from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.algorithms import L1ADLightningModule
from src.models.mcmc import SampleBuffer


class WNAE(L1ADLightningModule):
    """
    The WNAE algorithm differs from a standard Variational Autoencoder by
    replacing the KL divergence regularizer with a Wasserstein distance
    term computed between samples drawn from the model and the input
    data.  Negative samples are generated via a Metropolis-adjusted
    Langevin Monte Carlo (MALA) sampler.  Additionally, the NAE loss term
    encourages low-energy reconstructions by comparing the average
    reconstruction errors of the positive and negative samples.

    :param encoder: Neural network mapping input data to latent space.
    :param decoder: Neural network mapping latent vectors back to the input space.
    :param sampler: MCMC sampler module used to generate negative samples.
    :param features: Optional feature extractor applied to inputs before
        computing the reconstruction error. Defaults to the identity function.
    :param spherical: Whether to project latent representations to the unit hypersphere.
    :param clip: Optional boundaries for clipping or rejection.
    :param sampling: Sampling strategy for generating negative samples. One of: 
        "cd" (Contrastive Divergence), "pcd" (Persistent Contrastive Divergence)
        or "omi" (Out-of-Model Initialization).
        See :class:`src.utils.mcmc.SampleBuffer` for details. Default is "pcd".
    :param initial_dist: Distribution used to initialize the Langevin sampler. One of:
        "gaussian" or "uniform". Default is "gaussian".
    :param replay: Whether to use a replay buffer to initialize negative chains.
    :param replay_ratio: Fraction of initial points sampled from the replay buffer.
    :param buffer_size: Maximum size of the replay buffer.
    """
        
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sampler: nn.Module,
        features: Optional[nn.Module] = None,

        spherical: Optional[bool] = False,
        clip: Optional[Tuple[float, float]] = (0.0, 1.0),
        sampling: Optional[str] = "pcd",
        initial_dist: Optional[str] = "gaussian",
        replay: Optional[bool] = True,
        replay_ratio: Optional[float] = 0.95,
        buffer_size: Optional[int] = 10_000,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "sampler","loss"]
        )

        self.encoder, self.decoder = encoder, decoder
        self.sampler = sampler
        self.features = features if features is not None else nn.Identity()
        self.features.eval()
        self.spherical = spherical
        self.clip = tuple(clip)

        # Sampling parameters
        self.sampling = sampling
        self.initial_dist = initial_dist
        self.replay = replay
        self.replay_ratio = replay_ratio
        self.buffer_size = buffer_size

        # Initialize replay buffer
        self.buffer = SampleBuffer(max_samples=buffer_size, replay_ratio=replay_ratio)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z = self.encoder(x)
        if self.spherical:
            z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)

        reconstruction = self.decoder(z)
        return z, reconstruction
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error (energy) for input samples."""
        z, reconstruction = self.forward(x)
        energy = self.loss.losses.reconstruction(
            target=x,
            reconstruction=reconstruction,
            z=z,
        )
        return energy

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_pos, _ = batch
        x_pos = torch.flatten(x_pos, start_dim=1)
        z_pos, reconstruction_pos = self.forward(x_pos)
        energy_pos = self.loss.losses.reconstruction(
            target=x_pos,
            reconstruction=reconstruction_pos,
            z=z_pos,
        )

        # BUG: I am not sure about this, but how else is it supposed to work?
        # TODO: Check the wnae_ repo more carefully
        if self.training:
            # Generate negative samples and compute their energy
            x_neg = self.sample_negative(x_pos)
            z_neg, reconstruction_neg = self.forward(x_neg)
            energy_neg = self.loss.losses.reconstruction(
                target=x_neg,
                reconstruction=reconstruction_neg,
                z=z_neg
            )
        else:
            x_neg, energy_neg = x_pos, energy_pos


        # Normal reconstruction loss
        loss_reco = energy_pos.mean()
        # Wasserstein distance between positive and negative samples:
        loss_wasserstein = self.loss.losses.wasserstein(
            target=x_pos.detach(),
            negative_samples=x_neg.detach()
        )
        # Mean energy difference:
        loss_nae = self.loss.losses.nae(
            energy_pos=energy_pos.detach(),
            energy_neg=energy_neg.detach(),
        )
        loss_total = loss_reco + loss_wasserstein + loss_nae

        return {
            "loss": loss_total,
            "loss/total/full": loss_total.detach(),
            "loss/reconstruction": loss_reco.detach(),
            "loss/wasserstein": loss_wasserstein.detach(),
            "loss/nae": loss_nae.detach(),
            # "energy_positive": energy_pos.detach(),
            # "energy_negative": energy_neg.detach(),
        }

    def _initial_samples(
            self,
            n_samples: int,
            shape: Tuple[int, ...]
        ) -> torch.Tensor:
        """Generate initial points for the Langevin sampler."""
        samples = []
        n_replay = 0
        if self.replay and len(self.buffer) > 0:
            # Determine how many samples to draw from the buffer
            n_replay = int((np.random.rand(n_samples) < self.replay_ratio).sum())
            if n_replay > 0:
                samples.append(self.buffer.get(n_replay))

        # Remaining samples from the specified distribution
        n_new = n_samples - n_replay
        if n_new > 0:
            if self.initial_dist == "gaussian":
                x0_new = torch.randn((n_new,) + shape, dtype=torch.float)
            else: # uniform
                x0_new = torch.rand((n_new,) + shape, dtype=torch.float)
            
            # Rescale to bounds if clipping is enabled
            if self.sampling != "omi" and self.clip is not None:
                x0_new = x0_new * (self.clip[1] - self.clip[0]) + self.clip[0]
            samples.append(x0_new)
        return torch.cat(samples, dim=0)
    
    def sample_negative(self, x: torch.Tensor) -> torch.Tensor:
        """Generate negative samples via Langevin MCMC."""

        batch_size, sample_shape = x.size(0), x.size()[1:]

        # Initialize chain either from provided x (for CD) or from buffer / random
        if self.sampling == "cd":
            x0 = x.detach().to(self.device)
        else:
            x0 = self._initial_samples(batch_size, sample_shape).to(self.device)

        # Generate negative samples via Langevin MCMC
        negative = self.sampler(x=x0, model=self.energy)

        # Update replay buffer if enabled
        if self.replay and self.sampling == "pcd":
            self.buffer.push(negative.detach())
        return negative