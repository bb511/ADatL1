# Vanilla auto-encoder model implementations
from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.ae import HuberAELoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.data.utils import unpack_batch


class AE(ADLightningModule):
    """Autoencoder module.

    :param encoder: PyTorch module of the encoder.
    :param decoder: PyTorch module of the decoder.
    :param features: Optional PyTorch module to apply to the data before feeding it
        to the autoencoder.
    :param input_noise_std: Float specifying how much noise to add to the feature
        distributions before feeding them to the AE.
    :param delta: Float defining how close the loss is to L1 or L2, i.e., how much
        importance is given to tail examples. Parameter in the HuberLoss.
    :param target_rate: Float of the target background rate or FPR for the AE.
    :param base_rate: Float of the base rate, used to compute FPR given a target rate.
        If this is 'None', then target_rate is taken as the FPR directly.
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_noise_std: Optional[float] = 0.0,
        delta: float = 3.0,
        features: nn.Modul = None,
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.encoder, self.decoder = encoder, decoder
        self.input_noise_std = input_noise_std
        self.loss = HuberAELoss(delta=delta, reduction='none')
        self.ascore = MSEReconstructionLoss(reduction='none')

    def on_fit_start(self):
        inject_object_feature_map(self)

    def on_test_start(self):
        inject_object_feature_map(self)

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def model_step(self, batch: torch.Tensor) -> torch.Tensor:
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)

        m = b.mask
        if m is not None:
            m = torch.flatten(m, start_dim=1).float()

        x_noisy = x
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(x) * self.input_noise_std
            if m is not None:
                noise = noise * m

            x_noisy = x + noise

        z, reconstruction = self.forward(x_noisy)
        loss = self.loss(reco=reconstruction, target=x, mask=m)
        # The anomaly score is expected to be a distribution over events.
        ascore = self.ascore(x, reconstruction, m)
        if ascore.ndim != 1:
            raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

        del x, z

        with torch.no_grad():
            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))

            # If the operational tail is too small, use a top-k average for stability.
            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(ascore, 1.0 - self.target_fpr).item()

        return {
            # Used for backpropagation:
            "loss": loss.mean(),
            # Used for logging:
            "loss/mean": loss.mean(),
            "ascore/operational": operational_ascore,
            # Used for callbacks:
            "loss/full": loss.detach(),
            "ascore/full": ascore.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """The values of the loss that are logged."""
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
        }
