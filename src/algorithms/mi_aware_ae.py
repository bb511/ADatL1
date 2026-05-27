from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.components.bernoulli import BernoulliQuantizedBottleneck
from src.algorithms.losses.ae import HuberAELoss
from src.algorithms.losses.components.mi import BernoulliBottleneckMILoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.data.utils import unpack_batch


class MIAwareAE(ADLightningModule):
    """Autoencoder with Bernoulli bottleneck and unsupervised MI regularization."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_noise_std: float = 0.0,
        delta: float = 3.0,
        features: nn.Module | None = None,
        target_rate: float = 0.25,
        base_rate: float | None = None,
        gamma: float = 0.0,
        use_bernoulli_bottleneck: bool = True,
        bottleneck_temperature: float = 1.0,
        deterministic_eval: bool = True,
        mi_reduction: str = "sum",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=None, **kwargs)

        self.save_hyperparameters(
            ignore=[
                "model",
                "features",
                "encoder",
                "decoder",
                "loss",
                "ascore",
                "bottleneck",
                "mi_loss",
            ]
        )

        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.encoder = encoder
        self.decoder = decoder
        self.input_noise_std = input_noise_std

        self.gamma = gamma
        self.use_bernoulli_bottleneck = use_bernoulli_bottleneck

        if use_bernoulli_bottleneck:
            self.bottleneck = BernoulliQuantizedBottleneck(
                temperature=bottleneck_temperature,
                deterministic_eval=deterministic_eval,
            )
        else:
            self.bottleneck = nn.Identity()

        self.loss = HuberAELoss(delta=delta, reduction="none")
        self.ascore = MSEReconstructionLoss(reduction="none")
        self.mi_loss = BernoulliBottleneckMILoss(reduction=mi_reduction)

    def on_fit_start(self) -> None:
        inject_object_feature_map(self)

    def on_test_start(self) -> None:
        inject_object_feature_map(self)

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x = self.features(x)
        z = self.encoder(x)

        if self.use_bernoulli_bottleneck:
            z_for_decoder, z_info = self.bottleneck(z)
        else:
            z_for_decoder = z
            z_info = {
                "logits": z,
                "probs": torch.sigmoid(z),
                "hard": z,
            }

        reconstruction = self.decoder(z_for_decoder)
        return z_for_decoder, reconstruction, z_info

    def _compute_operational_ascore(self, ascore: torch.Tensor) -> float:
        n = ascore.numel()
        k = max(1, int(self.target_fpr * n))

        if k < 10:
            k_eff = min(max(10, k), n)
            return torch.topk(ascore, k_eff).values.mean().item()

        return torch.quantile(ascore, 1.0 - self.target_fpr).item()

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor | float]:
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

        z, reconstruction, z_info = self.forward(x_noisy)

        recon_loss_full = self.loss(reco=reconstruction, target=x, mask=m)
        recon_loss = recon_loss_full.mean()

        if self.use_bernoulli_bottleneck and self.gamma != 0.0:
            mi = self.mi_loss(z_info["probs"])
        else:
            mi = reconstruction.new_tensor(0.0)

        total_loss = recon_loss + self.gamma * mi

        ascore = self.ascore(x, reconstruction, m)
        if ascore.ndim != 1:
            raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

        with torch.no_grad():
            operational_ascore = self._compute_operational_ascore(ascore)

            probs_flat = torch.flatten(z_info["probs"], start_dim=1)
            latent_mean_prob = probs_flat.mean()
            latent_std_prob = probs_flat.std()
            latent_entropy = self.mi_loss.binary_entropy(probs_flat).mean()

        return {
            # Backpropagation
            "loss": total_loss,

            # Logging
            "loss/mean": total_loss.detach(),
            "loss/reconstruction": recon_loss.detach(),
            "loss/mi": mi.detach(),
            "latent/mean_prob": latent_mean_prob.detach(),
            "latent/std_prob": latent_std_prob.detach(),
            "latent/entropy": latent_entropy.detach(),
            "ascore/operational": operational_ascore,

            # Evaluator/callback compatibility
            "loss/full": recon_loss_full.detach(),
            "ascore/full": ascore.detach(),
            "reconstructed_data": reconstruction.detach(),

            # Optional debugging
            "latent_quantized": z.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "loss_reconstruction": outdict.get("loss/reconstruction"),
            "loss_mi": outdict.get("loss/mi"),
            "latent_mean_prob": outdict.get("latent/mean_prob"),
            "latent_std_prob": outdict.get("latent/std_prob"),
            "latent_entropy": outdict.get("latent/entropy"),
            "ascore_operational": outdict.get("ascore/operational"),
        }