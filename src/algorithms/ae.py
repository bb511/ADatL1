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
    :param features: Optional PyTorch module to apply to the data before feeding it to the
        autoencoder.
    :param input_noise_std: Float specifying how much noise to add to the feature distributions
        before feeding them to the AE.
    :param delta: Float defining how close the loss is to L1 or L2, i.e., how much importance is
        given to tail examples. Parameter in the HuberLoss.
    :param target_rate: Float of the target background rate or FPR for the AE.
    :param base_rate: Float of the base rate, used to compute FPR given a target rate. If this is
        'None', then target_rate is taken as the FPR directly.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_noise_std: float = 0.0,
        delta: float = 3.0,
        features: nn.Module = None,
        anomaly_score: str = "mse",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "encoder", "decoder", "loss"])
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.encoder, self.decoder = encoder, decoder
        self.input_noise_std = input_noise_std
        if anomaly_score not in {"mse", "residual_oas"}:
            raise ValueError("AE anomaly_score must be 'mse' or 'residual_oas'.")
        self.anomaly_score = anomaly_score
        self.loss = HuberAELoss(delta=delta, reduction="none")
        self.ascore = MSEReconstructionLoss(reduction="none")
        output_layers = [
            module for module in self.decoder.modules() if isinstance(module, nn.Linear)
        ]
        if not output_layers:
            raise ValueError("Dense AE decoder must contain a linear output layer.")
        residual_dim = int(output_layers[-1].out_features)
        self.register_buffer("residual_oas_location", torch.zeros(residual_dim))
        self.register_buffer("residual_oas_precision", torch.eye(residual_dim))
        self.register_buffer("residual_oas_ready", torch.tensor(False))

    def on_fit_start(self):
        """Install the object-feature map before fitting."""
        inject_object_feature_map(self)

    def on_test_start(self):
        """Install the object-feature map before testing."""
        inject_object_feature_map(self)

    @property
    def target_fpr(self) -> float:
        """Return the configured operational false-positive rate."""
        return self.compute_target_fpr()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input and reconstruct it from the latent representation."""
        x = self.features(x)
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def encode_flat(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode an already flattened feature tensor."""
        del m_flat
        x_flat = self.features(x_flat)
        return self.encoder(x_flat)

    def encode_batch(self, batch) -> torch.Tensor:
        """Unpack, flatten, and encode one repository-format batch."""
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        return self.encode_flat(x)

    def set_residual_oas_state(self, location: torch.Tensor, precision: torch.Tensor) -> None:
        """Install train-normal OAS state used by the covariance-aware score."""
        location = location.to(device=self.device, dtype=self.residual_oas_location.dtype)
        precision = precision.to(device=self.device, dtype=self.residual_oas_precision.dtype)
        if location.shape != self.residual_oas_location.shape:
            raise ValueError(
                f"Expected residual location {tuple(self.residual_oas_location.shape)}, "
                f"got {tuple(location.shape)}."
            )
        if precision.shape != self.residual_oas_precision.shape:
            raise ValueError(
                f"Expected residual precision {tuple(self.residual_oas_precision.shape)}, "
                f"got {tuple(precision.shape)}."
            )
        self.residual_oas_location.copy_(location)
        self.residual_oas_precision.copy_(precision)
        self.residual_oas_ready.fill_(True)

    def residual_oas_score(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return residual Mahalanobis energy over observed features only."""
        residual = target - reconstruction
        centered = residual - self.residual_oas_location
        if mask is None:
            denominator = residual.new_full((residual.shape[0],), residual.shape[1])
        else:
            mask = mask.to(device=centered.device, dtype=torch.bool)
            if mask.shape != centered.shape:
                raise ValueError(
                    f"Expected residual mask {tuple(centered.shape)}, got {tuple(mask.shape)}."
                )
            centered = centered.masked_fill(~mask, 0.0)
            denominator = mask.sum(dim=1).clamp_min(1).to(dtype=residual.dtype)
        return (
            torch.einsum("bi,ij,bj->b", centered, self.residual_oas_precision, centered)
            / denominator
        )

    def model_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute training loss and all callback-facing anomaly scores."""
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
        mse_ascore = self.ascore(x, reconstruction, m)
        residual_oas = (
            self.residual_oas_score(x, reconstruction, m)
            if bool(self.residual_oas_ready)
            else mse_ascore
        )
        ascore = {
            "mse": mse_ascore,
            "residual_oas": residual_oas,
        }[self.anomaly_score]
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
            "ascore/mse": mse_ascore.detach(),
            "ascore/residual_oas": residual_oas.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """The values of the loss that are logged."""
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
        }
