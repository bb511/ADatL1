# DeepSets per-type auto-encoder implementation.
from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.ae import HuberAELoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.data.utils import unpack_batch


class DeepSetsAE(ADLightningModule):
    """DeepSets per-type auto-encoder for anomaly detection.

    This model uses a per-type DeepSets encoder to build an event-level representation,
    and a standard decoder to reconstruct the flattened input.

    :param encoder: Per-type DeepSets encoder module.
    :param decoder: Decoder module mapping the event representation back to the
        flattened input space.
    :param features: Optional feature module to apply to the flattened input before the
        DeepSets split. Usually left as identity for object-wise inputs.
    :param input_noise_std: Standard deviation of Gaussian input noise used during
        training only.
    :param delta: Huber loss parameter controlling the L1/L2 transition.
    :param target_rate: Target background rate or FPR.
    :param base_rate: Base rate used to convert target_rate into an FPR. If None,
        target_rate is interpreted directly as an FPR.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_noise_std: float = 0.0,
        delta: float = 3.0,
        features: Optional[nn.Module] = None,
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder"]
        )

        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.encoder = encoder
        self.decoder = decoder
        self.input_noise_std = input_noise_std

        # Hard-coded algorithm definition.
        self.loss = HuberAELoss(delta=delta, reduction="none")
        self.ascore = MSEReconstructionLoss(reduction="none")

    def on_fit_start(self):
        inject_object_feature_map(self)

    def on_test_start(self):
        inject_object_feature_map(self)

    def forward(
        self,
        x_by_type: dict[str, torch.Tensor],
        m_by_type: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x_by_type, m_by_type)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        b = unpack_batch(batch)

        x_flat = torch.flatten(b.x, start_dim=1)

        m_flat = b.mask
        if m_flat is not None:
            m_flat = torch.flatten(m_flat, start_dim=1).float()

        x_input = x_flat
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(x_flat) * self.input_noise_std
            if m_flat is not None:
                noise = noise * m_flat
            x_input = x_flat + noise

        x_input = self.features(x_input)

        x_by_type, m_by_type = self._split_by_type_from_flat(x_input, m_flat)

        z, reconstruction = self.forward(x_by_type, m_by_type)
        loss = self.loss(reco=reconstruction, target=x_flat, mask=m_flat)
        ascore = self.ascore(x_flat, reconstruction, m_flat)

        if ascore.ndim != 1:
            raise ValueError(
                f"Expected per-event ascores, got {tuple(ascore.shape)}."
            )

        del x_flat, z

        with torch.no_grad():
            operational_ascore = self.compute_operational_ascore(ascore)

        loss_mean = loss.mean()

        return {
            # Used for backpropagation:
            "loss": loss_mean,
            # Used for logging:
            "loss/mean": loss_mean,
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
            "loss_reco": outdict.get("loss/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
        }
