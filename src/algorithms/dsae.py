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

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

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
            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))

            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(
                    ascore, 1.0 - self.target_fpr
                ).item()

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

    def _split_by_type_from_flat(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build per-type tensors from flattened inputs using object_feature_map."""
        object_feature_map = getattr(self, "object_feature_map", None)
        if object_feature_map is None:
            raise RuntimeError(
                "object_feature_map not found on module. "
                "Make sure inject_object_feature_map(self) was called in "
                "on_fit_start/on_test_start."
            )

        if m_flat is None:
            m_flat = torch.ones_like(x_flat, dtype=x_flat.dtype, device=x_flat.device)

        x_by_type = {}
        m_by_type = {}

        for obj_name, feature_map in object_feature_map.items():
            feat_names = list(feature_map.keys())
            feat_indices = [feature_map[feat_name] for feat_name in feat_names]

            n_obj = len(feat_indices[0])
            n_feat = len(feat_indices)

            if not all(len(idxs) == n_obj for idxs in feat_indices):
                raise ValueError(
                    f"Feature map for '{obj_name}' has inconsistent object counts."
                )

            obj_features = []
            obj_masks = []

            for obj_idx in range(n_obj):
                this_obj_feat_idx = [
                    feat_indices[f_idx][obj_idx] for f_idx in range(n_feat)
                ]
                idx_tensor = torch.tensor(
                    this_obj_feat_idx, device=x_flat.device, dtype=torch.long
                )

                x_obj = torch.index_select(x_flat, dim=1, index=idx_tensor)
                m_obj = torch.index_select(m_flat, dim=1, index=idx_tensor)

                obj_features.append(x_obj)
                obj_masks.append(m_obj.all(dim=1))

            x_by_type[obj_name] = torch.stack(obj_features, dim=1)
            m_by_type[obj_name] = torch.stack(obj_masks, dim=1)

        return x_by_type, m_by_type
