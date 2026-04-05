# DeepSets per-type auto-encoder implementation.
from typing import Optional

import torch
from torch import nn

from src.algorithms import L1ADLightningModule
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map


class DeepSetsAE(L1ADLightningModule):
    """DeepSets per-type auto-encoder for anomaly detection.

    This model uses a per-type DeepSets encoder to build an event-level representation,
    and a standard decoder to reconstruct the flattened input. The anomaly score is the
    masked MSE.

    :param encoder: Per-type DeepSets encoder module.
    :param decoder: Decoder module mapping the event representation back to the
        flattened input space.
    :param mse: Module computing the per-sample MSE anomaly score.
    :param mask: Bool whether to mask padded entries in the reconstruction loss.
    :param features: Optional feature module to apply to the flattened input before the
        DeepSets split. Usually left as identity for object-wise inputs.
    :param input_noise_std: Standard deviation of Gaussian input noise used during
        training only.
    :param operational_rate: Float operational trigger rate.
    :param bc_rate: Float bunch crossing rate used to derive the operational quantile.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        mse: nn.Module,
        mask: bool = True,
        features: Optional[nn.Module] = None,
        input_noise_std: float = 0.0,
        operational_rate: float = 0.25,
        bc_rate: float = 28608.8064,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.encoder = encoder
        self.decoder = decoder
        self.mse = mse
        self.input_noise_std = input_noise_std
        self.mask = mask
        self.operational_quantile = 1 - operational_rate / bc_rate

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
        x, m, _, _ = batch
        x_flat = torch.flatten(x, start_dim=1)
        m_flat = torch.flatten(m, start_dim=1).float()

        x_input = x_flat
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(x_flat) * self.input_noise_std
            noise = noise * m_flat
            x_input = x_flat + noise

        x_by_type, m_by_type = self._split_by_type_from_flat(x_input, m_flat)

        z, reconstruction = self.forward(x_by_type, m_by_type)
        reco_loss = self.loss(reco=reconstruction, target=x_flat, mask=m_flat)
        mse = self.mse(x_flat, reconstruction, m_flat)
        del x_flat, z

        with torch.no_grad():
            n = mse.numel()

            # Number of extreme tail samples
            k = max(10, int((1.0 - self.operational_quantile) * n))
            topk_vals = torch.topk(mse, k).values
            mean_top_vals = topk_vals.mean().item()

            # 99th quantile MSE
            mse_q99 = torch.quantile(mse, 0.99).item()

        return {
            # Used for backpropagation:
            "loss": reco_loss.mean(),
            # Used for logging:
            "loss/reco/mean": reco_loss.mean(),
            "loss/mse/mean_top_vals": mean_top_vals,
            "loss/mse/q99": mse_q99,
            # Used for callbacks:
            "loss/total/full": reco_loss.detach(),
            "loss/mse/full": mse.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """The values of the loss that are logged."""
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_mse_mean_top_vals": outdict.get("loss/mse/mean_top_vals"),
            "loss_mse_q99": outdict.get("loss/mse/q99"),
        }

    def _split_by_type_from_flat(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build per-type tensors from flattened inputs using object_feature_map.

        This method expects flattened input, such that it corresponds to the
        object_feature_map dictionary that is saved on the data side. It then returns
        a dictionary with the data categorised by type, e.g., into muons, jets, etc...
        """
        object_feature_map = getattr(self, "object_feature_map", None)
        if object_feature_map is None:
            raise RuntimeError(
                "object_feature_map not found on module. "
                "Make sure inject_object_feature_map(self) was called in "
                "on_fit_start/on_test_start. This needs to happen in the current "
                "L1ADLightningModule you are using."
            )

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
                this_obj_feat_idx = [feat_indices[f_idx][obj_idx] for f_idx in range(n_feat)]
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
