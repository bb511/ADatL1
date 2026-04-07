from typing import Optional, Tuple, Dict

import torch
from torch import nn

from src.algorithms import L1ADLightningModule
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class DeepSetsRealNVP(L1ADLightningModule):
    """RealNVP on top of a DeepSets variational encoder latent space.

    The flow is trained on the deterministic latent representation z_mean produced
    by the DS-VAE encoder. The anomaly score is the negative log-likelihood (NLL)
    under the flow.

    :param encoder: Pretrained DeepSets variational encoder.
    :param flow: LatentRealNVP module.
    :param features: Optional feature module applied before splitting by object type.
    :param freeze_encoder: Whether to freeze the encoder during flow training.
    :param ckpt: Optional checkpoint path for the full module.
    :param operational_rate: Float operational trigger rate.

    """

    def __init__(
        self,
        encoder: nn.Module,
        flow: nn.Module,
        features: Optional[nn.Module] = None,
        freeze_encoder: bool = True,
        ckpt: str = "",
        operational_rate: float = 0.25,
        bc_rate: float = 28608.8064,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)

        # Save non-heavy hyperparameters. Avoid serializing full modules here.
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "flow", "loss"]
        )

        self.encoder = encoder
        self.flow = flow

        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        # Whether the DS-VAE encoder stays fixed while training the flow.
        self.freeze_encoder = freeze_encoder
        self.ckpt_path = ckpt

        self.object_feature_map = None
        self.operational_quantile = 1 - operational_rate / bc_rate

    def on_fit_start(self):
        """Set up object_feature_map, move features to device, and freeze encoder if requested."""
        inject_object_feature_map(self)
        self.features.to(self.device)

        if self.freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        if self.ckpt_path:
            self._load_checkpoint()

    def on_test_start(self):
        """Inject object_feature_map for evaluation-only workflows."""
        inject_object_feature_map(self)

    def forward(
        self,
        x_by_type: dict[str, torch.Tensor],
        m_by_type: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder and flow."""
        # If encoder is frozen, disable gradient tracking through it.
        # If not frozen, allow gradients to flow through encoder as well.
        with torch.set_grad_enabled(not self.freeze_encoder):
            z_mean, z_log_var, _ = self.encoder(x_by_type, m_by_type)

        # Model the latent mean under the flow.
        log_prob = self.flow.log_prob(z_mean)
        return z_mean, z_log_var, log_prob

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shared train/val/test step.

        The input is flattened, split back into object-type tensors, encoded into
        latent space by the DS-VAE encoder, then scored by the flow.
        """
        x, m, _, _ = batch

        x_flat = torch.flatten(x, start_dim=1)
        m_flat = torch.flatten(m, start_dim=1).float()
        x_flat = self.features(x_flat)

        # Reconstruct per-type tensors from the flattened representation.
        x_by_type, m_by_type = self._split_by_type_from_flat(x_flat, m_flat)
        z_mean, z_log_var, log_prob = self.forward(x_by_type, m_by_type)

        # Flow anomaly score: negative log-likelihood.
        nll = -log_prob

        with torch.no_grad():
            # Number of per-sample NLL values in the batch.
            n = nll.numel()

            # Number of extreme-tail samples used for the operational proxy.
            k = max(10, int((1.0 - self.operational_quantile) * n))

            topk_vals = torch.topk(nll, k).values
            mean_top_vals = topk_vals.mean().item()

            nll_q50, nll_q95, nll_q99, nll_q999 = torch.quantile(
                nll,
                torch.tensor([0.5, 0.95, 0.99, 0.999], device=nll.device),
            ).tolist()

            # Same latent diagnostic you used elsewhere: mean squared norm of z_mean.
            z_mean_squared = torch.square(z_mean).sum(dim=1).mean().item()

        return {
            # Used for backpropagation:
            "loss": nll.mean(),
            # Used for logging:
            "loss/nll/mean": nll.mean(),
            "loss/nll/median": nll_q50,
            "loss/nll/q95": nll_q95,
            "loss/nll/q99": nll_q99,
            "loss/nll/q999": nll_q999,
            "loss/nll/mean_top_vals": mean_top_vals,
            "loss/z_mean_squared": z_mean_squared,
            # Used for callbacks:
            "loss/nll/full": nll.detach(),
            "latent_mean": z_mean.detach(),
            "latent_log_var": z_log_var.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """Values logged in the standard training loop."""
        return {
            "loss": outdict.get("loss"),
            "loss_nll": outdict.get("loss/nll/mean"),
            "loss_nll_median": outdict.get("loss/nll/median"),
            "loss_nll_q95": outdict.get("loss/nll/q95"),
            "loss_nll_q99": outdict.get("loss/nll/q99"),
            "loss_nll_q999": outdict.get("loss/nll/q999"),
            "loss_nll_mean_top_vals": outdict.get("loss/nll/mean_top_vals"),
            "z_mean_squared": outdict.get("loss/z_mean_squared"),
        }

    def _split_by_type_from_flat(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build per-type tensors from flattened inputs using object_feature_map.

        Expected input:
          - x_flat: [B, F_total]
          - m_flat: [B, F_total]

        This uses the injected object_feature_map to reconstruct object-wise tensors
        from the flattened representation.
        """
        object_feature_map = getattr(self, "object_feature_map", None)
        if object_feature_map is None:
            raise RuntimeError("object_feature_map not found on module.")

        x_by_type = {}
        m_by_type = {}

        # Loop over each object family, e.g. jets / muons / egammas / taus / FET.
        for obj_name, feature_map in object_feature_map.items():
            feat_names = list(feature_map.keys())
            feat_indices = [feature_map[feat_name] for feat_name in feat_names]

            n_obj = len(feat_indices[0])
            n_feat = len(feat_indices)

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

                # x_obj shape: [B, F_obj]
                # m_obj shape: [B, F_obj]
                obj_features.append(x_obj)

                # Object is considered present if all its feature entries are valid.
                obj_masks.append(m_obj.all(dim=1))

            x_by_type[obj_name] = torch.stack(obj_features, dim=1)
            m_by_type[obj_name] = torch.stack(obj_masks, dim=1)

        return x_by_type, m_by_type

    def _load_checkpoint(self):
        """Load a previously saved checkpoint."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)
