# DeepSets per-type variational auto-encoder implementation.
from typing import Optional, Tuple, Dict

import torch
from torch import nn

from src.algorithms import L1ADLightningModule
from src.algorithms.schedulers.linear import LinearWarmup
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class DeepSetsVAE(L1ADLightningModule):
    """DeepSets variational autoencoder architecture.

    Same structure and logging as the flat VAE, but using a DeepSets encoder.
    The anomaly score is the raw KL divergence.

    :param encoder: DeepSets variational encoder nn module.
    :param decoder: Decoder nn module.
    :param kl_warmup_frac: Fraction of total steps used to warm up KL scaling.
    :param features: Optional feature module applied before the DeepSets split.
    :param mask: Bool whether to mask padded input features in reconstruction loss.
    :param ckpt: Path to checkpoint to restart training from.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        kl_warmup_frac: float = 0.0,
        features: Optional[nn.Module] = None,
        mask: bool = True,
        ckpt: str = "",
        operational_rate: float = 0.25,
        bc_rate: float = 28608.8064,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )

        self.encoder = encoder
        self.decoder = decoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.kl_warmup_frac = kl_warmup_frac
        self.mask = mask
        self.ckpt_path = ckpt
        self.object_feature_map = None

        self.operational_quantile = 1 - operational_rate / bc_rate

    def on_fit_start(self):
        inject_object_feature_map(self)
        self.features.to(self.device)

        total_steps = int(self.trainer.estimated_stepping_batches)
        self._setup_kl_annealing(self.kl_warmup_frac, total_steps)

        if self.ckpt_path:
            self._load_checkpoint()

    def on_test_start(self):
        inject_object_feature_map(self)

    def forward(
        self,
        x_by_type: dict[str, torch.Tensor],
        m_by_type: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x_by_type, m_by_type)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, m, _, _ = batch

        x_flat = torch.flatten(x, start_dim=1)
        m_flat = torch.flatten(m, start_dim=1).float()
        x_flat = self.features(x_flat)

        # Split flat → per-type tensors
        x_by_type, m_by_type = self._split_by_type_from_flat(x_flat, m_flat)

        z_mean, z_log_var, z, reconstruction = self.forward(x_by_type, m_by_type)

        kl_current_scale = self.kl_scale(int(self.global_step))
        reco_loss, kl_raw, kl_scaled, total_loss = self.loss(
            reconstruction=reconstruction,
            z_mean=z_mean,
            z_log_var=z_log_var,
            target=x_flat,
            mask=m_flat if self.mask else None,
            kl_scale=kl_current_scale,
        )

        del z
        total_loss = self._add_hgq_loss(total_loss)

        # Diagnostics
        with torch.no_grad():
            kl_quantiles = torch.tensor([0.5, 0.95, 0.99, 0.999], device=kl_raw.device)
            kl_q50, kl_q95, kl_q99, kl_q999 = torch.quantile(
                kl_raw, kl_quantiles
            ).tolist()

            # Extreme tail KL (operational)
            n = kl_raw.numel()
            k = max(10, int((1.0 - self.operational_quantile) * n))
            topk_vals = torch.topk(kl_raw, k).values
            mean_top_vals = topk_vals.mean().item()

            z_mean_squared = torch.square(z_mean).sum(dim=1)

        return {
            # Backprop
            "loss": total_loss.mean(),
            # Logging
            "loss/reco/mean": reco_loss.detach().mean(),
            "loss/kl_scaled/mean": kl_scaled.detach().mean(),
            "loss/kl_raw/mean": kl_raw.detach().mean(),
            "loss/kl_raw/median": kl_q50,
            "loss/kl_raw/q95": kl_q95,
            "loss/kl_raw/q99": kl_q99,
            "loss/kl_raw/q999": kl_q999,
            "loss/kl_raw/mean_top_vals": mean_top_vals,
            "loss/z_mean_squared": z_mean_squared.detach().mean(),
            "kl_scale": kl_current_scale,
            # Callbacks
            "loss/total/full": total_loss.detach(),
            "loss/reco/full": reco_loss.detach(),
            "loss/kl_raw/full": kl_raw.detach(),
            "loss/z_mean_squared/full": z_mean_squared.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_kl_raw": outdict.get("loss/kl_raw/mean"),
            "loss_kl_scaled": outdict.get("loss/kl_scaled/mean"),
            "loss_kl_median": outdict.get("loss/kl_raw/median"),
            "loss_kl_raw_q95": outdict.get("loss/kl_raw/q95"),
            "loss_kl_raw_q99": outdict.get("loss/kl_raw/q99"),
            "loss_kl_raw_q999": outdict.get("loss/kl_raw/q999"),
            "loss_kl_raw_mean_top_vals": outdict.get("loss/kl_raw/mean_top_vals"),
            "z_mean_squared": outdict.get("loss/z_mean_squared"),
            "kl_scale": outdict.get("kl_scale"),
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
            raise RuntimeError("object_feature_map not found on module.")

        x_by_type = {}
        m_by_type = {}

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

                obj_features.append(x_obj)
                obj_masks.append(m_obj.all(dim=1))

            x_by_type[obj_name] = torch.stack(obj_features, dim=1)
            m_by_type[obj_name] = torch.stack(obj_masks, dim=1)

        return x_by_type, m_by_type

    def _setup_kl_annealing(self, kl_warmup_frac: float, total_steps: int):
        if hasattr(self.loss, "kl_scale_final"):
            fin_scale = float(self.loss.kl_scale_final)
        else:
            fin_scale = 1.0

        self.kl_scale = LinearWarmup(
            final_value=fin_scale,
            warmup_frac=kl_warmup_frac,
            total_steps=total_steps,
        )

    def _load_checkpoint(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def _add_hgq_loss(self, total_loss):
        add_loss = 0.0
        if hasattr(self.encoder, "losses") and len(self.encoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.encoder.losses]).sum()
        if hasattr(self.decoder, "losses") and len(self.decoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.decoder.losses]).sum()

        total_loss = total_loss + add_loss
        return total_loss
