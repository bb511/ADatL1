# DeepSets per-type variational auto-encoder implementation.
from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.vae import ClassicVAELoss
from src.algorithms.schedulers.linear import LinearWarmup
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.data.utils import unpack_batch


class DeepSetsVAE(ADLightningModule):
    """DeepSets variational autoencoder module.

    Same structure and logging convention as the flat VAE, but using a DeepSets encoder.
    The operational anomaly score is hard-configured to the raw KL divergence.

    :param encoder: DeepSets variational encoder nn module.
    :param decoder: Decoder nn module.
    :param kl_warmup_frac: Fraction of total steps used to warm up KL scaling.
    :param features: Optional feature module applied before the DeepSets split.
    :param mask: Whether to mask padded input features in reconstruction loss.
    :param ckpt: Optional checkpoint path to resume training from.
    :param target_rate: Target background rate or FPR.
    :param base_rate: Base rate used to convert target_rate into an FPR. If None,
        target_rate is interpreted directly as an FPR.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        kl_warmup_frac: float = 0.0,
        kl_scale: float = 1.0,
        features: Optional[nn.Module] = None,
        ckpt: str = "",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder"]
        )

        self.encoder = encoder
        self.decoder = decoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.kl_warmup_frac = kl_warmup_frac
        self.ckpt_path = ckpt

        # Hard-coded algorithm definition.
        self.loss = ClassicVAELoss(kl_scale=kl_scale, reduction="none")

    def on_fit_start(self):
        inject_object_feature_map(self)
        self.features.to(self.device)

        total_steps = int(self.trainer.estimated_stepping_batches)
        self._setup_kl_annealing(self.kl_warmup_frac, total_steps)

        if self.ckpt_path:
            self._load_checkpoint()

    def on_test_start(self):
        inject_object_feature_map(self)

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def forward(
        self,
        x_by_type: dict[str, torch.Tensor],
        m_by_type: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_by_type = {k: self.features(v) for k, v in x_by_type.items()}
        z_mean, z_log_var, z = self.encoder(x_by_type, m_by_type)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        b = unpack_batch(batch)

        x_flat = torch.flatten(b.x, start_dim=1)

        m_flat = b.mask
        if m_flat is not None:
            m_flat = torch.flatten(m_flat, start_dim=1).float()

        x_by_type, m_by_type = self._split_by_type_from_flat(x_flat, m_flat)

        z_mean, z_log_var, z, reconstruction = self.forward(x_by_type, m_by_type)

        kl_current_scale = self.kl_scale(int(self.global_step))
        reco_loss, kl_raw, kl_scaled, loss = self.loss(
            target=x_flat,
            reconstruction=reconstruction,
            mask=m_flat,
            z_mean=z_mean,
            z_log_var=z_log_var,
            kl_scale=kl_current_scale,
        )

        # The anomaly score is expected to be a distribution over events.
        ascore = kl_raw
        if ascore.ndim != 1:
            raise ValueError(
                f"Expected per-event ascores, got {tuple(ascore.shape)}."
            )

        loss = self._add_hgq_loss(loss)

        with torch.no_grad():
            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))

            # If the operational tail is too small, use a top-k average for stability.
            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(
                    ascore, 1.0 - self.target_fpr
                ).item()

            q50, q99 = torch.quantile(
                ascore, torch.tensor([0.5, 0.99], device=ascore.device)
            ).tolist()

            z_mean_squared = torch.square(z_mean).sum(dim=1)
            z_mean_squared_mean = z_mean_squared.mean().item()

        del x_flat, z

        loss_mean = loss.mean()
        reco_loss_mean = reco_loss.mean()
        kl_scaled_mean = kl_scaled.mean()
        kl_raw_mean = kl_raw.mean()

        return {
            # Used for backpropagation:
            "loss": loss_mean,
            # Used for logging:
            "loss/mean": loss_mean,
            "loss/reco/mean": reco_loss_mean,
            "loss/kl_scaled/mean": kl_scaled_mean,
            "loss/kl_raw/mean": kl_raw_mean,
            "ascore/operational": operational_ascore,
            "ascore/q50": q50,
            "ascore/q99": q99,
            "z_mean_squared": z_mean_squared_mean,
            "kl_scale": kl_current_scale,
            # Used for callbacks:
            "loss/full": loss.detach(),
            "loss/reco/full": reco_loss.detach(),
            "loss/kl_raw/full": kl_raw.detach(),
            "ascore/full": ascore.detach(),
            "z_mean_squared/full": z_mean_squared.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_kl_scaled": outdict.get("loss/kl_scaled/mean"),
            "loss_kl_raw": outdict.get("loss/kl_raw/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
            "ascore_q50": outdict.get("ascore/q50"),
            "ascore_q99": outdict.get("ascore/q99"),
            "z_mean_squared": outdict.get("z_mean_squared"),
            "kl_scale": outdict.get("kl_scale"),
        }

    def _split_by_type_from_flat(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build per-type tensors from flattened inputs using object_feature_map."""
        object_feature_map = getattr(self, "object_feature_map", None)
        if object_feature_map is None:
            raise RuntimeError("object_feature_map not found on module.")

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

    def _setup_kl_annealing(self, kl_warmup_frac: float, total_steps: int):
        """Set up KL annealing if supported by the loss."""
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

    def _add_hgq_loss(self, loss: torch.Tensor) -> torch.Tensor:
        add_loss = 0.0
        if hasattr(self.encoder, "losses") and len(self.encoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.encoder.losses]).sum()
        if hasattr(self.decoder, "losses") and len(self.decoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.decoder.losses]).sum()

        return loss + add_loss
