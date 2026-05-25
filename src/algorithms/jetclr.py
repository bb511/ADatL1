from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms import L1ADLightningModule
from src.algorithms.components.augmentation import (
    FastFeatureBlur,
    FastLorentzRotation,
    FastObjectMask,
)
from src.algorithms.components.mlp import MLP
from src.algorithms.utils.object_feature_map_loader import maybe_get_object_feature_map
from src.data.utils import unpack_batch


class JetCLR(L1ADLightningModule):
    """JetCLR-style contrastive encoder for CAP pair-table construction."""

    def __init__(
        self,
        projector: MLP,
        feature_blur: FastFeatureBlur | None = None,
        object_mask: FastObjectMask | None = None,
        lorentz_rotation: FastLorentzRotation | None = None,
        seed: int = 42,
        diagnosis_metrics: bool = True,
        ckpt: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(
            ignore=[
                "loss",
                "projector",
                "model",
                "feature_blur",
                "object_mask",
                "lorentz_rotation",
            ],
            logger=False,
        )
        self.projector = projector
        self.feature_blur = feature_blur
        self.object_mask = object_mask
        self.lorentz_rotation = lorentz_rotation
        self.seed = int(seed)
        self.diagnosis_metrics = diagnosis_metrics
        self.ckpt_path = ckpt

        self.feat_blurs = self._make_aug_pair(feature_blur)
        self.obj_masks = self._make_aug_pair(object_mask)
        self.lorentz_rot = {"1": nn.Identity(), "2": nn.Identity()}

    def on_fit_start(self):
        self.setup_pairing(self.trainer.datamodule, setup_lorentz=True)
        if self.ckpt_path:
            self._load_checkpoint(self.ckpt_path)

    def on_validation_start(self):
        self._setup_model_feature_map_from_trainer()

    def on_test_start(self):
        self._setup_model_feature_map_from_trainer()

    def setup_pairing(self, datamodule=None, setup_lorentz: bool = True) -> None:
        """Prepare object maps and augmentations outside the Lightning loop."""
        object_feature_map = None
        normalizer = None
        l1_scales = None

        if datamodule is not None:
            loader = getattr(datamodule, "loader", None)
            object_feature_map = getattr(loader, "object_feature_map", None)
            normalizer = getattr(datamodule, "normalizer", None)
            l1_scales = getattr(datamodule, "l1_scales", None)

        if object_feature_map is not None:
            self.object_feature_map = object_feature_map
            if hasattr(self.model, "set_object_feature_map"):
                self.model.set_object_feature_map(object_feature_map)

        if setup_lorentz and self.lorentz_rotation is not None and normalizer is not None:
            self.lorentz_rot = self._make_lorentz_pair(
                normalizer=normalizer,
                object_feature_map=object_feature_map,
                l1_scales=l1_scales,
            )

    def forward(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_flat(x_flat, m_flat)

    def encode_flat(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_flat.ndim != 2:
            x_flat = torch.flatten(x_flat, start_dim=1)
        if m_flat is not None and m_flat.ndim != 2:
            m_flat = torch.flatten(m_flat, start_dim=1).float()

        try:
            return self.model(x_flat, m_flat)
        except TypeError:
            return self.model(x_flat)

    def encode_batch(self, batch) -> torch.Tensor:
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        m = self._flat_mask(b.mask)
        return self.encode_flat(x, m)

    def augment_pair(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = x_flat.clone()
        x2 = x_flat.clone()
        x1 = self.lorentz_rot["1"](self.obj_masks["1"](self.feat_blurs["1"](x1)))
        x2 = self.lorentz_rot["2"](self.obj_masks["2"](self.feat_blurs["2"](x2)))
        return x1, x2

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        m = self._flat_mask(b.mask)

        x1, x2 = self.augment_pair(x)
        h1 = self.encode_flat(x1, m)
        h2 = self.encode_flat(x2, m)
        h = self.encode_flat(x, m)

        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.loss(z1, z2)

        with torch.no_grad():
            diag = self._diagnostics(h1, h2, z1, z2)

        return {
            "loss": loss,
            "loss/mean": loss.detach(),
            "pairing_rep_data": h.detach(),
            "pairing_view1_data": h1.detach(),
            "pairing_view2_data": h2.detach(),
            "jetclr_proj1_data": z1.detach(),
            "jetclr_proj2_data": z2.detach(),
            "loss/total/full": loss.detach().expand(h.shape[0]),
            **diag,
        }

    def outlog(self, outdict: dict) -> dict:
        diag_entries = {k: v for k, v in outdict.items() if k.startswith("diag_")}
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            **diag_entries,
        }

    def _setup_model_feature_map_from_trainer(self) -> None:
        object_feature_map = maybe_get_object_feature_map(self)
        if object_feature_map is None:
            return
        self.object_feature_map = object_feature_map
        if hasattr(self.model, "set_object_feature_map"):
            self.model.set_object_feature_map(object_feature_map)

    def _make_aug_pair(self, aug: nn.Module | None) -> dict[str, nn.Module]:
        aug1 = copy.deepcopy(aug) if aug is not None else nn.Identity()
        aug2 = copy.deepcopy(aug) if aug is not None else nn.Identity()
        if hasattr(aug1, "rng"):
            aug1.rng.set_seed(self.seed)
        if hasattr(aug2, "rng"):
            aug2.rng.set_seed(self.seed + 1)
        return {"1": aug1, "2": aug2}

    def _make_lorentz_pair(
        self,
        normalizer,
        object_feature_map: dict | None,
        l1_scales: dict | None,
    ) -> dict[str, nn.Module]:
        if object_feature_map is None or l1_scales is None:
            return {"1": nn.Identity(), "2": nn.Identity()}

        normalizer.setup_1d_denorm(object_feature_map)
        phi_idxs = []
        l1_scale_phi = []
        for obj_name, feature_map in object_feature_map.items():
            phi_values = feature_map.get("phi")
            if not phi_values:
                continue
            phi_idxs.extend(phi_values)
            scale = l1_scales.get(obj_name, {}).get("phi", 1.0)
            l1_scale_phi.extend(len(phi_values) * [scale])

        if not phi_idxs:
            return {"1": nn.Identity(), "2": nn.Identity()}

        phi_mask = torch.zeros_like(normalizer.scale_tensor, dtype=torch.bool)
        phi_mask[phi_idxs] = True
        l1_scale_phi = torch.tensor(l1_scale_phi, dtype=torch.float32)
        base = self.lorentz_rotation(
            normalizer=normalizer,
            phi_mask=phi_mask,
            l1_scale_phi=l1_scale_phi,
        )
        aug1 = copy.deepcopy(base)
        aug2 = copy.deepcopy(base)
        if hasattr(aug1, "rng"):
            aug1.rng.set_seed(self.seed)
        if hasattr(aug2, "rng"):
            aug2.rng.set_seed(self.seed + 1)
        return {"1": aug1, "2": aug2}

    @staticmethod
    def _flat_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        return torch.flatten(mask, start_dim=1).float()

    @torch.no_grad()
    def _diagnostics(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> dict[str, float]:
        if not self.diagnosis_metrics:
            return {}

        z1n = F.normalize(z1, dim=1)
        z2n = F.normalize(z2, dim=1)
        sim = z1n @ z2n.T
        batch_size = sim.shape[0]
        labels = torch.arange(batch_size, device=sim.device)
        recall1 = (sim.argmax(dim=1) == labels).float().mean().item()
        recall10 = (
            torch.topk(sim, k=min(10, batch_size), dim=1).indices == labels[:, None]
        ).any(dim=1).float().mean().item()

        h1_std = torch.sqrt(h1.var(dim=0, unbiased=False) + 1e-4).mean().item()
        h2_std = torch.sqrt(h2.var(dim=0, unbiased=False) + 1e-4).mean().item()
        pos_cos = F.cosine_similarity(z1, z2, dim=1).mean().item()
        neg_cos = self._offdiag_mean(sim)

        return {
            "diag_recall1": recall1,
            "diag_recall10": recall10,
            "diag_pos_cos": pos_cos,
            "diag_neg_cos": neg_cos,
            "diag_h1_std": h1_std,
            "diag_h2_std": h2_std,
        }

    @staticmethod
    def _offdiag_mean(sim: torch.Tensor) -> float:
        n = sim.shape[0]
        if n < 2:
            return float("nan")
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        return sim[mask].mean().item()

    def _load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["state_dict"], strict=False)
