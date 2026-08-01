from __future__ import annotations

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from src.algorithms import L1ADLightningModule
from src.algorithms.components.augmentation import (
    FastDetectorSmearing,
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
        detector_smearing: FastDetectorSmearing | None = None,
        lorentz_rotation: FastLorentzRotation | None = None,
        encoder_variance_weight: float = 0.0,
        encoder_covariance_weight: float = 0.0,
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
                "detector_smearing",
                "lorentz_rotation",
            ],
            logger=False,
        )
        self.projector = projector
        self.feature_blur = feature_blur
        self.object_mask = object_mask
        self.detector_smearing = detector_smearing
        self.lorentz_rotation = lorentz_rotation
        self.encoder_variance_weight = float(encoder_variance_weight)
        self.encoder_covariance_weight = float(encoder_covariance_weight)
        if self.encoder_variance_weight < 0.0 or self.encoder_covariance_weight < 0.0:
            raise ValueError("Encoder VICReg weights must be non-negative.")
        if self.encoder_covariance_weight > 0.0 and self.encoder_variance_weight == 0.0:
            raise ValueError(
                "encoder_covariance_weight requires a positive encoder_variance_weight "
                "to prevent a collapsed covariance-only solution."
            )
        self.seed = int(seed)
        self.diagnosis_metrics = diagnosis_metrics
        self.ckpt_path = ckpt

        self.feat_blurs = nn.ModuleDict(self._make_aug_pair(feature_blur))
        self.obj_masks = nn.ModuleDict(self._make_aug_pair(object_mask))
        self.detector_smears = nn.ModuleDict({"1": nn.Identity(), "2": nn.Identity()})
        self.lorentz_rot = nn.ModuleDict({"1": nn.Identity(), "2": nn.Identity()})

    def on_fit_start(self):
        self.setup_pairing(self.trainer.datamodule, setup_lorentz=True)
        if self.ckpt_path:
            self._load_checkpoint(self.ckpt_path)

    def on_validation_start(self):
        self._setup_model_feature_map_from_trainer()

    def on_test_start(self):
        self._setup_model_feature_map_from_trainer()
        self._reset_augmentation_rng(self.seed + 100_000)

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
            for augmenter in self.obj_masks.values():
                if hasattr(augmenter, "set_object_feature_map"):
                    augmenter.set_object_feature_map(object_feature_map)

        if normalizer is not None and object_feature_map is not None:
            normalizer.setup_1d_denorm(object_feature_map)
            if self.detector_smearing is not None and l1_scales is not None:
                self.detector_smears = nn.ModuleDict(
                    self._make_detector_smearing_pair(
                        normalizer=normalizer,
                        object_feature_map=object_feature_map,
                        l1_scales=l1_scales,
                    )
                ).to(self.device)

        if setup_lorentz and self.lorentz_rotation is not None and normalizer is not None:
            self.lorentz_rot = nn.ModuleDict(
                self._make_lorentz_pair(
                    normalizer=normalizer,
                    object_feature_map=object_feature_map,
                    l1_scales=l1_scales,
                )
            ).to(self.device)

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

    def augment_pair(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor | None = None,
        *,
        return_masks: bool = False,
    ):
        views = []
        for key in ("1", "2"):
            x_view = x_flat.clone()
            m_view = None if m_flat is None else m_flat.clone()
            for augmenter in (
                self.feat_blurs[key],
                self.detector_smears[key],
                self.obj_masks[key],
                self.lorentz_rot[key],
            ):
                x_view, m_view = self._apply_augmentation(augmenter, x_view, m_view)
            views.append((x_view, m_view))

        if return_masks:
            return views[0][0], views[1][0], views[0][1], views[1][1]
        return views[0][0], views[1][0]

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        m = self._flat_mask(b.mask)

        x1, x2, m1, m2 = self.augment_pair(x, m, return_masks=True)
        h1 = self.encode_flat(x1, m1)
        h2 = self.encode_flat(x2, m2)
        h = (h1 + h2) / 2.0 if self.training else self.encode_flat(x, m)

        z1 = self.projector(h1)
        z2 = self.projector(h2)
        ntxent_loss = self.loss(z1, z2)
        encoder_variance = ntxent_loss.new_zeros(())
        encoder_covariance = ntxent_loss.new_zeros(())
        loss = ntxent_loss
        if self.encoder_variance_weight > 0.0 or self.encoder_covariance_weight > 0.0:
            encoder_variance, encoder_covariance = self._encoder_vicreg_terms(h1, h2)
            loss = (
                ntxent_loss
                + self.encoder_variance_weight * encoder_variance
                + self.encoder_covariance_weight * encoder_covariance
            )

        with torch.no_grad():
            diag = self._diagnostics(h1, h2, z1, z2)

        outputs = {
            "loss": loss,
            "loss/mean": loss.detach(),
            "loss/ntxent": ntxent_loss.detach(),
            "loss/encoder_variance": encoder_variance.detach(),
            "loss/encoder_covariance": encoder_covariance.detach(),
            "loss/encoder_variance_weighted": (
                self.encoder_variance_weight * encoder_variance.detach()
            ),
            "loss/encoder_covariance_weighted": (
                self.encoder_covariance_weight * encoder_covariance.detach()
            ),
            "pairing_rep_data": h.detach(),
            "pairing_view1_data": h1.detach(),
            "pairing_view2_data": h2.detach(),
            "jetclr_proj1_data": z1.detach(),
            "jetclr_proj2_data": z2.detach(),
            "loss/total/full": loss.detach().expand(h.shape[0]),
            **diag,
        }
        # The clean projector representation is an evaluation diagnostic.  Keeping
        # this branch out of training avoids a third projector pass and leaves the
        # contrastive loss graph exactly unchanged.
        if not self.training:
            with torch.no_grad():
                outputs["jetclr_clean_proj_data"] = self.projector(h).detach()
        return outputs

    def outlog(self, outdict: dict) -> dict:
        diag_entries = {k: v for k, v in outdict.items() if k.startswith("diag_")}
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "loss_ntxent": outdict.get("loss/ntxent"),
            "loss_encoder_variance": outdict.get("loss/encoder_variance"),
            "loss_encoder_covariance": outdict.get("loss/encoder_covariance"),
            "loss_encoder_variance_weighted": outdict.get("loss/encoder_variance_weighted"),
            "loss_encoder_covariance_weighted": outdict.get("loss/encoder_covariance_weighted"),
            **diag_entries,
        }

    def _encoder_vicreg_terms(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return VICReg variance and covariance terms over the global batch."""
        global_h1 = self._gather_encoder_representation(h1)
        global_h2 = self._gather_encoder_representation(h2)
        if global_h1.shape[0] < 2:
            raise ValueError("Encoder VICReg regularization needs at least two global samples.")

        variance = 0.5 * (self._variance_penalty(global_h1) + self._variance_penalty(global_h2))
        covariance = global_h1.new_zeros(())
        if self.encoder_covariance_weight > 0.0:
            covariance = self._covariance_penalty(global_h1) + self._covariance_penalty(global_h2)
        return variance, covariance

    @staticmethod
    def _variance_penalty(h: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        std = torch.sqrt(h.var(dim=0, unbiased=True) + eps)
        return F.relu(1.0 - std).mean()

    @staticmethod
    def _covariance_penalty(h: torch.Tensor) -> torch.Tensor:
        n_samples, n_features = h.shape
        centered = h - h.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (n_samples - 1)
        off_diagonal = covariance.flatten()[:-1].view(n_features - 1, n_features + 1)[:, 1:]
        return off_diagonal.square().sum() / n_features

    @staticmethod
    def _gather_encoder_representation(h: torch.Tensor) -> torch.Tensor:
        """Differentiably gather equal-sized DDP batches for global VICReg statistics."""
        if not dist.is_available() or not dist.is_initialized():
            return h

        world_size = dist.get_world_size()
        sizes = [torch.zeros((), dtype=torch.long, device=h.device) for _ in range(world_size)]
        local_size = torch.tensor(h.shape[0], dtype=torch.long, device=h.device)
        dist.all_gather(sizes, local_size)
        observed_sizes = [int(size.item()) for size in sizes]
        if any(size != h.shape[0] for size in observed_sizes):
            raise RuntimeError(
                "Distributed encoder VICReg requires equal per-rank batch sizes; "
                f"received {observed_sizes}. Use drop_last=True or a fixed-size iterable batcher."
            )
        return torch.cat(differentiable_all_gather(h), dim=0)

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

    def _reset_augmentation_rng(self, seed: int) -> None:
        """Reset evaluation views independently of training history and batch size."""
        offset = 0
        for pair in (
            self.feat_blurs,
            self.detector_smears,
            self.obj_masks,
            self.lorentz_rot,
        ):
            for key in ("1", "2"):
                augmenter = pair[key]
                if hasattr(augmenter, "rng"):
                    augmenter.rng.set_seed(int(seed) + offset)
                offset += 1

    def _make_detector_smearing_pair(
        self,
        normalizer,
        object_feature_map: dict,
        l1_scales: dict,
    ) -> dict[str, nn.Module]:
        resolution = torch.zeros_like(normalizer.scale_tensor, dtype=torch.float32)
        nonnegative = torch.zeros_like(normalizer.scale_tensor, dtype=torch.bool)
        periodic_scales = torch.zeros_like(normalizer.scale_tensor, dtype=torch.float32)
        for object_name, feature_map in object_feature_map.items():
            object_scales = l1_scales.get(object_name, {})
            for feature_name, indices in feature_map.items():
                scale = object_scales.get(feature_name)
                if scale is not None:
                    # Inputs are stored as hardware codes, so one unit here is one
                    # detector LSB regardless of its physical conversion factor.
                    resolution[indices] = 1.0
                    if feature_name == "Et":
                        nonnegative[indices] = True
                    if feature_name == "phi":
                        periodic_scales[indices] = float(scale)

        base = self.detector_smearing(
            normalizer=normalizer,
            resolution_tensor=resolution,
            nonnegative_mask=nonnegative,
            periodic_scale_tensor=periodic_scales,
        )
        return self._make_aug_pair(base)

    def _make_lorentz_pair(
        self,
        normalizer,
        object_feature_map: dict | None,
        l1_scales: dict | None,
    ) -> dict[str, nn.Module]:
        if object_feature_map is None or l1_scales is None:
            return {"1": nn.Identity(), "2": nn.Identity()}

        phi_idxs = []
        phi_scale_by_index = torch.ones_like(normalizer.scale_tensor, dtype=torch.float32)
        for obj_name, feature_map in object_feature_map.items():
            phi_values = feature_map.get("phi")
            if not phi_values:
                continue
            phi_idxs.extend(phi_values)
            scale = l1_scales.get(obj_name, {}).get("phi", 1.0)
            phi_scale_by_index[phi_values] = float(scale)

        if not phi_idxs:
            return {"1": nn.Identity(), "2": nn.Identity()}

        phi_mask = torch.zeros_like(normalizer.scale_tensor, dtype=torch.bool)
        phi_mask[phi_idxs] = True
        l1_scale_phi = phi_scale_by_index[phi_mask]
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
    def _apply_augmentation(
        augmenter: nn.Module,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(augmenter, nn.Identity):
            return x, mask
        if isinstance(augmenter, (FastFeatureBlur, FastObjectMask)):
            result = augmenter(x)
        else:
            result = augmenter(x, mask)
        if isinstance(result, tuple):
            return result
        return result, mask

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
    ) -> dict[str, torch.Tensor]:
        if not self.diagnosis_metrics:
            return {}

        z1n = F.normalize(z1, dim=1)
        z2n = F.normalize(z2, dim=1)
        sim = z1n @ z2n.T
        batch_size = sim.shape[0]
        labels = torch.arange(batch_size, device=sim.device)
        recall1 = (sim.argmax(dim=1) == labels).float().mean()
        recall10 = (
            (torch.topk(sim, k=min(10, batch_size), dim=1).indices == labels[:, None])
            .any(dim=1)
            .float()
            .mean()
        )

        h1_std = torch.sqrt(h1.var(dim=0, unbiased=False) + 1e-4).mean()
        h2_std = torch.sqrt(h2.var(dim=0, unbiased=False) + 1e-4).mean()
        pos_cos = F.cosine_similarity(z1, z2, dim=1).mean()
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
    def _offdiag_mean(sim: torch.Tensor) -> torch.Tensor:
        n = sim.shape[0]
        if n < 2:
            return sim.new_tensor(float("nan"))
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        return sim[mask].mean()

    def _load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["state_dict"], strict=False)
