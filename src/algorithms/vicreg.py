from typing import Tuple, Dict, Union
import os
import copy
import numpy as np

import torch
import torch.nn as nn

from src.algorithms import L1ADLightningModule
from src.algorithms.components.mlp import MLP
from src.algorithms.components.augmentation import FastFeatureBlur
from src.algorithms.components.augmentation import FastObjectMask
from src.algorithms.components.augmentation import FastLorentzRotation
from src.algorithms.utils.weight_loader import load_weights
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

# LEGACY - WIP to MAKE IT MODERN.
class VICReg(L1ADLightningModule):
    """VICreg deep learning algorithm.

    See https://arxiv.org/abs/2105.04906 for more details.
    Used to generate an intermediate representation on which one can run the anomaly
    detection algorithm of choice on.

    :param projector: Projection head module.
    :param feature_blur: Feature blur augmentation module.
    :param object_mask: Object mask augmentation module.
    :param lorentz_rotation: Lorentz rotation augmentation module.
    :param seed: Int to seed the augmentations with.
    :param diagnosis_metrics: Bool whether to run all the diagnostics or not.
    :param ckpt: String with path pointing to a pytorch lightning checkpoint to restart
        the training from.
    """

    def __init__(
        self,
        projector: MLP,
        feature_blur: FastFeatureBlur = None,
        object_mask: FastObjectMask = None,
        lorentz_rotation: FastLorentzRotation = None,
        seed: int = 42,
        diagnosis_metrics: bool = False,
        ckpt: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Ignore saving hyperparams to avoid memory bloating.
        self.save_hyperparameters(
            ignore=[
                "loss",
                "projector",
                "feature_blur",
                "object_mask",
                "lorentz_rotation",
            ]
        )
        raise ValueError("VICReg is legacy for now! WIP to make it modern.")
        self.seed = seed
        self.diagnosis_metrics = diagnosis_metrics
        self.ckpt_path = ckpt

        self.projector = projector

        # Instantiate augmentation modules with different rngs
        self.feat_blurs = {}
        feature_blur_1 = copy.deepcopy(feature_blur)
        feature_blur_2 = copy.deepcopy(feature_blur)
        feature_blur_1.rng.set_seed(self.seed)
        feature_blur_2.rng.set_seed(self.seed + 1)
        self.feat_blurs.update({"1": feature_blur_1})
        self.feat_blurs.update({"2": feature_blur_2})

        self.obj_masks = {}
        object_mask_1 = copy.deepcopy(object_mask)
        object_mask_2 = copy.deepcopy(object_mask)
        object_mask_1.rng.set_seed(self.seed)
        object_mask_2.rng.set_seed(self.seed + 1)
        self.obj_masks.update({"1": object_mask_1})
        self.obj_masks.update({"2": object_mask_2})

        self.lorentz_rotation = lorentz_rotation

    def on_fit_start(self):
        """Set up the Lorentz augmentations and load weights if given ckpt.

        Save them as dictionaries so pytorch lightning does not instantiate them when
        loading from checkpoints, since they are not needed for inference.
        """
        self.lorentz_rot = {}
        lorentz_rotation = self._setup_phi_lorentz_rotation()
        lorentz_rotation.to(self.device)
        lorentz_rotation_1 = copy.deepcopy(lorentz_rotation)
        lorentz_rotation_2 = copy.deepcopy(lorentz_rotation)
        lorentz_rotation_1.rng.set_seed(self.seed)
        lorentz_rotation_2.rng.set_seed(self.seed + 1)
        self.lorentz_rot.update({"1": lorentz_rotation_1})
        self.lorentz_rot.update({"2": lorentz_rotation_2})

        # Load weights from checkpoint.
        if self.ckpt_path:
            self._load_checkpoint()

    def _setup_phi_lorentz_rotation(self) -> FastLorentzRotation:
        """Sets up the lorents rotation on the phi feature of each object.

        First, get the indices of the phi values in the batch torch tensor.
        Then, get the l1 scales for the phi values, to turn them into real physical phi
        values.
        """
        object_feature_map = self.trainer.datamodule.loader.object_feature_map
        normalizer = self.trainer.datamodule.normalizer
        normalizer.setup_1d_denorm(object_feature_map)
        l1_scales = self.trainer.datamodule.l1_scales

        phi_idxs = []
        l1_scale_phi = []
        for obj_name, feature_map in object_feature_map.items():
            for feat_name, idxs in feature_map.items():
                if feat_name != "phi":
                    continue

                phi_idxs.extend(idxs)
                nconst = len(idxs)
                l1_scale_phi.extend(nconst * [l1_scales[obj_name]["phi"]])

        phi_mask = torch.zeros_like(normalizer.scale_tensor, dtype=torch.bool)
        phi_mask[phi_idxs] = True
        l1_scale_phi = torch.tensor(l1_scale_phi, dtype=torch.float32)

        return self.lorentz_rotation(
            normalizer=normalizer, phi_mask=phi_mask, l1_scale_phi=l1_scale_phi
        )

    def forward(self, x: torch.Tensor):
        # Apply augmentations
        x1, x2 = x.clone(), x.clone()
        x1 = self.lorentz_rot["1"](self.obj_masks["1"](self.feat_blurs["1"](x1)))
        x2 = self.lorentz_rot["2"](self.obj_masks["2"](self.feat_blurs["2"](x2)))

        h1 = self.model(x1)
        h2 = self.model(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return h1, h2, z1, z2

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, _, _, _ = batch
        x = torch.flatten(x, start_dim=1)

        h1, h2, z1, z2 = self.forward(x)

        # Compute loss and return
        loss_inv, loss_var, loss_cov, loss_total = self.loss(z1, z2)
        loss_total = self._add_hgq_loss(loss_total)

        with torch.no_grad():
            inv_ratio = (loss_inv / loss_total).item()
            var_ratio = (loss_var / loss_total).item()
            cov_ratio = (loss_cov / loss_total).item()
            z1_std = torch.sqrt(z1.var(dim=0) + 1e-4).mean()
            z2_std = torch.sqrt(z2.var(dim=0) + 1e-4).mean()
            z1_var_min = z1.var(dim=0).min()
            z1_var_median = z1.var(dim=0).median()
            z1_var_max = z1.var(dim=0).max()
            outdict_diag = self._compute_diagnosis_metrics(h1, h2)

        return {
            # Used for backprop:
            "loss": loss_total,
            # Used for logging:
            # losses
            "loss/inv": loss_inv.detach(),
            "loss/var": loss_var.detach(),
            "loss/cov": loss_cov.detach(),
            "loss/inv/ratio": inv_ratio,
            "loss/var/ratio": var_ratio,
            "loss/cov/ratio": cov_ratio,
            # Projector diagnosis:
            "z1/std": z1_std,
            "z2/std": z2_std,
            "z1/var_min": z1_var_min,
            "z2/var_median": z1_var_median,
            "z2/var_max": z1_var_max,
            **outdict_diag,
        }

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ):
        x, _, _, _ = batch
        x = torch.flatten(x, start_dim=1)

        h1, h2, z1, z2 = self.forward(x)
        h = self.model(x).detach()

        # Compute loss and return
        loss_inv, loss_var, loss_cov, loss_total = self.loss(z1, z2)
        loss_total = self._add_hgq_loss(loss_total)

        inv_ratio = (loss_inv / loss_total).item()
        var_ratio = (loss_var / loss_total).item()
        cov_ratio = (loss_cov / loss_total).item()
        z1_std = torch.sqrt(z1.var(dim=0) + 1e-4).mean()
        z2_std = torch.sqrt(z2.var(dim=0) + 1e-4).mean()
        z1_var_min = z1.var(dim=0).min()
        z1_var_median = z1.var(dim=0).median()
        z1_var_max = z1.var(dim=0).max()
        outdict_diag = self._compute_diagnosis_metrics(h1, h2)
        h_efr = self._effective_rank_cov(h)

        outdict = {
            # Used for backprop:
            "loss": loss_total,
            # Used for logging:
            # losses
            "loss/inv": loss_inv.detach(),
            "loss/var": loss_var.detach(),
            "loss/cov": loss_cov.detach(),
            "loss/inv/ratio": inv_ratio,
            "loss/var/ratio": var_ratio,
            "loss/cov/ratio": cov_ratio,
            # Projector diagnosis:
            "z1/std": z1_std,
            "z2/std": z2_std,
            "z1/var_min": z1_var_min,
            "z2/var_median": z1_var_median,
            "z2/var_max": z1_var_max,
            # Backbone diagnosis:
            "effective_rank": h_efr,
            **outdict_diag,
            # Used for callbacks:
            "vicreg_rep_data": h,
        }

        self._log_dict(outdict, dataloader_idx=dataloader_idx)

        return outdict

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        x, m, _, _ = batch
        x = torch.flatten(x, start_dim=1)

        h1, h2, z1, z2 = self.forward(x)
        h = self.model(x).detach()

        loss_inv, loss_var, loss_cov, loss_total = self.loss(z1, z2)
        loss_total = self._add_hgq_loss(loss_total)

        h_efr = self._effective_rank_cov(h)
        return {
            "loss": loss_total.detach(),
            "vicreg_rep_data": h.detach(),
            "effective_rank": h_efr,
        }

    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        diag_entries = {k: v for k, v in outdict.items() if k.startswith("diag_")}
        return {
            "loss": outdict.get("loss"),
            "loss_inv": outdict.get("loss/inv"),
            "loss_var": outdict.get("loss/var"),
            "loss_cov": outdict.get("loss/cov"),
            "loss_inv_ratio": outdict.get("loss/inv/ratio"),
            "loss_var_ratio": outdict.get("loss/var/ratio"),
            "loss_cov_ratio": outdict.get("loss/cov/ratio"),
            "z1_std": outdict.get("z1/std"),
            "z2_std": outdict.get("z2/std"),
            "z1_var_min": outdict.get("z1/var_min"),
            "z1_var_median": outdict.get("z1/var_median"),
            "z1_var_max": outdict.get("z1/var_max"),
            **diag_entries,
        }

    def _compute_diagnosis_metrics(self, h1: torch.Tensor, h2: torch.Tensor):
        """Compute metrics to diagnose what the vicreg has learnt."""
        diag_outdict = {}
        if not self.diagnosis_metrics:
            return diag_outdict

        h1_efr, h1_max_ev = self._effective_rank_cov(h1)
        h2_efr, h2_max_ev = self._effective_rank_cov(h2)
        rep_eff_rank = min(h1_efr, h2_efr)
        max_ev = max(h1_max_ev, h2_max_ev)
        diag_outdict.update({"diag_eff_rank": rep_eff_rank, "diag_max_ev": max_ev})

        h1_cos_similarity = self._average_pairwise_cosine(h1)
        h2_cos_similarity = self._average_pairwise_cosine(h2)
        avg_cos_sim = max(h1_cos_similarity, h2_cos_similarity)
        diag_outdict.update({"diag_avg_cos_sim": avg_cos_sim})

        pair_cos = torch.nn.functional.cosine_similarity(h1, h2, dim=1).mean().item()
        diag_outdict.update({"diag_avg_pair_cos_sim": pair_cos})

        return diag_outdict

    @torch.no_grad()
    def _effective_rank_cov(self, z: torch.Tensor, eps: float = 1e-12):
        zc = z - z.mean(dim=0, keepdim=True)
        B = zc.shape[0]
        C = (zc.T @ zc) / (B - 1)

        eigvals = torch.linalg.eigvalsh(C).clamp_min(eps)
        trace = eigvals.sum()
        trace2 = (eigvals**2).sum()
        r_eff = (trace**2 / trace2).item()
        max_ev = eigvals.max().item()

        return r_eff, max_ev

    @torch.no_grad()
    def _average_pairwise_cosine(self, z: torch.Tensor, max_points: int = 2048):
        if z.shape[0] > max_points:
            z = z[:max_points]

        z = z - z.mean(dim=0, keepdim=True)
        z = torch.nn.functional.normalize(z, dim=1)  # L2 norm
        sim = z @ z.T  # [B, B]
        B = z.shape[0]

        # remove diagonal (self-similarity = 1)
        avg_cos = (sim.sum() - B) / (B * (B - 1))
        return avg_cos.item()

    def _load_checkpoint(self):
        """Load checkpoint weights to continue the training from, if provided."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]

        is_lightning_model = isinstance(self.model, MLP)
        is_lightning_projector = isinstance(self.projector, MLP)
        if is_lightning_model and is_lightning_projector:
            self.load_state_dict(state_dict, strict=True)
            return

        load_weights(self.model, state_dict, "model", False)
        load_weights(self.projector, state_dict, "projector", True)

    def _add_hgq_loss(self, loss_total):
        """Add additional HGQ losses if they exist."""
        add_loss = 0.0
        if hasattr(self.model, "losses") and len(self.model.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.model.losses]).sum()

        loss_total += add_loss

        return loss_total
