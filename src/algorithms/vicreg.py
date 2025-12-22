from typing import Tuple, Dict, Union
import os
import json
import pickle
import copy

import torch
import torch.nn as nn

from src.algorithms import L1ADLightningModule
from src.models.mlp import MLP
from src.models.augmentation import FastFeatureBlur
from src.models.augmentation import FastObjectMask
from src.models.augmentation import FastLorentzRotation


class VICReg(L1ADLightningModule):
    """
    Contrastive VAE.

    :param projector: Projection head module.
    :param feature_blur: Feature blur augmentation module.
    :param object_mask: Object mask augmentation module.
    :param lorentz_rotation: Lorentz rotation augmentation module.
    """

    def __init__(
        self,
        projector: MLP,
        feature_blur: FastFeatureBlur,
        object_mask: FastObjectMask,
        lorentz_rotation: FastLorentzRotation,
        seed: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Ignore saving hyperparams to avoid memory bloating.
        self.save_hyperparameters(
            ignore=[
                "model",
                "loss",
                "projector",
                "feature_blur",
                "object_mask",
                "lorentz_rotation",
            ]
        )
        self.seed = seed

        self.projector = projector

        # Instantiate augmentation modules with different rngs
        self.feature_blur_1 = copy.deepcopy(feature_blur)
        self.feature_blur_2 = copy.deepcopy(feature_blur)
        self.feature_blur_1.rng.set_seed(self.seed)
        self.feature_blur_2.rng.set_seed(self.seed + 1)

        self.object_mask_1 = copy.deepcopy(object_mask)
        self.object_mask_2 = copy.deepcopy(object_mask)
        self.object_mask_1.rng.set_seed(self.seed)
        self.object_mask_2.rng.set_seed(self.seed + 1)

        self.lorentz_rotation = lorentz_rotation

    def on_fit_start(self):
        """Set up the Lorentz augmentations."""
        lorentz_rotation = self._setup_phi_lorentz_rotation()
        self.lorentz_rotation_1 = copy.deepcopy(lorentz_rotation)
        self.lorentz_rotation_2 = copy.deepcopy(lorentz_rotation)
        self.lorentz_rotation_1.rng.set_seed(self.seed)
        self.lorentz_rotation_2.rng.set_seed(self.seed)

    def _setup_phi_lorentz_rotation(self) -> FastLorentzRotation:
        """Sets up the lorents rotation on the phi feature of each object.

        First, get the indices of the phi values in the batch torch tensor.
        Then, get the l1
        """
        object_feature_map = self.trainer.datamodule.loader.object_feature_map
        normalizer = self.trainer.datamodule.normalizer
        normalizer.setup_1d_denorm(object_feature_map)
        l1_scales = self.trainer.datamodule.l1_scales

        phi_idxs = []
        l1_scale_phi = []
        for obj_name, feature_map in object_feature_map.items():
            for feat_name, idxs in feature_map.items():
                if feat_name != 'phi':
                    continue

                phi_idxs.extend(idxs)
                nconst = len(idxs)
                l1_scale_phi.extend(nconst * [l1_scales[obj_name]['phi']])

        phi_mask = torch.zeros_like(normalizer.scale_tensor, dtype=torch.bool)
        phi_mask[phi_idxs] = True
        l1_scale_phi = torch.tensor(l1_scale_phi, dtype=torch.float32)

        return self.lorentz_rotation(
            normalizer=normalizer, phi_mask=phi_mask, l1_scale_phi=l1_scale_phi
        )

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, _ = batch
        x = torch.flatten(x, start_dim=1)

        # Apply augmentations
        x1, x2 = x.clone(), x.clone()
        del x
        x1 = self.lorentz_rotation_1(self.object_mask_1(self.feature_blur_1(x1)))
        x2 = self.lorentz_rotation_2(self.object_mask_2(self.feature_blur_2(x2)))

        # Get projections
        z1 = self.projector(self.model(x1))
        z2 = self.projector(self.model(x2))
        del x1, x2

        # Compute loss and return
        loss_inv, loss_var, loss_cov, loss_total = self.loss(z1, z2)

        return {
            "loss": loss_total,
            # Used for logging:
            "loss/inv": loss_inv.detach(),
            "loss/var": loss_var.detach(),
            "loss/cov": loss_cov.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss": outdict.get("loss"),
            "loss_inv": outdict.get("loss/inv"),
            "loss_var": outdict.get("loss/var"),
            "loss_cov": outdict.get("loss/cov"),
        }
