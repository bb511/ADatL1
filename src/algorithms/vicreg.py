from typing import Tuple, Dict, Union
import os
import json
import pickle
import copy

import torch
import torch.nn as nn

from src.algorithms import L1ADLightningModule


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
        projector: nn.Module,
        feature_blur: nn.Module,
        object_mask: nn.Module,
        lorentz_rotation: nn.Module,
        object_norm_params: Union[str, dict],
        object_feature_map: Union[str, dict],
        seed: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.projector = projector

        # Instantiate augmentation modules with different rngs
        self.fb1, self.fb2 = copy.deepcopy(feature_blur), copy.deepcopy(feature_blur)
        self.fb1.rng.set_seed(seed); self.fb2.rng.set_seed(seed + 1)
        self.om1, self.om2 = copy.deepcopy(object_mask), copy.deepcopy(object_mask)
        self.om1.rng.set_seed(seed); self.om2.rng.set_seed(seed + 1)

        if isinstance(object_feature_map, str):
            if not object_feature_map.endswith(".json"):
                raise ValueError("The 'object_feature_map' provided is invalid.")
        
            with open(object_feature_map, 'r') as file:
                object_feature_map = json.load(file)

        norm_scale, norm_bias = self._norm_tensor(object_feature_map, object_norm_params)
        phi_inds = [
            ind
            for _, feature_map in object_feature_map.items()
            for ind in feature_map.get("phi", [])
        ]
        phi_mask = torch.zeros_like(norm_scale, dtype=torch.bool)
        phi_mask[phi_inds] = True
        lorentz_rotation = lorentz_rotation(
            norm_scale=norm_scale,
            norm_bias=norm_bias,
            phi_mask=phi_mask
        )
        self.lor1, self.lor2 = copy.deepcopy(lorentz_rotation), copy.deepcopy(lorentz_rotation)
        self.lor1.rng.set_seed(seed); self.lor2.rng.set_seed(seed + 1)

    def _norm_tensor(self, object_feature_map: dict, object_norm_params: Union[str, dict]):
        """Get a 'scale' and 'shift' tensors to directly apply to the data."""

        if isinstance(object_norm_params, str):
            if not os.path.isdir(object_norm_params):
                raise ValueError("The 'object_norm_params' provided is invalid.")
            
            norm_params_dir = str(object_norm_params)
            object_norm_params = {}
            for obj_name in object_feature_map.keys():
                norm_params_path = os.path.join(norm_params_dir, f"{obj_name}_norm_params.pkl")
                if not os.path.isfile(norm_params_path):
                    raise ValueError(f"The normalization parameters for {obj_name} can't be found.")
                
                with open(norm_params_path, 'rb') as file:
                    object_norm_params[obj_name] = pickle.load(file)
        
        # Compute number of dimensions to preallocate scale and shift tensors:
        ndims = sum([
            len(inds)
            for _, feature_map in object_feature_map.items()
            for _, inds in feature_map.items()
        ])
        scale_tensor = torch.ones(ndims, dtype=torch.float32)
        shift_tensor = torch.zeros(ndims, dtype=torch.float32)
        
        for obj_name, feature_norm_params in object_norm_params.items():
            for feat, params in feature_norm_params.items():
                inds = object_feature_map[obj_name][feat]
                scale_tensor[inds] = float(params.get("scale", 1.0))
                shift_tensor[inds] = float(params.get("shift", 0.0))

        return scale_tensor, shift_tensor

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, _ = batch
        x = torch.flatten(x, start_dim=1)

        # Apply augmentations
        x1, x2 = x.clone(), x.clone()
        del x
        x1 = self.lor1(self.om1(self.fb1(x1)))
        x2 = self.lor2(self.om2(self.fb2(x2)))

        # Get projections
        z1 = self.projector(self.model(x1))
        z2 = self.projector(self.model(x2))
        del x1, x2

        # Compute loss and return
        loss_inv, loss_var, loss_cov, loss_total = self.loss(z1, z2)

        return {
            "loss": loss_total,
            "loss/total/full": loss_total.detach(),
            "loss/inv": loss_inv.detach(),
            "loss/var": loss_var.detach(),
            "loss/cov": loss_cov.detach(),
        }
    
    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss": outdict.get("loss"),
            "loss_inv": outdict.get("loss/inv").mean(),
            "loss_var": outdict.get("loss/var").mean(),
            "loss_cov": outdict.get("loss/cov").mean(),
        }