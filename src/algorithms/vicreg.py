from typing import Tuple, Dict
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

        # Instantiate augmentation modules
        self.fb1, self.fb2 = copy.deepcopy(feature_blur), copy.deepcopy(feature_blur)
        self.om1, self.om2 = copy.deepcopy(object_mask), copy.deepcopy(object_mask)
        # TODO: Here add the normalization, and then pass them so that they are fully initialized:
        self.lor1, self.lor2 = copy.deepcopy(lorentz_rotation), copy.deepcopy(lorentz_rotation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lor1(self.om1(self.fb1(x)))
        return self.projector(self.model(x))

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
            "loss": loss_total,
            "loss/inv": loss_inv.detach(),
            "loss/var": loss_var.detach(),
            "loss/cov": loss_cov.detach(),
        }
