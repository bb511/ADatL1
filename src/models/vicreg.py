from typing import Optional
import copy

import torch
import torch.nn as nn
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models import L1ADLightningModule
from src.models.quantization import Quantizer


class VICReg(L1ADLightningModule):
    """Contrastive VAE."""
    def __init__(
            self,
            projector: nn.Module,
            feature_blur: nn.Module,
            object_mask: nn.Module,
            lorentz_rotation: nn.Module,
            qdata: Optional[Quantizer] = None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model", "loss", "projector", "feature_blur", "object_mask", "lorentz_rotation"])
        self.projector = projector

        # Data quantization:
        self.qdata = qdata or Quantizer(None, None)
        
        # Instantiate augmentation modules
        self.fb1, self.fb2 = copy.deepcopy(feature_blur), copy.deepcopy(feature_blur)
        self.om1, self.om2 = copy.deepcopy(object_mask), copy.deepcopy(object_mask)
        self.lor1, self.lor2 = copy.deepcopy(lorentz_rotation), copy.deepcopy(lorentz_rotation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qdata(x)
        x = self.lor1(self.om1(self.fb1(x)))
        return self.projector(self.model(x))
    
    def model_step(self, x: torch.Tensor):
        # Quantize data
        x = self.qdata(x)
        
        # Apply augmentations
        x1, x2 = x.clone(), x.clone()
        del x; garbage_collection_cuda()
        x1 = self.lor1(self.om1(self.fb1(x1)))
        x2 = self.lor2(self.om2(self.fb2(x2)))
        
        # Get projections
        z1 = self.projector(self.model(x1))
        z2 = self.projector(self.model(x2))
        del x1, x2; garbage_collection_cuda()

        # Compute loss and return
        loss_inv, loss_var, loss_cov, loss_total = self.loss(z1, z2)

        return {
            "loss": loss_total,
            "loss_inv": loss_inv,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
        }