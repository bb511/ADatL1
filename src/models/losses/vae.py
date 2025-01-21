from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        
    def forward(self, target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        return self.scale * F.mse_loss(target, reconstruction)

class KLDivergenceLoss(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        
    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.mean(
            -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        )

class VAELoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.reco_scale = alpha * (1 - beta)
        self.kl_scale = beta
        
        self.reconstruction_loss = ReconstructionLoss(scale=self.reco_scale)
        self.kl_loss = KLDivergenceLoss(scale=self.kl_scale)
        
    def forward(
        self, 
        target: torch.Tensor, 
        reconstruction: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        reco_loss = self.reconstruction_loss(target, reconstruction)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        total_loss = reco_loss + kl_loss
        
        return total_loss, reco_loss, kl_loss
    

from typing import Dict
from src.models.losses.cylptpz import CylPtPzLoss, CylPtPzMAELoss

class CombinedVAELoss(nn.Module):
    def __init__(
        self, 
        norm_scales: torch.Tensor,
        norm_biases: torch.Tensor,
        mask: Dict[str, torch.Tensor],
        unscale_energy: bool = False,
        alpha: float = 1.0,
        beta: float = 0.1,
        use_mae: bool = False
    ):
        super().__init__()
        
        # Initialize component losses
        if use_mae:
            self.reco_loss = CylPtPzMAELoss(norm_scales, norm_biases, mask, unscale_energy)
        else:
            self.reco_loss = CylPtPzLoss(norm_scales, norm_biases, mask, unscale_energy)

        self.kl_loss = KLDivergenceLoss()
        
        # Loss scaling
        self.reco_scale = alpha * (1 - beta)
        self.kl_scale = beta
        
    def forward(self,
                target: torch.Tensor,
                reconstruction: torch.Tensor,
                z_mean: torch.Tensor,
                z_log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        reco_loss = self.reco_scale * self.reco_loss(target, reconstruction)
        kl_loss = self.kl_scale * self.kl_loss(z_mean, z_log_var)
        total_loss = reco_loss + kl_loss
        
        return total_loss, reco_loss, kl_loss