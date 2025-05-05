from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for VAE without reduction."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        
    def forward(self, target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(target, reconstruction, reduction='none')
        
        # Reduce across all dimensions except batch dimension
        loss_per_observation = torch.mean(mse_loss, dim=tuple(range(1, mse_loss.dim())))
        return self.scale * loss_per_observation


class KLDivergenceLoss(nn.Module):
    """Kullback-Leibler divergence loss for VAE without reduction."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        
    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        kl_per_observation = - 0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        return self.scale * kl_per_observation


class VAELoss(nn.Module):
    def __init__(
        self, 
        alpha: float = 1.0, 
        beta: float = 0.1,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ):
        super().__init__()
        self.reco_scale = alpha * (1 - beta)
        self.kl_scale = beta
        self.reduction = reduction
        
        self.reconstruction_loss = ReconstructionLoss(scale=self.reco_scale)
        self.kl_loss = KLDivergenceLoss(scale=self.kl_scale)
        
    def forward(
        self, 
        target: torch.Tensor, 
        reconstruction: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        reduction: Literal['none', 'mean', 'sum'] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        reduction = reduction if reduction is not None else self.reduction
        
        # Get per-observation losses
        reco_loss_per_obs = self.reconstruction_loss(target, reconstruction)
        kl_loss_per_obs = self.kl_loss(z_mean, z_log_var)
        total_loss_per_obs = reco_loss_per_obs + kl_loss_per_obs
        
        # Apply reduction
        if reduction == 'none':
            return total_loss_per_obs, reco_loss_per_obs, kl_loss_per_obs
        elif reduction == 'mean':
            return (
                torch.mean(total_loss_per_obs), 
                torch.mean(reco_loss_per_obs), 
                torch.mean(kl_loss_per_obs)
            )
        elif reduction == 'sum':
            return (
                torch.sum(total_loss_per_obs), 
                torch.sum(reco_loss_per_obs), 
                torch.sum(kl_loss_per_obs)
            )
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")
    

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