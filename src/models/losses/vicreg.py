from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    def __init__(
            self,
            inv_coef: Optional[float] = 25.0,
            var_coef: Optional[float] = 25.0,
            cov_coef: Optional[float] = 1.0,
        ):
        """
        VICReg-style loss combining invariance, variance, and covariance losses.

        :param inv_coef (float): Weight for the invariance loss (MSE loss).
        :param var_coef (float): Weight for the variance regularization.
        :param cov_coef (float): Weight for the covariance regularization.
        """
        super().__init__()
        self.inv_coef = inv_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef

    def invariance_loss(self, z1, z2):
        return F.mse_loss(z1, z2)

    def variance_loss(self, z):
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std_z))

    def covariance_loss(self, z):
        batch_size, feature_dim = z.shape
        z = z - z.mean(dim=0, keepdim=True)  # Center embeddings
        cov = (z.T @ z) / (batch_size - 1)  # Compute covariance matrix
        off_diag = cov.flatten()[:-1].view(feature_dim - 1, feature_dim + 1)[:, 1:].flatten()
        return off_diag.pow(2).sum() / float(feature_dim)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Computes the VICReg loss given two sets of embeddings.

        Args:
            z1 (torch.Tensor): First set of embeddings.
            z2 (torch.Tensor): Second set of embeddings.

        Returns:
            torch.Tensor: Total VICReg loss.
        """
        loss_inv = self.invariance_loss(z1, z2)
        loss_var = self.variance_loss(z1) + self.variance_loss(z2) 
        loss_cov = self.covariance_loss(z1) + self.covariance_loss(z2)

        loss_total = (
            self.inv_coef * loss_inv +
            self.var_coef * loss_var +
            self.cov_coef * loss_cov
        )
        return loss_inv, loss_var, loss_cov, loss_total
    

class L1VICRegLoss(VICRegLoss):

    def variance_loss(self, z):
        # TODO: Check why this implementation is different
        z = z - z.mean(dim=0, keepdim=True)
        std_z = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return torch.mean(F.relu(1.0 - std_z)) / 2