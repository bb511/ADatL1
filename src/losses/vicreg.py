from typing import Optional, Literal
import torch
import torch.nn.functional as F

from src.losses.components import L1ADLoss


class VICRegLoss(L1ADLoss):
    """
    VICReg-style loss combining invariance, variance, and covariance losses.

    The VICreg loss computes one statistic over the batch in any case, so there is no
    use for the reduction argument of other losses.
    Furthermore, the simple scale argument is replaced by three coefficients, as defined
    in the __init__ of this class.

    :param scale: Overall scaling factor for the loss.
    :param inv_coef: Weight for the invariance loss (MSE loss).
    :param rvar_coef: Ratio of how much the var coefficient is with respect to the inv.
    :param rcov_coef: Ratio of how much the cov coefficient is with respect to the inv.
    :param reduction: Reduction method to apply to the loss ("none", "mean", "sum").
    """

    def __init__(
        self,
        inv_coef: Optional[float] = 50,
        rvar_coef: Optional[float] = 50,
        rcov_coef: Optional[float] = 1,
    ):

        super().__init__(scale=None, reduction=None)
        self.inv_coef = inv_coef
        self.var_coef = inv_coef*rvar_coef
        self.cov_coef = inv_coef*rcov_coef

    def invariance_loss(self, z1, z2):
        z1n = F.normalize(z1, dim=1)
        z2n = F.normalize(z2, dim=1)
        return F.mse_loss(z1n, z2n)

    def variance_loss(self, z):
        z = z - z.mean(dim=0, keepdim=True)
        std_z = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return torch.mean(F.relu(1.0 - std_z))

    def covariance_loss(self, z):
        batch_size, feature_dim = z.shape
        # Compute covariance matrix.
        z = z - z.mean(dim=0, keepdim=True)
        z = z * ((batch_size - 1) ** -0.5)
        cov = z.T @ z
        # Remove the diagonal elements (i.e. variance terms).
        off_diag = cov[~torch.eye(feature_dim, device=z.device, dtype=torch.bool)]
        # Compute the covariance loss as the sum of squares of off-diagonal elements.
        return off_diag.pow(2).mean()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        loss_inv = self.invariance_loss(z1, z2)
        loss_var = 0.5 * (self.variance_loss(z1) + self.variance_loss(z2))
        loss_cov = self.covariance_loss(z1) + self.covariance_loss(z2)

        loss_inv = self.inv_coef * loss_inv
        loss_var = self.var_coef * loss_var
        loss_cov = self.cov_coef * loss_cov
        loss_total = loss_inv + loss_var + loss_cov

        return loss_inv, loss_var, loss_cov, loss_total

    def reduce(self):
        pass
