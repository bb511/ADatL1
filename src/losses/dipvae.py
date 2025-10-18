from typing import Literal
import torch

from src.losses import L1ADLoss


class DIPVAELoss(L1ADLoss):
    """
    DIP-VAE loss for disentangled representations. Decorrelates dimensions of the latent space.

    :param scale: Scaling factor for the loss.
    :param lambda_diag: Weight for the diagonal regularization term.
    :param lambda_offdiag: Weight for the off-diagonal regularization term.
    :param reduction: Reduction method to apply to the loss ("none", "mean", "sum").
    """
    
    def __init__(
        self,
        scale: float = 1.0,
        lambda_diag: float = 10.0,
        lambda_offdiag: float = 5.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag
        
    def forward(
        self,
        z_mean: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        batch_size = z_mean.shape[0]
        
        # Compute covariance matrix
        z_centered = z_mean - z_mean.mean(dim=0)
        cov = (z_centered.t() @ z_centered) / (batch_size - 1)
        
        # Diagonal loss (variance regularization)
        diag_loss = torch.mean((torch.diag(cov) - 1) ** 2)
        
        # Off-diagonal loss (covariance regularization)
        off_diag = cov - torch.diag(torch.diag(cov))
        offdiag_loss = torch.mean(off_diag ** 2)
        
        loss = self.lambda_diag * diag_loss + self.lambda_offdiag * offdiag_loss
        return self.scale * self.reduce(loss.expand(batch_size))