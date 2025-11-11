from typing import Literal
import torch

from src.losses.components import L1ADLoss


class MMDLoss(L1ADLoss):
    """
    Maximum Mean Discrepancy loss for VAE latent space regularization.
    
    :param scale: Scaling factor for the loss.
    :param kernel: Kernel type to use ("rbf", "linear", "poly", "imq").
    :param kernel_bandwidth: Bandwidth parameter for the kernel.
    :param prior: Prior distribution to sample from ("gaussian", "uniform").
    :param reduction: Reduction method to apply to the loss ("none", "mean", "sum").
    """

    name: str = "mmd"
    
    def __init__(
        self,
        scale: float = 1.0,
        kernel: Literal["rbf", "linear", "poly", "imq"] = "rbf",
        kernel_bandwidth: float = 1.0,
        prior: Literal["gaussian", "uniform"] = "gaussian",
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.prior = prior
        
    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix between x and y."""

        if self.kernel == "rbf":
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist.pow(2) / (2 * self.kernel_bandwidth ** 2))
        elif self.kernel == "linear":
            return torch.mm(x, y.t())
        elif self.kernel == "poly":
            return (torch.mm(x, y.t()) / self.kernel_bandwidth + 1).pow(3)
        elif self.kernel == "imq":  # Inverse multi-quadratic
            dist = torch.cdist(x, y, p=2)
            return (1 + dist.pow(2) / self.kernel_bandwidth).pow(-0.5)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def forward(
            self,
            z_mean: torch.Tensor, 
            **kwargs
        ) -> torch.Tensor:
        """Compute MMD between latent codes and prior."""
        
        batch_size = z_mean.shape[0]
        
        # Sample from prior
        if self.prior == "gaussian":
            z_prior = torch.randn_like(z_mean)
        elif self.prior == "uniform":
            z_prior = (torch.rand_like(z_mean) - 0.5) * 2 * torch.sqrt(torch.tensor(3.0))
        else:
            raise ValueError(f"Unknown prior: {self.prior}")
        
        # Compute MMD
        k_zz = self.compute_kernel(z_mean, z_mean)
        k_zp = self.compute_kernel(z_mean, z_prior)
        k_pp = self.compute_kernel(z_prior, z_prior)
        
        # Remove diagonal
        k_zz = k_zz * (1 - torch.eye(batch_size, device=z_mean.device))
        k_pp = k_pp * (1 - torch.eye(batch_size, device=z_prior.device))
        
        mmd = k_zz.sum() / (batch_size * (batch_size - 1))
        mmd -= 2 * k_zp.mean()
        mmd += k_pp.sum() / (batch_size * (batch_size - 1))
        return self.scale * self.reduce(mmd.expand(batch_size))