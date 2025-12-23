from typing import Literal, Optional
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
    :param block_size: Max block size for blockwise kernel computation.
    :param max_samples: If not None and batch_size > max_samples, randomly
                        subsample to this many points for the MMD.
    :param use_cpu: If True, perform the MMD computation on CPU (to save GPU
                    memory), then move the final scalar back to the original
                    device.
    """

    name: str = "mmd"

    def __init__(
        self,
        scale: float = 1.0,
        kernel: Literal["rbf", "linear", "poly", "imq"] = "rbf",
        kernel_bandwidth: float = 1.0,
        prior: Literal["gaussian", "uniform"] = "gaussian",
        reduction: Literal["none", "mean", "sum"] = "mean",
        block_size: int = 1024,
        max_samples: Optional[int] = None,
        use_cpu: bool = False,
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.prior = prior
        self.block_size = block_size
        self.max_samples = max_samples
        self.use_cpu = use_cpu

    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix between x and y."""
        if self.kernel == "rbf":
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist.pow(2) / (2 * self.kernel_bandwidth**2))
        elif self.kernel == "linear":
            return torch.mm(x, y.t())
        elif self.kernel == "poly":
            return (torch.mm(x, y.t()) / self.kernel_bandwidth + 1).pow(3)
        elif self.kernel == "imq":  # Inverse multi-quadratic
            dist = torch.cdist(x, y, p=2)
            return (1 + dist.pow(2) / self.kernel_bandwidth).pow(-0.5)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _self_kernel_sum_offdiag(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sum_{i != j} k(z_i, z_j) computed in blocks to avoid B x B allocation.
        """
        n = z.size(0)
        bs = self.block_size
        device = z.device

        total_sum = z.new_zeros((), device=device)
        diag_sum = z.new_zeros((), device=device)

        for i in range(0, n, bs):
            x_i = z[i : i + bs]
            for j in range(0, n, bs):
                x_j = z[j : j + bs]
                k_block = self.compute_kernel(x_i, x_j)
                total_sum = total_sum + k_block.sum()

                if i == j:
                    # Only this block contains diagonal elements
                    diag_sum = diag_sum + k_block.diagonal().sum()

        return total_sum - diag_sum  # off-diagonal only

    def _cross_kernel_sum(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Sum_{i,j} k(x_i, y_j) computed in blocks.
        """
        nx = x.size(0)
        ny = y.size(0)
        bs = self.block_size
        device = x.device

        total_sum = x.new_zeros((), device=device)

        for i in range(0, nx, bs):
            x_i = x[i : i + bs]
            for j in range(0, ny, bs):
                y_j = y[j : j + bs]
                k_block = self.compute_kernel(x_i, y_j)
                total_sum = total_sum + k_block.sum()

        return total_sum

    def forward(self, z_mean: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute MMD between latent codes and prior, using blockwise kernel
        computation (exact on the used sample) and optional subsampling.
        """
        orig_device = z_mean.device

        # Flatten in case encoder returns more dims
        if z_mean.dim() > 2:
            z_mean = z_mean.view(z_mean.size(0), -1)

        batch_size = z_mean.size(0)

        # Optional subsampling if batch is huge
        if self.max_samples is not None and batch_size > self.max_samples:
            idx = torch.randperm(batch_size, device=z_mean.device)[: self.max_samples]
            z_mean = z_mean[idx]
            batch_size = z_mean.size(0)

        # Possibly move to CPU for the heavy O(B^2) stuff
        if self.use_cpu:
            z_mean = z_mean.cpu()

        device_for_mmd = z_mean.device

        # Sample from prior on same device as z_mean
        if self.prior == "gaussian":
            z_prior = torch.randn_like(z_mean, device=device_for_mmd)
        elif self.prior == "uniform":
            z_prior = (
                (torch.rand_like(z_mean, device=device_for_mmd) - 0.5)
                * 2
                * torch.sqrt(torch.tensor(3.0, device=device_for_mmd))
            )
        else:
            raise ValueError(f"Unknown prior: {self.prior}")

        # Blockwise sums
        k_zz_offdiag = self._self_kernel_sum_offdiag(z_mean)  # sum_{i!=j} k(z_i,z_j)
        k_pp_offdiag = self._self_kernel_sum_offdiag(z_prior)  # sum_{i!=j} k(p_i,p_j)
        k_zp_sum = self._cross_kernel_sum(z_mean, z_prior)  # sum_{i,j} k(z_i,p_j)

        b = float(batch_size)

        # k_zz = k_zz_offdiag / (B(B-1)), k_pp same, k_zp = k_zp_sum / (B^2)
        mmd = (
            k_zz_offdiag / (b * (b - 1.0))
            - 2.0 * (k_zp_sum / (b * b))
            + k_pp_offdiag / (b * (b - 1.0))
        )
        loss = self.scale * self.reduce(mmd.expand(batch_size))
        loss = loss.to(orig_device)
        return loss
