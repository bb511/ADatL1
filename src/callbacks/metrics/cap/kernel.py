from typing import Optional, Callable
import torch
from torch import Tensor, nn


class ApproximationCapacityKernel(nn.Module):
    """Computes the approximation capacity for a classification problem."""

    def __init__(
            self,
            beta0: float,
            energy_fn: Callable,
            normalize_gradients: Optional[bool] = False
        ):  
        super().__init__()
        self.energy_fn = energy_fn
        self.normalize_gradients = normalize_gradients
        
        # Learnable parameters
        self.beta = torch.nn.Parameter(
            torch.tensor([beta0], dtype=torch.float), requires_grad=True
        )
        self.cap = torch.tensor([0.0], dtype=torch.float, device="cpu")
        self.total_samples = 0

    def compute_mutual_information(
        self,
        prob1: Tensor,
        prob2: Tensor,
        beta: Optional[float] = None
    ) -> Tensor:
        """
        Compute mutual information for relaxed hypothesis space (factorized form).
        This is the original implementation but with different energy functions.
        """
        beta = self.beta if beta is None else beta
        dev = self.beta.device

        # Compute energy:
        n = len(prob1)
        ones = torch.ones(n, dtype=torch.int8, device=dev)
        e1_0 = self.energy_fn(prob1.to(dev), 0 * ones)
        e1_1 = self.energy_fn(prob1.to(dev), ones)
        e2_0 = self.energy_fn(prob2.to(dev), 0 * ones)
        e2_1 = self.energy_fn(prob2.to(dev), ones)

        # Factorized mutual information computation
        num = torch.log(
            torch.exp(-beta * (e1_0 + e2_0)) + torch.exp(-beta * (e1_1 + e2_1))
        )
        den = torch.log(
            (torch.exp(-beta * e1_0) + torch.exp(-beta * e1_1))
            * (torch.exp(-beta * e2_0) + torch.exp(-beta * e2_1))
        )
        mi = torch.sum(num - den, dim=0)
        return mi

    def forward(self, loss1: Tensor, loss2: Tensor):
        """Forward pass with gradient computation. Used during optimization (unnormalized)."""
        self.beta.data.clamp_(min=0.0)
        with torch.set_grad_enabled(True):
            cap = self.compute_mutual_information(loss1, loss2)
            self.cap = self.cap + cap.detach().to(self.cap.device)
            return cap if not self.normalize_gradients else cap / len(loss1)

    def evaluate(self, loss1: Tensor, loss2: Tensor, beta: float):
        """Evaluation without gradient computation. Tracks total_samples for normalization."""
        with torch.set_grad_enabled(False):
            cap = self.compute_mutual_information(loss1, loss2, beta=beta)
            self.cap = self.cap + cap.cpu()
            self.total_samples += len(loss1)
            return cap

    def reset(self):
        """Reset accumulated state for a new computation sequence."""
        self.cap = torch.tensor([0.0], dtype=torch.float)
        self.total_samples = 0

    @property
    def capacity(self):
        """Get the final capacity with proper normalization."""
        if self.total_samples == 0:
            return self.cap
        return self.cap / self.total_samples

    @property
    def module(self):
        """Returns the kernel itself. It helps the kernel be accessed in both DDP and non-DDP mode."""
        return self
