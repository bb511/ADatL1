from typing import Literal
import torch

from src.losses import L1ADLoss
from src.losses.kl import KLDivergenceLoss
from src.losses.mmd import MMDLoss


class BetaVAELoss(L1ADLoss):
    """
    Beta-VAE style regularization with cyclical annealing.
    Helps with disentanglement and prevents posterior collapse.

    :param scale: Scaling factor for the loss.
    :param beta: Weight for the KL divergence term.
    :param gamma: Weight for the capacity control term.
    :param max_capacity: Maximum capacity for the latent bottleneck.
    :param capacity_leadin: Number of steps to reach max_capacity.
    :param distance: Distance metric to use ("kl", "mmd").
    :param reduction: Reduction method to apply to the loss ("none", "mean", "sum").
    """

    name: str = "betavae"
    
    def __init__(
        self,
        scale: float = 1.0,
        beta: float = 4.0,
        gamma: float = 1.0,
        max_capacity: float = 25.0,
        capacity_leadin: int = 100000,
        distance: Literal["kl", "mmd"] = "kl",
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.beta = beta
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.capacity_leadin = capacity_leadin

        self.steps = 0
        if distance == "mmd":
            self.distance = KLDivergenceLoss(scale=1.0, reduction="none")
        else:
            self.distance = MMDLoss(scale=1.0, reduction="none")

    def forward(
        self,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute β-VAE loss with optional capacity control."""
        
        # Compute distance-based loss
        loss = self.distance(z_mean=z_mean, z_log_var=z_log_var)

        # Apply capacity increase (for avoiding posterior collapse)
        if self.capacity_leadin > 0:
            capacity = min(
                self.max_capacity,
                self.max_capacity * self.steps / self.capacity_leadin
            )
            loss = self.gamma * torch.abs(loss - capacity)
        else:
            loss = self.beta * loss
            
        self.steps += 1
        return self.scale * self.reduce(loss)