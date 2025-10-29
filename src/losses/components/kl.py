from typing import Literal
import torch

from src.losses import L1ADLoss


class KLDivergenceLoss(L1ADLoss):
    """
    Kullback-Leibler divergence loss for VAE.

    :param scale: Scaling factor for the loss.
    :param reduction: Reduction method to apply to the loss ("none", "mean", "sum").
    """

    name: str = "kl"

    def __init__(
            self,
            scale: float = 1.0,
            reduction: Literal["none", "mean", "sum"] = "mean"
        ):
        super().__init__(scale=scale, reduction=reduction)

    def forward(
            self,
            z_mean: torch.Tensor,
            z_log_var: torch.Tensor,
            **kwargs
        ) -> torch.Tensor:

        kl_per_observation = -0.5 * torch.mean(
            1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1
        )
        return self.scale * self.reduce(kl_per_observation)