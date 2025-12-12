from typing import Literal
import torch

from src.losses.components import L1ADLoss


class SVDDLoss(L1ADLoss):
    """
    Loss function for SVDD.

    :param scale: Scaling factor for the loss.
    :param nu: Hyperparameter for SVDD (0 < nu <= 1).
    :param weight_decay: Weight decay for latent representations.
    :param soft_boundary: Whether to use soft-boundary SVDD.
    :param reduction: Reduction method to apply to the loss ("none", "mean", "sum").
    """

    def __init__(
        self,
        scale: float = 1.0,
        nu: float = 0.1,
        weight_decay: float = 1e-6,
        soft_boundary: bool = False,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.nu = nu
        self.weight_decay = weight_decay
        self.soft_boundary = soft_boundary

        if self.soft_boundary:
            self.register_buffer("radius", torch.tensor(1.0))

    def forward(
        self,
        distances: torch.Tensor,
        z: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        if self.soft_boundary:
            scores = distances - self.radius**2
            hinge_term = torch.mean(torch.max(torch.zeros_like(scores), scores))
            loss = self.radius**2 + (1 / self.nu) * hinge_term

            # Update radius (with gradient detached)
            with torch.no_grad():
                self.radius = torch.quantile(torch.sqrt(distances), 1 - self.nu).to(self.radius.device)

            loss = loss.expand(distances.shape[0])

        else:
            loss = distances
            if self.weight_decay > 0:
                reg_loss = self.weight_decay * 0.5 * torch.sum(z**2, dim=1)
                loss = loss + reg_loss

        return self.scale * self.reduce(loss)