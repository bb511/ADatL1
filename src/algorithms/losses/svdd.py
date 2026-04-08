from typing import Literal
import torch

from src.algorithms.losses.components import L1ADLoss


class SVDDLoss(L1ADLoss):
    """Loss function for Deep SVDD.

    :param scale: Global scaling factor applied to the loss.
    :param nu: Soft-boundary hyperparameter, with 0 < nu <= 1.
    :param weight_decay: L2 penalty applied to the latent representation in one-class
        mode.
    :param soft_boundary: Bool whether to use the soft-boundary Deep SVDD objective.
    :param reduction: Reduction method to apply to the loss ("none", "mean", "sum").
    """

    def __init__(
        self,
        scale: float = 1.0,
        nu: float = 0.1,
        weight_decay: float = 1e-6,
        soft_boundary: bool = False,
        reduction: Literal["none", "mean", "sum"] = "none",
    ):
        super().__init__(scale=scale, reduction=reduction)

        if not (0.0 < nu <= 1.0):
            raise ValueError(f"nu must satisfy 0 < nu <= 1, got {nu}.")

        self.nu = float(nu)
        self.weight_decay = float(weight_decay)
        self.soft_boundary = soft_boundary

        self.register_buffer("radius", torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        distances: torch.Tensor,
        z: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the Deep SVDD loss.

        :param distances: Tensor of shape [B] containing squared distances to the center.
        :param z: Tensor of shape [B, D] containing latent representations.
        :returns: distance_raw, reg_scaled, total_loss
        """
        if distances.ndim != 1:
            raise ValueError(
                f"distances must have shape [B], got shape {tuple(distances.shape)}"
            )

        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, D], got shape {tuple(z.shape)}")

        if z.shape[0] != distances.shape[0]:
            raise ValueError(
                f"Batch size mismatch: distances has {distances.shape[0]} samples, "
                f"z has {z.shape[0]} samples."
            )

        distance_raw = distances

        if self.soft_boundary:
            scores = distances - self.radius.square()
            hinge = torch.clamp(scores, min=0.0)

            # Per-sample soft-boundary contribution.
            reg_raw = self.radius.square() + (1.0 / self.nu) * hinge

            with torch.no_grad():
                new_radius = torch.quantile(torch.sqrt(distances.detach()), 1 - self.nu)
                self.radius.copy_(new_radius.to(self.radius.device))

        else:
            if self.weight_decay > 0.0:
                reg_raw = 0.5 * self.weight_decay * torch.sum(z.square(), dim=1)
            else:
                reg_raw = torch.zeros_like(distances)

        reg_scaled = reg_raw
        total_loss = self.scale * (distance_raw + reg_scaled)

        return (
            self.reduce(distance_raw),
            self.reduce(reg_scaled),
            self.reduce(total_loss),
        )
