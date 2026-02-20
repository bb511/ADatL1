from typing import Literal, Optional, List
import torch
import torch.nn as nn


class L1ADLoss(nn.Module):
    """
    Base class for all loss functions.

    :param scale: Scaling factor for the loss.
    :param reduction: Reduction method to apply to the loss.
        Options are 'none', 'mean', 'sum'.
    """

    name: str = "total"  # name for the logs

    def __init__(
        self,
        scale: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "none",
    ):
        super().__init__()
        self.scale = scale
        self.reduction = reduction

    def forward(self) -> torch.Tensor:
        """Forward method to compute the loss."""
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean" and loss.dim() > 0:
            return loss.mean()
        if self.reduction == "sum" and loss.dim() > 0:
            return loss.sum()
        if self.reduction == "none" and loss.dim() > 0:
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")
