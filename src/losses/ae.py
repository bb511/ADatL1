# Loss functions that work with the vanilla AE.
from typing import Literal

import torch
import torch.nn as nn

from src.losses.components.reconstruction import ReconstructionLoss


class ClassicAELoss(nn.Module):
    """The classic AE loss, i.e., reconstruction loss between input and output."""
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "none"):
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss(reduction=reduction)

    def forward(self, target: torch.Tensor, reco: torch.Tensor) -> torch.Tensor:
        # Get the reconstruction loss with the corersponding reduction applied.
        reco_loss = self.reconstruction_loss(target, reco)

        return reco_loss
