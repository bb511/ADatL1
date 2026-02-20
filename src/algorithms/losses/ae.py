# Loss functions that work with the vanilla AE.
from typing import Literal
import torch

from src.algorithms.losses.components import L1ADLoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.losses.components.reconstruction import HuberReconstructionLoss


class ClassicAELoss(L1ADLoss):
    """The classic AE loss, i.e., reconstruction loss between input and output."""

    def __init__(self, reduction: str = "none"):
        super().__init__(scale=None, reduction=reduction)
        self.reconstruction_loss = MSEReconstructionLoss(reduction=reduction)

    def forward(
        self,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        reco: torch.Tensor,
    ) -> torch.Tensor:
        # Get the reconstruction loss with the corersponding reduction applied.
        reco_loss = self.reconstruction_loss(target, reco, mask)

        return reco_loss


class HuberAELoss(L1ADLoss):
    """The classic AE loss, i.e., reconstruction loss between input and output."""

    def __init__(self, delta: float = 1.0, reduction: str = "none"):
        super().__init__(scale=None, reduction=reduction)
        self.reconstruction_loss = HuberReconstructionLoss(
            reduction=reduction, delta=delta
        )

    def forward(
        self,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        reco: torch.Tensor,
    ) -> torch.Tensor:
        # Get the reconstruction loss with the corersponding reduction applied.
        reco_loss = self.reconstruction_loss(target, reco, mask)

        return reco_loss
