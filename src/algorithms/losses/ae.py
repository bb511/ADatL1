# Loss functions that work with the vanilla AE.
from typing import Literal
import torch

from src.algorithms.losses.components import ADLoss
from src.algorithms.losses.components.mi import BernoulliBottleneckMILoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.algorithms.losses.components.reconstruction import HuberReconstructionLoss


class ClassicAELoss(ADLoss):
    """The classic AE loss, i.e., reconstruction loss between input and output."""
    def __init__(self, reduction: str = "none"):
        super().__init__(scale=None, reduction=reduction)
        self.reconstruction_loss = MSEReconstructionLoss(reduction=reduction)

    def forward(
        self,
        target: torch.Tensor,
        reco: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Get the reconstruction loss with the corersponding reduction applied.
        reco_loss = self.reconstruction_loss(target, reco, mask)

        return reco_loss


class HuberAELoss(ADLoss):
    """The classic AE loss, i.e., reconstruction loss between input and output."""
    def __init__(self, delta: float = 1.0, reduction: str = "none"):
        super().__init__(scale=None, reduction=reduction)
        self.reco_loss = HuberReconstructionLoss(reduction=reduction, delta=delta)

    def forward(
        self,
        target: torch.Tensor,
        reco: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Get the reconstruction loss with the corersponding reduction applied.
        reco_loss = self.reco_loss(target, reco, mask)

        return reco_loss

class PileupMIAELoss(HuberAELoss):
    """Huber reconstruction loss combined with a Bernoulli MI regulariser.

    total = reco_loss + γ · MI_loss

    The MI term is computed via :class:`BernoulliBottleneckMILoss` (see
    ``src.algorithms.losses.components.mi``).
    """

    def __init__(
        self,
        gamma: float = 1.0,
        delta: float = 1.0,
        mi_reduction: str = "sum",
        reduction: str = "none",
    ):
        super().__init__(delta=delta, reduction=reduction)
        self.gamma = gamma
        self.mi_loss = BernoulliBottleneckMILoss(reduction=mi_reduction)

    def forward(
        self,
        target: torch.Tensor,
        reco: torch.Tensor,
        mask: torch.Tensor | None,
        probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reco_loss = super().forward(target, reco, mask)

        if probs is not None and self.gamma != 0.0:
            mi = self.mi_loss(probs)
        else:
            mi = reco_loss.new_tensor(0.0)

        return self.gamma * mi