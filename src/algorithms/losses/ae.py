# Loss functions that work with the vanilla AE.
import torch

from src.algorithms.losses.components import ADLoss
from src.algorithms.losses.components.bernoulli_mi import BernoulliMILoss
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
    def __init__(self, delta: float = 1.0, scale: float = 1.0, reduction: str = "none"):
        super().__init__(scale=scale, reduction=reduction)
        self.reco_loss = HuberReconstructionLoss(reduction=reduction, delta=delta, scale=scale)

    def forward(
        self,
        target: torch.Tensor,
        reco: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Get the reconstruction loss with the corersponding reduction applied.
        reco_loss = self.reco_loss(target, reco, mask)

        return reco_loss

class PileupMIAELoss(ADLoss):
    """Huber reconstruction loss combined with a Bernoulli MI regulariser.

    total = reco_loss + γ · MI_loss

    The MI term is computed via :class:`BernoulliBottleneckMILoss` (see
    ``src.algorithms.losses.components.mi``).
    """

    def __init__(self, mi_temperature: float = 6.0, mi_reduction: str = "sum") -> None:
        super().__init__(scale=1.0, reduction="sum")
        # BernoulliMILoss does not accept a `reduction` parameter anymore; only
        # provide the temperature. Keep `mi_reduction` for API compatibility.
        self.mi_loss = BernoulliMILoss(temperature=mi_temperature, input_is_logits=True)

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        return self.mi_loss(latent=latent,sensitive=sensitive,)