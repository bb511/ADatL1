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
    """BinaryMI-style Bernoulli mutual-information regulariser for the AE.

    This wrapper computes only the MI term. Reconstruction loss is computed
    separately in AE.model_step.
    """

    def __init__(
        self,
        mi_temperature: float = 6.0,
        input_is_logits: bool = True,
        eps: float = 1e-20,
        use_float64: bool = True,
    ) -> None:
        super().__init__(scale=1.0, reduction="none")

        self.mi_loss = BernoulliMILoss(
            temperature=mi_temperature,
            eps=eps,
            input_is_logits=input_is_logits,
            use_float64=use_float64,
        )

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        return self.mi_loss(latent=latent, sensitive=sensitive)