# Loss functions that work with the vanilla AE.

from src.losses import L1ADLoss
from src.losses.components.reconstruction_loss import ReconstructionLoss


class ClassicAELoss(L1ADLoss):
    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction)
        self.reconstruction_loss = ReconstructionLoss(scale=self.reco_scale)

    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        reduction: Literal["none", "mean", "sum"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        reduction = reduction if reduction is not None else self.reduction

        # Get per-observation losses
        reco_loss_per_obs = self.reconstruction_loss(target, reconstruction)

        # Apply reduction
        if reduction == "none":
            return reco_loss_per_obs
        elif reduction == "mean":
            return torch.mean(reco_loss_per_obs)
        elif reduction == "sum":
            return (
                torch.sum(reco_loss_per_obs),
            )
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")
