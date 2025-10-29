# Variational AE loss function.

from src.losses import L1ADLoss
from src.losses.components.reconstruction import ReconstructionLoss
from src.losses.components.reconstruction import CylPtPzReconstructionLoss
from src.losses.components.kl import KLDivergenceLoss


class ClassicVAELoss(L1ADLoss):
    def __init__(
        self,
        scale: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction, scale=scale)

        self.reco_scale = 1 - self.scale
        self.kl_scale = self.scale

        self.reconstruction_loss = ReconstructionLoss(scale=self.reco_scale)
        self.kl_loss = KLDivergenceLoss(scale=self.kl_scale)

    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        reduction: Literal["none", "mean", "sum"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        reduction = reduction if reduction is not None else self.reduction

        # Get per-observation losses
        reco_loss_per_obs = self.reconstruction_loss(target, reconstruction)
        kl_loss_per_obs = self.kl_loss(z_mean, z_log_var)
        total_loss_per_obs = reco_loss_per_obs + kl_loss_per_obs

        # Apply reduction
        if reduction == "none":
            return total_loss_per_obs, reco_loss_per_obs, kl_loss_per_obs
        elif reduction == "mean":
            return (
                torch.mean(total_loss_per_obs),
                torch.mean(reco_loss_per_obs),
                torch.mean(kl_loss_per_obs),
            )
        elif reduction == "sum":
            return (
                torch.sum(total_loss_per_obs),
                torch.sum(reco_loss_per_obs),
                torch.sum(kl_loss_per_obs),
            )
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")


class AxoV4Loss(ClassicVAELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reconstruction_loss = CylPtPzReconstructionLoss(scale=self.reco_scale)
