from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for VAE without reduction."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, target: torch.Tensor, reco: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(target, reco, reduction="none")

        # Reduce across all dimensions except batch dimension
        loss_per_observation = torch.mean(mse_loss, dim=tuple(range(1, mse_loss.dim())))
        return self.scale * loss_per_observation


class ClassicAELoss(nn.Module):
    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__()
        self.reduction = reduction
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


class CylPtPzReconstructionLoss(nn.Module):
    """Convert the pT, eta, phi input into pT, pz and compute loss there.

    This trick is done since the training did not converge otherwise.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, target: torch.Tensor, reco: torch.Tensor) -> torch.Tensor:
        # Indexes of where the pTs of the objects are, given the axo v4 input data.
        pT_idxs = list(range(0, 55, 3))
        # Indexes of where the eta of the objects are, given the axo v4 input data.
        eta_idxs = list(range(1, 56, 3))

        # Compute the projection from pT to pz. Use real eta for the reco.
        pT, eta = target[:, pT_idxs], target[:, eta_idxs]
        pz = pT * torch.sinh(eta)
        MET_phi = target[:, -1]
        pT_pred = reco[:, pT_idxs]
        pz_pred = pT_pred * torch.sinh(eta)
        MET_phi_pred = reco[:, -1]

        loss_per_observation = torch.mean(
            torch.abs(pT - pT_pred) + torch.abs(pz - pz_pred), dim=1
        )

        return self.scale * (loss_per_observation + torch.abs(MET_phi - MET_phi_pred))


class CylLoss(ClassicAELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reconstruction_loss = CylPtPzReconstructionLoss(scale=self.reco_scale)
