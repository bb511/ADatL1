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


class KLDivergenceLoss(nn.Module):
    """Kullback-Leibler divergence loss for VAE without reduction."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        kl_per_observation = -0.5 * torch.sum(
            1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1
        )
        return self.scale * kl_per_observation


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
        eta_idxs = list(range(1, 54, 3))
        # Append eta index for Et, which is the last element of the array and is 0.
        # It's not in the same place as the other etas since it is zero padded.
        eta_idxs.append(56)

        # Compute the projection from pT to pz. Use real eta for the reco.
        pT, eta = target[:, pT_idxs], target[:, eta_idxs]
        pz = pT * torch.sinh(eta)
        pT_pred = reco[:, pT_idxs]
        pz_pred = pT_pred * torch.sinh(eta)

        loss_per_observation = torch.mean(
            torch.abs(pT - pT_pred) + torch.abs(pz - pz_pred), dim=1
        )

        return self.scale * loss_per_observation


class ClassicVAELoss(nn.Module):
    def __init__(
        self, 
        alpha: float = 1.0,
        reduction: Literal['none', 'mean', 'sum'] = 'mean',
    ):
        super().__init__()
        self.reco_scale = (1 - alpha)
        self.kl_scale = alpha
        self.reduction = reduction

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


class CylPtPzVAELoss(ClassicVAELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reconstruction_loss = CylPtPzReconstructionLoss(scale=self.reco_scale)
