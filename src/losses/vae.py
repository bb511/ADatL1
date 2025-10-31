# Variational AE loss function.

from typing import Literal

import torch
import torch.nn as nn

from src.losses.components.reconstruction import ReconstructionLoss
from src.losses.components.reconstruction import CylPtPzReconstructionLoss
from src.losses.components.kl import KLDivergenceLoss


class ClassicVAELoss(nn.Module):
    """The conventional variational AE loss.

    :param scale: Float between 0 and 1 that establishes the scale between the KL div
        loss and the reconstruction loss.
    :param reduct: String denoting the type of reduction used on the loss function. The
        separate loss functions return loss values per event. This is then aggregated
        by computing the mean over the batch, or the sum.
    """
    def __init__(self, scale: float, reduct: Literal["none", "mean", "sum"] = "none"):
        super().__init__()
        self.reduction = reduct
        self.reco_scale = 1 - scale
        self.kl_scale = scale

        self.reco_loss = ReconstructionLoss(scale=self.reco_scale)
        self.kl_loss = KLDivergenceLoss(scale=self.kl_scale)

    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        reco_loss = self.reco_loss(target, reconstruction)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        total_loss = reco_loss + kl_loss

        return self.reduce(reco_loss), self.reduce(kl_loss), self.reduce(total_loss)

    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Applies a reduction operation to a loss."""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

        return loss


class AxoV4Loss(ClassicVAELoss):
    """The conventional VAE loss, but the reconstruction is done in a special way.

    This assumes that the input is composed only of pT, eta, and phi features.
    Moreover, it assumes that the phi feature of each object is every third element
    in the torch Tensor that the loss function receives. Hence, THIS IS ONLY USABLE
    WITH A VERY PARTICULAR TYPE OF DATA PROCESSING.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reco_loss = CylPtPzReconstructionLoss(scale=self.reco_scale)
