# Variational AE loss function.

from typing import Literal

import torch

from src.losses.components import L1ADLoss
from src.losses.components.reconstruction import ReconstructionLoss
from src.losses.components.reconstruction import CylPtPzReconstructionLoss
from src.losses.components.kl import KLDivergenceLoss

from src.utils import pylogger
from colorama import Fore, Back, Style

log = pylogger.RankedLogger(__name__)


class ClassicVAELoss(L1ADLoss):
    """The conventional variational AE loss.

        L = MSE + scale*KL_div

    :param scale: Float to scale the total loss with.
    :param kl_scale: Float to scale the kl_scale with. IF kl_scale is set dynamically
        then this parameter is the final kl_scale that should be reached.
    :param reduct: String denoting the type of reduction used on the loss function. The
        separate loss functions return loss values per event. This is then aggregated
        by computing the mean over the batch, or the sum.
    """

    def __init__(self, scale: float = 1, kl_scale: float = 1, reduct: str = "none"):
        super().__init__(scale=scale, reduction=reduct)
        self.kl_scale_final = float(kl_scale)
        self.reco_loss = ReconstructionLoss(reduction=reduct)
        self.kl_loss = KLDivergenceLoss(scale=self.kl_scale_final, reduction=reduct)

    def forward(
        self,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        reconstruction: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        kl_scale: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The kl_scale argument allows for dynamically setting the kl_scale during
        # the training of the VAE.
        if not kl_scale is None:
            self.kl_loss.set_scale(float(kl_scale))

        reco_loss = self.reco_loss(target, reconstruction, mask)
        kl_raw, kl_scaled = self.kl_loss(z_mean, z_log_var)
        total_loss = self.scale*(reco_loss + kl_scaled)

        return reco_loss, kl_raw, kl_scaled, total_loss


class AxoV4Loss(ClassicVAELoss):
    """The conventional VAE loss, but the reconstruction is done in a special way.

    This assumes that the input is composed only of pT, eta, and phi features.
    Moreover, it assumes that the phi feature of each object is every third element
    in the torch Tensor that the loss function receives. Hence, THIS IS ONLY USABLE
    WITH A VERY PARTICULAR TYPE OF DATA PROCESSING.
    """

    def __init__(self, scale: float = 1, kl_scale: float = 1, reduct: str = "none"):
        super().__init__(scale=scale, kl_scale=kl_scale, reduct=reduct)
        log.warn(
            Fore.YELLOW
            + "Using AxoV4Loss. Expecting that the mlready data config is axov4."
            + "Moreover, expecting that the awkward2torch config is also axov4."
        )
        self.reco_loss = CylPtPzReconstructionLoss(reduction=reduct)
