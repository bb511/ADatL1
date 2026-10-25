# Variational AE loss function.
import math

import torch

from src.algorithms.losses.components import ADLoss
from src.algorithms.losses.components.kl import KLDivergenceLoss
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss


class ClassicVAELoss(ADLoss):
    """The conventional variational AE loss.

    L = MSE + scale*KL_div

    :param scale: Float to scale the total loss with.
    :param kl_scale: Float to scale the kl_scale with. IF kl_scale is set dynamically
        then this parameter is the final kl_scale that should be reached.
    :param reduciont: String denoting the type of reduction used on the loss function.
        The separate loss functions return loss values per event. This is then
        aggregated by computing the mean over the batch, or the sum.
    """

    def __init__(
        self,
        scale: float = 1.0,
        reco_scale: float = 1.0,
        kl_scale: float = 1.0,
        reduction: str = "none",
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.kl_scale_final = float(kl_scale)
        self.reco_loss = MSEReconstructionLoss(scale=reco_scale, reduction=reduction)
        self.kl_loss = KLDivergenceLoss(scale=self.kl_scale_final, reduction=reduction)

    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor | None,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        kl_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The kl_scale argument allows for dynamically setting the kl_scale during
        # the training of the VAE.
        if kl_scale is not None:
            self.kl_loss.set_scale(float(kl_scale))

        reco_loss = self.reco_loss(target, reconstruction, mask)
        kl_raw, kl_scaled = self.kl_loss(z_mean, z_log_var)
        total_loss = self.scale * (reco_loss + kl_scaled)

        return reco_loss, kl_raw, kl_scaled, total_loss


def gaussian_neg_elbo(
    mse: torch.Tensor,
    kl: torch.Tensor,
    beta: float,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return -ELBO per event, in nats, under the Gaussian decoder the VAE loss implies.

    With n observed features and the decoder variance sigma^2 = n beta / 2 per feature,
    -log p(x|z) = MSE / beta + (n / 2) log(pi n beta), so that

        -ELBO = (MSE + beta KL) / beta + (n / 2) log(pi n beta),

    the loss MSE + beta KL divided by beta, up to a constant per event. ``mse`` and
    ``kl`` have shape (batch,); n counts the observed entries of ``mask``, or every
    entry of an event of ``target`` without one, and is at least 1, as in the MSE.
    """
    # In float64, cast back at the end: beta, as small as 3e-5 on the chamber, is
    # inexact in float32, which biases MSE / beta (up to 10^3 nats) by about 5e-8
    # relative.
    mse_64, kl_64 = mse.double(), kl.double()
    if mask is None:
        n_observed = torch.full_like(mse_64, float(target[0].numel()))
    else:
        observed = mask.reshape(mask.shape[0], -1).double()
        n_observed = observed.sum(dim=1).clamp_min(1.0)
    constant = 0.5 * n_observed * torch.log(math.pi * n_observed * beta)
    return ((mse_64 + beta * kl_64) / beta + constant).to(mse.dtype)
