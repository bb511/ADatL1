from typing import Tuple, Literal

import torch

from src.losses.components import L1ADLoss


class MILoss(L1ADLoss):
    """
    Mutual Information loss for VAE models.

    Args:
        eps: Small constant for numerical stability.
        scale: Scaling factor for the loss.
        reduction: Reduction method to apply to the loss. Options are 'none', 'mean', 'sum'.
    """

    name: str = "mi"

    def __init__(
        self,
        eps: float = 1e-7,
        scale: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.eps = eps

    @staticmethod
    def mutual_information_bernoulli_loss(
        s: torch.Tensor,
        z: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Calculate mutual information loss for Bernoulli-distributed latent variables.

        This loss encourages the latent representation z to be informative about
        the signal/background label s.

        :param s: Signal/background labels (batch_size,)
        :param z: Latent representations (batch_size, latent_dim)
        :param eps: Small constant for numerical stability
        """
        # Convert z to probabilities using sigmoid
        z_probs = torch.sigmoid(z)

        # Calculate entropy H(Z) = -p*log(p) - (1-p)*log(1-p)
        entropy_z = -(
            z_probs * torch.log(z_probs + eps)
            + (1 - z_probs) * torch.log(1 - z_probs + eps)
        )
        entropy_z = entropy_z.sum(dim=1).mean()

        # Calculate conditional entropy H(Z|S)
        conditional_entropy = 0
        n_total = z.shape[0]
        mask_signal = s == 1 if s is not None else torch.zeros(n_total, dtype=torch.bool)
        if mask_signal.any():
            z_signal = z[mask_signal]
            z_signal_probs = torch.sigmoid(z_signal)
            h_z_given_s1 = -(
                z_signal_probs * torch.log(z_signal_probs + eps)
                + (1 - z_signal_probs) * torch.log(1 - z_signal_probs + eps)
            )
            conditional_entropy += (
                mask_signal.sum().float() / n_total
            ) * h_z_given_s1.sum(dim=1).mean()

        mask_background = s == 0 if s is not None else torch.ones(n_total, dtype=torch.bool)
        if mask_background.any():
            z_background = z[mask_background]
            z_background_probs = torch.sigmoid(z_background)
            h_z_given_s0 = -(
                z_background_probs * torch.log(z_background_probs + eps)
                + (1 - z_background_probs) * torch.log(1 - z_background_probs + eps)
            )
            conditional_entropy += (
                mask_background.sum().float() / n_total
            ) * h_z_given_s0.sum(dim=1).mean()

        # I(Z;S) = H(Z) - H(Z|S)
        return -(entropy_z - conditional_entropy)  # minimize -MI

    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        z: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward method to compute the MI loss."""
        mi_loss_batch = self.mutual_information_bernoulli_loss(s, z, eps=self.eps)
        mi_loss_per_obs = mi_loss_batch * torch.ones(z.shape[0], device=z.device)
        return self.scale * self.reduce(mi_loss_per_obs)