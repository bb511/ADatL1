from typing import Tuple, Literal, Optional
import torch

from src.models.losses.vae import ClassicVAELoss, AxoV4Loss


class MIVAELoss(ClassicVAELoss):
    """
    Loss function for MI-VAE.
    
    Args:
        alpha: Weight for KL divergence term.
        gamma: Weight for mutual information term.
        eps: Small constant for numerical stability in MI loss.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-7,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(alpha=alpha, reduction=reduction)
        self.gamma = gamma
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
        
        Args:
            s: Signal/background labels (batch_size,)
            z: Latent representations (batch_size, latent_dim)
            eps: Small constant for numerical stability
        
        Returns:
            Mutual information loss scalar
        """
        # Convert z to probabilities using sigmoid
        z_probs = torch.sigmoid(z)
        
        # Calculate entropy H(Z) = -p*log(p) - (1-p)*log(1-p)
        entropy_z = - (
            z_probs * torch.log(z_probs + eps) + 
            (1 - z_probs) * torch.log(1 - z_probs + eps)
        )
        entropy_z = entropy_z.sum(dim=1).mean()
        
        # Calculate conditional entropy H(Z|S)        
        conditional_entropy = 0
        n_total = z.shape[0]
        mask_signal = (s == 1)
        if mask_signal.any():
            z_signal = z[mask_signal]
            z_signal_probs = torch.sigmoid(z_signal)
            h_z_given_s1 = - (
                z_signal_probs * torch.log(z_signal_probs + eps) + 
                (1 - z_signal_probs) * torch.log(1 - z_signal_probs + eps)
            )
            conditional_entropy += (mask_signal.sum().float() / n_total) * h_z_given_s1.sum(dim=1).mean()
        
        mask_background = (s == 0)
        if mask_background.any():
            z_background = z[mask_background]
            z_background_probs = torch.sigmoid(z_background)
            h_z_given_s0 = - (
                z_background_probs * torch.log(z_background_probs + eps) + 
                (1 - z_background_probs) * torch.log(1 - z_background_probs + eps)
            )
            conditional_entropy += (mask_background.sum().float() / n_total) * h_z_given_s0.sum(dim=1).mean()
        
        # I(Z;S) = H(Z) - H(Z|S)
        return - (entropy_z - conditional_entropy) # minimize -MI
            
    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        reduction: Literal["none", "mean", "sum"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute MI-VAE loss.
        
        Args:
            target: Original input
            reconstruction: Reconstructed input
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution
            z: Latent representation (needed for MI loss)
            s: Signal/background labels (needed for MI loss)
            reduction: Type of reduction
            
        Returns:
            Tuple of (total_loss, reco_loss, kl_loss, mi_loss)
        """
        reduction = reduction if reduction is not None else self.reduction
        
        # Get standard VAE losses (per observation)
        total_loss, reco_loss, kl_loss = super().forward(
            target, reconstruction, z_mean, z_log_var, reduction="none"
        )
        
        # Compute MI loss if enabled
        if z is not None and s is not None:
            # MI loss is computed over the batch
            mi_loss_batch = self.mutual_information_bernoulli_loss(s, z, eps=self.eps)
            # Scale by gamma and distribute across observations
            mi_loss_per_obs = self.gamma * mi_loss_batch * torch.ones_like(total_loss)
            total_loss = total_loss + mi_loss_per_obs
        else:
            mi_loss_per_obs = torch.zeros_like(total_loss)
        
        # Apply reduction
        if reduction == "none":
            return total_loss, reco_loss, kl_loss, mi_loss_per_obs
        elif reduction == "mean":
            return (
                torch.mean(total_loss),
                torch.mean(reco_loss),
                torch.mean(kl_loss),
                torch.mean(mi_loss_per_obs),
            )
        elif reduction == "sum":
            return (
                torch.sum(total_loss),
                torch.sum(reco_loss),
                torch.sum(kl_loss),
                torch.sum(mi_loss_per_obs),
            )
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")


class AxoV4MIVAELoss(AxoV4Loss, MIVAELoss):
    """MI-VAE loss with AxoV4 cylindrical coordinate reconstruction."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-7,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(alpha=alpha, reduction=reduction)
        self.gamma = gamma
        self.eps = eps

    forward = MIVAELoss.forward