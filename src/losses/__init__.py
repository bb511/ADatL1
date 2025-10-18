from typing import Literal
import torch
import torch.nn as nn


class L1ADLoss(nn.Module):
    """
    Base class for all loss functions.

    :param scale: Scaling factor for the loss.
    :param reduction: Reduction method to apply to the loss. Options are 'none', 'mean', 'sum'.
    """
    
    def __init__(
            self,
            scale: float = 1.0,
            reduction: Literal["none", "mean", "sum"] = "none",
        ):
        super().__init__()
        self.scale = scale
        self.reduction = reduction

    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        z: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        s: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward method to compute the loss.

        :param target: Ground truth tensor.
        :param reconstruction: Reconstructed tensor from the model.
        :param z: Latent representations (batch_size, latent_dim).
        :param z_mean: Mean of the latent distribution (for VAE).
        :param z_log_var: Log variance of the latent distribution (for VAE).
        :param s: Signal/background labels (batch_size,).
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")
    
    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean" and loss.dim() > 0:
            return loss.mean()
        elif self.reduction == "sum" and loss.dim() > 0:
            return loss.sum()
        return loss


class MultiLoss(nn.Module):
    """Wrapper to combine multiple loss functions with individual scaling."""
    
    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "none",
        **losses,
    ):
        super().__init__()
        # Eliminate reduction of individual losses
        for loss_fn in losses.values():
            loss_fn.reduction = "none"

        self.losses = nn.ModuleDict({
            name: loss_fn
            for name, loss_fn in losses.items()
            if isinstance(loss_fn, nn.Module)
        })
        self.reduction = reduction
        
    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        z: torch.Tensor,
        s: torch.LongTensor,
    ) -> torch.Tensor:

        self.values = {}
        for name, loss_fn in self.losses.items():
            loss = loss_fn(**{
                'target': target,
                'reconstruction': reconstruction,
                'z_mean': z_mean,
                'z_log_var': z_log_var,
                'z': z,
                's': s,
            })
            self.values[f'{name}'] = loss
        
        total = torch.stack(
            list(self.values.values()),
            dim=0
        ).sum(dim=0)
        return self.reduce(total)
        
    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce the values and return total only."""
        if self.reduction == "mean":
            loss = loss.mean()
            self.values = {
                k: v.mean() if v.dim() > 0 else v
                for k, v in self.values.items()
            }
        elif self.reduction == "sum":
            loss = loss.sum()
            self.values = {
                k: v.sum() if v.dim() > 0 else v
                for k, v in self.values.items()
            }
        return loss