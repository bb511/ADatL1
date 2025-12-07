from typing import Literal, Optional, List
import torch
import torch.nn as nn


class L1ADLoss(nn.Module):
    """
    Base class for all loss functions.

    :param scale: Scaling factor for the loss.
    :param reduction: Reduction method to apply to the loss.
        Options are 'none', 'mean', 'sum'.
    """

    name: str = "total" # name for the logs
    
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
        y: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward method to compute the loss.

        :param target: Ground truth tensor.
        :param reconstruction: Reconstructed tensor from the model.
        :param z: Latent representations (batch_size, latent_dim).
        :param z_mean: Mean of the latent distribution (for VAE).
        :param z_log_var: Log variance of the latent distribution (for VAE).
        :param y: Labels: < 0 for background simulations, 0 for zerobias, > 0 for signal simulations (batch_size,).
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")
    
    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean" and loss.dim() > 0:
            return loss.mean()
        elif self.reduction == "sum" and loss.dim() > 0:
            return loss.sum()
        return loss


class MultiLoss(L1ADLoss):
    """
    Wrapper to combine multiple loss functions with individual scaling.
    
    :param reduction: Reduction method to apply to the total loss.
        Options are 'none', 'mean', 'sum'.
    :param list_losses: Optional list of loss names to add for backpropagation.
        If None, all provided losses are used.
    :param losses: Keyword arguments where keys are loss names and values are
        loss module instances.
    """
    
    def __init__(
        self,
        scale: int = 1.0,
        reduction: Literal["none", "mean", "sum"] = "none",
        list_losses: Optional[List[str]] = None,
        **losses,
    ):
        super().__init__(scale=scale, reduction=reduction)
        # Eliminate reduction of individual losses
        for loss_fn in losses.values():
            loss_fn.reduction = "none"

        self.list_losses = list_losses or list(losses.keys())
        self.losses = nn.ModuleDict({
            name: loss_fn
            for name, loss_fn in losses.items()
            if isinstance(loss_fn, nn.Module)
        })
        
    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        z: torch.Tensor,
        y: torch.LongTensor,
    ) -> torch.Tensor:

        self.values = {}
        for lname, loss_fn in self.losses.items():
            loss = loss_fn(**{
                'target': target,
                'reconstruction': reconstruction,
                'z_mean': z_mean,
                'z_log_var': z_log_var,
                'z': z,
                'y': y,
            })
            self.values[lname] = loss
        
        list_lvalues = [
            lvalue
            for lname, lvalue in self.values.items()
            if lname in self.list_losses
        ]
        return self.reduce(
            torch.stack(list_lvalues, dim=0).sum(dim=0)
        )

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
