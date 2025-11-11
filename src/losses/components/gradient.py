from typing import Literal

import torch

from src.losses.components import L1ADLoss


class GradientPenaltyLoss(L1ADLoss):
    """
    Gradient penalty inspired by WGAN-GP to enforce Lipshitz-smoothness
    of the reconstruction function. Encourage the norm of ∂reconstruction/∂x to
    be close to `target_norm`.
    """

    name: str = "gradient"

    def __init__(
        self,
        scale: float = 1.0,
        target_norm: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.target_norm = target_norm

    def forward(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        z: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        s: torch.LongTensor,
    ) -> torch.Tensor:
        B = target.shape[0]

        def zeros_per_sample():
            return reconstruction.new_zeros((B,), requires_grad=False)

        # Return zero penalty in eval/no-grad:
        if (
            not torch.is_grad_enabled()
            or not reconstruction.requires_grad
            or not target.requires_grad
        ):
            return self.scale * self.reduce(zeros_per_sample())

        # Compute per-sample Jacobian-vector grad of recon w.r.t. input
        recon_flat = reconstruction.view(B, -1)
        grad_outputs = torch.ones_like(recon_flat, device=reconstruction.device)
        grads = torch.autograd.grad(
            outputs=recon_flat,
            inputs=target,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]

        # If graph is disconnected (grads=None): return zero penalty
        if grads is None:
            return self.scale * self.reduce(zeros_per_sample())

        grad_norm = grads.view(B, -1).norm(2, dim=1)          
        penalty  = (grad_norm - self.target_norm).pow(2) 
        return self.scale * self.reduce(penalty)