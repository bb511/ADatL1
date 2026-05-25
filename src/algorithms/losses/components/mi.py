from __future__ import annotations

import torch
from torch import nn


class BernoulliBottleneckMILoss(nn.Module):
    """Batch estimate of I(X; Z) for Bernoulli latent units.

    The encoder output is converted to Bernoulli probabilities p(z_i=1|x).
    We estimate:

        I(X; Z) = H(Z) - H(Z | X)

    under a factorized Bernoulli approximation.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6) -> None:
        super().__init__()

        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")

        self.reduction = reduction
        self.eps = eps

    def binary_entropy(self, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(self.eps, 1.0 - self.eps)
        return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim < 2:
            raise ValueError(
                f"Expected probs with shape (batch, latent_dim, ...), got {tuple(probs.shape)}."
            )

        probs = torch.flatten(probs, start_dim=1)

        # Marginal entropy H(Z)
        marginal_probs = probs.mean(dim=0)
        h_z = self.binary_entropy(marginal_probs)

        # Conditional entropy H(Z | X)
        h_z_given_x = self.binary_entropy(probs).mean(dim=0)

        mi_per_unit = torch.clamp(h_z - h_z_given_x, min=0.0)

        if self.reduction == "sum":
            return mi_per_unit.sum()

        return mi_per_unit.mean()