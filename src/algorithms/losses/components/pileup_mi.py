# src/algorithms/losses/components/pileup_mi.py

from __future__ import annotations

import torch
from torch import nn


class BernoulliPileupMILoss(nn.Module):
    """Estimate sum_i I(T_i; S) for Bernoulli latent units.

    T_i is one stochastic Bernoulli latent neuron.
    S is the sensitive / nuisance variable.

    For pileup decorrelation:
        S = nPV_true

    probs has shape:
        (batch, latent_dim)

    s has shape:
        (batch,)
    """

    def __init__(
        self,
        reduction: str = "sum",
        eps: float = 1e-6,
        bin_width: int | None = None,
    ) -> None:
        super().__init__()

        if reduction not in {"sum", "mean", "none"}:
            raise ValueError(
                f"reduction must be one of 'sum', 'mean', 'none', got {reduction}"
            )

        self.reduction = reduction
        self.eps = eps
        self.bin_width = bin_width

    def binary_entropy(self, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(self.eps, 1.0 - self.eps)
        return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))

    def _prepare_sensitive(self, s: torch.Tensor) -> torch.Tensor:
        if s.ndim > 1:
            s = s.view(s.shape[0], -1)[:, 0]

        s = s.detach().to(dtype=torch.long)

        if self.bin_width is not None and self.bin_width > 1:
            s = torch.div(s, self.bin_width, rounding_mode="floor")

        return s

    def forward(self, probs: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if probs.ndim < 2:
            raise ValueError(
                f"Expected probs with shape (batch, latent_dim, ...), "
                f"got {tuple(probs.shape)}."
            )

        probs = torch.flatten(probs, start_dim=1)
        s = self._prepare_sensitive(s).to(device=probs.device)

        if probs.shape[0] != s.shape[0]:
            raise ValueError(
                f"Batch mismatch: probs has batch size {probs.shape[0]}, "
                f"but sensitive variable has batch size {s.shape[0]}."
            )

        # H(T_i)
        marginal_probs = probs.mean(dim=0)
        h_t = self.binary_entropy(marginal_probs)

        # H(T_i | S) = sum_s p(s) H(T_i | S=s)
        h_t_given_s = torch.zeros_like(h_t)

        unique_s = torch.unique(s)

        for value in unique_s:
            mask = s == value
            count = mask.sum()

            if count == 0:
                continue

            weight = count.to(dtype=probs.dtype) / probs.shape[0]
            conditional_probs = probs[mask].mean(dim=0)
            h_t_given_s = h_t_given_s + weight * self.binary_entropy(conditional_probs)

        mi_per_unit = torch.clamp(h_t - h_t_given_s, min=0.0)

        if self.reduction == "none":
            return mi_per_unit

        if self.reduction == "mean":
            return mi_per_unit.mean()

        return mi_per_unit.sum()