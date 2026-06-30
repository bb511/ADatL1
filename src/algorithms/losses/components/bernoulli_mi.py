from __future__ import annotations

import torch
from torch import nn


class BernoulliMILoss(nn.Module):
    """HepInfo-compatible Bernoulli mutual-information estimator.

    Computes
        I(L; S) = H(L) - sum_s p(S=s) H(L | S=s)

    where each latent activation is interpreted as a Bernoulli logit and mapped
    to p = sigmoid(temperature * activation) before the Bernoulli entropy is
    estimated from the batch mean probability.
    """

    def __init__(
        self,
        temperature: float = 6.0,
        eps: float = 1e-20,
        input_is_logits: bool = True,
        use_float64: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.input_is_logits = bool(input_is_logits)
        self.use_float64 = bool(use_float64)

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2:
            raise ValueError(
                f"Expected latent shape [batch, latent_dim, ...], got {tuple(latent.shape)}."
            )

        original_dtype = latent.dtype
        work_dtype = self._work_dtype(latent)

        latent = torch.flatten(latent, start_dim=1).to(dtype=work_dtype)
        sensitive = self._prepare_sensitive(
            sensitive=sensitive,
            batch_size=latent.shape[0],
            device=latent.device,
        )

        h_marginal = self._h_bernoulli(latent)
        h_conditional = latent.new_zeros(())

        batch_size = latent.shape[0]
        for value in torch.unique(sensitive, sorted=True):
            mask = sensitive == value
            latent_value = latent[mask]

            h_value = self._h_bernoulli(latent_value)
            weight = latent.new_tensor(latent_value.shape[0] / batch_size)

            h_conditional = h_conditional + weight * h_value

        mi = h_marginal - h_conditional

        # hepinfo masks NaNs; do not clamp MI positive.
        mi = torch.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)

        return mi.to(dtype=original_dtype)

    def _work_dtype(self, latent: torch.Tensor) -> torch.dtype:
        # TensorFlow path casts y_pred to float64.
        # MPS does not support float64, so use float32 there.
        if self.use_float64 and latent.device.type != "mps":
            return torch.float64

        return torch.float32

    def _prepare_sensitive(
        self,
        sensitive: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if sensitive.ndim == 0:
            raise ValueError("Sensitive variable must have a batch dimension.")

        if sensitive.ndim == 1:
            sensitive_flat = sensitive
        elif sensitive.ndim == 2 and sensitive.shape[1] == 1:
            sensitive_flat = sensitive[:, 0]
        else:
            raise ValueError(
                "Sensitive must have shape [batch] or [batch, 1]. "
                f"Got {tuple(sensitive.shape)}."
            )

        sensitive_flat = sensitive_flat.detach().reshape(-1).to(
            device=device,
            dtype=torch.long,
        )

        if sensitive_flat.numel() != batch_size:
            raise ValueError(
                f"Sensitive variable length ({sensitive_flat.numel()}) must match "
                f"batch size ({batch_size})."
            )

        return sensitive_flat

    def _bernoulli_probs(self, latent: torch.Tensor) -> torch.Tensor:
        if self.input_is_logits:
            return torch.sigmoid(self.temperature * latent)

        return latent

    def _log2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.eps) / torch.log(x.new_tensor(2.0))

    def _h_bernoulli(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.numel() == 0:
            return latent.new_zeros(())

        theta = self._bernoulli_probs(latent).mean(dim=0)

        entropy_per_unit = -(
            (1.0 - theta) * self._log2(1.0 - theta)
            + theta * self._log2(theta)
        )

        return entropy_per_unit.sum()