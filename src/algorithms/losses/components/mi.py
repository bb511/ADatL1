from __future__ import annotations

import torch
from torch import nn

from src.algorithms.losses.components.bernoulli import BernoulliActivation


class BernoulliMILoss(nn.Module):
    """HepInfo-compatible Bernoulli mutual-information estimator.

    Computes:

        I(L; S) = H(L) - sum_s p(S=s) H(L | S=s)

    Each latent activation is interpreted as a Bernoulli logit and mapped to a
    probability by :class:`~src.algorithms.losses.components.bernoulli.BernoulliActivation`,

        p = sigmoid(temperature * latent)

    unless input_is_logits=False. Everything to do with that mapping lives in
    ``bernoulli.py``; this class only does entropy bookkeeping on top of the
    probabilities it gets back.

    The returned dtype intentionally follows hepinfo and is float32.
    """

    def __init__(
        self,
        temperature: float = 6.0,
        eps: float = 1e-20,
        input_is_logits: bool = True,
        use_float64: bool = True,
    ) -> None:
        super().__init__()

        self.eps = float(eps)
        self.use_float64 = bool(use_float64)

        # The Bernoulli sampling front end, held as a submodule so that it
        # travels with .to() and shows up in the module repr.
        self.bernoulli = BernoulliActivation(
            temperature=temperature,
            input_is_logits=input_is_logits,
        )

    @property
    def temperature(self) -> float:
        return self.bernoulli.temperature

    @property
    def input_is_logits(self) -> bool:
        return self.bernoulli.input_is_logits

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2:
            raise ValueError(
                f"Expected latent shape [batch, latent_dim, ...], got {tuple(latent.shape)}."
            )

        work_dtype = self._work_dtype(latent)

        latent = torch.flatten(latent, start_dim=1).to(dtype=work_dtype)
        sensitive = self._prepare_sensitive(
            sensitive=sensitive,
            batch_size=latent.shape[0],
            device=latent.device,
        )

        h_marginal = self._h_bernoulli(latent)

        batch_size = latent.shape[0]
        h_conditional = latent.new_zeros(())

        for value in torch.unique(sensitive, sorted=True):
            mask = sensitive == value
            latent_value = latent[mask]

            h_value = self._h_bernoulli(latent_value)
            weight = latent.new_tensor(latent_value.shape[0] / batch_size)

            h_conditional = h_conditional + weight * h_value

        mi = h_marginal - h_conditional

        # hepinfo masks NaNs. Do not silently squash +/-inf to zero.
        mi = torch.where(torch.isnan(mi), mi.new_zeros(()), mi)

        # hepinfo returns tf.float32.
        return mi.to(dtype=torch.float32)

    def _work_dtype(self, latent: torch.Tensor) -> torch.dtype:
        # hepinfo casts y_pred to float64 before the entropy computation.
        # MPS does not support float64 well, so keep the previous MPS fallback.
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

        if sensitive.shape[0] != batch_size:
            raise ValueError(
                f"Sensitive first dimension ({sensitive.shape[0]}) must match "
                f"batch size ({batch_size}). Got shape {tuple(sensitive.shape)}."
            )

        sensitive_flat = sensitive.detach().reshape(batch_size, -1)

        if sensitive_flat.shape[1] != 1:
            raise ValueError(
                "Sensitive must contain exactly one scalar label/bin per event. "
                f"Got shape {tuple(sensitive.shape)}, which flattens to "
                f"{tuple(sensitive_flat.shape)}."
            )

        return sensitive_flat[:, 0].to(device=device, dtype=torch.long)

    def _log2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + x.new_tensor(self.eps)) / torch.log(x.new_tensor(2.0))

    def _h_bernoulli(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.numel() == 0:
            return latent.new_zeros(())

        theta = self.bernoulli(latent).mean(dim=0)

        entropy_per_unit = -(
            (1.0 - theta) * self._log2(1.0 - theta)
            + theta * self._log2(theta)
        )

        return entropy_per_unit.sum()
