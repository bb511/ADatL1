from __future__ import annotations

import torch


class BernoulliMILoss(torch.nn.Module):
    """HepInfo-compatible Bernoulli mutual-information estimator.

    This is a line-by-line PyTorch translation of HepInfo's
    ``mutual_information_bernoulli_loss`` for the non-quantized path:

        I(L; S) = H(L) - sum_s p(S=s) H(L | S=s)

    where every latent activation is interpreted as a Bernoulli logit and mapped
    to a Bernoulli probability with ``sigmoid(temperature * activation)`` before
    estimating the Bernoulli entropy from the batch mean probability.
    """

    def __init__(self, temperature: float = 6.0, eps: float = 1e-20, input_is_logits: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.input_is_logits = bool(input_is_logits)

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2:
            raise ValueError(
                f"Expected latent shape [batch, latent_dim, ...], got {tuple(latent.shape)}."
            )

        # HepInfo casts y_pred to float64 before applying the sigmoid/entropy path.
        original_dtype = latent.dtype
        latent = torch.flatten(latent, start_dim=1).to(dtype=torch.float64)
        sensitive = self._prepare_sensitive(sensitive=sensitive, batch_size=latent.shape[0], device=latent.device)

        h_marginal = self._h_bernoulli(latent)
        h_conditional = latent.new_zeros(())

        batch_size = latent.shape[0]
        for value in torch.unique(sensitive):
            mask = (sensitive == value)
            latent_value = latent[mask]
            h_value = self._h_bernoulli(latent_value)
            weight = latent.new_tensor(latent_value.shape[0] / batch_size)
            h_conditional = h_conditional + weight * h_value

        mi = h_marginal - h_conditional

        # HepInfo only replaces NaN by zero; it does not clamp MI to be positive.
        mi = torch.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        return mi.to(dtype=original_dtype)

    def _prepare_sensitive(self, sensitive: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Match HepInfo's handling of y_true.

        HepInfo uses the full 1-D tensor when y_true is rank 1.  When y_true is
        rank 2, it uses the first column as the discrete sensitive variable.
        """
        if sensitive.ndim == 0:
            raise ValueError("Sensitive variable must have a batch dimension.")

        if sensitive.ndim == 1:
            sensitive_flat = sensitive
        elif sensitive.ndim == 2 and sensitive.shape[1] == 1:
            sensitive_flat = sensitive[:, 0]
        else:
            raise ValueError(
                "For MS1, sensitive must have shapt [batch] or [batch, 1]."
                f"Got {tuple(sensitive.shape)}."
            )

        sensitive_flat = sensitive_flat.detach().reshape(-1).to(device=device, dtype=torch.long)
        
        if sensitive_flat.numel() != batch_size:
            raise ValueError(
                f"Sensitive variable length ({sensitive_flat.numel()}) must match "
                f"batch size ({batch_size})."
            )
        return sensitive_flat

    def _bernoulli_probs(self, latent: torch.Tensor) -> torch.Tensor:
        if self.input_is_logits:
            # HepInfo: theta = sigmoid(temperature * x / std), with std = 1.0.
            return torch.sigmoid(self.temperature * latent)
        return latent

    def _log2(self, x: torch.Tensor) -> torch.Tensor:
        # HepInfo computes log2(x + 1e-20), not log2(clamp(x)).
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
