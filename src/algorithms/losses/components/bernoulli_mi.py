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

    def __init__(
        self,
        temperature: float = 6.0,
        use_quantized_sigmoid: bool = False,
        bits_bernoulli_sigmoid: int = 8,
        eps: float = 1e-20,
        input_is_logits: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.use_quantized_sigmoid = bool(use_quantized_sigmoid)
        self.bits_bernoulli_sigmoid = int(bits_bernoulli_sigmoid)
        self.eps = float(eps)
        self.input_is_logits = bool(input_is_logits)

        if self.use_quantized_sigmoid:
            raise NotImplementedError(
                "HepInfo's quantized_sigmoid path depends on qkeras stochastic "
                "rounding and is not implemented in this PyTorch translation."
            )

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2:
            raise ValueError(
                f"Expected latent shape [batch, latent_dim, ...], got {tuple(latent.shape)}."
            )

        # HepInfo casts y_pred to float64 before applying the sigmoid/entropy path.
        latent = torch.flatten(latent, start_dim=1).to(dtype=torch.float32)
        sensitive = self._prepare_sensitive(sensitive=sensitive, batch_size=latent.shape[0], device=latent.device)

        h_marginal = self._h_bernoulli(latent)
        h_conditional = latent.new_zeros(())

        batch_size = latent.shape[0]
        for value in torch.unique(sensitive):
            latent_value = latent[sensitive == value]
            h_value = self._h_bernoulli(latent_value)
            weight = latent.new_tensor(latent_value.shape[0] / batch_size)
            h_conditional = h_conditional + weight * h_value

        mi = h_marginal - h_conditional

        # HepInfo only replaces NaN by zero; it does not clamp MI to be positive.
        mi = torch.where(torch.isnan(mi), mi.new_tensor(0.0), mi)
        return mi.to(dtype=torch.float32)

    def _prepare_sensitive(
        self,
        sensitive: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Match HepInfo's handling of y_true.

        HepInfo uses the full 1-D tensor when y_true is rank 1.  When y_true is
        rank 2, it uses the first column as the discrete sensitive variable.
        """
        if sensitive.ndim == 0:
            raise ValueError("Sensitive variable must have a batch dimension.")

        if sensitive.ndim == 1:
            sensitive_flat = sensitive
        else:
            sensitive_flat = sensitive[:, 0]

        sensitive_flat = sensitive_flat.reshape(-1).to(device=device, dtype=torch.long)
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
