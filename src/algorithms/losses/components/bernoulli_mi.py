from __future__ import annotations

import torch
from torch import nn


class BernoulliMILoss(nn.Module):
    """HepInfo-compatible Bernoulli mutual-information estimator.

    Computes:

        I(L; S) = H(L) - sum_s p(S=s) H(L | S=s)

    Each latent activation is interpreted as a Bernoulli logit and mapped to

        p = sigmoid(temperature * latent)

    unless input_is_logits=False.

    The returned dtype intentionally follows hepinfo and is float32.
    """

    def __init__(
        self,
        temperature: float = 6.0,
        eps: float = 1e-20,
        input_is_logits: bool = True,
        use_float64: bool = True,
        use_quantized_sigmoid: bool = False,
        bits_bernoulli_sigmoid: int = 8,
    ) -> None:
        super().__init__()

        if bits_bernoulli_sigmoid < 2:
            raise ValueError(
                f"bits_bernoulli_sigmoid must be >= 2, got {bits_bernoulli_sigmoid}."
            )

        self.temperature = float(temperature)
        self.eps = float(eps)
        self.input_is_logits = bool(input_is_logits)
        self.use_float64 = bool(use_float64)
        self.use_quantized_sigmoid = bool(use_quantized_sigmoid)
        self.bits_bernoulli_sigmoid = int(bits_bernoulli_sigmoid)

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

    def _bernoulli_probs(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.input_is_logits:
            return latent

        logits = self.temperature * latent

        if self.use_quantized_sigmoid:
            return self._quantized_hard_sigmoid(logits).to(dtype=latent.dtype)

        return torch.sigmoid(logits)

    def _log2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + x.new_tensor(self.eps)) / torch.log(x.new_tensor(2.0))

    def _h_bernoulli(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.numel() == 0:
            return latent.new_zeros(())

        theta = self._bernoulli_probs(latent).mean(dim=0)

        entropy_per_unit = -(
            (1.0 - theta) * self._log2(1.0 - theta)
            + theta * self._log2(theta)
        )

        return entropy_per_unit.sum()

    def _quantized_hard_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Closest PyTorch equivalent of hepinfo/qkerasV3.py quantized_sigmoid.

        hepinfo's quantized_sigmoid uses the qkeras internal hard sigmoid by default:

            hard_sigmoid(x) = clip(0.5 * x + 0.5, 0, 1)

        then quantizes to 2**bits levels with straight-through rounding.
        """

        x32 = x.to(dtype=torch.float32)
        p = torch.clamp(0.5 * x32 + 0.5, min=0.0, max=1.0)

        levels = float(2**self.bits_bernoulli_sigmoid)
        rounded = self._round_through(p * levels) / levels

        # hepinfo uses symmetric=True in the MI call:
        # min = 1 / levels, max = 1 - 1 / levels
        return torch.clamp(
            rounded,
            min=1.0 / levels,
            max=1.0 - 1.0 / levels,
        )

    def _round_through(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            rounded = self._stochastic_round(x, precision=0.5)
        else:
            rounded = torch.round(x)

        return x + (rounded - x).detach()

    @staticmethod
    def _stochastic_round(x: torch.Tensor, precision: float = 0.5) -> torch.Tensor:
        scale = 1.0 / precision
        scaled = x * scale
        floor = torch.floor(scaled)
        fraction = scaled - floor

        rnd = torch.rand_like(x)
        rounded_scaled = torch.where(fraction < rnd, floor, floor + 1.0)

        return rounded_scaled / scale