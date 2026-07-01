from __future__ import annotations

import torch
from torch import nn


class BernoulliSampling(nn.Module):
    """Straight-through Bernoulli sampling layer from hepinfo/qkerasV3.py.

    Forward pass:
        p = sigmoid(temperature * inputs / std)
        train: average num_samples Bernoulli(p) draws
        eval:  hard threshold p >= threshold

    Backward pass:
        identity straight-through estimator, matching
        inputs + stop_gradient(-inputs + out) in TensorFlow/Keras.
    """

    def __init__(
        self,
        num_samples: int = 10,
        std: float = 1.0,
        threshold: float = 0.5,
        temperature: float = 6.0,
        use_quantized: bool = False,
        bits_bernoulli_sigmoid: int = 8,
    ) -> None:
        super().__init__()

        if int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
        if float(std) <= 0.0:
            raise ValueError(f"std must be > 0, got {std}.")
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
        if int(bits_bernoulli_sigmoid) < 2:
            raise ValueError(
                f"bits_bernoulli_sigmoid must be >= 2, got {bits_bernoulli_sigmoid}."
            )

        self.num_samples = int(num_samples)
        self.std = float(std)
        self.temperature = float(temperature)
        self.use_quantized = bool(use_quantized)
        self.bits_bernoulli_sigmoid = int(bits_bernoulli_sigmoid)

        self.register_buffer(
            "threshold",
            torch.tensor(float(threshold), dtype=torch.float32),
            persistent=False,
        )

    def probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return Bernoulli probabilities."""

        logits = (self.temperature / self.std) * inputs

        if self.use_quantized:
            return self._quantized_hard_sigmoid(logits).to(dtype=inputs.dtype)

        return torch.sigmoid(logits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        p = self.probabilities(inputs)

        if self.training:
            out = torch.zeros_like(inputs)

            for _ in range(self.num_samples):
                r = torch.rand_like(inputs)

                # Exact hepinfo/qkerasV3.py Bernoulli draw logic:
                # q = sign(p - r)
                # q += 1.0 - abs(q)
                # q = (q + 1.0) / 2.0
                q = torch.sign(p - r)
                q = q + 1.0 - torch.abs(q)
                q = (q + 1.0) / 2.0

                out = out + q

            out = out / float(self.num_samples)
        else:
            threshold = self.threshold.to(device=inputs.device, dtype=inputs.dtype)
            out = torch.where(p >= threshold, torch.ones_like(p), torch.zeros_like(p))

        # TensorFlow equivalent:
        #   out = inputs + tf.stop_gradient(-inputs + out)
        return inputs + (out - inputs).detach()

    def _quantized_hard_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Closest PyTorch equivalent of hepinfo/qkerasV3.py quantized_sigmoid."""

        x32 = x.to(dtype=torch.float32)
        p = torch.clamp(0.5 * x32 + 0.5, min=0.0, max=1.0)

        levels = float(2**self.bits_bernoulli_sigmoid)
        rounded = self._round_through(p * levels) / levels

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