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
        ``inputs + stop_gradient(-inputs + out)`` in TensorFlow/Keras.
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
        if use_quantized:
            raise NotImplementedError(
                "hepinfo's quantized_sigmoid path is not implemented in this PyTorch port. "
                "Set mi_use_quantized_sigmoid=false."
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
        """Return Bernoulli probabilities p = sigmoid(temperature * x / std)."""
        return torch.sigmoid((self.temperature / self.std) * inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        p = self.probabilities(inputs)

        if self.training:
            out = torch.zeros_like(inputs)
            for _ in range(self.num_samples):
                r = torch.rand_like(inputs)

                # Exact hepinfo/qkerasV3.py logic:
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