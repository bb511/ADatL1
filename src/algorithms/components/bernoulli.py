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
    ) -> None:
        super().__init__()

        if int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
        if float(std) <= 0.0:
            raise ValueError(f"std must be > 0, got {std}.")
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}.")

        self.num_samples = int(num_samples)
        self.std = float(std)
        self.temperature = float(temperature)

        self.register_buffer(
            "threshold",
            torch.tensor(float(threshold), dtype=torch.float32),
            persistent=False,
        )

    def probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return Bernoulli probabilities."""

        return torch.sigmoid((self.temperature / self.std) * inputs)

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
            # TensorFlow equivalent:
            #   out = inputs + tf.stop_gradient(-inputs + out)
            return inputs + (out - inputs).detach()

        threshold = self.threshold.to(device=inputs.device, dtype=inputs.dtype)
        out = torch.where(p >= threshold, torch.ones_like(p), torch.zeros_like(p))

        # Return the hard code itself rather than routing it through the
        # straight-through identity. At evaluation there is no gradient to pass
        # back to ``inputs``, so the identity buys nothing, while
        # ``inputs + (out - inputs)`` equals ``out`` only in real arithmetic.
        # In float32 it holds while |inputs| < 2**24 and silently stops holding
        # above that. Downstream consumers (latent-collapse diagnostics,
        # leakage-probe extraction) assert hard zero/one codes and abort the run
        # after training has already completed, so this removes an entire class
        # of late failure at no cost.
        return out
