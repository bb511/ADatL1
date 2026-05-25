from __future__ import annotations

import torch
from torch import nn


class BernoulliQuantizedBottleneck(nn.Module):
    """Bernoulli stochastic bottleneck with straight-through estimator.

    Forward pass:
        logits -> sigmoid -> Bernoulli sample -> hard latent code

    Backward pass:
        gradients flow through the probabilities.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        deterministic_eval: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = temperature
        self.deterministic_eval = deterministic_eval
        self.eps = eps

    def forward(self, z_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probs = torch.sigmoid(z_logits / self.temperature)
        probs = probs.clamp(self.eps, 1.0 - self.eps)

        if self.training or not self.deterministic_eval:
            hard = torch.bernoulli(probs)
        else:
            hard = (probs >= 0.5).to(dtype=probs.dtype)

        # Straight-through estimator:
        # forward value is hard, backward gradient follows probs.
        z_quantized = hard.detach() - probs.detach() + probs

        return z_quantized, {
            "logits": z_logits,
            "probs": probs,
            "hard": hard,
        }