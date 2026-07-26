"""Predictor components for categorical Diffusion Time Estimation."""

from __future__ import annotations

import torch
from torch import nn


class DTEPredictor(nn.Module):
    """Paper-style MLP used to predict an ordered diffusion-time bin.

    The official DTE implementation applies dropout after every hidden layer except the first one.
    Keeping this component separate makes the algorithm usable with a different predictor without
    changing the corruption or scoring contract.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.5,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if out_dim < 2:
            raise ValueError("out_dim must be at least 2 for categorical DTE.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        activation_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }.get(activation.lower())
        if activation_cls is None:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        previous = int(in_dim)
        for index, width in enumerate(hidden_dims):
            layers.append(nn.Linear(previous, int(width)))
            layers.append(activation_cls())
            if index > 0 and dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(width)
        layers.append(nn.Linear(previous, int(out_dim)))

        self.in_dim = int(in_dim)
        self.hidden_dims = [int(width) for width in hidden_dims]
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"DTEPredictor expects [batch, features], got {tuple(x.shape)}.")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"DTEPredictor expected {self.in_dim} features, got {x.shape[1]}.")
        return self.net(x)
