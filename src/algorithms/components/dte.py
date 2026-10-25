"""Predictor components for categorical Diffusion Time Estimation, and the corruption of
its eval-mode passes."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from src.algorithms.components.mlp import ImageMLP

# The seed of the eval-mode corruption, the same in every module and every run.
EVAL_CORRUPTION_SEED = 20261025


def eval_corruption(x: torch.Tensor, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the diffusion times, shape (batch,), and the noise, the shape of ``x``, of
    an eval-mode pass.

    The time of an event is a fixed pseudo-random function of its input, t = floor(u
    n_steps) with u = frac(<x_flat, w>) and w ~ U[0, 1)^features drawn from
    ``EVAL_CORRUPTION_SEED``, the dot product in float64 on CPU. The noise is the next
    draw of the same generator, so it is fixed per position in the batch, not per event.
    Every epoch and every replay then evaluate the same corruption (common random
    numbers): the differences the epoch rule reads carry no Monte Carlo noise and a
    rescoring reproduces, without a global seed or a buffer that old state dicts lack.
    """
    generator = torch.Generator().manual_seed(EVAL_CORRUPTION_SEED)
    flat = x.detach().flatten(start_dim=1).to("cpu", torch.float64)
    weights = torch.rand(flat.shape[1], generator=generator, dtype=torch.float64)
    # remainder rather than frac, which keeps the sign of a negative projection.
    u = torch.remainder((flat * weights).sum(dim=1), 1.0)
    timesteps = (u * n_steps).long().clamp(0, n_steps - 1)
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    return timesteps.to(x.device), noise.to(x.device)


class DTEPredictor(nn.Module):
    """Paper-style MLP used to predict an ordered diffusion-time bin.

    The official DTE implementation applies dropout after every hidden layer except the
    first one. Keeping this component separate makes the algorithm usable with a
    different predictor without changing the corruption or scoring contract.
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
            raise ValueError(
                f"DTEPredictor expects [batch, features], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"DTEPredictor expected {self.in_dim} features, got {x.shape[1]}."
            )
        return self.net(x)


class ImageDTEPredictor(nn.Module):
    """Convolutional predictor of the ordered diffusion-time bin.

    Image-domain analogue of :class:`DTEPredictor`. ``nodes`` and ``strides`` follow exactly
    the same convention as :class:`src.algorithms.components.encoder.ImageEncoder` -- the last
    entry of ``nodes`` is the latent dimension and ``strides`` applies to the convolutional
    layers before it -- so that DTE explores the same architecture family as the other image
    models. A final linear head maps the latent onto ``out_dim`` bin logits.

    :param in_channels: Number of image input channels.
    :param nodes: List of ints specifying the hidden channel widths and the output
        latent dimension. The last entry is the latent dimension.
    :param out_dim: Number of ordered diffusion-time bins.
    :param input_size: Spatial input size as (height, width).
    :param strides: List of strides for the convolutional hidden layers.
    :param activation: Activation function name.
    :param dropout: Dropout probability applied to the latent before the bin-logit head.
    :param batchnorm: Whether to use batch normalization or not.
    :param init_weight: Callable method to initialize the weights.
    :param init_bias: Callable method to initialize the biases.
    """

    def __init__(
        self,
        in_channels: int,
        nodes: list[int],
        out_dim: int,
        input_size: tuple[int, int] = (32, 32),
        strides: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        init_weight: Callable | None = None,
        init_bias: Callable | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if len(nodes) < 1:
            raise ValueError("nodes must contain at least one entry.")
        if any(width <= 0 for width in nodes):
            raise ValueError("nodes must contain positive widths.")
        if out_dim < 2:
            raise ValueError("out_dim must be at least 2 for categorical DTE.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        conv_nodes = nodes[:-1]
        latent_dim = nodes[-1]

        if len(conv_nodes) == 0:
            raise ValueError(
                "ImageDTEPredictor requires at least one convolutional hidden layer before "
                "the latent dimension."
            )

        if strides is None:
            strides = [1] * len(conv_nodes)
        if len(strides) != len(conv_nodes):
            raise ValueError("strides must have length len(nodes) - 1.")

        self.net = ImageMLP(
            in_channels=in_channels,
            nodes=conv_nodes,
            strides=strides,
            transpose=False,
            batchnorm=batchnorm,
            activation=activation,
            final_activation=True,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        h, w = input_size
        for s in strides:
            h = (h + s - 1) // s
            w = (w + s - 1) // s

        self.in_channels = int(in_channels)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.nodes = [int(width) for width in nodes]
        self.feature_shape = (conv_nodes[-1], h, w)
        self.feature_dim = conv_nodes[-1] * h * w
        self.latent_dim = int(latent_dim)
        self.out_dim = int(out_dim)

        self.proj = nn.Linear(self.feature_dim, self.latent_dim)
        self.head_activation = self.net.activation
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(self.latent_dim, self.out_dim)
        for layer in (self.proj, self.head):
            if init_weight is not None:
                init_weight(layer.weight)
            if init_bias is not None and layer.bias is not None:
                init_bias(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"ImageDTEPredictor expects [batch, channels, height, width], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"ImageDTEPredictor expected {self.in_channels} channels, got {x.shape[1]}."
            )
        x = self.net(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head_activation(self.proj(x))
        return self.head(self.dropout(x))
