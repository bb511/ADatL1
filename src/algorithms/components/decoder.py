# Decoder models.
from typing import Optional, Callable
import math

import torch
import torch.nn as nn

from src.algorithms.components.mlp import MLP
from src.algorithms.components.mlp import ImageMLP


class Decoder(MLP):
    """Simple decoder model.

    :param nodes: List of ints, each int specifying the width of a layer.
    :param out_dim: Int of dimensionality of the output, needs to be same as input.
    :param activation: Pytorch module that defines the activation function.
    :param init_weight: Callable method to initialize the weights of the decoder nodes.
    :param init_bias: Callable method to initialize the biases of the decoder nodes.
    :param init_last_weight: Callable method to initialize the weights of the last layer.
    :param init_last_bias: Callable method to initialize the biases of the last layer.
    :param batchnorm: Whether to use batch normalization or not.
    """

    def __init__(
        self,
        nodes: list[int],
        out_dim: int,
        activation: str = "relu",
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
        batchnorm: bool = False,
        init_last_weight: Optional[Callable] = None,
        init_last_bias: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            in_dim=nodes[0],
            nodes=nodes[1:],
            out_dim=out_dim,
            activation=activation,
            batchnorm=batchnorm,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        # Apply initialization to the weight of the last Linear() layer
        last_linear_layer = self.net[-1]
        if init_last_weight != None:
            init_last_weight(last_linear_layer.weight)
            if last_linear_layer.bias != None and init_last_bias != None:
                init_last_bias(last_linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageDecoder(nn.Module):
    """Simple convolutional decoder model.

    This decoder follows the same config contract as the dense Decoder:
      - `nodes[0]` is the input dimension of the decoder,
      - `nodes[1:]` define the hidden layers,
      - `out_channels` defines the final output channels.

    For the image case:
      - `nodes[0]` is interpreted as the latent dimension,
      - the latent vector is projected to a feature map of shape
        [nodes[0], H0, W0],
      - `nodes[1:]` are the output channels of the transpose-convolutional stack,
      - `input_size` and `strides` are used to infer the starting spatial size.

    :param nodes: List of ints specifying the decoder widths. The first entry is the
        latent dimension. The remaining entries define the transpose-convolutional
        stack.
    :param out_channels: Number of output image channels.
    :param input_size: Final spatial image size as (height, width).
    :param strides: List of strides for the transpose-convolutional body. Must have
        length len(nodes) - 1.
    :param activation: Activation function name.
    :param init_weight: Callable method to initialize decoder weights.
    :param init_bias: Callable method to initialize decoder biases.
    :param batchnorm: Whether to use batch normalization or not.
    :param init_last_weight: Callable method to initialize the last layer weights.
    :param init_last_bias: Callable method to initialize the last layer biases.
    :param final_activation: Optional final activation after the output layer.
    """

    def __init__(
        self,
        nodes: list[int],
        out_channels: int,
        input_size: tuple[int, int] | list[int] = (32, 32),
        strides: Optional[list[int]] = None,
        activation: str = "relu",
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
        batchnorm: bool = False,
        init_last_weight: Optional[Callable] = None,
        init_last_bias: Optional[Callable] = None,
        final_activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        if len(nodes) < 2:
            raise ValueError(
                "nodes must contain at least two entries: "
                "[latent_dim, first_hidden_channels, ...]."
            )

        self.latent_dim = int(nodes[0])
        hidden_nodes = list(nodes[1:])
        self.input_size = tuple(int(x) for x in input_size)

        if strides is None:
            strides = [1] * len(hidden_nodes)
        if len(strides) != len(hidden_nodes):
            raise ValueError("strides must have length len(nodes) - 1.")

        self.strides = list(strides)

        h0, w0 = self._infer_start_size(self.input_size, self.strides)
        c0 = self.latent_dim

        self.feature_shape = (c0, h0, w0)
        self.feature_dim = c0 * h0 * w0

        self.proj = nn.Linear(self.latent_dim, self.feature_dim)
        if init_weight is not None:
            init_weight(self.proj.weight)
        if init_bias is not None and self.proj.bias is not None:
            init_bias(self.proj.bias)

        self.net = ImageMLP(
            in_channels=c0,
            nodes=hidden_nodes,
            strides=self.strides,
            transpose=True,
            batchnorm=batchnorm,
            activation=activation,
            final_activation=False,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        last_in_channels = hidden_nodes[-1]
        self.final_layer = nn.Conv2d(
            last_in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if init_last_weight is not None:
            init_last_weight(self.final_layer.weight)
        elif init_weight is not None:
            init_weight(self.final_layer.weight)

        if self.final_layer.bias is not None:
            if init_last_bias is not None:
                init_last_bias(self.final_layer.bias)
            elif init_bias is not None:
                init_bias(self.final_layer.bias)

        self.final_activation = None
        if final_activation is not None:
            self.final_activation = self._get_activation(final_activation)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected latent tensor of shape [B, D], got {tuple(z.shape)}.")

        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, got {z.shape[1]}."
            )

        x = self.proj(z)
        c, h, w = self.feature_shape
        x = x.view(z.shape[0], c, h, w)
        x = self.net(x)
        x = self.final_layer(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def _infer_start_size(
        self,
        output_size: tuple[int, int],
        strides: list[int],
    ) -> tuple[int, int]:
        """Infer the initial feature-map size needed before transpose convolutions."""
        h, w = output_size
        for s in reversed(strides):
            h = math.ceil(h / s)
            w = math.ceil(w / s)
        return h, w

    def _get_activation(self, activation: str) -> nn.Module:
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "silu":
            return nn.SiLU()
        if activation == "sigmoid":
            return nn.Sigmoid()
        if activation == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {activation}")
