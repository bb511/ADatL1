# MLP model class to build algorithms out of.
from typing import Optional, Callable
import functools

import torch
from torch import nn

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron.

    :param in_dim: Int of the input dimension.
    :param nodes: List of ints number of nodes composing each of the layers.
    :param out_dim: Int of the output dimension.
    :param batchnorm: Whether to use batch normalization after each layer or not.
    :param affine: Bool whether the batchnorm is affine if used.
    :param activation: Pytorch module that defines the activation function.
    :param final_activation: Bool whether to attach an activation after the final layer.
    :param init_weight: Callable method to initialize the weights of the decoder nodes.
    :param init_bias: Callable method to initialize the biases of the decoder nodes.
    """

    def __init__(
        self,
        in_dim: int,
        nodes: list[int],
        out_dim: int,
        batchnorm: Optional[bool] = False,
        affine: bool = True,
        activation: str = "relu",
        final_activation: bool = False,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = nodes
        self.out_dim = out_dim
        self.activation = self._get_activation(activation)
        self.final_activation = final_activation

        self.batchnorm = batchnorm
        self.affine = affine
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.net = self._construct_net()
        self._apply_weight_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _get_activation(self, activation: str) -> nn.Module:
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "silu":
            return nn.SiLU()

        raise ValueError(f"Unsupported activation: {activation}")

    def _construct_net(self):
        """Build the neural network."""
        layers: list[nn.Module] = []

        current_dim = self.in_dim

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(
                nn.BatchNorm1d(hidden_dim, affine=self.affine)
                if self.batchnorm
                else nn.Identity()
            )
            layers.append(self.activation)
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, self.out_dim))
        if self.final_activation:
            layers.append(self.activation)

        return nn.Sequential(*layers)

    def _init_weight_wrapper(self, layer: nn.Module):
        if isinstance(layer, nn.Linear):
            return self.init_weight(layer.weight)
        return None

    def _apply_weight_init(self):
        """Initialise the weights with the given weight initialisation method."""
        if self.init_weight is not None:
            self.net.apply(self._init_weight_wrapper)
        if self.init_bias is not None:
            self.net.apply(self._init_bias_wrapper)

    def _init_bias_wrapper(self, layer: nn.Module):
        """Initialise the biases according to the given bias initialisation method."""
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            return self.init_bias(layer.bias)
        return None


class ImageMLP(nn.Module):
    """Convolutional analogue of an MLP.

    :param in_channels: Number of input channels.
    :param nodes: List of channel widths for the hidden/output conv layers.
    :param kernel_size: Kernel size of each convolution.
    :param strides: List of strides for each layer.
    :param transpose: If True, use ConvTranspose2d instead of Conv2d.
    :param batchnorm: Whether to use BatchNorm2d after each layer.
    :param affine: Whether BatchNorm2d is affine if used.
    :param activation: Activation function name.
    :param final_activation: Whether to apply activation after the last layer.
    :param init_weight: Callable method to initialize the weights.
    :param init_bias: Callable method to initialize the biases.
    """

    def __init__(
        self,
        in_channels: int,
        nodes: list[int],
        kernel_size: int = 3,
        strides: Optional[list[int]] = None,
        transpose: bool = False,
        batchnorm: bool = False,
        affine: bool = True,
        activation: str = "relu",
        final_activation: bool = False,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()

        if strides is None:
            strides = [1] * len(nodes)
        if len(strides) != len(nodes):
            raise ValueError("strides must have the same length as nodes.")

        self.in_channels = in_channels
        self.hidden_dims = nodes
        self.kernel_size = kernel_size
        self.strides = strides
        self.transpose = transpose
        self.batchnorm = batchnorm
        self.affine = affine
        self.activation = self._get_activation(activation)
        self.final_activation = final_activation
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.net = self._construct_net()
        self._apply_weight_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _get_activation(self, activation: str) -> nn.Module:
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {activation}")

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> nn.Module:
        if not self.transpose:
            return nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.kernel_size // 2,
            )

        return nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.kernel_size // 2,
            output_padding=stride - 1 if stride > 1 else 0,
        )

    def _construct_net(self):
        layers: list[nn.Module] = []
        current_channels = self.in_channels

        for i, (hidden_dim, stride) in enumerate(zip(self.hidden_dims, self.strides)):
            layers.append(self._make_layer(current_channels, hidden_dim, stride))
            layers.append(
                nn.BatchNorm2d(hidden_dim, affine=self.affine)
                if self.batchnorm
                else nn.Identity()
            )

            is_last = i == len(self.hidden_dims) - 1
            if not is_last or self.final_activation:
                layers.append(self.activation)

            current_channels = hidden_dim

        return nn.Sequential(*layers)

    def _apply_weight_init(self):
        if self.init_weight is not None:
            self.net.apply(self._init_weight_wrapper)
        if self.init_bias is not None:
            self.net.apply(self._init_bias_wrapper)

    def _init_weight_wrapper(self, layer: nn.Module):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            self.init_weight(layer.weight)
        return None

    def _init_bias_wrapper(self, layer: nn.Module):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)) and layer.bias is not None:
            self.init_bias(layer.bias)
        return None
