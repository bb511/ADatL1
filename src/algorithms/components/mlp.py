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
