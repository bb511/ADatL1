from typing import Optional, Callable
import functools

import torch
from torch import nn

import keras
from keras import layers

from hgq.layers import QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope


class MLP(nn.Module):
    """
    Multi-layer perceptron.

    :param nodes: List of number of nodes composing each of the layers.
    :param batchnorm: Whether to use batch normalization after each layer or not.
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
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = nodes
        self.out_dim = out_dim

        self.batchnorm = batchnorm
        self.affine = affine
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.net = self._construct_net()
        self._apply_weight_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _construct_net(self):
        """Build the neural network."""
        layers: list[nn.Module] = []

        current_dim = self.in_dim

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(
                nn.BatchNorm1d(hidden_dim, affine=self.affine)
                if self.batchnorm else nn.Identity()
            )
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, self.out_dim))

        return nn.Sequential(*layers)

    def _apply_weight_init(self):
        if self.init_weight is not None:
            self.net.apply(self._init_weight_wrapper)
        if self.init_bias is not None:
            self.net.apply(self._init_bias_wrapper)

    def _init_weight_wrapper(self, layer: nn.Module):
        """Initialize the weights of each layer in a way dictated by a method."""
        if isinstance(layer, nn.Linear):
            return self.init_weight(layer.weight)
        return None

    def _init_bias_wrapper(self, layer: nn.Module):
        """Initialize the bias of each layer in a way dictated by a method."""
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            return self.init_bias(layer.bias)
        return None


def hgq_mlp(
    in_dim: int,
    nodes: list[int],
    out_dim: int,
    batchnorm: bool = False,
    affine: bool = True,
    kernel_initializer=None,
    bias_initializer=None,
    ebops: bool = False,
    name: str = 'hgq_mlp'
):
    """Multi-layer perceptron in HGQv2.

    :param in_dim: Int for initial dimension.
    :param nodes: List of number of nodes composing each of the layers.
    :param out_dim: Int for output dimension.
    :param batchnorm: Whether to use batch normalization after each layer or not.
    :param init_weight: Callable method to initialize the weights of the decoder nodes.
    :param init_bias: Callable method to initialize the biases of the decoder nodes.
    """
    inputs = keras.Input(shape=(in_dim,), name="x")
    x = inputs

    with (QuantizerConfigScope(place="all"), LayerConfigScope(enable_ebops=ebops)):
        for i, hidden_dim in enumerate(nodes):
            x = QDense(
                hidden_dim,
                name=f"qdense_{i}",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer
            )(x)

            if batchnorm:
                x = layers.BatchNormalization(
                        scale=affine, center=affine, name=f"bn_{i}"
                    )(x)

            x = layers.ReLU(name=f"relu_{i}")(x)

        outputs = QDense(out_dim, name="qdense_out")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)
