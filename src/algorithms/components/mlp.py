from typing import Optional, Callable
import functools

import torch
from torch import nn


class MLP(nn.Module):
    """
    Multi-layer perceptron.

    :param nodes: Number of nodes composing each of the layers.
    """

    def __init__(
        self,
        nodes: list[int],
        batchnorm: Optional[bool] = False,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()
        self.nodes = nodes
        self.batchnorm = batchnorm
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.net = self._construct_net()

        # Initialize weights and biases
        if self.init_weight is not None:
            self.net.apply(self._init_weight_wrapper)
        if self.init_bias is not None:
            self.net.apply(self._init_bias_wrapper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def _construct_net(self):
        """Build the neural network."""
        layers = []
        for ilayer in range(1, len(self.nodes) - 1):
            layer = nn.Sequential(
                nn.Linear(self.nodes[ilayer - 1], self.nodes[ilayer]),
                nn.BatchNorm1d(self.nodes[ilayer]) if self.batchnorm else nn.Identity(),
                nn.ReLU(),
            )
            layers.append(layer)

        return nn.Sequential(*layers, nn.Linear(self.nodes[-2], self.nodes[-1]))

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
