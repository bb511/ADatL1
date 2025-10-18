# Simple auto-encoder model.

# Variational autoencoder model.
from typing import Optional, Tuple, List, Callable

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Simple encoder model.

    :param nodes: List of ints, each int specifying the width of a layer.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """

    def __init__(
        self,
        nodes: List[int],
        init_weight: Optional[Callable] = lambda _: None,
        init_bias: Optional[Callable] = lambda _: None,
    ):
        super().__init__()

        self.init_weight = init_weight
        self.init_bias = init_bias

        # The encoder will be a QMLP up to the last layer
        net_list = []
        for idx in range(len(nodes) - 2):
            net_list.append(nn.Linear(nodes[idx], nodes[idx + 1]))
            net_list.append(nn.ReLU())

        self.net = nn.Sequential(*net_list)
        self.net.apply(self._init_weights_and_biases)

        # Mean and log variance layers
        self.output_layer = nn.Linear(nodes[-2], nodes[-1])
        init_weight(self.z_mean.weight)

    def _init_weights_and_biases(self, m):
        if isinstance(m, nn.Linear):
            self.init_weight(m.weight)
            self.init_bias(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # Forward pass through the network
        x = self.net(x)

        # Mean and log variance for reparemeterization
        z = self.output_layer(x)

        return z


class Decoder(nn.Module):
    """Decoder for AD@L1."""

    def __init__(
        self,
        nodes: List[int],
        init_weight: Optional[Callable] = lambda _: None,
        init_bias: Optional[Callable] = lambda _: None,
        init_last_weight: Optional[Callable] = None,
        init_last_bias: Optional[Callable] = None,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.init_weight = init_weight
        self.init_bias = init_bias

        init_last_weight = init_last_weight if init_last_weight else lambda _: None
        init_last_bias = init_last_bias if init_last_bias else lambda _: None

        # Build the decoder as a MLP (no quantization) with batch normalization
        net_list = []
        for idx in range(len(nodes) - 2):
            net_list.append(nn.Linear(nodes[idx], nodes[idx + 1]))
            net_list.append(nn.ReLU())

        # Remove last activation.
        net_list.pop()

        self.net = nn.Sequential(*net_list)
        self.net.apply(self._init_weights_and_biases)

        # Apply initialization to the weight of the last Linear() layer
        last_linear_layer = self.net[-1]
        if init_last_weight != None:
            init_last_weight(last_linear_layer.weight)
        if init_last_weight != None and last_linear_layer.bias != None:
            init_last_bias(last_linear_layer.bias)

    def _init_weights_and_biases(self, m):
        if isinstance(m, nn.Linear):
            self.init_weight(m.weight)
            self.init_bias(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
