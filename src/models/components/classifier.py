# Variational autoencoder model.
from typing import Optional, Tuple, List, Callable

import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    """Simple classifier architecture.

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
        for idx in range(len(nodes) - 1):
            net_list.append(nn.Linear(nodes[idx], nodes[idx + 1]))
            net_list.append(nn.ReLU())

        self.net = nn.Sequential(*net_list)
        self.net.apply(self._init_weights_and_biases)

        # Mean and log variance layers
        self.output = nn.Linear(nodes[-1], 1)

    def _init_weights_and_biases(self, m):
        if isinstance(m, nn.Linear):
            self.init_weight(m.weight)
            self.init_bias(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through the network
        x = self.net(x)

        # Mean and log variance for reparemeterization
        output = self.output(x)

        return output
