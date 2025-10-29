from typing import Optional, Callable, List
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
        nodes: List[int],
        batchnorm: Optional[bool] = False,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()
        layers = [
            nn.Sequential(
                nn.Linear(nodes[ilayer - 1], nodes[ilayer]),
                nn.BatchNorm1d(nodes[ilayer]) if batchnorm else nn.Identity(),
                nn.ReLU(),
            )
            for ilayer in range(1, len(nodes) - 1)
        ]
        self.net = nn.Sequential(*layers, nn.Linear(nodes[-2], nodes[-1]))

        # Initialize weights and biases
        if init_weight is not None and init_bias is not None:
            self.init_weights_and_biases(init_weight, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def init_weights_and_biases(self, init_weight: Callable, init_bias: Callable):
        self.net.apply(lambda m: init_weight(m.weight) if isinstance(m, nn.Linear) else None)
        self.net.apply(lambda m: init_bias(m.bias) if isinstance(m, nn.Linear) and m.bias is not None else None)
