# Encoder models.
from typing import Optional, Callable

import torch
import torch.nn as nn

from src.algorithms.components.mlp import MLP


class Encoder(MLP):
    """Simple vanilla encoder, i.e., just an MLP."""

    pass


class VariationalEncoder(nn.Module):
    """Simple variational encoder model.

    :param nodes: List of ints, each int specifying the width of a layer.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """

    def __init__(
        self,
        nodes: list[int],
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()

        # The encoder will be a MLP up to the last layer
        self.net = MLP(
            nodes[:-1], batchnorm=False, init_weight=init_weight, init_bias=init_bias
        )

        # Mean and log variance layers
        self.z_mean = nn.Linear(nodes[-2], nodes[-1])
        self.z_log_var = nn.Linear(nodes[-2], nodes[-1])

        if init_weight:
            init_weight(self.z_mean.weight)
            init_weight(self.z_log_var.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self.net(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.sample(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Performs reparametrization trick."""
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon
