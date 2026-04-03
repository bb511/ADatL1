# Encoder models.
from typing import Optional, Callable

import torch
import torch.nn as nn

from src.algorithms.components.mlp import MLP


class Encoder(nn.Module):
    """Simple vanilla encoder, i.e., just an MLP.

    :param in_dim: Integer for the input dimension to the encoder.
    :param nodes: List of ints, each int specifying the width of a layer, includes the
        output layer, i.e., the latent dimension.
    :param activation: Pytorch module that defines the activation function.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """

    def __init__(
        self,
        in_dim: int,
        nodes: list[int],
        activation: str = "relu",
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()

        self.net = MLP(
            in_dim=in_dim,
            nodes=nodes[:-1],
            out_dim=nodes[-1],
            batchnorm=False,
            activation=activation,
            final_activation=False,
            init_weight=init_weight,
            init_bias=init_bias,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.net(x)


class VariationalEncoder(nn.Module):
    """Simple variational encoder model.

    :param in_dim: Integer for the input dimension to the encoder.
    :param nodes: List of ints, each int specifying the width of a layer.
    :param activation: Pytorch module that defines the activation function.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    :param clamp_zlogvar_range: Tuple of two floats defining the range of the z_log_var
        value produced by the variational encoder.
    """

    def __init__(
        self,
        in_dim: int,
        nodes: list[int],
        activation: str = "relu",
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
        clamp_zlogvar_range: tuple[float, float] = (-20.0, 10.0),
    ):
        super().__init__()

        # The encoder will be a MLP up to the last layer
        self.net = MLP(
            in_dim=in_dim,
            nodes=nodes[:-2],
            out_dim=nodes[-2],
            batchnorm=False,
            activation=activation,
            final_activation=True,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        # Mean and log variance layers
        self.z_mean = nn.Linear(nodes[-2], nodes[-1])
        self.z_log_var = nn.Linear(nodes[-2], nodes[-1])

        self.clamp_zlogvar_range = clamp_zlogvar_range
        if init_weight:
            init_weight(self.z_mean.weight)
            init_weight(self.z_log_var.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self.net(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z_log_var = z_log_var.clamp(*self.clamp_zlogvar_range)
        z = self.sample(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Performs reparametrization trick."""
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon
