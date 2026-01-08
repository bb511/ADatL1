# Encoder models.
from typing import Optional, Callable

import torch
import torch.nn as nn

import keras
from keras import layers
from keras import ops
from hgq.layers import QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope

from src.algorithms.components.mlp import MLP
from src.algorithms.components.mlp import hgq_mlp


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
        in_dim: int,
        nodes: list[int],
        out_dim: int,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()

        # The encoder will be a MLP up to the last layer
        self.net = MLP(
            in_dim,
            nodes[:-1],
            nodes[-1],
            batchnorm=False,
            init_weight=init_weight,
            init_bias=init_bias
        )

        # Mean and log variance layers
        self.z_mean = nn.Linear(nodes[-1], out_dim)
        self.z_log_var = nn.Linear(nodes[-1], out_dim)

        if init_weight:
            init_weight(self.z_mean.weight)
            init_weight(self.z_log_var.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self.net(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z_log_var = z_log_var.clamp(-20, 10)
        z = self.sample(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Performs reparametrization trick."""
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon


def hgq_variational_encoder(
    in_dim: int,
    nodes: list[int],
    out_dim: int,
    kernel_initializer=None,
    bias_initializer=None,
    name: str = 'hgq_var_encoder'
):
    """Variational Encoder in HGQv2

    :param in_dim: Int specifying input dimension.
    :param nodes: List of ints, each int specifying the width of a layer.
    :param out_dim: Int specifying output dimension.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """

    x_in = keras.Input(shape=(in_dim,), name="x")
    with (QuantizerConfigScope(place="all"), LayerConfigScope(enable_ebops=False)):
        mlp_model = hgq_mlp(
            in_dim=in_dim,
            nodes=nodes[:-1],
            out_dim=nodes[-1],
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="enc_mlp",
        )
        h = mlp_model(x_in)

        # Mean and log-variance heads (quantized)
        z_mean = QDense(
            out_dim,
            name="z_mean",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(h)

        z_log_var = QDense(
            out_dim,
            name="z_log_var",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(h)

    z_mean, z_log_var, z = Sampling(name="sampling")([z_mean, z_log_var])

    return keras.Model(inputs=x_in, outputs=[z_mean, z_log_var, z], name=name)


@keras.saving.register_keras_serializable(package="adl1t")
class Sampling(keras.layers.Layer):
    """Reparameterization trick layer."""

    def __init__(self, clip_min: float = -20.0, clip_max: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs

        # clamp / clip log-variance
        z_log_var = ops.clip(z_log_var, self.clip_min, self.clip_max)

        std = ops.exp(0.5 * z_log_var)

        # backend-agnostic random normal
        eps = keras.random.normal(shape=ops.shape(std), dtype=std.dtype)

        z = z_mean + std * eps
        return z_mean, z_log_var, z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"clip_min": self.clip_min, "clip_max": self.clip_max})
        return cfg
