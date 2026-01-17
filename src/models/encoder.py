from typing import Optional, Callable

import torch
import torch.nn as nn

import keras
from keras import ops
from hgq.layers import QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope

from src.models.components.mlp import MLP, HGQMLP


class Encoder(MLP):
    """Simple vanilla encoder, i.e., just an MLP."""
    pass


class VariationalEncoder(nn.Module):
    """Variational encoder model.

    :param nodes: List of layer dimensions. nodes[0] is input, nodes[-1] is latent dim.
    :param init_weight: Callable to initialize layer weights.
    :param init_bias: Callable to initialize layer biases.
    """

    def __init__(
        self,
        nodes: list[int],
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()
        self.latent_dim = nodes[-1]

        self.net = MLP(
            nodes=nodes[:-1],
            batchnorm=False,
            final_activation=True,
            init_weight=init_weight,
            init_bias=init_bias
        )

        self.z_mean = nn.Linear(nodes[-2], self.latent_dim)
        self.z_log_var = nn.Linear(nodes[-2], self.latent_dim)

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
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon


@keras.saving.register_keras_serializable(package="adl1t")
class Sampling(keras.layers.Layer):
    """Reparameterization trick layer."""

    def __init__(self, clip_min: float = -20.0, clip_max: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        z_log_var = ops.clip(z_log_var, self.clip_min, self.clip_max)
        std = ops.exp(0.5 * z_log_var)
        eps = keras.random.normal(shape=ops.shape(std), dtype=std.dtype)
        z = z_mean + std * eps
        return z_mean, z_log_var, z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"clip_min": self.clip_min, "clip_max": self.clip_max})
        return cfg


class HGQVariationalEncoder(keras.Model):
    """Variational Encoder in HGQv2.

    :param nodes: List of layer dimensions. nodes[0] is input, nodes[-1] is latent dim.
    :param input_layer_config: Quantization config for input layer.
    :param output_layer_config: Quantization config for output layer.
    :param ebops: Whether to enable energy-based operations.
    """

    def __init__(
        self,
        nodes: list[int],
        input_layer_config: dict = None,
        output_layer_config: dict = None,
        ebops: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nodes = nodes
        self.latent_dim = nodes[-1]
        self.input_layer_config = input_layer_config
        self.output_layer_config = output_layer_config or {}
        self.ebops = ebops

        self.net = HGQMLP(
            nodes=nodes[:-1],
            input_layer_config=self.input_layer_config,
            ebops=self.ebops,
            final_activation=True,
            name="net"
        )

        with LayerConfigScope(enable_ebops=self.ebops):
            with QuantizerConfigScope(**self.output_layer_config):
                self.z_mean = QDense(self.latent_dim, name="z_mean")
                self.z_log_var = QDense(self.latent_dim, name="z_log_var")

        self.sampling = Sampling(name="sampling")

    def call(self, x):
        h = self.net(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z_mean, z_log_var, z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z
