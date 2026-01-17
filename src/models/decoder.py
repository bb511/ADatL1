from typing import Optional, Callable

import torch

import keras
from hgq.layers import QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope

from src.models.components.mlp import MLP, HGQMLP


class Decoder(MLP):
    """Simple decoder model.

    :param nodes: List of layer dimensions. nodes[0] is latent dim, nodes[-1] is output.
    :param batchnorm: Whether to use batch normalization.
    :param init_weight: Callable to initialize layer weights.
    :param init_bias: Callable to initialize layer biases.
    :param init_last_weight: Callable to initialize the last layer weights.
    :param init_last_bias: Callable to initialize the last layer biases.
    """

    def __init__(
        self,
        nodes: list[int],
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
        batchnorm: bool = False,
        init_last_weight: Optional[Callable] = None,
        init_last_bias: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            nodes=nodes,
            batchnorm=batchnorm,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        last_linear_layer = self.net[-1]
        if init_last_weight is not None:
            init_last_weight(last_linear_layer.weight)
            if last_linear_layer.bias is not None and init_last_bias is not None:
                init_last_bias(last_linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HGQDecoder(keras.Model):
    """Decoder in HGQv2.

    :param nodes: List of layer dimensions. nodes[0] is latent dim, nodes[-1] is output.
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
        self.input_layer_config = input_layer_config
        self.output_layer_config = output_layer_config or {}
        self.ebops = ebops

        self.net = HGQMLP(
            nodes=nodes[:-1],
            input_layer_config=self.input_layer_config,
            final_activation=True,
            name="net"
        )

        with LayerConfigScope(enable_ebops=False):
            with QuantizerConfigScope(**self.output_layer_config):
                self.output_layer = QDense(nodes[-1], name="qdense_out")

    def call(self, z):
        h = self.net(z)
        return self.output_layer(h)
