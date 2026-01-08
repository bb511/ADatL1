# Decoder models.
from typing import Optional, List, Callable

import torch
import keras
from keras import layers

from hgq.layers import QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope

from src.algorithms.components.mlp import MLP


class Decoder(MLP):
    """Simple decoder model.

    :param nodes: List of ints, each int specifying the width of a layer.
    :param init_weight: Callable method to initialize the weights of the decoder nodes.
    :param init_bias: Callable method to initialize the biases of the decoder nodes.
    :param init_last_weight: Callable method to initialize the weights of the last layer.
    :param init_last_bias: Callable method to initialize the biases of the last layer.
    :param batchnorm: Whether to use batch normalization or not.
    """

    def __init__(
        self,
        in_dim: int,
        nodes: List[int],
        out_dim: int,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
        batchnorm: bool = False,
        init_last_weight: Optional[Callable] = None,
        init_last_bias: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            in_dim,
            nodes,
            out_dim,
            batchnorm=batchnorm,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        # Apply initialization to the weight of the last Linear() layer
        last_linear_layer = self.net[-1]
        if init_last_weight != None:
            init_last_weight(last_linear_layer.weight)
            if last_linear_layer.bias != None and init_last_bias != None:
                init_last_bias(last_linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def hgq_decoder(
    in_dim: int,
    nodes: list[int],
    out_dim: int,
    batchnorm: bool = False,
    affine: bool = True,
    name: str = "hgq_decoder",
):
    """Simple dncoder in HGQv2.

    :param in_dim: Int specifying input dimension.
    :param nodes: List of ints, each int specifying the width of a layer.
    :param out_dim: Int specifying output dimension.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """
    z_in = keras.Input(shape=(in_dim,), name="z")

    with (QuantizerConfigScope(place="all"), LayerConfigScope(enable_ebops=False)):
        x = z_in
        for i, hidden_dim in enumerate(nodes):
            x = QDense(hidden_dim, name=f"dec_qdense_{i}")(x)

            if batchnorm:
                x = layers.BatchNormalization(
                    scale=affine,
                    center=affine,
                    name=f"dec_bn_{i}",
                )(x)

            x = layers.ReLU(name=f"dec_relu_{i}")(x)

        x = QDense(
            out_dim,
            name="dec_qdense_out",
            dtype="float32",
        )(x)

    return keras.Model(inputs=z_in, outputs=x, name=name)
