from typing import Optional, Callable

import torch
from torch import nn

import keras
from keras import layers as klayers
from hgq.layers import QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope


class MLP(nn.Module):
    """Multi-layer perceptron.

    :param nodes: List of layer dimensions. nodes[0] is input, nodes[-1] is output.
    :param batchnorm: Whether to use batch normalization after each hidden layer.
    :param affine: Whether batchnorm has learnable parameters.
    :param final_activation: Whether to apply ReLU after the output layer.
    :param init_weight: Callable to initialize layer weights.
    :param init_bias: Callable to initialize layer biases.
    """

    def __init__(
        self,
        nodes: list[int],
        batchnorm: Optional[bool] = False,
        affine: bool = True,
        final_activation: bool = False,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()
        self.nodes = nodes
        self.batchnorm = batchnorm
        self.affine = affine
        self.final_activation = final_activation
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.net = self._construct_net()
        self._apply_weight_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _construct_net(self):
        layers: list[nn.Module] = []

        for i in range(len(self.nodes) - 1):
            is_last = (i == len(self.nodes) - 2)
            layers.append(nn.Linear(self.nodes[i], self.nodes[i + 1]))

            if not is_last:
                if self.batchnorm:
                    layers.append(nn.BatchNorm1d(self.nodes[i + 1], affine=self.affine))
                layers.append(nn.ReLU())
            elif self.final_activation:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _apply_weight_init(self):
        if self.init_weight is not None:
            self.net.apply(self._init_weight_wrapper)
        if self.init_bias is not None:
            self.net.apply(self._init_bias_wrapper)

    def _init_weight_wrapper(self, layer: nn.Module):
        if isinstance(layer, nn.Linear):
            return self.init_weight(layer.weight)
        return None

    def _init_bias_wrapper(self, layer: nn.Module):
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            return self.init_bias(layer.bias)
        return None


class HGQMLP(keras.Model):
    """Multi-layer perceptron in HGQv2.

    :param nodes: List of layer dimensions. nodes[0] is input, nodes[-1] is output.
    :param batchnorm: Whether to use batch normalization after each hidden layer.
    :param affine: Whether batchnorm has learnable parameters.
    :param final_activation: Whether to apply ReLU after output layer.
    :param input_layer_config: Quantization config for input layer.
    :param output_layer_config: Quantization config for output layer.
    :param ebops: Whether to enable energy-based operations.
    """

    def __init__(
        self,
        nodes: list[int],
        batchnorm: bool = False,
        affine: bool = True,
        final_activation: bool = False,
        input_layer_config: dict = None,
        output_layer_config: dict = None,
        ebops: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nodes = nodes
        self.batchnorm = batchnorm
        self.affine = affine
        self.final_activation = final_activation
        self.input_layer_config = input_layer_config
        self.output_layer_config = output_layer_config
        self.ebops = ebops

        self.net = self._construct_net()

    def _construct_net(self):
        layers = []
        num_layers = len(self.nodes) - 1

        def make_qdense(out_dim, name, activation, config=None):
            if config:
                with QuantizerConfigScope(**config, heterogeneous_axis=()):
                    return QDense(out_dim, name=name, activation=activation)
            return QDense(out_dim, name=name, activation=activation)

        with LayerConfigScope(enable_ebops=self.ebops):
            with QuantizerConfigScope(place='all'):
                for i, out_dim in enumerate(self.nodes[1:]):
                    is_first = (i == 0)
                    is_last = (i == num_layers - 1)

                    if is_last:
                        activation = 'relu' if self.final_activation else None
                        layers.append(make_qdense(out_dim, "qdense_out", activation, self.output_layer_config))
                    else:
                        config = self.input_layer_config if is_first else None
                        layers.append(make_qdense(out_dim, f"qdense_{i}", 'relu', config))
                        if self.batchnorm:
                            layers.append(klayers.BatchNormalization(
                                scale=self.affine, center=self.affine, name=f"bn_{i}"
                            ))

        return layers

    def call(self, x):
        for layer in self.net:
            x = layer(x)
        return x
