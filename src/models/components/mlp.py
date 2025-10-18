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


from src.models.quantization import Quantizer
from src.models.quantization.layers import QuantizedLinear
from src.models.quantization.activations import QuantizedReLU, QuantizedBatchNorm1d


class QMLP(nn.Module):
    """
    Quantized multi-layer perceptron.

    :param nodes: Number of nodes composing each of the layers.
    :param batchnorm: Whether to use batch normalization or not.
    :param qweight: Quantizer for the weight parameters.
    :param qbias: Quantizer for the bias parameters.
    :param qactivation: Quantizer for the activation output.
    :param init_weight: Initialization function for the weights.
    :param init_bias: Initialization function for the biases.
    """

    def __init__(
        self,
        nodes: List[int],
        qweight: Optional[Quantizer] = None,
        qbias: Optional[Quantizer] = None,
        qactivation: Optional[Quantizer] = None,
        batchnorm: Optional[bool] = False,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()

        # Partial instantiation of QuantizedLinear and QuantizedBatchNorm1d
        QLinear = functools.partial(
            QuantizedLinear,
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation,
            init_weight=init_weight,
            init_bias=init_bias,
        )
        QBatchNorm1d = functools.partial(
            QuantizedBatchNorm1d,
            qweight=qweight,
            qbias=qbias
        )

        # Build QMLP
        layers = [
            nn.Sequential(
                QLinear(nodes[ilayer - 1], nodes[ilayer]),
                QBatchNorm1d(nodes[ilayer]) if batchnorm else nn.Identity(),
                QuantizedReLU(quantizer=qactivation),
            )
            for ilayer in range(1, len(nodes) - 1)
        ]
        self.net = nn.Sequential(*layers, QLinear(nodes[-2], nodes[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)