from typing import Optional, List
import functools

import torch
from torch import nn

from src.models.quantization import Quantizer
from src.models.quantization.layers import QuantizedLinear
from src.models.quantization.activations import QuantizedReLU, QuantizedBatchNorm1d

class QMLP(nn.Module):
    """
    Quantized multi-layer perceptron.

    :param nodes: A list of integers representing the number of nodes in each layer.
    :param qweight: Quantizer for the weight parameters.
    :param qbias: Quantizer for the bias parameters.
    :param qactivation: Quantizer for the activation output.
    """
    def __init__(
            self,
            nodes: List[int],
            qweight: Optional[Quantizer] = None,
            qbias: Optional[Quantizer] = None,
            qactivation: Optional[Quantizer] = None
        ):
        super().__init__()
       
        # Partial instantiation of QuantizedLinear
        QLinear = functools.partial(
            QuantizedLinear,
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation
        )

        # Build MLP
        layers = [
            nn.Sequential(
                QLinear(nodes[ilayer - 1], nodes[ilayer]),
                QuantizedReLU(quantizer=qactivation)
            )
            for ilayer in range(1, len(nodes) - 1)
        ]
        self.net = nn.Sequential(*layers, nn.Linear(nodes[-2], nodes[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class SimpleDenseNet(nn.Module):
    """
    Quantized simple fully-connected neural network.

    :param nodes: A list of integers representing the number of nodes in each layer.
    :param qweight: Quantizer for the weight parameters.
    :param qbias: Quantizer for the bias parameters.
    :param qactivation: Quantizer for the activation output.
    """

    def __init__(
        self,
        nodes: List[int],
        qweight: Optional[Quantizer] = None,
        qbias: Optional[Quantizer] = None,
        qactivation: Optional[Quantizer] = None
    ):
        super().__init__()

        # Partial instantiation of QuantizedLinear and QuantizedBatchNorm1d
        QLinear = functools.partial(
            QuantizedLinear,
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation
        )
        QBatchNorm1d = functools.partial(
            QuantizedBatchNorm1d,
            qweight=qweight,
            qbias=qbias
        )

        # Build net
        layers = [
            nn.Sequential(
                QLinear(nodes[ilayer - 1], nodes[ilayer]),
                QBatchNorm1d(nodes[ilayer]),
                QuantizedReLU(quantizer=qactivation)
            )
            for ilayer in range(1, len(nodes) - 1)
        ]
        self.net = nn.Sequential(*layers, nn.Linear(nodes[-2], nodes[-1]))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)