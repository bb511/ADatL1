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
    

class QSimpleDenseNet(nn.Module):
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
    

class QuantizedAutoEncoder(nn.Module):
    def __init__(
        self,
        encoder_nodes: List[int],
        decoder_nodes: List[int],
        quantizer: Quantizer,
        batchnorm: bool
    ):
        super().__init__()

        # Partial instantiation of QuantizedLinear and QuantizedBatchNorm1d
        QLinear = functools.partial(
            QuantizedLinear,
            qweight=quantizer,
            qbias=quantizer,
            qactivation=quantizer
        )
        QBatchNorm1d = functools.partial(
            QuantizedBatchNorm1d,
            qweight=quantizer,
            qbias=quantizer
        )

        # Build encoder net
        encoder_layers = [
            nn.Sequential(
                QLinear(encoder_nodes[ilayer - 1], encoder_nodes[ilayer]),
                QBatchNorm1d(encoder_nodes[ilayer]) if batchnorm else nn.Identity(),
                QuantizedReLU(quantizer)
            )
            for ilayer in range(1, len(encoder_nodes) - 1)
        ]
        self.encoder = nn.Sequential(*encoder_layers, nn.Linear(encoder_nodes[-2], encoder_nodes[-1]))

        # Build decoder net
        decoder_layers = [
            nn.Sequential(
                QLinear(decoder_nodes[ilayer - 1], decoder_nodes[ilayer]),
                QBatchNorm1d(decoder_nodes[ilayer]) if batchnorm else nn.Identity(),
                QuantizedReLU(quantizer)
            )
            for ilayer in range(1, len(decoder_nodes) - 1)
        ]
        self.decoder = nn.Sequential(*decoder_layers, nn.Linear(decoder_nodes[-2], decoder_nodes[-1]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x