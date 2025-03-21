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

    :param nodes: Number of nodes composing each of the layers.
    :param qweight: Quantizer for the weight parameters.
    :param qbias: Quantizer for the bias parameters.
    :param qactivation: Quantizer for the activation output.
    """
    def __init__(
            self,
            nodes: List[int],
            qweight: Optional[Quantizer] = None,
            qbias: Optional[Quantizer] = None,
            qactivation: Optional[Quantizer] = None,
            batchnorm: Optional[bool] = False
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

        # Build MLP
        layers = [
            nn.Sequential(
                QLinear(nodes[ilayer - 1], nodes[ilayer]),
                QBatchNorm1d(nodes[ilayer]) if batchnorm else nn.Identity(),
                QuantizedReLU(quantizer=qactivation)
            )
            for ilayer in range(1, len(nodes) - 1)
        ]
        self.net = nn.Sequential(*layers, QuantizedLinear(nodes[-2], nodes[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class QMLPAutoEncoder(nn.Module):
    """
    Quantized MLP encoder-decoder network.

    :param encoder_nodes: Number of nodes composing each of the encoder layers.
    :param decoder_nodes: Number of nodes composing each of the encoder layers.
    :param quantizer: Quantizer for all the parameters.
    :param batchnorm: Whether to use batch normalization or not.
    """

    def __init__(
        self,
        encoder_nodes: List[int],
        decoder_nodes: List[int],
        quantizer: Optional[Quantizer] = None,
        batchnorm: Optional[bool] = True
    ):
        super().__init__()
        self.encoder = QMLP(encoder_nodes, qweight=quantizer, qbias=quantizer, qactivation=quantizer, batchnorm=batchnorm)
        self.decoder = QMLP(decoder_nodes, qweight=quantizer, qbias=quantizer, qactivation=quantizer, batchnorm=batchnorm)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x