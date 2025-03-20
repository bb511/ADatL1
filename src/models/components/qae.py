from typing import Tuple, List
import functools

import torch
import torch.nn as nn

from src.models.quantization import Quantizer
from src.models.quantization.layers import QuantizedLinear
from src.models.quantization.activations import QuantizedReLU, QuantizedBatchNorm1d

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