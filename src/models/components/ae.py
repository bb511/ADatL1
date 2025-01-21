from typing import Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig

# TODO: These are not the otiginal implementations
from src.models.components.quantized import QuantizedLinear, QuantizedBatchNorm1d, QuantizedReLU


class QuantizedAutoEncoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        encoder_nodes: list,
        decoder_nodes: list,
        precision: Tuple[int, int],
        use_BN: bool
    ):
        super().__init__()
        
        # Build Encoder
        encoder_layers = []
        prev_size = input_features
        for i, node in enumerate(encoder_nodes):
            encoder_layers.append(
                QuantizedLinear(prev_size, node, bits=precision[0], alpha=1.0)
            )

            if use_BN and i != len(decoder_nodes) - 1:
                encoder_layers.append(
                    QuantizedBatchNorm1d(node, bits=precision[0], alpha=1.0)
                )
                
            if i != len(encoder_nodes) - 1:
                encoder_layers.append(QuantizedReLU(bits=precision[0]))
                
            prev_size = node
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build Decoder
        decoder_layers = []
        for i, node in enumerate(decoder_nodes):  # Changed to dot notation
            decoder_layers.append(
                QuantizedLinear(prev_size, node, bits=precision[0], alpha=1.0)
            )
            
            if use_BN and i != len(decoder_nodes) - 1:
                decoder_layers.append(
                    QuantizedBatchNorm1d(node, bits=precision[0], alpha=1.0)
                )
                
            if i != len(decoder_nodes) - 1:
                decoder_layers.append(QuantizedReLU(bits=precision[0]))
                
            prev_size = node
            
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x