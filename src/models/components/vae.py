from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Tuple, Dict
from omegaconf import DictConfig

from src.models.components.quantized import QuantizedBits, QuantizedLinear, QuantizedReLU

class Sampling(nn.Module):
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class QuantizedEncoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        encoder_config: DictConfig,
        latent_dim: int,
        precision_kernel: Tuple[int, int],
        precision_bias: Tuple[int, int],
        precision_activation: Tuple[int, int],
        precision_data: Tuple[int, int]
    ):
        super().__init__()
        
        # Input quantization
        self.input_quantizer = QuantizedBits(*precision_data)
        
        # Build encoder layers
        layers = []
        prev_size = input_features
        
        for i, node in enumerate(encoder_config.nodes):
            layers.extend([
                QuantizedLinear(
                    prev_size, 
                    node,
                    kernel_quantizer=QuantizedBits(*precision_kernel),
                    bias_quantizer=QuantizedBits(*precision_bias)
                ),
                QuantizedReLU(*precision_activation)
            ])
            prev_size = node
            
        self.encoder_layers = nn.Sequential(*layers)
        
        # Mean and log variance layers
        self.z_mean = QuantizedLinear(
            prev_size, 
            latent_dim,
            kernel_quantizer=QuantizedBits(*precision_kernel),
            bias_quantizer=QuantizedBits(*precision_bias),
            activation_quantizer=QuantizedBits(*precision_activation)
        )
        
        self.z_log_var = QuantizedLinear(
            prev_size, 
            latent_dim,
            kernel_quantizer=QuantizedBits(*precision_kernel),
            bias_quantizer=QuantizedBits(*precision_bias),
            activation_quantizer=QuantizedBits(*precision_activation)
        )
        
        self.sampling = Sampling()
        
        # Initialize log_var weights to zero (as in original)
        with torch.no_grad():
            nn.init.zeros_(self.z_log_var.linear.weight)
            nn.init.zeros_(self.z_log_var.linear.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_quantizer(x)
        x = self.encoder_layers(x)
        
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        
        return z_mean, z_log_var, z

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        decoder_config: DictConfig
    ):
        super().__init__()
        
        layers = []
        prev_size = latent_dim
        
        for i, node in enumerate(decoder_config.nodes):
            # Last layer has special initialization
            if i == len(decoder_config.nodes) - 1:
                layers.append(
                    nn.Linear(prev_size, node)
                )
                nn.init.uniform_(layers[-1].weight, -0.05, 0.05)
            else:
                layers.extend([
                    nn.Linear(prev_size, node),
                    nn.BatchNorm1d(node),
                    nn.ReLU()
                ])
            prev_size = node
            
        self.decoder_layers = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_layers(z)
