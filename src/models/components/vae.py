from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from src.models.quantization import Quantizer
from src.models.quantization.layers import QuantizedLinear
from src.models.quantization.layers import QuantizedLinear, QuantizedBatchNorm1d

from src.models.components.dense import QMLP

class Sampling(nn.Module):
    """Sampling layer for VAE. Performs reparameterization trick."""

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class QuantizedEncoder(nn.Module):
    """Quantized encoder for AD@L1."""

    def __init__(
        self,
        nodes: List[int],
        qdata: Optional[Quantizer] = None,
        qweight: Optional[Quantizer] = None,
        qbias: Optional[Quantizer] = None,
        qactivation: Optional[Quantizer] = None,
    ):
        super().__init__()
        
        # Input data quantization
        self.qdata = qdata or Quantizer(None, None)

        # The encoder will be a QMLP up to the last layer
        qmlp = QMLP(nodes, qweight=qweight, qbias=qbias, qactivation=qactivation, batchnorm=False)
        self.net = nn.Sequential(*list(qmlp.children())[:-1])
     
        # Mean and log variance layers
        self.z_mean = QuantizedLinear(
            in_features=nodes[-2],
            out_features=nodes[-1],
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation
        )
        self.z_log_var = QuantizedLinear(
            in_features=nodes[-2],
            out_features=nodes[-1],
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation
        )
        
        # Reparameterization layer
        self.sampling = Sampling()
        
        # TODO: Check if this makes sense
        # Initialize log_var weights to zero
        with torch.no_grad():
            nn.init.zeros_(self.z_log_var.linear.weight)
            nn.init.zeros_(self.z_log_var.linear.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through the network
        x = self.net(self.qdata(x))
        
        # Mean and log variance for reparemeterization
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """Decoder for AD@L1."""

    def __init__(self, nodes: List[int]):
        super().__init__()

        # Build the decoder as a MLP (no quantization) with batch normalization 
        self.net = QMLP(nodes, qweight=None, qbias=None, qactivation=None, batchnorm=True)

        # Apply uniform initialization to the weight of the last Linear() layer
        last_layer = list(self.net.children())[-1]
        nn.init.uniform_(last_layer.weight, -0.05, 0.05)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
