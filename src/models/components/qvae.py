from typing import Optional, Tuple, List, Callable
import torch
import torch.nn as nn

from src.models.quantization import Quantizer
from src.models.quantization.layers import QuantizedLinear
from src.models.components.qmlp import QMLP


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
        qmlp = QMLP(
            nodes,
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation,
            batchnorm=False,
        )
        self.net = nn.Sequential(*list(qmlp.net.children())[:-1])

        # Mean and log variance layers
        self.z_mean = QuantizedLinear(
            in_features=nodes[-2],
            out_features=nodes[-1],
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation,
        )
        self.z_log_var = QuantizedLinear(
            in_features=nodes[-2],
            out_features=nodes[-1],
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation,
            # Initialize log_var weights to zero
            init_weight=nn.init.zeros_,
            init_bias=nn.init.zeros_,
        )

        # Reparameterization layer
        self.sampling = Sampling()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through the network
        x = self.net(self.qdata(x))

        # Mean and log variance for reparemeterization
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """Decoder for AD@L1."""

    def __init__(
        self,
        nodes: List[int],
        init_last_weight: Optional[Callable] = None,
        init_last_bias: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        init_last_weight = init_last_weight if init_last_weight else lambda _: None
        init_last_bias = init_last_bias if init_last_bias else lambda _: None

        # Build the decoder as a MLP (no quantization) with batch normalization
        self.net = QMLP(
            nodes, qweight=None, qbias=None, qactivation=None, batchnorm=True
        )

        # Apply initialization to the weight of the last Linear() layer
        last_linear_layer = self.net.net[-1]
        if init_last_weight != None:
            init_last_weight(last_linear_layer.weight)
        if init_last_weight != None and last_linear_layer.bias != None:
            init_last_bias(last_linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
