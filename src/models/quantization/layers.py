from typing import Optional, Callable

import torch
from torch import nn
import torch.nn.functional as F

from src.models.quantization import Quantizer

class QuantizedLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        qweight: Optional["Quantizer"] = None,
        qbias: Optional["Quantizer"] = None,
        qactivation: Optional["Quantizer"] = None,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None
    ):
        super().__init__(in_features, out_features)

        # Default quantizers (identity if None)
        self.qweight = qweight or Quantizer(None, None)
        self.qbias = qbias or Quantizer(None, None)
        self.qactivation = qactivation or Quantizer(None, None)

        # Apply weight and bias initialization
        if init_weight != None: init_weight(self.weight)
        if init_bias != None and self.bias != None: init_bias(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights and biases
        weight = self.qweight(self.weight)
        bias = self.qbias(self.bias) if self.bias is not None else None

        # Apply the linear transformation and quantize activation
        return self.qactivation(F.linear(x, weight, bias))

    
    