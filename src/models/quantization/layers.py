from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from src.models.quantization import Quantizer

class QuantizedLinear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        qweight: Optional[Quantizer] = None,
        qbias: Optional[Quantizer] = None,
        qactivation: Optional[Quantizer] = None
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # If no quantizer provided, identity quantizer is used
        self.qweight = qweight or Quantizer(None, None)
        self.qbias = qbias or Quantizer(None, None)
        self.qactivation = qactivation or Quantizer(None, None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights and biases
        weight = self.qweight(self.linear.weight)
        bias = self.qbias(self.linear.bias) if self.linear.bias is not None else None
        
        # Quantize activation output
        return self.qactivation(
            F.linear(x, weight, bias)
        )