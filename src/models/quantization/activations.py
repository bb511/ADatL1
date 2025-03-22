from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.quantization import Quantizer

    
class QuantizedReLU(nn.ReLU):
    def __init__(self, quantizer: Quantizer):
        super().__init__()
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer(super().forward(x))

class QuantizedBatchNorm1d(nn.Module):
    def __init__(
            self,
            num_features: int,
            qweight: Optional[Quantizer] = None,
            qbias: Optional[Quantizer] = None,
            qmean: Optional[Quantizer] = None,
            qvar: Optional[Quantizer] = None
        ):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

        # If no quantizer provided, identity quantizer is used
        self.qweight = qweight or Quantizer(None, None)
        self.qbias = qbias or Quantizer(None, None)
        self.qmean = qmean or Quantizer(None, None)
        self.qvar = qvar or Quantizer(None, None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, use standard BatchNorm but quantize parameters
            self.bn.weight.data = self.qweight(self.bn.weight.data)
            self.bn.bias.data = self.qbias(self.bn.bias.data)
            return self.bn(x)
        else:
            # During inference, use quantized parameters and statistics
            weight = self.qweight(self.bn.weight)
            bias = self.qbias(self.bn.bias)
            mean = self.qmean(self.bn.running_mean)
            var = self.qvar(self.bn.running_var)
            
            return F.batch_norm(
                x, mean, var, weight, bias,
                training=False, momentum=self.bn.momentum, eps=self.bn.eps
            )