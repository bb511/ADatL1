from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.quantization import Quantizer


class QuantizedReLU(nn.ReLU):
    def __init__(self, quantizer: Optional[Quantizer] = None):
        super().__init__()
        self.quantizer = quantizer or Quantizer(None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer(super().forward(x))


class QuantizedSigmoid(nn.Module):
    """Quantized sigmoid activation function."""

    def __init__(self, quantizer: Optional[Quantizer] = None):
        super().__init__()
        self.quantizer = quantizer or Quantizer(None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer(torch.sigmoid(x))


class QuantizedTanh(nn.Module):
    """Quantized Tanh activation.
    
    :param quantizer: Quantizer to apply to output
    """
    def __init__(self, quantizer: Optional[Quantizer] = None):
        super().__init__()
        self.quantizer = quantizer or Quantizer(None, None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer(torch.tanh(x))
    

class QuantizedBatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features: int,
        qweight: Optional[Quantizer] = None,
        qbias: Optional[Quantizer] = None,
        qmean: Optional[Quantizer] = None,
        qvar: Optional[Quantizer] = None,
    ):
        super().__init__(num_features)
        self.qweight = qweight or Quantizer(None, None)
        self.qbias = qbias or Quantizer(None, None)
        self.qmean = qmean or Quantizer(None, None)
        self.qvar = qvar or Quantizer(None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.qweight(self.weight)
            bias = self.qbias(self.bias)
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                weight,
                bias,
                training=True,
                momentum=self.momentum,
                eps=self.eps,
            )
        else:
            weight = self.qweight(self.weight)
            bias = self.qbias(self.bias)
            mean = self.qmean(self.running_mean)
            var = self.qvar(self.running_var)
            return F.batch_norm(
                x,
                mean,
                var,
                weight,
                bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
