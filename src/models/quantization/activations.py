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


class QuantizedBatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features: int,
        qweight: Optional[Quantizer] = None,
        qbias: Optional[Quantizer] = None,
        qmean: Optional[Quantizer] = None,
        qvar: Optional[Quantizer] = None,
    ):
        super().__init__(num_features)  # Initialize nn.BatchNorm1d

        # If no quantizer provided, use an identity quantizer
        self.qweight = qweight or Quantizer(None, None)
        self.qbias = qbias or Quantizer(None, None)
        self.qmean = qmean or Quantizer(None, None)
        self.qvar = qvar or Quantizer(None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, use standard BatchNorm but quantize parameters
            self.weight.data = self.qweight(self.weight.data)
            self.bias.data = self.qbias(self.bias.data)
            return super().forward(x)
        else:
            # During inference, use quantized parameters and statistics
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
