import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedBits:
    def __init__(self, bits: int, integer: int, alpha: float = 1.0, symmetric: bool = True):
        self.bits = bits
        self.integer = integer
        self.alpha = alpha
        self.symmetric = symmetric
        
        # Calculate scaling factors
        self.fractional_bits = bits - integer - 1  # -1 for sign bit
        self.scale = 2.0 ** self.fractional_bits
        if symmetric:
            self.min_val = -2 ** (bits - 1) / self.scale
            self.max_val = (2 ** (bits - 1) - 1) / self.scale
        else:
            self.min_val = 0
            self.max_val = (2 ** bits - 1) / self.scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Scale input by alpha
        x = x * self.alpha
        
        # Clamp values to quantization range
        x = torch.clamp(x, self.min_val, self.max_val)
        
        # Scale to integer range for quantization
        x = x * self.scale
        
        # Quantize
        x = torch.round(x)
        
        # Scale back to original range
        x = x / self.scale
        
        return x

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bits: int, integer: int, alpha: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_quantizer = QuantizedBits(bits, integer, alpha)
        self.bias_quantizer = QuantizedBits(bits, integer, alpha)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights and biases
        weight_q = self.weight_quantizer(self.linear.weight)
        bias_q = self.bias_quantizer(self.linear.bias) if self.linear.bias is not None else None
        
        # Compute output using quantized parameters
        return F.linear(x, weight_q, bias_q)

class QuantizedReLU(nn.Module):
    def __init__(self, bits: int, integer: int):
        super().__init__()
        self.quantizer = QuantizedBits(bits, integer, symmetric=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer(F.relu(x))

class QuantizedBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, bits: int, integer: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.weight_quantizer = QuantizedBits(bits, integer)
        self.bias_quantizer = QuantizedBits(bits, integer)
        self.mean_quantizer = QuantizedBits(bits, integer)
        self.var_quantizer = QuantizedBits(bits, integer, symmetric=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, use standard BatchNorm but quantize parameters
            self.bn.weight.data = self.weight_quantizer(self.bn.weight.data)
            self.bn.bias.data = self.bias_quantizer(self.bn.bias.data)
            return self.bn(x)
        else:
            # During inference, use quantized parameters and statistics
            weight_q = self.weight_quantizer(self.bn.weight)
            bias_q = self.bias_quantizer(self.bn.bias)
            mean_q = self.mean_quantizer(self.bn.running_mean)
            var_q = self.var_quantizer(self.bn.running_var)
            
            return F.batch_norm(
                x, mean_q, var_q, weight_q, bias_q,
                training=False, momentum=self.bn.momentum, eps=self.bn.eps
            )