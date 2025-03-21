from typing import Optional
import torch

import torch

class Quantizer:
    """
    General quantizer class for reference. It can be used as identity operator: Quantizer(None, None).

    :param bits: The total number of bits used for quantization.
    :param integer: The number of bits allocated for the integer part.
    :param integer: The number of bits allocated for the integer part.

    :return: A quantized tensor with values constrained to the specified bit range.
    """
        
    def __init__(self, bits: int, integer: int, **kwargs):
        self.bits = bits
        self.integer = integer

    @classmethod
    def new(cls, method: str, **kwargs):
        """Allow to dinamically instantiate the quantizer based on the method only."""

        # Include here all accepted quantization methods
        assert method.lower() in ["fixed-point"]

        if method.lower() == "fixed-point":
            return FixedPointQuantizer(**kwargs)
        return cls(**kwargs)
      
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def __repr__(self):
        return f"Quantizer(bits={self.bits}, integer={self.integer})"


class FixedPointQuantizer(Quantizer):
    """
    A fixed-point quantizer that scales and quantizes a tensor to a fixed number of bits.

    :param bits: The total number of bits used for quantization.
    :param integer: The number of bits allocated for the integer part.
    :param alpha: A scaling factor applied before quantization (default: 1.0).
    :param symmetric: Whether to use symmetric quantization around zero (default: True).

    :return: A quantized tensor with values constrained to the specified bit range.
    """
        
    def __init__(
            self,
            alpha: Optional[float] = 1.0,
            symmetric: Optional[bool] = True,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.symmetric = symmetric
        
        # Calculate scaling factors
        self.fractional_bits = self.bits - self.integer - 1  # -1 for sign bit
        self.scale = 2.0 ** self.fractional_bits
        if symmetric:
            self.min_val = -2 ** (self.bits - 1) / self.scale
            self.max_val = (2 ** (self.bits - 1) - 1) / self.scale
        else:
            self.min_val = 0
            self.max_val = (2 ** self.bits - 1) / self.scale

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
    
    def __repr__(self):
        return f"FixedPointQuantizer(bits={self.bits}, integer={self.integer}, alpha={self.alpha}, symmetric={self.symmetric})"