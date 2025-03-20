import torch

class Quantizer:
    """
    General quantizer class for reference. It can be used as identity operator: Quantizer(None, None).

    :param bits: The total number of bits used for quantization.
    :param integer: The number of bits allocated for the integer part.

    :return: A quantized tensor with values constrained to the specified bit range.
    """
        
    def __init__(self, bits: int, integer: int):
        self.bits = bits
        self.integer = integer
      
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def __repr__(self):
        return f"Quantizer(bits={self.bits}, integer={self.integer})"