from typing import Optional
import torch
import torch.nn as nn

from src.models.quantization import Quantizer
from src.models.components.qvae import QuantizedEncoder
from src.models.quantization.activations import QuantizedSigmoid


class BernoulliSampling(nn.Module):
    """Bernoulli sampling layer for MI-VAE."""
    
    def __init__(
        self,
        num_samples: Optional[int] = 10,
        qsigmoid: Optional[Quantizer] = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.sigmoid = QuantizedSigmoid(qsigmoid)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Sample from Bernoulli distribution based on latent z."""
        # Convert z to probabilities
        probs = self.sigmoid(z)
        
        # Sample from Bernoulli distribution
        if self.training:
            samples = []
            for _ in range(self.num_samples):
                sample = torch.bernoulli(probs)
                samples.append(sample)
            z_sample = torch.stack(samples).mean(dim=0)
        else:
            z_sample = probs
        
        return z_sample


class MIVAEEncoder(QuantizedEncoder):
    """Encoder for MI-VAE."""
    
    def __init__(
        self,
        num_samples: Optional[int] = 10,
        qsigmoid: Optional[Quantizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bernoulli_sampling = BernoulliSampling(
            num_samples=num_samples,
            qsigmoid=qsigmoid
        )
    
    def forward(self, x: torch.Tensor):
        z_mean, z_log_var, z = super().forward(x)
        
        # Apply Bernoulli sampling
        z_sample = self.bernoulli_sampling(z)
        
        return z_mean, z_log_var, z, z_sample