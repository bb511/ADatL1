from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.algorithms import L1ADLightningModule
from src.models.components.coupling import CouplingLayer
from src.models.components.masking import ParticleMasking
from src.models.quantization import Quantizer
        

class RealNVP(L1ADLightningModule):
    """RealNVP normalizing flow for L1T anomaly detection.
    
    :param constituents: Data configuration for particles
    :param objects_features: Features per object type
    :param input_dim: Dimension of input features (default 57 for L1T)
    :param n_flows: Number of coupling layers in the flow
    :param hidden_dim: Hidden dimension for coupling networks
    :param conditional_dim: Dimension of conditional context (0 for unconditional)
    :param use_batch_norm: Whether to apply batch normalization in coupling layers
    :param noise_scale: Scale of regularization noise added during training
    :param qweight: Quantizer for weight parameters
    :param qbias: Quantizer for bias parameters
    :param qactivation: Quantizer for activation outputs
    """
    
    def __init__(
        self,
        constituents: Dict[str, List[bool]],
        objects_features: Dict[str, List[str]],
        input_dim: int = 57,
        n_flows: int = 6,
        hidden_dim: int = 32,
        conditional_dim: int = 0,
        use_batch_norm: bool = True,
        noise_scale: float = 0.01,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["loss", "qweight", "qbias", "qactivation"])
        
        self.input_dim = input_dim
        self.conditional_dim = conditional_dim
        self.noise_scale = noise_scale
        
        # Build coupling layers with alternating masks
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            mask_pattern = self._get_mask_pattern(i)
            masking = ParticleMasking(
                constituents=constituents,
                objects_features=objects_features,
                mask_probs=mask_pattern,
                mask_value=0.0,
                training_only=False  # Always apply, not training-specific
            )
            
            self.flows.append(
                CouplingLayer(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    masking=masking,
                    conditional_dim=conditional_dim,
                    use_batch_norm=use_batch_norm
                )
            )
        
        # Base distribution
        self.register_buffer('base_mean', torch.zeros(input_dim))
        self.register_buffer('base_std', torch.ones(input_dim))
    
    def _get_mask_pattern(self, layer_idx: int) -> Dict[str, float]:
        """Generate mask pattern for each layer - alternating which particles are masked."""
        if layer_idx % 2 == 0:
            # Even layers: mask muons and jets
            return {
                'muons': 1.0,
                'egammas': 0.0,
                'jets': 1.0,
                'ET': 0.0,
                'MET': 0.0,
            }
        else:
            # Odd layers: mask egammas and MET
            return {
                'muons': 0.0,
                'egammas': 1.0,
                'jets': 0.0,
                'ET': 1.0,
                'MET': 1.0,
            }
    
    def encode(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform data to latent space, computing log likelihood."""
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for flow in self.flows:
            z, log_det = flow(z, context)
            log_det_sum += log_det
        
        return z, log_det_sum
    
    def decode(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform from latent space back to data space."""
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        x = z
        
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x, context)
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log probability of data."""
        
        # Add small noise for regularization during training
        if self.training and self.noise_scale > 0:
            x = x + torch.randn_like(x) * self.noise_scale
        
        # Transform to latent space
        z, log_det = self.encode(x, context)
        
        # Compute log probability under base distribution
        log_pz = -0.5 * (
            ((z - self.base_mean) / self.base_std) ** 2 + 
            torch.log(2 * np.pi * self.base_std ** 2)
        ).sum(dim=1)
        
        # Change of variables formula
        log_px = log_pz + log_det
        
        return log_px
    
    def sample(self, n_samples: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from the learned distribution."""
        z = torch.randn(n_samples, self.input_dim, device=self.base_mean.device)
        x, _ = self.decode(z, context)
        return x
    
    # def get_context(self, x: torch.Tensor) -> Optional[torch.Tensor]:
    #     """Extract context features from input (e.g., multiplicity)."""
    #     if self.conditional_dim == 0:
    #         return None
            
    #     # Count non-zero particles (checking ET values)
    #     et_indices = list(range(0, 12, 3)) + list(range(12, 24, 3)) + list(range(24, 54, 3))
    #     multiplicity = (x[:, et_indices] > 0.01).sum(dim=1, keepdim=True).float()
        
    #     # Normalize multiplicity
    #     multiplicity = (multiplicity - 9.) / 3.  # Rough normalization
        
    #     # Could add more context features
    #     if self.conditional_dim > 1:
    #         # Total energy
    #         total_et = x[:, et_indices].sum(dim=1, keepdim=True)
    #         total_et = (total_et - total_et.mean()) / (total_et.std() + 1e-6)
            
    #         context = torch.cat([multiplicity, total_et], dim=1)
    #     else:
    #         context = multiplicity
            
    #     return context
    
    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass and loss computation."""
        x, _ = batch

        # context = self.get_context(x)
        context = None
        
        # Compute negative log likelihood
        loss = - self.log_prob(x, context)
        
        del context        

        return {
            "loss": loss.mean(),
            "loss/nll/mean": loss.mean(),
        }
    
    def _filter_log_dict(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss")
        }