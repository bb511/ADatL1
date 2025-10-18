from typing import Dict, List, Optional
import torch
import torch.nn as nn


class ParticleMasking(nn.Module):
    """Masks entire particles/objects based on the data configuration.

    :param constituents: Dictionary with a bool list of which constituents to
        include, corresponding to each particle object, e.g.
        muons: [True, True, True, True, False, False, False, False]
    :param objects_features: Dictionary with which objects to extract from the
        h5 file (keys), and the corresponding features to get from the object.
    :param mask_probs: Dict with masking probability per object type
    :param mask_value: Value to use for masked features
    :param training_only: Whether to apply masking only during training
    """
    
    def __init__(
        self,
        constituents: Dict[str, List[bool]],
        objects_features: Dict[str, List[str]],
        mask_probs: Optional[Dict[str, float]] = None,
        mask_value: float = 0.0,
        training_only: bool = True,
    ):
        super().__init__()
        self.mask_value = mask_value
        self.training_only = training_only
        
        # Default masking probabilities
        self.mask_probs = mask_probs or {
            'muons': 0.15,
            'egammas': 0.15,
            'jets': 0.15,
            'ET': 0.05,
            'MET': 0.05,
        }
        self.particle_slices = self._build_particle_slices(constituents, objects_features)
        
    def _build_particle_slices(self, constituents: Dict, objects_features: Dict) -> Dict:
        """Dynamically build particle boundaries from data configuration."""
        slices = {}
        current_idx = 0
        for obj_name, features in objects_features.items():
            n_features = len(features)
            
            if obj_name in constituents:
                # Particle objects with multiple constituents
                n_constituents = sum(constituents[obj_name])
                particle_slices = []
                for i in range(n_constituents):
                    start_idx = current_idx + i * n_features
                    end_idx = start_idx + n_features
                    particle_slices.append((start_idx, end_idx))
                slices[obj_name] = particle_slices
                current_idx += n_constituents * n_features
            else:
                # Single objects (ET, MET)
                end_idx = current_idx + n_features
                slices[obj_name] = [(current_idx, end_idx)]
                current_idx = end_idx
                
        return slices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training_only and not self.training:
            return x
            
        batch_size = x.shape[0]
        x_masked = x.clone()
        
        for obj_type, particle_slices in self.particle_slices.items():
            mask_prob = self.mask_probs.get(obj_type, 0.1)
            
            for start_idx, end_idx in particle_slices:
                mask = torch.rand(batch_size, device=x.device) < mask_prob
                x_masked[mask, start_idx:end_idx] = self.mask_value
                
        return x_masked


class MultiplicityMasking(nn.Module):
    """Masks particles to reduce multiplicity bias in anomaly detection.
    
    Args:
    :param constituents: Dictionary with a bool list of which constituents to
        include, corresponding to each particle object, e.g.
        muons: [True, True, True, True, False, False, False, False]
    :param objects_features: Dictionary with which objects to extract from the
        h5 file (keys), and the corresponding features to get from the object.
    :param high_mult_mask_prob: Probability of masking in high-multiplicity events
    :param low_mult_mask_prob: Probability of masking in low-multiplicity events
    :param mult_threshold_percentile: Percentile to determine high multiplicity
    :param et_feature_idx: Index of ET/pT feature in each object (default 0)
    :param training_only: Whether to apply masking only during training
    """
    
    def __init__(
        self,
        constituents: Dict[str, List[bool]],
        objects_features: Dict[str, List[str]],
        high_mult_mask_prob: float = 0.3,
        low_mult_mask_prob: float = 0.05,
        mult_threshold_percentile: float = 75.0,
        et_feature_idx: int = 0,
        training_only: bool = True,
    ):
        super().__init__()
        self.high_mult_mask_prob = high_mult_mask_prob
        self.low_mult_mask_prob = low_mult_mask_prob
        self.mult_threshold_percentile = mult_threshold_percentile
        self.et_feature_idx = et_feature_idx
        self.training_only = training_only
        
        # Build particle information from configuration
        self._build_particle_info(constituents, objects_features)
        
    def _build_particle_info(self, constituents: Dict, objects_features: Dict):
        """Extract particle boundaries and ET indices from configuration."""
        self.particle_info = {}
        current_idx = 0
        
        for obj_name, features in objects_features.items():
            n_features = len(features)
            
            # Skip non-particle objects for multiplicity counting
            if obj_name in ['ET', 'MET']:
                current_idx += n_features
                continue
                
            if obj_name in constituents:
                n_constituents = sum(constituents[obj_name])
                et_indices = []
                particle_boundaries = []
                
                for i in range(n_constituents):
                    start_idx = current_idx + i * n_features
                    end_idx = start_idx + n_features
                    et_idx = start_idx + self.et_feature_idx  # ET is typically first
                    
                    et_indices.append(et_idx)
                    particle_boundaries.append((start_idx, end_idx))
                
                self.particle_info[obj_name] = {
                    'et_indices': et_indices,
                    'boundaries': particle_boundaries,
                    'n_features': n_features,
                }
                
                current_idx += n_constituents * n_features
    
    def count_multiplicity(self, x: torch.Tensor) -> torch.Tensor:
        """Count active particles per sample."""
        multiplicities = torch.zeros(x.shape[0], device=x.device)
        
        for obj_name, info in self.particle_info.items():
            for et_idx in info['et_indices']:
                # Count particles with ET above threshold
                active = (x[:, et_idx] > 0.01).float()
                multiplicities += active
                
        return multiplicities
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training_only and not self.training:
            return x
            
        batch_size = x.shape[0]
        x_masked = x.clone()
        
        # Count multiplicities
        multiplicities = self.count_multiplicity(x)
        
        # Determine high multiplicity threshold
        if multiplicities.numel() > 1:
            threshold = torch.quantile(multiplicities, self.mult_threshold_percentile / 100.0)
        else:
            threshold = multiplicities.median()
        
        # Mask each sample based on its multiplicity
        for i in range(batch_size):
            mult = multiplicities[i].item()
            
            # Determine masking probability
            if mult > threshold:
                mask_prob = self.high_mult_mask_prob
            else:
                mask_prob = self.low_mult_mask_prob
            
            # Apply masking to particles
            for obj_name, info in self.particle_info.items():
                for (start_idx, end_idx), et_idx in zip(info['boundaries'], info['et_indices']):
                    # Only mask active particles
                    if x_masked[i, et_idx] > 0.01:
                        if torch.rand(1).item() < mask_prob:
                            x_masked[i, start_idx:end_idx] = 0.0
        
        return x_masked