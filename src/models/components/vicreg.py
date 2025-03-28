from typing import Optional, List
import torch
from torch import nn
    

class FastFeatureBlur(nn.Module):
    """
    Selectively blurs features by mixing with random values.

    :param p: Probability of applying the noise transformation (0 to 1).
    :param magnitude: Intensity of the blurring effect.
    :param strength: Probability of each feature being affected.
    """

    def __init__(self, prob: float, magnitude: float, strength: float):
        super().__init__()
        self.prob = prob
        self.magnitude = magnitude
        self.strength = strength

    def forward(self, x):
        batch_size, feature_dim = x.shape
        
        # Mask for items to be blurred and features within each item
        mask_p = (torch.rand(batch_size, 1) < self.prob).float()
        mask_strength = (torch.rand(batch_size, feature_dim) < self.strength).float()
        
        # Combined mask with magnitude
        mask = mask_p * mask_strength * self.magnitude
        
        # Apply blurring: original*(1-mask) + random*mask
        return x * (1 - mask) + torch.rand_like(x) * mask


class FastObjectMask(nn.Module):
    """
    Randomly zeros out a subset of features in the input tensor during training.

    :param p: Probability of applying the masking transformation (0 to 1).
    """

    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        # TODO: Check fast object mask implementation
        # This is not clear
        # batch = batch.reshape((-1, 19, 3))
        # mask = torch.rand((batch.shape[0], batch.shape[1]), device=self.device)
        # idx = mask < self.p
        # mask[idx] = 0
        # mask[~idx] = 1
        # batch = batch * mask[:, :, None]
        # batch = batch.reshape((-1, 57))

        if torch.rand(1).item() > self.prob:
            return x
        
        batch_size, feature_dim = x.shape
        mask = (torch.rand(batch_size, feature_dim) > self.prob).float()
        return x * mask


class FastLorentzRotation(nn.Module):
    """
    Applies a random rotation to the phi angles of input features with given probability.
    
    :param norm_scale: Normalization scale factors
    :param norm_bias: Normalization bias values
    :param p: Probability of applying the rotation to each batch item
    :param phi_indices: Indices of phi angles in the feature vector
    """
    def __init__(
            self,
            prob: float,
            norm_scale: List[float],
            norm_bias: List[float],
            phi_indices: Optional[list] = None
        ):
        super().__init__()
        self.prob = prob
        
        # Use register_buffer to store them in the state dictionary but not as model parameters
        phi_indices = phi_indices if phi_indices else torch.arange(0, 19, 1) + 2
        self.register_buffer('l1_scale', torch.tensor([144] * 1 + [144] * 4 + [576] * 4 + [144] * 10) / (2 * torch.pi))
        self.register_buffer('phi_indices', phi_indices)

        # TODO: Fix scale and bias:
        self.register_buffer('scale', torch.ones(self.phi_indices.shape[0]).float())
        self.register_buffer('bias', torch.tensor(self.phi_indices.shape[0]).float())
    

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        
        # Create mask for batch items to rotate
        bool_mask = (torch.rand(batch_size, device=device) < self.prob).float()
        
        # Extract and normalize phi values
        original_phi = (x[:, self.phi_indices] * self.scale + self.bias) / self.l1_scale
        
        # Generate random rotation angles
        rotation = (torch.rand(batch_size, device=device) * 2 * torch.pi)[:, None]
        
        # Apply rotation and normalize back
        rotated_phi = (torch.remainder((original_phi + rotation), 2 * torch.pi)) * self.l1_scale
        
        # Create new tensor to avoid modifying the input directly
        result = x.clone()
        
        # Replace values based on the mask
        result[:, self.phi_indices] = (
            (bool_mask[:, None] * rotated_phi + (1 - bool_mask[:, None]) * original_phi).float() - self.bias
        ) / self.scale
        return result