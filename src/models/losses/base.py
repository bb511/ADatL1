import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class ScalingConstants:
    MUON_PHI_SCALER: float = 2 * np.pi / 576
    CALO_PHI_SCALER: float = 2 * np.pi / 144
    MUON_ETA_SCALER: float = 0.0870 / 8
    CALO_ETA_SCALER: float = 0.0870 / 2
    PT_CALO_SCALER: float = 0.5
    PT_MUON_SCALER: float = 0.5

class L1ADBaseLoss(nn.Module):
    def __init__(self, norm_scales: np.ndarray, norm_biases: np.ndarray, 
                 mask: Dict[str, np.ndarray], unscale_energy: bool = False):
        super().__init__()
        
        constants = ScalingConstants()
        
        # Create scalers
        PT_SCALER = np.array([constants.PT_CALO_SCALER] * 13 + 
                            [constants.PT_MUON_SCALER] * 8 + 
                            [constants.PT_CALO_SCALER] * 12)
        ETA_SCALER = np.array([constants.CALO_ETA_SCALER] * 13 + 
                             [constants.MUON_ETA_SCALER] * 8 + 
                             [constants.CALO_PHI_SCALER] * 12)
        PHI_SCALER = np.array([constants.CALO_PHI_SCALER] * 13 + 
                             [constants.MUON_PHI_SCALER] * 8 + 
                             [constants.CALO_PHI_SCALER] * 12)
        SCALER = np.concatenate([PT_SCALER, ETA_SCALER, PHI_SCALER]).T.flatten()
        
        # Process mask
        mask = np.concatenate([np.array(mask[key]) for key in mask.keys()]).tolist()
        _mask = np.stack([mask] * 3, axis=-1).ravel()
        
        # Apply mask to scalers
        SCALER = SCALER[_mask]
        PT_SCALER = PT_SCALER[mask]
        ETA_SCALER = ETA_SCALER[mask]
        PHI_SCALER = PHI_SCALER[mask]
        
        # Calculate dimensions
        self.NOF_CONSTITUENTS = np.sum(mask)
        self.NOF_FEATURES = 3 * self.NOF_CONSTITUENTS
        
        # Process normalization parameters
        norm_scales_ = norm_scales[None, :, :].copy()
        norm_biases_ = norm_biases[None, :, :].copy()
        
        if unscale_energy:
            norm_scales_[:, :, 0] *= PT_SCALER
        norm_scales_[:, :, 1] *= ETA_SCALER
        norm_scales_[:, :, 2] *= PHI_SCALER
        norm_biases_[:, :, 1] *= ETA_SCALER
        norm_biases_[:, :, 2] *= PHI_SCALER
        
        if unscale_energy:
            norm_scales_[:, :, 0] *= PT_SCALER
            norm_biases_[:, :, 0] *= PT_SCALER
        else:
            norm_biases_[:, :, 0] /= norm_scales_[:, :, 0]
            norm_scales_[:, :, 0] = 1
            
        # Convert to PyTorch parameters
        self.register_buffer('scales', torch.tensor(norm_scales_, dtype=torch.float32))
        self.register_buffer('biases', torch.tensor(norm_biases_, dtype=torch.float32))