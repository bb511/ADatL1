import numpy as np
import tensorflow as tf
K = tf.keras.backend
from tensorflow.keras.losses import Loss

class L1ADBaseLoss(Loss):
    def __init__(self, norm_scales, norm_biases, mask, unscale_energy = False,name="L1ADLOSS"):
        super().__init__(name=name)
        
        MUON_PHI_SCALER = 2 * np.pi / 576
        CALO_PHI_SCALER = 2 * np.pi / 144
        MUON_ETA_SCALER = 0.0870 / 8
        CALO_ETA_SCALER = 0.0870 / 2
        PT_CALO_SCALER = 0.5
        PT_MUON_SCALER = 0.5

        
        PT_SCALER: np.ndarray = np.array([PT_CALO_SCALER] * 13 + [PT_MUON_SCALER] * 8 + [PT_CALO_SCALER] * 12)
        ETA_SCALER: np.ndarray = np.array([CALO_ETA_SCALER] * 13 + [MUON_ETA_SCALER] * 8 + [CALO_PHI_SCALER] * 12)
        PHI_SCALER: np.ndarray = np.array([CALO_PHI_SCALER] * 13 + [MUON_PHI_SCALER] * 8 + [CALO_PHI_SCALER] * 12)
        SCALER = np.concatenate([PT_SCALER, ETA_SCALER, PHI_SCALER]).T.flatten()
        
        mask = np.concatenate(list(np.array(mask[key]) for key in mask.keys())).tolist()
        
        _mask = np.stack([mask] * 3, axis=-1).ravel()
        SCALER = SCALER[_mask]
        PT_SCALER = PT_SCALER[mask]
        ETA_SCALER = ETA_SCALER[mask]
        PHI_SCALER = PHI_SCALER[mask]
        
        NOF_CONSTITUENTS = np.sum(mask)
        NOF_FEATURES = 3 * NOF_CONSTITUENTS
        
        
        norm_biases_ = norm_biases[None,:,:].copy()
        norm_scales_ = norm_scales[None,:,:].copy()
        
        if unscale_energy:
            norm_scales_[:, :, 0] *= PT_SCALER
        norm_scales_[:, :, 1] *= ETA_SCALER
        norm_scales_[:, :, 2] *= PHI_SCALER
        norm_biases_[:, :, 1] *= ETA_SCALER
        norm_biases_[:, :, 2] *= PHI_SCALER

        if unscale_energy:
            norm_scales_[:, :, 0] *= PT_SCALER
            norm_biases_[:, :, 0] *= PT_SCALER
        else:  # Undone bias only, but not scaling (e.g. 0->0, x->x/scale)
            norm_biases_[:, :, 0] /= norm_scales_[:, :, 0]
            norm_scales_[:, :, 0] = 1
            
        self.scales = tf.Variable(norm_scales_, dtype='float32')
        self.biases = tf.Variable(norm_biases_, dtype='float32')
        self.NOF_CONSTITUENTS = NOF_CONSTITUENTS
        self.NOF_FEATURES = NOF_FEATURES