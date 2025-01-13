import numpy as np
import tensorflow as tf
K = tf.keras.backend
from tensorflow.keras.losses import Loss
from .base import L1ADBaseLoss

class _cyl_PtPz_loss(L1ADBaseLoss):
    def __init__(self, norm_scales, norm_biases, mask, unscale_energy = False, name="Cyl_PtPz"):
        super().__init__(norm_scales, norm_biases, mask, unscale_energy, name = name)
        
    def call(self,y_true, y_pred):

        y_pred = K.reshape(y_pred, (-1, self.NOF_CONSTITUENTS, 3))
        y_true = K.reshape(y_true, (-1, self.NOF_CONSTITUENTS, 3))
        y_true = K.cast(y_true, dtype='float32') * self.scales + self.biases
        y_pred = K.cast(y_pred, dtype='float32') * self.scales + self.biases
        pt, eta = y_true[:, :, 0], y_true[:, :, 1]
        pz = pt * tf.math.sinh(eta)
        pt_pred, eta_pred = y_pred[:, :, 0], y_pred[:, :, 1]
        # pz_pred = pt_pred * tf.math.sinh(eta_pred) ## <----- This looks sane
        pz_pred = pt_pred * tf.math.sinh(eta) ## <----- This is the legacy implementation

        return K.mean(K.square(pt - pt_pred) + K.square(pz - pz_pred), axis=1)
