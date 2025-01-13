import numpy as np
import tensorflow as tf
K = tf.keras.backend
from tensorflow.keras.losses import Loss

class _kld(): # This is a special case and won't inherit the loss class unlike the other loss functions
    def __init__(self):
        super().__init__()
        
    def __call__(self,mu, log_var):
        log_var = K.cast(log_var, dtype='float32')
        mu = K.cast(mu, dtype='float32')
        return -0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
