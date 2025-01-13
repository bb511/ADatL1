import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from qkeras import QBatchNormalization
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu


class AutoEncoder(Model):
    def __init__(self, config):
        super().__init__()
        encoder_config = config["encoder_config"]
        decoder_config = config["decoder_config"]
        features = config["features"]
        ap_fixed = config["ap_fixed"]
        use_BN = config["use BN"]
        
        self.encoder = Sequential()
        
        for i,node in enumerate(encoder_config["nodes"]):
            self.encoder.add(QDense(node,
                               name=f'hd_encoder{i+1}',
                               kernel_quantizer=quantized_bits(*ap_fixed,alpha=1),
                               bias_quantizer=quantized_bits(*ap_fixed,alpha=1),
                               kernel_initializer='glorot_uniform',
                               # kernel_regularizer=regularizers.l1(0) ### commented out after Chang's finding ...
                              ))
            if use_BN and i!=len(decoder_config["nodes"])-1:
                self.encoder.add(QBatchNormalization(name=f'BN_encoder{i+1}',
                                              beta_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                              gamma_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                              mean_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                              variance_quantizer=quantized_bits(*ap_fixed,alpha=1)))

            if  i!= len(encoder_config["nodes"])-1:
                self.encoder.add(QActivation(name=f'act_encoder{i+1}',
                                      activation=quantized_relu(*ap_fixed)))
        
        
        self.decoder = Sequential()
        for i,node in enumerate(decoder_config["nodes"]):
            self.decoder.add(QDense(node,
                               name=f'hd_decoder{i+1}',
                               kernel_quantizer=quantized_bits(*ap_fixed,alpha=1),
                               bias_quantizer=quantized_bits(*ap_fixed,alpha=1),
                               kernel_initializer='glorot_uniform',
                               # kernel_regularizer=regularizers.l1(0) ### commented out after Chang's finding ...
                              ))
            
            
            if use_BN and i!=len(decoder_config["nodes"])-1:
                self.decoder.add(QBatchNormalization(name=f'BN_decoder{i+1}',
                                              beta_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                              gamma_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                              mean_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                              variance_quantizer=quantized_bits(*ap_fixed,alpha=1))) ### commented out after Chang's finding ...

            if  i!=len(decoder_config["nodes"])-1:
                self.decoder.add( QActivation(name=f'act_decoder{i+1}',
                                         activation=quantized_relu(*ap_fixed),))
        
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x