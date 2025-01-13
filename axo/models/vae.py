import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

import qkeras
from qkeras import QBatchNormalization
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

import tensorflow as tf
K = tf.keras.backend
Layer = tf.keras.layers.Layer
keras = tf.keras
 
############################################################################################
#Encoder
############################################################################################
## This is the asymetric VAE that is currently used for the V4 model

def get_encoder(config):
    encoder_config = config["encoder_config"]
    latent_dim = config["latent_dim"]
    features = config["features"]
    
    ap_fixed_kernel = config["ap_fixed_kernel"]
    ap_fixed_bias = config["ap_fixed_bias"]
    ap_fixed_activation = config["ap_fixed_activation"]
    ap_fixed_data = config["ap_fixed_data"]

    
    
    encoder_input = tf.keras.layers.Input(shape=(features,))
    
    input_quantiser = QActivation(activation=quantized_bits(*ap_fixed_data))
    
    x = input_quantiser(encoder_input)
    
    for i,node in enumerate(encoder_config["nodes"]):
            x = QDense(node,
                       name=f'hd_encoder{i+1}',
                       kernel_quantizer=quantized_bits(*ap_fixed_kernel,alpha=1),
                       bias_quantizer=quantized_bits(*ap_fixed_bias,alpha=1),
                       kernel_initializer="glorot_uniform",
                       bias_initializer="zero",
                       activation = quantized_relu(*ap_fixed_activation)
                       
                       # kernel_regularizer=regularizers.l1(0) ### commented out after Chang's finding ...
                      )(x)
    
    # Building the mean and the log var layers
    z_mean = QDense(latent_dim,
                           name=f'latent_mean',
                           kernel_quantizer=quantized_bits(*ap_fixed_kernel,alpha=1),
                           bias_quantizer=quantized_bits(*ap_fixed_bias,alpha=1),
                           activation=quantized_bits(*ap_fixed_activation,alpha=1)
                           # kernel_regularizer=regularizers.l1(0) ### commented out after Chang's finding ...
                          )(x)

    z_log_var = QDense(latent_dim,
                           name=f'latent_log_var',
                           kernel_quantizer=quantized_bits(*ap_fixed_kernel,alpha=1),
                           bias_quantizer=quantized_bits(*ap_fixed_bias,alpha=1),
                           activation=quantized_bits(*ap_fixed_activation,alpha=1)
                           # kernel_regularizer=regularizers.l1(0) ### commented out after Chang's finding ...
                          )(x)

    z = Sampling()([z_mean,z_log_var])

    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

    return encoder

############################################################################################
# Decoder
############################################################################################


def get_decoder(config):
    
    decoder_config = config["decoder_config"]
    latent_dim = config["latent_dim"]

    decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
    
    for i,node in enumerate(decoder_config["nodes"]):
        if i == 0:
            x = Dense(node,name=f'hd_decoder{i+1}')(decoder_input)
        
        else:
            if i == len(decoder_config["nodes"])-1: ## This is done to prevent blowup
                x = Dense(node,
                          name=f'hd_decoder{i+1}',
                          kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
                         )(x)
            else:
                x = Dense(node,
                          name=f'hd_decoder{i+1}'
                         )(x)


        if i!=len(decoder_config["nodes"])-1:
            x = BatchNormalization(name=f'BN_decoder{i+1}')(x)
            x = tf.keras.layers.ReLU()(x)
    
    decoder = keras.Model(decoder_input,x, name="decoder")

    return decoder


############################################################################################
# SAMPLING LAYER
############################################################################################

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_mean = K.cast(z_mean, dtype='float32')
        z_log_var = K.cast(z_log_var, dtype='float32')
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        epsilon = K.cast(epsilon, dtype='float32')
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
############################################################################################
#Variational Auto Encoder
############################################################################################

class VariationalAutoEncoder(Model):
    def __init__(self, config, reco_loss, kld_loss):
        super().__init__()
        encoder_config = config["encoder_config"]
        decoder_config = config["decoder_config"]
        latent_dim = config["latent_dim"]
        features = config["features"]
        
        # Peparing the encoder and the decoder >>>>
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        #### Loss bussiness .....
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        
        self.reco_scale = self.alpha * (1 - self.beta)
        self.kl_scale = self.beta
        
        self.reco_loss = reco_loss
        self.kl_loss = kld_loss
        
        #### Metrics .....
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reco_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.total_val_loss_tracker = keras.metrics.Mean(name="total_val_loss")
        self.reconstruction_val_loss_tracker = keras.metrics.Mean(name="val_reco_loss")
        self.kl_val_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
        ###.........
        
        ##### Taken from Chang
        log_var_k, log_var_b = self.encoder.get_layer('latent_log_var').get_weights()
        self.encoder.get_layer('latent_log_var').set_weights([log_var_k*0, log_var_b*0])
        #####
       

    def train_step(self, data):
        data_in, target = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data_in, training=True)
            reconstruction = self.decoder(z, training=True)

            reconstruction_loss = self.reco_scale * self.reco_loss(target, reconstruction)  # one value
            kl_loss = self.kl_scale * self.kl_loss(z_mean, z_log_var)  # type: ignore
            # kl_loss = self.kl_scale * kl_div_multivariate_normal(z_mean)
            total_loss = reconstruction_loss + kl_loss
            total_loss += K.sum(self.encoder.losses) + K.sum(self.decoder.losses)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reco_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        # validation
        data_in, target = data
        z_mean, z_log_var, z = self.encoder(data_in)
        reconstruction = self.decoder(z)

        reconstruction_loss = self.reco_scale * self.reco_loss(target, reconstruction)
        kl_loss = self.kl_scale * self.kl_loss(z_mean, z_log_var)  # type: ignore
        # kl_loss = self.kl_scale * kl_div_multivariate_normal(z_mean)
        total_loss = reconstruction_loss + kl_loss
        self.total_val_loss_tracker.update_state(total_loss)
        self.reconstruction_val_loss_tracker.update_state(reconstruction_loss)
        self.kl_val_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_val_loss_tracker.result(),
            "reco_loss": self.reconstruction_val_loss_tracker.result(),
            "kl_loss": self.kl_val_loss_tracker.result()
        }