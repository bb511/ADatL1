import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
from qkeras import quantized_bits

class distribution_plots():
    def __init__(self,model,config):
        self.config = config
        self.model = model
        
        self.threshold = None
        
        self.score_dict = {} #Internal variable not to be accessed directly
        self.data_file = None #Internal variable not to be accessed directly
        
        self.data_path = self.config["data_path"]
        self.HT_THRESHOLD = self.config["ht_threshold"]
        
        self.signal_hist = None
        self.background_hist = None
        
        # Init ...
        self._open_hdf5()
        ap_fixed = self.config["precision"]
        self.input_quantizer = quantized_bits(ap_fixed[0],ap_fixed[1],alpha=self.config["alpha"])
        self.make_dist()
    
    def _open_hdf5(self):
        self.data_file = h5py.File(self.data_path,"r")
    
    def make_dist(self):
        
        
        ## For the background
        
        x_test = self.data_file["Background_data"]["Test"]["DATA"][:]
        x_test = np.reshape(x_test,(x_test.shape[0],-1))       
        latent_axo_qk = self.model.predict(x_test,batch_size = 120000,verbose=0)
        y_axo_qk = np.sum(latent_axo_qk**2, axis=1)
        
        self.background_hist = np.histogram(y_axo_qk, bins=200)
        
        signal_names = self.data_file["Signal_data"].keys()
        self.signal_hist = {}
        
        for signal in signal_names:
            signal_data = self.data_file["Signal_data"][signal]["DATA"][:]
            signal_data = np.reshape(signal_data,(signal_data.shape[0],-1))
            latent_axo_qk = self.model.predict(signal_data,batch_size = signal_data.shape[0],verbose=0)
            y_axo_qk = np.sum(latent_axo_qk**2, axis=1)
            
            self.signal_hist[signal] = np.histogram(y_axo_qk, bins=200)