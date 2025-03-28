from typing import List, Tuple

import torch
import numpy as np
from torchmetrics import Metric

import h5py

from src.models.quantization.activations import QuantizedBits

class AXOScore(Metric):
    def __init__(
        self,
        # model: ???
        target_rate: List[float],
        bc_rate_khz: float,
        ht_threshold: float,
        precision: Tuple[int, int],
        alpha: float,
        data_path: str,
    ):
        super().__init__()
        self.model = model
        self.threshold = None
    
        self.data_file = h5py.File(self.data_path, "r")
    

        self.input_quantizer = QuantizedBits(precision[0], precision[1], alpha=alpha)
    
        self.get_threshold()
        self.get_axo_score()



        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds, target = self._input_format(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total
    


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

#!!!!! The definition of AXO IMPROVEMENT IS STILL NOT CLEAR !!!!!!!!!!!!1

class axo_threshold_manager():
    def __init__(self,model,config):
        self.config = config
        self.model = model
        
        self.target_rate = self.config["target_rate"]
        self.threshold = None
        self.bc_rate_khz = self.config["bc_khz"]
        
        self.score_dict = {} #Internal variable not to be accessed directly
        self.data_file = None #Internal variable not to be accessed directly
        
        self.data_path = self.config["data_path"]
        self.HT_THRESHOLD = self.config["ht_threshold"]
        
        # Init ...
        self._open_hdf5()
        precision = self.config["precision"]
        self.input_quantizer = quantized_bits(precision[0], precision[1],alpha=self.config["alpha"])
        self.get_threshold()
        self.get_axo_score()
        
    def _open_hdf5(self):
        self.data_file = h5py.File(self.data_path,"r")
    
    def get_threshold(self):
        
        x_test = self.data_file["Background_data"]["Test"]["DATA"][:]
        x_test = np.reshape(x_test,(x_test.shape[0],-1))       
        latent_axo_qk = self.model.predict(self.input_quantizer(x_test),batch_size = 120000)
        y_axo_qk = np.sum(latent_axo_qk**2, axis=1)
        
        threshold = {}
        for target_rate in self.target_rate:
            threshold[str(target_rate)] = np.percentile(y_axo_qk, 100-(target_rate/self.bc_rate_khz)*100)
        self.threshold = threshold
    
    def get_axo_score(self):
        
        HT_THRESHOLD = self.HT_THRESHOLD
        signal_names = self.data_file["Signal_data"].keys()
        score = {}
        score["SIGNAL_NAMES"] = signal_names
        score["SCORE"] = {}
        for thres in self.threshold.keys():
            _l1_rate = []
            _raw_rate = []
            _ht_rate = []
            _pure_rate = []
            _axo_improv_rate = []
            for signal in signal_names:
                signal_data = self.data_file["Signal_data"][signal]["DATA"][:]
                signal_data = np.reshape(signal_data,(signal_data.shape[0],-1))
                signal_ET = self.data_file["Signal_data"][signal]["ET"][:]
                signal_HT = self.data_file["Signal_data"][signal]["HT"][:]
                signal_L1 = self.data_file["Signal_data"][signal]["L1bits"][:]
                signal_PU = self.data_file["Signal_data"][signal]["PU"][:]

                latent_axo_qk = self.model.predict(self.input_quantizer(signal_data),batch_size = signal_data.shape[0],verbose=0)
                y_axo_qk = np.sum(latent_axo_qk**2, axis=1)

                nsamples = y_axo_qk.shape[0]

                axo_triggered = np.where(y_axo_qk > self.threshold[thres])[0].tolist()
                l1_triggered = np.where(signal_L1)[0].tolist()
                ht_triggered = np.where(signal_HT > HT_THRESHOLD)[0].tolist()

                raw_rate = len(axo_triggered)/nsamples
                l1_rate = len(l1_triggered)/nsamples
                ht_rate = len(ht_triggered)/nsamples

                axo_pure = list(set(axo_triggered)-set(l1_triggered))
                axo_pure_rate = len(axo_pure)/nsamples

                axo_improv = list(set(axo_triggered).union(set(l1_triggered)))
                axo_improv_rate = (len(axo_improv) - len(l1_triggered))/nsamples

                _l1_rate.append(l1_rate*100)
                _raw_rate.append(raw_rate*100)
                _pure_rate.append(axo_pure_rate*100)
                _axo_improv_rate.append(axo_improv_rate*100)
                _ht_rate.append(ht_rate*100)
                
            score["SCORE"][thres] = {
                "raw-axo":_raw_rate,
                "pure-axo":_pure_rate,
                "L1_rate":_l1_rate,
                "HT_rate":_ht_rate,
                "AXO Improvement":_axo_improv_rate,
            }

        self.score_dict = score ## Storing it here
        
    def get_raw_dict(self):
        return self.score_dict,self.threshold

    def get_score(self,thres):
                
        signal_names = self.data_file["Signal_data"].keys()
        df = pd.DataFrame()
        df["Signal Name"] = signal_names
        
        df["AXO SCORE"] = self.score_dict["SCORE"][str(thres)]['raw-axo']
        df["AXO pure SCORE"] = self.score_dict["SCORE"][str(thres)]['pure-axo']
        df["L1 SCORE"] = self.score_dict["SCORE"][str(thres)]['L1_rate']
        df["HT SCORE"] = self.score_dict["SCORE"][str(thres)]['HT_rate']
        df["AXO Improvement"] = self.score_dict["SCORE"][str(thres)]['AXO Improvement']
        return df