from typing import List
from dataclasses import dataclass
import h5py
import gc
import numpy as np

import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

@dataclass
class L1DataExtractor:
    """Parent class for the extraction of CMS data."""

    def _read_raw(self, filepath: str, **kwargs):
        """Reads the H5 file and returns the raw data."""
        pass

    def _strip(self, data: dict, **kwargs):
        """Strips the data into constituents."""
        pass
    
    def extract(self, filepath: str, **kwargs):
        data = self._read_raw(filepath, **kwargs)
        return self._strip(data, **kwargs)

@dataclass
class L1BackgroundDataExtractor(L1DataExtractor):
    """
    CMSDataExtractor for the extraction of CMS background data.
    
    :param train_sector: Portion of the file that will be used for training.
    :param test_sector: Portion of the file that will be used for evaluation.
    :param object_ranges: Ranges of each particle type in the data.
    :param constituents: Constituents of each particle type in the data.
    """

    train_sector: tuple
    test_sector: tuple
    object_ranges: dict
    constituents: dict
    
    def _read_raw(self, filepath: str):   
        f = h5py.File(filepath,"r")
        logger.info("H5 file opened successfully")
        
        dataset = f["full_data_cyl"]
        dat_met = dataset[self.train_sector[0]:self.train_sector[1],self.object_ranges["met"],:]
        dat_egs = dataset[self.train_sector[0]:self.train_sector[1],self.object_ranges["egs"][0]:self.object_ranges["egs"][1],:]
        dat_muons = dataset[self.train_sector[0]:self.train_sector[1],self.object_ranges["muons"][0]:self.object_ranges["muons"][1],:]
        dat_jets = dataset[self.train_sector[0]:self.train_sector[1],self.object_ranges["jets"][0]:self.object_ranges["jets"][1],:]
        logger.info("Train_Data read successful")

        # Few meta datas ....
        ET = f["ET"][self.train_sector[0]:self.train_sector[1]]
        HT = f["HT"][self.train_sector[0]:self.train_sector[1]]
        PU = f["event_info"][self.train_sector[0]:self.train_sector[1],:]
        L1_bits = f["L1bit"][self.train_sector[0]:self.train_sector[1]]
        logger.info("Train_Meta read successful")
            
        data = {}
        data["Train"]={
            "DATA":{
                "MET": dat_met,
                "EGAMMA": dat_egs,
                "MUON": dat_muons,
                "JET": dat_jets
            },
            "META":{
                "ET": ET,
                "HT": HT,
                "PU": PU,
                "L1bits": L1_bits
            }
        }
        del dat_met, dat_egs, dat_muons, dat_jets, ET, HT, PU, L1_bits
        gc.collect()
        logger.info("Registry and Garbage cleaning successful")
    
        dat_met = dataset[self.test_sector[0]:self.test_sector[1],self.object_ranges["met"],:]
        dat_egs = dataset[self.test_sector[0]:self.test_sector[1],self.object_ranges["egs"][0]:self.object_ranges["egs"][1],:]
        dat_muons = dataset[self.test_sector[0]:self.test_sector[1],self.object_ranges["muons"][0]:self.object_ranges["muons"][1],:]
        dat_jets = dataset[self.test_sector[0]:self.test_sector[1],self.object_ranges["jets"][0]:self.object_ranges["jets"][1],:]
        logger.info("Test_Data read successful")
        
        # Few meta datas ....
        ET = f["ET"][self.test_sector[0]:self.test_sector[1]]
        HT = f["HT"][self.test_sector[0]:self.test_sector[1]]
        PU = f["event_info"][self.test_sector[0]:self.test_sector[1],:]
        L1_bits = f["L1bit"][self.test_sector[0]:self.test_sector[1]]
        logger.info("Test_Meta read successful")
            
        data["Test"] = {
            "DATA":{
                "MET":dat_met,
                "EGAMMA":dat_egs,
                "MUON": dat_muons,
                "JET": dat_jets
            },
            "META":{
                "ET":ET,
                "HT":HT,
                "PU":PU,
                "L1bits":L1_bits
            }
        }
        del dat_met, dat_egs, dat_muons, dat_jets, ET, HT, PU, L1_bits
        gc.collect()
        f.close()
        logger.info("Registry and Garbage cleaning successful")
        return data
        
    def _strip(self, data: dict):
        logger.info("Starting...")
        for case in ["Train", "Test"]:
            for k in data[case]["DATA"].keys():
                if k !="MET":
                    data[case]["DATA"][k] = data[case]["DATA"][k][:, self.constituents[k],:]
                else:
                    data[case]["DATA"][k] = data[case]["DATA"][k][:,None,:]

        logger.info("Successful...")
        gc.collect()
        logger.info("Garbage cleaning successful")
        return data
    
    def extract(self, filepath: str):
        data = super().extract(filepath)

        background_datadict = {}
        for case in data.keys():
            background_datadict[case] = {
                "META": data[case]["META"],
                "DATA": np.concatenate(list(data[case]["DATA"].values()), axis = 1)
            }
        del data
        gc.collect()
        return background_datadict


@dataclass
class L1SignalDataExtractor(L1DataExtractor):
    """
    CMSDataExtractor for the extraction of CMS signal data.
    
    :param filepath: Path to the original H5 file.
    :param train_sector: Portion of the file that will be used for training.
    :param test_sector: Portion of the file that will be used for evaluation.
    """

    object_ranges: dict
    max_num_signals: int
    constituents: dict  

    def _read_raw(self, filepath: str):
        data = {}

        f = h5py.File(filepath,"r")
        signal_names = [k for k in f.keys() if f[k].ndim == 3]
        for signal_name in signal_names:
            dataset = f[signal_name]

            dat_met = dataset[:self.max_num_signals,self.object_ranges["met"],:]
            dat_egs = dataset[:self.max_num_signals,self.object_ranges["egs"][0]:self.object_ranges["egs"][1],:]
            dat_muons = dataset[:self.max_num_signals,self.object_ranges["muons"][0]:self.object_ranges["muons"][1],:]
            dat_jets = dataset[:self.max_num_signals,self.object_ranges["jets"][0]:self.object_ranges["jets"][1],:]

            ET = f[f'{signal_name}_ET'][:self.max_num_signals]  # type:ignore
            HT = f[f'{signal_name}_HT'][:self.max_num_signals]  # type:ignore
            PU = f[f'{signal_name}_nPV'][:self.max_num_signals]  # type:ignore
            L1_bits = f[f'{signal_name}_l1bit'][:self.max_num_signals]  # type:ignore

            data[signal_name] = {
                "DATA":{
                    "MET":dat_met,
                    "EGAMMA":dat_egs,
                    "MUON": dat_muons,
                    "JET": dat_jets},
                "META":{
                    "ET":ET,
                    "HT":HT,
                    "PU":PU,
                    "L1bits":L1_bits}}
            del dat_met, dat_egs, dat_muons, dat_jets, ET, HT, PU, L1_bits
            gc.collect()
        f.close()
        return data
    
    def _strip(self, data: dict):
        logger.info("Starting...")
        for case in data.keys():
            for k in data[case]["DATA"].keys():
                if k !="MET":
                    data[case]["DATA"][k] = data[case]["DATA"][k][:, self.constituents[k],:]
                else:
                    data[case]["DATA"][k] = data[case]["DATA"][k][:,None,:]
        logger.info("Successful...")
        
        gc.collect()
        logger.info("Garbage cleaning xuccessful...")
        return data


    def extract(self, filepath: str):
        data = super().extract(filepath)

        signal_datadict = {}
        for case in data.keys():
            signal_datadict[case] = {
                "META": data[case]["META"],
                "DATA": np.concatenate(list(data[case]["DATA"].values()), axis = 1)
            }
        del data
        gc.collect()   
        return signal_datadict

