# Extracts the data level 1 global trigger data from the h5 files and puts it into
# an effective format for further processing.

import h5py
import numpy as np

import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

from L1Trigger_data.h5convert import root2h5


@dataclass
class L1DataExtractor(root2h5.Root2h5):
    """Parent class for the extraction of CMS data."""

    def __init__():
        super().__init__()

    def _select_constituents(self, data: dict, constituent_numbers: dict):
        """Select the number of constituents for each of the objects.

        :param constituent_numbers: Dictionary specifying the number of constituents
            for each of the particle objects, e.g., "muons": 8.
        """
        if not self._check_subset(self.particles.keys(), constituent_numbers.keys()):
            raise KeyError("Wrong particles given in setting the constituent numbers. "
                           f"You can set constituents for {self.particles.keys()}.")

        for particle in self.particles.keys():
            if not particle in list(self.h5file.keys()):
                pass
            data[particle] = self.h5file[particle][:constituent_numbers[particle]]

        return data

    def _select_features(self, data: dict,  selected_features: dict):
        """Select the features to include for each object.

        :param selected_features: Dictionary specifying which features to include for
            each of the objects, e.g., "muons": ["pT", "eta", "phi"].
        """
        for particle in self.particles.keys():
            if not particle in list(self.h5file.keys()):
                pass
            self.particles[particle]

    def _check_subset(biglist: list, sublist: list) -> bool:
        """Checks if one list is a sublist of another list."""
        return set(biglist).intersection(sublist) == set(sublist)

    def extract(self, folder_paths: list[Path], **kwargs):
        """Extract the data from the h5 files and put it into a numpy array.

        :param folder_paths: Array of paths to the data to be extracted.
        """
        data = {}
        for folder in folder_paths:
            self.read_folder(folder)
            data = self._select_constituents(data, constituent_numbers)
            data = self._select_features(data, selected_features)
            self.close_h5()

        return data

# Select objects.
# Select constituents.

@dataclass
class L1ZeroBiasExtractor(L1DataExtractor):
    """
    Data extractor for the zero bias data.
    
    :param train_sector: Portion of the file that will be used for training.
    :param test_sector: Portion of the file that will be used for evaluation.
    :param object_ranges: Ranges of each particle type in the data.
    :param constituents: Constituents of each particle type in the data.
    """

    train_sector: tuple
    test_sector: tuple
    object_ranges: dict
    constituents: dict
    def _strip(self, data: h5py.File):
        logger.info("Starting...")
        for case in ["Train", "Test"]:
            for k in data[case]["DATA"].keys():
                if k != "MET":
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
                if k != "MET":
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

