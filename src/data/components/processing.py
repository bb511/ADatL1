from dataclasses import dataclass
from pathlib import Path
import numpy as np
import gc

from src.utils import pylogger
from colorama import Fore, Back, Style

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataProcessor:
    features: dict
    processed_data_dir: str = "/data"

    def process(self, data: dict, filename: str):
        """Applies processing to the data and stores result in a file."""
        self.processed_data_dir = Path(self.processed_data_dir)
        self.processed_file_path = self.processed_data_dir / filename

        log.info(Back.GREEN + "Processing data...")

        data = self._remove_saturation(data)
        data = self._feature_selection(data)
        data = self._dict2nparray(data)

        self._cache(data)

    def _remove_saturation(self, data: dict):
        """Remove saturation in the transverse momentum and the transverse energy.

        Events with saturated transverse energy are removed.
        Saturated transverse momentum is set to 0 for each object.
        """

        ET_thres = 4095
        # These values do not refer to the physical pT, but to the int values in the
        # h5 files. To get physical pT divide by 2.
        pT_thres = {"muons": 511, "egammas": 511, "jets": 2047, "MET": 4095}

        ET_mask = data["ET"][:, 0, 0] < ET_thres
        log.info(f"Removing events that have transverse energy larger than {ET_thres}.")
        for object_name in data.keys():
            data[object_name] = data[object_name][ET_mask]

        # All the features of an object are set to 0 if its pT is above threshold.
        # This treatment is applied only to the objects in the pT_thres disctionary.
        for obj_name in list(set(data.keys()) & set(pT_thres.keys())):
            log.info(f"Setting pT saturated {obj_name} to 0...")
            if obj_name in ["MET"]:
                mask = data[obj_name][:, 0, 0] > pT_thres[obj_name]
                mask.reshape(data[obj_name].shape[0], -1)
                data[obj_name][mask] = np.zeros(data[obj_name].shape[1])
            else:
                mask = data[obj_name][:, :, 1] > pT_thres[obj_name]
                mask.reshape(data[obj_name].shape[0], data[obj_name].shape[1], -1)
                data[obj_name][mask] = np.zeros(
                    (data[obj_name][mask].shape[0], data[obj_name].shape[2])
                )

        del mask
        gc.collect()

        return data

    def _feature_selection(self, data: dict) -> dict:
        """Selects which features to use for each object in the data dictionary.

        A dictionary given by the user is used to do this selection. Therein, each key
        contains an array of indexes which specify which features should be kept.
        The order of the features is the same as found in the table at
        https://github.com/bb511/l1trigger_datamaker/tree/main/l1trigger_datamaker/h5convert
        """
        max_nb_feats = max([len(feats_list) for feats_list in self.features.values()])
        for obj_name in list(data.keys()):
            if not obj_name in self.features.keys():
                del data[obj_name]
                continue

            data[obj_name] = data[obj_name][..., self.features[obj_name]]
            if len(self.features[obj_name]) < max_nb_feats:
                npad = max_nb_feats - len(self.features[obj_name])
                data[obj_name] = self._pad_last_dim(data[obj_name], npad)

        return data

    def _pad_last_dim(self, nparray: np.ndarray, npad: int) -> np.ndarray:
        """Pad last dimension of numpy array with npad zeros."""
        pad_scheme = [(0, 0)]*len(nparray.shape)
        pad_scheme[-1] = (0, npad)
        nparray = np.pad(nparray, pad_scheme, constant_values=0)

        return nparray

    def _dict2nparray(self, data: dict):
        """Converts the data dictionary to a numpy array.

        Each of the object types in the dictionary should contain the same number of
        features. For example, if a muon object has 5 features and a jet object has
        3 features, this code will fail.
        """
        data = [obj_data for obj_data in data.values()]
        data = np.concatenate(data, axis=1)

        return data

    def _cache(self, data: np.ndarray):
        """Store the processed data in a numpy file."""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.processed_file_path, data)
