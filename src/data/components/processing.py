from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
import h5py
import gc
from colorama import Fore, Back, Style

from src.utils import pylogger
from . import plots

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataProcessor:
    extracted_path: str = "data/extracted/"
    extraction_name: str = "default"
    remove_objects_features: dict = None
    cache: str = "data/processed/"
    name: str = "default"

    def process(self, datasets: dict, data_category: str):
        """Applies processing to the data and stores result in a file.

        A metadata file is also saved. This metadata files stores how many constituents
        each 'particle' type object contains, e.g., the data contains 8 muons; it also
        stores the features pertaining to each type of object, e.g.,
        'muons': ['pT', 'eta', 'phi'].

        :dataset: Dictionary with keys corresponding to the name of the processed data
            set and values corresponding to the path to the raw data.
        :param data_category: String specifying the kind of data that is being extracted
            e.g., 'zerobias', 'background', or 'signal'.
        """
        log.info(Fore.GREEN + "Processing data...")
        self.extracted_folder = Path(self.extracted_path) / self.extraction_name
        self.extracted_metadata_file = self.extracted_folder / "metadata.yaml"
        self.extracted_datapath = self.extracted_folder / data_category
        self.cache_folder = Path(self.cache) / self.name / data_category

        self.objects_feats_names = yaml.safe_load(open(self.extracted_metadata_file))
        self.proc_metadata = self._copy_metadata(self.objects_feats_names, {})

        # Process the data sets.
        for dataset_name in datasets.keys():
            if self._check_data_exists(dataset_name):
                continue

            extracted_filename = self.extracted_datapath / (dataset_name + ".h5")
            extracted_h5 = h5py.File(extracted_filename, mode="r")
            data = self._convert_hdf2dict(extracted_h5)
            extracted_h5.close()
            gc.collect()

            log.info(Fore.BLUE + f"Processing {dataset_name}")
            data = self._remove_saturation(data)
            data = self._remove_objects_features(data)
            data = self._dict2nparray(data)
            self._cache(data, dataset_name)

        self._cache_metadata()

    def _remove_saturation(self, data: dict):
        """Remove saturation in the transverse momentum and the transverse energy.

        Events with saturated transverse energy are removed.
        Saturated transverse momentum of objects is set to 0.
        The saturation threshold is always the same and fixed in this method.
        """

        # Saturation values.
        # These values do not refer to the physical energies, but to the int values in
        # the h5 files. To get physical values divide by 2.
        ET_thres = 4095
        pT_thres = {"muons": 511, "egammas": 511, "jets": 2047, "MET": 4095}

        data = self._remove_energy_saturation(data, ET_thres)
        data = self._mask_pt_saturation(data, pT_thres)

        return data

    def _remove_energy_saturation(self, data: dict, ET_thres: int) -> dict:
        """Remove events with the transverse energy larger than ET_thres."""
        log.info(f"Removing events that have transverse energy larger than {ET_thres}.")

        # Get the index of the transverse energy object corresponding to the energy.
        # The transverse energy object also contains the azimuthal angle.
        energy_idx = self.objects_feats_names["ET"].index("Et")
        ET_mask = data["ET"][:, 0, energy_idx] < ET_thres
        for object_name in data.keys():
            data[object_name] = data[object_name][ET_mask]

        return data

    def _mask_pt_saturation(self, data: dict, pT_thres: dict) -> dict:
        """Pad with zeros the objects in pT_thres if their pT value is above thres.

        All the features of an object are set to 0 if its pT is above threshold.
        This treatment is applied only to the objects in the pT_thres dictionary.
        """
        for obj_name in list(set(data.keys()) & set(pT_thres.keys())):
            log.info(f"Setting pT saturated {obj_name} to 0...")
            pT_idx = self._idx_containing_substr(
                self.objects_feats_names[obj_name], "Et"
            )
            if pT_idx == -1:
                raise ValueError(f"Could not find Et feat in object {obj_name}.")

            mask = data[obj_name][:, :, pT_idx] > pT_thres[obj_name]
            mask.reshape(data[obj_name].shape[0], data[obj_name].shape[1], -1)
            data[obj_name][mask] = np.zeros(
                (data[obj_name][mask].shape[0], data[obj_name].shape[2])
            )

        del mask
        gc.collect()

        return data

    def _remove_objects_features(self, data: dict) -> dict:
        """Removes given features in data.

        If all the features of an object are removed, the object itself is removed
        completely from the data dictionary.
        """
        if not self.remove_objects_features:
            return data

        log.info(f"Removing features {self.remove_objects_features}...")
        for obj_name in self.remove_objects_features.keys():
            if len(self.remove_objects_features[obj_name]) == len(
                self.objects_feats_names[obj_name]
            ):
                del data[obj_name]
                continue

            for feat_name in self._remove_objects_features[obj_name]:
                feat_idx = self.objects_feats_names[obj_name].index(feat_name)
                data[obj_name] = np.delete(data[obj_name], feat_idx, axis=-1)

        return data

    def _idx_containing_substr(self, the_list: list, substring: str) -> int:
        """Get index of string in list of strings that contains substring."""
        for i, s in enumerate(the_list):
            if substring in s:
                return i

        return -1

    def _dict2nparray(self, data: dict):
        """Converts the dictionary of numpy arrays to a concatenated numpy array.

        The numpy arrays in this dictionary are expected to be 3 dimensional.
        If the last dimension is not equal, then the arrays are padded with zero to
        the maximum length of array found in the dictionary.
        """
        max_nb_feats = max([len(feats) for feats in self.objects_feats_names.values()])
        for obj_name in data.keys():
            # Save the number of constituents the object has.
            self.proc_metadata[obj_name]["const"] = data[obj_name].shape[1]

            # Pad the objects up to the maximum number of features existing.
            if not data[obj_name].shape[-1] < max_nb_feats:
                continue
            npad = max_nb_feats - data[obj_name].shape[-1]
            data[obj_name] = self._pad_last_dim(data[obj_name], npad)

        data = [obj_data for obj_data in data.values()]
        data = np.concatenate(data, axis=1)

        return data

    def _pad_last_dim(self, nparray: np.ndarray, npad: int) -> np.ndarray:
        """Pad last dimension of numpy array with npad zeros."""
        pad_scheme = [(0, 0)] * len(nparray.shape)
        pad_scheme[-1] = (0, npad)
        nparray = np.pad(nparray, pad_scheme, constant_values=0)

        return nparray

    def _cache(self, data: np.ndarray, dataset_name: str):
        """Store the processed data in a numpy file."""
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_folder / (dataset_name + ".npy")

        self._plot_processed_data(data, dataset_name)
        np.save(cache_file, data)
        log.info("Cached processed data at " + Fore.GREEN + f"{cache_file}.")

    def _cache_metadata(self):
        """Store the metadata pertaining to which objs and feats are in the data."""
        metadata_filepath = self.cache_folder.parent / "metadata.yaml"
        if metadata_filepath.exists():
            return

        with open(metadata_filepath, "w") as metadata_file:
            yaml.dump(self.proc_metadata, metadata_file)

        log.info(f"Cached processed feature metadata at {metadata_filepath}.")

    def _convert_hdf2dict(self, h5file: h5py.File):
        """Converts an h5 file to a python dictionary. This will load it to memory."""
        data = {}
        for object_name in h5file.keys():
            data[object_name] = h5file[object_name][:]

        return data

    def _plot_processed_data(self, data: np.ndarray, dataset_name: str):
        """Plot the processed data into histograms.

        As the processed data is a numpy array, we use previously stored metadata
        information to keep track of which objects are plotted.
        """
        obj_starting_idx = 0
        for obj_name in self.proc_metadata.keys():
            obj_ending_idx = obj_starting_idx + self.proc_metadata[obj_name]["const"]
            plots.plot_hist_3d(
                data[:, obj_starting_idx:obj_ending_idx],
                obj_name,
                self.proc_metadata[obj_name]["feats"],
                self.cache_folder / dataset_name,
            )
            obj_starting_idx = obj_ending_idx

    def _check_data_exists(self, dataset_name: str) -> bool:
        """Check if the data was processed in the same way already."""
        if (self.cache_folder / (dataset_name + ".npy")).exists():
            log.info(Fore.YELLOW + f"{dataset_name} exists in {self.cache_folder}.")
            return 1

        return 0

    def _check_subset(self, biglist: list, sublist: list) -> bool:
        """Checks if one list is a sublist of another list."""
        return set(biglist).intersection(sublist) == set(sublist)

    def _copy_metadata(self, extracted_metadata: dict, proc_metadata: dict):
        """Copy the metadata from the extracted to the processed metadata dict."""
        for obj_name in extracted_metadata.keys():
            removed_feats = []
            if obj_name in self.remove_objects_features.keys():
                removed_feats = self.remove_objects_features[obj_name]
            proc_metadata[obj_name] = {}
            proc_metadata[obj_name]["feats"] = []

            for feat in extracted_metadata[obj_name]:
                if feat in removed_feats:
                    continue
                proc_metadata[obj_name]["feats"].append(feat)

            if len(proc_metadata[obj_name]["feats"]) == 0:
                del proc_metadata[obj_name]

        return proc_metadata
