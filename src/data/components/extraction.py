from pathlib import Path
from dataclasses import dataclass
import gc

import h5py
import numpy as np

from l1trigger_datamaker.h5convert.root2h5 import Root2h5
from src.utils import pylogger
from colorama import Fore, Back, Style

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

@dataclass
class L1DataExtractor(Root2h5):
    """Parent class for the extraction of CMS L1 global trigger data."""

    def __init__(self, objects: list[str], constituents: dict):
        """Extract data from level 1 global trigger data files saved in h5 format.


        :param objects: List of the specific objects to extract from the h5 file.
        :param constituents: Dictionary with a bool list of which constituents to
            include, corresponding to each particle object, e.g.
            muons: [True, True, True, True, False, False, False, False]
        """
        super().__init__()
        self.objects = objects
        self.constituents = constituents

    def extract(self, datasets: dict, **kwargs) -> dict:
        """Extract the data from the h5 files and put it into a pandas dataframe.

        :param datasets: Dictionary containing the name of the datasets and the
            corresponding folder path associated with it.
        """
        data_full = {}
        log.info(f"Extracting datasets: {list(datasets.keys())}.")
        for dataset_name, folder_path in datasets.items():
            folder_path = Path(folder_path)
            h5file = self.read_folder(folder_path)
            data = self._extract_objects(h5file)
            h5file.close()
            gc.collect()
            data_full = self._append_data(data_full, data)

        return data_full

    def _extract_objects(self, h5file: h5py.File):
        """Extract the objects in the h5file that are specified in the config file."""
        if not self._check_subset(h5file.keys(), self.objects):
            raise ValueError("The specified objects are not all present in the h5file.")

        data = {}
        for object_name in self.objects:
            if object_name in self.particles.keys():
                data[object_name] = self._extract_constituents(h5file, object_name)
            else:
                data[object_name] = h5file[object_name][:]

        return data

    def _extract_constituents(self, h5file: h5py.File, particle_type: str) -> dict:
        """Select the number of constituents given a particle type of object."""
        if not particle_type in h5file.keys():
            raise KeyError(f"Particle not found in the given data file.")

        return h5file[particle_type][:][:, self.constituents[particle_type]]

    def _check_subset(self, biglist: list, sublist: list) -> bool:
        """Checks if one list is a sublist of another list."""
        return set(biglist).intersection(sublist) == set(sublist)

    def _append_data(self, data_full: dict, data: dict) -> dict:
        """Merge a list of dictionaries into one dictionary. Dicts have same keys."""
        if not data_full.keys():
            data_full.update(data)
            return data_full

        for key, values in data.items():
            data_full[key] = np.append(data_full[key], values, axis=0)

        return data_full
