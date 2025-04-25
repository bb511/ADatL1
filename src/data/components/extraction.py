from pathlib import Path
from dataclasses import dataclass
from pathlib import Path
import gc

import h5py
import numpy as np
import yaml
from colorama import Fore, Back, Style

from l1trigger_datamaker.h5convert.root2h5 import Root2h5
from src.utils import pylogger
from . import plots

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@dataclass
class L1DataExtractor(Root2h5):
    """Parent class for the extraction of CMS L1 global trigger data."""

    def __init__(
        self,
        constituents: dict,
        objects_features: dict,
        cache: str = "data/extracted",
        name: str = "axol1tl",
    ):
        """Extract data from level 1 global trigger data files saved in h5 format.

        The data is then cached to a corresponding h5 file that only contains the
        desired constituents and the specified features for each constituent.

        :param constituents: Dictionary with a bool list of which constituents to
            include, corresponding to each particle object, e.g.
            muons: [True, True, True, True, False, False, False, False]
        :param objects_features: Dictionary with which objects to extract from the
            h5 file (keys), and the corresponding features to get from the object.
        :param cache: String specifying path to where to store the data.
        :param name: String used to name a specific method of doing the
            extraction of the data, e.g., "basic".
        """
        super().__init__()
        self.objects_features_names = objects_features.copy()
        self.objects_features_idxs = self._replace_feats_name_index(objects_features)
        self.constituents = constituents
        self.cache = Path(cache)
        self.extraction_name = name

    def extract(self, datasets: dict, data_category: str):
        """Extract the data.

        :param datasets: Dictionary containing the name of the datasets and the
            corresponding folder path associated with it.
        :param data_category: String specifying the kind of data that is being extracted
            e.g., 'zerobias', 'background', or 'signal'.
        """
        log.info(f"Extracting datasets: {list(datasets.keys())}.")

        self.cache_folder = self.cache / self.extraction_name / data_category
        for dataset_name, folder_path in datasets.items():
            if self._check_file_exists(dataset_name):
                continue
                
            h5file = self.read_folder(Path(folder_path))
            data = self._extract_objects(h5file)
            h5file.close()
            gc.collect()
            self._cache(data, dataset_name)

        self._cache_metadata()

    def _extract_objects(self, h5file: h5py.File) -> dict:
        """Extract the objects in the h5file that are specified in the config file."""
        if not self._check_subset(h5file.keys(), self.objects_features_idxs.keys()):
            raise ValueError("The specified objects are not all present in the h5file.")

        data = {}
        for obj in self.objects_features_idxs.keys():
            # If object is a particle, e.g. muon, then extract constituents and feats.
            if obj in self.particles.keys():
                data[obj] = self._extract_constituents_feats(h5file, obj)
                continue

            # If object is not a particle, then extract feats directly.
            data[obj] = h5file[obj][:][..., self.objects_features_idxs[obj]]

        return data

    def _extract_constituents_feats(self, h5file: h5py.File, particle: str) -> dict:
        """Select the constituents and corresp. features of a particle object.

        :param h5file: The h5file object that the data is being read from.
        :param particle: String specifying the name of the particle obj to be extracted.
        """
        if not particle in h5file.keys():
            raise KeyError(f"Particle not found in the given data file.")

        if self.objects_features_idxs[particle]:
            # Extract constituents.
            particle_data = h5file[particle][:][:, self.constituents[particle]]
            # Extract corresponding features for the constituents.
            particle_data = particle_data[..., self.objects_features_idxs[particle]]
            return particle_data

        return h5file[particle_type][:][:, self.constituents[particle]]

    def _replace_feats_name_index(self, selected_features: dict) -> dict:
        """Replaces the name of the feature with its index in the data array.

        This is done for a dictionary where for each object the user specifies which
        features to select.
        """
        for object_name in selected_features.keys():
            idxs = [
                # self.all_objects_feats[object_name].index(feat)
                self.objects_features_names[object_name].index(feat)
                for feat in selected_features[object_name]
            ]
            selected_features[object_name] = idxs

        return selected_features

    def _check_subset(self, biglist: list, sublist: list) -> bool:
        """Checks if one list is a sublist of another list."""
        return set(biglist).intersection(sublist) == set(sublist)

    def _cache(self, data: dict, filename: str):
        """Store the extracted data in a h5 file."""
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_folder / (filename + ".h5")

        h5file = h5py.File(cache_file, mode="w")
        for obj_name, obj_data in data.items():
            plots.plot_hist_3d(
                obj_data,
                obj_name,
                self.objects_features_names[obj_name],
                self.cache_folder / filename,
            )
            h5file.create_dataset(obj_name, data=obj_data, compression="gzip")
        h5file.close()

        log.info("Cached extracted data at " + Fore.GREEN + f"{cache_file}.")

    def _cache_metadata(self):
        """Cache the dictionary with the objects and features in the extracted data."""
        metadata_folder = self.cache_folder.parent
        metadata_filepath = metadata_folder / "metadata.yaml"

        if metadata_filepath.exists():
            return

        # Get the names of the features for each object in the right order.
        metadata_dict = {}
        for obj_name in self.objects_features_idxs.keys():
            object_feats = np.array(self.objects_features_names[obj_name])
            object_feats = object_feats[self.objects_features_idxs[obj_name]].tolist()
            metadata_dict[obj_name] = object_feats

        with open(metadata_filepath, "w") as metadata_file:
            yaml.dump(metadata_dict, metadata_file)

        log.info(f"Cached extracted feature metadata at {metadata_filepath}.")

    def _check_file_exists(self, dataset_name: str) -> bool:
        """Check if a specific data file exists."""
        cache_file = self.cache_folder / (dataset_name + ".h5")
        if cache_file.exists():
            log.info(Fore.YELLOW + f"Extracted {cache_file} already exists.")
            return 1

        return 0
