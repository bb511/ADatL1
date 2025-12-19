# Load and convert mlready parquet files to numpy arrays.
from typing import Union
import os
import json
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import awkward as ak
import pickle
from colorama import Fore, Back, Style

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataAwkward2Torch:
    workers: int
    nconst: dict
    verbose: bool = False

    def load_folder(self, folder_path: Path) -> torch.Tensor:
        """Loads folder of parquet files containing awkward arrays to a numpy array."""
        self.cache_filepath = folder_path / 'torch_cache.pt'
        self.object_feature_map = folder_path.parent / "object_feature_map.json"
        if self._cache_exists():
            return torch.load(self.cache_filepath)

        self.cached_objects = set()
        procd_folder = self._process_folder(folder_path)
        self._build_feature_mapping(procd_folder)
        data = np.concatenate([data_array for _, _, data_array in procd_folder], axis=1)
        data = torch.from_numpy(data)
        self._cache(data)

        return data

    def _cache_exists(self) -> bool:
        """Check whether the cache exists."""
        if self.cache_filepath.is_file() and self._check_cache_integrity():
            if self.verbose:
                log.warn(
                    "Loading data from torch cache. If the features in the parquets "
                    f"in {self.cache_filepath.parent} changed, this is not caught."
                )
            return True
        return False

    def _process_folder(self, folder_path: Path) -> tuple[str, np.ndarray, list[str]]:
        """Process an entire folder of parquet files.

        The given folder path is expected to contain multiple parquet files, where each
        parquet file stores data corresponding to an object, e.g., egammas.

        Returns a tuple containing at each entry:
            - the name of the processed object
            - a list of the features corresponding to that object
            - the processed object data in a numpy array
        """
        workers = min(self.workers, (os.cpu_count() or 4))
        obj_file_paths = self._get_parquet_fpaths(folder_path)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            processed_folder = list(ex.map(self._process_object, obj_file_paths))

        return processed_folder

    def _process_object(self, data_path: Path) -> np.ndarray:
        """Process one parquet file corresponding to an object from the data set."""
        obj_name = data_path.stem
        nconst = self.nconst.get(obj_name)

        self.cached_objects.add(obj_name)
        data = ak.from_parquet(data_path)

        # data = ak.with_field(data, None, 'dummy')
        data, padder = self._wrap_scalars(data)
        data, nconst = self._pad(data, nconst, padder)

        nevents = len(data)
        nfeats = len(data.fields)
        feats = list(data.fields)

        numpy_array = np.empty((nevents, nconst, nfeats), dtype=np.float32)
        numpy_array = self._funnel_to_nparray(data, numpy_array)

        return obj_name, feats, numpy_array

    def _pad(
        self, data: ak.Array, nconst: int, padder: Union[int, ak.Record]
    ) -> tuple[ak.Array, int]:
        """Pads the jagged arrays such that they are rectangular for each object.

        The None values in the awkward arrays are replaced by 0s.
        """
        if nconst is None:
            nconst = int(ak.max(ak.num(data, axis=1)))

        data = ak.pad_none(data, nconst, axis=1, clip=True)
        data = ak.fill_none(data, padder)
        data = ak.values_astype(data, np.float32)

        return data, nconst

    def _funnel_to_nparray(self, data: ak.Array, numpy_array: np.ndarray) -> np.ndarray:
        """Funnel the data from the awkward array into a numpy array of right dims."""
        for feat_idx, feature in enumerate(sorted(data.fields)):
            feature_data = ak.to_numpy(data[feature], allow_missing=False)
            if feature_data.ndim == 3 and feature_data.shape[-2] == 1:
                feature_data = np.squeeze(feature_data, axis=(2,))

            numpy_array[..., feat_idx] = feature_data

        return numpy_array

    def _wrap_scalars(self, data: ak.Array):
        """Gives one extra dimension to scalar data like ET, MET, etc.

        This is to be consistent with the other objects which are (nevents, nobj, feats)
        while the scalars are (nevents, feats). This method makes the scalars into the
        shape (nevents, 1, feats).
        """
        if isinstance(data[0], ak.Record):
            padder = 0
            data = self._make_consistent_dims(data)
            return ak.unflatten(data, 1, axis=0), padder

        padder = {feat: 0 for feat in data.fields}
        return data, padder

    def _make_consistent_dims(self, data: ak.Array):
        """Wrap fields which are purely scalar and are not filled with None.

        This is done because the original awkward arrays are of different structures:
        event_info is filled with scalars
        object data are all jagged arrays
        event level objects such as ET are fixed size arrays but with dim=2
        """
        out = data
        for name in data.fields:
            field = data[name]

            if field.ndim != 1:
                continue
            if ak.all(ak.is_none(field)):
                continue

            wrapped = ak.singletons(field)
            keep = ~ak.is_none(field)
            new_field = ak.mask(wrapped, keep)

            out = ak.with_field(out, new_field, where=name)

        return out

    def _cache(self, data: torch.Tensor):
        """Cache the converted data to disk."""
        log.info(f"Caching torch tensor at {self.cache_filepath}...")
        parent_folder = self.cache_filepath.parent
        if not parent_folder.exists():
            raise FileNotFoundError(f"Cache folder {parent_folder} missing!")

        with open(parent_folder / 'cached_objs.pkl', 'wb') as file:
            pickle.dump(self.cached_objects, file)

        torch.save(data, self.cache_filepath)

    def _check_cache_integrity(self) -> bool:
        """Checks whether the cache includes all the desired data objects.

        If the objects in the folder do not match what's been previously cached, as
        stored in the cache_objs.pkl, then rebuild the torch tensor cache.
        """
        cache_parent_folder = self.cache_filepath.parent
        obj_fpaths = self._get_parquet_fpaths(cache_parent_folder)
        existing_obj_names = {obj_path.stem for obj_path in obj_fpaths}

        cache_obj_file = cache_parent_folder / 'cached_objs.pkl'
        if cache_obj_file.is_file():
            with open(cache_obj_file, 'rb') as file:
                previously_cached_objects = pickle.load(file)
        else:
            previously_cached_objects = set()

        return existing_obj_names == previously_cached_objects

    def _get_parquet_fpaths(self, folder: Path) -> list[Path]:
        """Get the parquet files in a folder while avoiding system files."""
        parquet_file_paths = sorted([
            fpath for fpath in folder.glob("*.parquet")
            if fpath.is_file() and not fpath.name.startswith("._")
        ])

        return parquet_file_paths

    def _build_feature_mapping(self, procd_folder: tuple):
        """Build a map that contains metadata about the objs/feats in the torch tensor.

        The map is a dictionary with the structure:
            {
              obj_name: {
                 feature_name: [list of flattened column indices],
                 ...
              },
              ...
            }
        """
        if self.object_feature_map.is_file():
            return

        mapping: dict[str, dict[str, list[int]]] = {}
        offset = 0
        for obj_name, feat_names, data in processed:
            feats_to_idxs, offset = self._2d_feature_map(data, feat_names, offset)
            mapping[obj_name] = feats_to_idxs

        with open(self.object_feature_map, "w") as f:
            json.dump(mapping, f, indent=4)

    def _2d_feature_map(self, data: np.ndarray, feat_names: list[str], obj_offset: int):
        """Compute a feature to idx map for 2d data.

        Namely, map each feature within an object to its position in a flattened data
        array. The obj_offset applies an offset to indices, if one wants to process
        multiple objects.
        """
        nconst, nfeats = data.shape[-2:]
        feats_to_idxs = {feat_name: [] for feat_name in feat_names}

        # Compute the index of each feature if the array is flat.
        for const_idx in range(nconst):
            for feat_idx, feat_name in enumerate(feat_names):
                idx = offset + const_idx * nfeats + feat_idx
                feats_to_idxs[feat_name].append(idx)

        obj_offset += nconst * nfeats

        return feats_to_idxs, obj_offset
