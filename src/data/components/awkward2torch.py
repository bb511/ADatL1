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

from .normalization import L1DataNormalizer
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataAwkward2Torch:
    workers: int
    nconst: dict
    verbose: bool = False

    def __post_init__(self):
        self.object_feature_map = None

    def load_folder(self, folder_path: Path) -> torch.Tensor:
        """Loads folder of parquet files containing awkward arrays to a numpy array.

        To this end, the data is made uniform, i.e., each object contains the same
        number of features via padding.
        """
        self.cache_filepath = folder_path / "torch_cache.pt"
        self.cache_maskpath = folder_path / "torch_mask.pt"
        self.cache_l1bitpath = folder_path / "torch_l1bit.pt"
        self.object_feature_map_fpath = folder_path.parent / "object_feature_map.json"
        if self._cache_exists():
            if self.object_feature_map is None:
                self._set_obj_feat_map()
            return (
                torch.load(self.cache_filepath),
                torch.load(self.cache_maskpath),
                torch.load(self.cache_l1bitpath),
            )

        self.cached_objects = set()
        procd_folder = self._process_folder(folder_path)

        self._make_feature_map(procd_folder)
        data = np.concatenate([darr for _, _, darr, _ in procd_folder], axis=1)
        data = torch.from_numpy(data)
        self._cache_data(data)

        mask = np.concatenate([marr for _, _, _, marr in procd_folder], axis=1)
        mask = torch.from_numpy(mask)
        self._cache_mask(mask)

        l1bit_path = folder_path / "L1bit.parquet"
        if not l1bit_path.is_file():
            l1bit = torch.ones(data.size(0), dtype=torch.bool, device=data.device)
            log.warn(f"L1bit not found in {folder_path}.")
            return data, mask, l1bit

        l1bit = ak.from_parquet(l1bit_path)
        l1bit = ak.to_numpy(ak.flatten(l1bit["L1bit"]))
        l1bit = torch.from_numpy(l1bit)
        self._cache_l1bit(l1bit)

        return data, mask, l1bit

    def _cache_exists(self) -> bool:
        """Check whether the cache exists."""
        files_exist = (
            self.cache_filepath.is_file()
            and self.cache_maskpath.is_file()
            and self.cache_l1bitpath.is_file()
        )
        if files_exist and self._check_cache_integrity():
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

        padder = self._get_padder(data)
        data, mask, nconst = self._pad(data, nconst, padder)

        nevents = len(data)
        nfeats = len(data.fields)
        feats = sorted(list(data.fields))

        numpy_array = np.empty((nevents, nconst, nfeats), dtype=np.float32)
        numpy_array = self._funnel_to_nparray(data, numpy_array)
        paddy_array = np.empty((nevents, nconst, nfeats), dtype=np.bool)
        paddy_array = self._funnel_to_nparray(mask, paddy_array)

        return obj_name, feats, numpy_array, paddy_array

    def _pad(
        self, data: ak.Array, nconst: int, padder: Union[float, ak.Record]
    ) -> tuple[ak.Array, int]:
        """Pads the jagged arrays such that they are rectangular for each object."""
        if nconst is None:
            nconst = max(int(ak.max(ak.num(data[f]), initial=0)) for f in data.fields)

        data = ak.pad_none(data, nconst, axis=-1, clip=True)
        mask = ak.Array({f: ~ak.is_none(data[f], axis=-1) for f in data.fields})
        data = ak.fill_none(data, padder)
        data = ak.values_astype(data, np.float32)

        return data, mask, nconst

    def _funnel_to_nparray(self, data: ak.Array, numpy_array: np.ndarray) -> np.ndarray:
        """Funnel the data from the awkward array into a numpy array of right dims."""
        for feat_idx, feature in enumerate(sorted(data.fields)):
            feature_data = ak.to_numpy(data[feature], allow_missing=False)
            numpy_array[..., feat_idx] = feature_data

        return numpy_array

    def _get_padder(self, data: ak.Array):
        """Determines the padder for the different types of data.

        The energies and event_info are structurally different than the objects. These
        are always 1 per sample, and hence the padder can be a scalar. Meanwhile, objs
        like egammas can have different numbers of objects per sample, and hence the
        padder needs to be applied per-field.
        """
        if isinstance(data[0], ak.Record):
            padder = 0.0
            return padder

        padder = {feat: 0.0 for feat in data.fields}
        return padder

    def _cache_data(self, data: torch.Tensor):
        """Cache the converted data to disk."""
        log.info(f"Caching torch tensor at {self.cache_filepath}...")
        parent_folder = self.cache_filepath.parent
        if not parent_folder.exists():
            raise FileNotFoundError(f"Cache folder {parent_folder} missing!")

        with open(parent_folder / "cached_objs.pkl", "wb") as file:
            pickle.dump(self.cached_objects, file)

        torch.save(data, self.cache_filepath)

    def _cache_mask(self, mask: torch.Tensor):
        """Cache mask corresponding to data to disk."""
        log.info(f"Caching corresponding padding mask at {self.cache_maskpath}...")
        parent_folder = self.cache_maskpath.parent
        if not parent_folder.exists():
            raise FileNotFoundError(f"Cache folder {parent_folder} missing!")

        torch.save(mask, self.cache_maskpath)

    def _cache_l1bit(self, l1bit: torch.Tensor):
        """Cache the l1bit corresponding to the data."""
        log.info(f"Caching corresponding l1bit at {self.cache_l1bitpath}...")
        parent_folder = self.cache_l1bitpath.parent
        if not parent_folder.exists():
            raise FileNotFoundError(f"Cache folder {parent_folder} missing!")

        torch.save(l1bit, self.cache_l1bitpath)

    def _check_cache_integrity(self) -> bool:
        """Checks whether the cache includes all the desired data objects.

        If the objects in the folder do not match what's been previously cached, as
        stored in the cache_objs.pkl, then rebuild the torch tensor cache.
        """
        cache_parent_folder = self.cache_filepath.parent
        obj_fpaths = self._get_parquet_fpaths(cache_parent_folder)
        existing_obj_names = {obj_path.stem for obj_path in obj_fpaths}

        cache_obj_file = cache_parent_folder / "cached_objs.pkl"
        if cache_obj_file.is_file():
            with open(cache_obj_file, "rb") as file:
                previously_cached_objects = pickle.load(file)
        else:
            previously_cached_objects = set()

        return existing_obj_names == previously_cached_objects

    def _get_parquet_fpaths(self, folder: Path) -> list[Path]:
        """Get the parquet files in a folder while avoiding system files."""
        parquet_file_paths = sorted(
            [
                fpath
                for fpath in folder.glob("*.parquet")
                if fpath.is_file() and not "L1bit" in fpath.name
            ]
        )

        return parquet_file_paths

    def _make_feature_map(self, procd_folder: tuple):
        """Make a map that contains metadata about the objs/feats in the torch tensor.

        The map is a dictionary with the structure:
            {obj_name: {feature_name: [list of flattened column indices], ...},...}

        Save this map one level above the given directory to load, since we expect that
        this directory corresponds to train/val/test/etc and the structure of the data
        should not change between these.
        """
        mapping: dict[str, dict[str, list[int]]] = {}
        offset = 0
        for obj_name, feat_names, data, _ in procd_folder:
            feats_to_idxs, offset = self._2d_feature_map(data, feat_names, offset)
            mapping[obj_name] = feats_to_idxs

        with open(self.object_feature_map_fpath, "w") as f:
            json.dump(mapping, f, indent=4)

        self.object_feature_map = mapping

    def _2d_feature_map(self, data: np.ndarray, feat_names: list[str], obj_offset: int):
        """Compute a feature to idx map for input 2d data.

        Namely, map each feature within an object to its position in a flattened data
        array. The obj_offset applies an offset to indices, if one wants to process
        multiple objects. This is done because most torch tensors are flattened when
        they are passed to a neural network and the metadata info is important for
        some of the neural networks developed here.
        """
        nconst, nfeats = data.shape[-2:]
        feats_to_idxs = {feat_name: [] for feat_name in feat_names}

        # Compute the index of each feature if the array is flat.
        for const_idx in range(nconst):
            for feat_idx, feat_name in enumerate(feat_names):
                idx = obj_offset + const_idx * nfeats + feat_idx
                feats_to_idxs[feat_name].append(idx)

        obj_offset += nconst * nfeats

        return feats_to_idxs, obj_offset

    def _set_obj_feat_map(self):
        """Get index mapping of the processed torch tensor.

        Get the feature to index in torch array map. The torch array is generated with
        this class. This is important for in-place denomarlisation for some of the
        model workflows.
        """
        with open(self.object_feature_map_fpath, "r") as file:
            self.object_feature_map = json.load(file)
