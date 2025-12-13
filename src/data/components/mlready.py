# Splits the data into train, validation, test... And all the auxiliary data.
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import awkward as ak

import pyarrow
import pyarrow.dataset
from omegaconf import OmegaConf
from colorama import Fore

from src.utils import pylogger
from . import plots
from .normalization import L1DataNormalizer

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataMLReady:
    processed_datapath: str
    split: dict
    split_aux: float
    cache_root_dir: str = "data/"
    name: str = "default"
    seed: int = 42
    verbose: bool = False

    def prepare(
        self, normalizer: L1DataNormalizer, select_feats: dict, flag: str = ''
    ):
        """Makes train set, validation set, test set, and auxiliary data.

        Imports the processed data. Applies normalization through the normalizer object.
        See normalizer.py for the definition of this object and how to use it.
        Additionally, if an object is missing a feature that another object has, this
        feature is added to the object that is missing it and padded with None values.

        The flag is used in case one wants to store the mlready data in a new
        subdirectory inside the already established train/test/valid/aux subdirs.
        This is useful when, for example, one wants to process additional features from
        the data that are not used in the training data.
        """
        self.normalizer = normalizer

        self.processed_datapath = Path(self.processed_datapath)
        self.mlready_dir = Path(self.cache_root_dir) / 'mlready'
        self.cache_folder = self.mlready_dir / self.name
        self.cache_folder /= normalizer.name
        self.flag = flag

        self.select_feats = select_feats
        if OmegaConf.is_config(self.select_feats):
            self.select_feats = OmegaConf.to_container(self.select_feats, resolve=True)

        self.unified_schema = self._unify_schema()
        self.rng = np.random.default_rng(self.seed)

        self._prepare_maindata()
        self._prepare_auxdata()

    def _unify_schema(self):
        """Each object should have the same features for easier casting to numpy."""
        unified_schema = []
        for feat_list in self.select_feats.values():
            unified_schema.append(feat_list)

        return set().union(*unified_schema)

    def _prepare_maindata(self):
        """Generate the train, validation, test split of the data.

        This method assumes that only the zerobias data is used to generate this split.
        The object data, e.g., muons, are concatenated across the zero bias datasets
        available in the given processed data. Then, this is split into a training
        dataset, validation dataset, and test dataset in a deterministic seeded way.
        """
        obj_paths = self._get_files_per_object(self.processed_datapath / 'zerobias')
        obj_names = [obj_name for obj_name, _ in obj_paths.items()]
        if not set(self.select_feats.keys()) <= set(obj_paths.keys()):
            log.warn(
                "Some objects you're trying to select are not in the extracted data\n"
                f"selected objects = {self.select_feats.keys()}\n"
                f"available = {obj_names}"
            )

        log.info(Fore.MAGENTA + "Preparing train, valid, test data...")
        self._train_data_exists = 0
        self._valid_data_exists = 0
        self._test_data_exists = 0
        itrain, ivalid, itest = self._compute_main_split(obj_paths)
        for obj_name, file_paths in obj_paths.items():
            if obj_name not in self.select_feats.keys():
                continue

            obj_dataset = pyarrow.dataset.dataset(file_paths, format='parquet')
            self._cache_train_data(obj_dataset, obj_name, itrain)
            self._cache_valid_data(obj_dataset, obj_name, ivalid)
            self._cache_test_data(obj_dataset, obj_name, itest)

        if self._train_data_exists > 0:
            log.info(Fore.YELLOW + "Found existing train object data! Skipping...")
        if self._valid_data_exists > 0:
            log.info(Fore.YELLOW + "Found existing valid object data! Skipping...")
        if self._test_data_exists > 0:
            log.info(Fore.YELLOW + "Found existing test object data! Skipping...")

        if self._train_data_exists or self._valid_data_exists or self._test_data_exists:
            log.info(Fore.YELLOW + "For details, verbose=True...")

    def _compute_main_split(self, obj_paths: dict):
        """Get the indexes of the training, validation, and test events.

        This is computed on one object paruqet files and then applied to all other
        objects in the data. The assumption is that you have the same number of events
        for each object.
        """
        first_object_name = list(obj_paths)[0]
        first_object_filepaths = obj_paths[first_object_name]

        obj_dataset = pyarrow.dataset.dataset(first_object_filepaths, format='parquet')
        nevents = obj_dataset.count_rows()
        permutation = self.rng.permutation(nevents)
        ntrain = int(self.split['train']*nevents)
        nvalid = int(self.split['valid']*nevents) + ntrain

        itrain = permutation[:ntrain]
        ivalid = permutation[ntrain:nvalid]
        itest = permutation[nvalid:]

        return itrain, ivalid, itest

    def _cache_train_data(self, obj_data, obj_name: str, idxs: list):
        """Get the training data split, normalize it, and then cache it."""
        train_folder = self.cache_folder / 'train' / self.flag
        train_folder.mkdir(parents=True, exist_ok=True)
        cache_file = train_folder / f'{obj_name}.parquet'

        if self._check_data_exists(cache_file):
            self._train_data_exists += 1
            return

        train_events = pyarrow.array(idxs)
        train_data = ak.from_arrow(obj_data.take(train_events))
        train_data = train_data[self.select_feats[obj_name]]
        self.normalizer.fit(train_data, obj_name)
        self._cache_norm_params(obj_name)
        train_data = self.normalizer.norm(train_data, obj_name)

        train_data = self._add_missing_feats(train_data)
        self._plot(train_data, obj_name, train_folder)
        ak.to_parquet(train_data, cache_file)
        log.info(Fore.GREEN + f"Cached mlready train {obj_name} data at: {cache_file}")

    def _cache_valid_data(self, obj_data, obj_name: str, idxs: list):
        """Get the validation data split, normalize it, and then cache it."""
        valid_folder = self.cache_folder / 'valid' / self.flag
        valid_folder.mkdir(parents=True, exist_ok=True)
        cache_file = valid_folder / f'{obj_name}.parquet'
        self._load_norm_params(obj_name)
        if self._check_data_exists(cache_file):
            self._valid_data_exists += 1
            return

        valid_events = pyarrow.array(idxs)
        valid_data = ak.from_arrow(obj_data.take(valid_events))
        valid_data = valid_data[self.select_feats[obj_name]]
        valid_data = self.normalizer.norm(valid_data, obj_name)

        valid_data = self._add_missing_feats(valid_data)
        ak.to_parquet(valid_data, cache_file)
        self._plot(valid_data, obj_name, valid_folder)
        log.info(Fore.GREEN + f"Cached mlready valid {obj_name} data at: {cache_file}")

    def _cache_test_data(self, obj_data, obj_name: str, idxs: list):
        """Get the test data split, normalize it, and then cache it."""
        test_folder = self.cache_folder / 'test' / self.flag
        test_folder.mkdir(parents=True, exist_ok=True)
        cache_file = test_folder / f'{obj_name}.parquet'
        self._load_norm_params(obj_name)
        if self._check_data_exists(cache_file):
            self._test_data_exists += 1
            return

        test_events = pyarrow.array(idxs)
        test_data = ak.from_arrow(obj_data.take(test_events))
        test_data = test_data[self.select_feats[obj_name]]
        test_data = self.normalizer.norm(test_data, obj_name)

        test_data = self._add_missing_feats(test_data)
        ak.to_parquet(test_data, cache_file)
        self._plot(test_data, obj_name, test_folder)
        log.info(Fore.GREEN + f"Cached mlready test {obj_name} data at: {cache_file}")

    def _prepare_auxdata(self):
        """Prepare the signal and data categories, used only for validation."""
        log.info(Fore.MAGENTA + "Preparing auxiliary data...")
        self.aux_dir = self.cache_folder / 'aux'

        self._aux_data_exists = []
        dataset_paths = self.processed_datapath / 'background'
        for dataset_path in dataset_paths.iterdir():
            self._cache_aux(dataset_path)

        dataset_paths = self.processed_datapath / 'signal'
        for dataset_path in dataset_paths.iterdir():
            self._cache_aux(dataset_path)

        if self._aux_data_exists:
            log.info(Fore.YELLOW + "Found existing aux data! Skipping...")
            log.info(Fore.YELLOW + f"Existing datasets: {set(self._aux_data_exists)}")
            log.info(Fore.YELLOW + f"For more detail, set verbose=True.")

    def _cache_aux(self, dataset_path: Path):
        """Split the aux datasets into valid, test splits, normalize, and cache."""
        cache_dir = self.aux_dir / dataset_path.stem

        ivalid, itest = self._compute_aux_split(dataset_path)
        obj_names = []
        for obj_path in sorted(
            p for p in dataset_path.glob("*.parquet")
            if p.is_file() and not p.name.startswith("._")
        ):
            obj_name = obj_path.stem
            obj_names.append(obj_name)
            if obj_name not in self.select_feats.keys():
                continue

            obj_dataset = pyarrow.dataset.dataset(obj_path, format='parquet')
            self._cache_aux_valid(obj_dataset, cache_dir, obj_name, ivalid)
            self._cache_aux_test(obj_dataset, cache_dir, obj_name, itest)

        if not cache_dir.stem in self._aux_data_exists and cache_dir.is_dir():
            log.info(Fore.GREEN + f"Cached aux set {dataset_path.stem}: {cache_dir}")

    def _compute_aux_split(self, dataset_path: Path):
        """Compute the split into validation and test of the aux data."""
        obj_paths = sorted(
            p for p in dataset_path.glob("*.parquet")
            if p.is_file() and not p.name.startswith("._")
        )
        first_object_filepath = obj_paths[0]

        obj_dataset = pyarrow.dataset.dataset(first_object_filepath, format='parquet')
        nevents = obj_dataset.count_rows()
        permutation = self.rng.permutation(nevents)
        nvalid = int(self.split_aux*nevents)

        ivalid = permutation[:nvalid]
        itest = permutation[nvalid:]

        return ivalid, itest

    def _cache_aux_valid(self, obj_data, cache_dir: Path, obj_name: str, idxs: list):
        """Normalize and cache the validation split of an aux dataset."""
        valid_folder = cache_dir / 'valid' / self.flag
        valid_folder.mkdir(parents=True, exist_ok=True)
        cache_file = valid_folder / f'{obj_name}.parquet'
        self._load_norm_params(obj_name)
        if self._check_data_exists(cache_file):
            self._aux_data_exists.append(cache_dir.stem)
            return

        valid_events = pyarrow.array(idxs)
        valid_data = ak.from_arrow(obj_data.take(valid_events))
        valid_data = valid_data[self.select_feats[obj_name]]
        valid_data = self.normalizer.norm(valid_data, obj_name)

        valid_data = self._add_missing_feats(valid_data)
        ak.to_parquet(valid_data, cache_file)
        self._plot(valid_data, obj_name, valid_folder)

    def _cache_aux_test(self, obj_data, cache_dir: Path, obj_name: str, idxs: list):
        """Normalize and cache the test split of an aux dataset."""
        test_folder = cache_dir / 'test' / self.flag
        test_folder.mkdir(parents=True, exist_ok=True)
        cache_file = test_folder / f'{obj_name}.parquet'
        self._load_norm_params(obj_name)
        if self._check_data_exists(cache_file):
            self._aux_data_exists.append(cache_dir.stem)
            return

        test_events = pyarrow.array(idxs)
        test_data = ak.from_arrow(obj_data.take(test_events))
        test_data = test_data[self.select_feats[obj_name]]
        test_data = self.normalizer.norm(test_data, obj_name)

        test_data = self._add_missing_feats(test_data)
        ak.to_parquet(test_data, cache_file)
        self._plot(test_data, obj_name, test_folder)

    def _add_missing_feats(self, data: ak.Array):
        """If an array is missing a field present in the unified schema, add it.

        This field will be filled with None.
        """
        for feature in self.unified_schema:
            if feature in data.fields:
                continue
            data[feature] = None

        return data

    def _check_data_exists(self, data_filepath: Path):
        if data_filepath.is_file():
            if self.verbose:
                log.info(Fore.YELLOW + f"mlready data exists at {data_filepath}.")
                log.info(Fore.YELLOW + f"Double check if it has the right feats...")
            return True

        return False

    def _get_files_per_object(self, processed_path: Path) -> dict:
        """Get path to parquet files per object in the processed data folder."""
        obj_paths = defaultdict(list)
        for file_path in sorted(
            p for p in processed_path.rglob("*.parquet")
            if p.is_file() and not p.name.startswith("._")
        ):
            if len(file_path.relative_to(processed_path).parents) == 2:
                obj_paths[file_path.stem].append(file_path)

        return obj_paths

    def _cache_norm_params(self, obj_name: str):
        """Save the normalization parameters to a file."""
        norm_params_filepath = self.cache_folder / f'{obj_name}_norm_params.pkl'
        self.normalizer.export_norm_params(norm_params_filepath, obj_name)

    def _load_norm_params(self, obj_name: str):
        """Load normalization parameters into normalizer.

        Check if the normalization parameters do not exist for the normalizer and if
        that is the case, load them from existing pkl file. This might happen when
        the execution of this class is interrupted.
        """
        if obj_name in self.normalizer.norm_params.keys():
            return

        norm_params_filepath = self.cache_folder / f'{obj_name}_norm_params.pkl'
        log.info(Fore.YELLOW + f"Loading norm params from {norm_params_filepath}...")
        self.normalizer.import_norm_params(norm_params_filepath, obj_name)

    def _plot(self, obj_data: ak.Array, obj_name: str, dataset_dir: Path):
        """Read and then plot the processed data."""
        plots_dir = dataset_dir / 'PLOTS'
        obj_folder = plots_dir / obj_name
        obj_folder.mkdir(parents=True, exist_ok=True)

        for feature in obj_data.fields:
            plots.plot_hist(obj_data[feature], feature, obj_folder)
