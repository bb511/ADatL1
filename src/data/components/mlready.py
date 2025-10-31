# Splits the data into train, validation, test... And all the auxiliary data.
from collections.abc import Callable, Iterator
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import awkward as ak
import pickle

import pyarrow
import pyarrow.parquet as parquet
import pyarrow.dataset
from omegaconf import OmegaConf
from colorama import Fore, Back, Style

from src.utils import pylogger
from . import plots
from . import normalization

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataMLReady:
    processed_datapath: str
    select_features: dict
    split: dict
    split_aux: int
    cache_root_dir: str = "data/"
    name: str = "default"
    seed: int = 42

    def prepare(self, normalizer: normalization.L1DataNormalizer):
        """Makes train set, validation set, test set, and auxiliary data.

        Imports the processed data. Applies normalization through the normalizer object.
        See normalizer.py for the definition of this object and how to use it.
        Additionally, if an object is missing a feature that another object has, this
        feature is added to the object that is missing it and padded with None values.
        """
        self.normalizer = normalizer

        self.processed_datapath = Path(self.processed_datapath)
        self.mlready_dir = Path(self.cache_root_dir) / 'mlready'
        self.cache_folder = self.mlready_dir / self.name
        self.cache_folder /= normalizer.name

        self.select_feats = OmegaConf.to_container(self.select_features, resolve=True)
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

        train_exist = self._check_data_exists(self.cache_folder / 'train')
        valid_exist = self._check_data_exists(self.cache_folder / 'valid')
        test_exist = self._check_data_exists(self.cache_folder / 'test')
        if train_exist and valid_exist and test_exist:
            return

        for obj_name, file_paths in obj_paths.items():
            if not obj_name in self.select_feats.keys():
                continue
            obj_dataset = pyarrow.dataset.dataset(file_paths, format='parquet')
            obj_data = obj_dataset.to_table()
            permutation = self.rng.permutation(obj_data.num_rows)
            ntrain = int(self.split['train']*obj_data.num_rows)
            nvalid = int(self.split['valid']*obj_data.num_rows) + ntrain

            self._cache_train_data(obj_data, obj_name, permutation[:ntrain])
            self._cache_valid_data(obj_data, obj_name, permutation[ntrain:nvalid])
            self._cache_test_data(obj_data, obj_name, permutation[nvalid:])

        self._cache_norm_params()
        self._plot(self.cache_folder / 'train')
        self._plot(self.cache_folder / 'valid')
        self._plot(self.cache_folder / 'test')

    def _cache_train_data(self, obj_data: pyarrow.Table, obj_name: str, idxs: list):
        """Get the training data split, normalize it, and then cache it."""
        train_folder = self.cache_folder / 'train'
        train_folder.mkdir(parents=True, exist_ok=True)
        cache_file = train_folder / f'{obj_name}.parquet'

        train_events = pyarrow.array(idxs)
        train_data = ak.from_arrow(obj_data.take(train_events))
        train_data = train_data[self.select_feats[obj_name]]
        self.normalizer.fit(train_data, obj_name)
        train_data = self.normalizer.norm(train_data, obj_name)

        train_data = self._add_missing_feats(train_data)
        ak.to_parquet(train_data, cache_file)
        log.info(Fore.GREEN + f"Cached mlready train data at: {cache_file}")

    def _cache_valid_data(self, obj_data: pyarrow.Table, obj_name: str, idxs: list):
        """Get the validation data split, normalize it, and then cache it."""
        valid_folder = self.cache_folder / 'valid'
        valid_folder.mkdir(parents=True, exist_ok=True)
        cache_file = valid_folder / f'{obj_name}.parquet'

        valid_events = pyarrow.array(idxs)
        valid_data = ak.from_arrow(obj_data.take(valid_events))
        valid_data = valid_data[self.select_feats[obj_name]]
        valid_data = self.normalizer.norm(valid_data, obj_name)

        valid_data = self._add_missing_feats(valid_data)
        ak.to_parquet(valid_data, cache_file)
        log.info(Fore.GREEN + f"Cached mlready valid data at: {cache_file}")

    def _cache_test_data(self, obj_data: pyarrow.Table, obj_name: str, idxs: list):
        """Get the test data split, normalize it, and then cache it."""
        test_folder = self.cache_folder / 'test'
        test_folder.mkdir(parents=True, exist_ok=True)
        cache_file = test_folder / f'{obj_name}.parquet'
        ntest = int(self.split['test']*obj_data.num_rows)

        test_events = pyarrow.array(idxs)
        test_data = ak.from_arrow(obj_data.take(test_events))
        test_data = test_data[self.select_feats[obj_name]]
        test_data = self.normalizer.norm(test_data, obj_name)

        test_data = self._add_missing_feats(test_data)
        ak.to_parquet(test_data, cache_file)
        log.info(Fore.GREEN + f"Cached mlready test data at: {cache_file}")

    def _prepare_auxdata(self):
        """Prepare the signal and data categories, used only for validation."""
        log.info(Fore.BLUE + "Preparing auxiliary data...")
        self.aux_dir = self.cache_folder / 'aux'
        self.aux_dir.mkdir(parents=True, exist_ok=True)

        dataset_paths = self.processed_datapath / 'background'
        for dataset_path in dataset_paths.iterdir():
            self._cache_aux(dataset_path)

        dataset_paths = self.processed_datapath / 'signal'
        for dataset_path in dataset_paths.iterdir():
            self._cache_aux(dataset_path)

    def _cache_aux(self, dataset_path: Path):
        """Split the aux datasets into valid, test splits, normalize, and cache."""
        cache_dir = self.aux_dir / dataset_path.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        valid_exist = self._check_data_exists(cache_dir / 'valid')
        test_exist = self._check_data_exists(cache_dir / 'test')
        if valid_exist and test_exist:
            return

        for obj_path in dataset_path.glob('*.parquet'):
            obj_name = obj_path.stem
            if not obj_name in self.select_feats.keys():
                continue

            obj_data = parquet.read_table(obj_path)
            perm = self.rng.permutation(obj_data.num_rows)
            nvalid = int(self.split_aux*obj_data.num_rows)

            self._cache_aux_valid(obj_data, cache_dir, obj_name, perm[:nvalid])
            self._cache_aux_test(obj_data, cache_dir, obj_name, perm[nvalid:])

        self._plot(cache_dir / 'valid')
        self._plot(cache_dir / 'test')
        log.info(Fore.GREEN + f"Cached mlready auxiliary data at: {cache_dir}")

    def _cache_aux_valid(self, obj_data, cache_dir: Path, obj_name: str, idxs: list):
        """Normalize and cache the validation split of an aux dataset."""
        valid_folder = cache_dir / 'valid'
        valid_folder.mkdir(parents=True, exist_ok=True)
        cache_file = valid_folder / f'{obj_name}.parquet'

        valid_events = pyarrow.array(idxs)
        valid_data = ak.from_arrow(obj_data.take(valid_events))
        valid_data = valid_data[self.select_feats[obj_name]]
        valid_data = self.normalizer.norm(valid_data, obj_name)

        valid_data = self._add_missing_feats(valid_data)
        ak.to_parquet(valid_data, cache_file)

    def _cache_aux_test(self, obj_data, cache_dir: Path, obj_name: str, idxs: list):
        """Normalize and cache the test split of an aux dataset."""
        test_folder = cache_dir / 'test'
        test_folder.mkdir(parents=True, exist_ok=True)
        cache_file = test_folder / f'{obj_name}.parquet'

        test_events = pyarrow.array(idxs)
        test_data = ak.from_arrow(obj_data.take(test_events))
        test_data = test_data[self.select_feats[obj_name]]
        test_data = self.normalizer.norm(test_data, obj_name)

        test_data = self._add_missing_feats(test_data)
        ak.to_parquet(test_data, cache_file)

    def _add_missing_feats(self, data: ak.Array):
        """If an array is missing a field present in the unified schema, add it."""
        for feature in self.unified_schema:
            if feature in data.fields:
                continue
            data[feature] = None

        return data

    def _check_data_exists(self, dataset_folder: Path) -> bool:
        """Check if a specific data set was already processed."""
        if dataset_folder.exists():
            log.info(Fore.YELLOW + f"MLready data exists: {dataset_folder}.")
            return True

        return False

    def _get_files_per_object(self, processed_path: Path) -> dict:
        """Get path to parquet files per object in the processed data folder."""
        obj_paths = defaultdict(list)
        for file_path in processed_path.rglob('*.parquet'):
            if len(file_path.relative_to(processed_path).parents) == 2:
                obj_paths[file_path.stem].append(file_path)

        return obj_paths

    def _cache_norm_params(self):
        """Save the normalization parameters to a file."""
        params_file = self.cache_folder / 'norm_params.pkl'
        if self.normalizer.norm_params:
            with open(params_file, 'wb') as file:
                pickle.dump(self.normalizer.norm_params, file)

    def _plot(self, dataset_folder: Path):
        """Read and then plot the processed data."""
        plots_dir = dataset_folder / 'PLOTS'

        for object_file in dataset_folder.glob('*.parquet'):
            data = ak.from_parquet(object_file)
            obj_folder = plots_dir / object_file.stem
            obj_folder.mkdir(parents=True, exist_ok=True)

            for feature in data.fields:
                plots.plot_hist(data[feature], feature, obj_folder)
