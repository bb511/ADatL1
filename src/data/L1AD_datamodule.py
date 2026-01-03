from typing import Any, Optional, Union
from collections import defaultdict
from pathlib import Path
import gc
import warnings

import torch
import numpy as np
import awkward as ak

from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader

from src.utils import pylogger
from colorama import Fore, Back, Style
from src.data.components.dataset import L1ADDataset
from src.data.components.normalization import L1DataNormalizer

log = pylogger.RankedLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Your `IterableDataset` has `__len__` defined\..*",
    category=UserWarning,
)


class L1ADDataModule(LightningDataModule):
    def __init__(
        self,
        zerobias: dict,
        signal: dict,
        background: dict,
        data_extractor: "L1DataExtractor",
        data_processor: "L1DataProcessor",
        data_normalizer: "L1DataNormalizer",
        data_mlready: "L1DataMLReady",
        data_awkward2torch: "L1DataAwkward2Torch",
        train_features: dict,
        l1_scales: dict,
        batch_size: int = 16384,
        val_batches: int = -1,
        seed: int = 42,
    ) -> None:
        """Prepare the L1 data for using it to train and validate ML models.

        The files processed by this class originate from samples produced with
        https://gitlab.cern.ch/cms-l1-ad/l1tntuple-maker/-/tree/new_h5generation

        :param zerobias: Dictionary of paths to the zerobias data.
        :param signals: Dictionary of paths to simulation data of possible anomalies.
        :param background: Dictionary of paths to simulation of data that is guaranteed
            to not contain any anomalies.
        :param data_extractor: Class that extracts the data from the given h5 files.
        :param data_processor: Class that processes the extracted data.
        :param data_normalizer: Class that normalizes the processed data.
        :param data_mlready: Class formats the data to be ready for ML pipeline.
        :param data_awkward2np: Class that converts the data from jagged awkward arrays
            to fixed size numpy arrays to give to the torch dataloader.
        :param train_features: Dictionary where keys are strings of the objects that
            point to list of features to be used during
        :param l1_scales: Dictionary of the scales that the l1 trigger applies to
            all features that could be in the data set.
        :param batch_size: Integer specifying the batch size of the data.
        :param val_batches: Integer specifying how many batches to split each
            validation dataset into. Defaults to -1, which means to use the same batch
            size as the training data.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train, self.data_val, self.data_test = None, None, None
        self.labels_train, self.labels_val, self.labels_test = None, None, None
        self.aux_val, self.aux_val_labels = None, None
        self.aux_test, self.aux_test_labels = None, None

        self.l1_scales = l1_scales
        self.normalizer = None
        self.shuffler = torch.Generator()
        self.shuffler.manual_seed(seed)

    def prepare_data(self) -> None:
        """Get zero bias data and the simulated MC signal data."""
        log.info(Back.GREEN + "Extracting Data...")
        self.hparams.data_extractor.extract(self.hparams.zerobias, "zerobias")
        self.hparams.data_extractor.extract(self.hparams.background, "background")
        self.hparams.data_extractor.extract(self.hparams.signal, "signal")

        log.info(Back.GREEN + "Processing Data...")
        self.hparams.data_processor.process("zerobias")
        self.hparams.data_processor.process("background")
        self.hparams.data_processor.process("signal")

        log.info(Back.GREEN + "Splitting data into train, val, test and normalizing...")
        self.normalizer = self.hparams.data_normalizer
        self.hparams.data_mlready.prepare(self.normalizer, self.hparams.train_features)
        self.main_cache_folder = self.hparams.data_mlready.cache_folder

    def setup(self, stage: str = None) -> None:
        """Load data. Set `self.data_train`, `self.data_val`, `self.data_test`.

        Label the zerobias data with 0.
        Label the signal simulation with labels > 0.
        Label the background simulation with labels < 0.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `
            "predict"`. Defaults to ``None``.
        """
        self._set_batch_size()
        log.info(Back.GREEN + "Loading data in memory...")

        self.loader = self.hparams.data_awkward2torch
        data_dir = self.main_cache_folder

        if stage == "fit":
            self._load_train_data(data_dir)
            self._load_valid_data(data_dir)
        if stage == "validate":
            self._load_valid_data(data_dir)
        if stage == "test":
            self._load_test_data(data_dir)

        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented yet.")

        self._data_summary(stage)

    def _load_train_data(self, data_dir: Path):
        """Set up the training data, if it is not already loaded."""
        if self.data_train is None:
            label = 0
            self.data_train, self.mask_train = self.loader.load_folder(data_dir / "train")
            ntrain = self.data_train.size(dim=0)
            self.labels_train = torch.from_numpy(np.full(ntrain, label))

    def _load_valid_data(self, data_dir: Path):
        """Set up the validation data, if it is not already loaded."""
        if self.data_val is None:
            label = 0
            self.data_val, self.mask_val = self.loader.load_folder(data_dir / "valid")
            nvalid = self.data_val.size(dim=0)
            self.labels_val = torch.from_numpy(np.full(nvalid, label))

        if self.aux_val is None:
            self.aux_val, self.aux_val_mask, self.aux_val_labels =\
                self._load_aux_data("valid", data_dir)

    def _load_test_data(self, data_dir: Path):
        """Set up the test data, if it is not already loaded."""
        if self.data_test is None:
            label = 0
            self.data_test, self.mask_test = self.loader.load_folder(data_dir / "test")
            ntest = self.data_test.size(dim=0)
            self.labels_test = torch.from_numpy(np.full(ntest, label))

        if self.aux_test is None:
            self.aux_test, self.aux_test_mask, self.aux_test_labels =\
                self._load_aux_data("test", data_dir)

    def _load_aux_data(self, stage: str, data_dir: Path):
        """Load the auxiliary data, either for the test or validation stage."""
        aux_folder = data_dir / "aux"
        label_signal = 0
        label_background = 0

        data, mask, labels = {}, {}, {}
        for dataset_path in sorted(
            p
            for p in aux_folder.iterdir()
            if p.is_dir() and not p.name.startswith("._")
        ):
            name = dataset_path.stem
            if "SingleNeutrino" in name:
                label_background += -1
                label = label_background
            else:
                label_signal += 1
                label = label_signal
            data_tensor, mask_tensor = self.loader.load_folder(dataset_path / stage)
            label_tensor = torch.from_numpy(np.full(data_tensor.size(dim=0), label))
            data.update({name: data_tensor})
            mask.update({name: mask_tensor})
            labels.update({name: label_tensor})

        return data, mask, labels

    def _data_summary(self, stage: str):
        """Make a neat little summary of what data is being used."""
        log.info(Fore.MAGENTA + "-" * 5 + " Data Summary " + "-" * 5)

        if stage == "fit":
            log.info(Fore.GREEN + f"Training data:")
            log.info(f"Zero bias: {self.data_train.shape}")

            log.info(Fore.GREEN + f"Validation data:")
            log.info(f"Zero bias: {self.data_val.shape}")
            for data_name, data in self.aux_val.items():
                log.info(f"{data_name}: {data.shape}")

        if stage == "validate":
            log.info(Fore.GREEN + f"Validation data:")
            log.info(f"Zero bias: {self.data_val.shape}")
            for data_name, data in self.aux_val.items():
                log.info(f"{data_name}: {data.shape}")

        if stage == "test":
            log.info(Fore.GREEN + f"Test data:")
            log.info(f"Zero bias: {self.data_test.shape}")
            for data_name, data in self.aux_test.items():
                log.info(f"{data_name}: {data.shape}")

    def train_dataloader(self) -> Dataset[Any]:
        """Create and return the training dataloader.

        This dataloader is based on a custom dataset class from components/dataset.py,
        which basically makes the loading of numpy arrays that are already in memory
        a bit faster.
        """
        dataset = L1ADDataset(
            self.data_train,
            self.mask_train,
            self.labels_train,
            batch_size=self.batch_size_per_device,
            shuffler=self.shuffler,
        )

        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

    def val_dataloader(self) -> Dataset[Any]:
        """Create and return the validation dataloader.

        This is a dictionary containing all the loaded datasets, not just the main
        data validation split. Hence, we return a dictionary of dataloaders.
        """
        data_val = {}
        batch_size = self._get_validation_batchsize(self.data_val)

        # Make sure main_val is always first in the dict !!!
        # The rate callback expects this.
        main_val = L1ADDataset(
            self.data_val, self.mask_val, self.labels_val, batch_size=batch_size
        )
        main_val = DataLoader(
            main_val,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )
        data_val.update({"main_val": main_val})
        if not self.aux_val:
            return data_val

        for dataset_name in self.aux_val.keys():
            batch_size = self._get_validation_batchsize(self.aux_val[dataset_name])
            dataset = L1ADDataset(
                self.aux_val[dataset_name],
                self.aux_val_mask[dataset_name],
                self.aux_val_labels[dataset_name],
                batch_size=batch_size,
            )
            data_val[dataset_name] = DataLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=0,
                persistent_workers=False,
            )

        return data_val

    def test_dataloader(self) -> Dataset[Any]:
        """Create and return the test dataloader.

        This is a dictionary containing all the loaded datasets, not just the main
        data validation split. Hence, we return a dictionary of dataloaders.
        """
        # Shuffling is by default false for the custom dataset.
        batch_size = self._get_validation_batchsize(self.data_test)
        main_test = L1ADDataset(
            self.data_test, self.mask_test, self.labels_test, batch_size=batch_size
        )
        main_test = DataLoader(
            main_test,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )
        data_test = {}
        data_test.update({"main_test": main_test})
        if not self.aux_test:
            return data_test

        for dataset_name in self.aux_test.keys():
            batch_size = self._get_validation_batchsize(self.aux_test[dataset_name])
            dataset = L1ADDataset(
                self.aux_test[dataset_name],
                self.aux_test_mask[dataset_name],
                self.aux_test_labels[dataset_name],
                batch_size=batch_size,
            )
            data_test[dataset_name] = DataLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=0,
                persistent_workers=False,
            )

        return data_test

    def teardown(self, stage: str | None = None) -> None:
        # Drop references to large tensors/dicts so they become collectible
        gc.collect()

        # If CUDA is ever used in other experiments, this helps allocator reuse
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer custom dataset to gpu faster.

        If the transfer fails, which can happen when a lot of jobs are running at
        the same time and the gpu is out of memory, then try to transfer the data
        again after 1 second of delay. Attempt to do this 3 times before throwing
        an error and stopping the process.
        """
        data, mask, labels = batch
        data = data.contiguous()
        mask = mask.contiguous()
        labels = labels.contiguous()
        if "cuda" in str(device):
            data = data.pin_memory()
            data = data.to(device, non_blocking=True)
            mask = mask.pin_memory()
            mask = mask.to(device, non_blocking=True)
            labels = labels.pin_memory()
            labels = labels.to(device, non_blocking=True)
            return data, mask, labels

        return data.to(device), mask.to(device), labels.to(device)

    def _get_validation_batchsize(self, data: np.ndarray):
        if self.hparams.val_batches == -1:
            return self.batch_size_per_device

        return int(data.shape[0] / self.hparams.val_batches)

    def _set_batch_size(self):
        """Set the batch size per device if multiple devices are available."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

    def get_extra(self, normalizer: L1DataNormalizer, extra_feats: dict, flag: str):
        """Hook for callbacks to get additional data.

        The data provided through this hook should not be already included in the
        training data. Otherwise, no point in calling this hook.

        :param normalizer: Normalizer object for the additional data.
        :param extra_feats: Dictionary containing the object and the features to be
            extracted from that object.
        :param flag: String specifying subdirectory to put the extra feature parquet
            files in so they don't get mixed up at training time.
        """
        log.info(Back.GREEN + f"Extracting additional features: {extra_feats}...")
        self.hparams.data_mlready.prepare(normalizer, extra_feats, flag)
        data_dir = self.hparams.data_mlready.cache_folder

        log.info(Back.GREEN + "Loading extra data in memory...")
        val_data = {}
        test_data = {}

        data, mask = self.loader.load_folder(data_dir / "train" / flag)
        train_data = L1ADDataset(
            data,
            mask,
            self.labels_train,
            batch_size=self.batch_size_per_device,
        )
        data, mask = self.loader.load_folder(data_dir / "valid" / flag)
        bs = self._get_validation_batchsize(data)
        val_data.update({"main_val": L1ADDataset(data, mask, self.labels_val, bs)})

        data, mask = self.loader.load_folder(data_dir / "test" / flag)
        bs = self._get_validation_batchsize(data)
        test_data.update({"main_test": L1ADDataset(data, mask, self.labels_test, bs)})

        aux_folder = data_dir / "aux"
        for dataset_path in sorted(
            p
            for p in aux_folder.iterdir()
            if p.is_dir() and not p.name.startswith("._")
        ):
            name = dataset_path.stem
            val, val_mask = self.loader.load_folder(dataset_path / "valid" / flag)
            bs = self._get_validation_batchsize(val)
            val = L1ADDataset(val, val_mask, self.aux_val_labels[name], bs)

            test, test_mask = self.loader.load_folder(dataset_path / "test" / flag)
            bs = self._get_validation_batchsize(test)
            test = L1ADDataset(test, test_mask, self.aux_val_labels[name], bs)

            val_data.update({name: val})
            test_data.update({name: test})

        return train_data, val_data, test_data
