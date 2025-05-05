from typing import Any, Optional, Tuple
from pathlib import Path
import gc

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset

from src.utils import pylogger
from colorama import Fore, Back, Style

log = pylogger.RankedLogger(__name__)


class L1ADDataModule(LightningDataModule):
    def __init__(
        self,
        zerobias: dict,
        signal: dict,
        background: dict,
        data_extractor: "L1DataExtractor",
        data_processor: "L1DataProcessor",
        data_normalizer: "L1DataNormalizer",
        main_data: dict,
        additional_validation: dict,
        additional_test: dict,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `L1ADDataModule`.

        :param zerobias: The paths to the zerobias datasets.
        :param signals: The paths to a selection of possible anomalies.
        :param background: The paths to our best simulation of the zerobias data.
        :param data_extractor: The data extractor class.
        :param data_processor: The data processing class.
        :param processed_data_dir: Path to where the processed data is saved.
            Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split in ratios.
            Defaults to `(0.8, 0.1, 0.1)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.processed_data_folder = Path(self.hparams.data_processor.cache)
        self.processed_data_folder /= self.hparams.data_processor.name

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Get zero bias data and the simulated MC signal data."""
        log.info(Back.GREEN + "Preparing zerobias data...")
        self.hparams.data_extractor.extract(self.hparams.zerobias, "zerobias")
        self.hparams.data_processor.process(self.hparams.zerobias, "zerobias")

        log.info(Back.GREEN + "Preparing background data...")
        self.hparams.data_extractor.extract(self.hparams.background, "background")
        self.hparams.data_processor.process(self.hparams.background, "background")

        log.info(Back.GREEN + "Preparing signal data...")
        self.hparams.data_extractor.extract(self.hparams.signal, "signal")
        self.hparams.data_processor.process(self.hparams.signal, "signal")

    def setup(self, stage: str = None) -> None:
        """Load data. Set `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `
            "predict"`. Defaults to ``None``.
        """
        self._set_batch_size()

        if self.data_train is None and self.data_val is None and self.data_test is None:
            main_data = self._load_main_data(self.processed_data_folder)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=main_data,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.hparams.data_normalizer.fit(self.data_train[:])
            self.data_train = self.hparams.data_normalizer.norm(
                self.data_train[:], "train"
            )
            self.data_val = self.hparams.data_normalizer.norm(self.data_val[:], "val")
            self.data_test = self.hparams.data_normalizer.norm(
                self.data_test[:], "test"
            )

        if self.hparams.additional_validation:
            self._normalize_additional_data(self.hparams.additional_validation)
        if self.hparams.additional_test:
            self._normalize_additional_data(self.hparams.additional_test)

    def _normalize_additional_data(self, additional_data: dict):
        """Normalizes and caches additional validation or test data."""
        for data_category in additional_data.keys():
            for dataset in additional_data[data_category].keys():
                file = self.processed_data_folder / data_category / (dataset + ".npy")
                data = np.load(file)
                _ = self.hparams.data_normalizer.norm(data, dataset)

    def _load_main_data(self, processed_data_folder: Path):
        """Load the main data from given collection of files in the main_data dict."""
        log.info("Using main data that is split into train, val, test:")
        log.info(self.hparams.main_data)

        for data_category in self.hparams.main_data.keys():
            data = [
                np.load(processed_data_folder / data_category / (data_file + ".npy"), mmap_mode='r')
                for data_file in self.hparams.main_data[data_category].keys()
            ]
        data = np.concatenate(data, axis=0)
        return data

    def _set_batch_size(self):
        """Set the batch size per device if multiple devices are available."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

    def _remove_nans(self, data: np.ndarray) -> np.ndarray:
        """Check for NaN values in the data and remove them."""
        nan_mask = np.isnan(data).any(axis=(1, 2)) 

        num_nan_rows = np.sum(nan_mask)
        if num_nan_rows > 0:
            print(f"Removing {num_nan_rows} rows with NaN values out of {len(data)} total rows")

        return data[~nan_mask]

    def _check_file_exists(self, path: Path, filename: str) -> bool:
        """Checks if prepared data already exists at given path."""
        filepath = path / (filename + ".npy")
        if filepath.exists():
            log.info(f"Prepared {filename} data exists at {path}. Skipping prep...")
            return True

        return False

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self._remove_nans(self.data_train),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        main_val = DataLoader(
            dataset=self._remove_nans(self.data_val),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            shuffle=False,
        )

        dataloaders = self._dataloader_dict(self.hparams.additional_validation)
        dataloaders.update({"main": main_val})
        return dataloaders

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        main_test = DataLoader(
            dataset=self._remove_nans(self.data_test),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

        dataloaders = self._dataloader_dict(self.hparams.additional_test)
        dataloaders.update({"main": main_test})

        return dataloaders

    def _dataloader_dict(self, additional_data: dict):
        """Creates a dictionary of dataloaders of specified datasets found in dict."""
        norm_datapath = Path(self.hparams.data_normalizer.cache)

        dataloaders = {}
        for data_category in additional_data.keys():
            for dataset in additional_data[data_category].keys():
                file = Path(norm_datapath) / (self.hparams.data_normalizer.norm_scheme + "_" + dataset + ".npy")
                
                dataloader = DataLoader(
                    dataset=self._remove_nans(np.load(file)),
                    batch_size=self.batch_size_per_device,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                )
                dataloaders.update({dataset: dataloader})

        return dataloaders
