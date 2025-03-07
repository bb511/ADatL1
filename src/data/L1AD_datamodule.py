from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

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
        processed_data_dir: str = "data/",
        train_val_test_split: (float, float, float) = (0.8, 0.1, 0.1),
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

        self.hparams.processed_data_dir = Path(self.hparams.processed_data_dir)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Get zero bias data and the simulated MC signal data."""
        if not self._check_file_exists(self.hparams.processed_data_dir, "zerobias"):
            log.info(Back.GREEN + "Preparing zerobias data...")
            zerobias_data = self.hparams.data_extractor.extract(self.hparams.zerobias)
            self.hparams.data_processor.process(zerobias_data, "zerobias")

        if not self._check_file_exists(self.hparams.processed_data_dir, "background"):
            log.info(Back.GREEN + "Preparing background data...")
            backgr_data = self.hparams.data_extractor.extract(self.hparams.background)
            self.hparams.data_processor.process(backgr_data, "background")

        if not self._check_file_exists(self.hparams.processed_data_dir, "signal"):
            log.info(Back.GREEN + "Preparing signal data...")
            signal_data = self.hparams.data_extractor.extract(self.hparams.signal)
            self.hparams.data_processor.process(signal_data, "signal")

    def setup(self, stage: str = None) -> None:
        """Load data. Set `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `
            "predict"`. Defaults to ``None``.
        """
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            zerobias_data = np.load(self.hparams.processed_data_dir / "zerobias.npy")
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=zerobias_data,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42)
            )
            del zerobias_data

            self.data_train = self.hparams.data_normalizer.normalize(
                self.data_train[:], "train.npy", fit=True
            )
            self.data_val = self.hparams.data_normalizer.normalize(
                self.data_val[:], "val.npy"
            )
            self.data_test = self.hparams.data_normalizer.normalize(
                self.data_test[:], "test.npy"
            )

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
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
