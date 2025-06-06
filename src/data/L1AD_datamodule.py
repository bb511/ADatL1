from typing import Any, Optional, Tuple
from pathlib import Path
import gc

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
import mlflow


from src.utils import pylogger
from colorama import Fore, Back, Style
from src.data.components.dataset import L1ADDataset

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
        device: str = "cpu",
        specialized_loader: bool = False,
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
        :param device: String specifying device to load the data on.
        :param specialized_loader: Bool specifying whether to use a dataset class
            written specifically for the L1AD data (True) or use the default torch
            Dataloader (False).
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
        log.info(Back.GREEN + "Extracting Data....")
        self.hparams.data_extractor.extract(self.hparams.zerobias, "zerobias")
        self.hparams.data_extractor.extract(self.hparams.background, "background")
        self.hparams.data_extractor.extract(self.hparams.signal, "signal")

        log.info(Back.GREEN + "Processing Data....")
        self.hparams.data_processor.process(self.hparams.zerobias, "zerobias")
        self.hparams.data_processor.process(self.hparams.background, "background")
        self.hparams.data_processor.process(self.hparams.signal, "signal")

    def setup(self, stage: str = None) -> None:
        """Load data. Set `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `
            "predict"`. Defaults to ``None``.
        """
        self._set_batch_size()

        if "cuda" in self.hparams.device:
            log.info(Fore.YELLOW + f"All data will be loaded on {self.hparams.device}.")
            log.warn(Fore.MAGENTA + f"num workers is set to 0")
            log.warn(Fore.MAGENTA + f"pin_memory is set to False")
            self.hparams.num_workers = 0
            self.hparams.pin_memory = False

        if self.hparams.specialized_loader:
            log.info(Fore.GREEN + "Specialized dataset loader is being used!")
            if self.hparams.num_workers or self.hparams.pin_memory:
                log.warn(Fore.MAGENTA + "num_workers/pin_memory given but not used.")

        if self.data_train is None and self.data_val is None and self.data_test is None:
            main_data = self._load_main_data(self.processed_data_folder)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=main_data,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.hparams.data_normalizer.fit(self.data_train[:])
            self.data_train = self.hparams.data_normalizer.norm(
                data=self.data_train[:], dataset_name="train", logs=self.trainer.loggers
            )
            self.data_val = self.hparams.data_normalizer.norm(
                data=self.data_val[:], dataset_name="val", logs=self.trainer.loggers
            )
            self.data_test = self.hparams.data_normalizer.norm(
                data=self.data_test[:], dataset_name="test", logs=self.trainer.loggers
            )

            self.data_train = torch.from_numpy(self.data_train).to(self.hparams.device)
            self.data_val = torch.from_numpy(self.data_val).to(self.hparams.device)
            self.data_test = torch.from_numpy(self.data_test).to(self.hparams.device)

        if self.hparams.additional_validation:
            self._normalize_additional_data(self.hparams.additional_validation)
        if self.hparams.additional_test:
            self._normalize_additional_data(self.hparams.additional_test)

    def _load_main_data(self, processed_data_folder: Path):
        """Load the main data from given collection of files in the main_data dict."""
        log.info("Using main data that is split into train, val, test:")
        log.info(self.hparams.main_data)

        for data_category in self.hparams.main_data.keys():
            data = [
                np.load(
                    processed_data_folder / data_category / (data_file + ".npy"),
                    mmap_mode="r",
                )
                for data_file in self.hparams.main_data[data_category].keys()
            ]
        data = np.concatenate(data, axis=0)
        return data

    def _normalize_additional_data(self, additional_data: dict):
        """Normalizes and caches additional validation or test data."""
        for data_category in additional_data.keys():
            for dataset in additional_data[data_category].keys():
                file = self.processed_data_folder / data_category / (dataset + ".npy")
                data = np.load(file)
                _ = self.hparams.data_normalizer.norm(
                    data, dataset, self.trainer.loggers
                )

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

    def train_dataloader(self) -> DataLoader[Any]:
        """Creates an optimized dataloader for numpy arrays already in memory."""
        if self.hparams.specialized_loader:
            return L1ADDataset(
                self.data_train, batch_size=self.batch_size_per_device, shuffle=True
            )

        return DataLoader(
            dataset=self.data_train,
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
        if self.hparams.specialized_loader:
            main_val = L1ADDataset(
                self.data_val, batch_size=self.batch_size_per_device, shuffle=False
            )
            dataloaders = self._specialized_loader_dict(
                self.hparams.additional_validation or {}
            )
            dataloaders.update({"main_val": main_val})
            return dataloaders

        main_val = DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            shuffle=False,
        )

        dataloaders = self._dataloader_dict(self.hparams.additional_validation or {})
        dataloaders.update({"main_val": main_val})
        return dataloaders

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.hparams.specialized_loader:
            main_test = L1ADDataset(
                self.data_test, batch_size=self.batch_size_per_device, shuffle=False
            )
            dataloaders = self._specialized_loader_dict(
                self.hparams.additional_test or {}
            )
            dataloaders.update({"main_test": main_test})
            return dataloaders

        main_test = DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

        dataloaders = self._dataloader_dict(self.hparams.additional_test or {})
        dataloaders.update({"main_test": main_test})

        return dataloaders

    def _dataloader_dict(self, additional_data: dict):
        """Creates a dictionary of dataloaders of specified datasets found in dict."""
        norm_datapath = Path(self.hparams.data_normalizer.cache)
        norm_scheme = self.hparams.data_normalizer.norm_scheme

        dataloaders = {}
        for data_category in additional_data.keys():
            for dataset_name in additional_data[data_category].keys():
                file = norm_datapath / norm_scheme / (dataset_name + ".npy")
                data = np.load(file)
                dataloader = DataLoader(
                    dataset=data,
                    batch_size=self.batch_size_per_device,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                )
                dataloaders.update({dataset_name: dataloader})

        return dataloaders

    def _specialized_loader_dict(self, additional_data: dict):
        """Creates a dictionary of specialized datasets for our data."""
        norm_datapath = Path(self.hparams.data_normalizer.cache)
        norm_scheme = self.hparams.data_normalizer.norm_scheme

        dataloaders = {}
        for data_category in additional_data.keys():
            for dataset_name in additional_data[data_category].keys():
                file = norm_datapath / norm_scheme / (dataset_name + ".npy")
                data = np.load(file)
                data = torch.from_numpy(data).to(self.hparams.device)
                data = L1ADDataset(
                    data, batch_size=self.batch_size_per_device, shuffle=False
                )
                dataloaders.update({dataset_name: data})

        return dataloaders
