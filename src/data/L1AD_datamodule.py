from typing import Any, Optional, Union
from pathlib import Path

# from retry import retry

import torch
import numpy as np
import awkward as ak

import mlflow
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, Dataset, random_split

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
        data_mlready: "L1DataMLReady",
        data_awkward2torch: "L1DataAwkward2Torch",
        batch_size: int = 16384,
        val_batches: int = -1,
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
        :param batch_size: Integer specifying the batch size of the data.
        :param val_batches: Integer specifying how many batches to split each
            validation dataset into. Defaults to -1, which means to use the same batch
            size as the training data.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train, self.data_val, self.data_test = None, None, None
        self.labels_train, self.labels_val, self.labels_test = None, None, None
        self.aux_val, self.aux_val_labels = {}, {}
        self.aux_test, self.aux_test_labels = {}, {}

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
        self.hparams.data_mlready.prepare(self.hparams.data_normalizer)

    def setup(self, stage: str = None) -> None:
        """Load data. Set `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `
            "predict"`. Defaults to ``None``.
        """
        self._set_batch_size()
        log.info(Back.GREEN + "Loading data in memory...")

        data_folder = self.hparams.data_mlready.cache_folder
        loader = self.hparams.data_awkward2torch

        label = 0
        if self.data_train is None and self.data_val is None and self.data_test is None:
            self.data_train = loader.load_folder(data_folder / 'train')
            self.data_val = loader.load_folder(data_folder / 'valid')
            self.data_test = loader.load_folder(data_folder / 'test')

            self.label_train = np.full(self.data_train.size(dim=0), label)
            self.label_val = np.full(self.data_val.size(dim=0), label)
            self.label_test = np.full(self.data_test.size(dim=0), label)

        aux_folder = data_folder / 'aux'
        if not self.aux_val and not self.aux_test:
            for dataset_path in aux_folder.iterdir():
                name = dataset_path.stem
                val = loader.load_folder(dataset_path / 'valid')
                test = loader.load_folder(dataset_path / 'test')

                self.aux_val.update({name: val})
                self.aux_val_labels.update({name: np.full(val.size(dim=0), label)})
                self.aux_test.update({name: test})
                self.aux_test_labels.update({name: np.full(test.size(dim=0), label)})

                label +=1

        self._data_summary()
        exit(1)

    def _data_summary(self):
        """Make a neat little summary of what data is being used."""
        log.info(Fore.MAGENTA + '-'*5 + " Data Summary " + '-'*5)
        log.info(Fore.GREEN + f"Training data:")
        log.info(f"Zero bias: {self.data_train.shape}")

        log.info(Fore.GREEN + f"Validation data:")
        log.info(f"Zero bias: {self.data_val.shape}")
        for data_name, data in self.aux_val.items():
            log.info(f"{data_name}: {data.shape}")

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
        return L1ADDataset(
            self.data_train,
            self.labels_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
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
        main_val = L1ADDataset(self.data_val, self.labels_val, batch_size=batch_size)
        data_val.update({"main_val": main_val})
        if not self.aux_val:
            return data_val

        for dataset_name in self.aux_val.keys():
            batch_size = self._get_validation_batchsize(self.aux_val[dataset_name])
            data_val[dataset_name] = L1ADDataset(
                self.aux_val[dataset_name],
                self.aux_val_labels[dataset_name],
                batch_size=batch_size
            )

        return data_val

    def test_dataloader(self) -> Dataset[Any]:
        """Create and return the test dataloader.

        This is a dictionary containing all the loaded datasets, not just the main
        data validation split. Hence, we return a dictionary of dataloaders.
        """
        # Shuffling is by default false for the custom dataset.
        batch_size = self._get_validation_batchsize(self.data_test)
        main_test = L1ADDataset(self.data_test, self.labels_test, batch_size=batch_size)
        data_test = {}
        data_test.update({"main_test": main_test})
        if not self.aux_test:
            return data_test

        for dataset_name in self.aux_test.keys():
            batch_size = self._get_validation_batchsize(self.aux_test[dataset_name])
            data_test[dataset_name] = L1ADDataset(
                self.aux_test[dataset_name],
                self.aux_test_labels[dataset_name],
                batch_size=batch_size
            )

        return data_test

    # @retry((Exception), tries=3, delay=3, backoff=0)
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer custom dataset to gpu faster.

        If the transfer fails, which can happen when a lot of jobs are running at
        the same time and the gpu is out of memory, then try to transfer the data
        again after 1 second of delay. Attempt to do this 3 times before throwing
        an error and stopping the process.
        """
        data, labels = batch
        if "cuda" in str(device):
            return data.to(device, non_blocking=True), labels

        return data.to(device), labels

    def _get_validation_batchsize(self, data: np.ndarray):
        if self.hparams.val_batches == -1:
            return self.batch_size_per_device

        return int(len(data) / self.hparams.val_batches)

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
