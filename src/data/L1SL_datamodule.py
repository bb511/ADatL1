from typing import Any, Optional, Union
from pathlib import Path
from retry import retry

import torch
import numpy as np
import mlflow
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset, random_split
from sklearn.model_selection import train_test_split

from src.utils import pylogger
from colorama import Fore, Back, Style
from src.data.components.dataset import L1SLDataset, L1ADDataset

log = pylogger.RankedLogger(__name__)


class L1SLDatamodule(LightningDataModule):
    def __init__(
        self,
        data_extractor: "L1DataExtractor",
        data_processor: "L1DataProcessor",
        data_normalizer: "L1DataNormalizer",
        zerobias: dict,
        signal: dict,
        background: dict,
        partitions: dict,
        classifier_signals: list[str],
        split: tuple = (0.8, 0.1, 0.1),
        batch_size: int = 16384,
        val_batches: int = -1,
    ) -> None:
        """Prepare the L1 data from specific h5 files to be used for training.

        The h5 files need to be produced by the code at
        https://gitlab.cern.ch/cms-l1-ad/l1tntuple-maker/-/tree/new_h5generation

        :param data_extractor: Class that extracts the data from the given h5 files.
        :param data_processor: Class that processes the extracted data.
        :param data_normalizer: Class that normalizes the processed data.
        :param zerobias: Dictionary of paths to the zerobias data.
        :param signals: Dictionary of paths to simulation data of possible anomalies.
        :param background: Dictionary of paths to simulation of data that is guaranteed
            to not contain any anomalies.
        :param partitions: Dictionary for how the data is distributed between:
            - the main data, to be split in train, val, and test
            - auxiliary datasets to be used for validation
            - auxiliary datasets to be used at the test stage
        :param split: Tuple specifying the fractions of the main data to be assigned to
            training, validation, and test.
        :param batch_size: Integer specifying the batch size of the data.
        :param val_batches: Integer specifying how many batches to split each
            validation dataset into. Defaults to -1, which means to use the same batch
            size as the training data.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.aux_val: Optional[Dataset] = None
        self.aux_test: Optional[Dataset] = None
        self.nsignal = 0
        self.nbackground = 0

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
        self._configure_partitions()

        if self.data_train is None and self.data_val is None and self.data_test is None:
            main_data = self._get_processed_data(self.hparams.partitions["main_data"])
            main_data_y = self._get_data_labels(main_data)
            main_data = np.concatenate(list(main_data.values()), axis=0)
            self.data_train, self.data_test, self.data_train_y, self.data_test_y = \
                train_test_split(
                    main_data,
                    main_data_y,
                    test_size=self.hparams.split[-1],
                    random_state=42,
                    stratify=main_data_y
                )

            val_proportion = len(main_data)*self.hparams.split[1]/len(self.data_train)
            self.data_train, self.data_val, self.data_train_y, self.data_val_y = \
                train_test_split(
                    self.data_train,
                    self.data_train_y,
                    test_size=val_proportion,
                    random_state=42,
                    stratify=self.data_train_y
                )
            del main_data
            del main_data_y

            self.hparams.data_normalizer.fit(self.data_train)
            self.data_train = self._normalize_data(self.data_train, name="train")
            self.data_val = self._normalize_data(self.data_val, name="val")
            self.data_test = self._normalize_data(self.data_test, name="test")
            # Make every type of signal an anomaly.
            # Validation data is dealt with in a special way later.
            self.data_train_y[self.data_train_y > 0] = 1
            self.data_test_y[self.data_test_y > 0] = 1

        if self.hparams.partitions["aux_validation"]:
            dataset_names = self.hparams.partitions["aux_validation"]
            self.aux_val = self._get_processed_data(dataset_names)
            self.aux_val = self._normalize_data_dict(self.aux_val)

        if self.hparams.partitions["aux_test"]:
            dataset_names = self.hparams.partitions["aux_test"]
            self.aux_test = self._get_processed_data(dataset_names)
            self.aux_test = self._normalize_data_dict(self.aux_test)

    def _configure_partitions(self):
        """Partition the data into main data and auxiliary data."""
        self.hparams.partitions['main_data'].update(
            {'signal': self.hparams.classifier_signals}
        )

        if self.hparams.partitions['aux_validation']:
            for partition in self.hparams.partitions['aux_validation'].keys():
                for signal in self.hparams.classifier_signals:
                    if signal in self.hparams.partitions['aux_validation'][partition]:
                        self.hparams.partitions['aux_validation'][partition].remove(signal)

        if self.hparams.partitions['aux_test']:
            for partition in self.hparams.partitions['aux_test'].keys():
                for signal in self.hparams.classifier_signals:
                    if signal in self.hparams.partitions['aux_test'][partition]:
                        self.hparams.partitions['aux_test'][partition].remove(signal)

    def _normalize_data(self, data: np.ndarray, name: str):
        """Normalize a data set, given that the normalization parameters are known."""
        data = self.hparams.data_normalizer.norm(data, name, self.trainer.loggers)
        return torch.from_numpy(data)

    def _normalize_data_dict(self, data_dict: dict):
        """Normalize dictionary of data sets."""
        for data_name, data in data_dict.items():
            data_dict[data_name] = self._normalize_data(data, data_name)

        return data_dict

    def _get_processed_data(self, partition: dict):
        """Load processed data."""
        log.info(f"Loading processed {partition}...")
        processed_data_folder = Path(self.hparams.data_processor.cache)
        processed_data_folder /= self.hparams.data_processor.name

        data = {}
        for data_category in partition.keys():
            root_dir = processed_data_folder / data_category
            filenames = partition[data_category]
            data.update(self._load_datafiles(root_dir, filenames))

        return data

    def _load_datafiles(self, root_dir: Path, filenames: list[str]) -> dict:
        """Loads list of data files into a dictionary of numpy arrays."""
        data = {}
        for filename in filenames:
            data_nparray = np.load(root_dir / (filename + '.npy'), mmap_mode='r')
            data.update({filename: data_nparray})

        return data

    def _get_data_labels(self, main_data: dict):
        """Get the labels of the main data that the training happens on."""
        labels = []
        signal_number = 1
        self.signal_numbers = {}
        for dname in main_data.keys():
            if dname in set(self.hparams.partitions["main_data"]["zerobias"]):
                labels.append(np.zeros(len(main_data[dname])))
                self.nbackground += len(main_data[dname])
            elif dname in set(self.hparams.partitions["main_data"]["signal"]):
                labels.append(np.ones(len(main_data[dname]))*signal_number)
                self.nsignal += len(main_data[dname])
                self.signal_numbers.update({dname: signal_number})
                signal_number += 1
            else:
                raise KeyError("Dataset not found when construction labels!")

        return torch.from_numpy(np.concatenate(labels, axis=0))

    def get_pos_weight(self) -> float:
        """Get weight to balance the reduced population of signal samples."""
        pos_weight = self.nbackground/self.nsignal
        return pos_weight

    def _separate_sig_bkg(self, data: torch.Tensor, labels: torch.Tensor) -> dict:
        """Separates the signal and background in a data set."""
        separated_data = {}
        for signal_name, signal_label in self.signal_numbers.items():
            mask = labels == signal_label
            separated_data.update({signal_name: data[mask]})

        # Get the 0 bias signal data.
        separated_data.update({'main_val': data[labels == 0]})

        return separated_data

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

    @retry((Exception), tries=3, delay=3, backoff=0)
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer custom dataset to gpu faster.

        If the transfer fails, which can happen when a lot of jobs are running at
        the same time and the gpu is out of memory, then try to transfer the data
        again after 1 second of delay. Attempt to do this 3 times before throwing
        an error and stopping the process.
        """
        if 'cuda' in str(device):
            return batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        return batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

    def _get_validation_batchsize(self, data: np.ndarray):
        if self.hparams.val_batches == -1:
            return self.batch_size_per_device

        return int(len(data)/self.hparams.val_batches)

    def train_dataloader(self) -> Dataset[Any]:
        """Creates an optimized dataloader for numpy arrays already in memory."""
        return L1SLDataset(
            self.data_train,
            self.data_train_y,
            batch_size=self.batch_size_per_device,
            shuffle=True
        )

    def val_dataloader(self) -> Dataset[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        # Shuffling is by default false for the custom dataset.
        separated_data = self._separate_sig_bkg(self.data_val, self.data_val_y)

        for signal_name, signal_data in separated_data.items():
            if signal_name == 'main_val':
                continue
            batch_size_sig = self._get_validation_batchsize(signal_data)
            signal_labels = torch.from_numpy(np.ones(len(signal_data)))
            separated_data[signal_name] = L1SLDataset(
                signal_data,
                signal_labels,
                batch_size=batch_size_sig
            )

        bkg_data = separated_data['main_val'].detach().clone()
        del separated_data['main_val']
        bkg_labels = torch.from_numpy(np.zeros(len(bkg_data)))

        batch_size_bkg = self._get_validation_batchsize(bkg_data)
        bkg_data = L1SLDataset(
            bkg_data,
            bkg_labels,
            batch_size=batch_size_bkg
        )

        data_val = {}
        # Make sure main_val is always first in the dict !!!
        # The rate callback expects this.
        data_val.update({"main_val": bkg_data})
        data_val.update(separated_data)

        if not self.aux_val:
            return data_val

        for data_name, data in self.aux_val.items():
            batch_size = self._get_validation_batchsize(data)
            labels = torch.from_numpy(np.ones(len(data)))
            self.aux_val[data_name] = L1SLDataset(data, labels, batch_size=batch_size)

        data_val.update(self.aux_val)

        return data_val

    def test_dataloader(self) -> Dataset[Any]:
        """Create and return the test dataloader.

        :return: The validation dataloader.
        """
        # Shuffling is by default false for the custom dataset.
        batch_size = self._get_validation_batchsize(self.data_test)
        main_test = L1SLDataset(self.data_test, self.data_test_y, batch_size=batch_size)
        data_test = {}
        data_test.update({"main_test": main_test})
        if not self.aux_test:
            return data_test

        for data_name, data in self.aux_test.items():
            batch_size = self._get_validation_batchsize(data)
            labels = torch.from_numpy(np.ones(len(data)))
            self.aux_test[data_name] = L1SLDataset(data, labels, batch_size=batch_size)

        data_test.update(self.aux_test)

        return data_test
