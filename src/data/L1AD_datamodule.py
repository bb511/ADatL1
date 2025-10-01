from typing import Any, Optional, Union
from pathlib import Path

# from retry import retry

import torch
import numpy as np
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
        data_extractor: "L1DataExtractor",
        data_processor: "L1DataProcessor",
        data_normalizer: "L1DataNormalizer",
        zerobias: dict,
        signal: dict,
        background: dict,
        partitions: dict,
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

        self.data_train, self.data_val, self.data_test = None, None, None
        self.labels_train, self.labels_val, self.labels_test = None, None, None
        self.aux_val, self.aux_test = None, None

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

        if self.data_train is None and self.data_val is None and self.data_test is None:
            main_data = self._get_processed_data(self.hparams.partitions["main_data"])
            main_data_data, main_data_labels = [], []
            for key, data in main_data.items():
                main_data_data.append(data)
                main_data_labels.append(np.full(len(data), key, dtype=object))
            del main_data

            main_labels = np.concatenate(main_data_labels, axis=0)
            main_data = np.concatenate(main_data_data, axis=0)
            del main_data_data, main_data_labels

            splits = random_split(
                dataset=main_labels,
                lengths=self.hparams.split,
                generator=torch.Generator().manual_seed(42),
            )
            train_indices, val_indices, test_indices = (
                splits[0].indices,
                splits[1].indices,
                splits[2].indices,
            )

            # Use indices to get the corresponding data and labels
            self.data_train, self.data_val, self.data_test = (
                main_data[train_indices],
                main_data[val_indices],
                main_data[test_indices],
            )
            self.labels_train, self.labels_val, self.labels_test = (
                main_labels[train_indices],
                main_labels[val_indices],
                main_labels[test_indices],
            )
            del main_data, main_labels, splits, train_indices, val_indices, test_indices

            self.hparams.data_normalizer.fit(self.data_train)
            self.data_train = self._normalize_data(self.data_train, name="train")
            self.data_val = self._normalize_data(self.data_val, name="val")
            self.data_test = self._normalize_data(self.data_test, name="test")

        if self.hparams.partitions["aux_validation"]:
            dataset_names = self.hparams.partitions["aux_validation"]
            self.aux_val = self._get_processed_data(dataset_names)
            self.aux_val = self._normalize_data_dict(self.aux_val)

        if self.hparams.partitions["aux_test"]:
            dataset_names = self.hparams.partitions["aux_test"]
            self.aux_test = self._get_processed_data(dataset_names)
            self.aux_test = self._normalize_data_dict(self.aux_test)

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
            data_nparray = np.load(root_dir / (filename + ".npy"), mmap_mode="r")
            data.update({filename: data_nparray})

        return data

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

    def train_dataloader(self) -> Dataset[Any]:
        """Creates an optimized dataloader for numpy arrays already in memory."""
        return L1ADDataset(
            self.data_train,
            self.labels_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
        )

    def val_dataloader(self) -> Dataset[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        # Shuffling is by default false for the custom dataset.
        batch_size = self._get_validation_batchsize(self.data_val)
        main_val = L1ADDataset(self.data_val, self.labels_val, batch_size=batch_size)
        data_val = {}
        # Make sure main_val is always first in the dict !!!
        # The rate callback expects this.
        data_val.update({"main_val": main_val})
        if not self.aux_val:
            return data_val

        for data_name, data in self.aux_val.items():
            batch_size = self._get_validation_batchsize(data)
            self.aux_val[data_name] = L1ADDataset(data, None, batch_size=batch_size)

        data_val.update(self.aux_val)

        return data_val

    def test_dataloader(self) -> Dataset[Any]:
        """Create and return the test dataloader.

        :return: The validation dataloader.
        """
        # Shuffling is by default false for the custom dataset.
        batch_size = self._get_validation_batchsize(self.data_test)
        main_test = L1ADDataset(self.data_test, self.labels_test, batch_size=batch_size)
        data_test = {}
        data_test.update({"main_test": main_test})
        if not self.aux_test:
            return data_test

        for data_name, data in self.aux_test.items():
            batch_size = self._get_validation_batchsize(data)
            self.aux_test[data_name] = L1ADDataset(data, None, batch_size=batch_size)

        data_test.update(self.aux_test)

        return data_test


class L1ADDataModuleDebug(L1ADDataModule):
    """Debug version of L1ADDataModule that generates random data instead of loading real data."""

    def prepare_data(self) -> None:
        """Skip data extraction and processing for debugging."""
        log.info(Back.GREEN + "Debug mode: Skipping data preparation...")
        pass

    def setup(self, stage: str = None) -> None:
        """Generate random data instead of loading real data."""
        self._set_batch_size()

        if self.data_train is None and self.data_val is None and self.data_test is None:
            log.info("Debug mode: Generating 1000 random samples with 57 features...")

            main_data = np.random.randn(1000, 57).astype(np.float32)
            main_labels = np.concatenate(
                [
                    np.full(400, "background", dtype=object),
                    np.full(300, "simulation_1", dtype=object),
                    np.full(300, "simulation_2", dtype=object),
                ]
            )
            splits = random_split(
                dataset=main_labels,
                lengths=[700, 200, 100],
                generator=torch.Generator().manual_seed(42),
            )
            train_indices, val_indices, test_indices = (
                splits[0].indices,
                splits[1].indices,
                splits[2].indices,
            )
            self.data_train, self.data_val, self.data_test = (
                torch.from_numpy(main_data[train_indices]),
                torch.from_numpy(main_data[val_indices]),
                torch.from_numpy(main_data[test_indices]),
            )
            self.labels_train, self.labels_val, self.labels_test = (
                main_labels[train_indices],
                main_labels[val_indices],
                main_labels[test_indices],
            )

            log.info(
                f"Generated datasets - Train: {len(self.data_train)}, "
                f"Val: {len(self.data_val)}, Test: {len(self.data_test)}"
            )

        # Change the mean of the components:
        if hasattr(self.hparams, "partitions") and self.hparams.partitions.get(
            "aux_validation"
        ):
            self.aux_val = {}
            for ids, (_, filenames) in enumerate(
                self.hparams.partitions["aux_validation"].items()
            ):
                for filename in filenames:
                    random_data = ids + np.random.randn(100, 57).astype(np.float32)
                    self.aux_val[filename] = torch.from_numpy(random_data)

        if hasattr(self.hparams, "partitions") and self.hparams.partitions.get(
            "aux_test"
        ):
            self.aux_test = {}
            for ids, (_, filenames) in enumerate(
                self.hparams.partitions["aux_test"].items()
            ):
                for filename in filenames:
                    random_data = ids + np.random.randn(100, 57).astype(np.float32)
                    self.aux_test[filename] = torch.from_numpy(random_data)
