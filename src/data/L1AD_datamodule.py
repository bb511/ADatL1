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
        partitions: dict,
        split: tuple,
        batch_size: int,
        use_entire_val_dataset: bool = False,
        device: str = 'cpu',
        load_all: bool = False,
    ) -> None:
        """Prepare the L1 data from specific h5 files to be used for training.

        The h5 files need to be produced by the code at
        https://gitlab.cern.ch/cms-l1-ad/l1tntuple-maker/-/tree/new_h5generation

        :param zerobias: Dictionary of paths to the zerobias data.
        :param signals: Dictionary of paths to simulation data of possible anomalies.
        :param background: Dictionary of paths to simulation of data that is guaranteed
            to not contain any anomalies.
        :param data_extractor: Class that extracts the data from the given h5 files.
        :param data_processor: Class that processes the extracted data.
        :param data_normalizer: Class that normalizes the processed data.
        :param partitions: Dictionary for how the data is distributed between:
            - the main data, to be split in train, val, and test
            - auxiliary datasets to be used for validation
            - auxiliary datasets to be used at the test stage
        :param split: Tuple specifying the fractions of the main data to be assigned to
            training, validation, and test.
        :param loader: Dictionary with attributes handling the dataloading, such
            as the batch size, number of workers, whether to use pin_memory, etc.
        :param batch_size: Integer specifying the batch size of the data.
        :param use_entire_val_dataset: Bool whether to disregard batch_size and use
            the entire dataset when running validation on it.
        :param device: String specifying the device the data is loaded to.
        :param load_all: Bool whether to load all the data to memory (ram or gpu vram)
            before the training starts.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.aux_val: Optional[Dataset] = None
        self.aux_test: Optional[Dataset] = None

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
            main_data = np.concatenate(list(main_data.values()), axis=0)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=main_data,
                lengths=self.hparams.split,
                generator=torch.Generator().manual_seed(42),
            )

            self.hparams.data_normalizer.fit(self.data_train[:])
            self.data_train = self._normalize_data(self.data_train, name="train")
            self.data_val = self._normalize_data(self.data_val, name="val")
            self.data_test = self._normalize_data(self.data_test, name="test")
            self._load_data_to_device()

        if self.hparams.partitions["aux_validation"]:
            dataset_names = self.hparams.partitions["aux_validation"]
            self.aux_val = self._get_processed_data(dataset_names)
            self.aux_val = self._normalize_data_dict(self.aux_val)
            self.aux_val = self._load_aux_data_to_device(self.aux_val)

        if self.hparams.partitions["aux_test"]:
            dataset_names = self.hparams.partitions["aux_test"]
            self.aux_test = self._get_processed_data(dataset_names)
            self.aux_test = self._normalize_data_dict(self.aux_test)
            self.aux_test = self._load_aux_data_to_device(self.aux_test)

    def _load_data_to_device(self) -> None:
        """Loads the data numpy arrays to given device in the configuration."""
        log.info(Fore.YELLOW + f"All data will be loaded on {self.hparams.device}.")
        self.data_train = torch.from_numpy(self.data_train).to(self.hparams.device)
        self.data_val = torch.from_numpy(self.data_val).to(self.hparams.device)
        self.data_test = torch.from_numpy(self.data_test).to(self.hparams.device)

    def _load_aux_data_to_device(self, aux_data: dict) -> dict:
        for data_name, data in aux_data.items():
            aux_data[data_name] = torch.from_numpy(data).to(self.hparams.device)

        return aux_data

    def _normalize_data(self, data: np.ndarray, name: str):
        """Normalize a data set, given that the normalization parameters are known."""
        return self.hparams.data_normalizer.norm(data[:], name, self.trainer.loggers)

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

    def _get_validation_batchsize(self, data: np.ndarray):
        if self.hparams.use_entire_val_dataset:
            return len(data)

        return self.batch_size_per_device

    def train_dataloader(self) -> DataLoader[Any] | Dataset[Any]:
        """Creates an optimized dataloader for numpy arrays already in memory."""
        return L1ADDataset(
            self.data_train, batch_size=self.batch_size_per_device, shuffle=True
        )

    def val_dataloader(self) -> DataLoader[Any] | Dataset[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        # Shuffling is by default false for the custom dataset.
        batch_size = self._get_validation_batchsize(self.data_val)
        main_val = L1ADDataset(self.data_val, batch_size=batch_size)
        data_val = {}
        data_val.update({"main_val": main_val})
        if not self.aux_val:
            return data_val

        for data_name, data in self.aux_val.items():
            batch_size = self._get_validation_batchsize(data)
            self.aux_val[data_name] = L1ADDataset(data, batch_size=batch_size)

        data_val.update(self.aux_val)

        return data_val

    def test_dataloader(self) -> DataLoader[Any] | Dataset[Any]:
        """Create and return the test dataloader.

        :return: The validation dataloader.
        """
        # Shuffling is by default false for the custom dataset.
        batch_size = self._get_validation_batchsize(self.data_test)
        main_test = L1ADDataset(self.data_test, batch_size=batch_size)
        data_test = {}
        data_test.update({"main_test": main_test})
        if not self.aux_test:
            return data_test

        for data_name, data in self.aux_test.items():
            batch_size = self._get_validation_batchsize(data)
            self.aux_test[data_name] = L1ADDataset(data, batch_size=batch_size)

        data_test.update(self.aux_test)

        return data_test
