from typing import Any, Dict, Optional, List
import os
import gc
import h5py


from pytorch_lightning import LightningDataModule

# from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.datasets import MNIST
# from torchvision.transforms import transforms

# Temporally:
import numpy as np

class L1ADDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```
    """

    def __init__(
        self,
        zerobias_filepaths: str,
        signal_filepaths: str,
        data_extractor: "L1DataExtractor",
        data_processor: "L1DataProcessor",
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Get zero bias data and the simulated MC signal data."""
        zerobias_data = self.hparams.data_extractor.extract(self.hparams.background_filepaths)
        signal_data = self.hparams.data_extractor.extract(self.hparams.signal_filepaths)
        
        # Processing of the data:
        background_datadict, signal_datadict = self.hparams.data_processor.process(background_datadict, signal_datadict)
        
        # TODO: Store the processed data in processed_data_path.
        import ipdb; ipdb.set_trace()

        # NOT SURE HOW TO PROCEED:
        

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
