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
        background_filepath: str,
        background_data_extractor: "CMSDataExtractor",
        signal_filepath: str,
        signal_data_extractor: "CMSDataExtractor",
        data_processor: "CMSDataProcessor",
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

    def prepare_data(self) -> None:
        """Get background and signal data."""
        background_datadict = self.hparams.background_data_extractor.extract(self.hparams.background_filepath)
        signal_datadict = self.hparams.signal_data_extractor.extract(self.hparams.background_filepath)
        
        # Processing of the data:
        background_datadict, signal_datadict = self.hparams.data_processor.process(background_datadict, signal_datadict)
        
        # TODO: Store the processed data in processed_data_path.
        import ipdb; ipdb.set_trace()

        # NOT SURE HOW TO PROCEED:
        