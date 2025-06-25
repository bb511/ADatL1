# Custom dataset for AD data numpy array that is already loaded in memory.

import torch
import numpy as np
from torch.utils.data import Dataset


class L1ADDataset(Dataset):
    """Custom Dataset for loading anomaly detection numpy data.

    The data is assumed to be already loaded in memory.
    """
    def __init__(self, data: torch.Tensor, batch_size: int, shuffle: bool = False):
        self.data = data
        self.batch_size = batch_size
        self.max_idx = self.data.size()[0]
        self.indexer = np.arange(self.data.size()[0])
        self.shuffle = shuffle

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, batch_idx: int):
        if batch_idx == 0 and self.shuffle:
            np.random.shuffle(self.indexer)
        if batch_idx == self.max_idx-1:
            batch = self.data[self.indexer[self.batch_idx*self.batch_size:], ...]
        else:
            batch = self.data[
                self.indexer[
                    batch_idx*self.batch_size:batch_idx*self.batch_size + self.batch_size],
                    ...
                ]

        return batch
