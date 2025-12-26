import torch
import numpy as np
import math
from torch.utils.data import Dataset


class L1ADDataset(Dataset):
    """Custom Dataset for loading anomaly detection numpy data.

    The data is assumed to be already loaded in memory.
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        shuffler: np.random.Generator = None,
    ):
        self.data, self.labels = data, labels
        self.batch_size = batch_size
        self.indexer = np.arange(self.data.size()[0])
        self.shuffler = shuffler
        self.shuffled_indexer = self.indexer

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_idx == 0 and self.shuffler is not None:
            self.shuffled_indexer = self.indexer.copy()
            self.shuffler.shuffle(self.shuffled_indexer)

        idxer = self.shuffled_indexer if self.shuffler is not None else self.indexer

        start = batch_idx * self.batch_size
        stop = min(start + self.batch_size, self.data.shape[0])
        nidxs = idxer[start:stop]

        x = self.data[nidxs, ...]
        y = None if self.labels is None else self.labels[nidxs, ...]
        return x, y
