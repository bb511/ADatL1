import torch
import numpy as np
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
        self.max_idx = self.data.size()[0]
        self.indexer = np.arange(self.data.size()[0])
        self.shuffler = shuffler

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_idx == 0 and self.shuffler:
            self.shuffled_indexer = self.indexer.copy()
            self.shuffler.shuffle(self.shuffled_indexer)

        idxer = self.shuffled_indexer if self.shuffler else self.indexer

        if batch_idx == self.max_idx - 1:
            nidxs = idxer[batch_idx * self.batch_size :]
            x = self.data[nidxs, ...]

            if self.labels is None:
                y = None
            else:
                y = self.labels[nidxs, ...]

            return (x, y)

        nidxs = idxer[
            batch_idx * self.batch_size : batch_idx * self.batch_size + self.batch_size
        ]
        x = self.data[nidxs, ...]
        if self.labels is None:
            y = None
        else:
            y = self.labels[nidxs, ...]

        return (x, y)
