import torch
import numpy as np
import math
from torch.utils.data import IterableDataset


class L1ADDataset(IterableDataset):
    """Custom Dataset for loading anomaly detection numpy data.

    The data is assumed to be already loaded in memory.
    """

    def __init__(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None,
        batch_size: int,
        shuffler: torch.Generator | None = None,
    ):
        assert data.shape[0] == mask.shape[0]
        assert labels.shape[0] == data.shape[0]

        self.data, self.mask, self.labels = data, mask, labels
        self.batch_size = batch_size
        self.shuffler = shuffler

        self.n = data.shape[0]
        self.num_batches = (self.n + self.batch_size - 1) // self.batch_size
        self.starts = torch.arange(self.num_batches, dtype=torch.int64) * self.batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.shuffler:
            order = torch.randperm(self.num_batches, generator=self.shuffler)
            starts = self.starts[order]
        else:
            starts = self.starts

        bs = self.batch_size
        n = self.n

        for s in starts.tolist():
            e = s + bs
            if e > n:
                e = n

            x = self.data[s:e]
            m = self.mask[s:e]
            y = self.labels[s:e]

            yield x, m, y
