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
        l1bit: torch.Tensor,
        labels: torch.Tensor | None,
        batch_size: int,
        max_batches: int | None = None,
        shuffler: torch.Generator | None = None,
    ):
        assert data.shape[0] == mask.shape[0]
        assert labels.shape[0] == data.shape[0]

        self.data, self.mask, self.l1bit, self.labels = data, mask, l1bit, labels
        self.batch_size = batch_size
        self.shuffler = shuffler

        self.max_batches = max_batches
        self.n = data.shape[0]
        self.num_batches = (self.n + self.batch_size - 1) // self.batch_size
        self.starts = (
            torch.arange(self.num_batches, dtype=torch.int64) * self.batch_size
        )

    def __len__(self):
        if self.max_batches is None:
            return self.num_batches
        return min(self.num_batches, int(self.max_batches))

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.shuffler:
            order = torch.randperm(self.num_batches, generator=self.shuffler)
            starts = self.starts[order]
        else:
            starts = self.starts

        bs = self.batch_size
        n = self.n

        nb = 0
        for s in starts.tolist():
            if self.max_batches is not None and nb >= self.max_batches:
                break

            e = s + bs
            if e > n:
                e = n

            x = self.data[s:e]
            m = self.mask[s:e]
            l = self.l1bit[s:e]
            y = self.labels[s:e]

            yield x, m, l, y

            nb += 1
