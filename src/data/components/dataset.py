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
        for s in starts:
            s = int(s)

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


class CIFARADDataset(IterableDataset):
    """Custom dataset for loading anomaly detection image data already in memory.

    The dataset yields batches directly, in the style of L1ADDataset, but returns
    only (x, y). This is intended to work with `unpack_batch`, which supports
    2-tuples and sets mask/l1bit to None.
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        max_batches: int | None = None,
        shuffler: torch.Generator | None = None,
    ):
        if data.shape[0] != labels.shape[0]:
            raise ValueError("data and labels must have the same first dimension.")

        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.shuffler = shuffler

        self.n = data.shape[0]
        self.num_batches = (self.n + self.batch_size - 1) // self.batch_size
        self.starts = torch.arange(self.num_batches, dtype=torch.int64) * self.batch_size

    def __len__(self):
        if self.max_batches is None or self.max_batches < 0:
            return self.num_batches
        return min(self.num_batches, int(self.max_batches))

    def __iter__(self):
        if self.shuffler is not None:
            order = torch.randperm(self.num_batches, generator=self.shuffler)
            starts = self.starts[order]
        else:
            starts = self.starts

        bs = self.batch_size
        n = self.n

        nb = 0
        for s in starts:
            s = int(s)

            if self.max_batches is not None and self.max_batches >= 0 and nb >= self.max_batches:
                break

            e = min(s + bs, n)
            x = self.data[s:e]
            y = self.labels[s:e]

            yield x, y
            nb += 1


class RobustADDataset(IterableDataset):
    """Custom Dataset for loading RobustAD tensors already in memory."""

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None,
        batch_size: int,
        max_batches: int | None = None,
        shuffler: torch.Generator | None = None,
    ):
        assert labels.shape[0] == data.shape[0]
        if mask is not None:
            assert mask.shape[0] == data.shape[0]

        self.data = data
        self.labels = labels
        self.mask = mask
        self.batch_size = batch_size
        self.shuffler = shuffler

        self.max_batches = max_batches
        self.n = data.shape[0]
        self.num_batches = (self.n + self.batch_size - 1) // self.batch_size
        self.starts = torch.arange(self.num_batches, dtype=torch.int64) * self.batch_size

    def __len__(self):
        if self.max_batches is None or self.max_batches == -1:
            return self.num_batches
        return min(self.num_batches, int(self.max_batches))

    def __iter__(self):
        if self.shuffler is not None:
            order = torch.randperm(self.num_batches, generator=self.shuffler)
            starts = self.starts[order]
        else:
            starts = self.starts

        bs = self.batch_size
        n = self.n

        nb = 0
        for s in starts:
            s = int(s)

            if self.max_batches is not None and self.max_batches != -1 and nb >= self.max_batches:
                break

            e = min(s + bs, n)

            x = self.data[s:e]
            y = self.labels[s:e]

            if self.mask is None:
                yield x, y
            else:
                m = self.mask[s:e]
                yield x, m, y

            nb += 1
