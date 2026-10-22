"""Helpers shared by the criterion callbacks and the score dump."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def as_scores(values) -> np.ndarray:
    """Anomaly scores as a flat float64 array, from anything ``np.asarray`` accepts."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def placeholder_loader(
    x: torch.Tensor, batch_size: int, mask: torch.Tensor | None = None
) -> DataLoader:
    """Loader of made-up points with placeholder labels, which model_step does not read.

    Without a mask the batches are (x, y). With a data mask of the shape of x they are
    the (x, mask, l1bit, y) of the L1AD loaders, with placeholder l1bits and x set to +0
    wherever the mask is 0, as padding is in the L1AD data.
    """
    labels = torch.zeros(len(x), dtype=torch.long)
    if mask is None:
        return DataLoader(TensorDataset(x, labels), batch_size=batch_size)
    x = torch.where(mask.bool(), x, torch.zeros_like(x))
    l1bit = torch.zeros(len(x), dtype=torch.bool)
    return DataLoader(TensorDataset(x, mask, l1bit, labels), batch_size=batch_size)


def drawn_masks(mask: torch.Tensor | None, n: int, seed: int) -> torch.Tensor | None:
    """The masks of n rows of ``mask`` drawn uniformly with replacement with ``seed``,
    or None when the data carry no mask."""
    if mask is None:
        return None
    generator = torch.Generator().manual_seed(seed)
    return mask[torch.randint(len(mask), (n,), generator=generator)]


def row_subset(n: int, max_rows: int | None, seed: int) -> torch.Tensor | None:
    """Sorted indices of max_rows of n rows drawn without replacement, or None for every
    row when max_rows is None or at least n."""
    if max_rows is None or max_rows >= n:
        return None
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:max_rows].sort().values
