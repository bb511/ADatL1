"""Helpers shared by the criterion callbacks and the score dump."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def as_scores(values) -> np.ndarray:
    """Anomaly scores as a flat float64 array, from anything ``np.asarray`` accepts."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def placeholder_loader(x: torch.Tensor, batch_size: int) -> DataLoader:
    # The labels are placeholders: model_step reads only the inputs.
    labels = torch.zeros(len(x), dtype=torch.long)
    return DataLoader(TensorDataset(x, labels), batch_size=batch_size)


def row_subset(n: int, max_rows: int | None, seed: int) -> torch.Tensor | None:
    """Sorted indices of max_rows of n rows drawn without replacement, or None for every
    row when max_rows is None or at least n."""
    if max_rows is None or max_rows >= n:
        return None
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:max_rows].sort().values
