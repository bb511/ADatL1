"""The statistics of the lattice step: a run's entry at its epoch, the validity test
and the bootstrap standard error of Q'' that the gate reads."""

import functools
import math
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import rankdata

from src.algorithms import ADLightningModule

# The validity test's permutations, as in the criteria study of 20 Oct.
PERMUTATIONS, PERMUTATION_SEED = 10_000, 4242
GATE_DRAWS, GATE_SEED = 1_000, 20261021


def at(values: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    """Each run's entry at its epoch, from a (runs, epochs + 1, ...) array."""
    return values[np.arange(len(epochs)), epochs + 1]


def spearman_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman's rho and its two-sided permutation p-value.

    p = (1 + #{abs(rho*) >= abs(rho)}) / (1 + N) over N permutations of y.
    """
    if len(x) < 2:
        return math.nan, math.nan
    rx, ry = (r - r.mean() for r in (rankdata(x), rankdata(y)))
    scale = math.sqrt((rx @ rx) * (ry @ ry))
    if scale == 0:
        return math.nan, math.nan
    # A permutation keeps the norm of ry, so the numerators order the |rho|. Centred
    # ranks are multiples of 1/2, so the products are exact and ties compare equal.
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = rng.permuted(np.tile(ry, (PERMUTATIONS, 1)), axis=1) @ rx
    extreme = (np.abs(null) >= abs(rx @ ry)).sum()
    return float(rx @ ry / scale), float((1 + extreme) / (1 + PERMUTATIONS))


def operational_se(scores: np.ndarray, batch_sizes: list, target_fpr: float):
    """Bootstrap standard error of Q'' over the rows of validation normal.

    Each draw resamples the rows with replacement and recomputes Q'' as the
    evaluation does: the operational score of every batch of the stored layout,
    averaged over the batches.
    """
    module, sizes = SimpleNamespace(target_fpr=target_fpr), list(batch_sizes)
    score = functools.partial(ADLightningModule.compute_operational_ascore, module)
    rng = np.random.default_rng(GATE_SEED)
    rows = torch.from_numpy(rng.integers(len(scores), size=(GATE_DRAWS, len(scores))))
    draws = torch.from_numpy(scores)[rows]
    values = [np.mean([score(batch) for batch in draw.split(sizes)]) for draw in draws]
    return float(np.std(values, ddof=1))
