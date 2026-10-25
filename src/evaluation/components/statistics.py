"""The statistics of the lattice step: a run's entry at its epoch and the validity
test."""

import math

import numpy as np
from scipy.stats import rankdata

# The validity test's permutations, as in the criteria study of 20 Oct.
PERMUTATIONS, PERMUTATION_SEED = 10_000, 4242


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
