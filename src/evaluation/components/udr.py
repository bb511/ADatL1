"""The lattice criteria, UDR and model centrality, from the stored validation normal
scores of the runs and of their partners."""

import math
from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata


def _unit_ranks(scores: np.ndarray) -> np.ndarray:
    """Centred average ranks along the last axis, scaled to unit norm.

    As in UOMS (UDR.py and MC.py of github.com/yzhao062/UOMS, commit a27ebbb),
    non-finite scores pass through ``np.nan_to_num`` first, and a constant vector,
    whose correlation is undefined, gives zeros and so a rho of 0.
    """
    ranks = rankdata(np.nan_to_num(scores.astype(np.float64)), axis=-1)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=-1, keepdims=True)
    return np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)


def _udr(
    lattice: np.ndarray, partner: list[np.ndarray | None] | None
) -> np.ndarray | None:
    # With one other seed, the median of UDR is that seed's rho.
    if partner is None:
        return None
    udr = np.full(lattice.shape[:2], math.nan)
    paired = [i for i, scores in enumerate(partner) if scores is not None]
    if paired:
        units = _unit_ranks(np.stack([partner[i] for i in paired]))
        udr[paired] = (_unit_ranks(lattice[paired]) * units).sum(axis=-1)
    return udr


def _model_centrality(
    lattice: np.ndarray, partner: list[np.ndarray | None] | None
) -> np.ndarray:
    # Ranked once per (run, epoch); rho is (epochs, runs, runs), with 1 on the
    # diagonal (0 for a constant score), which the mean over the others leaves out.
    units = _unit_ranks(lattice).transpose(1, 0, 2)
    rho = units @ units.transpose(0, 2, 1)
    others = rho.sum(axis=2) - np.diagonal(rho, axis1=1, axis2=2)
    return (others / (len(lattice) - 1)).T


# The criteria of Ma et al. (SIGKDD Explor. 25(1), 2023):
LATTICE_CRITERIA: dict[str, Callable] = {"udr": _udr, "mc": _model_centrality}
