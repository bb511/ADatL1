"""The pushed points of ANCAP v8 on the six plain-vector domains.

The definition is the frozen page
``scratchpad/adl1t/theory/theory_xxxii/redo/POINTS.md`` (md5 06fc2f99...), sections
"Domains", "Rows" and "The possible events", which ``theory_xxxii/redo/FREEZE_2.md``
takes into the freeze. ``SCALES`` holds the scales read there (``RESULTS.md``, "Per
domain"). At these scales the points built here are the reference files of
``theory_xxxii/reference/`` byte for byte. The yardstick that read the scales and the
page's other statistics are not implemented.

A detector's input x has the raw value ``v = a x + b``: a and b are the robust spread
and median of a tabular domain, and 255 times the channel's std and mean of an image
domain, whose pixels are flattened in the order (channel, row, column). The datamodule's
float32 a and b are taken to float64, and all arithmetic is float64. The possible events
are read from the training inputs X_train alone. Per input j, with the tolerance
``tol_j = 1e-6 (abs(b_j) + 10 a_j)``:

- an input that is not logged is never negative when its smallest training value is at
  least ``-tol_j`` (floor 0), and integer when every training value lies within
  ``tol_j`` of an integer;
- the ceiling is 1 for the dijet's tau21, the largest training value for its tau32, 255
  for a pixel, and none otherwise;
- the bounds of a tabular input are ``max(floor, b - 10 a)`` and
  ``min(ceiling, b + 10 a)``, where the datamodules clip x at 10 in absolute value, and
  those of a pixel its floor and ceiling; an integer input takes the ceiling of its
  lower bound and the floor of its upper bound.

The projection Π clips an input that is not integer in the detector's units, so that a
value inside its bounds keeps its bits, rounds an integer input in raw units and clips
it there, and clips a tabular point at ±10 last. RH holds at most 10,000 distinct rows
of the held-out typical rows X_H (the test sets of ``HELD_OUT``, concatenated in that
order), drawn by ``src.callbacks.sds._distinct_rows`` with seed 0. The point pushed from
a start row of RH is ``Π(start + s (out - start))`` in float32, where out is SDS's
pseudo-outlier of that row and s the domain's scale.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.callbacks.sds import _distinct_rows, sds_pseudo_sets

# As read on the grid and frozen (RESULTS.md, "Per domain").
SCALES = {
    "cchamber": 1.414,
    "cchamber_wt": 4.0,
    "cifar10": 1.414,
    "mscct": 1.414,
    "dijet": 2.0,
    "sbc": 4.0,
}
# The test sets whose rows, concatenated in this order, are X_H.
HELD_OUT = {
    "cchamber": ("normal", "reference_normal"),
    "cchamber_wt": ("normal", "reference_normal"),
    "cifar10": ("normal", "reference_normal"),
    "mscct": ("normal", "reference_normal"),
    "dijet": ("reference_normal",),
    "sbc": ("reference_normal",),
}
IMAGE_DOMAINS = ("cifar10", "mscct")
IMAGE_SHAPE = (3, 32, 32)
CLIP = 10.0
PIXEL_MAX = 255.0
# tau21 = tau2 / tau1 is at most 1; typical tau32 exceed 1 (training maxima near 1.19
# and 1.22), so their ceiling is read from the data.
CEILING_ONE = ("tau21_j1", "tau21_j2")
CEILING_DATA = ("tau32_j1", "tau32_j2")
MAX_ROWS, ROW_SEED = 10_000, 0


@dataclass(frozen=True)
class Possible:
    """The possible events of a domain, each field (d,) over the flattened inputs.

    ``lo`` and ``hi`` are the bounds in raw units as used, ±inf where there is none.
    """

    a: np.ndarray
    b: np.ndarray
    tol: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    integer: np.ndarray
    never_negative: np.ndarray
    tabular: bool


def _named(names: list[str], chosen: list[str] | tuple[str, ...]) -> np.ndarray:
    return np.array([name in chosen for name in names], dtype=bool)


def _possible(
    v: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    logged: np.ndarray,
    ceiling: np.ndarray,
    tabular: bool,
) -> Possible:
    """Return the possible events read from the raw training values v (n, d)."""
    # The size of a float32 round trip through the scaling, for x within ±10.
    tol = 1e-6 * (np.abs(b) + CLIP * a)
    never_negative = ~logged & (v.min(axis=0) >= -tol)
    integer = ~logged & (np.abs(v - np.rint(v)).max(axis=0) <= tol)
    lo, hi = np.where(never_negative, 0.0, -np.inf), ceiling
    if tabular:
        lo, hi = np.maximum(lo, b - CLIP * a), np.minimum(hi, b + CLIP * a)
    lo, hi = np.where(integer, np.ceil(lo), lo), np.where(integer, np.floor(hi), hi)
    return Possible(a, b, tol, lo, hi, integer, never_negative, tabular)


def tabular_events(
    x_train: np.ndarray,
    median: np.ndarray,
    spread: np.ndarray,
    names: list[str],
    logged: list[str],
) -> Possible:
    """Return the possible events of a tabular domain, read from X_train (n, d).

    ``median`` and ``spread`` are the datamodule's float32 scaling (d,), and ``names``
    the inputs in model-input order.
    """
    a, b = spread.astype(np.float64), median.astype(np.float64)
    v = a * x_train.astype(np.float64) + b
    ceiling = np.where(_named(names, CEILING_ONE), 1.0, np.inf)
    ceiling = np.where(_named(names, CEILING_DATA), v.max(axis=0), ceiling)
    return _possible(v, a, b, _named(names, logged), ceiling, tabular=True)


def image_events(x_train: np.ndarray, mean: np.ndarray, std: np.ndarray) -> Possible:
    """Return the possible events of an image domain, read from X_train (n, c, h, w).

    ``mean`` and ``std`` are the datamodule's float32 statistics (c,) of the pixels
    divided by 255.
    """
    x = x_train.reshape(len(x_train), -1).astype(np.float64)
    pixels = x.shape[1] // len(mean)
    a = np.repeat(PIXEL_MAX * std.astype(np.float64), pixels)
    b = np.repeat(PIXEL_MAX * mean.astype(np.float64), pixels)
    unlogged, ceiling = np.zeros(len(a), dtype=bool), np.full(len(a), PIXEL_MAX)
    return _possible(a * x + b, a, b, unlogged, ceiling, tabular=False)


def project(p: np.ndarray, events: Possible) -> np.ndarray:
    """Return Π of the points p (n, d), in float64."""
    a, b = events.a, events.b
    x = np.clip(p, (events.lo - b) / a, (events.hi - b) / a)
    v = np.clip(np.rint(a * p + b), events.lo, events.hi)
    x = np.where(events.integer, (v - b) / a, x)
    return np.clip(x, -CLIP, CLIP) if events.tabular else x


def n_violations(x: np.ndarray, events: Possible) -> int:
    """Return the number of float32 rows x (n, d) outside the possible events.

    A row is outside when an input lies more than its tolerance beyond its bounds, or an
    integer input more than its tolerance from an integer.
    """
    v = events.a * x.astype(np.float64) + events.b
    bad = (v < events.lo - events.tol) | (v > events.hi + events.tol)
    bad |= events.integer & (np.abs(v - np.rint(v)) > events.tol)
    return int(bad.any(axis=1).sum())


def distinct_rows(x: torch.Tensor) -> torch.Tensor:
    """Return R of X_train or RH of X_H: at most 10,000 distinct rows, flattened."""
    return _distinct_rows(x, None, MAX_ROWS, ROW_SEED)[0]


def push(
    rh: np.ndarray, scale: float, events: Possible
) -> tuple[np.ndarray, np.ndarray]:
    """Return the points pushed from RH at ``scale`` and their start rows.

    ``rh`` is RH in float64 (m, d). The points are float32 (n, d), and the start rows
    their indices in RH, (n,).
    """
    _, out, _, start_rows = sds_pseudo_sets(rh, return_rows=True)
    start = rh[start_rows]
    return project(start + scale * (out - start), events).astype(np.float32), start_rows


def first_rows(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return the index in x (n, d) of the first row equal to each of ``rows``, int64.

    Raises KeyError for a row that x lacks.
    """
    # Adding 0.0 turns -0.0 into +0.0, so that rows equal in value have equal bytes.
    first = {}
    for i, row in enumerate(x + 0.0):
        first.setdefault(row.tobytes(), i)
    return np.array([first[row.tobytes()] for row in rows + 0.0], dtype=np.int64)
