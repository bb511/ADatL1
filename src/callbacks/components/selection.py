"""The epoch rule and the hyperpoint rule of the selection protocol.

A strategy picks among candidates by two quantities, its own Q' and the operational
score Q'' of validation normal, minimising

    w rank(Q') + (1 - w) rank(Q''),    w = k / W_STEPS, k = 0, ..., W_STEPS,

ties to the better Q', then to the earlier candidate. Over the epochs of a run both
quantities are first smoothed by the exponential moving average of the training
callbacks and rounded to float32, as Lightning logs them; at k = W_STEPS the pick is then
the epoch that ModelCheckpoint(save_top_k=1) keeps for Q', which ``best_epoch`` gives.
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata

# The weight of the previous value in the EMA of every training callback.
EMA_WEIGHT = 0.9
# The registered grid w in {0, 0.1, ..., 1}, as integer steps so that rank sums of
# average ranks compare exactly.
W_STEPS = 10


def ema_update(
    average: float | None,
    value: float,
    weight: float = EMA_WEIGHT,
    skip_nonfinite: bool = False,
) -> float | None:
    """One step of the EMA of the training callbacks, None or NaN before the first.

    The first value is taken as it is, then w average + (1 - w) value. By default a
    non-finite value propagates, as in the drift, Wasserstein and efficiency callbacks;
    with ``skip_nonfinite`` it leaves the average unchanged, as in CAPCallback, whose
    average stays NaN until the first finite value.
    """
    value = float(value)
    if skip_nonfinite and not math.isfinite(value):
        return average
    if average is None or (skip_nonfinite and math.isnan(average)):
        return value
    return weight * average + (1 - weight) * value


def ema(
    values: Sequence[float], weight: float = EMA_WEIGHT, skip_nonfinite: bool = False
) -> list[float]:
    """The running EMA of a per-epoch sequence, ``ema_update`` applied in order."""
    smoothed_values, average = [], math.nan if skip_nonfinite else None
    for value in values:
        average = ema_update(average, value, weight, skip_nonfinite)
        smoothed_values.append(average)
    return smoothed_values


def as_logged(value: float) -> float:
    """The value as one epoch-level ``self.log`` call holds it, a float32."""
    return float(np.float32(value))


def smoothed(
    values: Sequence[float], weight: float = EMA_WEIGHT, skip_nonfinite: bool = False
) -> list[float]:
    """The logged EMA of a per-epoch sequence, the values ModelCheckpoint compares."""
    return [as_logged(value) for value in ema(values, weight, skip_nonfinite)]


def best_epoch(values: Sequence[float], mode: str) -> int:
    """Index that ModelCheckpoint(save_top_k=1) keeps for a monitored sequence.

    The first value is always kept, a NaN counts as the worst value, and only a strict
    improvement replaces the kept index, so a tie keeps the earlier one.
    """
    worst = -math.inf if mode == "max" else math.inf
    best_index, best = 0, worst if math.isnan(values[0]) else values[0]
    for index, value in enumerate(values[1:], start=1):
        if (value > best) if mode == "max" else (value < best):
            best_index, best = index, value
    return best_index


def _goodness(values: Sequence[float], mode: str) -> np.ndarray:
    goodness = np.asarray(values, dtype=np.float64)
    goodness = goodness if mode == "max" else -goodness
    return np.where(np.isnan(goodness), -np.inf, goodness)


def ranks(values: Sequence[float], mode: str) -> np.ndarray:
    """Average ranks, 1 for the best value in the direction ``mode``, a NaN the
    worst."""
    return rankdata(-_goodness(values, mode), method="average")


def _doubled_ranks(values: Sequence[float], mode: str) -> np.ndarray:
    # Average ranks are multiples of 1/2; doubled they are integers.
    return (2 * ranks(values, mode)).astype(np.int64)


def pick(
    q_main: Sequence[float],
    q_sec: Sequence[float],
    k: int,
    mode_main: str,
    mode_sec: str = "min",
) -> int:
    """Index minimising k rank(Q') + (W_STEPS - k) rank(Q''), with average ranks (1 for
    the best, a NaN the worst); ties go to the better Q', then to the lower index."""
    if not 0 <= k <= W_STEPS:
        raise ValueError(f"k must lie in [0, {W_STEPS}], got {k}.")
    total = k * _doubled_ranks(q_main, mode_main)
    total = total + (W_STEPS - k) * _doubled_ranks(q_sec, mode_sec)
    good_main = _goodness(q_main, mode_main)
    # lexsort orders by its last key first.
    return int(np.lexsort((np.arange(len(total)), -good_main, total))[0])


def epoch_pick(
    q_main: Sequence[float],
    q_sec: Sequence[float],
    k: int,
    mode_main: str,
    skip_nonfinite: bool = False,
) -> int:
    """The epoch rule: ``pick`` on the logged EMAs of Q', averaged as its callback
    averages it, and of Q'' (minimised)."""
    q_main = smoothed(q_main, skip_nonfinite=skip_nonfinite)
    return pick(q_main, smoothed(q_sec), k, mode_main)
