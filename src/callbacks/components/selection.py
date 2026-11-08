"""The epoch rule and the hyperpoint rule of the selection protocol.

A strategy picks among candidates by two quantities, its own Q' and the fit Q'' of
validation normal (the module's ``fit/full`` averaged over its rows), minimising

    w rank(Q') + (1 - w) rank(Q''),    w = k / W_STEPS, k = 0, ..., W_STEPS,

ties to the better Q', then to the earlier candidate. Over the epochs of a run both
quantities are first smoothed by the exponential moving average of the training
callbacks and rounded to float32, as Lightning logs them; at k = W_STEPS the pick is then
the epoch that ModelCheckpoint(save_top_k=1) keeps for Q', which ``best_epoch`` gives.

A banded strategy (``ancap_v3_spec.md``, section 2.3) first keeps the candidates whose
level, a second maximised quantity, lies within band_se of its standard errors of the
best level, and applies the rule among them alone; at k = W_STEPS Q' then decides only
differences beyond band_se of its own standard errors, and the level decides within
them. Over the epochs the level and Q' are smoothed as Q' is above, and the standard
errors are not. ``epoch_band_pick`` is its epoch rule and ``banded_pick`` its rule over
candidates as given, which the hyperpoint stage applies to every run at its epoch.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class BandPick:
    """A pick of the band rule and what is reported beside it: the number of eligible
    candidates and, at k = W_STEPS, the size of the top group and whether the level
    decided, that is whether the pick is other than the eligible candidate of best Q';
    the last two are None at k < W_STEPS."""

    index: int
    eligible: int
    top: int | None
    level_decided: bool | None


def band_columns(spec: dict) -> list[str]:
    """The val.csv columns that a banded registry entry reads: Q', the level and the two
    standard errors."""
    band = spec["band"]
    return [spec["column"], band["level"], band["level_se"], band["se"]]


def _floats(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def band(
    level: Sequence[float], level_se: Sequence[float], band_se: float
) -> np.ndarray:
    """Indices, ascending, of the candidates of finite level at least the best level
    less band_se of its standard errors.

    The best is the first candidate of maximal level, a NaN the worst, and is always
    among them, so that the band is never empty.
    """
    level = _floats(level)
    best = best_epoch(level, "max")
    bound = level[best] - band_se * _floats(level_se)[best]
    within = np.flatnonzero(np.isfinite(level) & (level >= bound))
    return np.union1d(within, [best]).astype(np.int64)


def _top_group(q_main: np.ndarray, q_main_se: np.ndarray, band_se: float, mode: str):
    """The indices of the candidates whose Q' lies within band_se standard errors of the
    best Q', and the index of the best: the first of best Q', a NaN the worst, always in
    the group."""
    goodness, best = _goodness(q_main, mode), best_epoch(q_main, mode)
    bound = goodness[best] - band_se * q_main_se[best]
    return np.union1d(np.flatnonzero(goodness >= bound), [best]), best


def _flat_pick(level, q_main, q_main_se, band_se: float, mode: str) -> tuple:
    """(index, size of the top group, whether the level decided): the top group's
    candidate of highest level, ties to the better Q', then to the lower index."""
    top, best = _top_group(q_main, q_main_se, band_se, mode)
    # lexsort orders by its last key first.
    keys = (top, -_goodness(q_main[top], mode), -_goodness(level[top], "max"))
    index = int(top[np.lexsort(keys)[0]])
    return index, len(top), index != best


def banded_pick(
    level: Sequence[float],
    level_se: Sequence[float],
    q_main: Sequence[float],
    q_main_se: Sequence[float],
    q_sec: Sequence[float],
    k: int,
    band_se: float,
    mode: str = "max",
) -> BandPick:
    """The band rule over the candidates as given (ancap_v3_spec.md, section 2.3).

    Among the candidates that ``band`` keeps, the pick at k = W_STEPS is the top
    group's candidate of highest level, and at k < W_STEPS that of ``pick`` of Q' and
    Q''. The level is maximised, Q' in the direction ``mode``.
    """
    eligible, values = band(level, level_se, band_se), (level, q_main, q_main_se, q_sec)
    level, q_main, q_main_se, q_sec = (_floats(v)[eligible] for v in values)
    if k == W_STEPS:
        index, top, decided = _flat_pick(level, q_main, q_main_se, band_se, mode)
        return BandPick(int(eligible[index]), len(eligible), top, decided)
    index = pick(q_main, q_sec, k, mode)
    return BandPick(int(eligible[index]), len(eligible), None, None)


def epoch_band_pick(
    level: Sequence[float],
    level_se: Sequence[float],
    q_main: Sequence[float],
    q_main_se: Sequence[float],
    q_sec: Sequence[float],
    k: int,
    band_se: float,
    mode: str = "max",
    skip_nonfinite: bool = False,
) -> BandPick:
    """The epoch rule of a banded strategy: ``banded_pick`` on the logged EMAs of the
    level and Q', both averaged as Q' is, and of Q''; the standard errors are taken
    as they are."""
    level = smoothed(level, skip_nonfinite=skip_nonfinite)
    q_main = smoothed(q_main, skip_nonfinite=skip_nonfinite)
    q_sec = smoothed(q_sec)
    return banded_pick(level, level_se, q_main, q_main_se, q_sec, k, band_se, mode)
