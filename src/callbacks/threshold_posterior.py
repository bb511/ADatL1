"""ANCAP, CAP for anomaly detection: the capacity of a detector's threshold posterior,
as functions of scores and as a training callback.

A detector proposes one region per threshold, and the cost of a threshold is the typical
events it rejects plus the reference points it accepts. One model's scores on two
typical views, s_1 (X1) and s_2 (X2), and on a reference P define, on a grid T of
thresholds::

    T        = {u_(floor(j (n - 1) / L)) : j = 0, ..., L}, u the n pooled scores of the
               two views in ascending order, duplicates kept: the 'lower' order
               statistics of torch.quantile at the levels j / L; P is not pooled
    R_d(tau) = mean(s_d > tau) + mean(s_P < tau), d = 1, 2, both inequalities strict
    p_d(tau) = exp(-beta R_d(tau)) / sum over T
    A(beta)  = log(|T| sum_T p_1 p_2), zero for two flat posteriors, at most log|T|

P is the set of SDS pseudo-outliers (``src/callbacks/sds.py``): built from the training
inputs alone, it sits at the edge of the typical data and states no signal model.

The capacity is the supremum over the resolution beta of the expected log agreement of
the posteriors of two instances of the data (Buhmann, Dumazert, Gronskiy and
Szpankowski, Theor. Comput. Sci. 745 (2018), Definition 3). The instances are n_pairs
bootstrap pairs: each view is redrawn with replacement at its own size and takes its own
redraw of P at P's size, and the thresholds of the observed pair serve every pair::

    Abar(beta)  = mean over the pairs of A(beta), on n_beta log-spaced betas
    I           = max over beta of Abar, in nats, at most log|T|
    beta_star   = the smallest beta whose Abar lies within TIE_TOLERANCE of I, so that
                  rounding cannot settle it: an output, never declared
    I_0         = I on pairs of its own, both views redrawn from the pooled typical
                  sample at their own sizes, as if the two views came from one source
    confidence  = I_0
    consistency = I - I_0
    saturated   = 1.0 if beta_star >= min(n_1, n_2, n_P), else 0.0

The capacity sees neither the direction nor the level of the separation. The direction
check reads them on view 1 and P alone, so it does not depend on the grade::

    depth = 1 - min of R_1(tau) over every real tau
    t     = (depth - mean of the null's depths) / their standard deviation, ddof 1

The content null reassigns the reference label at random within the pooled scores of
view 1 and P, counts kept. A null of zero spread gives NaN (0/0).

T and every R read the scores through their order alone, so that every output is
unchanged by one strictly increasing map applied to the sets. A set that is empty or
holds a non-finite score makes NaN of every column that reads it, and of no other.

A redraw is one multinomial draw of the counts in the len(T) + 1 cells that T cuts the
line into. Those counts fix R and have the law of the cell counts of a redraw of the
rows, at a cost independent of the number of rows.

Seed rule. Three streams ``numpy.random.default_rng([seed, k])`` are created afresh at
every call: k = 0 draws the pairs of I, k = 1 the pairs of I_0 and k = 2 the content
null. In streams 0 and 1 the draws come in the order view 1, the reference for view 1,
view 2, the reference for view 2, each of size n_pairs. Stream 2 makes n_draws
sequential ``rng.permutation(n_1 + n_P)`` over cat(view 1, P), the first n_1 entries
being view 1's relabelled scores and the rest the reference's. Every column is therefore
a pure function of the scores and the configuration, and no grade depends on another.
Since every candidate shares these draws, the error of a finite n_pairs or n_draws is
common to the candidates and does not average out over them: it enters a comparison with
an absolute bar in full. Everything is computed in float64 on the CPU.

The entry points on plain scores, which the callbacks call and the test bed reads
offline, are ``capacity`` (one grade), ``direction``, ``depth`` and ``ancap`` (every
column of a set of grades).

ANCAP-W, the capacity at the working point (``scratchpad/adl1t/theoryx/ancap_wp_spec.md``
and its rulings), poses the learning problem as the labelling of the reference points,
flagged or not, at the working rate alpha. View d sets its own cut and view 1 the scale::

    tau_d   = Q_{1 - alpha}(s_d), torch.quantile's linear interpolation
    scale   = Q_0.75(s_1) - Q_0.25(s_1)
    m_i^d   = (s_i - tau_d) / scale, i over P
    p_i^d   = sigmoid(beta m_i^d + c), c = logit(alpha): the prior at beta = 0
    a_i     = log[p_i^1 p_i^2 / alpha + (1 - p_i^1)(1 - p_i^2) / (1 - alpha)]
            = sp(beta (m_i^1 + m_i^2) + c) - sp(beta m_i^1 + c) - sp(beta m_i^2 + c)
              - log(1 - alpha), sp the softplus
    A(beta) = mean over P of a_i, zero for the prior and at most log(1 / alpha)

The bootstrap pairs redraw view 1, view 2 and P, P once per pair for both views; each
pair takes its cuts and scale from its redrawn views and its margins over its redrawn P.
I_W and beta_star follow from the mean curve as I and beta_star do. I_0 takes both cuts
from redraws of the pooled typical sample at sizes n_1 and n_2 and the scale from a redraw
of view 1 itself. Beside them stand the consistency I_W - I_0, the levels at the observed
cuts f_1 = mean(s_P > tau_1) and f_2 = mean(s_P > max(tau_1, tau_2)), and a mark where
beta_star is the grid's last point. An observed scale of zero makes NaN of every column.

Three more columns read the draws of I_W and add none (``ancap_v3_spec.md``, section
2.2, and ruling 1 of its rulings)::

    f1_se = standard deviation, ddof 1, over the pairs of stream 0 of each pair's f_1,
            its redrawn P's share above its redrawn tau_1; one per call, as f_1
    se    = standard deviation, ddof 1, over the pairs of each pair's A(beta_star),
            beta_star that of I_W
    r_2   = mean(s_2 > tau_1), view 2's share above view 1's observed float64 cut; NaN
            where tau_1 lies below view 2's smallest kept value

View 2 and the pool enter through their largest ceil(n / 20) values alone (``Tail``),
whether a set arrives whole or as a quantile record's kept tail, so that both give the
same bits. A redraw of n_s values from a source of n sorted values draws n_s indices
uniformly in [0, n) and counts them per cell: one cell per kept value and one lump cell
below for the rest. A cut whose interpolation reaches the lump is NaN, and so is any
column that reads it. The pool keeps view 1's values at least view 2's smallest kept value
and view 2's kept values; above that value its order is exact.

Seed rule of ANCAP-W. Two streams ``numpy.random.default_rng([seed, k])`` are created
afresh for every grade: in k = 0 (the pairs of I_W) the redraws of view 1, P and view 2,
in k = 1 (the pairs of I_0) those of view 1, P, the pool at size n_1 and the pool at size
n_2, each one ``rng.integers`` of shape [n_pairs, n_s]. View 1 and P come first in both
streams, so that their draws are common to the grades; ``ancap_w`` draws them once and
restores the stream's state for each grade, which gives the bits of fresh streams.
"""

import math
from typing import NamedTuple

import numpy as np
import torch
from pytorch_lightning import Callback
from torch import Tensor

from src.callbacks.components.selection import EMA_WEIGHT, ema_update
from src.callbacks.components.utils import placeholder_loader
from src.callbacks.score_dump import score_in_eval_mode
from src.callbacks.sds import sds_inputs
from src.data.utils import unpack_batch

# The reference's name among the sets of the SDS pseudo-sets.
REFERENCE = "sds_outliers"
# The keys of capacity, in order, with the prefix of each one's column in ancap.
COLUMNS = {
    "capacity": "ancap",
    "beta_star": "ancap_beta",
    "confidence": "ancap_conf",
    "consistency": "ancap_cons",
    "saturated": "ancap_sat",
}
KEYS = tuple(COLUMNS)
# The streams of the seed rule.
PAIRS_STREAM, POOLED_STREAM, NULL_STREAM = 0, 1, 2
# The pairs of one chunk: an array [pairs, n_beta, len(T)] holds about 8 MB at the
# registered sizes.
PAIR_CHUNK = 8
# In float64 exp is exactly 0 below this exponent (the last non-zero value is at
# -745.13), and torch's exp takes about ten times as long there (12 against 1 ns per
# element on olgpu), whereas a third of the exponents of a curve lie below it.
UNDERFLOW = -746.0
# Points of the mean curve this close, in nats, are tied: on a plateau flat to rounding
# the arithmetic alone would decide an exact maximum, and so beta_star.
TIE_TOLERANCE = 1e-9
# The columns of ancap_w per grade, by key of _grade_w.
W_COLUMNS = {
    "capacity": "ancapw",
    "beta_star": "ancapw_beta",
    "confidence": "ancapw_conf",
    "consistency": "ancapw_cons",
    "f2": "ancapw_f2",
    "end": "ancapw_end",
    "se": "ancapw_se",
    "r2": "ancapw_r2",
}
W_KEYS = tuple(W_COLUMNS)
# The columns of ancap_w that read no grade.
W_LEVELS = ("ancapw_f1", "ancapw_f1_se")
# The pairs of one call of the agreement sums: each temporary [pairs, n_beta, n_P] holds
# about 23 MB at the light tunnel's sizes. A caller that compiles the sums may raise it.
W_PAIR_CHUNK = 2
LOG_KWARGS = dict(
    prog_bar=False,
    on_step=False,
    on_epoch=True,
    logger=True,
    sync_dist=True,
    add_dataloader_idx=False,
)


def _as_scores(scores) -> Tensor:
    return torch.as_tensor(scores).detach().reshape(-1).to("cpu", torch.float64)


def _usable(*sets: Tensor) -> bool:
    return all(len(s) > 0 and bool(torch.isfinite(s).all()) for s in sets)


def _stream(config, k: int) -> np.random.Generator:
    return np.random.default_rng([config["seed"], k])


def grid_positions(n: int, levels: int) -> Tensor:
    """Return floor(j (n - 1) / levels) for j = 0, ..., levels.

    These are the positions, in n ascending values, of the 'lower' order statistics of
    torch.quantile at the levels j / levels, computed in integers so that no rounding
    of j / levels moves one.
    """
    return torch.arange(levels + 1) * (n - 1) // levels


def threshold_grid(scores_1, scores_2, levels: int) -> Tensor:
    """Return the levels + 1 thresholds of the pooled scores of the two views."""
    pooled = torch.cat([_as_scores(scores_1), _as_scores(scores_2)]).sort().values
    return pooled[grid_positions(len(pooled), levels)]


def costs(scores, reference, grid: Tensor) -> Tensor:
    """Return R(tau) = mean(scores > tau) + mean(reference < tau) at every tau of
    grid."""
    scores = _as_scores(scores).sort().values
    reference = _as_scores(reference).sort().values
    above = len(scores) - torch.searchsorted(scores, grid, right=True)
    below = torch.searchsorted(reference, grid)
    return above.double() / len(scores) + below.double() / len(reference)


def beta_grid(config) -> Tensor:
    """Return the n_beta log-spaced betas of the curve."""
    low, high = math.log10(config["beta_min"]), math.log10(config["beta_max"])
    return torch.logspace(low, high, config["n_beta"], dtype=torch.float64)


def _cells(scores: Tensor, grid: Tensor, right: bool) -> np.ndarray:
    """Return the number of scores in each of the len(grid) + 1 cells of grid.

    The first j + 1 cells hold the scores at most tau_j with ``right`` False and the
    scores below tau_j with it True: the two counts of R, whatever the ties and the
    duplicated thresholds.
    """
    cell = torch.searchsorted(grid, scores, right=right)
    return torch.bincount(cell, minlength=len(grid) + 1).numpy()


def _cell_costs(typical: np.ndarray, reference: np.ndarray) -> Tensor:
    """Return R [B, len(T)] of the redrawn cell counts of the typical and reference
    sets, each [B, len(T) + 1].

    The arithmetic is that of ``costs``, so that the two agree to the last bit.
    """
    n, n_reference = int(typical[0].sum()), int(reference[0].sum())
    above = n - torch.from_numpy(typical).cumsum(dim=1)[:, :-1]
    below = torch.from_numpy(reference).cumsum(dim=1)[:, :-1]
    return above.double() / n + below.double() / n_reference


def _redrawn_costs(
    rng: np.random.Generator,
    sources: list[np.ndarray],
    sizes: list[int],
    reference: np.ndarray,
    n_pairs: int,
) -> list[Tensor]:
    """Return R [n_pairs, len(T)] of each view, redrawn at sizes from the cell counts of
    sources, each view with its own redraw of the reference's cell counts."""
    n_reference, drawn = int(reference.sum()), []
    for cells, n in zip(sources, sizes):
        typical = rng.multinomial(n, cells / cells.sum(), size=n_pairs)
        references = rng.multinomial(n_reference, reference / n_reference, size=n_pairs)
        drawn.append(_cell_costs(typical, references))
    return drawn


def _from_zero(cost: Tensor) -> Tensor:
    return cost - cost.amin(dim=-1, keepdim=True)


def _log_partition(cost: Tensor, betas: Tensor) -> Tensor:
    """Return log sum_T exp(-beta R) [..., n_beta] of a cost [..., len(T)] with a zero
    in every row, so that no exponent is positive and the sum is at least one.

    An exponent below ``UNDERFLOW`` sits at 0 during the exp and is zeroed after it,
    which gives the bits of the plain sum at a third of its cost.
    """
    exponents = cost.unsqueeze(-2) * -betas.unsqueeze(-1)
    underflowing = exponents < UNDERFLOW
    exponents.masked_fill_(underflowing, 0.0).exp_().masked_fill_(underflowing, 0.0)
    return exponents.sum(dim=-1).log_()


def agreement(cost_1: Tensor, cost_2: Tensor, betas: Tensor) -> Tensor:
    """Return A(beta) [..., n_beta] of the posteriors of two costs over T, their last
    axis.

    A = log|T| + z_12 - z_1 - z_2, with z = log sum_T exp(-beta R) and z_12 on
    R_1 + R_2. A posterior keeps its value when its cost loses a constant, so that each
    cost enters from zero and exactly flat posteriors give exactly zero. The minimum of
    their sum is taken out by hand: disjoint minimisers leave it above zero, and at
    beta = 1e6 every term of z_12 would underflow.

    Args:
        cost_1: R_1 [..., len(T)].
        cost_2: R_2 [..., len(T)].
        betas: The betas [n_beta].
    """
    cost_1, cost_2 = _from_zero(cost_1), _from_zero(cost_2)
    joint = cost_1 + cost_2
    floor = joint.amin(dim=-1, keepdim=True)
    z_12 = _log_partition(joint - floor, betas) - betas * floor
    z_1, z_2 = _log_partition(cost_1, betas), _log_partition(cost_2, betas)
    return math.log(cost_1.shape[-1]) + z_12 - z_1 - z_2


def _mean_curve(cost_1: Tensor, cost_2: Tensor, betas: Tensor) -> Tensor:
    """Return the mean over the pairs [n_pairs] of each pair's log agreement, [n_beta].

    This is the expectation of the log, not the log of the mean agreement.
    """
    chunks = zip(cost_1.split(PAIR_CHUNK), cost_2.split(PAIR_CHUNK))
    return torch.cat([agreement(*chunk, betas) for chunk in chunks]).mean(dim=0)


def _first_maximum(curve: Tensor, betas: Tensor) -> tuple[float, float]:
    """Return the maximum of curve and the smallest beta within ``TIE_TOLERANCE`` of it.

    A saturated curve rises to its limit in steps that shrink with exp(-beta d), d the
    smallest step of the costs: they fall below the tolerance near beta = 21 / d and
    below the resolution of float64 near beta = 36 / d. Its beta_star marks the start of
    that plateau, not the end of the grid.
    """
    maximum = curve.max()
    best = int((curve >= maximum - TIE_TOLERANCE).int().argmax())
    return maximum.item(), betas[best].item()


def _capacity(
    rng: np.random.Generator,
    sources: list[np.ndarray],
    sizes: list[int],
    reference: np.ndarray,
    config,
) -> tuple[float, float]:
    """Return the maximum of the mean curve of one stream's pairs, and its beta."""
    betas = beta_grid(config)
    cost_1, cost_2 = _redrawn_costs(rng, sources, sizes, reference, config["n_pairs"])
    return _first_maximum(_mean_curve(cost_1, cost_2, betas), betas)


def _capacities(sets: list[Tensor], config) -> tuple[float, float, float]:
    """Return I, beta_star and I_0 of the three sets, each usable."""
    grid = threshold_grid(sets[0], sets[1], config["levels"])
    cells = [_cells(s, grid, right=False) for s in sets[:2]]
    args = ([len(s) for s in sets[:2]], _cells(sets[2], grid, right=True), config)
    own, beta_star = _capacity(_stream(config, PAIRS_STREAM), cells, *args)
    pooled = [cells[0] + cells[1]] * 2
    return own, beta_star, _capacity(_stream(config, POOLED_STREAM), pooled, *args)[0]


def capacity(scores_1, scores_2, reference, config) -> dict[str, float]:
    """Return the capacity of one grade and what is reported beside it.

    The keys are those of ``KEYS``. If one of the three sets is empty or holds a
    non-finite score, every value is NaN.

    Args:
        scores_1: Scores of X1 [n_1], as anything ``torch.as_tensor`` accepts.
        scores_2: Scores of X2 [n_2].
        reference: Scores of the reference [n_P].
        config: The posterior's configuration, as in ``configs/cap/posterior.yaml``.
    """
    sets = [_as_scores(s) for s in (scores_1, scores_2, reference)]
    if not _usable(*sets):
        return dict.fromkeys(KEYS, math.nan)
    own, beta_star, confidence = _capacities(sets, config)
    saturated = float(beta_star >= min(len(s) for s in sets))
    return dict(zip(KEYS, (own, beta_star, confidence, own - confidence, saturated)))


def depth(scores, reference) -> float:
    """Return 1 - min R(tau) over every real tau, for finite, non-empty sets.

    A tie between the two sets counts as no error at that tau, by the strict
    inequalities. The pooled values suffice: above a pooled value v and below the next
    one, R has lost no score above v and may have gained a reference point below, so
    that R(v) is no larger; and beyond either end R is 1, no less than at the end.
    """
    scores, reference = _as_scores(scores), _as_scores(reference)
    points = torch.cat([scores, reference])
    return 1.0 - costs(scores, reference, points).min().item()


def _ranked(pool: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return the position of every value of pool in ascending order [N], and per
    position the number of values below it and at most it [N]."""
    ordered, order = torch.sort(pool, stable=True)
    position = torch.empty_like(order)
    position[order] = torch.arange(len(pool))
    below = torch.searchsorted(ordered, ordered)
    return position, below, torch.searchsorted(ordered, ordered, right=True)


def _relabelled_depth(ranked: tuple, view: np.ndarray, n_reference: int) -> float:
    """Return ``depth`` of a relabelled pool, the indices view [n] being the view's.

    R at every pooled value is read off a running count of the view's positions, which
    spares two sorts per draw; the arithmetic is that of ``costs``, so that the two
    agree to the last bit.
    """
    position, below, at_most = ranked
    hits = torch.zeros(len(position) + 1, dtype=torch.int64)
    hits[position[view] + 1] = 1
    # count[i] is the number of the view's values among the i smallest of the pool.
    count = hits.cumsum(dim=0)
    above, under = len(view) - count[at_most], below - count[below]
    cost = above.double() / len(view) + under.double() / n_reference
    return 1.0 - cost.min().item()


def _null_depths(scores: Tensor, reference: Tensor, config) -> Tensor:
    """Return the depth of each draw of the content null, [n_draws]."""
    rng, n, n_reference = _stream(config, NULL_STREAM), len(scores), len(reference)
    ranked = _ranked(torch.cat([scores, reference]))
    drawn = (rng.permutation(n + n_reference) for _ in range(config["n_draws"]))
    depths = [_relabelled_depth(ranked, perm[:n], n_reference) for perm in drawn]
    return torch.tensor(depths, dtype=torch.float64)


def direction(scores_1, reference, config) -> float:
    """Return t, the depth of view 1 against the content null in units of its spread.

    The value is NaN if either set is empty or holds a non-finite score, and for a null
    of zero spread.

    Args:
        scores_1: Scores of X1 [n_1], as anything ``torch.as_tensor`` accepts.
        reference: Scores of the reference [n_P].
        config: The posterior's configuration, as in ``configs/cap/posterior.yaml``.
    """
    scores, reference = _as_scores(scores_1), _as_scores(reference)
    if not _usable(scores, reference):
        return math.nan
    draws = _null_depths(scores, reference, config)
    # Tensors, since a Python float raises on the division by a zero spread.
    return ((depth(scores, reference) - draws.mean()) / draws.std()).item()


def ancap(scores_1, views: dict, reference, config) -> dict[str, float]:
    """Return every column of a set of grades.

    The keys are ``ancap_<grade>`` (the capacity), ``ancap_beta_<grade>``,
    ``ancap_conf_<grade>``, ``ancap_cons_<grade>``, ``ancap_sat_<grade>`` and
    ``ancap_dir``. A grade's columns read view 1, its own view and the reference;
    ``ancap_dir`` reads view 1 and the reference.

    Args:
        scores_1: Scores of X1 [n_1], as anything ``torch.as_tensor`` accepts.
        views: The scores of X2 of each grade, by grade.
        reference: Scores of the reference [n_P].
        config: The posterior's configuration, as in ``configs/cap/posterior.yaml``.
    """
    values = {}
    for grade, scores_2 in views.items():
        stats = capacity(scores_1, scores_2, reference, config)
        values |= {f"{COLUMNS[key]}_{grade}": stats[key] for key in KEYS}
    return values | {"ancap_dir": direction(scores_1, reference, config)}


class Tail(NamedTuple):
    """The largest ``kept_size(n)`` scores of a set of n, in any order, and n: what a
    quantile record keeps of a view."""

    top: Tensor
    n: int


class _Redraws(NamedTuple):
    """The cuts and scale of each pair [n_pairs] and its redrawn P's counts [n_pairs,
    n_P], P ascending."""

    tau_1: Tensor
    tau_2: Tensor
    scale: Tensor
    weights: Tensor


class _Shared(NamedTuple):
    """One stream after the redraws of view 1 and P, which every grade shares."""

    rng: np.random.Generator
    state: dict
    tau: Tensor
    scale: Tensor
    weights: Tensor


def kept_size(n: int) -> int:
    """Return ceil(n / 20), in integers, the number of values a tail keeps."""
    return -(-n // 20)


def tail_of(scores) -> Tail:
    """Return the tail of a whole set."""
    ordered = _as_scores(scores).sort().values
    return Tail(ordered[len(ordered) - kept_size(len(ordered)) :], len(ordered))


def _as_tail(view) -> Tail:
    """Return a view as a tail with its kept values ascending; raise on a record that
    keeps other than ``kept_size(n)`` values.

    A whole set with a non-finite score anywhere gives a tail of NaN, so that its grade
    is NaN as in ``capacity``, although its tail alone may be finite.
    """
    if not isinstance(view, Tail):
        whole = _as_scores(view)
        if _usable(whole):
            return tail_of(whole)
        nan = torch.full((kept_size(len(whole)),), math.nan, dtype=torch.float64)
        return Tail(nan, len(whole))
    if len(view.top) != kept_size(int(view.n)):
        raise ValueError(f"A tail of {view.n} keeps {kept_size(int(view.n))} values.")
    return Tail(_as_scores(view.top).sort().values, int(view.n))


def _rank(q: float, n: int) -> tuple[int, int, Tensor]:
    """Return the two positions among n ascending values and the weight of
    torch.quantile's linear interpolation at q, by its own arithmetic."""
    rank = torch.tensor(q, dtype=torch.float64) * (n - 1)
    below = rank.long()
    return int(below), int(rank.ceil().long()), rank - below


def _cell_values(kept: Tensor) -> Tensor:
    """Return the value of each cell: NaN for the lump, then the kept values."""
    return torch.cat([torch.full((1,), math.nan, dtype=torch.float64), kept])


def _observed_quantile(kept: Tensor, n: int, q: float) -> Tensor:
    """Return torch.quantile at q of a set of n whose largest values kept holds
    ascending, NaN where the interpolation reaches below them."""
    low, high, weight = _rank(q, n)
    values, offset = _cell_values(kept), n - len(kept)
    below, above = (values[max(p - offset + 1, 0)] for p in (low, high))
    return torch.lerp(below, above, weight)


def _redraw_cells(
    rng: np.random.Generator, n: int, n_lump: int, size: int, n_pairs: int
) -> np.ndarray:
    """Return the cell counts [n_pairs, n - n_lump + 1] of size indices drawn
    uniformly in [0, n) per pair, cell 0 being the lump of the n_lump smallest
    values."""
    width = n - n_lump + 1
    cells = np.maximum(rng.integers(0, n, size=(n_pairs, size)) - n_lump + 1, 0)
    cells += width * np.arange(n_pairs)[:, None]
    counts = np.bincount(cells.ravel(), minlength=n_pairs * width)
    return counts.reshape(n_pairs, width)


def _redrawn_quantile(counts: np.ndarray, values: Tensor, q: float) -> Tensor:
    """Return torch.quantile at q of each redraw [n_pairs] from its cell counts."""
    cumulative = torch.from_numpy(counts.cumsum(axis=1))
    low, high, weight = _rank(q, int(counts[0].sum()))
    positions = torch.tensor([low, high]).expand(len(counts), 2).contiguous()
    cells = torch.searchsorted(cumulative, positions, right=True)
    return torch.lerp(values[cells[:, 0]], values[cells[:, 1]], weight)


def _shared(config, k: int, view_1: Tensor, n_reference: int) -> _Shared:
    """Return stream k after its redraws of view 1, ascending, and of P.

    A redrawn scale of zero is NaN, as the observed one makes every column NaN.
    """
    rng, n_1, n_pairs = _stream(config, k), len(view_1), config["n_pairs"]
    counts = _redraw_cells(rng, n_1, 0, n_1, n_pairs)
    weights = _redraw_cells(rng, n_reference, 0, n_reference, n_pairs)[:, 1:]
    values = _cell_values(view_1)
    tau, upper, lower = (
        _redrawn_quantile(counts, values, q)
        for q in (1.0 - config["alpha"], 0.75, 0.25)
    )
    scale = torch.where(upper > lower, upper - lower, math.nan)
    weights = torch.from_numpy(weights).double()
    return _Shared(rng, rng.bit_generator.state, tau, scale, weights)


def _restored(shared: _Shared) -> np.random.Generator:
    shared.rng.bit_generator.state = shared.state
    return shared.rng


def _tail_cut(rng: np.random.Generator, tail: Tail, size: int, config) -> Tensor:
    """Return the cut of each redraw at size from a tail [n_pairs]."""
    n_lump = tail.n - len(tail.top)
    counts = _redraw_cells(rng, tail.n, n_lump, size, config["n_pairs"])
    return _redrawn_quantile(counts, _cell_values(tail.top), 1.0 - config["alpha"])


def _pool(view_1: Tensor, tail: Tail) -> Tail:
    """Return the pooled typical sample above view 2's lump, ascending."""
    above = view_1[view_1 >= tail.top[0]]
    return Tail(torch.cat([above, tail.top]).sort().values, len(view_1) + tail.n)


def _log1p_exp(x: Tensor) -> Tensor:
    return torch.log1p(torch.exp(-x.abs()))


def _agreement_sums(m_1, m_2, weights, betas, alpha: float) -> Tensor:
    """Return the sum over P of w_i a_i(beta) [pairs, n_beta], of margins and weights
    [pairs, n_P] and betas [n_beta].

    sp(x) = max(x, 0) + log(1 + exp(-abs(x))), and the three max terms are combined
    by case: where both points are flagged (x >= 0) they give -c exactly, which spares
    the cancellation of terms of order beta abs(m) at large beta.
    """
    c = math.log(alpha / (1.0 - alpha))
    u, v = betas[:, None] * m_1[:, None, :], betas[:, None] * m_2[:, None, :]
    x_1, x_2, x_12 = u + c, v + c, u + v + c
    linear = torch.where(
        x_1 >= 0,
        torch.where(x_2 >= 0, -c, torch.maximum(v, -x_1)),
        torch.where(x_2 >= 0, torch.maximum(u, -x_2), x_12.clamp_min(0.0)),
    )
    curved = _log1p_exp(x_12) - _log1p_exp(x_1) - _log1p_exp(x_2)
    return ((linear + curved - math.log1p(-alpha)) * weights[:, None, :]).sum(dim=-1)


def _chunk_sums(tau_1, tau_2, scale, weights, reference, betas, alpha) -> Tensor:
    """Return the agreement sums of a chunk of pairs, on the reference's device."""
    on = [t.to(reference.device) for t in (tau_1, tau_2, scale, weights)]
    m_1 = (reference - on[0][:, None]) / on[2][:, None]
    m_2 = (reference - on[1][:, None]) / on[2][:, None]
    return _agreement_sums(m_1, m_2, on[3], betas, alpha)


def _sample_sd(values: Tensor) -> float:
    """Return the standard deviation, ddof 1, of values [n_pairs], by numpy."""
    return float(np.std(values.cpu().numpy(), ddof=1))


def _pair_curves(redraws: _Redraws, reference: Tensor, config, device) -> Tensor:
    """Return each pair's mean over its redrawn P of a_i(beta) [n_pairs, n_beta], on
    device."""
    betas, on = beta_grid(config).to(device), reference.to(device)
    chunks = zip(*(t.split(W_PAIR_CHUNK) for t in redraws))
    sums = [_chunk_sums(*chunk, on, betas, config["alpha"]) for chunk in chunks]
    return torch.cat(sums) / len(reference)


def _capacity_w(redraws: _Redraws, reference: Tensor, config, device) -> tuple:
    """Return the maximum of the mean curve over the pairs, its beta and the standard
    deviation over the pairs of their curves at that beta, all three NaN where the mean
    curve holds a non-finite value."""
    curves = _pair_curves(redraws, reference, config, device)
    curve = curves.mean(dim=0).cpu()
    if not bool(torch.isfinite(curve).all()):
        return math.nan, math.nan, math.nan
    betas = beta_grid(config)
    capacity, beta_star = _first_maximum(curve, betas)
    at_star = curves[:, int(torch.nonzero(betas == beta_star)[0, 0])]
    return capacity, beta_star, _sample_sd(at_star)


def _share_above(tail: Tail, cut: Tensor) -> float:
    """Return the share of a tail's set strictly above cut, NaN where cut lies below the
    smallest kept value.

    At or above it the share is exact, since no value of the lump exceeds that value.
    """
    if not bool(cut >= tail.top[0]):
        return math.nan
    return (tail.top > cut).sum().item() / tail.n


def _levels(view_1: Tensor, tail: Tail, reference: Tensor, alpha: float) -> tuple:
    """Return f_1 and f_2 at the observed cuts, and r_2."""
    tau_1 = _observed_quantile(view_1, len(view_1), 1.0 - alpha)
    tau_2 = _observed_quantile(tail.top, tail.n, 1.0 - alpha)
    f_1, f_2 = (reference > tau_1).double().mean().item(), math.nan
    if not torch.isnan(tau_2):
        f_2 = (reference > torch.maximum(tau_1, tau_2)).double().mean().item()
    return f_1, f_2, _share_above(tail, tau_1)


def _level_columns(view_1, reference, own: _Shared, alpha: float) -> dict[str, float]:
    """Return f_1 and f1_se, the spread of each pair's f_1 in stream ``own``."""
    tau_1 = _observed_quantile(view_1, len(view_1), 1.0 - alpha)
    above = (reference > own.tau[:, None]).double()
    shares = (own.weights * above).sum(dim=1) / len(reference)
    f_1 = (reference > tau_1).double().mean().item()
    return dict(zip(W_LEVELS, (f_1, _sample_sd(shares))))


def _confidence_w(pooled: _Shared, sets: tuple, config, device) -> float:
    """Return I_0: both cuts from the pool, at sizes n_1 and n_2 in that order."""
    view_1, tail, reference = sets
    rng, pool = _restored(pooled), _pool(view_1, tail)
    cuts = [_tail_cut(rng, pool, size, config) for size in (len(view_1), tail.n)]
    redraws = _Redraws(*cuts, pooled.scale, pooled.weights)
    return _capacity_w(redraws, reference, config, device)[0]


def _grade_w(streams: list[_Shared], sets: tuple, config, device) -> dict[str, float]:
    """Return the columns of one grade, by key of ``W_KEYS``."""
    (own, pooled), (view_1, tail, reference) = streams, sets
    tau_2 = _tail_cut(_restored(own), tail, tail.n, config)
    redraws = _Redraws(own.tau, tau_2, own.scale, own.weights)
    capacity, beta_star, se = _capacity_w(redraws, reference, config, device)
    confidence = _confidence_w(pooled, sets, config, device)
    end = float(beta_star == beta_grid(config)[-1].item())
    end = math.nan if math.isnan(beta_star) else end
    f_2, r_2 = _levels(view_1, tail, reference, config["alpha"])[1:]
    stats = (capacity, beta_star, confidence, capacity - confidence, f_2, end, se, r_2)
    return dict(zip(W_KEYS, stats))


def _undefined(view_1: Tensor, reference: Tensor) -> bool:
    """Return whether view 1 or P make every column NaN, the scale of zero included."""
    if not _usable(view_1, reference):
        return True
    n = len(view_1)
    scale = _observed_quantile(view_1, n, 0.75) - _observed_quantile(view_1, n, 0.25)
    return not bool(scale > 0)


def ancap_w(scores_1, views: dict, reference, config, device="cpu") -> dict[str, float]:
    """Return every column of ANCAP-W of a set of grades.

    The keys are ``<prefix>_<grade>`` for each prefix of ``W_COLUMNS`` and those of
    ``W_LEVELS``. A grade whose view is empty or holds a non-finite kept value has NaN
    columns; an unusable view 1 or P, or an observed scale of zero, makes every column
    NaN. The bootstrap draws run on the CPU and the agreement sums on ``device``.

    Args:
        scores_1: Scores of X1 [n_1], as anything ``torch.as_tensor`` accepts.
        views: The scores of X2 of each grade, by grade, each whole or as a ``Tail``.
        reference: Scores of the reference [n_P].
        config: The posterior's configuration, as in ``configs/cap/posterior.yaml``.
        device: Where the agreement sums are computed, in float64.
    """
    view_1 = _as_scores(scores_1).sort().values
    reference = _as_scores(reference).sort().values
    tails = {grade: _as_tail(view) for grade, view in views.items()}
    names = [f"{W_COLUMNS[k]}_{g}" for g in tails for k in W_KEYS] + [*W_LEVELS]
    if _undefined(view_1, reference):
        return dict.fromkeys(names, math.nan)
    n_reference, values = len(reference), dict.fromkeys(names, math.nan)
    streams = [
        _shared(config, k, view_1, n_reference) for k in (PAIRS_STREAM, POOLED_STREAM)
    ]
    for grade, tail in tails.items():
        if _usable(tail.top):
            stats = _grade_w(streams, (view_1, tail, reference), config, device)
            values |= {f"{W_COLUMNS[k]}_{grade}": stats[k] for k in W_KEYS}
    return values | _level_columns(view_1, reference, streams[0], config["alpha"])


class ThresholdPosteriorCallback(Callback):
    """Log the moving average of the capacity of every grade.

    The scores of dataset_1 and of the views are collected batch by batch at every
    validation epoch after the sanity pass. The reference, the pseudo-outliers of
    ``sds_inputs`` over the training inputs, is built once at fit start as SDSCallback
    builds it and scored at every epoch end in eval mode under forked RNG streams. Every
    epoch logs ``val/summary/ancap_<grade>_ema`` for each grade, averaged as
    CAPBankCallback averages: a non-finite value leaves the average unchanged.

    Args:
        output_name: Key in the outputs dict of the per-event anomaly scores.
        dataset_1: Name of the validation dataloader of X1.
        views: Name of the validation dataloader of X2, by grade.
        posterior_config: The posterior's configuration, as in
            ``configs/cap/posterior.yaml``.
        ema_weight: Weight of the previous value in the exponential moving average.
        max_rows: Largest number of distinct training rows the reference is built from.
        row_seed: Seed of the subset of rows.
        batch_size: Batch size of the pass over the reference.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        views: dict[str, str],
        posterior_config: dict,
        ema_weight: float = EMA_WEIGHT,
        max_rows: int | None = 10000,
        row_seed: int = 0,
        batch_size: int = 1024,
    ):
        super().__init__()
        self.output_name, self.ema_weight = output_name, ema_weight
        self.dataset_1, self.views = dataset_1, dict(views)
        self.datasets = [dataset_1, *self.views.values()]
        self.config, self.max_rows = posterior_config, max_rows
        self.row_seed, self.batch_size = row_seed, batch_size
        self.keys = {grade: f"val/summary/ancap_{grade}_ema" for grade in self.views}

    def on_fit_start(self, trainer, pl_module):
        """Instantiate the moving averages and build the reference's loader."""
        self.emas = dict.fromkeys(self.views, math.nan)
        train = unpack_batch(trainer.datamodule.data_train.tensors)
        inputs = sds_inputs(train.x, self.max_rows, self.row_seed, train.mask)
        mask = inputs.get(f"{REFERENCE}_mask")
        loader = placeholder_loader(inputs[REFERENCE], self.batch_size, mask)
        self.loaders = {REFERENCE: loader}

    def on_validation_epoch_start(self, trainer, pl_module):
        """Check the dataloaders and reset the scores."""
        dset_names = list(trainer.val_dataloaders.keys())
        missing = [name for name in self.datasets if name not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires validation dataloaders {missing}. "
                f"Available validation dataloaders: {dset_names}"
            )
        self.scores = {name: [] for name in self.datasets}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect the scores of the views."""
        if trainer.sanity_checking:
            return
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name in self.scores:
            score = outputs[self.output_name].detach().reshape(-1)
            self.scores[dset_name].append(score.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Score the reference, compute each grade's capacity and log the averages."""
        if trainer.sanity_checking:
            return
        reference = self._reference_scores(trainer, pl_module)
        scores = {name: torch.cat(self.scores[name]) for name in self.datasets}
        for grade, name in self.views.items():
            view_1, view_2 = scores[self.dataset_1], scores[name]
            self._average(grade, capacity(view_1, view_2, reference, self.config))
        logged = {self.keys[grade]: value for grade, value in self.emas.items()}
        pl_module.log_dict(logged, **LOG_KWARGS)

    def _average(self, grade: str, stats: dict[str, float]) -> None:
        self.emas[grade] = ema_update(
            self.emas[grade], stats["capacity"], self.ema_weight, skip_nonfinite=True
        )

    def _reference_scores(self, trainer, pl_module) -> Tensor:
        keys = [self.output_name]
        scored = score_in_eval_mode(trainer, pl_module, self.loaders, keys).scores()
        # An empty pseudo-set serves no batch and so leaves no entry.
        if REFERENCE not in scored:
            return torch.empty(0)
        return scored[REFERENCE][self.output_name]
