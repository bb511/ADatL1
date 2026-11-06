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
"""

import math

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
