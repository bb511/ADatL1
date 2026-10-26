"""The threshold posterior with a probe, track 2 of the CAP test bed, as functions of
scores and as a training callback.

Registered in the test plan 08_tests.md (part C and its amendments of 24 Oct, evening,
and 26 Oct, 02:25). One model's scores on two typical views, s_1 (X1) and s_2 (X2), and
on a probe set P define, on a grid T of thresholds::

    T        = {u_(floor(j (n - 1) / L)) : j = 0, ..., L}, u the n pooled scores of the
               two views in ascending order, duplicates kept: the 'lower' order
               statistics of torch.quantile at the levels j / L
    R_d(tau) = mean(s_d > tau) + mean(s_P < tau), d = 1, 2, both inequalities strict
    p_d(tau) = exp(-beta R_d(tau)) / sum over T
    A(beta)  = log(|T| sum_T p_1 p_2) = C + F, zero for two flat posteriors
    C(beta)  = log cos(p_1, p_2) <= 0, the consistency
    F(beta)  = log(|T| |p_1| |p_2|), the confidence
    A-fix = A(beta_fix); A-sup = max of A over the beta grid, at beta_star;
    D-fix = A-fix - A_pi, A_pi the mean of A over the transfer null.

T and every R_d read the scores through their order alone, so that every output is
unchanged by one strictly increasing map applied to the three sets. Two permutation
nulls, each of n_draws draws, give A, C and F per draw at beta_fix:

- transfer: the two views' scores are pooled and the view label is reassigned with the
  counts kept; T and the probe term are those of the observed sets.
- content: on each view apart, its scores are pooled with the probe's and the probe
  label is reassigned with the counts kept; T is rebuilt from the draw's two typical
  sets.

A draw relabels a pool cat(first, second) by ``torch.randperm``, whose first len(first)
entries make the first set; a content draw takes view 1's permutation, then view 2's.
Each null seeds a generator of its own with ``seed`` at every call, so that every probe,
epoch and run sees the same permutations. The spreads are standard deviations over the
draws with ddof 1, as the CAP bank's sigma_pi. Everything is computed in float64 on the
CPU.

The entry points on plain scores, which the callbacks call and the test bed reads
offline, are ``agreement`` (A, C and F at any betas), ``null_draws`` (A, C and F of
every draw of both nulls) and ``threshold_posterior`` (every statistic at beta_fix);
``half_entropy_beta`` gives the beta of the beta_fix rule.
"""

import math
from collections.abc import Iterator

import torch
from pytorch_lightning import Callback
from torch import Tensor

from src.callbacks.components.selection import EMA_WEIGHT, ema_update
from src.callbacks.components.utils import drawn_masks, placeholder_loader
from src.callbacks.score_dump import aux_inputs, input_moments, score_in_eval_mode
from src.data.utils import unpack_batch

# The keys of threshold_posterior, in order.
KEYS = (
    "A-fix",
    "D-fix",
    "C",
    "F",
    "beta_fix",
    "A_pi",
    "sigma_pi",
    "C_pi",
    "sigma_C_pi",
    "F_pi",
    "sigma_F_pi",
    "A_content",
    "sigma_A_content",
    "C_content",
    "sigma_C_content",
    "F_content",
    "sigma_F_content",
    "A-sup",
    "beta_star",
    "at_bound",
)
# The keys of the mean and of the spread of A, C and F over each null.
TRANSFER_KEYS = {
    "A": ("A_pi", "sigma_pi"),
    "C": ("C_pi", "sigma_C_pi"),
    "F": ("F_pi", "sigma_F_pi"),
}
CONTENT_KEYS = {q: (f"{q}_content", f"sigma_{q}_content") for q in ("A", "C", "F")}
# The statistics whose moving averages the training callback logs.
LOGGED = ("A-fix", "D-fix")
# The entries of one chunk of draws [draws, values]: each int64 array of a chunk holds
# about 8 MB, which bounds the memory of the nulls on the wind tunnel's pools.
CHUNK_ELEMENTS = 2**20


def _as_scores(scores) -> Tensor:
    return torch.as_tensor(scores).detach().reshape(-1).to("cpu", torch.float64)


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


def costs(scores, probe, grid: Tensor) -> Tensor:
    """Return R(tau) = mean(scores > tau) + mean(probe < tau) at every tau of grid."""
    scores, probe = _as_scores(scores).sort().values, _as_scores(probe).sort().values
    above = len(scores) - torch.searchsorted(scores, grid, right=True)
    below = torch.searchsorted(probe, grid)
    return above.double() / len(scores) + below.double() / len(probe)


def beta_grid(config) -> Tensor:
    """Return the grid of A-sup, that of the CAP bank."""
    low, high = math.log10(config["beta_min"]), math.log10(config["beta_max"])
    return torch.logspace(low, high, config["n_beta"], dtype=torch.float64)


def _log_posterior(cost: Tensor, beta) -> Tensor:
    logits = -beta * cost
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def kernel(cost_1: Tensor, cost_2: Tensor, beta) -> dict[str, Tensor]:
    """Return A, C and F of the posteriors of two costs over T, their last axis.

    beta is a float or a tensor that broadcasts against the costs; betas [n_beta, 1]
    against costs [len(T)] give curves [n_beta].
    """
    log_p1, log_p2 = _log_posterior(cost_1, beta), _log_posterior(cost_2, beta)
    log_size = math.log(cost_1.shape[-1])
    overlap = torch.logsumexp(log_p1 + log_p2, dim=-1)
    norms = 0.5 * (torch.logsumexp(2 * log_p1, -1) + torch.logsumexp(2 * log_p2, -1))
    return {"A": log_size + overlap, "C": overlap - norms, "F": log_size + norms}


def entropy(cost: Tensor, beta: float) -> float:
    """Return the entropy in nats of the posterior of a cost at beta."""
    log_p = _log_posterior(cost, beta)
    return -(log_p.exp() * log_p).sum().item()


def _bracket(cost: Tensor, target: float) -> tuple[float, float]:
    """Return beta_low < beta_high with H(beta_low) > target >= H(beta_high), doubling
    or halving from beta = 1."""
    low = high = 1.0
    while entropy(cost, high) > target:
        low, high = high, 2.0 * high
    while entropy(cost, low) <= target:
        low, high = low / 2.0, low
    return low, high


def half_entropy_beta(cost: Tensor, rtol: float = 1e-10) -> float:
    """Return the beta at which the posterior of cost has half the entropy of the flat.

    H(beta) falls from log|T| at beta = 0 (dH/dbeta = -beta Var_p(R)) to the log of the
    number of minimisers of R as beta grows; where that limit is at least (1/2) log|T|
    the level is never reached and +inf is returned. Otherwise the root is bracketed and
    bisected in log beta to the relative tolerance rtol.
    """
    target = 0.5 * math.log(cost.numel())
    if math.log(int((cost == cost.min()).sum())) >= target:
        return math.inf
    low, high = _bracket(cost, target)
    while high > low * (1.0 + rtol):
        middle = math.sqrt(low * high)
        low, high = (middle, high) if entropy(cost, middle) > target else (low, middle)
    return math.sqrt(low * high)


def _ranked(values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the values in ascending order [N], the position of every value in that
    order [N], and per position the number of values below it and at most it [N]."""
    ordered, order = torch.sort(values, stable=True)
    position = torch.empty_like(order)
    position[order] = torch.arange(len(values))
    below = torch.searchsorted(ordered, ordered)
    return ordered, position, below, torch.searchsorted(ordered, ordered, right=True)


def _counts(positions: Tensor, size: int) -> Tensor:
    """Return per row of positions [B, m] the number of its entries below each of the
    positions 0, ..., size, [B, size + 1]."""
    hits = torch.zeros(len(positions), size + 1, dtype=torch.int64)
    hits.scatter_add_(1, positions + 1, torch.ones_like(positions))
    return hits.cumsum(dim=1)


def _view_cost(counts: Tensor, probe: Tensor, upper: Tensor, lower: Tensor) -> Tensor:
    """Return R of one view [B, levels + 1] from the counts of its typical set and its
    probe, read at the numbers of values at most (upper) and below (lower) each
    threshold."""
    n_draws, n = len(upper), int(counts[0, -1])
    above = n - counts.expand(n_draws, -1).gather(1, upper)
    under = probe.expand(n_draws, -1).gather(1, lower)
    return above.double() / n + under.double() / int(probe[0, -1])


def _drawn_costs(ranked, typical, probes, levels: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return R_1, R_2 and the thresholds, each [B, levels + 1], of B draws.

    typical and probes hold per view the counts of ``_counts`` over the positions of
    ``ranked``, [B, N + 1], or [1, N + 1] when every draw shares them. A threshold is
    the value at the first position where the joint count of the typical sets exceeds
    its grid position, the order statistic that ``threshold_grid`` takes; counting at
    the ends of its run of tied values keeps both inequalities of R strict.
    """
    ordered, _, below, at_most = ranked
    joint = typical[0] + typical[1]
    targets = (grid_positions(int(joint[0, -1]), levels) + 1).repeat(len(joint), 1)
    index = torch.searchsorted(joint, targets) - 1
    upper, lower = at_most[index], below[index]
    cost_1, cost_2 = (_view_cost(t, p, upper, lower) for t, p in zip(typical, probes))
    return cost_1, cost_2, ordered[index]


def _chunk_sizes(n_draws: int, n_values: int) -> list[int]:
    size = max(1, CHUNK_ELEMENTS // n_values)
    return [min(size, n_draws - start) for start in range(0, n_draws, size)]


def _split(pool: Tensor, perms: list[Tensor], n_first: int, n_values: int) -> list:
    """Return the counts of the first n_first entries of pool under every permutation of
    perms, and those of the rest, each [len(perms), n_values + 1]."""
    first = _counts(pool[torch.stack(perms)[:, :n_first]], n_values)
    # The rest by difference from the whole pool, which halves the scatters.
    return [first, _counts(pool[None], n_values) - first]


def _transfer_counts(ranked, n_1: int, n_2: int, n_draws: int, seed: int) -> Iterator:
    """Yield per chunk of draws the counts of the two typical sets and of the probe."""
    position, n_values = ranked[1], len(ranked[0])
    probe = _counts(position[None, n_1 + n_2 :], n_values)
    generator = torch.Generator().manual_seed(seed)
    for size in _chunk_sizes(n_draws, n_values):
        perms = [torch.randperm(n_1 + n_2, generator=generator) for _ in range(size)]
        yield _split(position[: n_1 + n_2], perms, n_1, n_values), [probe, probe]


def _view_permutations(pools: list[Tensor], size: int, generator) -> list[list]:
    """Return per pool the permutations of size draws, drawn draw by draw and, within a
    draw, in the order of the pools."""
    draws = [
        [torch.randperm(len(pool), generator=generator) for pool in pools]
        for _ in range(size)
    ]
    return [list(view) for view in zip(*draws)]


def _content_counts(ranked, n_1: int, n_2: int, n_draws: int, seed: int) -> Iterator:
    """Yield per chunk of draws the counts of the two typical sets and of the two
    probes."""
    position, n_values = ranked[1], len(ranked[0])
    typical, probe = position[: n_1 + n_2], position[n_1 + n_2 :]
    pools = [torch.cat([typical[:n_1], probe]), torch.cat([typical[n_1:], probe])]
    generator = torch.Generator().manual_seed(seed)
    for size in _chunk_sizes(n_draws, n_values):
        perms = _view_permutations(pools, size, generator)
        views = zip(pools, perms, (n_1, n_2))
        splits = [_split(pool, p, n, n_values) for pool, p, n in views]
        yield [split[0] for split in splits], [split[1] for split in splits]


def _draws(counts: Iterator, ranked, beta: float, levels: int) -> dict[str, Tensor]:
    """Return A, C and F at beta of every draw of a null, [n_draws] each."""
    chunks = []
    for typical, probes in counts:
        cost_1, cost_2, _ = _drawn_costs(ranked, typical, probes, levels)
        chunks.append(kernel(cost_1, cost_2, beta))
    return {q: torch.cat([chunk[q] for chunk in chunks]) for q in ("A", "C", "F")}


def _summary(draws: dict[str, Tensor], names: dict) -> dict[str, float]:
    # torch's default std, ddof 1, as the CAP bank's sigma_pi.
    values = {}
    for q, (mean_key, sigma_key) in names.items():
        values |= {mean_key: draws[q].mean().item(), sigma_key: draws[q].std().item()}
    return values


def agreement(scores_1, scores_2, probe, betas, levels: int) -> dict[str, Tensor]:
    """Return A, C and F of the observed posteriors at every beta, [n_beta] each.

    The scores are not checked for finiteness, which ``threshold_posterior`` does.

    Args:
        scores_1: Scores of X1 [n_1], as anything ``torch.as_tensor`` accepts.
        scores_2: Scores of X2 [n_2].
        probe: Scores of the probe set [n_P].
        betas: One beta or a 1-D grid of them [n_beta].
        levels: L, the grid T holding L + 1 thresholds.
    """
    sets = [_as_scores(s) for s in (scores_1, scores_2, probe)]
    grid = threshold_grid(sets[0], sets[1], levels)
    cost_1, cost_2 = (costs(s, sets[2], grid) for s in sets[:2])
    betas = torch.as_tensor(betas, dtype=torch.float64).reshape(-1, 1)
    return kernel(cost_1, cost_2, betas)


def null_draws(
    scores_1, scores_2, probe, beta: float, config, seed: int | None = None
) -> dict[str, dict[str, Tensor]]:
    """Return A, C and F at beta of every draw of each null, [n_draws] each.

    The keys are 'transfer' and 'content'. The draws are those that
    ``threshold_posterior`` summarises when seed is None, which takes
    ``config["seed"]``; ``config["n_draws"]`` and ``config["levels"]`` are read too.
    The arguments are otherwise those of ``agreement``, and as there the scores are
    not checked for finiteness.
    """
    sets = [_as_scores(s) for s in (scores_1, scores_2, probe)]
    ranked = _ranked(torch.cat(sets))
    seed = config["seed"] if seed is None else seed
    args = (ranked, len(sets[0]), len(sets[1]), config["n_draws"], seed)
    nulls = {"transfer": _transfer_counts, "content": _content_counts}
    return {
        name: _draws(counts(*args), ranked, beta, config["levels"])
        for name, counts in nulls.items()
    }


def _nulls(sets: list[Tensor], beta: float, config) -> dict[str, float]:
    """Return the means and spreads of A, C and F over the transfer and content
    nulls."""
    draws = null_draws(*sets, beta, config)
    transfer = _summary(draws["transfer"], TRANSFER_KEYS)
    return transfer | _summary(draws["content"], CONTENT_KEYS)


def _supremum(curve: Tensor, betas: Tensor) -> dict[str, float]:
    """Return the maximum of curve, its beta, and 1.0 if that beta ends the grid."""
    best = int(torch.argmax(curve))
    at_bound = float(best in (0, len(betas) - 1))
    return {
        "A-sup": curve[best].item(),
        "beta_star": betas[best].item(),
        "at_bound": at_bound,
    }


def _observed(sets: list[Tensor], beta: float, config) -> dict[str, float]:
    """Return A-fix, C and F at beta, and A-sup with its beta_star and at_bound."""
    betas = beta_grid(config)
    at_betas = torch.cat([torch.tensor([beta], dtype=torch.float64), betas])
    curves = agreement(*sets, at_betas, config["levels"])
    at_beta = {q: v[0].item() for q, v in curves.items()}
    values = _supremum(curves["A"][1:], betas)
    return values | {"A-fix": at_beta["A"], "C": at_beta["C"], "F": at_beta["F"]}


def threshold_posterior(
    scores_1, scores_2, probe, beta_fix: float, config
) -> dict[str, float]:
    """Return every statistic of the threshold posterior of two views and one probe.

    The keys are those of ``KEYS``: A-fix, D-fix, C and F at beta_fix, and beta_fix;
    the means and spreads over the transfer null of A (A_pi, sigma_pi), C (C_pi,
    sigma_C_pi) and F (F_pi, sigma_F_pi) and over the content null of each
    (<q>_content, sigma_<q>_content), at beta_fix; A-sup, its beta_star and at_bound,
    1.0 if beta_star is either end of the grid. If a score of the three sets is not
    finite, every value but beta_fix is NaN.

    Args:
        scores_1: Scores of X1 [n_1], as anything ``torch.as_tensor`` accepts.
        scores_2: Scores of X2 [n_2].
        probe: Scores of the probe set [n_P].
        beta_fix: The fixed beta of the probe.
        config: The posterior's configuration, as in ``configs/cap/posterior.yaml``;
            its own ``beta_fix`` is not read.
    """
    sets = [_as_scores(s) for s in (scores_1, scores_2, probe)]
    if not all(bool(torch.isfinite(s).all()) for s in sets):
        return dict.fromkeys(KEYS, math.nan) | {"beta_fix": beta_fix}
    values = _observed(sets, beta_fix, config) | _nulls(sets, beta_fix, config)
    values |= {"D-fix": values["A-fix"] - values["A_pi"], "beta_fix": beta_fix}
    return {key: values[key] for key in KEYS}


def declared_betas(config) -> dict[str, float]:
    """Return the fixed beta of every registered probe.

    Raises:
        ValueError: If a registered probe's beta_fix is null.
    """
    betas = {probe: config["beta_fix"].get(probe) for probe in config["probes"]}
    missing = [probe for probe, beta in betas.items() if beta is None]
    if missing:
        raise ValueError(
            f"beta_fix of the probes {missing} is null. The test bed declares it, by "
            "the rule of 08_tests.md (amendment of 26 Oct, 02:25), before any label "
            "is read; set it under cap_posterior.beta_fix."
        )
    return {probe: float(beta) for probe, beta in betas.items()}


def _logged_keys(probes, dataset_1: str, dataset_2: str) -> dict[tuple, str]:
    """Return the logged key of every probe and statistic of ``LOGGED``."""
    pair_name = "_vs_".join(name.replace("/", "_") for name in (dataset_1, dataset_2))
    prefix = "val/summary/posterior"
    return {
        (probe, statistic): f"{prefix}_{probe}_{statistic}_ema_{pair_name}"
        for probe in probes
        for statistic in LOGGED
    }


class ThresholdPosteriorCallback(Callback):
    """Log the moving averages of A-fix and D-fix of the threshold posterior.

    The scores of dataset_1 and dataset_2 are collected batch by batch at every
    validation epoch after the sanity pass. The probes, the auxiliary sets of the score
    dump named in ``posterior_config["probes"]``, are drawn once at fit start as
    NPDCallback draws ``autouad`` and scored at every epoch end in eval mode under
    forked RNG streams. Every epoch logs, for each probe,
    ``val/summary/posterior_<probe>_<A-fix|D-fix>_ema_<dataset_1>_vs_<dataset_2>``,
    averaged as CAPBankCallback averages: a non-finite value leaves the average
    unchanged.

    Args:
        output_name: Key in the outputs dict of the per-event anomaly scores.
        dataset_1: Name of the validation dataloader of X1.
        dataset_2: Name of the validation dataloader of X2.
        posterior_config: The posterior's configuration, as in
            ``configs/cap/posterior.yaml``. A null beta_fix of a registered probe
            raises at the first epoch end, so that the configuration composes before
            the test bed declares it.
        ema_weight: Weight of the previous value in the exponential moving average.
        aux_seed: Seed of the auxiliary sets, as ScoreDumpCallback's, and of the
            training rows whose masks the points take.
        aux_size: Number of points in each auxiliary set.
        batch_size: Batch size of the passes over the probes.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        posterior_config: dict,
        ema_weight: float = EMA_WEIGHT,
        aux_seed: int = 0,
        aux_size: int = 10000,
        batch_size: int = 1024,
    ):
        super().__init__()
        self.output_name, self.ema_weight = output_name, ema_weight
        self.datasets, self.config = (dataset_1, dataset_2), posterior_config
        self.probes = list(posterior_config["probes"])
        self.aux_seed, self.aux_size, self.batch_size = aux_seed, aux_size, batch_size
        self.keys = _logged_keys(self.probes, dataset_1, dataset_2)
        self.log_kwargs = dict(
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def on_fit_start(self, trainer, pl_module):
        """Instantiate the moving averages and draw the probes."""
        self.emas = {name: math.nan for name in self.keys}
        train = unpack_batch(trainer.datamodule.data_train.tensors)
        meta = {"aux_seed": self.aux_seed, "aux_size": self.aux_size}
        inputs = aux_inputs({**meta, **input_moments(train.x)})
        mask = drawn_masks(train.mask, self.aux_size, self.aux_seed)
        self.loaders = {
            probe: placeholder_loader(inputs[probe], self.batch_size, mask)
            for probe in self.probes
        }

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
        """Collect the scores of the two sets."""
        if trainer.sanity_checking:
            return
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name in self.scores:
            score = outputs[self.output_name].detach().reshape(-1)
            self.scores[dset_name].append(score.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Score the probes, compute the posterior per probe and log the averages."""
        if trainer.sanity_checking:
            return
        betas = declared_betas(self.config)
        typical = [torch.cat(self.scores[name]) for name in self.datasets]
        for probe, scores in self._probe_scores(trainer, pl_module).items():
            values = threshold_posterior(*typical, scores, betas[probe], self.config)
            self._average(probe, values)
        logged = {key: self.emas[name] for name, key in self.keys.items()}
        pl_module.log_dict(logged, **self.log_kwargs)

    def _probe_scores(self, trainer, pl_module) -> dict[str, Tensor]:
        keys = [self.output_name]
        scored = score_in_eval_mode(trainer, pl_module, self.loaders, keys).scores()
        return {probe: scored[probe][self.output_name] for probe in self.probes}

    def _average(self, probe: str, values: dict[str, float]) -> None:
        for statistic in LOGGED:
            name = (probe, statistic)
            self.emas[name] = ema_update(
                self.emas[name], values[statistic], self.ema_weight, skip_nonfinite=True
            )
