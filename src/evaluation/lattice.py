"""The lattice step of the evaluation stage: the run-comparing criteria, the selection
over the (k_epoch, k_point) grid and the tables of one lattice.

``Evaluator.evaluate_lattice`` writes one directory per run, ``<out_dir>/runs/<run>/``,
holding ``val.csv``, ``test.csv``, ``picks.csv`` and ``run.json``. ``select_lattice``
reads them, with the score dumps of the lattice and of its partner, and writes its
tables into ``<out_dir>``. The runs are taken in name order, so that a tie between runs
goes to the first, and must hold equal numbers of trained epochs.

The lattice criteria compare runs and are computed here, each a function of the stored
validation ``normal`` scores (``ascore/full``) of every epoch, ``initial.pt`` included
as epoch -1 (``LATTICE_CRITERIA``): ``udr`` is Spearman's rho of a run with the partner
lattice's run of equal hyperpoint (UDR over one other seed) and ``mc`` its mean rho with
every other run of the lattice at the same epoch (model centrality), both as
``label_free`` computes them. Their EMAs and epoch picks follow the per-run criteria.

A strategy takes each run's epoch from its epoch rule at k_epoch, then the run by
``selection.pick`` on the raw Q' and Q'' of every run at its epoch, at k_point. The
labelled rows (``oracle``, ``cvar25``, ``cvar10``) and ``q_sec`` pick by Q' alone at
both levels. The picks read validation values only; test values are looked up at picks
already made. Outputs:

    consensus.csv        run, epoch, the lattice criteria and their EMAs (ema/udr,
                         ema/mc); udr is empty without a partner
    selection.csv        one row per strategy and k = k_epoch = k_point (empty k for
                         the rows that pick by Q' alone): strategy, k, w, point,
                         epoch, q_main, q_sec, test_eff, oracle_test_eff, regret,
                         regret_lo, regret_hi, regret_epoch, regret_point,
                         realised_fpr, null_rate, floor_mean, floor_p05, floor_p95,
                         limit, best_test_eff (oracle row only)
    validity.csv         strategy, k, rho, p, p_holm, n
    tradeoff.csv         strategy, k_epoch, k_point (0 to 10, or gate), point, epoch,
                         test_eff, regret, joint_only
    oracle_position.csv  strategy, k_epoch, oracle_point, rank_q_main, rank_q_sec, n
    interventions.csv    strategy, k, intervention, target_cluster, test_eff,
                         oracle_test_eff
    <lattice>.csv, <lattice>_eval.csv
                         the schema of scripts/analysis/harvest_pool_effs.py

``test_eff`` is the mean test efficiency over the signal sets at the pick, and
``regret`` = test_eff - oracle_test_eff, negative when the pick does worse than the
oracle. Its interval holds the 2.5 and 97.5 percentiles of a bootstrap that resamples
the target clusters of the signal sets, the per-set differences paired. ``regret_epoch``
compares the pick with the same point at the oracle's epoch for that point, and
``regret_point`` = regret - regret_epoch. The floor is the test efficiency of every run
at the strategy's epoch for it: mean, 5th and 95th percentiles. Validity is Spearman's
rho, over the runs, between the value a pick ranks them by and the test efficiency at
each run's epoch, runs where either is not finite left out. The value is oriented so
that a positive rho means a valid ranking: Q' in its own direction at k = 10, minus Q''
at k = 0, minus the point-level rank sum between. The gate takes, among the runs whose
Q'' lies within one standard error of the best, the best Q'; the standard error comes
from a bootstrap of the best run's validation normal rows. A point is ``joint_only``
when neither k_point = 0 nor 10 reaches it at that k_epoch. Ranks are average ranks, 1
for the best and a NaN the worst.
"""

import functools
import json
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata

from src.algorithms import ADLightningModule
from src.callbacks.metrics.selection import W_STEPS, ema, epoch_pick, pick, ranks

SCORE_KEY = "ascore/full"
SCORE_SPLITS = ("val", "test", "aux", "train")
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 20261020
# The validity test's permutations, as in the criteria study of 20 Oct.
PERMUTATIONS, PERMUTATION_SEED = 10_000, 4242
GATE_DRAWS, GATE_SEED = 1_000, 20261021
# The harvest's labels where they differ from the strategy names.
LABELS = {"drift": "stability", "w1": "wasserstein"}
# The column lists of scripts/analysis/harvest_pool_effs.py, which the notebooks read.
OLD_COLUMNS = [
    "run_name",
    "strategy",
    "ckpt",
    "epoch",
    "optimized_main",
    "optimized_sec",
    "mean_sig_eff",
    "n_signals",
]
NEW_COLUMNS = [
    "experiment",
    "run",
    "run_id",
    "tier",
    "monitor",
    "maximize",
    "monitor_value",
    "q_source",
    "is_pareto",
    "test_mean_sig_eff",
    "test_n_signals",
]
EVAL_COLUMNS = [
    "run_name",
    "strategy",
    "split",
    "context",
    "experiment",
    "run",
    "run_id",
    "tier",
]


def _record_scores(path: Path) -> tuple[int, dict]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    scores = {
        split: {
            name: {key: value.numpy() for key, value in keys.items()}
            for name, keys in record[split].items()
        }
        for split in SCORE_SPLITS
        if split in record
    }
    return record["epoch"], scores


def load_scores(run_dir: Path) -> dict[int, dict]:
    """Every stored score of a dump, ``{epoch: {split: {set: {key: (n,)}}}}``.

    ``initial.pt`` is epoch -1; the splits are those its records hold (``val``,
    ``test``, ``aux``, ``train``), each with every set and key it stores.
    """
    run_dir = Path(run_dir)
    paths = [run_dir / "initial.pt", *sorted(run_dir.glob("epoch_*.pt"))]
    scores = dict(_record_scores(path) for path in paths)
    if list(scores) != list(range(-1, len(paths) - 1)):
        raise ValueError(f"{run_dir} does not hold epochs -1 to {len(paths) - 2}.")
    return scores


def dump_hyperpoint(run_dir: Path, keys) -> dict[str, str]:
    """The entries of a dump's ``overrides.yaml`` whose key is in ``keys``, as the value
    strings of the command line."""
    overrides = yaml.safe_load((Path(run_dir) / "overrides.yaml").read_text())
    entries = (override.split("=", 1) for override in overrides)
    return {key: value for key, value in entries if key in keys}


def _frozen(hyperpoint: dict) -> str:
    return json.dumps(hyperpoint, sort_keys=True)


def pair_runs(hyperpoints: dict[str, dict], partner: dict[str, dict]) -> dict[str, str]:
    """Map every run to the partner lattice's run of equal hyperpoint.

    Both arguments map run names to hyperpoints. Raise ValueError when two runs of a
    lattice share a hyperpoint or a run has no partner.
    """
    by_point = {_frozen(point): run for run, point in partner.items()}
    own = {_frozen(point) for point in hyperpoints.values()}
    if len(by_point) != len(partner) or len(own) != len(hyperpoints):
        raise ValueError("Two runs of a lattice share a hyperpoint.")
    alone = sorted(r for r, p in hyperpoints.items() if _frozen(p) not in by_point)
    if alone:
        raise ValueError(f"No partner run has the hyperpoint of {alone}.")
    return {run: by_point[_frozen(point)] for run, point in hyperpoints.items()}


def _unit_ranks(scores: np.ndarray) -> np.ndarray:
    """Centred average ranks along the last axis, scaled to unit norm.

    As in ``label_free``, non-finite scores pass through ``np.nan_to_num`` first, and
    a constant vector, whose correlation is undefined, gives zeros and so a rho of 0.
    """
    ranks = rankdata(np.nan_to_num(scores.astype(np.float64)), axis=-1)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=-1, keepdims=True)
    return np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)


def _udr(lattice: np.ndarray, partner: np.ndarray | None) -> np.ndarray | None:
    # With one other seed, the median of UDR is that seed's rho.
    if partner is None:
        return None
    return (_unit_ranks(lattice) * _unit_ranks(partner)).sum(axis=-1)


def _model_centrality(lattice: np.ndarray, partner: np.ndarray | None) -> np.ndarray:
    # Ranked once per (run, epoch); rho is (epochs, runs, runs), with 1 on the
    # diagonal (0 for a constant score), which the mean over the others leaves out.
    units = _unit_ranks(lattice).transpose(1, 0, 2)
    rho = units @ units.transpose(0, 2, 1)
    others = rho.sum(axis=2) - np.diagonal(rho, axis1=1, axis2=2)
    return (others / (len(lattice) - 1)).T


# The lattice criteria. Each maps the stored validation normal scores of every run,
# (runs, epochs + 1, n) with epoch -1 first, and those of each run's partner, aligned
# with the runs or None without a partner lattice, to (runs, epochs + 1) values, or to
# None when it is undefined without a partner.
LATTICE_CRITERIA: dict[str, Callable] = {"udr": _udr, "mc": _model_centrality}


class _Runs(NamedTuple):
    names: list[str]
    info: list[dict]  # run.json
    val: list[pd.DataFrame]  # indexed by epoch, -1 first
    test: list[pd.DataFrame]
    picks: dict  # (strategy, k or None) -> (runs,) epochs


class _Strategy(NamedTuple):
    name: str
    mode: str
    q_main: np.ndarray  # (runs, epochs + 1), epoch -1 first
    ema: np.ndarray  # (runs, epochs + 1), NaN at epoch -1
    picks: dict  # k_epoch, None for Q' alone -> (runs,) epochs
    key: str  # the MLflow key of its EMA, empty when nothing logs one


class _Test(NamedTuple):
    eff: np.ndarray  # (runs, epochs + 1), the mean over the signal sets
    realised_fpr: np.ndarray
    null_rate: np.ndarray
    set_effs: np.ndarray  # (runs, epochs + 1, signals)


class _Lattice(NamedTuple):
    names: list[str]
    q_sec: np.ndarray  # (runs, epochs + 1), validation
    val_effs: np.ndarray  # (runs, epochs + 1, signals), validation
    test: _Test
    signals: list[str]
    clusters: list[str]  # the target cluster of every signal set
    oracle: tuple  # point, epoch, the oracle's epoch of every run
    limits: dict
    gate_se: Callable[[int, int], float]


def _epoch_table(path: Path) -> pd.DataFrame:
    # Round-trip parsing reads every value exactly as written, so that ties survive.
    table = pd.read_csv(path, float_precision="round_trip")
    return table.set_index("epoch").sort_index()


def _stack_picks(tables: list[pd.DataFrame]) -> dict:
    per_run = [
        {(s, None if pd.isna(k) else int(k)): int(e) for s, k, e in t.to_numpy()}
        for t in (table[["strategy", "k", "epoch"]] for table in tables)
    ]
    if any(picks.keys() != per_run[0].keys() for picks in per_run):
        raise ValueError("The runs' picks.csv name different strategies or k.")
    return {key: np.array([picks[key] for picks in per_run]) for key in per_run[0]}


def _check_epochs(info: list[dict], tables: list[pd.DataFrame]) -> None:
    counts = sorted({run["epochs"] for run in info})
    if len(counts) != 1:
        raise ValueError(f"The runs hold different numbers of epochs: {counts}.")
    if any(list(table.index) != list(range(-1, counts[0])) for table in tables):
        raise ValueError(f"A per-run table does not hold epochs -1 to {counts[0] - 1}.")


def _read_runs(out_dir: Path) -> _Runs:
    dirs = sorted(path for path in (out_dir / "runs").iterdir() if path.is_dir())
    info = [json.loads((path / "run.json").read_text()) for path in dirs]
    val = [_epoch_table(path / "val.csv") for path in dirs]
    test = [_epoch_table(path / "test.csv") for path in dirs]
    _check_epochs(info, val + test)
    picks = _stack_picks([pd.read_csv(path / "picks.csv") for path in dirs])
    return _Runs([path.name for path in dirs], info, val, test, picks)


def _matrix(tables: list[pd.DataFrame], column: str) -> np.ndarray:
    return np.stack([table[column].to_numpy(dtype=np.float64) for table in tables])


def _set_matrix(tables: list[pd.DataFrame], prefix: str, sets: list[str]):
    return np.stack([_matrix(tables, prefix + name) for name in sets], axis=-1)


def _normal_scores(run_dir: Path) -> np.ndarray:
    scores = load_scores(run_dir)
    return np.stack([scores[epoch]["val"]["normal"][SCORE_KEY] for epoch in scores])


def _load_lattice(lattice_dir: Path, names: list[str]) -> tuple[np.ndarray, list]:
    """Stored validation normal scores, (runs, epochs + 1, n), and every meta.pt."""
    normal = np.stack([_normal_scores(lattice_dir / name) for name in names])
    metas = [torch.load(lattice_dir / n / "meta.pt", weights_only=True) for n in names]
    return normal, metas


def _partner_scores(scores_dir: Path, lattice: str, partner: str, runs: _Runs):
    """Stored validation normal scores of each run's partner, aligned with the runs."""
    keys = set().union(*(run["hyperpoint"] for run in runs.info))
    own = {n: dump_hyperpoint(scores_dir / lattice / n, keys) for n in runs.names}
    dirs = sorted(path.parent for path in (scores_dir / partner).glob("*/meta.pt"))
    pairs = pair_runs(own, {path.name: dump_hyperpoint(path, keys) for path in dirs})
    return np.stack([_normal_scores(scores_dir / partner / pairs[n]) for n in own])


def _picks_of(runs: _Runs, name: str) -> dict:
    picks = {k: epochs for (s, k), epochs in runs.picks.items() if s == name}
    if not picks:
        raise ValueError(f"picks.csv holds no epochs of {name}.")
    return picks


def _from_val(runs: _Runs, specs: list[tuple]) -> list[_Strategy]:
    """Strategies whose Q' is a column of val.csv, from (name, column, mode, key)."""
    return [
        _Strategy(
            name,
            mode,
            _matrix(runs.val, column),
            _matrix(runs.val, f"ema/{name}"),
            _picks_of(runs, name),
            key,
        )
        for name, column, mode, key in specs
    ]


def _criteria_specs(selection: dict) -> list[tuple]:
    criteria = selection["criteria"]
    return [(n, c["column"], c["mode"], c["key"]) for n, c in criteria.items()]


def _labelled_specs(selection: dict) -> list[tuple]:
    specs = [(n, n, "max", c["key"]) for n, c in selection["labelled"].items()]
    q_sec = selection["q_sec"]
    return specs + [("q_sec", q_sec["column"], "min", q_sec["key"])]


def _lattice_strategy(name: str, mode: str, values, q_sec: np.ndarray) -> _Strategy:
    trained, q_sec = values[:, 1:], q_sec[:, 1:]
    ema_values = np.array([[math.nan, *ema(row)] for row in trained])
    picks = {
        k: np.array([epoch_pick(v, s, k, mode) for v, s in zip(trained, q_sec)])
        for k in range(W_STEPS + 1)
    }
    return _Strategy(name, mode, values, ema_values, picks, "")


def _lattice_strategies(normal, partner, modes: dict, q_sec) -> list[_Strategy]:
    if partner is not None and partner.shape != normal.shape:
        raise ValueError(f"Partner scores {partner.shape}, lattice {normal.shape}.")
    values = {name: LATTICE_CRITERIA[name](normal, partner) for name in modes}
    return [
        _lattice_strategy(name, modes[name], v, q_sec)
        for name, v in values.items()
        if v is not None
    ]


def _at(values: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    """Each run's entry at its epoch, from a (runs, epochs + 1, ...) array."""
    return values[np.arange(len(epochs)), epochs + 1]


def _ks(strategy: _Strategy) -> list:
    return [None] if None in strategy.picks else list(range(W_STEPS + 1))


def _choose(strategy: _Strategy, q_sec: np.ndarray, k_epoch, k_point: int):
    """(run index, epoch) of a pick; k_epoch None takes the epochs of Q' alone."""
    epochs = strategy.picks[k_epoch]
    q_main = _at(strategy.q_main, epochs)
    point = pick(q_main, _at(q_sec, epochs), k_point, strategy.mode)
    return point, int(epochs[point])


def _pick_at(strategy: _Strategy, k, q_sec: np.ndarray) -> tuple[int, int]:
    return _choose(strategy, q_sec, k, W_STEPS if k is None else k)


def _operational_se(scores: np.ndarray, batch_sizes: list, target_fpr: float):
    """Bootstrap standard error of Q'' over the rows of validation normal.

    Each draw resamples the rows with replacement and recomputes Q'' as the
    evaluation does: the operational score of every batch of the stored layout,
    averaged over the batches.
    """
    module = SimpleNamespace(target_fpr=target_fpr)
    rng = np.random.default_rng(GATE_SEED)
    rows = torch.from_numpy(rng.integers(len(scores), size=(GATE_DRAWS, len(scores))))
    values = [
        np.mean(
            [
                ADLightningModule.compute_operational_ascore(module, batch)
                for batch in draw.split(list(batch_sizes))
            ]
        )
        for draw in torch.from_numpy(scores)[rows]
    ]
    return float(np.std(values, ddof=1))


def _gate_se(normal: np.ndarray, metas: list[dict]) -> Callable[[int, int], float]:
    @functools.cache
    def standard_error(point: int, epoch: int) -> float:
        meta = metas[point]
        layout = meta["val"]["batch_sizes"]["normal"]
        return _operational_se(normal[point, epoch + 1], layout, meta["target_fpr"])

    return standard_error


def _signals(info: list[dict]) -> list[str]:
    signals = [list(run["signals"]) for run in info]
    if any(names != signals[0] for names in signals):
        raise ValueError("The runs name different signal sets.")
    return signals[0]


def _clusters(path, signals: list[str]) -> list[str]:
    """The target cluster of every signal set; without a file each is its own."""
    if path is None:
        return list(signals)
    table = pd.read_csv(path, dtype=str).set_index("intervention")["target_cluster"]
    missing = sorted(set(signals) - set(table.index))
    if missing:
        raise ValueError(f"{path} gives no target cluster for {missing}.")
    return [table[name] for name in signals]


def _test_values(runs: _Runs, prefix: str, signals: list[str]) -> _Test:
    return _Test(
        _matrix(runs.test, "test_mean_eff"),
        _matrix(runs.test, "realised_fpr"),
        _matrix(runs.test, "null_rate"),
        _set_matrix(runs.test, prefix, signals),
    )


def _context(runs: _Runs, selection: dict, strategies: list, q_sec, gate_se):
    signals, prefix = _signals(runs.info), selection["efficiency"]
    by_name = {strategy.name: strategy for strategy in strategies}
    point, epoch = _choose(by_name["oracle"], q_sec, None, W_STEPS)
    oracle = (point, epoch, by_name["oracle"].picks[None])
    val_effs = _set_matrix(runs.val, prefix, signals)
    test = _test_values(runs, prefix, signals)
    clusters = _clusters(selection.get("clusters"), signals)
    limits = dict(selection.get("limits") or {})
    return _Lattice(
        runs.names, q_sec, val_effs, test, signals, clusters, oracle, limits, gate_se
    )


@functools.cache
def _bootstrap_draws(clusters: int) -> np.ndarray:
    # One stream for every row, so that the intervals of two strategies share draws.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(clusters, size=(BOOTSTRAP_DRAWS, clusters))


def _cluster_interval(differences: np.ndarray, clusters: list[str]) -> tuple:
    """95 % percentile interval of the mean per-set difference, the target clusters
    resampled with replacement and every set of a drawn cluster counted."""
    codes = np.unique(clusters, return_inverse=True)[1]
    sums, sizes = np.bincount(codes, weights=differences), np.bincount(codes)
    draws = _bootstrap_draws(len(sums))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def _regret(lat: _Lattice, point: int, epoch: int) -> dict:
    o_point, o_epoch, o_epochs = lat.oracle
    eff, set_effs = lat.test.eff, lat.test.set_effs
    test_eff, oracle_eff = eff[point, epoch + 1], eff[o_point, o_epoch + 1]
    at_oracle_epoch = eff[point, o_epochs[point] + 1]
    paired = set_effs[point, epoch + 1] - set_effs[o_point, o_epoch + 1]
    lo, hi = _cluster_interval(paired, lat.clusters)
    row = {"test_eff": test_eff, "oracle_test_eff": oracle_eff}
    row.update(regret=test_eff - oracle_eff, regret_lo=lo, regret_hi=hi)
    return {**row, "regret_epoch": test_eff - at_oracle_epoch}


def _floor(eff: np.ndarray, epochs: np.ndarray) -> dict:
    values = _at(eff, epochs)
    p05, p95 = np.percentile(values, [5, 95])
    return {"floor_mean": values.mean(), "floor_p05": p05, "floor_p95": p95}


def _selection_row(strategy: _Strategy, k, lat: _Lattice) -> dict:
    point, epoch = _pick_at(strategy, k, lat.q_sec)
    row = {"strategy": strategy.name, "k": k, "w": None if k is None else k / W_STEPS}
    at = (point, epoch + 1)
    row.update(point=lat.names[point], epoch=epoch)
    row.update(q_main=strategy.q_main[at], q_sec=lat.q_sec[at])
    row.update(_regret(lat, point, epoch))
    row["regret_point"] = row["regret"] - row["regret_epoch"]
    test = {"realised_fpr": lat.test.realised_fpr, "null_rate": lat.test.null_rate}
    row.update({name: values[at] for name, values in test.items()})
    row.update(_floor(lat.test.eff, strategy.picks[k]))
    best = np.nanmax(lat.test.eff[:, 1:]) if strategy.name == "oracle" else None
    return {**row, "limit": lat.limits.get(strategy.name, ""), "best_test_eff": best}


def _ranking_value(strategy: _Strategy, k, q_sec: np.ndarray) -> np.ndarray:
    """The value a pick at k ranks the runs by, oriented so that larger is better."""
    epochs = strategy.picks[k]
    q_main, q_sec = _at(strategy.q_main, epochs), _at(q_sec, epochs)
    if k is None or k == W_STEPS:
        return q_main if strategy.mode == "max" else -q_main
    if k == 0:
        return -q_sec
    rank_sum = k * ranks(q_main, strategy.mode) + (W_STEPS - k) * ranks(q_sec, "min")
    return -rank_sum


def _centred_ranks(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x)
    return ranks - ranks.mean()


def _spearman_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman's rho and its two-sided permutation p-value.

    p = (1 + #{abs(rho*) >= abs(rho)}) / (1 + N) over N permutations of y.
    """
    if len(x) < 2:
        return math.nan, math.nan
    rx, ry = _centred_ranks(x), _centred_ranks(y)
    scale = math.sqrt((rx @ rx) * (ry @ ry))
    if scale == 0:
        return math.nan, math.nan
    # A permutation keeps the norm of ry, so the numerators order the |rho|. Centred
    # ranks are multiples of 1/2, so the products are exact and ties compare equal.
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = rng.permuted(np.tile(ry, (PERMUTATIONS, 1)), axis=1) @ rx
    extreme = (np.abs(null) >= abs(rx @ ry)).sum()
    return float(rx @ ry / scale), float((1 + extreme) / (1 + PERMUTATIONS))


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm's adjusted p-values over the finite entries: the running maximum of
    (m - j) p_(j), capped at 1."""
    adjusted = np.full(len(p), math.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite], kind="stable")]
    scaled = (len(order) - np.arange(len(order))) * p[order]
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def _validity_row(strategy: _Strategy, k, lat: _Lattice) -> dict:
    value = _ranking_value(strategy, k, lat.q_sec)
    test = _at(lat.test.eff, strategy.picks[k])
    kept = np.isfinite(value) & np.isfinite(test)
    rho, p = _spearman_test(value[kept], test[kept])
    row = {"strategy": strategy.name, "k": k, "rho": rho, "p": p}
    return {**row, "p_holm": math.nan, "n": int(kept.sum())}


def _validity_rows(strategies: list[_Strategy], lat: _Lattice) -> list[dict]:
    rows = [_validity_row(s, k, lat) for s in strategies for k in _ks(s)]
    p_holm = _holm(np.array([row["p"] for row in rows], dtype=np.float64))
    for row, adjusted in zip(rows, p_holm):
        row["p_holm"] = adjusted
    return rows


def _gate(strategy: _Strategy, lat: _Lattice, k_epoch: int) -> tuple[int, int]:
    epochs = strategy.picks[k_epoch]
    q_main, q_sec = _at(strategy.q_main, epochs), _at(lat.q_sec, epochs)
    best = pick(q_main, q_sec, 0, strategy.mode)
    bound = q_sec[best] + lat.gate_se(best, int(epochs[best]))
    within = np.union1d(np.flatnonzero(q_sec <= bound), [best])
    point = int(within[pick(q_main[within], q_sec[within], W_STEPS, strategy.mode)])
    return point, int(epochs[point])


def _tradeoff_row(strategy: _Strategy, lat: _Lattice, k_epoch: int, k_point, ends):
    if k_point == "gate":
        point, epoch = _gate(strategy, lat, k_epoch)
    else:
        point, epoch = _choose(strategy, lat.q_sec, k_epoch, k_point)
    test_eff = lat.test.eff[point, epoch + 1]
    o_point, o_epoch, _ = lat.oracle
    row = {"strategy": strategy.name, "k_epoch": k_epoch, "k_point": k_point}
    row.update(point=lat.names[point], epoch=epoch, test_eff=test_eff)
    regret = test_eff - lat.test.eff[o_point, o_epoch + 1]
    return {**row, "regret": regret, "joint_only": point not in ends}


def _tradeoff_rows(strategy: _Strategy, lat: _Lattice) -> list[dict]:
    rows = []
    for k_epoch in range(W_STEPS + 1):
        ends = {_choose(strategy, lat.q_sec, k_epoch, k)[0] for k in (0, W_STEPS)}
        for k_point in [*range(W_STEPS + 1), "gate"]:
            rows.append(_tradeoff_row(strategy, lat, k_epoch, k_point, ends))
    return rows


def _oracle_position_row(strategy: _Strategy, lat: _Lattice, k_epoch: int) -> dict:
    epochs, point = strategy.picks[k_epoch], lat.oracle[0]
    rank_main = ranks(_at(strategy.q_main, epochs), strategy.mode)[point]
    rank_sec = ranks(_at(lat.q_sec, epochs), "min")[point]
    row = {"strategy": strategy.name, "k_epoch": k_epoch}
    row.update(oracle_point=lat.names[point], rank_q_main=rank_main)
    return {**row, "rank_q_sec": rank_sec, "n": len(epochs)}


def _intervention_rows(strategy: _Strategy, k, lat: _Lattice) -> list[dict]:
    point, epoch = _pick_at(strategy, k, lat.q_sec)
    o_point, o_epoch, _ = lat.oracle
    picked = lat.test.set_effs[point, epoch + 1]
    oracle = lat.test.set_effs[o_point, o_epoch + 1]
    head = {"strategy": strategy.name, "k": k}
    keys = ("intervention", "target_cluster", "test_eff", "oracle_test_eff")
    values = zip(lat.signals, lat.clusters, picked, oracle)
    return [{**head, **dict(zip(keys, entry))} for entry in values]


def _write(path: Path, rows) -> None:
    table = pd.DataFrame(rows)
    if "k" in table:
        # Nullable, so that the k of a pick by Q' alone is written empty rather than
        # as a NaN that turns every k into a float.
        table["k"] = table["k"].astype("Int64")
    table.to_csv(path, index=False)


def _consensus_table(names: list[str], strategies: list, modes, n_epochs: int):
    """Every lattice criterion and its EMA by run and epoch; empty where undefined."""
    found = {strategy.name: strategy for strategy in strategies}
    table = {"run": np.repeat(names, n_epochs)}
    table["epoch"] = np.tile(np.arange(-1, n_epochs - 1), len(names))
    for name in modes:
        table[name] = found[name].q_main.ravel() if name in found else math.nan
    for name in modes:
        table[f"ema/{name}"] = found[name].ema.ravel() if name in found else math.nan
    return pd.DataFrame(table)


def _write_tables(out_dir: Path, strategies: list[_Strategy], lat: _Lattice) -> None:
    graded = [strategy for strategy in strategies if None not in strategy.picks]
    rows = [_selection_row(s, k, lat) for s in strategies for k in _ks(s)]
    _write(out_dir / "selection.csv", rows)
    _write(out_dir / "validity.csv", _validity_rows(strategies, lat))
    rows = [row for s in graded for row in _tradeoff_rows(s, lat)]
    _write(out_dir / "tradeoff.csv", rows)
    ks = range(W_STEPS + 1)
    rows = [_oracle_position_row(s, lat, k) for s in graded for k in ks]
    _write(out_dir / "oracle_position.csv", rows)
    groups = [_intervention_rows(s, k, lat) for s in strategies for k in _ks(s)]
    _write(out_dir / "interventions.csv", [row for rows in groups for row in rows])


def _tier_name(rate: float, base_rate: float | None) -> str:
    # The operational tier as harvest_pool_effs._tier_name names it: by its rate in Hz
    # (rates are in kHz) with a base rate, else by the quantile its rate sets.
    if base_rate is not None:
        return f"{1e3 * rate:g}"
    return f"q{100 * (1 - rate):g}"


def _harvest_name(name: str, k) -> str:
    label = LABELS.get(name, name)
    return label if k is None else f"{label}_w{k / W_STEPS:g}"


def _harvest_row(strategy: _Strategy, k, i: int, lat: _Lattice) -> dict:
    epoch, name = int(strategy.picks[k][i]), _harvest_name(strategy.name, k)
    effs = lat.val_effs[i, epoch + 1]
    row = {"run_name": f"{name}_{lat.names[i]}", "strategy": name, "epoch": epoch}
    row.update(optimized_main=strategy.q_main[i, epoch + 1])
    row.update(optimized_sec=lat.q_sec[i, epoch + 1], mean_sig_eff=effs.mean())
    row.update(monitor=strategy.key, maximize=strategy.mode == "max")
    row.update(monitor_value=strategy.ema[i, epoch + 1], n_signals=len(effs))
    row.update(test_mean_sig_eff=lat.test.eff[i, epoch + 1], test_n_signals=len(effs))
    return {**row, **dict(zip(lat.signals, effs))}


def _harvest_rows(strategies, runs: _Runs, lat: _Lattice, fixed: dict) -> list[dict]:
    """One row per run and strategy at k = 10 and k = 0, a row by Q' alone once."""
    rows = []
    for i, info in enumerate(runs.info):
        head = {**fixed, "run": runs.names[i], "run_id": info["mlflow_run_id"]}
        head.update({f"hp/{key}": v for key, v in info["hyperpoint"].items()})
        for strategy in strategies:
            ks = [None] if None in strategy.picks else [W_STEPS, 0]
            rows += [{**_harvest_row(strategy, k, i, lat), **head} for k in ks]
    return rows


def _harvest_table(rows: list[dict], signals: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(rows).sort_values("run_name", kind="stable")
    hp = sorted(column for column in table.columns if column.startswith("hp/"))
    return table[OLD_COLUMNS + NEW_COLUMNS + sorted(signals) + hp]


def _contexts(columns) -> dict[str, list[str]]:
    """The recorded <cb_name>/<scalar> columns of a per-run table, by callback."""
    contexts = {}
    for column in columns:
        callback, _, scalar = column.partition("/")
        if scalar and callback != "ema":
            contexts.setdefault(callback, []).append(column)
    return contexts


def _eval_records(row: dict, tables: dict[str, pd.DataFrame]) -> list[dict]:
    head = {column: row[column] for column in EVAL_COLUMNS if column in row}
    records = []
    for split, table in sorted(tables.items()):
        values = table.loc[row["epoch"]]
        for context, columns in sorted(_contexts(table.columns).items()):
            scalars = {column.partition("/")[2]: values[column] for column in columns}
            records.append({**head, "split": split, "context": context, **scalars})
    return records


def _eval_table(harvest: pd.DataFrame, runs: _Runs) -> pd.DataFrame:
    tables = {
        name: {"val": val, "test": test}
        for name, val, test in zip(runs.names, runs.val, runs.test)
    }
    rows = harvest.to_dict("records")
    records = [r for row in rows for r in _eval_records(row, tables[row["run"]])]
    table = pd.DataFrame(records).sort_values(["run_name", "split", "context"])
    return table[EVAL_COLUMNS + sorted(set(table.columns) - set(EVAL_COLUMNS))]


def _write_harvest(out_dir: Path, lattice: str, strategies, runs, lat, meta) -> None:
    tier = _tier_name(meta["target_rate"], meta["base_rate"])
    fixed = {"experiment": lattice, "tier": tier, "ckpt": "epoch"}
    fixed.update(q_source="eval", is_pareto=None)
    table = _harvest_table(_harvest_rows(strategies, runs, lat, fixed), lat.signals)
    table.to_csv(out_dir / f"{lattice}.csv", index=False)
    _eval_table(table, runs).to_csv(out_dir / f"{lattice}_eval.csv", index=False)


def _strategies(runs: _Runs, selection: dict, consensus: list) -> list[_Strategy]:
    """The criteria, the lattice criteria, the labelled rows and q_sec, in order."""
    criteria = _from_val(runs, _criteria_specs(selection))
    return criteria + consensus + _from_val(runs, _labelled_specs(selection))


def select_lattice(
    out_dir: Path,
    selection: dict,
    scores_dir: Path,
    lattice: str,
    partner: str | None,
) -> None:
    """Compute the lattice criteria, apply every strategy and write the tables.

    :param out_dir: The stage's output directory of the lattice, whose ``runs/``
        holds the per-run outputs of ``Evaluator.evaluate_lattice``.
    :param selection: The ``selection`` node of the eval domain config: ``criteria``,
        ``q_sec``, ``labelled``, ``efficiency``, ``clusters`` (a CSV with columns
        intervention and target_cluster, or None), ``lattice_criteria`` (name: mode)
        and ``limits`` (strategy: note).
    :param scores_dir: The root of the dumps, ``<scores_dir>/<lattice>/<run>/``.
    :param lattice: The lattice's name, its directory under ``scores_dir``.
    :param partner: The lattice whose runs, paired by hyperpoint, give UDR; None
        leaves UDR out.
    """
    out_dir, scores_dir = Path(out_dir), Path(scores_dir)
    runs = _read_runs(out_dir)
    normal, metas = _load_lattice(scores_dir / lattice, runs.names)
    paired = None
    if partner is not None:
        paired = _partner_scores(scores_dir, lattice, partner, runs)
    q_sec = _matrix(runs.val, selection["q_sec"]["column"])
    modes = selection["lattice_criteria"]
    consensus = _lattice_strategies(normal, paired, modes, q_sec)
    strategies = _strategies(runs, selection, consensus)
    lat = _context(runs, selection, strategies, q_sec, _gate_se(normal, metas))
    table = _consensus_table(runs.names, consensus, modes, q_sec.shape[1])
    _write(out_dir / "consensus.csv", table)
    _write_tables(out_dir, strategies, lat)
    _write_harvest(out_dir, lattice, strategies, runs, lat, metas[0])
