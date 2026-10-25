#!/usr/bin/env python3
"""Harvest the shared-pool runs into one row per evaluated checkpoint.

Every run of a physics pool experiment (``physics_<model>_pool``) saves one checkpoint
per selection strategy, and the evaluator scores each checkpoint on the validation and
test splits. For every (run, evaluated checkpoint, rate tier) this writes the strategy
label, the strategy's own validation quantity Q′ (``optimized_main``) and the second
objective Q″ (``optimized_sec``) at the checkpoint, the mean signal efficiency on both
splits, the parameters that vary across the pool (``hp/...``), and ``is_pareto``:
whether the row is non-dominated in (Q′, Q″) among its strategy's rows at that tier. The
pools of the other domains keep no checkpoint; ``src/evaluation/harvest.py`` writes
their tables in this schema.

The checkpoints are read off the run. Each ``callbacks/<name>/monitor`` parameter
belongs to a checkpoint callback whose ``dirpath`` ends in the
``<strategy>/<metric>/<criterion>`` folders by which the evaluator keys its metrics,
``eval/<split>/<strategy>/<callback>/<metric>/<criterion>/[<ckpt>/]<scalar>``; the
callback counts as evaluated when such keys exist. Its ``mode`` gives the direction of
Q′, and the first best value of the monitored history gives the epoch. ``FAMILIES`` maps
the monitored key, as the training callbacks name it, to the label the effs notebooks
use, to the rate tier, and to the evaluation scalar that recomputes Q′ on the
validation split; Q′ of a CVaR strategy is the CVaR itself, not the 10³-scaled value the
search reports to Optuna. A key outside ``FAMILIES`` (the validation loss) keeps the
checkpoint's metric folder as its label and its best monitored value as Q′, which
``q_source`` records. The label-free criteria of ``HISTORY_FAMILIES`` have no evaluation
scalar on the LHC; their Q′ is the raw value the training callback logs beside the EMA,
read at the checkpoint's epoch.

The files keep the schema that ``notebooks/lib/effs_data.wl`` reads, one pair per tier:
``<experiment>.csv`` and ``<experiment>_eval.csv`` for the operational rate, and
``<domain>_<model>_<tier>_pool.csv`` with its ``_eval`` file for every further rate, as
the q99 Pareto tables were named. The ``tier`` column holds the same names (``250``,
``q99``), and a rate-free strategy appears in every tier. ``run_name`` is
``<strategy>_<run>``, the pool's counterpart of the ``<strategy>_t<trial>`` names of the
Pareto retrainings.

Usage::

    python scripts/analysis/harvest_pool_effs.py
    python scripts/analysis/harvest_pool_effs.py physics_ae_pool \\
        --tracking-uri file:logs/mlflow/mlruns --outdir notebooks/pareto_effs
"""
import argparse
import math
import re
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

REPO = Path(__file__).resolve().parents[2]

# Q″ is read as optimized_metric_config.sec_metric reads it in every physics experiment.
SEC_CALLBACK, SEC_SCALAR = "fit", "normal"

# Validation keys of src/callbacks (after "val/summary/"), the strategy label of the
# effs notebooks, and the evaluation scalar that recomputes the quantity. CVaR has no
# scalar: the evaluation logs a single tail level, so it is recomputed from the
# per-signal efficiencies.
FAMILIES = [
    (r"eff_cvar(?P<pct>\d+)_ema_(?P<tier>.+)", "cvar{pct}", None),
    (r"(?P<tier>operational|trate.+kHz)_drift_ema", "stability", "drift_{tier}"),
    (r"w1dist_ema_.+", "wasserstein", "w1dist"),
    (r"cap_ema_.+", "cap", "cap"),
    (r"consistency_ema_.+", "consistency", "consistency"),
    (r"fit_ema", "q_sec", SEC_SCALAR),
]
# The label-free criteria have no evaluation scalar on the LHC: Q′ is the raw value the
# training callback logs beside its EMA, read at the checkpoint's epoch. SDS cut at a
# rate other than the operational one carries the rate token in its key.
HISTORY_FAMILIES = [
    (r"(?P<name>npd|rtm|eag|entropy|sds|sds_val)(_(?P<tier>\d+_\d+))?_ema", "{name}"),
]

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


class Checkpoint(NamedTuple):
    folder: tuple[str, ...]  # (strategy, metric, criterion), as the evaluator reads it
    monitor: str
    maximize: bool


class Tier(NamedTuple):
    rate: float
    label: str  # the evaluation scalars' name: "operational" or "286.088064"
    name: str  # the submitted tables' name: "250" or "q99"


class Strategy(NamedTuple):
    ckpt: Checkpoint
    label: str
    tier: Tier | None  # None for a rate-free strategy
    q_scalar: str | None
    alpha: float | None  # CVaR tail fraction
    q_history: str | None = None  # training key of the raw Q′


def _checkpoints(params: dict[str, str]) -> list[Checkpoint]:
    ckpts = []
    for key, monitor in params.items():
        name = key.removesuffix("/monitor")
        if name == key or not key.startswith("callbacks/") or monitor == "None":
            continue
        folder = tuple(params.get(f"{name}/dirpath", "").rstrip("/").split("/")[-3:])
        ckpts.append(Checkpoint(folder, monitor, params.get(f"{name}/mode") == "max"))
    return sorted(ckpts)


def _eval_index(metrics: dict[str, float]) -> dict:
    """Group a run's checkpoint metrics by folder, split, context and scalar.

    The context is the part of the key between split and scalar, as in the Pareto
    harvest; keys without the folders (``last``, ``optimized``) are left out.
    """
    index: dict = {}
    for key, value in metrics.items():
        parts = key.split("/")
        if parts[0] != "eval" or len(parts) < 7:
            continue
        splits = index.setdefault((parts[2], parts[4], parts[5]), {})
        context = splits.setdefault(parts[1], {}).setdefault("/".join(parts[2:-1]), {})
        context[parts[-1]] = value
    return index


def _rates(params: dict[str, str]) -> tuple[float, float | None]:
    """Return the run's target rate and base rate, None when it sets no base rate."""
    base_rate = params.get("algorithm/base_rate", "None")
    target = float(params["algorithm/target_rate"])
    return target, None if base_rate == "None" else float(base_rate)


def _tier_name(rate: float, operational: bool, base_rate: float | None) -> str:
    # As the submitted tables were named: the operational physics tier by its rate in Hz
    # (rates are in kHz), every other tier by the quantile its false-positive rate sets.
    if operational and base_rate is not None:
        return f"{1e3 * rate:g}"
    false_positive_rate = rate / base_rate if base_rate is not None else rate
    return f"q{100 * (1 - false_positive_rate):g}"


def _tier(token: str, target: float, base_rate: float | None) -> Tier:
    """Map a rate token (operational, 286_088064, trate286_088064kHz) to its tier."""
    rate = target
    if token != "operational":
        rate = float(token.removeprefix("trate").removesuffix("kHz").replace("_", "."))
    operational = math.isclose(rate, target)
    label = "operational" if operational else str(rate)
    return Tier(rate, label, _tier_name(rate, operational, base_rate))


def _strategy(ckpt: Checkpoint, target: float, base_rate: float | None) -> Strategy:
    for pattern, label, scalar in FAMILIES:
        match = re.fullmatch(f"val/summary/{pattern}", ckpt.monitor)
        if match is None:
            continue
        groups = match.groupdict()
        tier = _tier(groups["tier"], target, base_rate) if "tier" in groups else None
        alpha = int(groups["pct"]) / 100 if "pct" in groups else None
        q_scalar = scalar.format(**groups) if scalar else None
        return Strategy(ckpt, label.format(**groups), tier, q_scalar, alpha)
    return _history_strategy(ckpt, target, base_rate)


def _history_strategy(
    ckpt: Checkpoint, target: float, base_rate: float | None
) -> Strategy:
    """A strategy whose Q′ is read from the training history, rate-free unless its key
    carries a rate token, or the checkpoint's metric folder as label for a key outside
    every family."""
    for pattern, label in HISTORY_FAMILIES:
        match = re.fullmatch(f"val/summary/{pattern}", ckpt.monitor)
        if match is not None:
            raw = ckpt.monitor.removesuffix("_ema")
            groups = match.groupdict()
            token = groups.get("tier")
            tier = None if token is None else _tier(token, target, base_rate)
            return Strategy(ckpt, label.format(**groups), tier, None, None, raw)
    return Strategy(ckpt, ckpt.folder[1], None, None, None)


def _best_epoch(
    client: MlflowClient, run_id: str, ckpt: Checkpoint
) -> tuple[int | None, float | None]:
    """Return the epoch at which the checkpoint was kept and the monitored value there.

    ModelCheckpoint keeps the first epoch with the best finite value, and the history
    holds one value per validation epoch.
    """
    values = _history(client, run_id, ckpt.monitor)
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None, None
    best = max(finite) if ckpt.maximize else min(finite)
    return values.index(best), best


def _history(client: MlflowClient, run_id: str, key: str) -> list[float]:
    """The values of a per-epoch key in epoch order."""
    history = client.get_metric_history(run_id, key)
    return [m.value for m in sorted(history, key=lambda m: (m.step, m.timestamp))]


def _value_at(
    client: MlflowClient, run_id: str, key: str | None, epoch: int | None
) -> float | None:
    if key is None or epoch is None:
        return None
    values = _history(client, run_id, key)
    return values[epoch] if epoch < len(values) else None


def _scalars(contexts: dict[str, dict[str, float]]) -> dict[str, float]:
    """Merge a checkpoint's contexts; the first callback in name order wins a clash."""
    merged: dict[str, float] = {}
    for context in sorted(contexts):
        for scalar, value in contexts[context].items():
            merged.setdefault(scalar, value)
    return merged


def _signal_effs(
    scalars: dict[str, float], tier: Tier, bkg_re: re.Pattern
) -> dict[str, float]:
    prefix = f"eff_{tier.label}_"
    effs = {k[len(prefix) :]: v for k, v in scalars.items() if k.startswith(prefix)}
    return {ds: eff for ds, eff in effs.items() if not bkg_re.search(ds)}


def _sec(contexts: dict[str, dict[str, float]]) -> float | None:
    for context in sorted(contexts):
        if context.split("/")[1] == SEC_CALLBACK:
            return contexts[context].get(SEC_SCALAR)
    return None


def _cvar(values: list[float], alpha: float) -> float:
    # The ceil(alpha n) lowest values, as in the evaluation efficiency callback, whose
    # background term is zero: no efficiency dataset list holds a background set.
    k = max(1, math.ceil(alpha * len(values)))
    return statistics.fmean(sorted(values)[:k])


def _mean(values: dict[str, float]) -> float | None:
    return statistics.fmean(values.values()) if values else None


def _q_eval(
    strategy: Strategy, val: dict[str, float], val_effs: dict[str, float]
) -> float | None:
    if strategy.alpha is not None:
        return _cvar(list(val_effs.values()), strategy.alpha) if val_effs else None
    return val.get(strategy.q_scalar)


def _q_main(
    strategy: Strategy, val: dict[str, float], val_effs: dict[str, float], best: tuple
) -> tuple[float | None, str]:
    """Q′ of a row and its source: the training history, the evaluation or the monitored
    value."""
    if strategy.q_history is not None:
        return best[2], "history"
    if strategy.alpha is not None or strategy.q_scalar is not None:
        return _q_eval(strategy, val, val_effs), "eval"
    return best[1], "monitor"


def _row(
    run: Run,
    strategy: Strategy,
    tier: Tier,
    evals: dict,
    best: tuple,
    bkg_re: re.Pattern,
) -> dict:
    """Return one output row; the underscored entries are internal to the writers."""
    val, test = _scalars(evals.get("val", {})), _scalars(evals.get("test", {}))
    val_effs = _signal_effs(val, tier, bkg_re)
    test_effs = _signal_effs(test, tier, bkg_re)
    q_main, q_source = _q_main(strategy, val, val_effs, best)
    return {
        "run_name": f"{strategy.label}_{run.info.run_name}",
        "strategy": strategy.label,
        "ckpt": "strategy",
        "epoch": best[0],
        "optimized_main": q_main,
        "optimized_sec": _sec(evals.get("val", {})),
        "mean_sig_eff": _mean(val_effs),
        "n_signals": len(val_effs),
        "run": run.info.run_name,
        "run_id": run.info.run_id,
        "tier": tier.name,
        "monitor": strategy.ckpt.monitor,
        "maximize": strategy.ckpt.maximize,
        "monitor_value": best[1],
        "q_source": q_source,
        "test_mean_sig_eff": _mean(test_effs),
        "test_n_signals": len(test_effs),
        "_label": tier.label,
        "_evals": evals,
        **val_effs,
    }


def _run_rows(client: MlflowClient, run: Run, bkg_re: re.Pattern) -> list[dict]:
    rates = _rates(run.data.params)
    index = _eval_index(run.data.metrics)
    ckpts = [c for c in _checkpoints(run.data.params) if c.folder in index]
    strategies = [_strategy(c, *rates) for c in ckpts]
    operational = [_tier("operational", *rates)]
    tiers = sorted({s.tier for s in strategies if s.tier}) or operational
    rows = []
    for s in strategies:
        epoch, monitored = _best_epoch(client, run.info.run_id, s.ckpt)
        raw = _value_at(client, run.info.run_id, s.q_history, epoch)
        best = (epoch, monitored, raw)
        for tier in [s.tier] if s.tier else tiers:
            rows.append(_row(run, s, tier, index[s.ckpt.folder], best, bkg_re))
    return rows


def _latest_runs(client: MlflowClient, name: str) -> list[Run]:
    """Return the newest run per run name among those the evaluator logged to.

    Every experiment of that name is searched, since a launch race can create two.
    """
    exps = client.search_experiments(filter_string=f"name = '{name}'")
    if not exps:
        return []
    runs = client.search_runs(
        [e.experiment_id for e in exps],
        max_results=5000,
        order_by=["attributes.start_time ASC"],
    )
    evaluated = [r for r in runs if any(k.startswith("eval/") for k in r.data.metrics)]
    return list({r.info.run_name: r for r in evaluated}.values())


def _hyperparameters(runs: list[Run]) -> pd.DataFrame:
    """Return the parameters whose value differs between runs, prefixed with hp/."""
    params = pd.DataFrame(
        [r.data.params for r in runs], index=[r.info.run_id for r in runs]
    )
    varying = [c for c in sorted(params) if params[c].nunique(dropna=False) > 1]
    return params[varying].add_prefix("hp/")


def _non_dominated(goods: np.ndarray) -> np.ndarray:
    """Flag the rows of an (n, 2) larger-is-better array that no other row dominates.

    A row with a missing objective is never flagged.
    """
    ok = np.isfinite(goods).all(axis=1)
    g = goods[ok]
    no_worse = (g[None, :, :] >= g[:, None, :]).all(axis=2)
    better = (g[None, :, :] > g[:, None, :]).any(axis=2)
    flags = np.zeros(len(goods), dtype=bool)
    flags[ok] = ~(no_worse & better).any(axis=1)
    return flags


def _pareto_flags(table: pd.DataFrame) -> pd.Series:
    """Flag non-dominated rows per strategy and tier, Q′ by its mode, Q″ minimised.

    The table must carry a RangeIndex.
    """
    q = pd.to_numeric(table["optimized_main"], errors="coerce")
    sec = pd.to_numeric(table["optimized_sec"], errors="coerce")
    goods = np.column_stack([q.where(table["maximize"], -q), -sec])
    flags = pd.Series(False, index=table.index)
    for idx in table.groupby(["strategy", "tier"]).groups.values():
        flags[idx] = _non_dominated(goods[idx])
    return flags


def _tier_tokens(label: str) -> set[str]:
    """Return the name parts that mark the evaluation scalars of one tier."""
    if label == "operational":
        return {"operational"}
    return {label, f"trate{label.replace('.', '_')}kHz"}


def _eval_records(row: pd.Series, foreign: set[str]) -> Iterator[dict]:
    head = {c: row[c] for c in EVAL_COLUMNS if c in row.index}
    for split, contexts in sorted(row["_evals"].items()):
        for context, scalars in sorted(contexts.items()):
            kept = {
                k: v
                for k, v in scalars.items()
                if not any(f"_{t}_" in f"_{k}_" for t in foreign)
            }
            yield {**head, "split": split, "context": context, **kept}


def _write_eval(rows: pd.DataFrame, foreign: set[str], path: Path) -> None:
    records = [rec for _, row in rows.iterrows() for rec in _eval_records(row, foreign)]
    table = pd.DataFrame(records).sort_values(["run_name", "split", "context"])
    leaves = sorted(set(table.columns) - set(EVAL_COLUMNS))
    table[EVAL_COLUMNS + leaves].to_csv(path, index=False)
    print(f"  wrote {path} ({len(table)} rows, {len(leaves)} scalar columns)")


def _write_main(rows: pd.DataFrame, path: Path) -> None:
    fixed = OLD_COLUMNS + NEW_COLUMNS
    hp = [c for c in rows.columns if c.startswith("hp/")]
    datasets = sorted(set(rows.columns) - set(fixed) - set(hp) - {"_label", "_evals"})
    rows.sort_values("run_name")[fixed + datasets + hp].to_csv(path, index=False)
    missing = rows["optimized_main"].isna().sum()
    print(f"  wrote {path} ({len(rows)} rows, {missing} without Q')")


def _stem(name: str, label: str, tier: str) -> str:
    """Return the file stem of a tier: physics_ae_pool, then physics_ae_q99_pool."""
    if label == "operational":
        return name
    return re.sub(r"(_pool)?$", rf"_{tier}\1", name, count=1)


def _write_tiers(table: pd.DataFrame, name: str, outdir: Path) -> None:
    labels = set(table["_label"])
    for (label, tier), rows in table.groupby(["_label", "tier"]):
        stem = _stem(name, label, tier)
        foreign = set().union(*(_tier_tokens(o) for o in labels - {label}))
        _write_main(rows, outdir / f"{stem}.csv")
        _write_eval(rows, foreign, outdir / f"{stem}_eval.csv")


def harvest_experiment(
    client: MlflowClient, name: str, outdir: Path, bkg_re: re.Pattern
) -> None:
    runs = _latest_runs(client, name)
    rows = [row for run in runs for row in _run_rows(client, run, bkg_re)]
    if not rows:
        print(f"  {name}: no evaluated runs, nothing written")
        return
    table = pd.DataFrame(rows).join(_hyperparameters(runs), on="run_id")
    table["experiment"] = name
    table["is_pareto"] = _pareto_flags(table)
    _write_tiers(table, name, outdir)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "experiments",
        nargs="*",
        help="MLflow experiment names (default: every experiment named *_pool).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=f"file:{REPO / 'logs' / 'mlflow' / 'mlruns'}",
        help="MLflow tracking URI (default: file:<repo>/logs/mlflow/mlruns).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO / "notebooks" / "pareto_effs",
        help="Output directory (default: <repo>/notebooks/pareto_effs).",
    )
    parser.add_argument(
        "--bkg-regex",
        type=re.compile,
        default="normal|SingleNeutrino|ZB_|reference",
        help="Datasets matching this regex are not signals.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    client = MlflowClient(args.tracking_uri)
    args.outdir.mkdir(parents=True, exist_ok=True)
    names = args.experiments or sorted(
        {e.name for e in client.search_experiments() if e.name.endswith("_pool")}
    )
    for name in names:
        print(f"harvesting {name}...")
        harvest_experiment(client, name, args.outdir, args.bkg_regex)


if __name__ == "__main__":
    main()
