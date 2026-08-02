#!/usr/bin/env python3
"""Harvest per-signal validation efficiencies at selected checkpoints.

For every run of the given MLflow experiments, reads the per-dataset
efficiency metric histories that the training-time AnomalyEfficiencyCallback
logs every validation epoch (``val/<dataset>/eff__ascore_full__brate_...``)
and extracts the full per-signal array at two checkpoints:

* ``strategy`` -- the epoch selected by the run's own strategy checkpoint,
  found as the arg-best of the exact ``val/summary/...`` curve that the
  checkpoint callback monitored (cvar25/cvar10 ema max, cap ema max, drift ema
  min, w1dist ema min, inferred from the ``<strategy>_t<trial>`` run name);
* ``last`` -- the final epoch (the 200-epoch model).

Each row also carries the run's Optuna pair (``eval/val/optimized_main/sec``)
and the mean efficiency over signal datasets (datasets not matching
``--bkg-regex``), i.e. the rule-(c) downstream scalar. One CSV per experiment
is written to ``notebooks/pareto_effs/`` for effs.nb-style ingestion.

Values are val-split (training-time validation loader). Runs without training
histories (e.g. eval-only reruns) are skipped with a note; when several runs
share a name, the newest one with histories wins.

A second mode, ``--mode eval``, harvests the flat ``eval/*`` metrics written by
the evaluation phase instead: one row per (run, split, checkpoint context) with
a column per scalar -- including the **test-split** per-signal efficiencies
``eff_<rate>_<dataset>`` produced by the post-campaign eval-only pass
(``train=false`` resubmission with the patched efficiency callback). Per run
name, the newest run entry carrying per-signal eval keys is preferred (the
eval-only rerun); output goes to ``<experiment>_eval.csv``.

Usage (any machine holding the mlruns folder, e.g. after copying from
clariden)::

    python scripts/analysis/harvest_pareto_effs.py physics_ae_pareto physics_ae_q99_pareto
    python scripts/analysis/harvest_pareto_effs.py --mode eval physics_ae_pareto ...
    python scripts/analysis/harvest_pareto_effs.py cifar10_ae_pareto \\
        --tracking-uri file:logs/mlflow/mlruns
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

from mlflow.tracking import MlflowClient

# Run-name prefix -> (monitored summary-curve prefix, direction).
STRATEGY_MONITORS = {
    "cvar25": ("val/summary/eff_cvar25_ema", "max"),
    "cvar10": ("val/summary/eff_cvar10_ema", "max"),
    "cap": ("val/summary/cap_ema_", "max"),
    "consistency": ("val/summary/consistency_ema_", "max"),
    "stability": ("val/summary/", "min_drift"),  # *_drift_ema, min
    "wasserstein": ("val/summary/w1dist_ema_", "min"),
}

EFF_KEY_RE = re.compile(r"^val/(?!summary/)([^/]+)/eff__")

# A run*_pareto.sh block carries both keys, so the catalogue is self-describing
# and no experiment-name -> file-path guessing is needed.
BLOCK_EXP_RE = re.compile(r"^#\s+experiment_name=(\S+?)\s*\\?$", re.M)
BLOCK_RUN_RE = re.compile(r"^#\s+run_name=(\S+?)\s*\\?$", re.M)


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def catalogue_run_names(repo: Path) -> dict:
    """experiment_name -> {run_name} over every checked-in Pareto catalogue.

    An experiment accumulates runs forever, so a front that was re-fetched after
    its Optuna study changed leaves the *previous* front's retrainings behind in
    the same MLflow experiment (the pre-bugfix svdd fronts are the live case:
    126 superseded run names across the four svdd experiments). Harvesting those
    silently mixes superseded models into the fronts, so --restrict-to-catalogues
    keeps only what the current catalogue actually asks for.
    """
    allowed: dict = {}
    for path in sorted(repo.glob("scripts/*/run*_pareto.sh")):
        text = path.read_text()
        for block in text.split("# ---"):
            exps = BLOCK_EXP_RE.findall(block)
            runs = BLOCK_RUN_RE.findall(block)
            if len(exps) == 1 and len(runs) == 1:
                allowed.setdefault(exps[0], set()).add(runs[0])
    return allowed


def history_values(client: MlflowClient, run_id: str, key: str) -> list:
    return [m.value for m in client.get_metric_history(run_id, key)]


def monitored_key(metric_keys: list, prefix: str, direction: str):
    """Find the summary curve a strategy checkpoint monitored."""
    if direction == "min_drift":
        cands = [k for k in metric_keys if k.startswith(prefix) and k.endswith("_drift_ema")]
    else:
        cands = [k for k in metric_keys if k.startswith(prefix)]
    return cands[0] if cands else None


def strategy_of(run_name: str):
    m = re.match(r"([a-z0-9]+)_t\d+$", run_name)
    return m.group(1) if m else None


def harvest_experiment(client, exp, bkg_re, outdir, allowed=None):
    runs = client.search_runs(
        [exp.experiment_id], max_results=5000, order_by=["attributes.start_time ASC"]
    )
    if allowed is not None:
        dropped = {r.info.run_name for r in runs} - allowed
        runs = [r for r in runs if r.info.run_name in allowed]
        if dropped:
            print(f"  restricted to catalogue: dropped {len(dropped)} superseded "
                  f"run name(s), e.g. {sorted(dropped)[:3]}")
    # Newest run with training histories per run name.
    by_name = {}
    for run in runs:
        if any(EFF_KEY_RE.match(k) for k in run.data.metrics):
            by_name[run.info.run_name] = run  # ascending start_time: last wins
        else:
            print(f"  note: {run.info.run_name} ({run.info.run_id[:8]}) has no "
                  "training eff histories (eval-only?), skipped")

    all_ds = sorted({
        EFF_KEY_RE.match(k).group(1)
        for run in by_name.values()
        for k in run.data.metrics
        if EFF_KEY_RE.match(k)
    })
    sig_ds = [d for d in all_ds if not bkg_re.search(d)]
    if not sig_ds:
        print(f"  warn: every dataset matches the background regex "
              f"{bkg_re.pattern!r}, mean_sig_eff will be empty")

    rows = []
    for run_name, run in sorted(by_name.items()):
        keys = list(run.data.metrics)
        eff_keys = {EFF_KEY_RE.match(k).group(1): k for k in keys if EFF_KEY_RE.match(k)}
        histories = {ds: history_values(client, run.info.run_id, k)
                     for ds, k in eff_keys.items()}
        # by_name only holds runs with at least one eff history, so this is
        # non-empty in practice -- but a run whose only history is empty would
        # otherwise crash the whole harvest on max() of an empty sequence.
        n_epochs = max((len(h) for h in histories.values()), default=0)
        if n_epochs == 0:
            print(f"  note: {run_name} has no efficiency history points, skipped")
            continue

        targets = {"last": n_epochs - 1}
        strat = strategy_of(run_name)
        mon = STRATEGY_MONITORS.get(strat)
        if mon:
            key = monitored_key(keys, *mon)
            if key:
                curve = history_values(client, run.info.run_id, key)
                best = max if mon[1] == "max" else min
                targets["strategy"] = curve.index(best(curve))
            else:
                print(f"  warn: {run_name}: monitored curve for '{strat}' not found")
        elif strat is not None:
            print(f"  warn: {run_name}: unknown strategy prefix '{strat}', last only")

        for ckpt_kind, epoch in sorted(targets.items()):
            effs = {
                ds: (h[epoch] if epoch < len(h) else "")
                for ds, h in histories.items()
            }
            sig_vals = [effs[d] for d in sig_ds if effs.get(d) != ""]
            row = {
                "run_name": run_name,
                "strategy": strat or "",
                "ckpt": ckpt_kind,
                "epoch": epoch,
                "optimized_main": run.data.metrics.get("eval/val/optimized_main", ""),
                "optimized_sec": run.data.metrics.get("eval/val/optimized_sec", ""),
                "mean_sig_eff": statistics.mean(sig_vals) if sig_vals else "",
                "n_signals": len(sig_vals),
            }
            row.update({ds: effs.get(ds, "") for ds in all_ds})
            rows.append(row)

    out = outdir / f"{exp.name}.csv"
    header = ["run_name", "strategy", "ckpt", "epoch", "optimized_main",
              "optimized_sec", "mean_sig_eff", "n_signals"] + all_ds
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out} ({len(rows)} rows, {len(sig_ds)} signal datasets)")


# Per-signal eval leaves are eff_<rate>_<dataset>; exclude the eff_med_<rate> /
# eff_min_<rate> summary leaves that unpatched evaluations log as well.
PER_SIGNAL_EVAL_RE = re.compile(r"^eval/(val|test)/.*/eff_(?!med_|min_)[^/]+_[^/]+$")


def harvest_experiment_eval(client, exp, outdir, allowed=None):
    """Harvest the flat eval/* metrics (one row per run x split x context)."""
    runs = client.search_runs(
        [exp.experiment_id], max_results=5000, order_by=["attributes.start_time ASC"]
    )
    if allowed is not None:
        dropped = {r.info.run_name for r in runs} - allowed
        runs = [r for r in runs if r.info.run_name in allowed]
        if dropped:
            print(f"  restricted to catalogue: dropped {len(dropped)} superseded "
                  f"run name(s), e.g. {sorted(dropped)[:3]}")
    # Per run name, prefer the newest entry with per-signal eval keys (the
    # patched eval-only rerun), else the newest with any eval metrics.
    by_name = {}
    for run in runs:
        keys = run.data.metrics
        has_eval = any(k.startswith("eval/") for k in keys)
        has_per_signal = any(PER_SIGNAL_EVAL_RE.match(k) for k in keys)
        if not has_eval:
            continue
        prev = by_name.get(run.info.run_name)
        if prev is None or has_per_signal or not prev[1]:
            by_name[run.info.run_name] = (run, has_per_signal)

    rows_by_key = {}
    leaves = set()
    for run_name, (run, has_per_signal) in sorted(by_name.items()):
        if not has_per_signal:
            print(f"  note: {run_name}: no per-signal eval keys "
                  "(eval-only pass not run with patched callback?)")
        for key, value in run.data.metrics.items():
            if not key.startswith("eval/"):
                continue
            parts = key.split("/")
            split, context, leaf = parts[1], "/".join(parts[2:-1]), parts[-1]
            row = rows_by_key.setdefault(
                (run_name, split, context),
                {
                    "run_name": run_name,
                    "strategy": strategy_of(run_name) or "",
                    "split": split,
                    "context": context,
                },
            )
            row[leaf] = value
            leaves.add(leaf)

    out = outdir / f"{exp.name}_eval.csv"
    header = ["run_name", "strategy", "split", "context"] + sorted(leaves)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_by_key[k] for k in sorted(rows_by_key))
    print(f"  wrote {out} ({len(rows_by_key)} rows, {len(leaves)} scalar columns, "
          f"{len(by_name)} runs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", nargs="+", help="MLflow experiment names.")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: file:<repo>/logs/mlflow/mlruns).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output dir for CSVs (default: <repo>/notebooks/pareto_effs).",
    )
    parser.add_argument(
        "--bkg-regex",
        default="normal|SingleNeutrino|ZB_|reference|shifted_normal",
        help="Datasets matching this regex are excluded from mean_sig_eff.",
    )
    parser.add_argument(
        "--mode",
        choices=["history", "eval"],
        default="history",
        help="'history': per-signal val arrays from training metric histories "
        "(default); 'eval': flat eval/* metrics incl. test-split per-signal "
        "efficiencies from the eval-only pass.",
    )
    parser.add_argument(
        "--restrict-to-catalogues",
        action="store_true",
        help="Keep only run names listed for that experiment in "
        "scripts/*/run*_pareto.sh. Off by default (harvest everything); needed "
        "wherever a front was re-fetched, since the superseded front's "
        "retrainings still live in the same MLflow experiment.",
    )
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    uri = args.tracking_uri or f"file:{repo / 'logs' / 'mlflow' / 'mlruns'}"
    outdir = Path(args.outdir) if args.outdir else repo / "notebooks" / "pareto_effs"
    outdir.mkdir(parents=True, exist_ok=True)
    bkg_re = re.compile(args.bkg_regex)

    catalogues = catalogue_run_names(repo) if args.restrict_to_catalogues else {}

    client = MlflowClient(uri)
    for name in args.experiments:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            print(f"experiment {name!r} not found in {uri}")
            continue
        allowed = None
        if args.restrict_to_catalogues:
            allowed = catalogues.get(name)
            if not allowed:
                print(f"  warn: {name} has no catalogue block, harvesting unrestricted")
        print(f"harvesting {name}...")
        if args.mode == "eval":
            harvest_experiment_eval(client, exp, outdir, allowed)
        else:
            harvest_experiment(client, exp, bkg_re, outdir, allowed)


if __name__ == "__main__":
    main()
