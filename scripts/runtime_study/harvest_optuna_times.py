#!/usr/bin/env python3
"""Harvest per-trial wall-clock durations from the Optuna study databases.

``scripts/optuna/fetch_optuna_pareto.py`` exports the trials' objective values and
parameters but drops their timing (its ``attrs=`` omits ``duration`` and
``datetime_*``). The sweeps were run with ``logger=none``, so the study
databases are the *only* record of how long a hyperparameter-search trial took,
and hence the only way to price the sweeps for the paper's compute table.

One row per completed trial is written to
``notebooks/run_times/optuna_<domain>_<model>.csv``::

    domain, model, strategy, tier, study, number, minutes

``strategy`` and ``tier`` are parsed from the study name, which follows
``<strategy>_vs_<objective>[q99]_b16k``: the tier is ``q99`` when the second
objective carries that marker and ``250`` otherwise -- the same rule
``make_pareto_scripts.py`` uses.

Timing is read with raw ``sqlite3`` over the ``trials`` table rather than
through the Optuna API, so the databases' 4.7.0 schema does not have to match
whatever Optuna is installed. Durations are computed in SQL (``julianday``) for
the same reason -- no date parsing, so no dependency on the Python version.

Read-only: the databases are opened with ``mode=ro``.

Usage (on olqti, where the databases live; see this directory's README)::

    conda activate optuna-ui
    python scripts/runtime_study/harvest_optuna_times.py

By default only the four strategies of the paper's table are exported, at the
250 Hz tier; pass ``--all-studies`` to export every study in every database.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

# Study-name prefix -> the column it fills in the paper's table. Matches
# STRATEGIES in scripts/optuna/make_pareto_scripts.py.
TABLE_STRATEGIES = ["cvar25eff", "drift", "wasserstein", "cap"]

# Databases holding side studies rather than the model/strategy sweeps.
SKIP_DB_STEMS = {"ae_cap_exploration"}

# Minutes per day, for julianday() differences.
MINUTES_PER_DAY = 1440.0


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".project-root").exists():
            return p
    return start


def split_study(study_name: str) -> tuple[str, str]:
    """``<strategy>_vs_<objective>`` -> (strategy, tier)."""
    strategy, _, objective = study_name.partition("_vs_")
    return strategy, ("q99" if "q99" in objective else "250")


def trial_times(conn: sqlite3.Connection, study_id: int) -> list[tuple[int, float]]:
    """(trial number, minutes) for every COMPLETE trial of one study."""
    rows = conn.execute(
        "SELECT number, (julianday(datetime_complete) - julianday(datetime_start))"
        " * ? FROM trials WHERE study_id = ? AND state = 'COMPLETE'"
        " AND datetime_start IS NOT NULL AND datetime_complete IS NOT NULL",
        (MINUTES_PER_DAY, study_id),
    )
    return [(n, m) for n, m in rows if m is not None]


def harvest_db(db_path: Path, domain: str, keep: set | None, outdir: Path) -> int:
    """Write one CSV for a single <model>.db; returns the number of trials."""
    model = db_path.stem
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = []
    for study_id, study_name in conn.execute(
        "SELECT study_id, study_name FROM studies ORDER BY study_name"
    ):
        strategy, tier = split_study(study_name)
        if keep is not None and (strategy not in keep or tier != "250"):
            continue
        for number, minutes in trial_times(conn, study_id):
            rows.append(
                {
                    "domain": domain,
                    "model": model,
                    "strategy": strategy,
                    "tier": tier,
                    "study": study_name,
                    "number": number,
                    "minutes": f"{minutes:.6f}",
                }
            )
    conn.close()

    if not rows:
        print(f"  {domain}/{model}: no matching studies")
        return 0

    out_path = outdir / f"optuna_{domain}_{model}.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_study = {}
    for r in rows:
        per_study.setdefault(r["strategy"], 0)
        per_study[r["strategy"]] += 1
    summary = " ".join(f"{s}={n}" for s, n in sorted(per_study.items()))
    print(f"  wrote {out_path.name}: {len(rows)} trials  ({summary})")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default="/data/deodagiu/adl1t/logs/optuna",
        help="Directory holding <domain>/<model>.db (default: the olqti path).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output dir for CSVs (default: <repo>/notebooks/run_times).",
    )
    parser.add_argument(
        "--all-studies",
        action="store_true",
        help="Export every study, not just the paper's four strategies at 250 Hz.",
    )
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    outdir = Path(args.outdir) if args.outdir else repo / "notebooks" / "run_times"
    outdir.mkdir(parents=True, exist_ok=True)
    keep = None if args.all_studies else set(TABLE_STRATEGIES)

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"optuna root not found: {root}")

    total = 0
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        print(f"{domain_dir.name}:")
        for db_path in sorted(domain_dir.glob("*.db")):
            if db_path.stem in SKIP_DB_STEMS:
                print(f"  skipping {db_path.name} (side study)")
                continue
            total += harvest_db(db_path, domain_dir.name, keep, outdir)
    print(f"\ntotal trials: {total}")


if __name__ == "__main__":
    main()
