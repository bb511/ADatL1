#!/usr/bin/env python3
"""Per-family tables of the label-free selection criteria beside trivial baselines.

Three inputs of the chamber analysis are read:

    --design-table  design_table.csv of ``scripts/cchamber/design_study.py table
                    --held-out``, one row per strategy and one CAP configuration
    --selection     selection.csv files of ``scripts/cchamber/offline_select.py``, the
                    selected row of every run under every strategy
    --ranking       ranking_validity_efficiency.csv of
                    ``scripts/cchamber/ranking_validity.py``

Per family, ``criteria_<family>.csv`` and ``criteria_<family>.md`` are written to
``--out``. Their rows are the strategies present, in the order cap (the frozen
configuration), drift, w1, the established criteria (npd, em, mv, rtm, eag, entropy),
the consensus criteria (udr, mc), sds, last and the reference; their columns are

    selected        the candidate a strategy selects for each seed (its best run),
                    seeds in increasing order
    efficiency, se, auprc
                    criterion_<family>, se_<family> and auprc_<family> of the design
                    table: the validation efficiency at the 1 % cut of the selected
                    runs, its crossed-bootstrap standard error, and the AUPRC
    random_mean     mean efficiency_val_mean over every (candidate, seed) row of the
                    strategy, the expectation for one candidate drawn per seed
    band_5, band_95 5 % and 95 % quantiles (numpy's linear interpolation) of the seed
                    mean of efficiency_val_mean over every pick of one candidate per
                    seed, enumerated exactly (17^3 picks for 17 candidates and 3 seeds)
    pool_median     median over the candidates of their seed-mean efficiency
    default         the seed-mean efficiency of candidate 0, the default configuration
    rho, p_holm     ranking validity of the strategy's criterion, efficiency endpoint

A strategy's baselines read the efficiency at the epochs its own chain selects. The
last pseudo-strategy keeps every run at its last epoch and selects no candidate, so its
row holds the baselines alone, and the reference row has none. The selection rows are
those of the declared score ``ascore/full``, and the cap rows those of the design
table's CAP configuration.

Usage
-----
    python scripts/analysis/criteria_table.py \\
        --design-table wt_table/design_table.csv \\
        --selection frozen/selection.csv baselines/selection.csv \\
            criteria/selection.csv \\
        --ranking ranking_criteria/ranking_validity_efficiency.csv --out criteria
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.cchamber.ranking_validity import STRATEGIES  # noqa: E402

DECLARED_KEY = "ascore/full"
REFERENCE = "reference"
ORDER = ["cap", "drift", "w1", "npd", "em", "mv", "rtm", "eag", "entropy", "udr", "mc"]
ORDER += ["sds", "last", REFERENCE]
# The ranking-validity criterion of each strategy; the frozen CAP row is cap_frozen.
CRITERION = {"cap": "cap_frozen", **STRATEGIES}
COLUMNS = ["strategy", "selected", "efficiency", "se", "auprc", "random_mean"]
COLUMNS += ["band_5", "band_95", "pool_median", "default", "rho", "p_holm"]
# Columns printed as they are; the others are numbers.
TEXT = ("strategy", "selected")
BAND = (0.05, 0.95)


def _read(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as file:
            rows += csv.DictReader(file)
    return rows


def _design_rows(path: Path) -> dict[str, dict]:
    rows = _read([path])
    if sum(row["strategy"] == "cap" for row in rows) != 1:
        raise SystemExit(f"{path} must hold exactly one CAP configuration.")
    return {row["strategy"]: row for row in rows}


def _selection_units(paths: list[Path], cap_config: str) -> dict:
    """Rows of the declared score by (strategy, family), the cap rows of one
    configuration."""
    units = defaultdict(list)
    for row in _read(paths):
        other_cap = row["strategy"] == "cap" and row["config"] != cap_config
        if row["score_key"] == DECLARED_KEY and not other_cap:
            units[(row["strategy"], row["family"])].append(row)
    return dict(units)


def _efficiency(units: list[dict]) -> tuple[np.ndarray, list[int]]:
    """efficiency_val_mean per candidate and seed, (candidates, seeds), and the
    candidate ids in increasing order."""
    candidates = sorted({int(u["candidate"]) for u in units})
    seeds = sorted({int(u["seed"]) for u in units})
    cells = {(int(u["candidate"]), int(u["seed"])) for u in units}
    if len(units) != len(cells) or len(cells) != len(candidates) * len(seeds):
        strategy = units[0]["strategy"]
        raise SystemExit(f"{strategy}: not one row per candidate and seed.")
    matrix = np.empty((len(candidates), len(seeds)))
    for u in units:
        i, j = candidates.index(int(u["candidate"])), seeds.index(int(u["seed"]))
        matrix[i, j] = float(u["efficiency_val_mean"])
    return matrix, candidates


def _all_picks(matrix: np.ndarray) -> np.ndarray:
    """Seed mean of every pick of one candidate per seed, (candidates ** seeds,)."""
    return np.mean(np.meshgrid(*matrix.T, indexing="ij"), axis=0).ravel()


def _baselines(units: list[dict]) -> dict:
    matrix, candidates = _efficiency(units)
    if 0 not in candidates:
        raise SystemExit(f"{units[0]['strategy']}: no default candidate 0.")
    seed_means, band = matrix.mean(axis=1), np.quantile(_all_picks(matrix), BAND)
    row = {"random_mean": matrix.mean(), "band_5": band[0], "band_95": band[1]}
    row.update(pool_median=np.median(seed_means))
    return {**row, "default": seed_means[candidates.index(0)]}


def _selected(units: list[dict]) -> str:
    best = [(int(u["seed"]), u["candidate"]) for u in units if u["best"] == "True"]
    return " / ".join(candidate for _, candidate in sorted(best))


def _design_columns(design: dict | None, family: str) -> dict:
    if design is None:
        return {}
    quantities = {"efficiency": "criterion", "se": "se", "auprc": "auprc"}
    return {column: design[f"{q}_{family}"] for column, q in quantities.items()}


def _row(strategy: str, family: str, data: SimpleNamespace) -> dict:
    row = {"strategy": strategy, **_design_columns(data.design.get(strategy), family)}
    units = data.units.get((strategy, family))
    if units:
        row.update(_baselines(units))
        row["selected"] = "" if strategy == "last" else _selected(units)
    ranking = data.ranking.get((CRITERION.get(strategy), family), {})
    row.update({k: ranking[k] for k in ("rho", "p_holm") if k in ranking})
    return row


def _cell(value, column: str) -> str:
    if value is None or value == "":
        return ""
    return str(value) if column in TEXT else f"{float(value):.4f}"


def _markdown(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(c), c) for c in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _load(args: argparse.Namespace) -> SimpleNamespace:
    design = _design_rows(args.design_table)
    units = _selection_units(args.selection, design["cap"]["config"])
    ranking = {(r["criterion"], r["family"]): r for r in _read([args.ranking])}
    families = sorted({family for _, family in units} - {REFERENCE})
    data = {"design": design, "units": units, "ranking": ranking}
    return SimpleNamespace(**data, families=families)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--design-table", type=Path, required=True)
    parser.add_argument("--selection", nargs="+", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    data = _load(args)
    args.out.mkdir(parents=True, exist_ok=True)
    for family in data.families:
        present = [s for s in ORDER if s in data.design or (s, family) in data.units]
        rows = [_row(strategy, family, data) for strategy in present]
        _write_csv(args.out / f"criteria_{family}.csv", rows)
        (args.out / f"criteria_{family}.md").write_text(_markdown(rows))


if __name__ == "__main__":
    main()
