#!/usr/bin/env python3
"""Pick one model per strategy from the harvested pools, with each pick's regret.

Reads the tables that ``harvest_pool_effs.py`` writes (``*_eval.csv`` files are skipped)
and applies the picking rules of ``notebooks/lib/effs_rules.wl`` to every (domain, model,
tier, strategy): the best-Q′ pick over every run of the pool, the knee of the strategy's
Pareto rows in (Q′, Q″), and the oracle, the best test metric over every run. The regret
of a pick is the oracle's test metric minus the pick's; Spearman's ρ ranks the
goodness-oriented Q′ against the test metric over the pool. The test metric is the mean
signal efficiency on the test split, the ``eff`` of the notebooks, and ties go to the
first run in name order, as in ``argBest``. One CSV and one Markdown table are written.

Usage::

    python scripts/analysis/select_pool.py notebooks/pareto_effs/*_pool*.csv \\
        --out notebooks/pareto_effs/pool_selection
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TEST = "test_mean_sig_eff"
KEYS = ["domain", "model", "tier", "strategy"]


def _maximize(group: pd.DataFrame) -> bool:
    return bool(group["maximize"].iloc[0])


def _oriented_q(rows: pd.DataFrame, maximize: bool) -> pd.Series:
    return rows["optimized_main"] if maximize else -rows["optimized_main"]


def _arg_best(values: pd.Series, maximize: bool) -> str | None:
    """Return the label of the best value, the first in name order among equals."""
    values = values.dropna().sort_index()
    if values.empty:
        return None
    best = values.max() if maximize else values.min()
    return values.index[values == best][0]


def _best_q(group: pd.DataFrame) -> str | None:
    rows = group.dropna(subset=["optimized_main", TEST])
    return _arg_best(rows["optimized_main"], _maximize(group))


def _knee(group: pd.DataFrame) -> str | None:
    """Return the Pareto row farthest from the chord joining the front's ends.

    Port of kneePick: both objectives goodness-oriented and min-max normalised; fewer
    than three distinct points, or an objective constant over them, leave no knee.
    """
    rows = group[group["is_pareto"]].dropna(
        subset=["optimized_main", "optimized_sec", TEST]
    )
    xy = pd.DataFrame(
        {"x": _oriented_q(rows, _maximize(group)), "y": -rows["optimized_sec"]}
    )
    if len(xy.drop_duplicates()) < 3 or (xy.max() == xy.min()).any():
        return None
    xy = (xy - xy.min()) / (xy.max() - xy.min())
    xy = xy.assign(name=xy.index).sort_values(["x", "y", "name"])
    (x1, y1), (x2, y2) = xy[["x", "y"]].iloc[0], xy[["x", "y"]].iloc[-1]
    dist = ((x2 - x1) * (y1 - xy["y"]) - (x1 - xy["x"]) * (y2 - y1)).abs()
    return dist.iloc[1:-1].idxmax()


def _spearman(group: pd.DataFrame) -> float:
    """Return Spearman's rho of goodness-oriented Q′ against the test metric."""
    rows = group.dropna(subset=["optimized_main", TEST])
    if len(rows) < 3:
        return math.nan
    ranks = np.column_stack(
        [_oriented_q(rows, _maximize(group)).rank(), rows[TEST].rank()]
    )
    if (ranks.std(axis=0) == 0).any():
        return math.nan
    return float(np.corrcoef(ranks, rowvar=False)[0, 1])


def _summary(group: pd.DataFrame) -> dict:
    """Return the picks of one (domain, model, tier, strategy), indexed by run_name."""
    row = {"n_runs": len(group), "n_pareto": int(group["is_pareto"].sum())}
    picks = {
        "best_q": _best_q(group),
        "knee": _knee(group),
        "oracle": _arg_best(group[TEST], True),
    }
    for rule, pick in picks.items():
        row[f"{rule}_run"] = group.at[pick, "run"] if pick else None
        row[f"{rule}_test"] = group.at[pick, TEST] if pick else math.nan
    row["best_q_regret"] = row["oracle_test"] - row["best_q_test"]
    row["knee_regret"] = row["oracle_test"] - row["knee_test"]
    row["spearman"] = _spearman(group)
    return row


def select(table: pd.DataFrame) -> pd.DataFrame:
    parts = table["experiment"].str.extract(r"^(?P<domain>[^_]+)_(?P<model>.+)_pool$")
    table = table.join(parts).set_index("run_name")
    groups = table.groupby(KEYS, dropna=False)
    return pd.DataFrame([dict(zip(KEYS, k)) | _summary(g) for k, g in groups])


def _cell(value) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return "" if value is None else str(value)


def _markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + " --- |" * table.shape[1]]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csvs", nargs="+", type=Path, help="Harvested pool tables.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "notebooks" / "pareto_effs" / "pool_selection",
        help="Output path without suffix; .csv and .md are appended.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = [p for p in args.csvs if not p.name.endswith("_eval.csv")]
    dtypes = {"run_name": str, "run": str, "tier": str}
    table = pd.concat([pd.read_csv(p, dtype=dtypes) for p in paths], ignore_index=True)
    result = select(table)
    result.to_csv(args.out.with_name(args.out.name + ".csv"), index=False)
    args.out.with_name(args.out.name + ".md").write_text(_markdown(result))
    print(f"wrote {args.out}.csv and .md ({len(result)} rows from {len(paths)} tables)")


if __name__ == "__main__":
    main()
