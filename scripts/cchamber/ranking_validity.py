#!/usr/bin/env python3
"""Ranking validity of the frozen CAP configuration, threshold drift and Wasserstein.

The secondary analysis ``ranking_validity`` of ``registration_limits.yaml``: does the
value a criterion takes at the checkpoint its own chain selects rank the candidates of
a family as their endpoint at that checkpoint does? The inputs are ``selection.csv``
files of ``offline_select.py``, whose rows name their criterion through the strategy
and the configuration:

    cap_frozen                    strategy cap, a configuration run without --pairs
    cap_frozen_nearest_neighbour  strategy cap, --pairs pairs_nearest_neighbour.pt
    cap_frozen_random             strategy cap, --pairs pairs_random.pt
    drift, wasserstein            strategies drift and w1

Per criterion, a candidate's value is ``q_main`` of its selected row and its endpoint
is ``efficiency_val_mean`` (efficiency at 1 %, the headline) or ``auprc_val_mean`` of
the same row, both averaged over the seeds. Over the candidates of each family:

    rho          Spearman rank correlation of value and endpoint
    p            two-sided permutation p-value, (1 + #{abs(rho*) >= abs(rho)}) / (1 + N),
                 for N permutations of the endpoint
    p_holm       Holm's step-down adjustment over all criteria and families of one
                 endpoint
    rho_drop3    rho without the three candidates of lowest endpoint (ties drop the
                 lower candidate id first)
    rho_partial  partial Spearman with the seed-averaged selected epoch: the
                 correlation of the residuals of the value and endpoint ranks after a
                 least-squares line on the epoch ranks

Values are not re-signed: CAP is maximised and drift and Wasserstein are minimised, so
a valid ranking gives a positive rho for the CAP criteria and a negative one for the
other two.

Outputs, in ``--out``: ``ranking_validity_<endpoint>.csv`` and a figure
``ranking_validity_<endpoint>.pdf`` with one panel per criterion and family.

Usage
-----
    python scripts/cchamber/ranking_validity.py --selection \\
        frozen/selection.csv nn/selection.csv random/selection.csv \\
        baselines/selection.csv --out ranking
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

CRITERIA = [
    "cap_frozen",
    "drift",
    "wasserstein",
    "cap_frozen_nearest_neighbour",
    "cap_frozen_random",
]
ENDPOINTS = {"efficiency": "efficiency_val_mean", "auprc": "auprc_val_mean"}
COLUMNS = {"value": "q_main", "epoch": "selected_epoch"}
STRATEGIES = {"drift": "drift", "w1": "wasserstein"}
PERMUTATIONS, PERMUTATION_SEED, DROPPED = 10000, 4242, 3
FIELDS = ["criterion", "family", "rho", "p", "p_holm", "rho_drop3", "rho_partial", "n"]


def pairing_of(config: str) -> str:
    """Pairing of an offline_select.py configuration name: its --pairs stem, or cdf."""
    return config.partition("_pairs_")[2] or "cdf"


def _criterion(row: dict) -> str:
    if row["strategy"] != "cap":
        return STRATEGIES[row["strategy"]]

    pairing = pairing_of(row["config"])
    return "cap_frozen" if pairing == "cdf" else f"cap_frozen_{pairing}"


def _check_rows(rows: list[dict]) -> None:
    keys = [(r["criterion"], r["family"], r["candidate"], r["seed"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit(
            "Two rows share a criterion and run: pass one CAP configuration, with "
            "--strategies cap for the pairing runs."
        )
    missing = set(CRITERIA) - {r["criterion"] for r in rows}
    if missing:
        raise SystemExit(f"No selection rows for {sorted(missing)}.")


def _read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as file:
            rows += [{**r, "criterion": _criterion(r)} for r in csv.DictReader(file)]

    _check_rows(rows)
    return rows


def _candidate_means(rows: list[dict], column: str) -> dict[tuple, float]:
    """Seed mean of one column per (criterion, family, candidate)."""
    values = defaultdict(list)
    for row in rows:
        key = (row["criterion"], row["family"], int(row["candidate"]))
        values[key].append(float(row[column]))

    return {key: float(np.mean(v)) for key, v in values.items()}


def _family_arrays(means: dict[str, dict], criterion: str, family: str) -> dict:
    """(n_candidates,) arrays of every averaged column, ordered by candidate id."""
    keys = sorted(k for k in means["value"] if k[:2] == (criterion, family))
    return {name: np.array([m[k] for k in keys]) for name, m in means.items()}


def _centred_ranks(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x)
    return ranks - ranks.mean()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _centred_ranks(x), _centred_ranks(y)
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def _permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    rng = np.random.default_rng(PERMUTATION_SEED)
    rx, ry = _centred_ranks(x), _centred_ranks(y)
    # A permutation keeps the norm of ry, so the numerators order the |rho|. Centred
    # ranks are multiples of 1/2, so the products are exact and ties compare equal.
    null = rng.permuted(np.tile(ry, (PERMUTATIONS, 1)), axis=1) @ rx
    extreme = np.abs(null) >= abs(rx @ ry)
    return float((1 + extreme.sum()) / (1 + PERMUTATIONS))


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm's adjusted p-values: the running maximum of (m - j) p_(j), capped at 1."""
    order = np.argsort(p, kind="stable")
    scaled = (len(p) - np.arange(len(p))) * p[order]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def _drop_lowest(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.argsort(y, kind="stable")[DROPPED:]
    return _spearman(x[keep], y[keep])


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(z)), rankdata(z)])
    residuals = [
        r - design @ np.linalg.lstsq(design, r, rcond=None)[0]
        for r in (rankdata(x), rankdata(y))
    ]
    return float(np.corrcoef(*residuals)[0, 1])


def _test_row(criterion: str, family: str, arrays: dict) -> dict:
    x, y = arrays["value"], arrays["endpoint"]
    return {
        "criterion": criterion,
        "family": family,
        "rho": _spearman(x, y),
        "p": _permutation_p(x, y),
        "rho_drop3": _drop_lowest(x, y),
        "rho_partial": _partial_spearman(x, y, arrays["epoch"]),
        "n": len(x),
    }


def _table(rows: list[dict], endpoint: str) -> tuple[list[dict], dict]:
    """Test rows of one endpoint, Holm-adjusted together, and the arrays behind them."""
    columns = {**COLUMNS, "endpoint": ENDPOINTS[endpoint]}
    means = {name: _candidate_means(rows, col) for name, col in columns.items()}
    families = sorted({row["family"] for row in rows})
    arrays = {(c, f): _family_arrays(means, c, f) for c in CRITERIA for f in families}
    table = [_test_row(c, f, a) for (c, f), a in arrays.items()]
    for row, p_holm in zip(table, _holm(np.array([r["p"] for r in table]))):
        row["p_holm"] = float(p_holm)

    return table, arrays


def _plot(arrays: dict, endpoint: str, path: Path) -> None:
    """Endpoint against criterion value, a row per criterion and a column per family.

    The criteria differ in scale and units, so each panel has its own x-axis.
    """
    families = sorted({family for _, family in arrays})
    shape = (len(CRITERIA), len(families))
    size = (2.6 * shape[1], 2.0 * shape[0])
    fig, axes = plt.subplots(*shape, figsize=size, sharey="col", squeeze=False)
    for (i, criterion), (j, family) in itertools.product(
        enumerate(CRITERIA), enumerate(families)
    ):
        values = arrays[criterion, family]
        axes[i, j].scatter(values["value"], values["endpoint"], s=12, color="0.2")
        axes[i, j].set_xlabel(criterion, fontsize=8)
        axes[0, j].set_title(family)

    fig.supylabel(ENDPOINTS[endpoint])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rows = _read_rows(args.selection)
    args.out.mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINTS:
        table, arrays = _table(rows, endpoint)
        _write_csv(args.out / f"ranking_validity_{endpoint}.csv", table)
        _plot(arrays, endpoint, args.out / f"ranking_validity_{endpoint}.pdf")


if __name__ == "__main__":
    main()
