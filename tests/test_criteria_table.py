import csv
import itertools

import pytest

from scripts.analysis import criteria_table as table

SEEDS = (101, 202, 303)
# Seed-mean efficiency 0.2 + 0.1 (i + j + k) for the pick (i, j, k) of candidates.
EFFICIENCY = {
    (c, s): 0.1 + 0.3 * c + 0.1 * j for c in range(3) for j, s in enumerate(SEEDS)
}
OFFSETS = {"cap": 0.0, "drift": 0.01, "npd": 0.02, "last": 0.03}
BEST = {"cap": (2, 0, 1), "drift": (1, 1, 1), "npd": (0, 2, 2), "last": (0, 0, 0)}


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def selection_rows():
    rows = []
    for strategy, (c, s) in itertools.product(OFFSETS, EFFICIENCY):
        best = BEST[strategy][SEEDS.index(s)] == c
        row = {"config": "frozen" if strategy == "cap" else "none"}
        row.update(strategy=strategy, family="ae", candidate=c, seed=s)
        row.update(score_key="ascore/full", best=str(best))
        row["efficiency_val_mean"] = EFFICIENCY[c, s] + OFFSETS[strategy]
        rows.append(row)
        # Another CAP configuration, which the table must leave out.
        if strategy == "cap":
            rows.append({**rows[-1], "config": "other", "efficiency_val_mean": 9.0})
    return rows


def design_rows():
    rows = []
    for i, strategy in enumerate(("cap", "drift", "npd", "reference")):
        config = "frozen" if strategy == "cap" else strategy
        row = {"config": config, "strategy": strategy, "criterion_ae": 0.6 + 0.01 * i}
        rows.append({**row, "se_ae": 0.05, "auprc_ae": 0.7})
    return rows


def test_criteria_table_rows_and_baselines(tmp_path):
    ranking = [
        {"criterion": "cap_frozen", "family": "ae", "rho": 0.5, "p_holm": 0.2},
        {"criterion": "npd", "family": "ae", "rho": -0.1, "p_holm": 1.0},
    ]
    argv = ["--design-table", str(write_csv(tmp_path / "design.csv", design_rows()))]
    argv += ["--selection", str(write_csv(tmp_path / "sel.csv", selection_rows()))]
    argv += ["--ranking", str(write_csv(tmp_path / "ranking.csv", ranking))]
    table.main([*argv, "--out", str(tmp_path / "out")])

    with open(tmp_path / "out" / "criteria_ae.csv", newline="") as file:
        rows = {row["strategy"]: row for row in csv.DictReader(file)}
    assert list(rows) == ["cap", "drift", "npd", "last", "reference"]
    cap = rows["cap"]
    assert cap["selected"] == "2 / 0 / 1"
    assert [float(cap[k]) for k in ("efficiency", "se", "auprc")] == [0.6, 0.05, 0.7]
    assert (float(cap["rho"]), float(cap["p_holm"])) == (0.5, 0.2)
    # Of the 27 picks, the sums i + j + k at ranks 1.3 and 24.7 are 1 and 5.
    hand = {"random_mean": 0.5, "band_5": 0.3, "band_95": 0.7, "pool_median": 0.5}
    hand["default"] = 0.2
    for strategy in ("cap", "drift", "npd", "last"):
        for column, value in hand.items():
            expected = value + OFFSETS[strategy]
            assert float(rows[strategy][column]) == pytest.approx(expected)
    assert rows["drift"]["rho"] == rows["drift"]["p_holm"] == ""
    assert rows["last"]["selected"] == rows["last"]["efficiency"] == ""
    assert rows["reference"]["random_mean"] == ""
    assert float(rows["reference"]["efficiency"]) == pytest.approx(0.63)
    markdown = (tmp_path / "out" / "criteria_ae.md").read_text().splitlines()
    assert markdown[0] == "| " + " | ".join(table.COLUMNS) + " |"
    assert len(markdown) == 2 + len(rows)
