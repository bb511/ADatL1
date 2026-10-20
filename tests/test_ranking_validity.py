import csv
import itertools

import numpy as np
import pytest

from scripts.cchamber import ranking_validity as ranking

FAMILIES, CANDIDATES, SEEDS = ("ae", "vae"), range(5), (101, 202, 303)
STRATEGIES = ("drift", "w1", "npd", "mv", "last")


def write_selection(path):
    rng = np.random.default_rng(3)
    rows = []
    for strategy, family, k, seed in itertools.product(
        STRATEGIES, FAMILIES, CANDIDATES, SEEDS
    ):
        efficiency = 0.1 * k + 0.01 * rng.random()
        # npd ranks the ae candidates exactly as their endpoint does.
        perfect = (strategy, family) == ("npd", "ae")
        rows.append(
            {
                "config": "none",
                "strategy": strategy,
                "family": family,
                "candidate": k,
                "seed": seed,
                "q_main": efficiency if perfect else rng.random(),
                "selected_epoch": int(rng.integers(0, 150)),
                "efficiency_val_mean": efficiency,
                "auprc_val_mean": rng.random(),
            }
        )
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_holm_family_follows_the_criteria(tmp_path):
    selection = write_selection(tmp_path / "selection.csv")
    criteria = ["drift", "npd", "mass_volume"]
    out = tmp_path / "out"
    argv = ["--selection", str(selection), "--criteria", *criteria]
    ranking.main([*argv, "--out", str(out)])

    with open(out / "ranking_validity_efficiency.csv", newline="") as file:
        table = list(csv.DictReader(file))
    pairs = {(row["criterion"], row["family"]) for row in table}
    assert pairs == set(itertools.product(criteria, FAMILIES))
    p = np.array([float(row["p"]) for row in table])
    p_holm = np.array([float(row["p_holm"]) for row in table])
    # Holm's smallest adjusted p-value is the family size times the smallest p.
    assert p_holm.min() == pytest.approx(min(1.0, len(table) * p.min()))
    (npd,) = (r for r in table if (r["criterion"], r["family"]) == ("npd", "ae"))
    assert float(npd["rho"]) == pytest.approx(1.0)
