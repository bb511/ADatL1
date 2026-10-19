import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/select_pool.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("select_pool", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_pool = _load_script()


def _rows(strategy, maximize, points, tier="250"):
    return pd.DataFrame(
        {
            "run_name": f"{strategy}_{run}",
            "run": run,
            "strategy": strategy,
            "experiment": "physics_toy_pool",
            "tier": tier,
            "maximize": maximize,
            "optimized_main": q,
            "optimized_sec": sec,
            "is_pareto": pareto,
            "test_mean_sig_eff": test,
        }
        for run, q, sec, pareto, test in points
    )


@pytest.fixture
def pool():
    # cap: the front a-d in goodness space (q, 2 - sec) is (0, 1), (0.2, 0.95),
    # (0.5, 0.5), (1, 0), so b lies farthest from the chord; b dominates e.
    cap = _rows(
        "cap",
        True,
        [
            ("a", 0.0, 1.0, True, 0.30),
            ("b", 0.2, 1.05, True, 0.50),
            ("c", 0.5, 1.5, True, 0.40),
            ("d", 1.0, 2.0, True, 0.20),
            ("e", 0.1, 1.9, False, 0.50),
        ],
    )
    stability = _rows(
        "stability",
        False,
        [
            ("x", 0.3, 1.0, True, 0.1),
            ("y", 0.1, 2.0, True, 0.3),
            ("z", 0.2, 3.0, False, 0.2),
        ],
    )
    ties = _rows(
        "cvar25",
        True,
        [
            ("t2", 0.7, 1.0, True, 0.2),
            ("t1", 0.7, 1.0, True, 0.1),
            ("t0", 9.0, 1.0, True, math.nan),
        ],
    )
    return pd.concat([cap, stability, ties], ignore_index=True)


def _pick(result, strategy):
    rows = result[result["strategy"] == strategy]
    assert len(rows) == 1
    return rows.iloc[0]


def test_picks_and_regrets(pool):
    cap = _pick(select_pool.select(pool), "cap")
    assert (cap["domain"], cap["model"], cap["tier"]) == ("physics", "toy", "250")
    assert (cap["n_runs"], cap["n_pareto"]) == (5, 4)
    assert (cap["best_q_run"], cap["knee_run"], cap["oracle_run"]) == ("d", "b", "b")
    assert cap["best_q_regret"] == pytest.approx(0.30)
    assert cap["knee_regret"] == pytest.approx(0.0)


def test_spearman_of_oriented_q(pool):
    result = select_pool.select(pool)
    # Ranks of q (1, 3, 4, 5, 2) against average ranks of test (2, 4.5, 3, 1, 4.5).
    assert _pick(result, "cap")["spearman"] == pytest.approx(-3.5 / math.sqrt(95))
    assert _pick(result, "stability")["spearman"] == pytest.approx(1.0)


def test_minimised_q_and_degenerate_knee(pool):
    stability = _pick(select_pool.select(pool), "stability")
    assert stability["best_q_run"] == "y" and stability["best_q_regret"] == 0.0
    assert stability["knee_run"] is None or pd.isna(stability["knee_run"])
    assert math.isnan(stability["knee_regret"])


def test_ties_go_to_the_first_name_and_missing_test_is_skipped(pool):
    ties = _pick(select_pool.select(pool), "cvar25")
    assert ties["best_q_run"] == "t1"
    assert ties["oracle_run"] == "t2"


def test_main_writes_csv_and_markdown(pool, tmp_path):
    pool.to_csv(tmp_path / "physics_toy_pool.csv", index=False)
    pd.DataFrame({"run_name": ["unused"]}).to_csv(
        tmp_path / "physics_toy_pool_eval.csv"
    )
    tables = sorted(str(p) for p in tmp_path.glob("*.csv"))
    select_pool.main([*tables, "--out", str(tmp_path / "selection")])
    result = pd.read_csv(tmp_path / "selection.csv", dtype={"tier": str})
    assert len(result) == 3
    lines = (tmp_path / "selection.md").read_text().splitlines()
    assert len(lines) == 2 + len(result)
    assert lines[0].startswith("| domain | model | tier | strategy |")
