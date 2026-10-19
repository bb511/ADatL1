import importlib.util
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/harvest_pool_effs.py"
EXPERIMENT = "physics_toy_pool"
SIGNALS = ["sigA", "sigB", "sigC", "sigD", "sigE"]
Q99 = "286.088064"
LABELS = ["operational", Q99]
# Checkpoint callback -> (dirpath folders, monitored key, mode), named as in the pools.
CKPTS = {
    "cvar25_ema_ckpt": (
        "summary/cvar25_ema/max",
        "val/summary/eff_cvar25_ema_operational",
        "max",
    ),
    "cvar10_q99_ckpt": (
        "summary/cvar10_ema_286_088064/max",
        "val/summary/eff_cvar10_ema_286_088064",
        "max",
    ),
    "drift_q99_ckpt": (
        "summary/trate286_088064kHz_drift_ema/min",
        "val/summary/trate286_088064kHz_drift_ema",
        "min",
    ),
    "cap_ckpt": (
        "summary/cap_ema_normal_vs_ref/max",
        "val/summary/cap_ema_normal_vs_ref",
        "max",
    ),
    "loss_ckpt": ("normal/loss/min", "val/normal/loss", "min"),
    "w1_ckpt": (
        "summary/w1dist_ema_normal_vs_ref/min",
        "val/summary/w1dist_ema_normal_vs_ref",
        "min",
    ),
}
EVALUATED = [
    "cvar25_ema_ckpt",
    "cvar10_q99_ckpt",
    "drift_q99_ckpt",
    "cap_ckpt",
    "loss_ckpt",
]
CAP = [0.1, 0.2, 0.3]
SEC = [1.0, 0.5, 2.0]
HISTORY = [[0.9, 0.4, 0.6, 0.4], [0.8, 0.7, 0.3, 0.5], [0.5, 0.6, 0.7, 0.8]]


def _load_script():
    spec = importlib.util.spec_from_file_location("harvest_pool_effs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harvest = _load_script()


def _eff(run, ckpt, dataset, split, label):
    rate_shift = 0.1 if label == "operational" else 0.0
    split_shift = 0.002 if split == "test" else 0.0
    return 0.05 * (dataset + 1) + 0.01 * run + 0.001 * ckpt + rate_shift + split_shift


def _eval_metrics(run, ckpt):
    strategy, metric, criterion = CKPTS[EVALUATED[ckpt]][0].split("/")
    metrics = {}
    for split in ("val", "test"):
        prefix = f"eval/{split}/{strategy}/{{}}/{metric}/{criterion}"
        for label in LABELS:
            for i, dataset in enumerate([*SIGNALS, "normal"]):
                eff = _eff(run, ckpt, i, split, label)
                metrics[f"{prefix.format('eff')}/eff_{label}_{dataset}"] = eff
            metrics[f"{prefix.format('eff')}/eff_{label}"] = 0.5
            metrics[f"{prefix.format('eff')}/eff_med_{label}"] = 0.5
        metrics[f"{prefix.format('ascore_operational')}/normal"] = SEC[run]
        metrics[f"{prefix.format('thres_drift')}/drift_operational"] = 0.3 + run
        drift_q99 = f"{prefix.format('thres_drift')}/drift_trate286_088064kHz"
        metrics[drift_q99] = 0.6 + run
        metrics[f"{prefix.format('cap')}/cap"] = CAP[run]
    return metrics


def _params(run):
    params = {
        "algorithm/target_rate": "0.25",
        "algorithm/base_rate": "28608.8064",
        "algorithm/optimizer/lr": str(10.0**-run),
        "seed": "123",
        "callbacks/last_epoch_ckpt/monitor": "None",
        "callbacks/last_epoch_ckpt/dirpath": "${paths.checkpoints_dir}/${run_name}/",
    }
    for name, (folders, monitor, mode) in CKPTS.items():
        params[f"callbacks/{name}/dirpath"] = f"${{run_name}}/{folders}"
        params[f"callbacks/{name}/monitor"] = monitor
        params[f"callbacks/{name}/mode"] = mode
    return params


def _log_run(client, experiment_id, run):
    run_id = client.create_run(experiment_id, run_name=f"run{run}").info.run_id
    now = int(time.time() * 1000)
    metrics = []
    for ckpt in range(len(EVALUATED)):
        metrics += [Metric(k, v, now, 0) for k, v in _eval_metrics(run, ckpt).items()]
    metrics += [Metric("eval/val/last/eff/eff_operational", 0.5, now, 0)]
    metrics += [Metric("eval/val/optimized/optimized_main", 500.0, now, 0)]
    for _, monitor, _ in CKPTS.values():
        metrics += [Metric(monitor, v, now + e, e) for e, v in enumerate(HISTORY[run])]
    params = [Param(k, v) for k, v in _params(run).items()]
    client.log_batch(run_id, metrics=metrics, params=params)
    client.set_terminated(run_id)


@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("pool")
    uri = f"file:{root / 'mlruns'}"
    client = MlflowClient(uri)
    experiment_id = client.create_experiment(EXPERIMENT)
    for run in range(3):
        _log_run(client, experiment_id, run)
    harvest.main([EXPERIMENT, "--tracking-uri", uri, "--outdir", str(root / "out")])
    return root / "out"


def _read(outdir, stem):
    return pd.read_csv(outdir / f"{stem}.csv", dtype={"tier": str, "run": str})


def _row(table, strategy, run):
    rows = table[(table["strategy"] == strategy) & (table["run"] == f"run{run}")]
    assert len(rows) == 1
    return rows.iloc[0]


def _per_signal_base(header):
    # perSignalCols of notebooks/lib/effs_data.wl.
    effs = [c for c in header if c.startswith("eff_")]
    effs = [c for c in effs if not c.startswith(("eff_med_", "eff_min_"))]
    for base in sorted(effs, key=len):
        if sum(c.startswith(f"{base}_") for c in effs) >= 3:
            return base


def test_one_table_pair_per_tier(outdir):
    names = {p.name for p in outdir.iterdir()}
    assert names == {
        "physics_toy_pool.csv",
        "physics_toy_pool_eval.csv",
        "physics_toy_q99_pool.csv",
        "physics_toy_q99_pool_eval.csv",
    }


def test_rows_per_tier_and_strategy(outdir):
    main, q99 = _read(outdir, "physics_toy_pool"), _read(outdir, "physics_toy_q99_pool")
    assert Counter(main["strategy"]) == {"cvar25": 3, "cap": 3, "loss": 3}
    assert Counter(q99["strategy"]) == {
        "cvar10": 3,
        "stability": 3,
        "cap": 3,
        "loss": 3,
    }
    assert set(main["tier"]) == {"250"} and set(q99["tier"]) == {"q99"}
    assert main["run_name"].is_unique and q99["run_name"].is_unique
    assert _row(main, "cap", 1)["run_name"] == "cap_run1"
    assert set(main["ckpt"]) == {"strategy"}


def test_q_prime_and_second_objective(outdir):
    main, q99 = _read(outdir, "physics_toy_pool"), _read(outdir, "physics_toy_q99_pool")
    val_effs = sorted(_eff(2, 0, i, "val", "operational") for i in range(5))
    cvar25 = _row(main, "cvar25", 2)
    assert cvar25["optimized_main"] == pytest.approx(statistics.fmean(val_effs[:2]))
    assert cvar25["q_source"] == "eval" and cvar25["maximize"]
    cvar10 = _row(q99, "cvar10", 2)
    assert cvar10["optimized_main"] == pytest.approx(_eff(2, 1, 0, "val", Q99))
    stability = _row(q99, "stability", 1)
    assert stability["optimized_main"] == pytest.approx(1.6)
    assert not stability["maximize"]
    for table in (main, q99):
        assert _row(table, "cap", 2)["optimized_main"] == pytest.approx(CAP[2])
        assert _row(table, "cap", 0)["optimized_sec"] == pytest.approx(SEC[0])
    loss = _row(main, "loss", 0)
    assert loss["q_source"] == "monitor"
    assert loss["optimized_main"] == pytest.approx(0.4) and loss["epoch"] == 1
    assert _row(main, "cap", 0)["epoch"] == 0


def test_efficiencies_at_the_row_tier(outdir):
    q99 = _read(outdir, "physics_toy_q99_pool")
    row = _row(q99, "cap", 1)
    test_effs = [_eff(1, 3, i, "test", Q99) for i in range(5)]
    assert row["test_mean_sig_eff"] == pytest.approx(statistics.fmean(test_effs))
    assert row["sigA"] == pytest.approx(_eff(1, 3, 0, "val", Q99))
    assert row["n_signals"] == 5 and row["test_n_signals"] == 5
    assert "normal" not in q99.columns


def test_is_pareto_within_strategy_and_tier(outdir):
    for stem in ("physics_toy_pool", "physics_toy_q99_pool"):
        cap = _read(outdir, stem).query("strategy == 'cap'").sort_values("run")
        assert cap["is_pareto"].tolist() == [False, True, True]


def test_varying_parameters_only(outdir):
    main = _read(outdir, "physics_toy_pool")
    assert "hp/algorithm/optimizer/lr" in main.columns
    assert "hp/seed" not in main.columns


def test_eval_tables_hold_their_tier(outdir):
    main = pd.read_csv(outdir / "physics_toy_pool_eval.csv", dtype={"tier": str})
    q99 = pd.read_csv(outdir / "physics_toy_q99_pool_eval.csv", dtype={"tier": str})
    assert _per_signal_base(main.columns) == "eff_operational"
    assert _per_signal_base(q99.columns) == f"eff_{Q99}"
    assert "drift_trate286_088064kHz" not in main.columns
    assert "drift_operational" not in q99.columns
    assert not main["context"].str.startswith("last").any()
    test_cap = main.query("run_name == 'cap_run1' and split == 'test'")
    assert set(test_cap["context"]) == {
        f"summary/{cb}/cap_ema_normal_vs_ref/max"
        for cb in ("eff", "ascore_operational", "thres_drift", "cap")
    }
    eff = test_cap.query("context == 'summary/eff/cap_ema_normal_vs_ref/max'")
    expected = _eff(1, 3, 1, "test", "operational")
    assert eff["eff_operational_sigB"].item() == pytest.approx(expected)
