import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from scipy.stats import rankdata, spearmanr

from src.callbacks.metrics import label_free, selection
from src.evaluation import lattice

HARVEST = Path(__file__).resolve().parents[1] / "scripts/analysis/harvest_pool_effs.py"
EFF = "eff/eff_operational_"
# Batch sizes of validation normal, and the target FPR of every run.
LAYOUT = [16, 16, 8]
TARGET_FPR = 0.1
NAMES = ["r0", "r1", "r2", "r3", "r4", "r5"]
COLUMNS = {
    "cap": "cap/cap",
    "drift": "thres_drift/drift_operational",
    "oracle": "oracle",
    "cvar25": "cvar25",
    "cvar10": "cvar10",
    "q_sec": "ascore_operational/normal",
}
MODES = {"cap": "max", "drift": "min", "q_sec": "min", "udr": "max", "mc": "max"}
SELECTION = {
    "criteria": {
        "cap": {
            "column": "cap/cap",
            "mode": "max",
            "key": "val/summary/cap_ema_normal_vs_reference_normal",
            "skip_nonfinite": True,
        },
        "drift": {
            "column": "thres_drift/drift_operational",
            "mode": "min",
            "key": "val/summary/operational_drift_ema",
        },
    },
    "q_sec": {
        "column": "ascore_operational/normal",
        "key": "val/summary/ascore_operational_ema",
        "raw_key": "val/normal/ascore_operational",
    },
    "labelled": {
        "oracle": {"statistic": "mean", "key": "val/summary/eff_mean_ema_operational"},
        "cvar25": {
            "statistic": "cvar",
            "alpha": 0.25,
            "key": "val/summary/eff_cvar25_ema_operational",
        },
        "cvar10": {
            "statistic": "cvar",
            "alpha": 0.10,
            "key": "val/summary/eff_cvar10_ema_operational",
        },
    },
    "efficiency": EFF,
    "clusters": None,
    "lattice_criteria": {"udr": "max", "mc": "max"},
    "limits": {"drift": "a known limit"},
}
CRITERIA = list(SELECTION["criteria"])
ONCE = [*SELECTION["labelled"], "q_sec"]


def _spec(names=NAMES, epochs=4, signals=("sig_a", "sig_b", "sig_c"), seed=0):
    rng = np.random.default_rng(seed)
    shape, sets = (len(names), epochs + 1), ["normal", *signals, "null_x"]
    val = {COLUMNS["cap"]: rng.normal(size=shape)}
    val["cap/cap_beta_star"] = rng.uniform(size=shape)
    val[COLUMNS["drift"]] = rng.uniform(size=shape)
    val[COLUMNS["q_sec"]] = rng.uniform(1, 2, size=shape)
    val.update({EFF + name: rng.uniform(size=shape) for name in sets})
    signal_effs = np.stack([val[EFF + name] for name in signals])
    val["oracle"] = signal_effs.mean(axis=0)
    val["cvar25"] = val["cvar10"] = signal_effs.min(axis=0)
    test = {EFF + name: rng.uniform(size=shape) for name in sets}
    test["test_mean_eff"] = np.mean([test[EFF + name] for name in signals], axis=0)
    test["realised_fpr"], test["null_rate"] = test[EFF + "normal"], test[EFF + "null_x"]
    size = len(names)
    ks = range(11)
    picks = {(s, k): rng.integers(epochs, size=size) for s in CRITERIA for k in ks}
    picks.update({(s, None): rng.integers(epochs, size=size) for s in ONCE})
    normal = rng.normal(size=(*shape, sum(LAYOUT))).astype(np.float32)
    return SimpleNamespace(
        names=list(names),
        epochs=epochs,
        signals=list(signals),
        val=val,
        test=test,
        picks=picks,
        normal=normal,
        # The scores of each run's partner, aligned with the runs.
        partner=rng.normal(size=normal.shape).astype(np.float32),
        hyperpoints=[{"algorithm.lr": f"{0.001 * (i + 1):g}"} for i in range(size)],
    )


def _overrides(hyperpoint: dict, seed: int) -> list[str]:
    entries = [f"{key}={value}" for key, value in hyperpoint.items()]
    return [*entries, "experiment=cchamber/ae", f"seed={seed}"]


def _write_dump(run_dir: Path, normal: np.ndarray, overrides: list[str]) -> None:
    run_dir.mkdir(parents=True)
    meta = {"target_rate": TARGET_FPR, "base_rate": None, "target_fpr": TARGET_FPR}
    meta["val"] = {"batch_sizes": {"normal": LAYOUT}}
    torch.save(meta, run_dir / "meta.pt")
    (run_dir / "overrides.yaml").write_text(yaml.safe_dump(overrides))
    for epoch, scores in zip(range(-1, len(normal) - 1), normal):
        scores = torch.from_numpy(scores)
        record = {"epoch": epoch, "state_dict": {"weight": torch.zeros(2)}}
        record["val"] = {"normal": {"ascore/full": scores, "ascore/mse": 2 * scores}}
        record["test"] = {"normal": {"ascore/full": torch.zeros(3)}}
        record["aux"] = {"gaussian": {"ascore/full": torch.ones(5)}}
        record["train"] = {"train": {"loss/full": torch.ones(7)}}
        name = "initial.pt" if epoch < 0 else f"epoch_{epoch:04d}.pt"
        torch.save(record, run_dir / name)


def _write_run(run_dir: Path, spec, i: int) -> None:
    run_dir.mkdir(parents=True)
    epochs = np.arange(-1, spec.epochs)
    val = {"epoch": epochs, **{column: v[i] for column, v in spec.val.items()}}
    for name, column in COLUMNS.items():
        val[f"ema/{name}"] = [math.nan, *selection.ema(spec.val[column][i, 1:])]
    pd.DataFrame(val).to_csv(run_dir / "val.csv", index=False)
    test = {"epoch": epochs, **{column: v[i] for column, v in spec.test.items()}}
    pd.DataFrame(test).to_csv(run_dir / "test.csv", index=False)
    rows = [{"strategy": s, "k": k, "epoch": e[i]} for (s, k), e in spec.picks.items()]
    picks = pd.DataFrame(rows).astype({"k": "Int64"})
    picks.to_csv(run_dir / "picks.csv", index=False)
    info = {"run": spec.names[i], "lattice": "lat", "hyperpoint": spec.hyperpoints[i]}
    info.update(mlflow_run_id=f"id_{spec.names[i]}", epochs=spec.epochs)
    info.update(signals=spec.signals, nulls=["null_x"])
    (run_dir / "run.json").write_text(json.dumps(info))


def _run(root: Path, spec, partner: bool = True, config: dict = SELECTION) -> Path:
    out_dir, scores_dir, size = root / "eval" / "lat", root / "scores", len(spec.names)
    for i, name in enumerate(spec.names):
        _write_run(out_dir / "runs" / name, spec, i)
        overrides = _overrides(spec.hyperpoints[i], 123)
        _write_dump(scores_dir / "lat" / name, spec.normal[i], overrides)
        # Other names, in the reverse order: only the hyperpoint pairs the runs.
        partner_dir = scores_dir / "partner" / f"p{size - i}"
        overrides = _overrides(spec.hyperpoints[i], 456)
        _write_dump(partner_dir, spec.partner[i], overrides)
    name = "partner" if partner else None
    lattice.select_lattice(out_dir, config, scores_dir, "lat", name)
    return out_dir


def _read(out_dir: Path, name: str) -> pd.DataFrame:
    dtype = {"point": str, "oracle_point": str}
    return pd.read_csv(out_dir / name, dtype=dtype, float_precision="round_trip")


def _k(value):
    return None if pd.isna(value) else int(value)


def _inputs(spec, consensus: pd.DataFrame, strategy: str, k):
    # The per-run epochs, Q' (runs, epochs + 1) and mode a strategy picks from.
    q_sec = spec.val[COLUMNS["q_sec"]]
    if strategy in ("udr", "mc"):
        values = consensus[strategy].to_numpy().reshape(len(spec.names), -1)
        rows = zip(values[:, 1:], q_sec[:, 1:])
        epochs = [selection.epoch_pick(v, s, k, "max") for v, s in rows]
        return np.array(epochs), values, "max"
    mode = MODES.get(strategy, "max")
    return spec.picks[strategy, k], spec.val[COLUMNS[strategy]], mode


def _at(values: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    return values[np.arange(len(epochs)), np.asarray(epochs) + 1]


def test_load_scores_reads_every_stored_set_with_initial_as_minus_one(tmp_path):
    spec = _spec()
    _write_dump(tmp_path / "run", spec.normal[0], [])
    scores = lattice.load_scores(tmp_path / "run")
    assert list(scores) == [-1, 0, 1, 2, 3]
    for epoch, record in scores.items():
        assert set(record) == {"val", "test", "aux", "train"}
        normal = record["val"]["normal"]
        np.testing.assert_array_equal(normal["ascore/full"], spec.normal[0, epoch + 1])
        expected = 2 * spec.normal[0, epoch + 1]
        np.testing.assert_array_equal(normal["ascore/mse"], expected)
        assert record["train"]["train"]["loss/full"].shape == (7,)

    (tmp_path / "run" / "epoch_0001.pt").unlink()
    with pytest.raises(ValueError, match="epochs -1 to 2"):
        lattice.load_scores(tmp_path / "run")


def test_dump_hyperpoint_keeps_the_sweep_parameters(tmp_path):
    overrides = ["algorithm.lr=0.001", "algorithm.encoder.nodes=[32,16]", "seed=456"]
    (tmp_path / "overrides.yaml").write_text(yaml.safe_dump(overrides))
    keys = {"algorithm.lr", "algorithm.encoder.nodes", "algorithm.delta"}
    point = lattice.dump_hyperpoint(tmp_path, keys)
    assert point == {"algorithm.lr": "0.001", "algorithm.encoder.nodes": "[32,16]"}


def test_pair_runs_matches_equal_hyperpoints():
    own = {"a": {"lr": "0.1", "nodes": "[8,4]"}, "b": {"lr": "0.2", "nodes": "[8,4]"}}
    partner = {
        "x": {"nodes": "[8,4]", "lr": "0.2"},
        "y": {"lr": "0.1", "nodes": "[8,4]"},
        "z": {"lr": "0.3", "nodes": "[8,4]"},
    }
    assert lattice.pair_runs(own, partner) == {"a": "y", "b": "x"}
    with pytest.raises(ValueError, match="share"):
        lattice.pair_runs(own, {**partner, "w": {"lr": "0.1", "nodes": "[8,4]"}})
    with pytest.raises(ValueError, match="No partner run"):
        lattice.pair_runs({**own, "c": {"lr": "0.4", "nodes": "[8,4]"}}, partner)


def test_consensus_is_label_free_udr_and_mc(tmp_path):
    spec = _spec()
    # A constant score, whose rho is undefined, and an infinite one.
    spec.normal[1, 2] = 1.0
    spec.normal[2, 3, 0] = np.inf
    table = _read(_run(tmp_path, spec), "consensus.csv")
    assert list(table.columns) == ["run", "epoch", "udr", "mc", "ema/udr", "ema/mc"]
    assert list(table["run"]) == list(np.repeat(spec.names, spec.epochs + 1))
    for i, name in enumerate(spec.names):
        rows = table[table["run"] == name]
        assert list(rows["epoch"]) == list(range(-1, spec.epochs))
        others = np.delete(spec.normal, i, axis=0)
        for j in range(spec.epochs + 1):
            own = spec.normal[i, j]
            udr = label_free.udr(own, spec.partner[i, j][None])
            mc = label_free.model_centrality(own, others[:, j])
            assert rows["udr"].iloc[j] == pytest.approx(udr, abs=1e-12)
            assert rows["mc"].iloc[j] == pytest.approx(mc, abs=1e-12)
        for name in ("udr", "mc"):
            assert math.isnan(rows[f"ema/{name}"].iloc[0])
            expected = selection.ema(rows[name].to_numpy()[1:])
            np.testing.assert_allclose(rows[f"ema/{name}"].to_numpy()[1:], expected)


def test_without_a_partner_udr_is_left_out(tmp_path):
    out_dir = _run(tmp_path, _spec(), partner=False)
    assert _read(out_dir, "consensus.csv")["udr"].isna().all()
    strategies = set(_read(out_dir, "selection.csv")["strategy"])
    assert strategies == {*CRITERIA, "mc", *ONCE}


def test_every_pick_is_the_two_level_rule(tmp_path):
    spec = _spec()
    out_dir = _run(tmp_path, spec)
    consensus = _read(out_dir, "consensus.csv")
    table = _read(out_dir, "selection.csv")
    graded = [(s, k) for s in [*CRITERIA, "udr", "mc"] for k in range(11)]
    keys = list(zip(table["strategy"], map(_k, table["k"])))
    once = {(s, None) for s in ONCE}
    assert len(keys) == len(set(keys)) and set(keys) == {*graded, *once}
    q_sec = spec.val[COLUMNS["q_sec"]]
    for row in table.itertuples():
        k = _k(row.k)
        epochs, q_main, mode = _inputs(spec, consensus, row.strategy, k)
        k_point = selection.W_STEPS if k is None else k
        point = selection.pick(_at(q_main, epochs), _at(q_sec, epochs), k_point, mode)
        assert (row.point, row.epoch) == (spec.names[point], epochs[point])
        assert row.q_main == q_main[point, epochs[point] + 1]
        assert row.q_sec == q_sec[point, epochs[point] + 1]
        assert math.isnan(row.w) if k is None else row.w == pytest.approx(k / 10)


def _hand_spec():
    spec = _spec(names=["a", "b", "c"], epochs=2, signals=("s1", "s2"))
    nan = math.nan
    spec.val["cap/cap"] = np.array([[nan, 0.2, 0.5], [nan, 0.4, 0.9], [nan, 0.7, 0.3]])
    q_sec = [[9.0, 3.0, 2.0], [9.0, 2.0, 4.0], [9.0, 2.5, 1.5]]
    spec.val["ascore_operational/normal"] = np.array(q_sec)
    oracle = [[nan, 0.6, 0.65], [nan, 0.5, 0.8], [nan, 0.55, 0.7]]
    spec.val["oracle"] = np.array(oracle)
    for k in range(11):
        spec.picks["cap", k] = np.array({10: [1, 1, 0], 0: [0, 0, 1]}.get(k, [1, 0, 1]))
    spec.picks["oracle", None] = np.array([0, 1, 0])
    spec.picks["q_sec", None] = np.array([1, 0, 0])
    # (run, epoch, signal), epoch -1 first.
    effs = [
        [[0, 0], [0.95, 0.85], [0.6, 0.6]],
        [[0, 0], [0.4, 0.6], [0.9, 0.7]],
        [[0, 0], [0.3, 0.5], [0.7, 0.7]],
    ]
    effs = np.array(effs)
    spec.test[EFF + "s1"], spec.test[EFF + "s2"] = effs[..., 0], effs[..., 1]
    means = [[0, 0.9, 0.6], [0, 0.5, 0.8], [0, 0.4, 0.7]]
    spec.test["test_mean_eff"] = np.array(means)
    spec.test["realised_fpr"] = np.arange(9.0).reshape(3, 3) / 100
    spec.test["null_rate"] = np.arange(9.0).reshape(3, 3) / 10
    return spec


def test_selection_by_hand(tmp_path):
    table = _read(_run(tmp_path, _hand_spec()), "selection.csv").set_index("strategy")
    cap = table.loc["cap"].set_index("k")
    # k = 10: Q' at epochs (1, 1, 0) is (0.5, 0.9, 0.7), so b at epoch 1, the oracle's.
    expected = dict(point="b", epoch=1, q_main=0.9, q_sec=4.0, test_eff=0.8)
    expected.update(regret=0.0, regret_lo=0.0, regret_hi=0.0, regret_epoch=0.0)
    expected.update(regret_point=0.0, realised_fpr=0.05, null_rate=0.5)
    expected.update(floor_mean=0.6, floor_p05=0.42, floor_p95=0.78)
    # k = 0: Q'' at epochs (0, 0, 1) is (3.0, 2.0, 1.5), so c at epoch 1; the oracle
    # takes c at epoch 0 (test 0.4) and b at epoch 1 (test 0.8).
    expected_0 = dict(point="c", epoch=1, q_main=0.3, q_sec=1.5, test_eff=0.7)
    expected_0.update(regret=-0.1, regret_lo=-0.2, regret_hi=0.0, regret_epoch=0.3)
    expected_0.update(regret_point=-0.4, realised_fpr=0.08, null_rate=0.8)
    expected_0.update(floor_mean=0.7, floor_p05=0.52, floor_p95=0.88)
    # k = 5: ranks of Q' (1, 2, 3) and of Q'' (2.5, 2.5, 1), so a at epoch 1.
    expected_5 = dict(point="a", epoch=1, q_main=0.5, q_sec=2.0, test_eff=0.6)
    expected_5.update(regret=-0.2, regret_lo=-0.3, regret_hi=-0.1, regret_epoch=-0.3)
    expected_5.update(regret_point=0.1, realised_fpr=0.02, null_rate=0.2)
    expected_5.update(floor_mean=0.6, floor_p05=0.51, floor_p95=0.69)
    for k, values in ((10, expected), (0, expected_0), (5, expected_5)):
        for column, value in values.items():
            assert cap.loc[k, column] == pytest.approx(value), (k, column)
        assert cap.loc[k, "oracle_test_eff"] == pytest.approx(0.8)
        assert cap.loc[k, "w"] == pytest.approx(k / 10)

    oracle = table.loc["oracle"]
    assert (oracle["point"], oracle["epoch"], oracle["regret"]) == ("b", 1, 0.0)
    assert oracle["best_test_eff"] == pytest.approx(0.9)
    assert pd.isna(oracle["k"]) and pd.isna(table.loc["q_sec", "best_test_eff"])
    # Q'' alone: a at epoch 1 and b at epoch 0 tie at 2.0, and the first run wins.
    assert (table.loc["q_sec", "point"], table.loc["q_sec", "epoch"]) == ("a", 1)
    assert set(table.loc["drift", "limit"]) == {"a known limit"}
    assert table.loc["cap", "limit"].isna().all()


def test_one_target_cluster_collapses_the_interval(tmp_path):
    clusters = tmp_path / "clusters.csv"
    clusters.write_text("intervention,target_cluster\ns1,x\ns2,x\nother,y\n")
    config = {**SELECTION, "clusters": str(clusters)}
    table = _read(_run(tmp_path, _hand_spec(), config=config), "selection.csv")
    cap = table[table["strategy"] == "cap"].set_index("k")
    for k, regret in ((10, 0.0), (0, -0.1), (5, -0.2)):
        assert cap.loc[k, "regret_lo"] == pytest.approx(regret)
        assert cap.loc[k, "regret_hi"] == pytest.approx(regret)
    interventions = _read(tmp_path / "eval" / "lat", "interventions.csv")
    assert set(interventions["target_cluster"]) == {"x"}


def test_the_interval_resamples_target_clusters(tmp_path):
    clusters = tmp_path / "clusters.csv"
    rows = ["intervention,target_cluster", "sig_a,x", "sig_b,x", "sig_c,y"]
    clusters.write_text("\n".join(rows) + "\n")
    spec = _spec()
    out_dir = _run(tmp_path, spec, config={**SELECTION, "clusters": str(clusters)})
    table = _read(out_dir, "selection.csv")
    oracle = table[table["strategy"] == "oracle"].iloc[0]
    index = {name: i for i, name in enumerate(spec.names)}
    rng = np.random.default_rng(lattice.BOOTSTRAP_SEED)
    draws = rng.integers(2, size=(lattice.BOOTSTRAP_DRAWS, 2))
    for row in table.itertuples():
        effs = [spec.test[EFF + name] for name in spec.signals]
        pick = [e[index[row.point], row.epoch + 1] for e in effs]
        best = [e[index[oracle["point"]], oracle["epoch"] + 1] for e in effs]
        differences = np.subtract(pick, best)
        sums = np.array([differences[:2].sum(), differences[2]])
        means = sums[draws].sum(axis=1) / np.array([2, 1])[draws].sum(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        assert (row.regret_lo, row.regret_hi) == (pytest.approx(lo), pytest.approx(hi))
        assert row.regret == pytest.approx(differences.mean())


def _permuted_test(spec, seed: int = 1):
    order = np.random.default_rng(seed).permutation(spec.test["test_mean_eff"].size)
    test = {c: v.ravel()[order].reshape(v.shape) for c, v in spec.test.items()}
    return SimpleNamespace(**{**vars(spec), "test": test})


def test_permuting_the_test_values_leaves_every_pick_unchanged(tmp_path):
    spec = _spec()
    first = _run(tmp_path / "first", spec)
    second = _run(tmp_path / "second", _permuted_test(spec))
    columns = {
        "selection.csv": ["strategy", "k", "point", "epoch", "q_main", "q_sec"],
        "tradeoff.csv": ["strategy", "k_epoch", "k_point", "point", "epoch"],
        "oracle_position.csv": ["strategy", "k_epoch", "oracle_point", "rank_q_main"],
    }
    for name, kept in columns.items():
        tables = [_read(out_dir, name)[kept] for out_dir in (first, second)]
        pd.testing.assert_frame_equal(*tables)
    effs = [_read(out_dir, "selection.csv")["test_eff"] for out_dir in (first, second)]
    assert not np.allclose(*effs)


def _ranking_value(spec, consensus, strategy: str, k):
    epochs, q_main, mode = _inputs(spec, consensus, strategy, k)
    q_main, q_sec = _at(q_main, epochs), _at(spec.val[COLUMNS["q_sec"]], epochs)
    goodness = q_main if mode == "max" else -q_main
    if k is None or k == 10:
        return goodness, epochs
    if k == 0:
        return -q_sec, epochs
    ranks = rankdata(-np.where(np.isnan(goodness), -np.inf, goodness))
    return -(k * ranks + (10 - k) * rankdata(q_sec)), epochs


def _permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x) - rankdata(x).mean(), rankdata(y) - rankdata(y).mean()
    rng = np.random.default_rng(lattice.PERMUTATION_SEED)
    null = rng.permuted(np.tile(ry, (lattice.PERMUTATIONS, 1)), axis=1) @ rx
    return (1 + (np.abs(null) >= abs(rx @ ry)).sum()) / (1 + lattice.PERMUTATIONS)


def test_validity_is_spearman_of_the_ranking_value(tmp_path):
    spec = _spec()
    # Without Q', r0 leaves the rows of Q' alone but keeps a rank sum.
    spec.val["cap/cap"][0] = math.nan
    out_dir = _run(tmp_path, spec)
    consensus = _read(out_dir, "consensus.csv")
    table = _read(out_dir, "validity.csv")
    assert list(table.columns) == ["strategy", "k", "rho", "p", "p_holm", "n"]
    test = spec.test["test_mean_eff"]
    for row in table.itertuples():
        value, epochs = _ranking_value(spec, consensus, row.strategy, _k(row.k))
        kept = np.isfinite(value)
        x, y = value[kept], _at(test, epochs)[kept]
        assert row.n == kept.sum()
        assert row.rho == pytest.approx(spearmanr(x, y).statistic)
        assert row.p == pytest.approx(_permutation_p(x, y))
    cap = table[table["strategy"] == "cap"].set_index("k")
    assert (cap.loc[10, "n"], cap.loc[5, "n"]) == (len(NAMES) - 1, len(NAMES))

    p, holm = table["p"].to_numpy(), np.empty(len(table))
    running = 0.0
    for j, index in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - j) * p[index]))
        holm[index] = running
    np.testing.assert_allclose(table["p_holm"], holm)


def _grid_spec(spread_at_best: float):
    spec = _spec(names=["a", "b", "c", "d"], epochs=2, signals=("s1", "s2"))
    # At epoch 0 the ranks of Q' are (1, 4, 2, 3) and of Q'' (4, 1, 2, 3): the rank sums
    # 4 - 3w, 1 + 3w, 2 and 3 make c the pick for w in (1/3, 2/3) alone.
    spec.val["cap/cap"][:, 1] = [0.9, 0.1, 0.7, 0.4]
    spec.val["ascore_operational/normal"][:, 1] = [4.0, 1.0, 2.0, 3.0]
    for key in spec.picks:
        spec.picks[key] = np.zeros(4, dtype=int)
    # b is the best Q'': constant scores give a standard error of 0, spread ones a
    # standard error far above the range of Q''.
    spec.normal[1, 1] = spread_at_best * spec.normal[1, 1] + 1.0
    return spec


@pytest.mark.parametrize("spread, gate", [(0.0, "b"), (1000.0, "a")])
def test_tradeoff_grid_gate_and_joint_only(tmp_path, spread, gate):
    spec = _grid_spec(spread)
    table = _read(_run(tmp_path, spec, partner=False), "tradeoff.csv")
    assert len(table) == len(["cap", "drift", "mc"]) * 11 * 12
    cap = table[table["strategy"] == "cap"]
    oracle = _read(tmp_path / "eval" / "lat", "selection.csv")
    oracle_eff = oracle.loc[oracle["strategy"] == "oracle", "test_eff"].item()
    for k_epoch in range(11):
        rows = cap[cap["k_epoch"] == k_epoch].set_index("k_point")
        points = {str(k): "b" if k < 4 else "c" if k < 7 else "a" for k in range(11)}
        assert rows["point"].to_dict() == {**points, "gate": gate}
        joint = {str(k): k in (4, 5, 6) for k in range(11)}
        assert rows["joint_only"].to_dict() == {**joint, "gate": False}
        np.testing.assert_allclose(rows["regret"], rows["test_eff"] - oracle_eff)


def test_oracle_position_ranks_the_oracle_point(tmp_path):
    spec = _spec()
    out_dir = _run(tmp_path, spec)
    consensus = _read(out_dir, "consensus.csv")
    selected = _read(out_dir, "selection.csv")
    oracle = selected.loc[selected["strategy"] == "oracle", "point"].item()
    point = spec.names.index(oracle)
    table = _read(out_dir, "oracle_position.csv")
    assert len(table) == 4 * 11
    for row in table.itertuples():
        epochs, q_main, mode = _inputs(spec, consensus, row.strategy, row.k_epoch)
        goodness = _at(q_main, epochs) * (1 if mode == "max" else -1)
        q_sec = _at(spec.val[COLUMNS["q_sec"]], epochs)
        assert row.oracle_point == oracle and row.n == len(NAMES)
        assert row.rank_q_main == rankdata(-goodness)[point]
        assert row.rank_q_sec == rankdata(q_sec)[point]


def test_interventions_hold_the_per_set_test_efficiencies(tmp_path):
    spec = _spec()
    out_dir = _run(tmp_path, spec)
    selected = _read(out_dir, "selection.csv")
    table = _read(out_dir, "interventions.csv")
    assert len(table) == len(selected) * len(spec.signals)
    oracle = selected[selected["strategy"] == "oracle"].iloc[0]
    o_point, o_epoch = spec.names.index(oracle["point"]), oracle["epoch"] + 1
    for row in table.itertuples():
        pick = selected[(selected["strategy"] == row.strategy)]
        pick = pick[pick["k"].isna() if pd.isna(row.k) else pick["k"] == row.k].iloc[0]
        effs = spec.test[EFF + row.intervention]
        point = spec.names.index(pick["point"])
        assert row.test_eff == effs[point, pick["epoch"] + 1]
        assert row.oracle_test_eff == effs[o_point, o_epoch]
        assert row.target_cluster == row.intervention


def _harvest_module():
    module_spec = importlib.util.spec_from_file_location("harvest_pool_effs", HARVEST)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_harvest_schema(tmp_path):
    harvest = _harvest_module()
    assert lattice.OLD_COLUMNS == harvest.OLD_COLUMNS
    assert lattice.NEW_COLUMNS == harvest.NEW_COLUMNS
    assert lattice.EVAL_COLUMNS == harvest.EVAL_COLUMNS
    for rate, base_rate in ((0.01, None), (0.1, None), (0.25, 28608.8)):
        expected = harvest._tier_name(rate, True, base_rate)
        assert lattice._tier_name(rate, base_rate) == expected

    spec = _spec()
    out_dir = _run(tmp_path, spec)
    table = _read(out_dir, "lat.csv")
    fixed = harvest.OLD_COLUMNS + harvest.NEW_COLUMNS
    assert list(table.columns) == fixed + sorted(spec.signals) + ["hp/algorithm.lr"]
    assert len(table) == len(NAMES) * (2 * 4 + len(ONCE))
    assert list(table["run_name"]) == sorted(table["run_name"])
    assert set(table["tier"]) == {"q90"} and set(table["ckpt"]) == {"epoch"}
    labels = {"cap_w1", "cap_w0", "stability_w1", "stability_w0", "udr_w1", "udr_w0"}
    labels |= {"mc_w1", "mc_w0", *ONCE}
    assert set(table["strategy"]) == labels
    for row in table.itertuples():
        assert row.run_name == f"{row.strategy}_{row.run}"
    i = 2
    row = table.set_index("run_name").loc[f"stability_w0_{NAMES[i]}"]
    epoch = spec.picks["drift", 0][i]
    assert row["epoch"] == epoch and not row["maximize"]
    assert row["optimized_main"] == spec.val[COLUMNS["drift"]][i, epoch + 1]
    assert row["optimized_sec"] == spec.val[COLUMNS["q_sec"]][i, epoch + 1]
    ema = selection.ema(spec.val[COLUMNS["drift"]][i, 1:])[epoch]
    assert row["monitor_value"] == pytest.approx(ema)
    assert row["monitor"] == "val/summary/operational_drift_ema"
    assert row["test_mean_sig_eff"] == spec.test["test_mean_eff"][i, epoch + 1]
    assert row["sig_b"] == spec.val[EFF + "sig_b"][i, epoch + 1]
    assert row["mean_sig_eff"] == pytest.approx(spec.val["oracle"][i, epoch + 1])
    assert row["hp/algorithm.lr"] == pytest.approx(0.003)
    assert row["run_id"] == f"id_{NAMES[i]}" and row["q_source"] == "eval"

    evals = _read(out_dir, "lat_eval.csv")
    assert list(evals.columns[: len(harvest.EVAL_COLUMNS)]) == harvest.EVAL_COLUMNS
    contexts = {"val": {"cap", "thres_drift", "ascore_operational", "eff"}}
    contexts["test"] = {"eff"}
    assert len(evals) == len(table) * 5
    for split, names in contexts.items():
        assert set(evals.loc[evals["split"] == split, "context"]) == names
    cap = evals.set_index(["run_name", "split", "context"])
    cap = cap.loc[(f"stability_w0_{NAMES[i]}", "val", "cap")]
    assert cap["cap"] == spec.val["cap/cap"][i, epoch + 1]
    assert cap["cap_beta_star"] == spec.val["cap/cap_beta_star"][i, epoch + 1]


def test_runs_with_different_epoch_counts_raise(tmp_path):
    spec, out_dir = _spec(), tmp_path / "eval" / "lat"
    for i, name in enumerate(spec.names):
        _write_run(out_dir / "runs" / name, spec, i)
    path = out_dir / "runs" / "r1" / "run.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "epochs": 5}))
    with pytest.raises(ValueError, match="different numbers of epochs"):
        lattice.select_lattice(out_dir, SELECTION, tmp_path / "scores", "lat", None)
