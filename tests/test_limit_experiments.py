import copy
import itertools
import math

import numpy as np
import pytest
import torch
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score

from scripts.cchamber import limit_experiments as limits
from scripts.cchamber import make_pairs
from scripts.cchamber import offline_select as offline
from scripts.cchamber import ranking_validity as ranking
from src.callbacks.cap import CAPCallback
from src.callbacks.metrics.cap.metric import ApproximationCapacity
from src.callbacks.metrics.rate import AnomalyRate, compute_tie_weight
from tests.test_offline_select import (
    CAP_CONFIG,
    CAP_METRIC,
    KEY,
    SEED,
    read_csv,
    write_run,
)

FAMILIES, CANDIDATES, SEEDS = ("ae", "vae"), range(5), (101, 202, 303)


def save_pairs(path, normal, reference, seed=None):
    torch.save({"normal": normal, "reference_normal": reference, "seed": seed}, path)
    return path


def callback_cap(val, pairs=None):
    callback = CAPCallback(output_name=KEY, **offline.PAIR, **copy.deepcopy(CAP_CONFIG))
    if pairs is not None:
        callback.pairing_fn = lambda s1, s2: pairs
    callback.capmetric = ApproximationCapacity(**callback.cap_metric_config)
    callback.dataset_1_scores = [val["normal"]]
    callback.dataset_2_scores = [val["reference_normal"]]
    return callback._compute_cap()


def test_nearest_neighbour_is_an_optimal_permutation():
    gen = torch.Generator().manual_seed(0)
    x, y = torch.randn(6, 11, generator=gen), torch.randn(6, 11, generator=gen)
    normal, reference = make_pairs.nearest_neighbour(x, y)
    assert sorted(normal.tolist()) == sorted(reference.tolist()) == list(range(6))

    cost = torch.cdist(x.double(), y.double())
    brute = min(
        cost[torch.arange(6), torch.tensor(p)].sum().item()
        for p in itertools.permutations(range(6))
    )
    assert cost[normal, reference].sum().item() == pytest.approx(brute, rel=1e-12)

    shuffle = torch.randperm(6, generator=gen)
    normal, reference = make_pairs.nearest_neighbour(x, x[shuffle] + 1e-3)
    assert torch.equal(shuffle[reference], normal)


def test_random_pairs_enter_offline_select(tmp_path):
    pairs = make_pairs.random_pairs(60, 4242)
    assert pairs[0].tolist() == list(range(60))
    assert sorted(pairs[1].tolist()) == list(range(60))
    assert torch.equal(pairs[1], make_pairs.random_pairs(60, 4242)[1])
    assert not torch.equal(pairs[1], make_pairs.random_pairs(60, 4243)[1])

    run = write_run(tmp_path / "pool" / "ae_000_s1" / "scores")
    config = tmp_path / "cap_a.yaml"
    config.write_text(yaml.safe_dump(CAP_CONFIG))
    path = save_pairs(tmp_path / "pairs_random.pt", *pairs, 4242)
    argv = [str(run), "--cap-config", str(config), "--strategies", "cap"]
    argv += ["--torch-seed", str(SEED)]
    offline.main([*argv, "--out", str(tmp_path / "cdf")])
    offline.main([*argv, "--pairs", str(path), "--out", str(tmp_path / "stored")])

    trace = read_csv(tmp_path / "stored" / "trace.csv")
    assert {row["config"] for row in trace} == {"cap_a_pairs_random"}
    cdf = read_csv(tmp_path / "cdf" / "trace.csv")
    epochs = offline._load_run(run, KEY)["epochs"]
    for row, cdf_row, scores in zip(trace, cdf, epochs):
        assert float(row["cap"]) == callback_cap(scores["val"], pairs)
        assert row["cap"] != cdf_row["cap"]

    short = save_pairs(tmp_path / "pairs_short.pt", pairs[0][:50], pairs[1][:50])
    with pytest.raises(ValueError, match="50 stored pairs for 60 and 60 scores"):
        offline.main([*argv, "--pairs", str(short), "--out", str(tmp_path / "x")])


def brute_energy_distance(x, y):
    def mean_distance(a, b):
        return np.mean([np.linalg.norm(u - v) for u in a for v in b])

    return 2 * mean_distance(x, y) - mean_distance(x, x) - mean_distance(y, y)


def test_energy_distance_matches_brute_force():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(20, 11)), rng.normal(0.5, 1.0, size=(15, 11))
    expected = brute_energy_distance(x, y)
    distance = limits._energy_distance(torch.tensor(x), torch.tensor(y))
    assert distance == pytest.approx(expected, rel=1e-10)
    assert expected > 0

    x = torch.tensor(x, dtype=torch.float32)
    assert limits._energy_distance(x, x.clone()) == 0.0


def test_ranking_statistics_match_scipy_and_hand_computation():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=17), rng.normal(size=17)
    y[3] = y[5]
    assert ranking._spearman(x, y) == pytest.approx(stats.spearmanr(x, y)[0], abs=1e-12)
    assert ranking._permutation_p(x, y) == ranking._permutation_p(x, -y)

    # Seven points: scipy enumerates all 5040 permutations; ours draws 10,000.
    x, y = x[:7], 0.5 * x[:7] + y[:7]
    exact = stats.permutation_test(
        (y,),
        lambda v: abs(stats.spearmanr(x, v)[0]),
        permutation_type="pairings",
        alternative="greater",
    )
    assert 0.02 < exact.pvalue < 0.98
    assert ranking._permutation_p(x, y) == pytest.approx(exact.pvalue, abs=0.02)

    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    # Ascending: 5(0.005), 4(0.01), 3(0.03), 2(0.04), 1(0.5); then running maxima.
    assert ranking._holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.5])
    assert ranking._holm(np.array([0.6, 0.9])) == pytest.approx([1.0, 1.0])

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y = np.array([0.3, 0.0, 0.5, 0.1, 0.9, 0.6, 0.7])
    keep = [2, 4, 5, 6]
    assert ranking._drop_lowest(x, y) == pytest.approx(0.4)
    assert ranking._drop_lowest(x, y) == pytest.approx(
        stats.spearmanr(x[keep], y[keep])[0]
    )

    z = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0])
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    r_xy, r_xz, r_yz = (
        np.corrcoef(a, b)[0, 1] for a, b in ((rx, ry), (rx, rz), (ry, rz))
    )
    expected = (r_xy - r_xz * r_yz) / math.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    assert ranking._partial_spearman(x, y, z) == pytest.approx(expected)


TOY_MAP = {
    "up": {"affected_readouts": ["ir_1"], "direction": {"ir_1": "increase"}},
    "down": {
        "affected_readouts": ["ir_2", "vis_2"],
        "direction": {"ir_2": "decrease", "vis_2": "unknown"},
        "direction_inferred": {"vis_2": "increase"},
    },
    "osr": {
        "affected_readouts": ["angle_1"],
        "direction": {"angle_1": "variance_decrease"},
    },
    "saturated": {
        "affected_readouts": ["angle_2"],
        "direction": {"angle_2": "saturation"},
        "direction_inferred": {"angle_2": "increase"},
    },
}


def toy_scores(seed=4):
    gen = torch.Generator().manual_seed(seed)
    sizes = {"normal": 200, "up": 60, "down": 60, "osr": 60, "saturated": 60}
    inputs = {
        split: {n: torch.randn(size, 11, generator=gen) for n, size in sizes.items()}
        for split in ("val", "test")
    }
    for split in inputs.values():
        split["up"][:, 3] += 3.0
        split["down"][:, 5] -= 3.0
    return {
        s: {n: limits._signed_readouts(x) for n, x in v.items()}
        for s, v in inputs.items()
    }


def test_blind_direction_candidates_comparators_and_summary():
    scores = limits._signed_readouts(torch.arange(22.0).reshape(2, 11))
    assert list(scores)[:4] == ["+current", "-current", "+angle_1", "-angle_1"]
    assert len(scores) == 22
    assert torch.equal(scores["-v_reg"], -torch.tensor([10.0, 21.0]))

    rows = [
        limits._comparator(entry, readout)
        for entry in TOY_MAP.values()
        for readout in entry["affected_readouts"]
    ]
    assert [(r["comparator"], r["block"]) for r in rows] == [
        ("+ir_1", "documented"),
        ("-ir_2", "documented"),
        ("+vis_2", "inferred"),
        ("", "none"),
        ("", "none"),
    ]

    rows = limits._blind_rows(toy_scores(), "+ir_1", TOY_MAP)
    assert [r["selected_is_comparator"] for r in rows] == [True] + [False] * 4
    summary = {
        s["block"]: s for s in (limits._block_summary(b, rows) for b in limits.BLOCKS)
    }
    assert summary["documented"]["selected_is_comparator"] == 1
    assert summary["signed"]["interventions"] == 2 and summary["none"]["rows"] == 2
    assert summary["none"]["efficiency_val_gap"] is None
    gaps = [r["auprc_val_comparator"] - r["auprc_val_selected"] for r in rows[:2]]
    assert summary["documented"]["auprc_val_gap"] == pytest.approx(np.mean(gaps))
    assert (
        rows[1]["efficiency_val_comparator"] > 0.5 > rows[1]["efficiency_val_selected"]
    )


def test_outcomes_count_ties_as_anomaly_rate_does():
    gen = torch.Generator().manual_seed(4)
    # Integer scores: the 1 % threshold of normal, 4, is shared by about 40 scores.
    scores = {
        split: {
            "normal": {"c": torch.randint(0, 5, (200,), generator=gen).float()},
            "down": {"c": torch.randint(3, 7, (60,), generator=gen).float()},
        }
        for split in ("val", "test")
    }
    outcomes = limits._outcomes(scores, "c", "down")
    rate = AnomalyRate(0.01, None)
    rate.set_threshold(scores["val"]["normal"]["c"])
    threshold, weight = rate.threshold.item(), rate.tie_weight.item()
    assert threshold == 4.0 and 0.0 < weight < 1.0
    for split in ("val", "test"):
        signal = scores[split]["down"]["c"]
        rate = AnomalyRate(0.01, None)
        rate.set_threshold(scores["val"]["normal"]["c"])
        rate.update(signal)
        assert outcomes[f"efficiency_{split}"] == rate.compute("efficiency").item()
        x = signal.numpy()
        expected = np.mean(x > threshold) + weight * np.mean(x == threshold)
        assert outcomes[f"efficiency_{split}"] == pytest.approx(expected)
        normal = scores[split]["normal"]["c"].numpy()
        target = np.r_[np.zeros(len(normal)), np.ones(len(x))]
        expected = average_precision_score(target, np.r_[normal, x])
        assert outcomes[f"auprc_{split}"] == pytest.approx(expected, rel=1e-5)


def test_drift_and_model_responses(tmp_path):
    # Integer scores put ties at the calibration threshold.
    gen = torch.Generator().manual_seed(5)
    normal = torch.randint(0, 20, (100,), generator=gen).float()
    callback = offline._Drift(output_name=KEY, **offline.DRIFT_KWARGS)
    callback.on_fit_start(None, None)
    module, monitor = offline._Module(limits.META), offline.MONITORS["drift"][0]
    val = {"normal": normal}
    drift, _ = offline._validation_epoch(callback, module, val, KEY, SEED, monitor)
    calibration, evaluation = callback._split_scores(normal)
    threshold = callback._compute_threshold(calibration, exceedance_prob=0.01)
    assert 0.0 < compute_tie_weight(calibration, threshold, 0.01) < 1.0
    assert (evaluation == threshold).any()
    assert limits._drift(normal, evaluation) == drift
    assert limits.DEPARTURE == {"cap": -1.0, "drift": 1.0, "w1": 1.0}

    run = write_run(tmp_path / "ae_000_s1" / "scores")
    row = {"run": str(run), "selected_epoch": "3", "score_key": KEY}
    row.update(family="ae", candidate="0", seed="1")
    # The frozen configuration in the flat format of the cap config group.
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump(CAP_METRIC))
    cap_cfg = limits._frozen_cap(path)
    shifts = {"sig_a": 0.1, "sig_b": 0.2}
    responses = limits._model_responses(row, shifts, cap_cfg, SEED)
    assert len(responses) == 2 * 3

    val = offline._load_run(run, KEY)["epochs"][3]["val"]
    order = torch.randperm(60, generator=torch.Generator().manual_seed(4242))
    reference = {
        "normal": val["normal"],
        "reference_normal": val["reference_normal"][order],
    }
    cap = callback_cap(reference)
    for response in responses:
        if response["criterion"] == "cap":
            assert response["reference"] == cap
        sign = limits.DEPARTURE[response["criterion"]]
        diff = response["value"] - response["reference"]
        assert response["response"] == sign * diff


@pytest.fixture(scope="module")
def pool_outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("limits")
    runs = []
    for family, candidate, seed in itertools.product(FAMILIES, CANDIDATES, SEEDS):
        path = root / "pool" / f"{family}_{candidate:03d}_s{seed}" / "scores"
        runs.append(str(write_run(path, seed=len(runs))))
    config = root / "frozen_cap.yaml"
    config.write_text(yaml.safe_dump(CAP_CONFIG))
    argv = [*runs, "--cap-config", str(config), "--torch-seed", str(SEED)]
    offline.main([*argv, "--out", str(root / "cdf")])
    for name, seed in (("nearest_neighbour", 7), ("random", 4242)):
        pairs = save_pairs(
            root / f"pairs_{name}.pt", *make_pairs.random_pairs(60, seed)
        )
        out = ["--strategies", "cap", "--pairs", str(pairs), "--out", str(root / name)]
        offline.main([*argv, *out])
    return root


def test_ranking_validity_tables(pool_outputs):
    out = pool_outputs / "ranking"
    paths = [
        pool_outputs / d / "selection.csv"
        for d in ("cdf", "random", "nearest_neighbour")
    ]
    ranking.main(["--selection", *map(str, paths), "--out", str(out)])

    selection = [r for p in paths for r in read_csv(p)]
    for endpoint, column in ranking.ENDPOINTS.items():
        table = read_csv(out / f"ranking_validity_{endpoint}.csv")
        assert {(r["criterion"], r["family"]) for r in table} == set(
            itertools.product(ranking.CRITERIA, FAMILIES)
        )
        assert (out / f"ranking_validity_{endpoint}.pdf").stat().st_size > 0
        row = next(
            r for r in table if r["criterion"] == "wasserstein" and r["family"] == "vae"
        )
        units = [u for u in selection if u["strategy"] == "w1" and u["family"] == "vae"]
        means = [
            [
                np.mean([float(u[c]) for u in units if u["candidate"] == str(k)])
                for k in CANDIDATES
            ]
            for c in ("q_main", column)
        ]
        assert float(row["rho"]) == pytest.approx(stats.spearmanr(*means)[0])
        assert int(row["n"]) == len(CANDIDATES)
        assert all(float(r["p_holm"]) >= float(r["p"]) for r in table)


def test_pairing_table(pool_outputs, tmp_path):
    clusters = tmp_path / "clusters.csv"
    clusters.write_text("intervention,target_cluster\nsig_a,a\nsig_b,b\n")
    outputs = [str(pool_outputs / d) for d in ("cdf", "nearest_neighbour", "random")]
    limits.main(
        ["pairing", *outputs, "--clusters", str(clusters), "--out", str(tmp_path)]
    )

    table = read_csv(tmp_path / "pairing.csv")
    assert [(r["pairing"], r["family"]) for r in table] == [
        (p, f)
        for p in ("cdf", "nearest_neighbour", "random")
        for f in (*FAMILIES, "all")
    ]
    for row in table:
        if row["family"] == "all":
            continue
        selection = read_csv(pool_outputs / row["pairing"] / "selection.csv")
        best = [
            u
            for u in selection
            if u["strategy"] == "cap"
            and u["best"] == "True"
            and u["family"] == row["family"]
        ]
        assert len(best) == len(SEEDS)
        expected = np.mean([float(u["efficiency_val_mean"]) for u in best])
        assert float(row["efficiency"]) == pytest.approx(expected)
        assert float(row["se_efficiency"]) > 0
    for pairing in ("cdf", "random"):
        rows = [r for r in table if r["pairing"] == pairing]
        assert float(rows[-1]["auprc"]) == pytest.approx(
            np.mean([float(r["auprc"]) for r in rows[:-1]])
        )


def test_crossed_bootstrap_weights():
    clusters = np.array([0, 0, 1, 2, 2, 2])
    seed_w, item_w = limits._bootstrap_weights(3, clusters)
    assert seed_w.shape == (limits.REPLICATES, 3)
    assert item_w.shape == (limits.REPLICATES, 6)
    assert (seed_w.sum(axis=1) == 3).all()
    assert (item_w[:, 0] == item_w[:, 1]).all() and (item_w[:, 3] == item_w[:, 5]).all()
    assert (item_w[:, [0, 2, 3]].sum(axis=1) == 3).all()

    matrix = np.random.default_rng(6).random((3, 6, 2))
    replicated = limits._replicated(matrix, seed_w, item_w)
    for b in (0, 7, 1999):
        expanded = np.repeat(np.repeat(matrix, seed_w[b], axis=0), item_w[b], axis=1)
        assert replicated[b] == pytest.approx(expanded.mean(axis=(0, 1)))
