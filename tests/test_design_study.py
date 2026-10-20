import csv
import hashlib
import itertools

import numpy as np
import pytest
import torch
import yaml
from sklearn.covariance import OAS

from scripts.cchamber import design_study as design
from scripts.cchamber import offline_select as offline
from scripts.cchamber import reference_row as reference
from src.callbacks.metrics.cap.metric import ApproximationCapacity
from src.data.CausalChamber_datamodule import READOUTS

ROOT = design.HERE.parents[1]
REGISTRATION = design.HERE / "registration.yaml"
PROVISIONAL = ROOT / "configs/cap/provisional.yaml"
KEY = "ascore/full"
FAMILIES = ["ae", "vae"]
SEEDS = [101, 202, 303]
CANDIDATES = 3
CLUSTERS = {"sig_a": "x", "sig_b": "x", "sig_c": "y", "sig_d": "y"}
SPACE = {
    "normalization_type": ["sigmoid", "rank"],
    "normalization_mode": ["joint", "split"],
    "energy": [{"energy_type": "baseline"}, {"energy_type": "adaptive", "scale": 0.5}],
    "beta_max": [10],
    "n_configurations": 8,
}
# Simplicity rank of each configuration of SPACE: energy, then normalisation, then mode.
RANKS = {
    "sigmoid-joint-baseline-B10": 0,
    "sigmoid-split-baseline-B10": 1,
    "rank-joint-baseline-B10": 2,
    "rank-split-baseline-B10": 3,
    "sigmoid-joint-adaptive0.5-B10": 4,
    "sigmoid-split-adaptive0.5-B10": 5,
    "rank-joint-adaptive0.5-B10": 6,
    "rank-split-adaptive0.5-B10": 7,
}
# Per family, seed and intervention; the best candidate adds nothing to it.
OFFSETS = np.random.default_rng(1).normal(0.0, 0.05, (2, 3, 4))
REFERENCE_OFFSETS = np.array([0.01, -0.02, 0.03, 0.0])


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def write_registration(tmp_path):
    reg = yaml.safe_load(REGISTRATION.read_text())
    reg["families"] = {family: reg["families"][family] for family in FAMILIES}
    reg["design_space"].update(SPACE)
    reg["system"]["interventions"] = len(CLUSTERS)
    reg["criterion"]["standard_error"]["units"] = {"seeds": 3, "target_clusters": 2}
    path = tmp_path / "registration.yaml"
    path.write_text(yaml.safe_dump(reg, sort_keys=False))
    return path, reg


def write_clusters(tmp_path, clusters=CLUSTERS):
    rows = [
        {"intervention": name, "target_cluster": c, "family": c, "strength": "mid"}
        for name, c in clusters.items()
    ]
    return write_csv(tmp_path / "clusters.csv", rows)


def metric_rows(run, key, epoch, efficiency):
    rows = []
    for split, shift in (("val", 0.0), ("test", 0.01)):
        head = {"run": run, "score_key": key, "epoch": epoch, "split": split}
        rows.append({**head, "dataset": "normal", "efficiency": 0.01, "auprc": ""})
        for name, value in zip(CLUSTERS, efficiency + shift):
            auprc = 0.5 * value + 0.4
            rows.append({**head, "dataset": name, "efficiency": value, "auprc": auprc})
    return rows


def unit_row(config, strategy, run, family, candidate, seed, epoch, q_main, key):
    return {
        "config": config,
        "run": run,
        "family": family,
        "candidate": candidate,
        "seed": seed,
        "score_key": key,
        "strategy": strategy,
        "selected_epoch": epoch,
        "q_main": q_main,
        "best": str(candidate == 0),
    }


def write_outputs(path, levels, q_main=None, key=KEY):
    path.mkdir(parents=True)
    selection, metrics = [], []
    for epoch, ((config, strategy), level) in enumerate(levels.items(), start=1):
        for family, seed, k in itertools.product(FAMILIES, SEEDS, range(CANDIDATES)):
            q = -k if q_main is None else q_main(config, family, seed, k)
            run = f"pool/{family}_{k:03d}_s{seed}/scores"
            unit = unit_row(config, strategy, run, family, k, seed, epoch, q, key)
            offsets = OFFSETS[FAMILIES.index(family), SEEDS.index(seed)]
            metrics += metric_rows(run, key, epoch, level + offsets + 0.1 * k)
            selection.append(unit)
    selection_path = write_csv(path / "selection.csv", selection)
    return selection_path, write_csv(path / "metrics.csv", metrics)


def write_reference(path, level):
    path.mkdir(parents=True)
    run = "design/reference_000_s0/scores"
    unit = unit_row("any", "cap", run, "reference", 0, 0, 0, 0.0, KEY)
    metrics = metric_rows(run, KEY, 0, level + REFERENCE_OFFSETS)
    selection_path = write_csv(path / "selection.csv", [unit])
    return selection_path, write_csv(path / "metrics.csv", metrics)


def input_args(tmp_path, levels, q_main=None):
    files = [
        write_outputs(tmp_path / "pool", levels, q_main),
        write_reference(tmp_path / "reference", 0.33),
    ]
    selection = ["--selection", *(str(s) for s, _ in files)]
    return selection + ["--metrics", *(str(m) for _, m in files)]


def run_table(tmp_path, q_main=None):
    registration, reg = write_registration(tmp_path)
    names = [entry["config"] for entry in design._design(reg["design_space"])]
    levels = {(name, "cap"): 0.30 + 0.01 * i for i, name in enumerate(names)}
    levels |= {("none", "drift"): 0.25, ("none", "w1"): 0.20}
    out = tmp_path / "design"
    design.main(
        ["table", "--registration", str(registration), "--out", str(out)]
        + ["--frozen", str(out / "frozen.yaml")]
        + input_args(tmp_path, levels, q_main)
        + ["--clusters", str(write_clusters(tmp_path))]
    )
    return out, registration, levels


def table_rows(out):
    return {row["config"]: row for row in read_csv(out / "design_table.csv")}


def test_criterion_aggregates_the_best_runs(tmp_path):
    out, _, levels = run_table(tmp_path)
    rows = table_rows(out)
    assert len(rows) == 8 + 3
    for (config, strategy), level in levels.items():
        row = rows[config if strategy == "cap" else strategy]
        criterion = level + OFFSETS.mean()
        assert float(row["criterion"]) == pytest.approx(criterion)
        assert float(row["auprc"]) == pytest.approx(0.5 * criterion + 0.4)
        assert float(row["efficiency_test"]) == pytest.approx(criterion + 0.01)
        for f, family in enumerate(FAMILIES):
            per_family = level + OFFSETS[f].mean()
            assert float(row[f"criterion_{family}"]) == pytest.approx(per_family)
            assert float(row[f"auprc_{family}"]) == pytest.approx(
                0.5 * per_family + 0.4
            )
    ref = rows["reference"]
    assert float(ref["criterion"]) == pytest.approx(0.33 + REFERENCE_OFFSETS.mean())
    for family in FAMILIES:
        assert float(ref[f"criterion_{family}"]) == pytest.approx(
            float(ref["criterion"])
        )
    assert {config: int(rows[config]["simplicity_rank"]) for config in RANKS} == RANKS
    # Every row shares the offsets, and the bootstrap ignores a shift.
    errors = [float(rows[config]["se"]) for config in [*RANKS, "drift", "w1"]]
    assert np.allclose(errors, errors[0]) and errors[0] > 0
    assert 0 < float(ref["se"]) and float(ref["se_ae"]) == pytest.approx(
        float(ref["se"])
    )


def test_constant_cap_is_ineligible_and_skipped(tmp_path):
    top = "rank-split-adaptive0.5-B10"

    def q_main(config, family, seed, k):
        return 0.7 if (config, family, seed) == (top, "vae", 202) else -k

    out, registration, _ = run_table(tmp_path, q_main)
    rows = table_rows(out)
    assert rows[top]["eligible"] == "False" and float(rows[top]["min_spread"]) == 0.0
    others = [row for config, row in rows.items() if config in RANKS and config != top]
    assert all(
        r["eligible"] == "True" and float(r["min_spread"]) == 1.0 for r in others
    )

    best = max(others, key=lambda row: float(row["criterion"]))
    floor = float(best["criterion"]) - float(best["se"])
    within = [row for row in others if float(row["criterion"]) >= floor]
    frozen = min(within, key=lambda row: int(row["simplicity_rank"]))
    assert [c for c, row in rows.items() if row["frozen"] == "True"] == [
        frozen["config"]
    ]
    names = {row["config"] for row in within}
    for config in RANKS:
        if config != top:
            assert rows[config]["within_one_se"] == str(config in names)

    header = (out / "frozen.yaml").read_text()
    assert f"\n# configuration: {frozen['config']}, " in header
    assert f"\n# best eligible: {best['config']}, " in header
    assert f"of the best: {len(within)}\n" in header
    markdown = (out / "design_table.md").read_text()
    assert frozen["config"] in markdown and "| reference |" in markdown


def test_held_out_table_scores_the_given_configurations(tmp_path, monkeypatch):
    monkeypatch.setattr(design, "FROZEN", tmp_path / "default_frozen.yaml")
    registration, _ = write_registration(tmp_path)
    # Equal levels tie the two criteria, which have no simplicity rank to break it.
    levels = {("frozen", "cap"): 0.30, ("frozen_pairs_random", "cap"): 0.30}
    levels |= {("none", "drift"): 0.25, ("none", "w1"): 0.20}
    # Four target clusters where the registration holds two.
    clusters = write_clusters(tmp_path, {name: name for name in CLUSTERS})
    out = tmp_path / "held_out"
    argv = ["--registration", str(registration), "--out", str(out)]
    argv += input_args(tmp_path, levels) + ["--clusters", str(clusters)]
    design.main(["table", "--held-out", *argv])

    rows = table_rows(out)
    assert list(rows) == ["frozen", "frozen_pairs_random", "drift", "w1", "reference"]
    frozen = rows["frozen"]
    assert float(frozen["criterion"]) == pytest.approx(0.30 + OFFSETS.mean())
    assert frozen["simplicity_rank"] == frozen["within_one_se"] == ""
    assert float(frozen["se"]) > 0
    assert sorted(path.name for path in out.iterdir()) == [
        "design_table.csv",
        "design_table.md",
    ]
    assert not design.FROZEN.exists()

    with pytest.raises(SystemExit, match="4 interventions in 4 clusters"):
        design.main(["table", *argv])
    assert not design.FROZEN.exists()
    with pytest.raises(SystemExit) as error:
        design.main(["table", "--held-out", *argv, "--frozen", str(design.FROZEN)])
    assert error.value.code == 2


def test_table_adds_every_other_strategy_but_last(tmp_path):
    registration, _ = write_registration(tmp_path)
    levels = {("frozen", "cap"): 0.30, ("none", "drift"): 0.25, ("none", "w1"): 0.20}
    levels |= {("none", "npd"): 0.22, ("none", "mc"): 0.21, ("none", "last"): 0.40}
    out = tmp_path / "held_out"
    argv = ["--registration", str(registration), "--out", str(out)]
    argv += input_args(tmp_path, levels) + ["--clusters", str(write_clusters(tmp_path))]
    design.main(["table", "--held-out", *argv])

    rows = table_rows(out)
    assert list(rows) == ["frozen", "drift", "w1", "mc", "npd", "reference"]
    for strategy in ("mc", "npd"):
        criterion = levels[("none", strategy)] + OFFSETS.mean()
        assert float(rows[strategy]["criterion"]) == pytest.approx(criterion)


def test_one_standard_error_rule():
    def row(config, criterion, se, eligible, rank):
        return dict(
            config=config,
            criterion=criterion,
            se=se,
            eligible=eligible,
            simplicity_rank=rank,
            frozen=False,
        )

    rows = [
        row("ineligible_best", 0.60, 0.01, False, 5),
        row("best", 0.50, 0.02, True, 4),
        row("within", 0.49, 0.10, True, 2),
        row("outside", 0.47, 0.00, True, 0),
        row("ineligible_within", 0.485, 0.00, False, 1),
    ]
    best, frozen, n_within = design._freeze(rows)
    assert (best["config"], frozen["config"], n_within) == ("best", "within", 2)
    assert [r["within_one_se"] for r in rows] == [True, True, True, False, True]
    assert [r["frozen"] for r in rows] == [False, False, True, False, False]


def test_bootstrap_is_seeded_and_weights_clusters():
    se = {"seed": 271828, "replicates": 2000}
    cluster_of = np.array([0, 0, 1, 1])
    seed_counts, weights = design._draws(se, 3, cluster_of)
    again = design._draws(se, 3, cluster_of)
    assert np.array_equal(seed_counts, again[0]) and np.array_equal(weights, again[1])
    assert (seed_counts.sum(axis=1) == 3).all() and weights.shape == (2000, 4)
    assert (weights[:, 0] == weights[:, 1]).all() and (weights.sum(axis=1) == 4).all()

    eff = np.random.default_rng(3).uniform(0.2, 0.8, (1, 2, 3, 4))
    counts, w = np.array([[2, 1, 0]]), np.array([[2, 2, 0, 0]])
    per_family = [
        sum(n * (w[0] @ eff[0, f, s]) / w.sum() for s, n in enumerate(counts[0])) / 3
        for f in range(2)
    ]
    replicated = design._replicates(eff, counts, w)
    assert replicated[0, 0] == pytest.approx(per_family)

    def error(values):
        return design._replicates(values, seed_counts, weights).mean(axis=2).std(ddof=1)

    mean = eff.mean()
    assert error(mean + 0.1 * (eff - mean)) == pytest.approx(0.1 * error(eff))
    assert error(np.full_like(eff, 0.4)) == pytest.approx(0.0, abs=1e-12)


def test_frozen_file_is_the_flat_cap_group(tmp_path):
    out, registration, _ = run_table(tmp_path)
    data = (out / "frozen.yaml").read_bytes()
    text, digest = data.decode(), hashlib.sha256(data).hexdigest()
    # Hydra stops reading directives at the first other line, comments included.
    assert text.splitlines()[0] == "# @package cap"
    assert (out / "frozen.sha256").read_text() == f"{digest}  frozen.yaml\n"
    reg_digest = hashlib.sha256(registration.read_bytes()).hexdigest()
    assert f"# registration: registration.yaml, sha256 {reg_digest}" in text

    config = yaml.safe_load(text)
    assert list(config) == list(yaml.safe_load(PROVISIONAL.read_text()))
    frozen = next(
        r for r in read_csv(out / "design_table.csv") if r["frozen"] == "True"
    )
    entries = design._design(yaml.safe_load(registration.read_text())["design_space"])
    cap = next(e["cap"] for e in entries if e["config"] == frozen["config"])
    assert config == cap["cap_metric_config"]
    ApproximationCapacity(**config)


def test_design_space_files_are_distinct_and_build(tmp_path):
    design.main(["configs", "--out", str(tmp_path)])
    paths = sorted(tmp_path.glob("*.yaml"))
    texts = {path.read_text() for path in paths}
    assert len(paths) == len(texts) == 336
    assert (tmp_path / "sigmoid-joint-adaptive0.5-B10.yaml") in paths

    gen = torch.Generator().manual_seed(0)
    scores_1, scores_2 = torch.rand(16, generator=gen), torch.rand(16, generator=gen)
    for path in paths:
        config = yaml.safe_load(path.read_text())["cap_metric_config"]
        assert path.stem.endswith(f"-B{config['beta_max']}")
        metric = ApproximationCapacity(**config)
        metric.update(scores_1.clone(), scores_2.clone())
        assert np.isfinite(metric.compute())


def test_native_table_sets_scores_side_by_side(tmp_path):
    registration, _ = write_registration(tmp_path)
    keys = {KEY: 0.30, "ascore/mse": 0.20, "ascore/kl": 0.25}
    files = [
        write_outputs(tmp_path / f"native{i}", {("frozen", "cap"): level}, key=key)
        for i, (key, level) in enumerate(keys.items())
    ]
    files.append(write_reference(tmp_path / "reference", 0.33))
    design.main(
        ["native", "--registration", str(registration), "--out", str(tmp_path / "out")]
        + ["--selection", *(str(s) for s, _ in files)]
        + ["--metrics", *(str(m) for _, m in files)]
        + ["--clusters", str(write_clusters(tmp_path))]
    )
    rows = {
        row["family"]: row for row in read_csv(tmp_path / "out" / "native_table.csv")
    }
    for f, (family, native) in enumerate((("ae", "mse"), ("vae", "kl"))):
        row = rows[family]
        assert (row["declared_score"], row["native_score"]) == ("residual_oas", native)
        offset = OFFSETS[f].mean()
        assert float(row["criterion_declared"]) == pytest.approx(0.30 + offset)
        assert float(row["criterion_native"]) == pytest.approx(
            keys[f"ascore/{native}"] + offset
        )
    criterion = float(rows["reference"]["criterion_declared"])
    assert criterion == pytest.approx(0.33 + REFERENCE_OFFSETS.mean())


def write_light_tunnel(root, signals):
    rng = np.random.default_rng(0)
    dataset = root / "lt_interventions_standard_v1"
    dataset.mkdir(parents=True)
    mixing = rng.normal(size=(len(READOUTS), len(READOUTS)))
    tables = {"uniform_reference": rng.normal(size=(100, len(READOUTS))) @ mixing}
    for shift, name in enumerate(signals, start=1):
        tables[name] = rng.normal(size=(50, len(READOUTS))) @ mixing + shift
    for name, values in tables.items():
        lines = [",".join(READOUTS)] + [
            ",".join(map(repr, row)) for row in values.tolist()
        ]
        (dataset / f"{name}.csv").write_text("\n".join(lines) + "\n")


def test_reference_row_is_scored_like_a_run(tmp_path, capsys):
    signals = ["uniform_red_mid", "uniform_t_ir_3_strong"]
    write_light_tunnel(tmp_path / "causal_chamber", signals)
    data_config = yaml.safe_load((ROOT / "configs/data/cchamber.yaml").read_text())
    data_config.update(signal_experiments=signals, batch_size=8)
    config_path = tmp_path / "cchamber.yaml"
    config_path.write_text(yaml.safe_dump(data_config))
    scores = tmp_path / "design" / "reference_000_s0" / "scores"
    data_dir = str(tmp_path / "causal_chamber")
    argv = ["dump", "--data-config", str(config_path), "--data-dir", data_dir]
    reference.main([*argv, "--out", str(scores)])

    meta = torch.load(scores / "meta.pt", weights_only=True)
    record = torch.load(scores / "epoch_0000.pt", weights_only=True)
    assert meta["val"]["names"] == ["normal", "reference_normal", *signals]
    assert meta["val"]["batch_sizes"]["normal"] == [8, 2]
    assert meta["target_fpr"] == 0.01 and record["epoch"] == 0
    dm = reference._datamodule(config_path, data_dir)
    oas = OAS(store_precision=True, assume_centered=False)
    oas.fit(dm.data_train.tensors[0].double().numpy())
    for split, loaders in (
        ("val", dm.val_dataloader()),
        ("test", dm.test_dataloader()),
    ):
        for name, loader in loaders.items():
            x = (
                torch.cat([batch[0] for batch in loader]).double().numpy()
                - oas.location_
            )
            expected = np.einsum("bi,ij,bj->b", x, oas.precision_, x) / len(READOUTS)
            stored = record[split][name]
            assert torch.equal(stored["ascore/full"], stored["ascore/residual_oas"])
            np.testing.assert_allclose(
                stored["ascore/full"].numpy(), expected, rtol=1e-4
            )

    cap_config = tmp_path / "cap.yaml"
    entries = design._design(yaml.safe_load(REGISTRATION.read_text())["design_space"])
    cap_config.write_text(yaml.safe_dump(entries[0]["cap"], sort_keys=False))
    out = tmp_path / "offline"
    argv = [str(scores), "--cap-config", str(cap_config), "--strategies", "cap"]
    offline.main([*argv, "--torch-seed", "0", "--out", str(out)])
    (unit,) = read_csv(out / "selection.csv")
    assert (unit["family"], unit["selected_epoch"], unit["best"]) == (
        "reference",
        "0",
        "True",
    )
    assert len(read_csv(out / "metrics.csv")) == 2 * (1 + len(signals))

    pool = []
    for candidate, factor in ((0, 1.0), (1, 1.0 + 1e-3)):
        run = tmp_path / "pool" / f"ae_{candidate:03d}_s101" / "scores"
        run.mkdir(parents=True)
        normal = record["val"]["normal"]
        epoch = {"epoch": 3, "val": {"normal": {KEY: normal[KEY] * factor}}}
        torch.save(epoch, run / "epoch_0003.pt")
        pool.append(
            unit_row("frozen", "cap", str(run), "ae", candidate, 101, 3, 0.0, KEY)
        )
    selection = write_csv(tmp_path / "pool_selection.csv", pool)
    check = tmp_path / "check.csv"
    argv = ["check", "--reference", str(scores), "--selection", str(selection)]
    reference.main([*argv, "--config", "frozen", "--out", str(check)])
    assert [row["equal"] for row in read_csv(check)] == ["True", "False"]
    assert "ae: 1 of 2 runs equal the reference" in capsys.readouterr().out
