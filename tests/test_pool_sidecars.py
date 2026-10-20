import pytest
import torch
import yaml

from scripts.analysis import label_free_criteria as lfc
from scripts.cchamber import pool_sidecars as sidecars
from tests.test_offline_select import KEY, write_run
from tests.test_score_dump import dump_callback, fit, toy_sets


def toy_run(root, **kwargs):
    run_dir = root / "pool" / "ae_000_s101"
    fit(root, [dump_callback(run_dir / "scores", **kwargs)])
    (run_dir / ".hydra").mkdir()
    config = {"algorithm": {"_target_": "tests.test_score_dump.ToyAE"}}
    (run_dir / ".hydra" / "config.yaml").write_text(yaml.safe_dump(config))
    return run_dir / "scores"


def load(path):
    return torch.load(path, weights_only=True)


def test_rescore_inputs_follow_the_training_inputs():
    x_train = toy_sets()["train"][0]
    sets, description = sidecars.rescore_inputs(x_train, 70)
    again, _ = sidecars.rescore_inputs(x_train, 70)
    assert all(torch.equal(sets[name], again[name]) for name in sets)

    mean, std = x_train.mean(dim=0), x_train.std(dim=0, correction=0)
    assert sets["autouad"].shape == (70, 4)
    assert torch.equal(description["aux_mean"], mean)
    assert torch.equal(description["aux_std"], std)
    uniform = sets["uniform_100k"]
    lower, upper = x_train.min(dim=0).values, x_train.max(dim=0).values
    assert uniform.shape == (100_000, 4)
    assert bool(((uniform >= lower) & (uniform <= upper)).all())
    targets, outliers = lfc.sds_pseudo_sets(x_train.double().numpy())
    assert torch.equal(sets["sds_targets"], torch.from_numpy(targets).float())
    assert torch.equal(sets["sds_outliers"], torch.from_numpy(outliers).float())


def test_rescore_reproduces_the_stored_scores(tmp_path):
    scores = toy_run(tmp_path)
    meta = load(scores / "meta.pt")
    _, description = sidecars.rescore_inputs(toy_sets()["train"][0], 70)
    # The toy dump holds an autouad set of its own, which is not rescored.
    gaussian = meta["aux_inputs"]["gaussian"]
    sets = {"again": gaussian, "autouad": gaussian}
    row = sidecars.rescore_run(scores, sets, description, tmp_path / "out")
    assert row["problem"] == "" and row["epochs"] == 2

    out = tmp_path / "out" / "ae_000_s101"
    for epoch in (0, 1):
        name = f"epoch_{epoch:04d}.pt"
        sidecar, record = load(out / name), load(scores / name)
        assert (sidecar["epoch"], sidecar["values"]) == (epoch, {})
        assert list(sidecar["aux"]) == ["again"]
        stored = record["aux"]["gaussian"][KEY]
        torch.testing.assert_close(sidecar["aux"]["again"][KEY], stored)
    assert load(out / "meta.pt")["sizes"] == {"again": 50}


def test_rescore_skips_a_run_that_fails_its_check(tmp_path):
    sets, description = sidecars.rescore_inputs(toy_sets()["train"][0], 70)
    # The n largest of 1,000 stored gaussian scores scaled by 1 + r, large enough to
    # leave atol behind: a few points off by 1e-3 pass, more than 1 % of them or one
    # off by 5e-2 fail.
    cases = {"few": (5, 1e-3, True), "many": (20, 1e-3, False), "far": (1, 5e-2, False)}
    for case, (n, relative, passes) in cases.items():
        scores = toy_run(tmp_path / case, aux_size=1000)
        record = load(scores / "epoch_0000.pt")
        stored = record["aux"]["gaussian"][KEY]
        stored[stored.argsort(descending=True)[:n]] *= 1 + relative
        torch.save(record, scores / "epoch_0000.pt")
        row = sidecars.rescore_run(scores, sets, description, tmp_path / case / "out")
        assert row["fraction_beyond_rtol"] == pytest.approx(n / 1000)
        expected = relative / (1 + relative)
        assert row["max_relative_difference"] == pytest.approx(expected, rel=1e-3)
        problem = "rescoring does not reproduce the stored gaussian scores"
        assert row["problem"] == ("" if passes else problem)
        assert (tmp_path / case / "out" / "ae_000_s101").exists() == passes


def test_consensus_values_follow_the_functions(tmp_path):
    names = ["ae_000_s101", "ae_000_s202", "ae_001_s101", "ae_001_s202"]
    runs = [
        write_run(tmp_path / "pool" / name / "scores", seed=i)
        for i, name in enumerate(names)
    ]
    sidecars.main(["consensus", *map(str, runs), "--out", str(tmp_path / "out")])

    normal = {
        name: [load(p)["val"]["normal"][KEY] for p in sorted(run.glob("epoch_*.pt"))]
        for name, run in zip(names, runs)
    }
    seed_peer = {"ae_000_s101": "ae_000_s202", "ae_000_s202": "ae_000_s101"}
    seed_peer |= {"ae_001_s101": "ae_001_s202", "ae_001_s202": "ae_001_s101"}
    for name in names:
        others = [n for n in names if n != name]
        for epoch in range(14):
            values = load(tmp_path / "out" / name / f"epoch_{epoch:04d}.pt")["values"]
            own = normal[name][epoch]
            udr = lfc.udr(own, torch.stack([normal[seed_peer[name]][epoch]]))
            pool = torch.stack([normal[n][epoch] for n in others])
            assert values == {"udr": udr, "mc": lfc.model_centrality(own, pool)}
        meta = load(tmp_path / "out" / name / "meta.pt")
        assert (meta["udr_runs"], meta["mc_runs"]) == ([seed_peer[name]], others)
