import json
import re

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from src.callbacks.sds import SDS_SETS
from src.evaluation.components import rebuild
from src.evaluation.dump import _check_stored_sds
from src.utils.omegaconf import register_resolvers
from tests.test_eval_stage import (
    SELECTION_TABLES,
    TABLES,
    TOY_RUN_CONFIG,
    _evaluate_and_select,
    _outputs,
    _strip_weights,
)
from tests.test_score_dump import PASSES, assert_identical, dump_callback, fit, load

# As extra_sets.sds of configs/eval/cifar10.yaml and the image DDL experiments.
SDS = {"max_rows": 10000, "row_seed": 0}
RUNS = ["run_a", "run_b"]
PUSH_SCALE = "+evaluation.evaluator.extra_sets.sds.push_scale={}"


@pytest.fixture(scope="module", autouse=True)
def _resolvers() -> None:
    # A second registration raises.
    if not OmegaConf.has_resolver("short_hash"):
        register_resolvers()


@pytest.fixture(scope="module")
def sds_lattices(tmp_path_factory):
    root = tmp_path_factory.mktemp("sds_lattices")
    for name, sds in [("rescored", None), ("stored", SDS)]:
        for seed, run in enumerate(RUNS):
            path = root / name / "toy" / run
            fit(root, [dump_callback(path, sds=sds)], seed=seed)
            (path / "config.yaml").write_text(yaml.safe_dump(TOY_RUN_CONFIG))
    _strip_weights(root / "stored" / "toy", root / "stripped" / "toy")
    return root


def test_storing_sds_adds_its_part_and_changes_nothing_else(sds_lattices):
    for file in ["meta.pt", *PASSES]:
        off = load(sds_lattices / "rescored" / "toy" / "run_a" / file)
        on = load(sds_lattices / "stored" / "toy" / "run_a" / file)
        added = (
            {"sds", "sds_max_rows", "sds_row_seed"} if file == "meta.pt" else {"sds"}
        )
        assert set(on) - set(off) == added and set(off) <= set(on)
        assert_identical(off, {key: on[key] for key in off})

    meta = load(sds_lattices / "stored" / "toy" / "run_a" / "meta.pt")
    record = load(sds_lattices / "stored" / "toy" / "run_a" / "epoch_0001.pt")
    assert (meta["sds_max_rows"], meta["sds_row_seed"]) == (10000, 0)
    assert meta["sds"]["names"] == list(SDS_SETS) == list(record["sds"])
    for name in SDS_SETS:
        n = sum(meta["sds"]["batch_sizes"][name])
        assert n > 0 and record["sds"][name]["ascore/full"].shape == (n,)


def _replayed(sds_lattices, tmp_path, name: str, *overrides: str) -> dict:
    _evaluate_and_select(sds_lattices / name, tmp_path / name, *overrides)
    return _outputs(tmp_path / name)


def test_stored_sds_scores_replay_to_the_tables_of_rescored_ones(
    sds_lattices, tmp_path
):
    # The stripped copy holds no weights after initial.pt, so a pseudo-set rescored
    # from them would raise.
    outputs = {
        name: _replayed(sds_lattices, tmp_path, name)
        for name in ("rescored", "stored", "stripped")
    }
    runs = [f"runs/{run}/{table}" for run in RUNS for table in TABLES]
    assert set(outputs["rescored"]) == set(runs) | SELECTION_TABLES
    assert b"sds/sds" in outputs["rescored"]["runs/run_a/val.csv"]
    for path, content in outputs["rescored"].items():
        for name in ("stored", "stripped"):
            if not path.endswith("run.json"):
                assert outputs[name][path] == content, (name, path)
                continue
            info = {n: json.loads(outputs[n][path]) for n in ("rescored", name)}
            assert {**info["rescored"], "dump": None} == {**info[name], "dump": None}


def test_stored_sds_scores_of_other_pseudo_sets_raise(sds_lattices, tmp_path):
    message = (
        "The dump stores the SDS pseudo-sets of {'max_rows': 10000, 'row_seed': 0}"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        override = "evaluation.evaluator.extra_sets.sds.row_seed=1"
        _replayed(sds_lattices, tmp_path, "stored", override)


def test_stored_sds_scores_at_another_push_scale_raise(sds_lattices, tmp_path):
    for lattice in ("stored", "stripped"):
        with pytest.raises(ValueError, match="push_scale"):
            _replayed(sds_lattices, tmp_path, lattice, PUSH_SCALE.format(2))


def test_a_meta_without_a_push_scale_passes_the_check(sds_lattices):
    meta = load(sds_lattices / "stored" / "toy" / "run_a" / "meta.pt")
    assert not [key for key in meta if "scale" in key]
    _check_stored_sds(meta, SDS)
    _check_stored_sds(meta, {**SDS, "push_scale": 1})
    with pytest.raises(ValueError, match="push_scale"):
        _check_stored_sds(meta, {**SDS, "push_scale": 0.5})


def test_rescored_sds_replays_at_another_push_scale(sds_lattices, tmp_path):
    default = _replayed(sds_lattices, tmp_path / "default", "rescored")
    pushed = _replayed(
        sds_lattices, tmp_path / "pushed", "rescored", PUSH_SCALE.format(0.5)
    )
    assert default.keys() == pushed.keys()
    assert default["runs/run_a/val.csv"] != pushed["runs/run_a/val.csv"]


def test_the_sds_cache_key_carries_a_scale_other_than_the_default(monkeypatch):
    monkeypatch.setattr(rebuild, "SDS_CACHE", {})
    x = torch.randn(64, 3, generator=torch.Generator().manual_seed(0))
    default = rebuild.cached_sds(x, 100, 0)
    assert rebuild.cached_sds(x, 100, 0, None, 1) is default
    pushed = rebuild.cached_sds(x, 100, 0, None, 2.0)
    assert sorted(map(len, rebuild.SDS_CACHE)) == [5, 6]
    assert torch.equal(default["sds_targets"], pushed["sds_targets"])
    assert not torch.equal(default["sds_outliers"], pushed["sds_outliers"])
