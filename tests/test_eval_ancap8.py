import hashlib
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import Callback

from src.callbacks.components.points import load_points
from src.callbacks.components.utils import row_subset
from src.evaluation.callbacks.utils.mlflow import log_metrics_to_mlflow
from src.evaluation.components import ancap8 as ancap8_components
from src.evaluation.components.rebuild import check_record_cache, rebuild
from src.evaluation.dump import RunEvaluation, evaluate_dump
from src.evaluation.lattice import select_lattice
from src.evaluation.replay import _load_weights
from tests.test_eval_physics import MODULE, SN, PhysicsDataModule, _dump, physics_sets
from tests.test_eval_stage import SELECTION_TABLES, _compose, _read_csv
from tests.test_score_dump import (
    PASSES,
    MaskedToyAE,
    ToyAE,
    ToyDataModule,
    _points_x,
    as_points,
    dump_callback,
    fit,
    load,
)

LATTICE, EPOCHS = "toy", 2
TABLES = ["candidates", "picks", "beside", "kinds", "condition"]
# The keys the stand-in records per cell, in place of the criterion's keys(n_kinds).
STAND_IN_KEYS = [f"{s}_{k}" for s in ("n", "sum") for k in ("s1", "s2", "k1", "k2")]
# Stands in for the criterion's callback in the wiring tests.
STAND_IN = "tests.test_eval_ancap8.StandInANCAP8"
# The views' rule at the toy's size: fewer rows than its 120 and 80, so that the rule
# draws a subset.
SEED, CAPS = 20261106, {"normal": 100, SN: 60}
# The physics dump's auxiliary batch size (tests.test_eval_physics.BATCH), at which the
# points are scored, and the toy datamodule's, which the capped views keep.
POINT_BATCH, SET_BATCH = 16, 32
RUN_CONFIG = {
    "data": {"_target_": "tests.test_eval_physics.PhysicsDataModule"},
    "algorithm": {"_target_": "tests.test_score_dump.MaskedToyAE", **MODULE},
}
# FREEZE.md of Theory XXXI: the md5s of x and mask of lhc_kind1.npz to lhc_kind4.npz.
FROZEN = {
    "lhc_k1": ("d0bc2d0f1cc997978ee3261febe59771", "f7876aa159f40c63c7ef70b0b6aed5d6"),
    "lhc_k2": ("82c0bd01630aef22db48b3a9acd90a02", "f7876aa159f40c63c7ef70b0b6aed5d6"),
    "lhc_k3": ("50ac6de682b644f41a0aa53e23c23983", "bba0ed75b410fb8c02e8ab77ea35cf9f"),
    "lhc_k4": ("e26e31de29cad4de4ee8747644b007c0", "380a141646b2c88a2afdf76fedc02f30"),
}


HUB = "src.data.L1AD_HF_datamodule.L1ADHFDataModule"
# The record's cache_dir as its loader lays it out (adl1t_datamaker 97b465d), under
# paths.data_dir, <cache_dir>/data_2025E+G, with the stage names of its basis config.
RECORD_STAGES = {
    "extracted": "extracted/minimalTauFET/{category}/{name}",
    "processed": "processed/eminimalTauFET_pdefaultTauFET/{category}/{name}",
    "mlready": "mlready/eminimalTauFET_pdefaultTauFET_default/robust/aux/{name}",
}
GUN = [(stage, "background", SN) for stage in RECORD_STAGES]


class StandInANCAP8(Callback):
    # The pinned signature of src.evaluation.callbacks.ancap8.ANCAP8. It records the
    # size and the float64 sum of every sample it reads, per cell, which identify the
    # scores served to it.
    def __init__(
        self,
        output_name,
        dataset_1,
        views,
        points,
        kinds=False,
        sizes=None,
        name="ancap8",
    ):
        super().__init__()
        self.output_name, self.name, self.views = output_name, name, dict(views)
        self.samples = {"s1": dataset_1}
        self.samples |= {f"k{k}": p for k, p in enumerate(points, start=1)}

    def on_test_epoch_start(self, trainer, pl_module):
        self.scores = defaultdict(list)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        name = list(trainer.test_dataloaders)[dataloader_idx]
        self.scores[name].append(outputs[self.output_name].reshape(-1))

    def on_test_epoch_end(self, trainer, pl_module):
        values = {}
        for cell, view in self.views.items():
            for key, name in {**self.samples, "s2": view}.items():
                scores = torch.cat(self.scores[name])
                values[f"n_{key}_{cell}"] = len(scores)
                values[f"sum_{key}_{cell}"] = scores.double().sum().item()
        ckpt_name = Path(pl_module._ckpt_path).stem
        log_metrics_to_mlflow(trainer, values, ckpt_name=ckpt_name, cb_name=self.name)


class TrackedDataModule(PhysicsDataModule):
    def val_dataloader(self):
        self.handed = super().val_dataloader()
        return self.handed

    def test_dataloader(self):
        self.calls.append("test_dataloader")
        return super().test_dataloader()


def _md5(array: np.ndarray) -> str:
    return hashlib.md5(array.tobytes(), usedforsecurity=False).hexdigest()


def _masked_points(root: Path, name: str, n: int, seed: int) -> dict:
    # Whole objects present or padded, as in masked_sets, with x left nonzero under the
    # padding, which the loader sets to +0.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2, 2)).astype(np.float32) + 1.0
    mask = np.repeat(rng.random((n, 2, 1)) < 0.6, 2, axis=2)
    path = root / f"{name}.npz"
    np.savez(path, x=x, mask=mask)
    return {"path": str(path), "md5_x": _md5(x), "md5_mask": _md5(mask)}


def _record_cache(cache: Path, sets: list[tuple[str, str, str]]) -> None:
    # The zero bias at every stage and the ml-ready splits, then a directory per
    # (stage, category, set) given; every directory is empty.
    data = cache / "data_2025E+G"
    ml_ready = data / RECORD_STAGES["mlready"].split("/aux/")[0]
    for split in ("train", "valid", "test"):
        (ml_ready / split).mkdir(parents=True, exist_ok=True)
    zero_bias = [(s, "zerobias", "ZB_run396102") for s in ("extracted", "processed")]
    for stage, category, name in [*zero_bias, *sets]:
        path = data / RECORD_STAGES[stage].format(category=category, name=name)
        path.mkdir(parents=True, exist_ok=True)


def _config(points: dict, target: str = STAND_IN):
    cfg = _compose("physics_ancap8")
    with open_dict(cfg):
        evaluator = cfg.evaluation.evaluator
        callback = evaluator.dump_callbacks.ancap8
        callback["_target_"], callback.points = target, list(points)
        callback.sizes = None
        evaluator.extra_sets.points = points
        evaluator.row_caps.val = CAPS
        cfg.selection.ancap8.n_kinds = len(points)
    return cfg


def _train_runs(scores: Path) -> None:
    for seed, run in enumerate(["run_a", "run_b"]):
        path = scores / LATTICE / run
        module, datamodule = MaskedToyAE(**MODULE), PhysicsDataModule()
        fit(scores.parent, [_dump(path)], EPOCHS, seed, module, datamodule)
        (path / "config.yaml").write_text(yaml.safe_dump(RUN_CONFIG))


@pytest.fixture(scope="module")
def lattice(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("ancap8")
    _train_runs(root / "scores")
    # 37 and 21 points, which the batch of 16 does not divide.
    points = {
        name: _masked_points(root, name, n, seed)
        for seed, (name, n) in enumerate([("k1", 37), ("k2", 21)])
    }
    cfg = _config(points)
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    runs = evaluator.evaluate_lattice(root / "scores", root / "eval", LATTICE)
    return SimpleNamespace(
        scores=root / "scores", out=root / "eval", points=points, cfg=cfg, runs=runs
    )


def _scores(record: dict, batches: list, module=None) -> torch.Tensor:
    module = MaskedToyAE(**MODULE) if module is None else module
    _load_weights(module, record["state_dict"])
    module.eval()
    with torch.no_grad():
        return torch.cat([module.model_step(batch)["ascore/full"] for batch in batches])


def _batches(tensors: tuple, size: int) -> list[tuple]:
    starts = range(0, len(tensors[0]), size)
    return [tuple(t[i : i + size] for t in tensors) for i in starts]


def _view(name: str) -> tuple:
    tensors = physics_sets()["val"][name]
    rows = row_subset(len(tensors[0]), CAPS[name], SEED)
    return tuple(t[rows] for t in tensors)


def _points(spec: dict) -> tuple:
    with np.load(spec["path"]) as npz:
        x, mask = torch.from_numpy(npz["x"]), torch.from_numpy(npz["mask"])
    n = len(x)
    zeros = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.long)
    return (torch.where(mask, x, 0.0), mask, *zeros)


def test_points_are_served_with_their_masks_and_rescored(lattice):
    val = _read_csv(lattice.out / "runs" / "run_a" / "val.csv")
    dump = lattice.scores / LATTICE / "run_a"
    sums = []
    for row, name in zip(val, PASSES):
        record = load(dump / name)
        points = lattice.points.items()
        batches = {k: _batches(_points(p), POINT_BATCH) for k, p in points}
        batches["s1"] = _batches(_view("normal"), SET_BATCH)
        batches["s2"] = _batches(_view(SN), SET_BATCH)
        for key, sample in batches.items():
            scores = _scores(record, sample)
            assert float(row[f"ancap8/n_{key}_gun"]) == len(scores)
            assert float(row[f"ancap8/sum_{key}_gun"]) == scores.double().sum().item()
        sums.append(row["ancap8/sum_k1_gun"])
    assert len(set(sums)) == len(PASSES)


def test_signal_sets_are_dropped_before_any_pass(lattice, tmp_path):
    evaluator = hydra.utils.instantiate(lattice.cfg.evaluation.evaluator)
    datamodule, dump = TrackedDataModule(), lattice.scores / LATTICE / "run_a"
    settings = evaluator._dump_settings()
    run = RunEvaluation(dump, MaskedToyAE(**MODULE), datamodule, **settings)
    run.evaluate(tmp_path)
    assert list(run.loaders["val"]) == ["normal", SN, "k1", "k2"]
    assert run.loaders["test"] == {} and run.signals == {"val": [], "test": []}
    assert datamodule.calls == ["prepare_data", "setup:fit"]
    assert all(datamodule.handed[name].passes == 0 for name in ("sig_a", "sig_b"))
    for key, spec in lattice.points.items():
        served = run.loaders["val"][key].dataset.tensors
        assert all(torch.equal(a, b) for a, b in zip(served, _points(spec)))


def test_sets_build_no_test_loader_without_a_test_pass(lattice):
    evaluator = hydra.utils.instantiate(lattice.cfg.evaluation.evaluator)
    datamodule, dump = TrackedDataModule(), lattice.scores / LATTICE / "run_a"
    settings = {**evaluator._dump_settings(), "sets": "replaced"}
    run = RunEvaluation(dump, MaskedToyAE(**MODULE), datamodule, **settings)
    assert datamodule.calls == ["prepare_data", "setup:fit"]
    assert run.signals == {"val": [], "test": []}


def test_the_ancap8_columns_are_written_per_run(lattice):
    samples = ("s1", "s2", "k1", "k2")
    columns = {f"ancap8/{s}_{k}_gun" for s in ("n", "sum") for k in samples}
    sums = []
    for run in ("run_a", "run_b"):
        out = lattice.out / "runs" / run
        val = _read_csv(out / "val.csv")
        assert [row["epoch"] for row in val] == ["-1", "0", "1"]
        assert set(val[0]) == {"epoch"} | columns
        assert _read_csv(out / "test.csv") == [{"epoch": e} for e in ("-1", "0", "1")]
        assert (out / "picks.csv").read_text().strip() == ""
        sums.append([row["ancap8/sum_k1_gun"] for row in val])
    assert sums[0] != sums[1]
    assert all(run["signals"] == [] and run["nulls"] == [] for run in lattice.runs)


def test_stored_point_scores_are_not_read(tmp_path, points_files):
    dump = tmp_path / "scores" / LATTICE / "run_a"
    fit(tmp_path, [dump_callback(dump, points=as_points(points_files))], EPOCHS)
    # Stored scores that no model gives, so that a served value would show itself.
    for name in PASSES:
        record = load(dump / name)
        for scores in record["points"].values():
            scores["ascore/full"] += 100.0
        torch.save(record, dump / name)
    points = {
        name: {"path": str(path), "md5_x": _md5(_points_x(path)), "md5_mask": None}
        for name, path in points_files.items()
    }
    callback = {"_target_": STAND_IN, "output_name": "ascore/full"}
    callback.update(dataset_1="normal", views={"ref": "reference_normal"})
    callback.update(points=list(points), name="ancap8")
    settings = {"dump_callbacks": {"ancap8": callback}, "test_callbacks": []}
    settings.update(extra_sets={"points": points}, selection={})
    evaluate_dump(dump, ToyAE(), ToyDataModule(), tmp_path / "eval", **settings)
    val = _read_csv(tmp_path / "eval" / "runs" / "run_a" / "val.csv")
    for row, name in zip(val, PASSES):
        for k, path in enumerate(points_files.values(), start=1):
            x = torch.from_numpy(_points_x(path))
            batches = [(b, torch.zeros(len(b))) for b in x.split(POINT_BATCH)]
            scores = _scores(load(dump / name), batches, ToyAE())
            assert float(row[f"ancap8/sum_k{k}_ref"]) == scores.double().sum().item()


@pytest.mark.parametrize("change", ["md5_x", "md5_mask", "no_mask", "float64"])
def test_a_points_file_unlike_its_declaration_is_refused(tmp_path, change):
    spec = _masked_points(tmp_path, "kind", 5, 0)
    with np.load(spec["path"]) as npz:
        found = [_md5(npz["x"]), _md5(npz["mask"])]
    if change == "float64":
        x = np.zeros((5, 2, 2))
        np.savez(spec["path"], x=x)
        spec.update(md5_x=_md5(x), md5_mask=None)
        found = [_md5(x), "None"]
    elif change == "no_mask":
        spec["md5_mask"] = None
    else:
        spec[change] = "0" * 32
    with pytest.raises(ValueError) as error:
        load_points(spec["path"], spec["md5_x"], spec["md5_mask"])
    message = str(error.value)
    assert spec["path"] in message and all(md5 in message for md5 in found)
    assert spec["md5_x"] in message and str(spec["md5_mask"]) in message
    assert ("float64" in message) == (change == "float64")


def test_a_refused_file_stops_the_lattice_before_any_run(lattice, tmp_path):
    points = {k: dict(spec) for k, spec in lattice.points.items()}
    points["k2"]["md5_x"] = "0" * 32
    evaluator = hydra.utils.instantiate(_config(points).evaluation.evaluator)
    message = re.escape(f"{points['k2']['path']} holds x float32, mask bool")
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate_lattice(lattice.scores, tmp_path / "eval", LATTICE)
    assert not (tmp_path / "eval" / "runs").exists()


@pytest.fixture
def stand_in_tables(monkeypatch) -> list:
    calls = []

    def tables(rows: pd.DataFrame, cells: list[str], n_kinds: int) -> dict:
        calls.append(SimpleNamespace(rows=rows.copy(), cells=cells, n_kinds=n_kinds))
        other = pd.DataFrame({"value": [0.1 + 0.2, math.nan], "count": [3, 4]})
        return {name: rows if name == "candidates" else other for name in TABLES}

    monkeypatch.setattr(ancap8_components, "tables", tables)
    monkeypatch.setattr("src.evaluation.lattice.ancap8_keys", lambda n: STAND_IN_KEYS)
    return calls


def test_the_lattice_step_writes_the_five_tables(lattice, stand_in_tables, tmp_path):
    out = tmp_path / "eval"
    shutil.copytree(lattice.out, out)
    selection = OmegaConf.to_container(lattice.cfg.selection, resolve=True)
    select_lattice(out, selection, lattice.scores, LATTICE, None)
    [call] = stand_in_tables
    assert (call.cells, call.n_kinds) == (["gun"], 2)
    assert list(call.rows.columns[:2]) == ["run", "epoch"]
    assert call.rows["run"].tolist() == ["run_a"] * 3 + ["run_b"] * 3
    val = pd.read_csv(out / "runs" / "run_b" / "val.csv", float_precision="round_trip")
    assert call.rows["sum_k1_gun"].tolist()[3:] == val["ancap8/sum_k1_gun"].tolist()

    files = {path.name for path in out.iterdir() if path.is_file()}
    assert files == {f"ancap8_{name}.csv" for name in TABLES}
    assert not files & SELECTION_TABLES
    for name in TABLES:
        path = out / f"ancap8_{name}.csv"
        table = pd.read_csv(path, float_precision="round_trip")
        assert list(table.columns[:2]) == ["domain", "lattice"]
        assert set(table["domain"]) == {"physics"} and set(table["lattice"]) == {"toy"}
    beside = (out / "ancap8_beside.csv").read_text().splitlines()
    assert beside[1:] == ["physics,toy,0.30000000000000004,3", "physics,toy,,4"]
    path = out / "ancap8_candidates.csv"
    candidates = pd.read_csv(path, float_precision="round_trip")
    written = candidates.drop(columns=["domain", "lattice"])
    pd.testing.assert_frame_equal(written, call.rows, check_exact=True)


def test_the_record_is_composed_without_its_signal_sets(tmp_path):
    hub = {"_target_": "src.data.L1AD_HF_datamodule.L1ADHFDataModule", "overrides": []}
    config = {"data": hub, "algorithm": {"_target_": "tests.test_score_dump.ToyAE"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    overrides = list(_compose("physics_ancap8").data_overrides)
    assert overrides == ["data.signal={}"]
    datamodule = rebuild(tmp_path, str(tmp_path / "cache"), None, overrides)[1]
    record = datamodule._record_overrides(str(tmp_path / "record"))
    assert record[-1] == "data.signal={}"
    assert f"paths.base_data_dir={tmp_path / 'cache'}" in record
    toy = {**config, "data": {"_target_": "tests.test_score_dump.ToyDataModule"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(toy))
    with pytest.raises(ValueError, match="data_overrides apply"):
        rebuild(tmp_path, None, None, overrides)


def test_a_resumed_out_dir_without_the_ancap8_columns_raises(lattice, tmp_path):
    out = tmp_path / "eval"
    shutil.copytree(lattice.out, out)
    # As eval=physics leaves its out_dir: complete runs without v8's columns.
    for path in out.glob("runs/*/val.csv"):
        val = pd.read_csv(path)[["epoch"]].assign(**{"fit/normal": 0.5})
        val.to_csv(path, index=False)
    evaluator = hydra.utils.instantiate(_config(lattice.points).evaluation.evaluator)
    runs = evaluator.evaluate_lattice(lattice.scores, out, LATTICE)
    assert [run["run"] for run in runs] == ["run_a", "run_b"]
    selection = OmegaConf.to_container(lattice.cfg.selection, resolve=True)
    with pytest.raises(ValueError, match="run_a's val.csv lacks"):
        select_lattice(out, selection, lattice.scores, LATTICE, None)


@pytest.mark.parametrize(
    "stray",
    [
        None,
        ("extracted", "signal", "haa-4b-ma15"),
        ("processed", "signal", "smj-case-A"),
        ("mlready", "signal", "GluGluHTo2B_Par-MH-125"),
        ("processed", "background", "another_gun"),
    ],
)
def test_a_record_cache_with_a_set_outside_val_sets_is_refused(tmp_path, stray):
    cache, run = tmp_path / "cache", tmp_path / "run"
    _record_cache(cache, GUN + ([stray] if stray else []))
    run.mkdir()
    hub = {"_target_": HUB, "cache_dir": str(cache)}
    (run / "config.yaml").write_text(yaml.safe_dump({"data": hub}))
    check_record_cache(run, None, None)
    if stray is None:
        check_record_cache(run, None, ["normal", SN])
        return
    for data_dir in (None, str(cache)):
        with pytest.raises(ValueError, match=re.escape(stray[2])):
            check_record_cache(run, data_dir, ["normal", SN])


def test_a_record_cache_is_refused_before_any_datamodule(
    lattice, tmp_path, monkeypatch
):
    cache, run = tmp_path / "cache", tmp_path / "scores" / "lhc" / "run_x"
    _record_cache(cache, [*GUN, ("mlready", "signal", "haa-4b-ma15")])
    run.mkdir(parents=True)
    (run / "meta.pt").touch()
    (run / "config.yaml").write_text(yaml.safe_dump({"data": {"_target_": HUB}}))

    def built(*args, **kwargs):
        raise AssertionError("A datamodule was built.")

    monkeypatch.setattr("src.evaluation.dump.rebuild", built)
    evaluator = hydra.utils.instantiate(_config(lattice.points).evaluation.evaluator)
    with pytest.raises(ValueError, match="haa-4b-ma15"):
        evaluator.evaluate_lattice(
            tmp_path / "scores", tmp_path / "eval", "lhc", data_dir=str(cache)
        )


def test_the_config_runs_ancap8_alone_on_the_two_views(tmp_path):
    cfg = _compose("physics_ancap8")
    assert cfg.data_dir is None and cfg.tracking_uri is None
    assert cfg.out_dir.endswith("/eval/x_ancap8")
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    (tmp_path / "x").mkdir()
    with pytest.raises(ValueError, match="data_overrides need data_dir"):
        evaluator.evaluate_lattice(
            tmp_path, tmp_path / "out", "x", data_overrides=cfg.data_overrides
        )
    assert list(evaluator.dump_callbacks) == ["ancap8"]
    assert evaluator.test_callbacks == []
    callback = evaluator.dump_callbacks["ancap8"]
    assert callback["dataset_1"] == "normal" and callback["views"] == {"gun": SN}
    assert callback["points"] == list(FROZEN) and callback["kinds"] is True
    assert callback["sizes"] == {"n1": 50000, "n2": 50000, "n_p": 40000}
    points = evaluator.extra_sets["points"]
    assert list(evaluator.extra_sets) == ["points"] and list(points) == list(FROZEN)
    for k, (name, spec) in enumerate(points.items(), start=1):
        assert Path(spec["path"]).name == f"lhc_kind{k}.npz"
        assert (spec["md5_x"], spec["md5_mask"]) == FROZEN[name]
    assert evaluator.val_sets == ["normal", SN]
    assert evaluator.row_caps == {"seed": SEED, "val": {"normal": 50000, SN: 50000}}
    assert evaluator.rescored_sets == ["normal", SN]
    entry = {"name": "ancap8", "cells": ["gun"], "n_kinds": 4, "domain": "physics"}
    assert evaluator.selection == {"ancap8": entry}


def test_the_criterion_takes_the_configured_arguments():
    pytest.importorskip("src.evaluation.callbacks.ancap8")
    evaluator = hydra.utils.instantiate(_compose("physics_ancap8").evaluation.evaluator)
    callback = hydra.utils.instantiate(evaluator.dump_callbacks["ancap8"])
    assert isinstance(callback, Callback)


def test_the_criterion_runs_through_the_wiring(lattice, tmp_path):
    pytest.importorskip("src.evaluation.callbacks.ancap8")
    cfg = _config(lattice.points, "src.evaluation.callbacks.ancap8.ANCAP8")
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    out = tmp_path / "eval"
    evaluator.evaluate_lattice(lattice.scores, out, LATTICE)
    selection = OmegaConf.to_container(cfg.selection, resolve=True)
    select_lattice(out, selection, lattice.scores, LATTICE, None)
    for run in ("run_a", "run_b"):
        columns = list(_read_csv(out / "runs" / run / "val.csv")[0])
        assert columns[0] == "epoch" and len(columns) > 1
        assert all(c.startswith("ancap8/") and c.endswith("_gun") for c in columns[1:])
    for name in TABLES:
        table = pd.read_csv(out / f"ancap8_{name}.csv")
        assert list(table.columns[:2]) == ["domain", "lattice"]
