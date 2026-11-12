import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

from src.callbacks.ancap8 import ancap8, keys
from src.data.Dijet_datamodule import DijetDataModule
from src.data.SBC_datamodule import SBCDataModule
from src.evaluation.components.rebuild import L1AD_DATAMODULES, rebuild
from tests import test_dijet_datamodule as dijet
from tests import test_sbc_datamodule as sbc
from tests.test_eval_stage import CONFIGS, _compose, _read_csv
from tests.test_lattice import _eval_entry
from tests.test_score_dump import ToyAE, dump_callback, fit

# The lattice name that _compose gives every config.
LATTICE = "x"
TABLES = ["candidates", "picks", "beside", "kinds", "condition"]
# Per config: its cells, declared sizes, md5 of the points' x, sets and empty_signals.
DOMAINS = {
    "dijet": SimpleNamespace(
        cells=["simulation"],
        sizes={"n1": 50000, "n2": 50000, "n_p": 6363},
        md5_x="15152a32cdf170061bad273f3b2e129d",
        sets="dijet",
        empty_signals=True,
    ),
    "sbc": SimpleNamespace(
        cells=["leipzig_validation", "greifswald"],
        sizes={"n1": 50000, "n2": 50000, "n_p": 3822},
        md5_x="26649eab4670ca52b41c5986a3a0d987",
        sets="sbc",
        empty_signals=True,
    ),
    "cchamber": SimpleNamespace(
        cells=["sensor_weak", "sensor_mid", "sensor_strong"],
        sizes={"n1": 1000, "n2": 5400, "n_p": 1611},
        md5_x="7d24bd0aeb57732bad11ebff54c5706c",
        sets="cchamber",
        empty_signals=False,
    ),
    "cchamber_wt": SimpleNamespace(
        cells=["sensor_osr", "walk_null"],
        sizes={
            "n1": 1000,
            "n2": {"sensor_osr": 60000, "walk_null": 30000},
            "n_p": 1420,
        },
        md5_x="f9cd543df9b09be292b607122d9c8ec3",
        sets="cchamber_wt",
        empty_signals=False,
    ),
}
# Rows per file of a toy dijet record, every one finite, so that S1 and the cell are
# large enough for the criterion's cuts.
DIJET_ROWS = {
    "qcd_data_samples/qcd_0.txt": 400,
    "qcd_data_samples/qcd_1.txt": 400,
    "qcd_test_samples/qcd_0.txt": 200,
    "qcd_mc_samples/qcd_0.txt": 300,
    "sig_samples/sig_a.txt": 50,
    "sig_samples/sig_b.txt": 50,
}
# The toy's validation normal, simulation and points.
SIZES = {"n1": 200, "n2": 150, "n_p": 120}
DIJET_AE = "tests.test_eval_ancap8_domains.DijetToyAE"


class DijetToyAE(ToyAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = torch.nn.Linear(len(dijet.INPUTS), len(dijet.INPUTS))


def _kwargs(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "_target_"}


def _built(cls, kwargs: dict):
    datamodule = cls(**kwargs)
    datamodule.setup("fit")
    return datamodule


def _rows(datamodule) -> dict[str, torch.Tensor]:
    # The inputs of every evaluation set, keyed '<split>/<set>', and the training rows.
    loaders = {"val": datamodule.val_dataloader(), "test": datamodule.test_dataloader()}
    rows = {"train": datamodule.data_train.tensors[0]}
    for split, named in loaders.items():
        for name, loader in named.items():
            rows[f"{split}/{name}"] = torch.cat([batch[0] for batch in loader])
    return rows


@pytest.fixture(scope="module")
def dijet_data(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("dijet")
    for seed, (name, rows) in enumerate(DIJET_ROWS.items()):
        dijet._write(root / "dijet" / name, rows, 0, 0, seed)
    files = list(DIJET_ROWS)
    return {
        "_target_": "src.data.Dijet_datamodule.DijetDataModule",
        "data_dir": str(root),
        "dataset_name": "dijet",
        "url": "https://example.invalid/{name}",
        "archives": {},
        "inputs": dijet.INPUTS,
        "log_inputs": dijet.LOGGED,
        "reference": {"files": files[:2], "n_train": 400, "n_valid": SIZES["n1"]},
        "test_reference": {"files": [files[2]], "n_test": 100},
        "signals": {Path(f).stem: {"files": [f]} for f in files[4:]},
        "nulls": {"simulation": {"files": [files[3]], "n_valid": 150, "n_test": 150}},
        "batch_size": 32,
    }


@pytest.fixture(scope="module")
def sbc_kwargs(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("sbc")
    rows = sbc._synthetic_rows(np.random.default_rng(0))
    sbc._write(root / "extdata" / "sbcdata.csv", rows)
    data = OmegaConf.to_container(OmegaConf.load(CONFIGS / "data" / "sbc.yaml"))
    names = ("url", "md5", "inputs", "log_inputs", "reference", "signals", "nulls")
    lists = {name: data[name] for name in names}
    return lists | sbc.SIZES | {"data_dir": str(root), "batch_size": 8}


@pytest.mark.parametrize("domain", list(DOMAINS))
def test_each_config_runs_ancap8_alone_on_its_cells(domain):
    spec = DOMAINS[domain]
    cfg = _compose(f"{domain}_ancap8")
    assert cfg.scores_dir.endswith(f"/lattices/{domain}")
    assert cfg.out_dir.endswith("/eval/x_ancap8") and cfg.tracking_uri is None
    assert (cfg.sets, cfg.empty_signals) == (spec.sets, spec.empty_signals)
    assert cfg.data_dir is None and cfg.data_overrides is None
    assert cfg.partial_runs is None and cfg.typical is None
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    assert list(evaluator.dump_callbacks) == ["ancap8"]
    assert evaluator.test_callbacks == []
    callback = evaluator.dump_callbacks["ancap8"]
    assert callback["dataset_1"] == "normal"
    assert callback["views"] == {cell: cell for cell in spec.cells}
    assert callback["points"] == ["push"] and callback["kinds"] is False
    assert callback["sizes"] == spec.sizes
    assert list(evaluator.extra_sets) == ["points"]
    push = dict(evaluator.extra_sets["points"]["push"])
    path = Path(cfg.paths.root_dir) / "pushed_points" / domain / f"{domain}_pushed.npz"
    assert Path(push.pop("path")) == path
    assert push == {"md5_x": spec.md5_x, "md5_mask": None}
    assert evaluator.val_sets == ["normal", *spec.cells]
    assert evaluator.row_caps is None
    assert evaluator.rescored_sets == ["normal", *spec.cells]
    entry = {"name": "ancap8", "cells": spec.cells, "n_kinds": 0, "domain": domain}
    assert evaluator.selection == {"ancap8": entry}
    assert isinstance(hydra.utils.instantiate(callback), Callback)


@pytest.mark.parametrize(
    "path", sorted((CONFIGS / "eval").glob("*_ancap8.yaml")), ids=lambda p: p.stem
)
def test_every_v8_config_rescores_its_views_in_the_pass_of_its_points(path):
    # The ruling of 6 Nov 04:00 (plans/adl1t/08_tests.md): the points are rescored at
    # every replay, so S1 and the cells must be too, never served from the dump.
    evaluator = _compose(path.stem).evaluation.evaluator
    callback = evaluator.dump_callbacks.ancap8
    views = {callback.dataset_1, *callback.views.values()}
    assert views <= set(evaluator.rescored_sets or [])


def test_dijet_rows_do_not_depend_on_the_signals(dijet_data):
    full = _built(DijetDataModule, _kwargs(dijet_data))
    empty = _built(DijetDataModule, _kwargs(dijet_data) | {"signals": {}})
    rows, kept = _rows(full), _rows(empty)
    signals = {f"{split}/sig_{s}" for split in ("val", "test") for s in "ab"}
    assert rows.keys() - kept.keys() == signals
    assert all(torch.equal(rows[name], x) for name, x in kept.items())
    assert full.counts.keys() - empty.counts.keys() == {"sig_a", "sig_b"}


def test_sbc_rows_do_not_depend_on_the_signals(sbc_kwargs):
    full = _built(SBCDataModule, sbc_kwargs)
    empty = _built(SBCDataModule, sbc_kwargs | {"signals": {}})
    rows, kept = _rows(full), _rows(empty)
    names = set(sbc_kwargs["signals"])
    assert rows.keys() - kept.keys() == {
        f"{s}/{n}" for s in ("val", "test") for n in names
    }
    assert all(torch.equal(rows[name], x) for name, x in kept.items())
    assert full.case_counts.keys() - empty.case_counts.keys() == names


def test_empty_signals_builds_no_signal_set(dijet_data, tmp_path):
    run = {"data": dijet_data, "algorithm": {"_target_": DIJET_AE}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(run))
    datamodule = rebuild(tmp_path, None, None, None, True)[1]
    datamodule.setup("fit")
    sets = ["normal", "reference_normal", "simulation"]
    assert list(datamodule.val_dataloader()) == sets
    assert list(datamodule.test_dataloader()) == sets
    assert not {"sig_a", "sig_b"} & datamodule.counts.keys()
    # After sets: the signals of configs/data/dijet.yaml are emptied, its nulls kept.
    datamodule = rebuild(tmp_path, None, "dijet", None, True)[1]
    nulls = OmegaConf.load(CONFIGS / "data" / "dijet.yaml").nulls
    assert datamodule.hparams.signals == {} and datamodule.hparams.nulls == nulls


@pytest.mark.parametrize(
    "data, message",
    [
        ({"_target_": L1AD_DATAMODULES[0], "signals": {"sig_a": {}}}, "L1AD"),
        ({"_target_": L1AD_DATAMODULES[1], "overrides": []}, "L1AD"),
        ({"_target_": "tests.test_score_dump.ToyDataModule"}, "takes no signals"),
    ],
)
def test_empty_signals_is_refused_without_a_signal_list(tmp_path, data, message):
    run = {"data": data, "algorithm": {"_target_": "tests.test_score_dump.ToyAE"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(run))
    with pytest.raises(ValueError, match=message):
        rebuild(tmp_path, None, None, None, True)


def test_emptied_signals_fetch_no_signal_archive(dijet_data, tmp_path, monkeypatch):
    fetched = []

    def fake_download(url: str, path: Path) -> None:
        fetched.append(path.name)

    module = "src.data.Dijet_datamodule"
    monkeypatch.setattr(f"{module}.download", fake_download)
    monkeypatch.setattr(f"{module}.check_md5", lambda path, md5: None)
    monkeypatch.setattr(f"{module}._extract_folders", lambda *args: None)
    # The record's two archives, on a data directory that lacks the signal folder.
    ignored = shutil.ignore_patterns("sig_samples")
    shutil.copytree(dijet_data["data_dir"], tmp_path / "data", ignore=ignored)
    archives = OmegaConf.load(CONFIGS / "data" / "dijet.yaml").archives
    kwargs = _kwargs(dijet_data) | {"data_dir": str(tmp_path / "data")}
    kwargs |= {"archives": OmegaConf.to_container(archives)}
    DijetDataModule(**kwargs | {"signals": {}}).prepare_data()
    assert fetched == []
    DijetDataModule(**kwargs).prepare_data()
    assert fetched == ["sig_samples.zip"]


def test_n2_may_be_declared_per_cell():
    rng = np.random.default_rng(0)
    scores_1, points = rng.normal(size=400), rng.normal(1.0, 1.0, size=200)
    views = {"a": rng.normal(size=300), "b": rng.normal(size=250)}
    sizes = {"n1": 400, "n2": {"a": 300, "b": 200}, "n_p": 200}
    values = ancap8(scores_1, views, [points], sizes=sizes)
    assert (values["usable_a"], values["usable_b"]) == (1, 0)
    assert values["n2_a"] == 300


@pytest.fixture(scope="module")
def dijet_lattice(dijet_data, tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("dijet_lattice")
    run_config = {"data": dijet_data, "algorithm": {"_target_": DIJET_AE}}
    for seed, run in enumerate(["run_a", "run_b"]):
        path = root / "scores" / LATTICE / run
        # The dijet dumps' form: weights and probes, no validation set stored.
        callback = dump_callback(path, stored_sets=[], train_pass=False)
        datamodule = DijetDataModule(**_kwargs(dijet_data))
        fit(root, [callback], seed=seed, module=DijetToyAE(), datamodule=datamodule)
        (path / "config.yaml").write_text(yaml.safe_dump(run_config))
    # The evaluation's copy of the record lacks the signal files, which a read would
    # find missing.
    ignored = shutil.ignore_patterns("sig_samples")
    shutil.copytree(dijet_data["data_dir"], root / "data", ignore=ignored)
    shape = (SIZES["n_p"], len(dijet.INPUTS))
    x = np.random.default_rng(0).normal(0.0, 2.0, shape).astype(np.float32)
    # The layout of the frozen files: x, rows and scale, no mask.
    np.savez(root / "push.npz", x=x, rows=np.arange(len(x)), scale=np.float64(1.0))
    md5_x = hashlib.md5(x.tobytes(), usedforsecurity=False).hexdigest()
    points = {"path": str(root / "push.npz"), "md5_x": md5_x, "md5_mask": None}
    return SimpleNamespace(root=root, points=points)


def test_a_plain_domain_reaches_the_lattice_tables(dijet_lattice, monkeypatch):
    root, out = dijet_lattice.root, dijet_lattice.root / "eval"
    overrides = [f"scores_dir={root / 'scores'}", f"out_dir={out}"]
    overrides += [f"data_dir={root / 'data'}", "~extras"]
    # The toy's own lists: data/dijet.yaml's null set draws 100,000 rows.
    overrides += ["sets=null"]
    cfg = _compose("dijet_ancap8", *overrides)
    cfg.evaluation.evaluator.extra_sets.points.push = dijet_lattice.points
    cfg.evaluation.evaluator.dump_callbacks.ancap8.sizes = SIZES
    _eval_entry(monkeypatch).main(cfg)

    columns = {f"ancap8/{key}_simulation" for key in keys(0)}
    for run in ("run_a", "run_b"):
        val = _read_csv(out / "runs" / run / "val.csv")
        assert [row["epoch"] for row in val] == ["-1", "0", "1"]
        assert set(val[0]) == {"epoch"} | columns
        info = json.loads((out / "runs" / run / "run.json").read_text())
        assert info["signals"] == [] and info["nulls"] == []
    tables = {name: pd.read_csv(out / f"ancap8_{name}.csv") for name in TABLES}
    for table in tables.values():
        assert list(table.columns[:2]) == ["domain", "lattice"]
        assert set(table["domain"]) <= {"dijet"} and set(table["lattice"]) <= {"x"}
    assert tables["kinds"].empty
    candidates = tables["candidates"]
    assert len(candidates) == 6 and set(candidates["cell"]) == {"simulation"}
    assert (candidates["usable"] == 1).all()
    assert (candidates[["n1", "n2", "nP"]] == list(SIZES.values())).all(axis=None)
    assert tables["picks"]["cell"].tolist() == ["simulation"]
    assert tables["condition"]["cell"].tolist() == ["simulation"]
