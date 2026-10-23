import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from src.callbacks.cap_bank import CAPBankCallback
from src.callbacks.components.selection import smoothed
from src.evaluation.callbacks.components.cap.bank import cap_bank
from src.evaluation.callbacks.components.cap.binary import pairing
from src.evaluation.callbacks.components.cap.binary.pairing import CDFCoupling, cdf
from src.evaluation.callbacks.components.cap.metric import ApproximationCapacity
from src.utils.omegaconf import register_resolvers
from tests.test_eval_b9 import (
    AUX,
    BATCH,
    EPOCHS,
    LATTICE,
    RUN,
    _history,
    _read_csv,
    _stage_config,
    _train,
)
from tests.test_eval_stage import TOY_SETS, _compose, _evaluator
from tests.test_score_dump import ToyAE, ToyDataModule, dump_callback, fit

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BANK = OmegaConf.to_container(OmegaConf.load(CONFIGS / "cap" / "bank.yaml"))
PROVISIONAL = OmegaConf.to_container(
    OmegaConf.load(CONFIGS / "cap" / "provisional.yaml")
)
CONTROL = "cap_bank_reference_normal"
SENSORS = ["sensor_weak", "sensor_mid", "sensor_strong"]
DIAGNOSTICS = ["beta_star", "at_bound", "sigma_pi", "A_pi", "C", "F", "C_pi"]
LEGACY = ["A-sup", "beta_star", "at_bound"]
# The pairings whose couplings are cached; cdf reads the scores alone.
CACHED = ["nn", "ot", "strat"]
TRAINING_ARGS = ["output_name", "dataset_1", "dataset_2", "reference", "bank_config"]
LOG_2 = math.log(2.0)
# A run directory of a real dump, for the regression guard on stored scores.
REAL_DUMP = os.environ.get("ADL1T_CAP_DUMP")


@pytest.fixture(scope="module", autouse=True)
def _resolvers() -> None:
    # A second registration raises.
    if not OmegaConf.has_resolver("short_hash"):
        register_resolvers()


def _cap_config(control: dict) -> dict:
    # The NeurIPS CAP callback on the pair of the control.
    return {
        "_target_": "src.evaluation.callbacks.cap.CAP",
        "output_name": control["output_name"],
        "dataset_1": control["dataset_1"],
        "dataset_2": control["dataset_2"],
        "pairing_type": "cdf",
        "cap_metric_config": PROVISIONAL,
        "log_raw_mlflow": False,
        "name": "cap",
    }


@pytest.fixture(scope="module")
def stage(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("stage")
    path = root / "scores" / "toy" / "run_a"
    fit(root, [dump_callback(path)])
    evaluator = _evaluator(TOY_SETS)
    control = evaluator.dump_callbacks[CONTROL]
    evaluator.dump_callbacks["cap"] = _cap_config(control)
    out = root / "eval"
    evaluator.evaluate_dump(path, ToyAE(), ToyDataModule(), out, "toy")
    val = _read_csv(out / "runs" / "run_a" / "val.csv")
    return SimpleNamespace(
        evaluator=evaluator, control=control, path=path, out=out, val=val
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("trained")
    uri, dump = f"file:{root / 'mlruns'}", root / "scores" / LATTICE / RUN
    cfg = _stage_config()
    control = OmegaConf.to_container(cfg.evaluation.evaluator.dump_callbacks[CONTROL])
    callback = CAPBankCallback(**{key: control[key] for key in TRAINING_ARGS})
    callbacks = [dump_callback(dump, aux_batch_size=BATCH, **AUX), callback]
    run_id = _train(root, callbacks, uri)
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    evaluator.evaluate_dump(dump, ToyAE(), ToyDataModule(), root / "eval", LATTICE)
    client = MlflowClient(uri)
    return SimpleNamespace(
        keys=callback.keys,
        train=lambda key: _history(client, run_id, key),
        val=_read_csv(root / "eval" / "runs" / RUN / "val.csv"),
    )


def _bank_keys(config: dict) -> set[str]:
    pairings, maps = config["pairings"], config["maps"]
    columns = [*config["statistics"], *DIAGNOSTICS]
    keys = {f"{p}_{m}_{c}" for p in pairings for m in maps for c in columns}
    keys |= {f"{m}_beta_fix" for m in maps} | {f"{p}_discarded" for p in pairings}
    return keys | {f"cdf_legacy_{key}" for key in LEGACY}


def _columns(row: dict, name: str) -> dict[str, float]:
    prefix = f"{name}/"
    return {
        key.removeprefix(prefix): float(value)
        for key, value in row.items()
        if key.startswith(prefix)
    }


def _paired_cap(scores_1: torch.Tensor, scores_2: torch.Tensor):
    # As src/evaluation/callbacks/cap.py pairs them: cdf, then the shorter length.
    index_1, index_2 = cdf(scores_1, scores_2)
    paired_1, paired_2 = scores_1[index_1], scores_2[index_2]
    n = min(len(paired_1), len(paired_2))
    metric = ApproximationCapacity(**PROVISIONAL)
    metric.update(paired_1[:n], paired_2[:n])
    return metric


def _fail(*args, **kwargs):
    raise AssertionError("A cached coupling was built again.")


def test_bank_callbacks_copy_the_control():
    cfg = _compose("cchamber")
    assert "cap" not in cfg and OmegaConf.to_container(cfg.cap_bank) == BANK
    callbacks = cfg.evaluation.evaluator.dump_callbacks
    callbacks = OmegaConf.to_container(callbacks, resolve=True)
    control = callbacks[CONTROL]
    assert control["bank_config"] == BANK
    assert control["dataset_1"] == "normal"
    assert control["dataset_2"] == control["reference"] == "reference_normal"
    for sensor in SENSORS:
        name = f"cap_bank_{sensor}"
        assert callbacks[name] == {**control, "dataset_2": sensor, "name": name}
    columns = [spec.column for spec in cfg.selection.criteria.values()]
    assert not [column for column in columns if column.startswith("cap_bank")]


def test_stage_writes_finite_bank_columns(stage):
    assert [row["epoch"] for row in stage.val] == ["-1", "0", "1"]
    for row in stage.val:
        values = _columns(row, CONTROL)
        assert values.keys() == _bank_keys(BANK)
        assert [key for key, value in values.items() if not math.isfinite(value)] == []
        for name, beta in BANK["beta_fix"].items():
            assert values[f"{name}_beta_fix"] == beta


def test_legacy_corner_is_the_old_cap_on_the_same_pass(stage):
    for row in stage.val:
        bank, old = _columns(row, CONTROL), _columns(row, "cap")
        legacy = bank["cdf_legacy_A-sup"] - LOG_2
        assert legacy == pytest.approx(old["cap"], rel=0, abs=1e-10)
        assert bank["cdf_legacy_beta_star"] == old["cap_beta_star"]
        assert bank["cdf_legacy_at_bound"] == old["cap_at_bound"]


def test_second_stage_reads_the_cached_couplings(stage, monkeypatch):
    cache = stage.out / "cap_bank"
    names = sorted(path.name for path in cache.iterdir())
    pairs = [name for name in CACHED for _ in range(2)]
    assert [name.split("_")[0] for name in names] == pairs
    assert [Path(name).suffix for name in names] == [".json", ".pt"] * len(CACHED)
    label = {key: stage.control[key] for key in ("dataset_1", "dataset_2", "reference")}
    for name in names[::2]:
        assert json.loads((cache / name).read_text())["label"] == label

    for name in CACHED:
        monkeypatch.setitem(pairing.COUPLINGS, name, _fail)
    stage.evaluator.evaluate_dump(stage.path, ToyAE(), ToyDataModule(), stage.out)
    again = _read_csv(stage.out / "runs" / "run_a" / "val.csv")
    expected = [_columns(row, CONTROL) for row in stage.val]
    assert [_columns(row, CONTROL) for row in again] == expected
    assert sorted(path.name for path in cache.iterdir()) == names


def test_training_logs_the_stage_values(trained):
    groups = ("pairings", "maps", "statistics")
    assert len(trained.keys) == math.prod(len(BANK[group]) for group in groups)
    key = "val/summary/cap_ot_std_A-sup_ema_reference_normal_vs_normal"
    assert trained.keys["ot_std_A-sup"] == key
    for name, key in trained.keys.items():
        steps, logged = trained.train(key)
        assert len(steps) == EPOCHS, key
        raw = [float(row[f"{CONTROL}/{name}"]) for row in trained.val[1:]]
        expected = smoothed(raw, skip_nonfinite=True)
        np.testing.assert_array_equal(np.float32(logged), expected, err_msg=key)


@pytest.mark.skipif(REAL_DUMP is None, reason="ADL1T_CAP_DUMP names no real run.")
def test_legacy_corner_is_the_old_cap_on_a_real_run():
    files = sorted(Path(REAL_DUMP).glob("epoch_*.pt"))
    for path in (files[0], files[len(files) // 2], files[-1]):
        val = torch.load(path, weights_only=True)["val"]
        normal = val["normal"]["ascore/full"]
        reference = val["reference_normal"]["ascore/full"]
        values = cap_bank(normal, reference, CDFCoupling(), BANK)
        old = _paired_cap(normal, reference)
        legacy = values["legacy_A-sup"] - LOG_2
        assert legacy == pytest.approx(old.compute(), rel=0, abs=1e-10)
        assert values["legacy_beta_star"] == old.beta_star
        assert values["legacy_at_bound"] == float(old.at_bound)
