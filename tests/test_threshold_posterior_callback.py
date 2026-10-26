import math
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.components.selection import smoothed
from src.callbacks.threshold_posterior import (
    KEYS,
    ThresholdPosteriorCallback,
    threshold_posterior,
)
from src.evaluation.callbacks.threshold_posterior import ThresholdPosterior
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
from tests.test_eval_stage import TOY_BETA_FIX, _compose
from tests.test_score_dump import ToyAE, ToyDataModule, dump_callback, fit

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
POSTERIOR = OmegaConf.to_container(OmegaConf.load(CONFIGS / "cap" / "posterior.yaml"))
PROBES = ["autouad", "uniform", "gaussian"]
CONTROL = "posterior_reference_normal"
TARGET = "src.evaluation.callbacks.threshold_posterior.ThresholdPosterior"
TRAINING_ARGS = ["output_name", "dataset_1", "dataset_2", "posterior_config"]
# Fewer draws than registered, for speed.
DECLARED = POSTERIOR | {"beta_fix": TOY_BETA_FIX, "n_draws": 20}
SHIFTS = {
    "normal": 0.0,
    "sensor_weak": 0.4,
    "gaussian": 1.0,
    "uniform": 2.5,
    "autouad": 1.5,
}


@pytest.fixture(scope="module", autouse=True)
def _resolvers() -> None:
    # A second registration raises.
    if not OmegaConf.has_resolver("short_hash"):
        register_resolvers()


class Scorer(pl.LightningModule):
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return {"ascore/full": x.sum(dim=1)}


def _loaders(names) -> dict[str, DataLoader]:
    g = torch.Generator().manual_seed(0)
    loaders = {}
    for name in names:
        n = {"normal": 60, "sensor_weak": 90}.get(name, 40)
        x = torch.randn(n, 3, generator=g) + SHIFTS[name]
        y = torch.zeros(n, dtype=torch.long)
        loaders[name] = DataLoader(TensorDataset(x, y), batch_size=16)
    return loaders


def _evaluate(callback, loaders) -> dict:
    trainer = pl.Trainer(
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[callback],
    )
    trainer.metric_records = {}
    module = Scorer()
    module._ckpt_path = "epoch_0003.pt"
    trainer.test(module, dataloaders=loaders, verbose=False)
    return trainer.metric_records


def _scores(loader) -> torch.Tensor:
    return torch.cat([x.sum(dim=1) for x, _ in loader])


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("trained")
    uri, dump = f"file:{root / 'mlruns'}", root / "scores" / LATTICE / RUN
    cfg = _stage_config()
    control = OmegaConf.to_container(cfg.evaluation.evaluator.dump_callbacks[CONTROL])
    callback = ThresholdPosteriorCallback(
        **{key: control[key] for key in TRAINING_ARGS},
        aux_seed=AUX["aux_seed"],
        aux_size=AUX["aux_size"],
        batch_size=BATCH,
    )
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


def test_evaluation_records_every_key_of_every_probe():
    loaders = _loaders(["normal", "sensor_weak", *PROBES])
    callback = ThresholdPosterior(
        "ascore/full", "normal", "sensor_weak", DECLARED, name="posterior_x"
    )
    records = _evaluate(callback, loaders)
    assert set(records) == {f"posterior_x/{p}_{key}" for p in PROBES for key in KEYS}
    scores = {name: _scores(loader) for name, loader in loaders.items()}
    for probe in PROBES:
        beta = TOY_BETA_FIX[probe]
        expected = threshold_posterior(
            scores["normal"], scores["sensor_weak"], scores[probe], beta, DECLARED
        )
        for key, value in expected.items():
            assert math.isfinite(value), (probe, key)
            assert records[f"posterior_x/{probe}_{key}"] == value, (probe, key)


def test_evaluation_names_the_missing_loaders():
    loaders = _loaders(["normal", "sensor_weak", "autouad", "gaussian"])
    callback = ThresholdPosterior("ascore/full", "normal", "sensor_weak", DECLARED)
    with pytest.raises(ValueError, match=r"\['uniform'\].*Available test dataloaders"):
        _evaluate(callback, loaders)


def test_a_null_beta_fix_raises_at_the_first_computation_only(tmp_path):
    loaders = _loaders(["normal", "sensor_weak", *PROBES])
    undeclared = DECLARED | {"beta_fix": TOY_BETA_FIX | {"uniform": None}}
    callback = ThresholdPosterior("ascore/full", "normal", "sensor_weak", undeclared)
    with pytest.raises(ValueError, match=r"\['uniform'\].*declares it"):
        _evaluate(callback, loaders)

    undeclared = POSTERIOR | {"beta_fix": dict.fromkeys(PROBES)}
    training = ThresholdPosteriorCallback(
        "ascore/full", "normal", "reference_normal", undeclared, aux_size=50
    )
    with pytest.raises(ValueError, match="before any label"):
        fit(tmp_path, [training], epochs=1)


def test_training_logs_the_stage_values(trained):
    assert len(trained.keys) == 2 * len(PROBES)
    key = "val/summary/posterior_uniform_D-fix_ema_normal_vs_reference_normal"
    assert trained.keys[("uniform", "D-fix")] == key
    for (probe, statistic), key in trained.keys.items():
        steps, logged = trained.train(key)
        assert len(steps) == EPOCHS, key
        column = f"{CONTROL}/{probe}_{statistic}"
        raw = [float(row[column]) for row in trained.val[1:]]
        assert all(math.isfinite(value) for value in raw), key
        expected = smoothed(raw, skip_nonfinite=True)
        np.testing.assert_array_equal(np.float32(logged), expected, err_msg=key)


def test_stage_writes_every_control_column(trained):
    columns = {f"{CONTROL}/{p}_{key}" for p in PROBES for key in KEYS}
    for row in trained.val:
        assert columns <= row.keys()
        for probe in PROBES:
            values = {key: float(row[f"{CONTROL}/{probe}_{key}"]) for key in KEYS}
            assert all(math.isfinite(value) for value in values.values()), probe
            assert values["beta_fix"] == TOY_BETA_FIX[probe]
            assert values["D-fix"] == pytest.approx(values["A-fix"] - values["A_pi"])
            assert values["A-fix"] == pytest.approx(values["C"] + values["F"])


@pytest.mark.parametrize(
    "domain, sets",
    [
        ("cchamber", ["sensor_weak", "sensor_mid", "sensor_strong"]),
        ("cchamber_wt", ["sensor_osr"]),
    ],
)
def test_posterior_instances_copy_the_control_and_feed_the_registry(domain, sets):
    cfg = _compose(domain)
    posterior = OmegaConf.to_container(cfg.cap_posterior)
    # The wind tunnel never inherits the light tunnel's declaration.
    if domain == "cchamber_wt":
        assert posterior == POSTERIOR | {"beta_fix": dict.fromkeys(PROBES)}
    else:
        assert posterior == POSTERIOR
    callbacks = cfg.evaluation.evaluator.dump_callbacks
    callbacks = OmegaConf.to_container(callbacks, resolve=True)
    control = callbacks[CONTROL]
    assert control["dataset_1"] == "normal"
    assert control["dataset_2"] == "reference_normal"
    assert control["posterior_config"] == posterior
    instances = {name for name, c in callbacks.items() if c["_target_"] == TARGET}
    assert instances == {CONTROL, *(f"posterior_{x2}" for x2 in sets)}
    for x2 in sets:
        name = f"posterior_{x2}"
        assert callbacks[name] == {**control, "dataset_2": x2, "name": name}
        assert hydra.utils.instantiate(callbacks[name]).name == name

    criteria = {
        name: OmegaConf.to_container(spec)
        for name, spec in cfg.selection.criteria.items()
        if name.startswith("posterior")
    }
    produced = {
        f"{name}/{probe}_{key}"
        for name in instances
        for probe in PROBES
        for key in KEYS
    }
    registered = set()
    for x2 in sets:
        keys = ThresholdPosteriorCallback("ascore/full", "normal", x2, posterior).keys
        grade = x2.removeprefix("sensor_")
        for (probe, statistic), key in keys.items():
            name = f"posterior_{probe}_{statistic.replace('-', '').lower()}_{grade}"
            column = f"posterior_{x2}/{probe}_{statistic}"
            spec = {"column": column, "mode": "max", "key": key}
            assert criteria[name] == spec | {"skip_nonfinite": True}, name
            assert column in produced
            registered.add(name)
    assert registered == criteria.keys() and len(registered) == 6 * len(sets)
