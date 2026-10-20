import csv
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from src.callbacks.autouad import EAGCallback, NPDCallback, RTMCallback
from src.callbacks.cap import CAPCallback
from src.callbacks.consistency import PosteriorConsistencyCallback
from src.callbacks.efficiency import AnomalyEfficiencyCallback
from src.callbacks.entropy import LossEntropyCallback
from src.callbacks.goix import ExcessMassCallback, MassVolumeCallback
from src.callbacks.metrics.selection import W_STEPS
from src.callbacks.operational import OperationalScoreCallback
from src.callbacks.sds import SDSCallback
from src.callbacks.thres_drift import ThresholdDriftCallback
from src.callbacks.wasserstein import WassersteinCallback
from tests.test_score_dump import ToyAE, ToyDataModule, dump_callback

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EPOCHS = 10
LATTICE, RUN = "toy", "run_a"
# The auxiliary sets and batch size of the dump, so that the callbacks score the points
# the dump stores, in the same batches.
AUX = {"aux_seed": 7, "aux_size": 50}
BATCH = 16
PAIR = {
    "output_name": "ascore/full",
    "dataset_1": "normal",
    "dataset_2": "reference_normal",
}
# The key under which the training efficiency callback logs one set's efficiency.
EFFICIENCY_KEY = "val/{}/eff__ascore_full__brate_operational"
# The labelled rows that the training efficiency callback logs; the oracle has no key.
LABELLED = ["cvar25", "cvar10"]
# Q'' of an epoch is the mean of per-batch values. The module sums them in float64 and
# logs the mean v from the hook of the last batch of normal, so that Lightning infers
# that batch's size n and logs fl(fl(n v) / n); the stage averages them with a float32
# MeanMetric. The raw value and its EMA differ by up to 2 float32 ulps on the toy.
MAX_ULP = {"val/normal/ascore_operational": 2, "val/summary/ascore_operational_ema": 2}


def _stage_config():
    cfg = OmegaConf.load(CONFIGS / "eval" / "cchamber.yaml")
    cfg.cap = OmegaConf.load(CONFIGS / "cap" / "provisional.yaml")
    # paths is not composed here, and evaluate_dump does not read the cluster file.
    cfg.selection.clusters = None
    OmegaConf.resolve(cfg)
    return cfg


def _trained(selection: dict) -> dict[str, dict]:
    # As the stage reads the registry: a labelled row is the val.csv column of its own
    # name and is maximised, Q'' is minimised.
    strategies = dict(selection["criteria"])
    for name in LABELLED:
        spec = selection["labelled"][name]
        strategies[name] = {**spec, "column": name, "mode": "max"}
    strategies["q_sec"] = {**selection["q_sec"], "mode": "min"}
    return strategies


def _strategy_callbacks(cap: dict) -> list:
    aux = {**AUX, "batch_size": BATCH}
    paired = {**PAIR, "pairing_type": "cdf", "cap_metric_config": cap}
    return [
        AnomalyEfficiencyCallback(output_name="ascore/full"),
        ThresholdDriftCallback(
            output_name="ascore/full", calibration_fraction=0.5, split_seed=12345
        ),
        WassersteinCallback(**PAIR, apply_log1p=True),
        CAPCallback(**paired),
        PosteriorConsistencyCallback(**paired),
        NPDCallback(**aux),
        RTMCallback(),
        EAGCallback(),
        ExcessMassCallback(**aux),
        MassVolumeCallback(**aux),
        LossEntropyCallback(batch_size=BATCH),
        SDSCallback(source="train", batch_size=BATCH),
        OperationalScoreCallback(),
    ]


def _checkpoint(root: Path, key: str, mode: str) -> ModelCheckpoint:
    # The layout of the physics experiments: summary/<metric>/<mode>/<metric>.ckpt.
    metric = key.removeprefix("val/summary/")
    return ModelCheckpoint(
        dirpath=root / "summary" / metric / mode,
        filename=metric,
        monitor=key,
        mode=mode,
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )


def _train(root: Path, callbacks: list, uri: str) -> str:
    # As fit of test_score_dump, whose Trainer fixes logger=False.
    logger = MLFlowLogger(experiment_name=LATTICE, run_name=RUN, tracking_uri=uri)
    pl.seed_everything(0)
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        devices=1,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        default_root_dir=root,
        num_sanity_val_steps=-1,
    )
    trainer.fit(ToyAE(), datamodule=ToyDataModule())
    return logger.run_id


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def _history(client: MlflowClient, run_id: str, key: str) -> tuple[list, list]:
    metrics = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
    return [m.step for m in metrics], [m.value for m in metrics]


def run_b9(root: Path, cfg) -> SimpleNamespace:
    uri, dump = f"file:{root / 'mlruns'}", root / "scores" / LATTICE / RUN
    selection = OmegaConf.to_container(cfg.selection)
    strategies = _trained(selection)
    checkpoints = {
        name: _checkpoint(root, spec["key"], spec["mode"])
        for name, spec in strategies.items()
    }
    callbacks = [dump_callback(dump, aux_batch_size=BATCH, **AUX)]
    callbacks += _strategy_callbacks(OmegaConf.to_container(cfg.cap))
    train_id = _train(root, [*callbacks, *checkpoints.values()], uri)

    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    out = root / "eval"
    run = evaluator.evaluate_dump(dump, ToyAE(), ToyDataModule(), out, LATTICE, uri)
    client = MlflowClient(uri)
    return SimpleNamespace(
        selection=selection,
        strategies=strategies,
        signals=run["signals"],
        train=lambda key: _history(client, train_id, key),
        stage=lambda key: _history(client, run["mlflow_run_id"], key),
        val=_read_csv(out / "runs" / RUN / "val.csv"),
        picks=_read_csv(out / "runs" / RUN / "picks.csv"),
        kept={
            name: torch.load(ckpt.best_model_path, weights_only=False)["epoch"]
            for name, ckpt in checkpoints.items()
        },
    )


@pytest.fixture(scope="module")
def b9(tmp_path_factory) -> SimpleNamespace:
    return run_b9(tmp_path_factory.mktemp("b9"), _stage_config())


def _pairs(b9) -> list[tuple[str, str]]:
    pairs = []
    for name, spec in b9.strategies.items():
        pairs.append((spec["key"], f"ema/{name}"))
        if "raw_key" in spec:
            pairs.append((spec["raw_key"], spec["column"]))
    prefix = b9.selection["efficiency"]
    return pairs + [(EFFICIENCY_KEY.format(s), prefix + s) for s in b9.signals]


def _assert_float32(actual: list, expected: list, max_ulp: int) -> None:
    actual, expected = np.float32(actual), np.float32(expected)
    if max_ulp:
        np.testing.assert_array_max_ulp(actual, expected, maxulp=max_ulp)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_stage_reproduces_the_logged_values(b9):
    assert [row["epoch"] for row in b9.val] == [str(e) for e in range(-1, EPOCHS)]
    for key, column in _pairs(b9):
        steps, logged = b9.train(key)
        assert len(steps) == EPOCHS, key
        table = [float(row[column]) for row in b9.val[1:]]
        _assert_float32(logged, table, MAX_ULP.get(key, 0))
        stage_steps, stage_logged = b9.stage(key)
        assert stage_steps == list(range(EPOCHS)), key
        _assert_float32(logged, stage_logged, MAX_ULP.get(key, 0))


def test_untrained_epoch_is_a_row_without_emas(b9):
    initial = b9.val[0]
    assert initial["epoch"] == "-1"
    names = {*b9.selection["criteria"], *b9.selection["labelled"], "q_sec"}
    assert all(initial[f"ema/{name}"] == "" for name in names)
    for _, column in _pairs(b9):
        if not column.startswith("ema/"):
            assert initial[column] != "", column


def test_picks_are_the_kept_checkpoints(b9):
    for name in b9.strategies:
        k = str(W_STEPS) if name in b9.selection["criteria"] else ""
        rows = [r for r in b9.picks if r["strategy"] == name and r["k"] == k]
        assert [int(row["epoch"]) for row in rows] == [b9.kept[name]], name
    # Some strategy keeps an epoch inside the run, which a monotone EMA never does, so
    # that the check is not one of the first and last epochs alone.
    assert set(b9.kept.values()) - {0, EPOCHS - 1}
