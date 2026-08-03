"""Tests for cross-dataset threshold-transfer model selection."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.callbacks.thres_drift import ThresholdDriftCallback as TrainingDrift
from src.evaluation.callbacks.thres_drift import (
    ThresholdDriftCallback as EvaluationDrift,
)


class _Module:
    """Minimal Lightning-module stand-in for direct callback tests."""

    def __init__(self, ckpt_path: Path | None = None):
        self.device = torch.device("cpu")
        self.hparams = SimpleNamespace(target_rate=0.25, base_rate=1.0)
        self.logged = {}
        if ckpt_path is not None:
            self._ckpt_path = ckpt_path

    def log_dict(self, values, **kwargs):
        """Capture logged metrics without a Lightning trainer."""
        del kwargs
        self.logged.update(values)


def _feed(callback, hook_name: str, trainer, module) -> None:
    """Feed fixed calibration and shifted evaluation scores to a callback."""
    hook = getattr(callback, hook_name)
    for index, scores in enumerate(
        (torch.tensor([0.0, 1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 4.0, 5.0]))
    ):
        hook(
            trainer,
            module,
            {"ascore/full": scores},
            None,
            0,
            dataloader_idx=index,
        )


def test_training_drift_transfers_background0_threshold_to_background1() -> None:
    """Training drift must calibrate on dataset 1 and apply to dataset 2."""
    callback = TrainingDrift(
        output_name="ascore/full",
        dataset_1="background0",
        dataset_2="background1",
        beta=0.0,
    )
    trainer = SimpleNamespace(val_dataloaders={"background0": None, "background1": None})
    module = _Module()

    callback.on_fit_start(trainer, module)
    callback.on_validation_epoch_start(trainer, module)
    _feed(callback, "on_validation_batch_end", trainer, module)
    callback.on_validation_epoch_end(trainer, module)

    # Calibration threshold is 2; every background1 score exceeds it.
    expected = math.log(3.0)
    assert module.logged["val/summary/operational_drift_ema"] == pytest.approx(expected)


def test_evaluation_drift_uses_two_named_datasets(tmp_path, monkeypatch) -> None:
    """Evaluation drift must retain the same cross-dataset direction."""
    callback = EvaluationDrift(
        output_name="ascore/full",
        dataset_1="background0",
        dataset_2="background1",
        log_raw_mlflow=False,
    )
    monkeypatch.setattr(callback, "_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "src.evaluation.callbacks.thres_drift.utils.mlflow.log_plots_to_mlflow",
        lambda *args, **kwargs: None,
    )
    trainer = SimpleNamespace(
        test_dataloaders={"background0": None, "background1": None}, split="test"
    )
    module = _Module(tmp_path / "last.ckpt")

    callback.on_test_epoch_start(trainer, module)
    _feed(callback, "on_test_batch_end", trainer, module)
    callback.on_test_epoch_end(trainer, module)

    assert callback.transfer_summary["operational"]["last"] == pytest.approx(math.log(3.0))
