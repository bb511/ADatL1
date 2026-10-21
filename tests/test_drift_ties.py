import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.callbacks.components.rate import compute_tie_weight, count_exceedances
from src.callbacks.thres_drift import ThresholdDriftCallback as TrainDrift
from src.evaluation.callbacks.thres_drift import ThresholdDriftCallback as EvalDrift

# fpr: drift of the code before October 2026 on the `scores` fixture.
REGRESSION = {
    1e-3: 0.06124362524071867,
    1e-2: 0.044951387862266275,
    3.65e-4: 0.40546510810816444,
}


@pytest.fixture(scope="module")
def scores() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(100_000)


def _module(fpr: float, **kwargs) -> SimpleNamespace:
    hparams = SimpleNamespace(target_rate=fpr, base_rate=None)
    return SimpleNamespace(device=torch.device("cpu"), hparams=hparams, **kwargs)


def _train_drift(scores: torch.Tensor, fpr: float) -> float:
    logged = {}
    module = _module(fpr, log_dict=lambda metrics, **_: logged.update(metrics))
    trainer = SimpleNamespace(val_dataloaders={"normal": None}, sanity_checking=False)
    callback = TrainDrift(output_name="s")
    callback.on_fit_start(trainer, module)
    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(trainer, module, {"s": scores}, None, 0, 0)
    callback.on_validation_epoch_end(trainer, module)
    return logged["val/summary/operational_drift_ema"]


def _eval_drift(scores: torch.Tensor, fpr: float, tmp_path: Path) -> float:
    module = _module(fpr, _ckpt_path=str(tmp_path / "ckpt.ckpt"))
    trainer = SimpleNamespace(
        test_dataloaders={"normal": None}, split="val", logger=None, loggers=[]
    )
    callback = EvalDrift(output_name="s")
    callback._plot = lambda *args, **kwargs: None
    callback.on_test_epoch_start(trainer, module)
    callback.on_test_batch_end(trainer, module, {"s": scores}, None, 0, 0)
    callback.on_test_epoch_end(trainer, module)
    return callback.transfer_summary["operational"]["ckpt"]


def _old_drift(scores: torch.Tensor, fpr: float) -> float:
    calibration, evaluation = TrainDrift(output_name="s")._split_scores(scores)
    n = calibration.numel()
    index = max(0, min(n - 1, int(math.ceil((1.0 - fpr) * n) - 1)))
    threshold = torch.sort(calibration).values[index]
    p_hat = int((evaluation >= threshold).sum().item()) / float(evaluation.numel())
    eps = 0.5 / float(evaluation.numel())
    return abs(math.log((p_hat + eps) / (fpr + eps)))


@pytest.mark.parametrize("fpr", list(REGRESSION))
def test_untied_drift_unchanged(scores, fpr, tmp_path):
    old = _old_drift(scores, fpr)
    assert old == pytest.approx(REGRESSION[fpr], rel=1e-6)
    assert _train_drift(scores, fpr) == pytest.approx(old, rel=1e-6)
    assert _eval_drift(scores, fpr, tmp_path) == pytest.approx(old, rel=1e-6)


@pytest.mark.parametrize("value", [0.0, 2.5])
def test_constant_score_has_no_drift(value, tmp_path):
    fpr = 0.01
    constant = torch.full((10_000,), value)
    callback = TrainDrift(output_name="s")
    calibration, evaluation = callback._split_scores(constant)
    threshold = callback._compute_threshold(calibration, exceedance_prob=fpr)
    tie_weight = compute_tie_weight(calibration, threshold, fpr)
    p_hat = count_exceedances(evaluation, threshold, tie_weight).item()
    assert p_hat / evaluation.numel() == pytest.approx(fpr, rel=1e-12)

    assert _train_drift(constant, fpr) == pytest.approx(0.0, abs=1e-12)
    assert _eval_drift(constant, fpr, tmp_path) == pytest.approx(0.0, abs=1e-12)
