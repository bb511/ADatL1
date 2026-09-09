import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.evaluation.callbacks import auroc as auroc_module
from src.evaluation.callbacks.auroc import AnomalyAUROCCallback


def test_partial_auroc_interpolates_the_operating_region_endpoint() -> None:
    callback = AnomalyAUROCCallback(
        output_name="ascore/full",
        ds=["signal"],
        max_false_positive_rate=0.25,
    )

    metrics = callback._signal_metrics(
        normal_scores=np.array([0.1, 0.3]),
        signal_scores=np.array([0.2, 0.4]),
    )

    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["partial_auroc_raw"] == pytest.approx(0.125)
    assert metrics["partial_auroc"] == pytest.approx(0.5)


def test_auroc_summary_persists_per_signal_metrics_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = AnomalyAUROCCallback(
        output_name="ascore/full",
        ds=["signal_a", "signal_b"],
        max_false_positive_rate=0.25,
        ckpts={"loss_total": True},
        log_raw_mlflow=False,
    )
    callback._active = True
    callback._normal_score_chunks = [np.array([0.1, 0.2])]
    callback._signal_score_chunks = {
        "signal_a": [np.array([0.8, 0.9])],
        "signal_b": [np.array([0.15, 0.25])],
    }
    callback._plot = Mock()
    monkeypatch.setattr(
        auroc_module.utils.mlflow,
        "log_plots_to_mlflow",
        Mock(),
    )

    checkpoint_path = tmp_path / "loss_total.ckpt"
    trainer = SimpleNamespace(split="val")
    module = SimpleNamespace(_ckpt_path=checkpoint_path)

    callback.on_test_epoch_end(trainer, module)

    summary_path = (
        tmp_path
        / "plots"
        / "val"
        / "loss_total"
        / "auroc"
        / "auroc_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["checkpoint"] == "loss_total.ckpt"
    assert payload["split"] == "val"
    assert payload["score_direction"] == "higher_score_is_more_anomalous"
    assert payload["roc_convention"]["partial_auroc_max_false_positive_rate"] == (
        pytest.approx(0.25)
    )
    assert payload["normal_event_count"] == 2
    assert payload["num_signal_datasets"] == 2
    assert set(payload["per_signal"]) == {"signal_a", "signal_b"}
    assert payload["per_signal"]["signal_a"]["auroc"] == pytest.approx(1.0)
    assert payload["summaries"]["min_auroc"] == pytest.approx(0.75)


def test_auroc_callback_runs_only_for_loss_total() -> None:
    callback = AnomalyAUROCCallback(
        output_name="ascore/full",
        ds=["signal"],
        max_false_positive_rate=0.25,
        ckpts={"loss_total": True},
    )

    assert callback._should_run_for_current_ckpt(
        SimpleNamespace(strat_name="loss_total", metric_name=None, criterion_name=None)
    )
    assert not callback._should_run_for_current_ckpt(
        SimpleNamespace(strat_name="last", metric_name=None, criterion_name=None)
    )


def test_auroc_callback_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="not finite"):
        AnomalyAUROCCallback._scores_to_numpy(
            torch.tensor([0.0, float("nan")]),
            "signal",
        )
