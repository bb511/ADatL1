import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.evaluation.callbacks import efficiency as efficiency_module
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback


def test_operational_efficiency_summary_is_written_for_named_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callback = AnomalyEfficiencyCallback(
        output_name="ascore/full",
        ds=["signal_a", "signal_b", "signal_c"],
        log_raw_mlflow=False,
    )
    callback.target_rates_resolved = [0.25]
    callback.operational_rate = 0.25
    callback.base_rate_resolved = 28608.8064
    callback.main_rate = object()
    callback.sig_rates = object()
    callback.bkg_rates = object()
    callback._compute_eff = Mock(
        side_effect=[
            {"normal": 0.25 / 28608.8064},
            {"signal_a": 0.1, "signal_b": 0.4, "signal_c": 0.7},
            {},
        ]
    )
    callback._plot = Mock()
    monkeypatch.setattr(
        efficiency_module.utils.mlflow,
        "log_plots_to_mlflow",
        Mock(),
    )

    checkpoint_path = (
        tmp_path / "checkpoints" / "physics_ae_models" / "run_1" / "loss_total.ckpt"
    )
    trainer = SimpleNamespace(split="val")
    module = SimpleNamespace(_ckpt_path=checkpoint_path)

    callback.on_test_epoch_end(trainer, module)

    summary_path = (
        checkpoint_path.parent
        / "plots"
        / "val"
        / "loss_total"
        / "eff"
        / "eff_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["checkpoint"] == "loss_total.ckpt"
    assert payload["split"] == "val"
    assert payload["anomaly_score"] == "ascore/full"
    assert payload["operating_point"] == {
        "label": "operational",
        "target_rate": pytest.approx(0.25),
        "base_rate": pytest.approx(28608.8064),
    }
    assert payload["num_signal_datasets"] == 3
    assert payload["mean_efficiency"] == pytest.approx(0.4)
    assert payload["min_efficiency"] == pytest.approx(0.1)
    assert payload["min_efficiency_dataset"] == "signal_a"
    assert payload["signal_efficiencies"] == {
        "signal_a": pytest.approx(0.1),
        "signal_b": pytest.approx(0.4),
        "signal_c": pytest.approx(0.7),
    }
