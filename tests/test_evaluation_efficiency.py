import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.evaluation.callbacks import efficiency as efficiency_module
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback
from src.evaluation.artifact_provenance import EventManifest


def test_operational_efficiency_summary_is_written_for_named_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callback = AnomalyEfficiencyCallback(
        output_name="ascore/full",
        ds=["signal_a", "signal_b", "signal_c"],
        log_raw_mlflow=False,
        artifact_provenance={
            "protocol_version": "fet-et-pareto-v1",
            "configuration_id": "seed-independent-configuration",
            "autoencoder_seed": 123,
        },
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
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"frozen checkpoint")
    callback._event_manifest = EventManifest("valid")
    for dataset in ("normal", "signal_a", "signal_b", "signal_c"):
        batch = (
            torch.zeros((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            torch.ones(1),
            torch.zeros((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
        )
        callback._event_manifest.update_batch(dataset, batch)
    trainer = SimpleNamespace(
        split="val",
        artifact_provenance_datamodule=SimpleNamespace(
            main_cache_folder=tmp_path / "mlready-cache",
            control_object_feature_map={"FET": {"Et": [0]}},
        ),
    )
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
    assert payload["median_efficiency"] == pytest.approx(0.4)
    assert payload["min_efficiency"] == pytest.approx(0.1)
    assert payload["min_efficiency_dataset"] == "signal_a"
    assert payload["cvar25_efficiency"] == pytest.approx(0.1)
    assert payload["signal_efficiencies"] == {
        "signal_a": pytest.approx(0.1),
        "signal_b": pytest.approx(0.4),
        "signal_c": pytest.approx(0.7),
    }
    assert payload["provenance"]["configuration_id"] == "seed-independent-configuration"
    assert payload["provenance"]["evaluation_mode"] == "validation"
    assert payload["metric_contract"]["efficiency"]["unit"] == "fraction"


def test_efficiency_summary_cvar25_averages_five_worst_of_twenty_signals(
    tmp_path: Path,
) -> None:
    callback = AnomalyEfficiencyCallback(
        output_name="ascore/full",
        ds=[],
        log_raw_mlflow=False,
    )
    callback.base_rate_resolved = 28608.8064
    output_path = tmp_path / "eff_summary.json"
    signal_efficiencies = {
        f"signal_{index:02d}": index / 100 for index in range(1, 21)
    }

    callback._write_efficiency_summary(
        output_path,
        checkpoint_name="loss_total.ckpt",
        split="val",
        target_rate=0.25,
        signal_efficiencies=signal_efficiencies,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["cvar25_efficiency"] == pytest.approx(0.03)


def test_efficiency_summary_uses_the_signal_median_not_the_mean(
    tmp_path: Path,
) -> None:
    callback = AnomalyEfficiencyCallback(
        output_name="ascore/full",
        ds=[],
        log_raw_mlflow=False,
    )
    callback.base_rate_resolved = 28608.8064
    output_path = tmp_path / "eff_summary.json"

    callback._write_efficiency_summary(
        output_path,
        checkpoint_name="loss_total.ckpt",
        split="val",
        target_rate=0.25,
        signal_efficiencies={
            "signal_a": 0.1,
            "signal_b": 0.2,
            "signal_c": 0.9,
        },
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mean_efficiency"] == pytest.approx(0.4)
    assert payload["median_efficiency"] == pytest.approx(0.2)


def test_efficiency_summary_distinguishes_zero_efficiency_from_missing_signals(
    tmp_path: Path,
) -> None:
    callback = AnomalyEfficiencyCallback(
        output_name="ascore/full",
        ds=[],
        log_raw_mlflow=False,
    )
    callback.base_rate_resolved = 28608.8064

    measured_path = tmp_path / "measured" / "eff_summary.json"
    measured_path.parent.mkdir()
    callback._write_efficiency_summary(
        measured_path,
        checkpoint_name="loss_total.ckpt",
        split="val",
        target_rate=0.25,
        signal_efficiencies={"signal_zero": 0.0},
    )

    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    assert measured["num_signal_datasets"] == 1
    assert measured["mean_efficiency"] == pytest.approx(0.0)
    assert measured["median_efficiency"] == pytest.approx(0.0)
    assert measured["min_efficiency"] == pytest.approx(0.0)
    assert measured["min_efficiency_dataset"] == "signal_zero"
    assert measured["cvar25_efficiency"] == pytest.approx(0.0)
    assert measured["signal_efficiencies"] == {"signal_zero": pytest.approx(0.0)}

    missing_path = tmp_path / "missing" / "eff_summary.json"
    missing_path.parent.mkdir()
    callback._write_efficiency_summary(
        missing_path,
        checkpoint_name="loss_total.ckpt",
        split="val",
        target_rate=0.25,
        signal_efficiencies={},
    )

    missing = json.loads(missing_path.read_text(encoding="utf-8"))
    assert missing["num_signal_datasets"] == 0
    assert missing["mean_efficiency"] is None
    assert missing["median_efficiency"] is None
    assert missing["min_efficiency"] is None
    assert missing["min_efficiency_dataset"] is None
    assert missing["cvar25_efficiency"] is None
    assert missing["signal_efficiencies"] == {}
