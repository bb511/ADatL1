"""Focused contracts for authenticated persistence and training-curve context."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation import (
    physics_background_pairing_confirmation_context as context_module,
)
from src.evaluation.physics_background_pairing_confirmation_analysis import (
    _content_hash,
)
from src.evaluation.physics_background_pairing_confirmation_context import (
    build_confirmation_context,
)
from src.evaluation.physics_background_pairing_confirmation_report import (
    MODELS,
    SELECTORS,
)


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_context_authenticates_42_cells_and_mlflow_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact 50/200 cells and a sealed 200-point MLflow curve produce context."""
    confirmation = tmp_path / "confirmation"
    discovery = tmp_path / "discovery"
    mlflow = tmp_path / "mlruns"
    confirmation.mkdir()
    discovery.mkdir()
    run_dir = mlflow / "1/run123"
    metric_dir = run_dir / "metrics/val/summary"
    metric_dir.mkdir(parents=True)
    (run_dir / "tags").mkdir()
    (run_dir / "tags/mlflow.runName").write_text("ae_run", encoding="utf-8")
    metric = metric_dir / "cap_native_cdf_ema_sources"
    metric.write_text(
        "".join(f"{1000 + epoch} {epoch / 200} {epoch}\n" for epoch in range(200)),
        encoding="utf-8",
    )

    confirmation_rows = []
    discovery_rows = []
    for model in MODELS:
        for selector, pairing, _ in SELECTORS:
            pairing = pairing or "not_applicable"
            for endpoint in ("efficiency_250hz", "auprc"):
                confirmation_rows.append(
                    {
                        "model": model,
                        "selector": selector,
                        "pairing": pairing,
                        "endpoint": endpoint,
                        "mean": 0.2,
                    }
                )
            discovery_rows.append(
                {
                    "model": model,
                    "score": "native",
                    "metric": selector,
                    "strategy": pairing,
                    "mean_efficiency": 0.1,
                    "mean_auprc": 0.15,
                }
            )
    confirmation_table = confirmation / "endpoints_across_seeds.csv"
    _csv(confirmation_table, confirmation_rows)
    marker = tmp_path / "marker.json"
    marker.write_text(
        json.dumps(
            {
                "mlflow_run_id": "run123",
                "run_name": "ae_run",
                "row": {
                    "model": "ae",
                    "trial_number": 1,
                    "seed": 2,
                    "logical_objectives": ["native__cap__cdf"],
                },
            }
        ),
        encoding="utf-8",
    )
    source_markers = [
        {"marker": str(marker), "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest()}
    ]
    endpoint = confirmation / "endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "source_markers": source_markers,
                "cardinality": {"checkpoint_selections": 1},
            }
        ),
        encoding="utf-8",
    )
    confirmation_manifest = {
        "schema_version": 1,
        "inputs": {
            "endpoint_receipt": str(endpoint),
            "endpoint_receipt_sha256": hashlib.sha256(endpoint.read_bytes()).hexdigest(),
        },
        "artifacts": {
            confirmation_table.name: hashlib.sha256(confirmation_table.read_bytes()).hexdigest()
        },
    }
    confirmation_manifest["content_sha256"] = _content_hash(confirmation_manifest)
    (confirmation / "confirmation_endpoint_analysis.json").write_text(
        json.dumps(confirmation_manifest), encoding="utf-8"
    )

    discovery_table = discovery / "primary_front_summary.csv"
    _csv(discovery_table, discovery_rows)
    discovery_manifest = {
        "artifact_hashes": {
            discovery_table.name: hashlib.sha256(discovery_table.read_bytes()).hexdigest()
        }
    }
    discovery_manifest_path = discovery / "endpoint_analysis.json"
    discovery_manifest_path.write_text(json.dumps(discovery_manifest), encoding="utf-8")
    monkeypatch.setattr(
        context_module,
        "DISCOVERY_ENDPOINT_ANALYSIS_SHA256",
        hashlib.sha256(discovery_manifest_path.read_bytes()).hexdigest(),
    )

    result = build_confirmation_context(confirmation, discovery, mlflow)
    assert len(result["persistence"]) == 84
    assert len(result["hypotheses"]) == 72
    assert len(result["training_curves"]) == 200
    assert result["training_curve_summary"][0]["mean_epoch_200"] == pytest.approx(0.995)
    assert {
        (row["endpoint"], row["mean_50_epoch"], row["mean_200_epoch"])
        for row in result["persistence"]
    } == {
        ("efficiency_250hz", 0.1, 0.2),
        ("auprc", 0.15, 0.2),
    }

    original_marker = marker.read_text()
    marker.write_text(original_marker + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="marker changed"):
        build_confirmation_context(confirmation, discovery, mlflow)
    marker.write_text(original_marker, encoding="utf-8")

    metric.write_text(metric.read_text() + "9999 1.1 200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 200 validation epochs"):
        build_confirmation_context(confirmation, discovery, mlflow)
