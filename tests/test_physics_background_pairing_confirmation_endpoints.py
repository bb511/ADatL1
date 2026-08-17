"""Contracts for extracting sealed confirmation efficiency and AUPRC values."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import physics_background_pairing_campaign as campaign
from scripts import physics_background_pairing_confirmation_endpoints as endpoints


def _values_file(checkpoint: Path, folder: str, metric: str, value: float) -> Path:
    """Write one synthetic exact 20-anomaly callback values file."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    path = checkpoint.parent / "plots/test" / checkpoint.stem / folder / "values.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("checkpoint", "intervention", "metric", "value")
        )
        writer.writeheader()
        writer.writerows(
            {
                "checkpoint": checkpoint.stem,
                "intervention": anomaly,
                "metric": metric,
                "value": value,
            }
            for anomaly in sorted(campaign.DOWNSTREAM_DATASET_NAMES)
        )
    return path


def test_endpoint_reader_authenticates_efficiency_and_auprc(tmp_path: Path) -> None:
    """Both endpoint callbacks must cover the same exact 20-anomaly panel."""
    checkpoint = tmp_path / "summary/metric_a/max/epoch=199.ckpt"
    efficiency = _values_file(checkpoint, "eff_native", "efficiency_operational", 0.25)
    auprc = _values_file(checkpoint, "auprc_native", "auprc", 0.75)
    marker = {
        "value_files": [
            {"path": str(path), "sha256": endpoints._sha256(path)} for path in (efficiency, auprc)
        ]
    }
    assert set(
        endpoints._values_for_checkpoint(
            marker,
            checkpoint=checkpoint,
            score="native",
            endpoint="efficiency_250hz",
        ).values()
    ) == {0.25}
    assert set(
        endpoints._values_for_checkpoint(
            marker, checkpoint=checkpoint, score="native", endpoint="auprc"
        ).values()
    ) == {0.75}


def test_endpoint_reader_rejects_hash_or_coverage_drift(tmp_path: Path) -> None:
    """A changed file or missing anomaly cannot enter the final analysis table."""
    checkpoint = tmp_path / "last.ckpt"
    path = _values_file(checkpoint, "auprc_native", "auprc", 0.75)
    marker = {"value_files": [{"path": str(path), "sha256": endpoints._sha256(path)}]}
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Endpoint artifact changed"):
        endpoints._values_for_checkpoint(
            marker, checkpoint=checkpoint, score="native", endpoint="auprc"
        )

    checkpoint = tmp_path / "second/single/ascore_operational/stable/epoch=050.ckpt"
    path = _values_file(checkpoint, "auprc_native", "auprc", 0.75)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))[:-1]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    marker = {"value_files": [{"path": str(path), "sha256": endpoints._sha256(path)}]}
    with pytest.raises(RuntimeError, match="Expected the exact 20-dataset"):
        endpoints._values_for_checkpoint(
            marker, checkpoint=checkpoint, score="native", endpoint="auprc"
        )


def test_same_stem_in_two_summary_families_is_bound_by_exact_path(tmp_path: Path) -> None:
    """Two family checkpoints with the same stem cannot collide during extraction."""
    first_checkpoint = tmp_path / "summary/metric_a/max/epoch=199.ckpt"
    second_checkpoint = tmp_path / "summary/metric_b/max/epoch=199.ckpt"
    first = _values_file(first_checkpoint, "auprc_native", "auprc", 0.2)
    second = _values_file(second_checkpoint, "auprc_native", "auprc", 0.8)
    marker = {
        "value_files": [
            {"path": str(path), "sha256": endpoints._sha256(path)} for path in (first, second)
        ]
    }
    assert set(
        endpoints._values_for_checkpoint(
            marker, checkpoint=first_checkpoint, score="native", endpoint="auprc"
        ).values()
    ) == {0.2}
    assert set(
        endpoints._values_for_checkpoint(
            marker, checkpoint=second_checkpoint, score="native", endpoint="auprc"
        ).values()
    ) == {0.8}
