"""Focused contracts for isolated search50 AUPRC evaluation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.physics_background_pairing_search50_auprc import (
    _canonical_hash,
    _marker_valid,
    _stage,
    _value_rows,
)


def _task(source: Path, *, oas: bool = True) -> dict:
    """Build a minimal frozen evaluation task."""
    return {
        "task_index": 0,
        "model": "ae" if oas else "svdd",
        "trial_number": 3,
        "params": {},
        "scores": ["native", "residual_oas"] if oas else ["native"],
        "checkpoints": [
            {
                "checkpoint_id": "selected_000",
                "source_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
        "memberships": [],
    }


def _write_values(task_dir: Path, callback: str) -> Path:
    """Write exact synthetic 20-anomaly AUPRC callback output."""
    path = (
        task_dir
        / "checkpoint_store/a/b/c/plots/test/ds=selected_000__selected"
        / callback
        / "values.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["checkpoint", "intervention", "metric", "value"]
        )
        writer.writeheader()
        for index in range(20):
            writer.writerow(
                {
                    "checkpoint": "ds=selected_000__selected",
                    "intervention": f"signal_{index:02d}",
                    "metric": "auprc",
                    "value": index / 20,
                }
            )
    return path


def test_stage_is_confined_to_auprc_workspace_and_hash_checks(tmp_path: Path) -> None:
    """Staging remains isolated and rejects changed checkpoint bytes."""
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"checkpoint")
    task_dir, experiment, run_name = _stage(tmp_path, _task(source, oas=False))
    assert task_dir == tmp_path / "auprc_evaluation/tasks/000"
    staged = (
        task_dir
        / "checkpoint_store"
        / experiment
        / run_name
        / "candidate/checkpoint/selected/ds=selected_000__selected.ckpt"
    )
    assert staged.is_symlink() and staged.resolve() == source.resolve()
    source.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen checkpoint changed"):
        _stage(
            tmp_path,
            _task(source, oas=False)
            | {
                "checkpoints": [
                    {**_task(source, oas=False)["checkpoints"][0], "source_sha256": "0" * 64}
                ]
            },
        )


def test_auprc_values_require_exact_native_and_oas_coverage(tmp_path: Path) -> None:
    """AE/VAE tasks require both native and OAS outputs over all anomalies."""
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"x")
    task = _task(source)
    _write_values(tmp_path, "auprc_native")
    with pytest.raises(RuntimeError, match="coverage differs"):
        _value_rows(tmp_path, task)
    _write_values(tmp_path, "auprc_residual_oas")
    values, files = _value_rows(tmp_path, task)
    assert len(values) == len(files) == 2
    assert all(len(value["rows"]) == 20 for value in values)


def test_marker_reauthenticates_auprc_file_hashes(tmp_path: Path) -> None:
    """Changing a persisted AUPRC value invalidates its task marker."""
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"x")
    task = _task(source, oas=False)
    values_path = _write_values(tmp_path, "auprc_native")
    values, files = _value_rows(tmp_path, task)
    manifest = {"content_sha256": "manifest"}
    marker = {
        "manifest_content_sha256": "manifest",
        "task_sha256": _canonical_hash(task),
        "task_dir": str(tmp_path),
        "values": values,
        "value_files": files,
    }
    assert _marker_valid(manifest, task, marker)
    values_path.write_text(values_path.read_text().replace("0.95", "0.96"), encoding="utf-8")
    assert not _marker_valid(manifest, task, marker)


def test_retrain_suites_define_native_and_oas_auprc_callbacks() -> None:
    """Retrain suites expose score-specific AUPRC callback names and outputs."""
    root = Path(__file__).parents[1] / "configs/retrain_suite"
    native = OmegaConf.load(root / "physics_background_native.yaml")
    oas = OmegaConf.load(root / "physics_background_native_oas.yaml")
    native_cb = native.evaluation.callbacks.anomaly_auprc
    oas_cb = oas.evaluation.callbacks.anomaly_auprc_oas
    assert native_cb.output_name == "ascore/full" and native_cb.name == "auprc_native"
    assert oas_cb.output_name == "ascore/residual_oas" and oas_cb.name == "auprc_residual_oas"
    assert len(native_cb.ds) == len(oas_cb.ds) == 20
