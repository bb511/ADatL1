import csv
from pathlib import Path

import pytest

from scripts.physics_background_pairing_search50_evaluation import (
    EXPECTED_ANOMALIES,
    _canonical_hash,
    _marker_valid,
    _stage_task,
    _values_rows,
)


def _task(source: Path, source_sha256: str, *, oas: bool = True) -> dict:
    return {
        "task_index": 0,
        "model": "ae" if oas else "svdd",
        "trial_number": 7,
        "params": {},
        "scores": ["native", "residual_oas"] if oas else ["native"],
        "checkpoints": [
            {
                "checkpoint_id": "selected_000",
                "checkpoint_family": "last",
                "optimized_ckpt_name": "last",
                "source_path": str(source),
                "source_sha256": source_sha256,
                "logical_objectives": ["native__drift"],
            }
        ],
        "memberships": [],
    }


def _write_values(root: Path, score: str) -> Path:
    callback = "eff_native" if score == "native" else "eff_residual_oas"
    path = (
        root
        / "checkpoint_store/x/y/candidate/checkpoint/selected/plots/test"
        / "ds=selected_000__selected"
        / callback
        / "values.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["checkpoint", "intervention", "metric", "value"]
        )
        writer.writeheader()
        for index in range(EXPECTED_ANOMALIES):
            writer.writerow(
                {
                    "checkpoint": "ds=selected_000__selected",
                    "intervention": f"signal_{index:02d}",
                    "metric": "efficiency_operational",
                    "value": index / EXPECTED_ANOMALIES,
                }
            )
    return path


def test_stage_task_creates_only_authenticated_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"checkpoint")
    import hashlib

    task = _task(source, hashlib.sha256(source.read_bytes()).hexdigest(), oas=False)
    task_dir, experiment, run_name = _stage_task(tmp_path, {}, task)
    staged = (
        task_dir
        / "checkpoint_store"
        / experiment
        / run_name
        / "candidate/checkpoint/selected/ds=selected_000__selected.ckpt"
    )
    assert staged.is_symlink()
    assert staged.resolve() == source.resolve()
    _stage_task(tmp_path, {}, task)  # idempotent


def test_values_rows_requires_exact_native_and_oas_coverage(tmp_path: Path) -> None:
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"x")
    task = _task(source, "unused")
    _write_values(tmp_path, "native")
    with pytest.raises(RuntimeError, match="coverage differs"):
        _values_rows(tmp_path, task)
    _write_values(tmp_path, "residual_oas")
    values, files = _values_rows(tmp_path, task)
    assert len(values) == len(files) == 2
    assert all(len(value["rows"]) == EXPECTED_ANOMALIES for value in values)


def test_marker_reauthenticates_exact_value_file_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"x")
    task = _task(source, "unused", oas=False)
    value_path = _write_values(tmp_path, "native")
    values, files = _values_rows(tmp_path, task)
    manifest = {"content_sha256": "manifest"}
    marker = {
        "manifest_content_sha256": "manifest",
        "task_sha256": _canonical_hash(task),
        "attempt_dir": str(tmp_path),
        "values": values,
        "value_files": files,
    }
    assert _marker_valid(tmp_path, manifest, task, marker)
    value_path.write_text(value_path.read_text().replace("0.95", "0.96"), encoding="utf-8")
    assert not _marker_valid(tmp_path, manifest, task, marker)
