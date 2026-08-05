"""Tests for the frozen-campaign operational control plane."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import physics_background_pairing_operations as operations


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    """Create one minimal frozen campaign and execution tree."""
    root = tmp_path / "campaign"
    repository = tmp_path / "execution"
    root.mkdir()
    repository.mkdir()
    (root / "design.json").write_text(
        json.dumps({"schema_version": 2, "code_commit": "a" * 40}), encoding="utf-8"
    )
    script = repository / "scripts" / "clariden" / "physics_background_pairing_sweep.sbatch"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    for name in ("data", "raw", "pairing"):
        (tmp_path / name).mkdir()
    return root, repository


def _authenticate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace Git authentication for isolated filesystem tests."""
    monkeypatch.setattr(operations, "_authenticate_execution_repository", lambda *args: None)


def test_submit_registers_exact_frozen_job(monkeypatch, tmp_path) -> None:
    """Submission exports frozen roots and atomically records the Slurm ID."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345;cluster\n")

    monkeypatch.setattr(operations.subprocess, "run", run)
    job_id = operations.submit_search(
        root,
        repository,
        "ae",
        48,
        48,
        720,
        tmp_path / "data",
        tmp_path / "raw",
        tmp_path / "pairing",
    )

    assert job_id == "12345"
    assert "--nodes=12" in calls[0]
    assert any("PHYSICS_BGPAIR_REPOSITORY_ROOT=" in value for value in calls[0])
    with (root / "state" / "job_history.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["state"] == "PENDING"
    assert row["commit"] == "a" * 40


def test_submit_rejects_active_same_model(monkeypatch, tmp_path) -> None:
    """A second chunk cannot overlap an active writer for the same study."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    path = root / "state" / "job_history.csv"
    operations._atomic_csv(
        path,
        [{"job_id": "1", "kind": "search_chunk", "model": "vae", "state": "RUNNING"}],
    )
    with pytest.raises(RuntimeError, match="overlap"):
        operations.submit_search(
            root,
            repository,
            "vae",
            4,
            4,
            720,
            tmp_path / "data",
            tmp_path / "raw",
            tmp_path / "pairing",
        )


def test_reconcile_hashes_only_terminal_logs(monkeypatch, tmp_path) -> None:
    """Only immutable terminal logs receive an authentication digest."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    terminal_log = tmp_path / "terminal.out"
    terminal_log.write_text("done\n", encoding="utf-8")
    running_log = tmp_path / "running.out"
    running_log.write_text("live\n", encoding="utf-8")
    operations._atomic_csv(
        root / "state" / "job_history.csv",
        [
            {"job_id": "1", "model": "ae", "state": "RUNNING", "log_path": str(terminal_log)},
            {"job_id": "2", "model": "vae", "state": "PENDING", "log_path": str(running_log)},
        ],
    )
    monkeypatch.setattr(
        operations,
        "_accounting",
        lambda ids: {
            "1": {"state": "COMPLETED", "elapsed": "01:00:00", "exit_code": "0:0"},
            "2": {"state": "RUNNING", "elapsed": "00:10:00", "exit_code": "0:0"},
        },
    )

    rows = operations.reconcile(root, repository)

    assert rows[0]["log_sha256"] == operations._sha256(terminal_log)
    assert rows[1]["log_sha256"] == ""


def test_accounting_uses_unambiguous_json_timestamps(monkeypatch) -> None:
    """Slurm epoch fields become stable UTC timestamps instead of local clock times."""
    monkeypatch.setattr(operations.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(
                {
                    "jobs": [
                        {
                            "job_id": 7,
                            "state": {"current": ["COMPLETED"]},
                            "time": {"elapsed": 3661, "start": 1, "end": 2},
                            "exit_code": {
                                "return_code": {"number": 0},
                                "signal": {"id": {"number": 0}},
                            },
                        }
                    ]
                }
            )
        ),
    )

    row = operations._accounting(["7"])["7"]

    assert row["elapsed"] == "01:01:01"
    assert row["started"] == "1970-01-01T00:00:01+00:00"
    assert row["ended"] == "1970-01-01T00:00:02+00:00"


def test_consistent_backup_refuses_live_trial_and_hashes_copy(monkeypatch, tmp_path) -> None:
    """SQLite backup is quiescent, consistent, and accompanied by its digest."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    database = root / "studies" / "svdd.db"
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE trials (state TEXT)")
        connection.execute("INSERT INTO trials VALUES ('RUNNING')")
    with pytest.raises(RuntimeError, match="not quiescent"):
        operations.backup_study(root, repository, "svdd")

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE trials SET state='COMPLETE'")
    backup = operations.backup_study(root, repository, "svdd")

    assert backup.is_file()
    assert (
        backup.with_suffix(".sha256")
        .read_text(encoding="utf-8")
        .startswith(operations._sha256(backup))
    )
