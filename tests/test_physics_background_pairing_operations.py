"""Tests for the frozen-campaign operational control plane."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import physics_background_pairing_operations as operations
from scripts import physics_background_pairing_report as report


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
    retrain = repository / "scripts" / "clariden" / "physics_background_pairing_retrain.sbatch"
    retrain.write_text("#!/bin/sh\n", encoding="utf-8")
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
    monkeypatch.setattr(operations, "_require_canary_gate", lambda *args: None)
    monkeypatch.setattr(operations, "_usable_search_trials", lambda *args: 4)
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


def test_submit_requires_terminal_canary_gate(monkeypatch, tmp_path) -> None:
    """A quiescent individual model cannot start before all six canaries pass."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)

    with pytest.raises(RuntimeError, match="terminal validation gate"):
        operations.submit_search(
            root,
            repository,
            "ae",
            4,
            4,
            720,
            tmp_path / "data",
            tmp_path / "raw",
            tmp_path / "pairing",
        )


def test_terminal_canary_gate_authenticates_exact_campaign_cardinalities(tmp_path) -> None:
    """The search gate accepts only the complete six-model validator contract."""
    root, _ = _campaign(tmp_path)
    design = operations._design(root)
    payload = {
        "schema_version": 1,
        "code_commit": "a" * 40,
        "bindings": 24,
        "checkpoint_selections": 224,
        "mlflow_runs": [{} for _ in range(24)],
        "candidate_checkpoints": [{} for _ in range(272)],
        "candidate_availability": {
            "configured_slots": 272,
            "available": 272,
            "required_available": 248,
            "stable_configured": 24,
            "stable_available": 24,
            "stable_missing": 0,
        },
        "jobs": [
            {
                "model": model,
                "state": "COMPLETED",
                "exit_code": "0:0",
                "commit": "a" * 40,
                "log_sha256": "b" * 64,
            }
            for model in operations.MODELS
        ],
        "studies": [
            {
                "model": model,
                "total": 4,
                "complete": 4,
                "usable_complete": 4,
                "running": 0,
                "waiting": 0,
                "fail": 0,
                "pruned": 0,
            }
            for model in operations.MODELS
        ],
    }
    gate = root / "state" / "production_canary_validation.json"
    gate.parent.mkdir()
    gate.write_text(json.dumps(payload), encoding="utf-8")

    operations._require_canary_gate(root, design)
    payload["bindings"] = 23
    gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete or inconsistent"):
        operations._require_canary_gate(root, design)


def test_submit_refuses_trial_target_overshoot(monkeypatch, tmp_path) -> None:
    """The final chunk cannot create more than 600 potentially usable trials."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    monkeypatch.setattr(operations, "_require_canary_gate", lambda *args: None)
    monkeypatch.setattr(operations, "_usable_search_trials", lambda *args: 580)

    with pytest.raises(RuntimeError, match="at most 20 remain"):
        operations.submit_search(
            root,
            repository,
            "vae",
            48,
            48,
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


def test_accounting_maps_array_allocation_and_task(monkeypatch) -> None:
    """Array accounting maps Slurm's numeric allocation back to the ledger task ID."""
    monkeypatch.setattr(operations.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(
                {
                    "jobs": [
                        {
                            "job_id": 101,
                            "array": {
                                "job_id": 100,
                                "task_id": {"set": True, "number": 7},
                            },
                            "state": {"current": ["RUNNING"]},
                            "time": {"elapsed": 4, "start": 1, "end": 0},
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

    observed = operations._accounting(["100_7"])

    assert set(observed) == {"100_7"}
    assert observed["100_7"]["state"] == "RUNNING"


def test_submit_retrains_uses_exact_missing_indices(monkeypatch, tmp_path) -> None:
    """Retraining submits a compressed missing-index array and records every task."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    selection = root / "selection"
    selection.mkdir()
    rows = [{"retrain_index": index, "model": "ae", "trial_number": index} for index in range(4)]
    (selection / "pareto_retrain_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "code_commit": "a" * 40,
                "retrain_runs": rows,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        operations, "_validate_retrain_marker", lambda path, row: row["retrain_index"] in {0, 2}
    )
    calls = []
    monkeypatch.setattr(operations.shutil, "which", lambda name: f"/bin/{name}")

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="987\n")

    monkeypatch.setattr(operations.subprocess, "run", run)
    job_id = operations.submit_retrains(
        root,
        repository,
        24,
        tmp_path / "data",
        tmp_path / "raw",
        tmp_path / "pairing",
    )

    assert job_id == "987"
    assert "--array=1,3%2" in calls[0]
    with (root / "state" / "job_history.csv").open(encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    assert [row["job_id"] for row in ledger] == ["987_1", "987_3"]


def test_missing_retrains_rejects_changed_marker_artifact(monkeypatch, tmp_path) -> None:
    """A present marker is not complete when any authenticated artifact changed."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    row = {"retrain_index": 0, "model": "vae", "trial_number": 3}
    selection = root / "selection"
    selection.mkdir()
    (selection / "pareto_retrain_manifest.json").write_text(
        json.dumps({"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": [row]}),
        encoding="utf-8",
    )
    artifact = tmp_path / "optimized.json"
    artifact.write_text("before\n", encoding="utf-8")
    values = tmp_path / "values.csv"
    values.write_text("value\n1\n", encoding="utf-8")
    marker = root / "retraining" / "markers" / "0000.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "row": row,
                "optimized_metric_artifact": str(artifact),
                "optimized_metric_artifact_sha256": "0" * 64,
                "value_files": [{"path": str(values), "sha256": operations._sha256(values)}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing or changed"):
        operations.missing_retrains(root, repository)


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


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        (
            {"score": "native", "metric": "cap", "strategy": "jetclr"},
            "summary/cap_native_jetclr_ema_ZB_run396102_vs_ZB_run398183/max",
        ),
        (
            {"score": "residual_oas", "metric": "wasserstein", "strategy": None},
            "summary/w1dist_residual_oas_ema_ZB_run396102_vs_ZB_run398183/min",
        ),
        (
            {"score": "native", "metric": "drift", "strategy": None},
            "summary/native_operational_drift_ema/min",
        ),
    ],
)
def test_primary_checkpoint_family_is_exact(objective, expected) -> None:
    """Every objective maps to its only allowed primary checkpoint family."""
    assert operations._primary_checkpoint_family(objective) == expected


def test_candidate_checkpoint_requires_independent_oas_threshold(tmp_path) -> None:
    """AE/VAE candidates require finite independent native and OAS thresholds."""
    checkpoint = tmp_path / "candidate.ckpt"
    torch.save({"state_dict": {"thres_operational": torch.tensor(1.0)}}, checkpoint)
    with pytest.raises(RuntimeError, match="threshold keys differ"):
        operations._candidate_checkpoint(checkpoint, "ae")

    torch.save(
        {
            "state_dict": {
                "thres_operational": torch.tensor(1.0),
                "thres_operational__residual_oas": torch.tensor(2.0),
            }
        },
        checkpoint,
    )
    audited = operations._candidate_checkpoint(checkpoint, "ae")

    assert audited["thresholds"] == {
        "thres_operational": 1.0,
        "thres_operational__residual_oas": 2.0,
    }


def test_validate_canaries_covers_all_trials_and_candidate_families(monkeypatch, tmp_path) -> None:
    """The terminal gate audits all candidate slots and tolerates absent stable checkpoints."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    scores = {
        "ae": ("native", "residual_oas"),
        "vae": ("native", "residual_oas"),
        "dsae": ("native",),
        "dsvae": ("native",),
        "svdd": ("native",),
        "realnvp": ("native",),
    }
    studies = []
    for model in operations.MODELS:
        objectives = []
        for score in scores[model]:
            objectives.extend(
                {
                    "id": f"{score}__cap__{strategy}",
                    "score": score,
                    "metric": "cap",
                    "strategy": strategy,
                    "directions": ["maximize", "minimize"],
                }
                for strategy in (
                    "flat_physical",
                    "physics_summary",
                    "typed_sliced_wasserstein",
                    "jetclr",
                    "cdf",
                )
            )
            objectives.extend(
                [
                    {
                        "id": f"{score}__wasserstein",
                        "score": score,
                        "metric": "wasserstein",
                        "strategy": None,
                        "directions": ["minimize", "minimize"],
                    },
                    {
                        "id": f"{score}__drift",
                        "score": score,
                        "metric": "drift",
                        "strategy": None,
                        "directions": ["minimize", "minimize"],
                    },
                ]
            )
        studies.append({"model": model, "objectives": objectives})
    design = {"schema_version": 2, "code_commit": "a" * 40, "studies": studies}
    bindings = {}
    for study in studies:
        model = study["model"]
        for index in range(4):
            hydra = root / "sweeps" / model / "stamp" / str(index)
            (hydra / ".hydra").mkdir(parents=True)
            (hydra / ".hydra" / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "test": False,
                        "data": {
                            "validation_aux_datasets": [],
                            "expose_zerobias_sources": True,
                        },
                        "paths": {
                            "zerobias": {
                                "ZB_run396102": "e",
                                "ZB_run398183": "g",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_root = root / "checkpoints" / model / str(index)
            checkpoint_root.mkdir(parents=True)
            (checkpoint_root / "last.ckpt").write_text("last", encoding="utf-8")
            families = ["single/ascore_operational/stable"] + [
                operations._primary_checkpoint_family(objective)
                for objective in study["objectives"]
            ]
            for family in families:
                directory = checkpoint_root / family
                directory.mkdir(parents=True)
                if not (
                    model == "ae" and index == 0 and family == "single/ascore_operational/stable"
                ):
                    (directory / "candidate.ckpt").write_text("candidate", encoding="utf-8")
            artifact = hydra / "optimized_metric.json"
            artifact.write_text(
                json.dumps({"run_provenance": {"checkpoint_run_dir": str(checkpoint_root)}}),
                encoding="utf-8",
            )
            bindings[(model, index)] = {
                "hydra_run_dir": str(hydra),
                "optimized_metric_artifact": str(artifact),
            }

    jobs = [
        {
            "kind": "production_canary",
            "model": model,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "commit": "a" * 40,
            "log_sha256": "1" * 64,
        }
        for model in operations.MODELS
    ]
    status = [
        {
            "model": model,
            "total": 4,
            "complete": 4,
            "usable_complete": 4,
            "running": 0,
            "waiting": 0,
            "fail": 0,
            "pruned": 0,
        }
        for model in operations.MODELS
    ]
    monkeypatch.setattr(operations, "reconcile", lambda *args: jobs)
    monkeypatch.setattr(report, "_design", lambda root: design)
    monkeypatch.setattr(report, "_trial_exports", lambda *args: (status, []))
    monkeypatch.setattr(report, "_binding_index", lambda root: bindings)
    monkeypatch.setattr(report, "_validate_bindings", lambda *args: None)
    monkeypatch.setattr(
        report, "_checkpoint_rows", lambda *args, **kwargs: ([{}] * 224, [{}] * 24)
    )
    monkeypatch.setattr(report, "_annotate_checkpoint_families", lambda *args: [{}] * 224)
    monkeypatch.setattr(report, "_validate_mlflow_runs", lambda *args, **kwargs: [{}] * 24)
    monkeypatch.setattr(
        operations,
        "_candidate_checkpoint",
        lambda path, model: {"path": str(path), "sha256": "2" * 64, "thresholds": {}},
    )

    output = operations.validate_canaries(root, repository)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["bindings"] == 24
    assert payload["checkpoint_selections"] == 224
    assert len(payload["candidate_checkpoints"]) == 272
    assert payload["candidate_availability"] == {
        "configured_slots": 272,
        "available": 271,
        "required_available": 248,
        "stable_configured": 24,
        "stable_available": 23,
        "stable_missing": 1,
    }
