"""Tests for the frozen-campaign operational control plane."""

from __future__ import annotations

import csv
import json
import random
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import physics_background_pairing_operations as operations
from scripts import physics_background_pairing_report as report


def _write_exact_checkpoint(path: Path, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "callbacks": {
                "physics-background-pairing-exact-retrain-rng-v1": {
                    "schema_version": 1,
                    "python": random.getstate(),
                    "numpy": {
                        "algorithm": "MT19937",
                        "keys": torch.zeros(624, dtype=torch.uint32),
                        "position": 0,
                        "has_gauss": 0,
                        "cached_gaussian": 0.0,
                    },
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": [torch.get_rng_state()],
                    "datamodule_shuffler": torch.Generator().manual_seed(123).get_state(),
                }
            },
        },
        path,
    )


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
    monkeypatch.setattr(operations, "_live_search_job_ids", lambda: set())
    monkeypatch.setattr(operations, "_live_retrain_job_ids", lambda: set())


def _write_retrain_progress(root: Path, row: dict, job_ids: list[str]) -> Path:
    """Write mutable campaign state with immutable artifacts through each boundary."""
    index = int(row["retrain_index"])
    manifest = root / "selection" / "pareto_retrain_manifest.json"
    state_dir = root / "retraining" / "restart_state" / f"{index:04d}"
    segments_dir = state_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    attempts = []
    for attempt_number, job_id in enumerate(job_ids, start=1):
        boundary = attempt_number * 40
        checkpoint = segments_dir / f"epoch_{boundary:03d}.ckpt"
        config = segments_dir / f"epoch_{boundary:03d}_config.yaml"
        _write_exact_checkpoint(checkpoint, epoch=boundary - 1)
        config.write_text(f"boundary: {boundary}\n", encoding="utf-8")
        array_job_id, task_id = job_id.split("_")
        attempts.append(
            {
                "attempt_number": attempt_number,
                "phase": "training",
                "target_epoch_boundary": boundary,
                "slurm_array_job_id": array_job_id,
                "slurm_array_task_id": task_id,
                "completed_at": f"done-{boundary}",
            }
        )
        segments.append(
            {
                "epoch_boundary": boundary,
                "attempt_number": attempt_number,
                "continuation_checkpoint": str(checkpoint.resolve()),
                "continuation_checkpoint_sha256": operations._sha256(checkpoint),
                "resolved_config": str(config.resolve()),
                "resolved_config_sha256": operations._sha256(config),
            }
        )
    identity = {
        "code_commit": "a" * 40,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": operations._sha256(manifest),
        "retrain_index": index,
        "row": row,
        "row_sha256": operations._json_sha256(row),
    }
    state = state_dir / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "attempts": attempts,
                "segments": segments,
            }
        ),
        encoding="utf-8",
    )
    return state


def _freeze_design() -> dict:
    """Return the smallest target-one design with the production 56-front shape."""
    counts = {"ae": 14, "vae": 14, "dsae": 7, "dsvae": 7, "svdd": 7, "realnvp": 7}
    return {
        "schema_version": 2,
        "code_commit": "a" * 40,
        "target_trials_per_study": 1,
        "retrain_epochs": 200,
        "studies": [
            {
                "model": model,
                "study_name": f"study_{model}",
                "objectives": [
                    {
                        "id": f"objective_{index}",
                        "score": "native",
                        "metric": "wasserstein",
                        "strategy": None,
                        "directions": ["minimize", "minimize"],
                        "value_indices": [2 * index, 2 * index + 1],
                    }
                    for index in range(count)
                ],
            }
            for model, count in counts.items()
        ],
    }


def _freeze_manifest(design: dict) -> dict:
    """Build one nonempty membership per logical front and one retrain per model."""
    retrains = []
    memberships = []
    for retrain_index, study in enumerate(design["studies"]):
        objective_ids = [row["id"] for row in study["objectives"]]
        retrains.append(
            {
                "model": study["model"],
                "trial_number": 0,
                "params": {},
                "logical_objectives": objective_ids,
                "seed": 123,
                "epochs": 200,
                "retrain_index": retrain_index,
            }
        )
        for objective in study["objectives"]:
            memberships.append(
                {
                    "membership_index": len(memberships),
                    "model": study["model"],
                    "logical_objective": objective["id"],
                    "score": objective["score"],
                    "metric": objective["metric"],
                    "strategy": objective["strategy"],
                    "trial_number": 0,
                    "objective_values": [0.1, 0.2],
                    "retrain_index": retrain_index,
                }
            )
    return {
        "schema_version": 2,
        "code_commit": design["code_commit"],
        "selection_uses_downstream_anomalies": False,
        "front_memberships": memberships,
        "retrain_runs": retrains,
    }


def _oracle(manifest: dict) -> dict:
    """Build a valid aggregate result from a minimal frozen manifest."""
    retrains = {row["retrain_index"]: row for row in manifest["retrain_runs"]}
    memberships = []
    for row in manifest["front_memberships"]:
        values = {f"anomaly_{index:02d}": 0.5 for index in range(20)}
        memberships.append(
            {
                **row,
                "params": retrains[row["retrain_index"]]["params"],
                "optimized_ckpt_name": "last",
                "retrained_objective_values": [0.1, 0.2],
                "mean_downstream_efficiency": 0.5,
                "downstream_efficiencies": values,
            }
        )
    return {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "oracle_uses_downstream_anomalies": True,
        "winners": memberships.copy(),
        "all_front_memberships": memberships,
    }


def _aggregate_campaign_files(root: Path, design: dict, manifest: dict) -> None:
    """Persist the frozen inputs consumed by the aggregation transition."""
    (root / "design.json").write_text(json.dumps(design), encoding="utf-8")
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    freeze_path = root / "state" / "freeze_validation.json"
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code_commit": design["code_commit"],
                "logical_fronts": 56,
                "front_memberships": len(manifest["front_memberships"]),
                "unique_retrains": len(manifest["retrain_runs"]),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": operations._sha256(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    marker_root = root / "retraining" / "markers"
    marker_root.mkdir(parents=True)
    for row in manifest["retrain_runs"]:
        (marker_root / f"{row['retrain_index']:04d}.json").write_text(
            json.dumps({"retrain_index": row["retrain_index"]}), encoding="utf-8"
        )


def test_aggregate_gate_seals_exact_quiescent_oracle(monkeypatch, tmp_path) -> None:
    """Aggregation validates all retrains, 56 fronts, memberships, and hashes."""
    root, repository = _campaign(tmp_path)
    design = _freeze_design()
    manifest = _freeze_manifest(design)
    oracle = _oracle(manifest)
    _aggregate_campaign_files(root, design, manifest)
    runner = repository / "scripts" / "physics_background_pairing_campaign.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    _authenticate(monkeypatch)
    monkeypatch.setattr(operations, "_reconcile_locked", lambda *args: [])
    monkeypatch.setattr(operations, "_validate_frozen_manifest_contract", lambda *args: None)
    monkeypatch.setattr(operations, "_validate_retrain_marker", lambda *args: True)

    def run(command, **kwargs):
        oracle_path = root / "results" / "oracle_winners.json"
        oracle_path.parent.mkdir(parents=True)
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(operations.subprocess, "run", run)
    output = operations.aggregate_campaign(root, repository)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["logical_fronts"] == 56
    assert payload["front_memberships"] == 56
    assert payload["unique_retrains"] == 6
    assert payload["oracle_winners"] == 56
    assert payload["oracle_memberships"] == 56
    assert len(payload["retrain_markers"]) == 6
    assert payload["manifest_sha256"] == operations._sha256(
        root / "selection" / "pareto_retrain_manifest.json"
    )
    assert payload["oracle_sha256"] == operations._sha256(root / "results" / "oracle_winners.json")
    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("idempotent aggregation reran the campaign runner"),
    )
    assert operations.aggregate_campaign(root, repository) == output


def test_aggregate_gate_rejects_active_retrain(monkeypatch, tmp_path) -> None:
    """A ledger-active retrain prevents the terminal aggregation transition."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    monkeypatch.setattr(
        operations,
        "_reconcile_locked",
        lambda *args: [{"kind": "retrain", "state": "REQUEUED", "job_id": "1_0"}],
    )
    with pytest.raises(RuntimeError, match="remain active"):
        operations.aggregate_campaign(root, repository)


def test_aggregate_gate_rejects_changed_membership(monkeypatch, tmp_path) -> None:
    """An oracle cannot change a frozen front membership while retaining its index."""
    root, repository = _campaign(tmp_path)
    design = _freeze_design()
    manifest = _freeze_manifest(design)
    oracle = _oracle(manifest)
    oracle["all_front_memberships"][0]["trial_number"] = 99
    _aggregate_campaign_files(root, design, manifest)
    runner = repository / "scripts" / "physics_background_pairing_campaign.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    _authenticate(monkeypatch)
    monkeypatch.setattr(operations, "_reconcile_locked", lambda *args: [])
    monkeypatch.setattr(operations, "_validate_frozen_manifest_contract", lambda *args: None)
    monkeypatch.setattr(operations, "_validate_retrain_marker", lambda *args: True)

    def run(command, **kwargs):
        oracle_path = root / "results" / "oracle_winners.json"
        oracle_path.parent.mkdir(parents=True)
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(operations.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="identity differs"):
        operations.aggregate_campaign(root, repository)


@pytest.mark.parametrize("tampered_artifact", ["marker", "oracle"])
def test_aggregate_gate_rejects_tampered_completed_transition(
    monkeypatch, tmp_path, tampered_artifact: str
) -> None:
    """An idempotent rerun fails closed if a receipt-bound artifact changed."""
    root, repository = _campaign(tmp_path)
    design = _freeze_design()
    manifest = _freeze_manifest(design)
    oracle = _oracle(manifest)
    _aggregate_campaign_files(root, design, manifest)
    runner = repository / "scripts" / "physics_background_pairing_campaign.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    _authenticate(monkeypatch)
    monkeypatch.setattr(operations, "_reconcile_locked", lambda *args: [])
    monkeypatch.setattr(operations, "_validate_frozen_manifest_contract", lambda *args: None)
    monkeypatch.setattr(operations, "_validate_retrain_marker", lambda *args: True)

    def run(command, **kwargs):
        oracle_path = root / "results" / "oracle_winners.json"
        oracle_path.parent.mkdir(parents=True)
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(operations.subprocess, "run", run)
    operations.aggregate_campaign(root, repository)
    if tampered_artifact == "marker":
        marker = root / "retraining" / "markers" / "0000.json"
        marker.write_text(marker.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        oracle_path = root / "results" / "oracle_winners.json"
        changed = json.loads(oracle_path.read_text(encoding="utf-8"))
        changed["created_at"] = "2026-01-02T00:00:00+00:00"
        oracle_path.write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("a completed transition was rerun"),
    )
    with pytest.raises(RuntimeError, match="receipt is changed or inconsistent"):
        operations.aggregate_campaign(root, repository)


def test_freeze_gate_seals_exact_quiescent_first_trial_pool(monkeypatch, tmp_path) -> None:
    """Freeze authenticates search evidence, backups, and the resulting 56-front manifest."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    design = _freeze_design()
    manifest = _freeze_manifest(design)
    (repository / "scripts" / "physics_background_pairing_campaign.py").write_text(
        "#!/usr/bin/env python\n", encoding="utf-8"
    )
    for model in operations.MODELS:
        database = root / "studies" / f"{model}.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        database.write_bytes(f"database-{model}".encode())

    monkeypatch.setattr(operations, "TARGET_TRIALS", 1)
    monkeypatch.setattr(report, "_design", lambda campaign_root: design)
    monkeypatch.setattr(operations, "_reconcile_locked", lambda *args: [])
    monkeypatch.setattr(operations, "_assert_no_live_search_overlap", lambda *args: set())
    monkeypatch.setattr(operations, "_assert_no_live_retrain_overlap", lambda *args: set())
    monkeypatch.setattr(operations, "_assert_database_quiescent", lambda *args: None)
    statuses = [
        {
            "model": model,
            "usable_complete": 1,
            "selected_for_freeze": 1,
            "running": 0,
            "waiting": 0,
        }
        for model in operations.MODELS
    ]
    trials = [
        {"model": model, "trial_number": 0, "selected_for_freeze": True}
        for model in operations.MODELS
    ]
    monkeypatch.setattr(report, "_trial_exports", lambda *args: (statuses, trials))
    bindings = {
        (model, 0): {"optimized_metric_artifact": str(tmp_path / f"{model}.json")}
        for model in operations.MODELS
    }
    monkeypatch.setattr(report, "_binding_index", lambda *args: bindings)
    monkeypatch.setattr(report, "_validate_bindings", lambda *args: None)
    checkpoint_rows = [
        {"model": study["model"], "trial_number": 0, "logical_objective": objective["id"]}
        for study in design["studies"]
        for objective in study["objectives"]
    ]
    mlflow_rows = [{"model": model, "trial_number": 0} for model in operations.MODELS]
    monkeypatch.setattr(
        report, "_checkpoint_rows", lambda *args, **kwargs: (checkpoint_rows, mlflow_rows)
    )
    monkeypatch.setattr(report, "_annotate_checkpoint_families", lambda *args: args[1])
    monkeypatch.setattr(report, "_validate_mlflow_runs", lambda *args, **kwargs: args[2])
    candidates = [
        {"slot": "last", "required": True, "available": True}
        for _ in range(len(operations.MODELS))
    ]
    monkeypatch.setattr(report, "_checkpoint_candidate_rows", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(report, "_validate_manifest_against_studies", lambda *args: None)

    def backup(campaign_root: Path, model: str) -> Path:
        path = campaign_root / "studies" / "backups" / model / "frozen.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((campaign_root / "studies" / f"{model}.db").read_bytes())
        path.with_suffix(".sha256").write_text("digest  frozen.db\n", encoding="utf-8")
        return path

    monkeypatch.setattr(operations, "_create_study_backup", backup)

    def run(command, **kwargs):
        path = root / "selection" / "pareto_retrain_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(operations.subprocess, "run", run)
    output = operations.freeze_campaign(root, repository)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_trials"] == 6
    assert payload["logical_fronts"] == 56
    assert payload["front_memberships"] == 56
    assert payload["unique_retrains"] == 6
    assert payload["mlflow_runs"] == 6
    assert len(payload["study_databases"]) == 6
    assert all(Path(row["backup"]).is_file() for row in payload["study_databases"])


def test_freeze_gate_rejects_nonexact_trial_budget(monkeypatch, tmp_path) -> None:
    """A quiescent study with fewer than the exact budget cannot be frozen."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    design = _freeze_design()
    monkeypatch.setattr(operations, "TARGET_TRIALS", 1)
    monkeypatch.setattr(report, "_design", lambda campaign_root: design)
    monkeypatch.setattr(operations, "_reconcile_locked", lambda *args: [])
    monkeypatch.setattr(operations, "_assert_no_live_search_overlap", lambda *args: set())
    monkeypatch.setattr(operations, "_assert_no_live_retrain_overlap", lambda *args: set())
    monkeypatch.setattr(operations, "_assert_database_quiescent", lambda *args: None)
    statuses = [
        {
            "model": model,
            "usable_complete": 0 if model == "ae" else 1,
            "selected_for_freeze": 0 if model == "ae" else 1,
            "running": 0,
            "waiting": 0,
        }
        for model in operations.MODELS
    ]
    monkeypatch.setattr(report, "_trial_exports", lambda *args: (statuses, []))

    with pytest.raises(RuntimeError, match="not exact and quiescent"):
        operations.freeze_campaign(root, repository)


def test_freeze_gate_rejects_active_ledger_job(monkeypatch, tmp_path) -> None:
    """Reconciled scheduler activity blocks the immutable transition."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    design = _freeze_design()
    monkeypatch.setattr(report, "_design", lambda campaign_root: design)
    monkeypatch.setattr(
        operations,
        "_reconcile_locked",
        lambda *args: [
            {"job_id": "42", "kind": "search_chunk", "model": "ae", "state": "RUNNING"}
        ],
    )
    monkeypatch.setattr(operations, "_assert_no_live_search_overlap", lambda *args: set())
    monkeypatch.setattr(operations, "_assert_no_live_retrain_overlap", lambda *args: set())
    monkeypatch.setattr(operations, "_assert_database_quiescent", lambda *args: None)

    with pytest.raises(RuntimeError, match="remain active"):
        operations.freeze_campaign(root, repository)


def test_freeze_gate_rejects_active_optuna_database(monkeypatch, tmp_path) -> None:
    """A RUNNING/WAITING Optuna writer blocks freeze even without a live ledger row."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    design = _freeze_design()
    monkeypatch.setattr(report, "_design", lambda campaign_root: design)
    monkeypatch.setattr(operations, "_reconcile_locked", lambda *args: [])
    monkeypatch.setattr(operations, "_assert_no_live_search_overlap", lambda *args: set())
    monkeypatch.setattr(operations, "_assert_no_live_retrain_overlap", lambda *args: set())

    def quiescent(campaign_root: Path, model: str) -> None:
        if model == "ae":
            raise RuntimeError("Study ae is not quiescent: ['RUNNING'].")

    monkeypatch.setattr(operations, "_assert_database_quiescent", quiescent)

    with pytest.raises(RuntimeError, match="not quiescent"):
        operations.freeze_campaign(root, repository)


def test_existing_frozen_manifest_must_authenticate(monkeypatch, tmp_path) -> None:
    """A pre-existing manifest is never accepted merely because the file exists."""
    design = _freeze_design()
    manifest = _freeze_manifest(design)
    manifest["selection_uses_downstream_anomalies"] = True
    monkeypatch.setattr(report, "_validate_manifest_against_studies", lambda *args: None)

    with pytest.raises(RuntimeError, match="violates the campaign contract"):
        operations._validate_frozen_manifest_contract(tmp_path, design, manifest)


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


def test_submit_canary_registers_fixed_four_trial_one_node_job(monkeypatch, tmp_path) -> None:
    """The first-class production canary has no operator-tunable search shape."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    monkeypatch.setattr(operations, "_require_restart_canary_gate", lambda *args: None)
    monkeypatch.setattr(operations, "_study_trial_count", lambda *args: 0)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="24680;cluster\n")

    monkeypatch.setattr(operations.subprocess, "run", run)
    job_id = operations.submit_canary(
        root,
        repository,
        "ae",
        tmp_path / "data",
        tmp_path / "raw",
        tmp_path / "pairing",
    )

    assert job_id == "24680"
    assert "--nodes=1" in calls[0]
    export = next(value for value in calls[0] if value.startswith("--export="))
    assert "CHUNK_TRIALS=4" in export
    assert "CHUNK_JOBS=4" in export
    assert "TIMEOUT_MIN=720" in export
    with (root / "state" / "job_history.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["kind"] == "production_canary"
    assert row["model"] == "ae"
    assert row["state"] == "PENDING"
    assert row["commit"] == "a" * 40


def test_submit_canary_requires_validated_forced_restart_gate(monkeypatch, tmp_path) -> None:
    """No real Optuna canary starts before scheduler restart recovery is proved."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)

    with pytest.raises(RuntimeError, match="Forced-restart canary"):
        operations.submit_canary(
            root,
            repository,
            "vae",
            tmp_path / "data",
            tmp_path / "raw",
            tmp_path / "pairing",
        )


def test_submit_canary_requires_exact_empty_study(monkeypatch, tmp_path) -> None:
    """A production canary cannot be mislabeled after any persisted trial exists."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    monkeypatch.setattr(operations, "_require_restart_canary_gate", lambda *args: None)
    monkeypatch.setattr(operations, "_study_trial_count", lambda *args: 1)

    with pytest.raises(RuntimeError, match="requires an empty vae study"):
        operations.submit_canary(
            root,
            repository,
            "vae",
            tmp_path / "data",
            tmp_path / "raw",
            tmp_path / "pairing",
        )


def test_study_trial_count_includes_every_persisted_trial_state(tmp_path) -> None:
    """Canary emptiness is total trial cardinality, not merely usable completions."""
    import optuna

    root = tmp_path / "campaign"
    (root / "studies").mkdir(parents=True)
    design = {
        "studies": [{"model": "ae", "study_name": "study_ae"}],
    }
    study = optuna.create_study(
        study_name="study_ae",
        storage=f"sqlite:///{root / 'studies' / 'ae.db'}",
        directions=["minimize"],
    )
    assert operations._study_trial_count(root, design, "ae") == 0

    trial = study.ask()
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    assert operations._study_trial_count(root, design, "ae") == 1


def test_submit_canary_refuses_a_second_registered_model_canary(monkeypatch, tmp_path) -> None:
    """Exactly one production-canary allocation may claim each model."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    operations._atomic_csv(
        root / "state" / "job_history.csv",
        [
            {
                "job_id": "123",
                "kind": "production_canary",
                "model": "ae",
                "state": "COMPLETED",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="already contains"):
        operations.submit_canary(
            root,
            repository,
            "ae",
            tmp_path / "data",
            tmp_path / "raw",
            tmp_path / "pairing",
        )


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
                "success_sha256": "c" * 64,
                "bindings_sha256": "d" * 64,
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


def test_validate_restart_canary_persists_idempotent_authenticated_evidence(
    monkeypatch, tmp_path
) -> None:
    """Repeated validation preserves one immutable gate rather than rewriting it."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    canary_root = tmp_path / "canaries" / "retrain_restart_canary_example"
    evidence = {
        "schema_version": 1,
        "code_commit": "a" * 40,
        "canary_root": str(canary_root.resolve()),
        "job_id": "987",
        "complete_sha256": "b" * 64,
    }
    monkeypatch.setattr(operations, "_validate_restart_canary_evidence", lambda *args: evidence)

    first = operations.validate_restart_canary(root, repository, canary_root, "987")
    first_bytes = first.read_bytes()
    second = operations.validate_restart_canary(root, repository, canary_root, "987")

    assert second == first
    assert second.read_bytes() == first_bytes
    assert json.loads(first.read_text(encoding="utf-8"))["evidence"] == evidence


def test_restart_canary_gate_reauthenticates_persisted_evidence(monkeypatch, tmp_path) -> None:
    """A gate file alone is insufficient when its external evidence has changed."""
    root, _ = _campaign(tmp_path)
    design = operations._design(root)
    canary_root = tmp_path / "canaries" / "retrain_restart_canary_example"
    evidence = {
        "schema_version": 1,
        "code_commit": "a" * 40,
        "canary_root": str(canary_root.resolve()),
        "job_id": "987",
        "complete_sha256": "b" * 64,
    }
    gate = root / "state" / "restart_canary_validation.json"
    gate.parent.mkdir()
    gate.write_text(
        json.dumps({"schema_version": 1, "validated_at": "now", "evidence": evidence}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        operations,
        "_validate_restart_canary_evidence",
        lambda *args: {**evidence, "complete_sha256": "c" * 64},
    )

    with pytest.raises(RuntimeError, match="changed after validation"):
        operations._require_restart_canary_gate(root, design)


def test_restart_canary_accounting_reads_terminal_restart_count(monkeypatch) -> None:
    """The forced gate uses allocation-level JSON, including Slurm RestartCnt."""
    payload = {
        "jobs": [
            {
                "job_id": 987,
                "name": "phys-bgpair-restart-canary",
                "state": {"current": ["COMPLETED"]},
                "exit_code": {
                    "return_code": {"number": 0},
                    "signal": {"id": {"number": 0}},
                },
                "restart_cnt": 1,
                "stdout_expanded": "/scratch/phys-bgpair-restart-canary-987.out",
            }
        ]
    }
    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    observed = operations._restart_canary_accounting("987")

    assert observed == {
        "job_id": "987",
        "job_name": "phys-bgpair-restart-canary",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "restart_count": 1,
        "stdout": "/scratch/phys-bgpair-restart-canary-987.out",
    }


def test_restart_canary_accounting_rejects_nonallocation_identity() -> None:
    """Task-like or otherwise nonnumeric scheduler IDs cannot satisfy the gate."""
    with pytest.raises(ValueError, match="must be numeric"):
        operations._restart_canary_accounting("987_0")


def test_restart_canary_evidence_authenticates_exact_epoch_oracle_and_hashes(
    monkeypatch, tmp_path
) -> None:
    """The independent validator covers scheduler, MLflow, rollback, and oracle proof."""
    root, _ = _campaign(tmp_path)
    commit = "a" * 40
    job_id = "987"
    canary_root = tmp_path / "canaries" / "retrain_restart_canary_schema2"
    evidence_root = canary_root / "evidence"
    retained_root = canary_root / "retained"
    evidence_root.mkdir(parents=True)
    retained_root.mkdir()
    hpc = evidence_root / "hpc_input_001.ckpt"
    exact_resume = evidence_root / "exact_resume_001.ckpt"
    retained = retained_root / "stable.ckpt"
    for path, value in (
        (hpc, "hpc"),
        (retained, "stable"),
    ):
        path.write_text(value, encoding="utf-8")
    rng_state = {
        "schema_version": 1,
        "python": (3, (1, 2, 3), None),
        "numpy": {
            "algorithm": "MT19937",
            "keys": torch.tensor([1, 2, 3], dtype=torch.uint32),
            "position": 1,
            "has_gauss": 0,
            "cached_gaussian": 0.0,
        },
        "torch_cpu": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "torch_cuda": [torch.tensor([4, 5, 6], dtype=torch.uint8)],
        "datamodule_shuffler": torch.tensor([7, 8, 9], dtype=torch.uint8),
    }
    torch.save(
        {
            "epoch": 0,
            "loops": {
                "fit_loop": {
                    "epoch_progress": {"total": {"ready": 1, "started": 1, "processed": 1}}
                }
            },
            "callbacks": {"physics-background-pairing-exact-retrain-rng-v1": rng_state},
        },
        exact_resume,
    )
    retained_rows = [{"path": str(retained.resolve()), "sha256": operations._sha256(retained)}]
    restore = evidence_root / "restore_001.json"
    restore.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "restored_at": "now",
                "restart_count": 1,
                "callback_states": {
                    "exact_rng": operations._restart_canary_jsonable(rng_state),
                    **{
                        name: {"state": name}
                        for name in ("efficiency", "cap", "wasserstein", "drift", "stable")
                    },
                },
                "retained_checkpoints": retained_rows,
            }
        ),
        encoding="utf-8",
    )
    snapshot = evidence_root / "scheduler_log_snapshot.txt"
    snapshot_text = (
        "CANARY_ATTEMPT_START restart=0 job=987\n"
        "CANARY_LIGHTNING_CHILD restart=0\n"
        "Handling auto-requeue signal\n"
        "CANARY_ATTEMPT_START restart=1 job=987\n"
        "CANARY_LIGHTNING_CHILD restart=1\n"
    )
    snapshot.write_text(snapshot_text, encoding="utf-8")
    complete_path = canary_root / "COMPLETE.json"
    scheduler_log = tmp_path / "phys-bgpair-restart-canary-987.out"
    scheduler_log.write_text(
        snapshot_text
        + f"CANARY_COMPLETE path={complete_path}\n"
        + "CANARY_ATTEMPT_FINISH restart=1 job=987\n",
        encoding="utf-8",
    )
    tracking_uri = f"file:{(canary_root / 'mlflow' / 'mlruns').resolve()}"
    (canary_root / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "code_commit": commit,
                "mlflow_run_id": "run",
                "tracking_uri": tracking_uri,
            }
        ),
        encoding="utf-8",
    )
    expected_tags = operations._restart_canary_expected_tags(canary_root, commit)
    sentinels = operations._restart_canary_scheduler_sentinels(snapshot_text)
    complete = {
        "schema_version": 2,
        "completed_at": "now",
        "code_commit": commit,
        "slurm_job_id": job_id,
        "slurm_restart_count": 1,
        "cuda_device": "gpu",
        "hpc_checkpoint_evidence": {
            "path": str(hpc.resolve()),
            "sha256": operations._sha256(hpc),
            "resume_eligible": False,
        },
        "exact_resume_checkpoint": {
            "path": str(exact_resume.resolve()),
            "sha256": operations._sha256(exact_resume),
            "completed_epochs": 1,
            "mlflow_epoch_steps": [0],
            "mlflow_epoch_values": [0.0],
        },
        "restore_evidence": {
            "path": str(restore.resolve()),
            "sha256": operations._sha256(restore),
        },
        "retained_checkpoints": retained_rows,
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_name": operations.RESTART_CANARY_EXPERIMENT,
            "run_id": "run",
            "allocation_steps": [0, 1],
            "epoch_steps": list(range(6)),
            "epoch_values": [float(value) for value in range(6)],
            "tags": expected_tags,
        },
        "scheduler_log": {
            "source_path": str(scheduler_log.resolve()),
            "snapshot_path": str(snapshot.resolve()),
            "snapshot_sha256": operations._sha256(snapshot),
            "sentinels": sentinels,
        },
        "initial_state_sha256": "1" * 64,
        "final_state_sha256": "2" * 64,
        "uninterrupted_reference_state_sha256": "2" * 64,
        "parameters_match_uninterrupted_reference": True,
        "final_epoch": 6,
    }
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    expected_tracking_uri = tracking_uri

    class Client:
        def __init__(self, tracking_uri):
            assert tracking_uri == expected_tracking_uri

        def get_run(self, run_id):
            assert run_id == "run"
            return SimpleNamespace(
                info=SimpleNamespace(experiment_id="experiment", status="FINISHED"),
                data=SimpleNamespace(tags=expected_tags),
            )

        def get_experiment(self, experiment_id):
            assert experiment_id == "experiment"
            return SimpleNamespace(name=operations.RESTART_CANARY_EXPERIMENT)

        def get_metric_history(self, run_id, name):
            values = range(6) if name == "epoch_idx" else range(2)
            return [
                SimpleNamespace(step=value, value=float(value), timestamp=value)
                for value in values
            ]

    import mlflow.tracking

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", Client)
    monkeypatch.setattr(
        operations,
        "_restart_canary_accounting",
        lambda _: {
            "job_id": job_id,
            "job_name": "phys-bgpair-restart-canary",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "restart_count": 1,
            "stdout": str(scheduler_log.resolve()),
        },
    )

    observed = operations._validate_restart_canary_evidence(root, canary_root, job_id, commit)
    assert observed["exact_resume_completed_epochs"] == 1
    assert observed["mlflow_run_id"] == "run"

    restore_payload = json.loads(restore.read_text(encoding="utf-8"))
    restored_original = json.loads(json.dumps(restore_payload))
    restore_payload["callback_states"]["exact_rng"]["torch_cpu"][0] = 255
    restore.write_text(json.dumps(restore_payload), encoding="utf-8")
    complete["restore_evidence"]["sha256"] = operations._sha256(restore)
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(RuntimeError, match="callback restoration evidence"):
        operations._validate_restart_canary_evidence(root, canary_root, job_id, commit)

    restore.write_text(json.dumps(restored_original), encoding="utf-8")
    complete["restore_evidence"]["sha256"] = operations._sha256(restore)
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    exact_resume.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact reference is invalid"):
        operations._validate_restart_canary_evidence(root, canary_root, job_id, commit)


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


def test_submit_refuses_timeout_beyond_normal_partition_limit(monkeypatch, tmp_path) -> None:
    """The mutable helper cannot advertise more time than Slurm grants."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)

    with pytest.raises(ValueError, match="720-minute"):
        operations.submit_search(
            root,
            repository,
            "ae",
            4,
            4,
            721,
            tmp_path / "data",
            tmp_path / "raw",
            tmp_path / "pairing",
        )


def test_submit_refuses_unregistered_live_packed_job(monkeypatch, tmp_path) -> None:
    """A live scheduler allocation cannot hide outside the campaign ledger."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    monkeypatch.setattr(operations, "_live_search_job_ids", lambda: {"123"})

    with pytest.raises(RuntimeError, match="absent from the campaign ledger"):
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


def test_operation_lock_refuses_concurrent_operator(tmp_path) -> None:
    """A second process-equivalent file description cannot enter the control plane."""
    root = tmp_path / "campaign"
    root.mkdir()

    with operations._operation_lock(root):
        with pytest.raises(RuntimeError, match="Another campaign operation"):
            with operations._operation_lock(root):
                pass


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


def test_reconcile_seals_successful_search_manifest_and_bindings(monkeypatch, tmp_path) -> None:
    """A successful search row authenticates its chunk and exact trial bindings."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    log = tmp_path / "search.out"
    log.write_text("done\n", encoding="utf-8")
    chunk = root / "sweeps" / "ae" / "stamp" / "chunk.json"
    chunk.parent.mkdir(parents=True)
    chunk.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slurm_job_id": "7",
                "model": "ae",
                "controller_status": "complete",
                "completed_at": "now",
                "requested_trials": 2,
            }
        ),
        encoding="utf-8",
    )
    bindings = chunk.parent / "trial_bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slurm_job_id": "7",
                "model": "ae",
                "bindings": [
                    {"optimized_metric_artifact": "a"},
                    {"optimized_metric_artifact": "b"},
                ],
            }
        ),
        encoding="utf-8",
    )
    operations._atomic_csv(
        root / "state" / "job_history.csv",
        [
            {
                "job_id": "7",
                "kind": "search_chunk",
                "model": "ae",
                "state": "RUNNING",
                "log_path": str(log),
            }
        ],
    )
    monkeypatch.setattr(
        operations,
        "_accounting",
        lambda ids: {"7": {"state": "COMPLETED", "exit_code": "0:0"}},
    )
    validation = chunk.parent / "chunk_validation.json"
    validation.write_text('{"validated": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        operations,
        "_validate_completed_search_bindings",
        lambda *args: validation,
    )

    row = operations.reconcile(root, repository)[0]

    assert row["success_sha256"] == operations._sha256(validation)
    assert row["bindings_sha256"] == operations._sha256(bindings)


def test_terminal_chunk_validation_authenticates_only_supplied_bindings(
    monkeypatch, tmp_path
) -> None:
    """Per-chunk sealing checks objectives, candidates, MLflow, and exact identities."""
    root = tmp_path / "campaign"
    chunk = root / "sweeps" / "ae" / "stamp" / "chunk.json"
    run_dir = chunk.parent / "0"
    run_dir.mkdir(parents=True)
    chunk.write_text(json.dumps({"model": "ae"}), encoding="utf-8")
    bindings_path = chunk.parent / "trial_bindings.json"
    bindings_path.write_text("{}\n", encoding="utf-8")
    artifact = run_dir / "optimized_metric.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "objective_order": ["native__wasserstein"],
                "optimized_metric": [1.0, 2.0],
                "selections": {
                    "native__wasserstein": {
                        "optimized_metric": [1.0, 2.0],
                        "optimized_ckpt_name": "last",
                    }
                },
                "run_provenance": {
                    "experiment_name": "experiment",
                    "run_name": "run",
                    "hydra_output_dir": str(run_dir),
                    "checkpoint_run_dir": str(root / "runtime" / "checkpoints" / "run"),
                    "loggers": [{"experiment_id": "1", "run_id": "abc"}],
                },
            }
        ),
        encoding="utf-8",
    )
    binding = {
        "model": "ae",
        "study_name": "study",
        "trial_number": 4,
        "hydra_job_num": 0,
        "state": "COMPLETE",
        "params": {"x": 1},
        "values": [1.0, 2.0],
        "optimized_metric_artifact": str(artifact),
        "optimized_metric_artifact_sha256": operations._sha256(artifact),
    }
    design = {
        "schema_version": 2,
        "code_commit": "a" * 40,
        "studies": [
            {
                "model": "ae",
                "study_name": "study",
                "objectives": [
                    {
                        "id": "native__wasserstein",
                        "score": "native",
                        "metric": "wasserstein",
                        "strategy": None,
                        "directions": ["minimize", "minimize"],
                    }
                ],
            }
        ],
    }
    observed = {}
    monkeypatch.setattr(
        report,
        "_validate_bindings",
        lambda campaign, frozen, bindings: observed.setdefault("bindings", bindings),
    )
    monkeypatch.setattr(report, "_annotate_checkpoint_families", lambda *args: args[1])
    monkeypatch.setattr(
        report,
        "_checkpoint_candidate_rows",
        lambda *args, **kwargs: [
            {"required": True, "available": True},
            {"required": False, "available": False},
            {"required": True, "available": True},
        ],
    )
    monkeypatch.setattr(
        report, "_validate_mlflow_runs", lambda campaign, frozen, rows, final: rows
    )

    output = operations._validate_completed_search_bindings(
        root, design, chunk, bindings_path, [binding]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(observed["bindings"]) == {("ae", 4)}
    assert payload["usable_trials"] == 1
    assert payload["selection_rows"] == 1
    assert payload["candidate_slots"] == 3
    assert payload["mlflow_runs"] == 1


def test_terminal_chunk_validation_rejects_noncomplete_binding(tmp_path) -> None:
    """A finite vector is insufficient when its explicit binding is not COMPLETE."""
    root = tmp_path / "campaign"
    chunk = root / "sweeps" / "ae" / "stamp" / "chunk.json"
    run_dir = chunk.parent / "0"
    run_dir.mkdir(parents=True)
    chunk.write_text(json.dumps({"model": "ae"}), encoding="utf-8")
    bindings = chunk.parent / "trial_bindings.json"
    bindings.write_text("{}\n", encoding="utf-8")
    artifact = run_dir / "optimized_metric.json"
    artifact.write_text("{}\n", encoding="utf-8")
    design = {
        "studies": [
            {
                "model": "ae",
                "study_name": "study",
                "objectives": [{"id": "objective"}],
            }
        ]
    }
    binding = {
        "model": "ae",
        "study_name": "study",
        "trial_number": 4,
        "hydra_job_num": 0,
        "state": "RUNNING",
        "values": [1.0, 2.0],
        "optimized_metric_artifact": str(artifact),
        "optimized_metric_artifact_sha256": operations._sha256(artifact),
    }

    with pytest.raises(RuntimeError, match="not usable"):
        operations._validate_completed_search_bindings(root, design, chunk, bindings, [binding])


def test_repair_stale_backs_up_and_fails_only_explicit_trials(monkeypatch, tmp_path) -> None:
    """Stale repair is scheduler-proven, API-based, backed up, and auditable."""
    import optuna

    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    design = {
        "schema_version": 2,
        "code_commit": "a" * 40,
        "studies": [{"model": "ae", "study_name": "study", "objectives": []}],
    }
    (root / "design.json").write_text(json.dumps(design), encoding="utf-8")
    database = root / "studies" / "ae.db"
    database.parent.mkdir()
    study = optuna.create_study(study_name="study", storage=f"sqlite:///{database}")
    study.ask()
    study.enqueue_trial({"x": 1})
    monkeypatch.setattr(operations, "_reconcile_locked", lambda *args: [])

    output = operations.repair_stale(root, repository, "ae", [0, 1])

    repaired = optuna.load_study(study_name="study", storage=f"sqlite:///{database}")
    assert [trial.state.name for trial in repaired.get_trials()] == ["FAIL", "FAIL"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["trials_before"] == [
        {"state": "RUNNING", "trial_number": 0},
        {"state": "WAITING", "trial_number": 1},
    ]
    assert Path(payload["backup"]).is_file()
    assert operations._sha256(Path(payload["backup"])) == payload["backup_sha256"]
    events = [
        json.loads(line)
        for line in (root / "state" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["outcome"] == "repaired"


def test_repair_stale_refuses_live_same_model_and_records_evidence(monkeypatch, tmp_path) -> None:
    """No Optuna mutation is attempted while the scheduler still owns the model."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    monkeypatch.setattr(
        operations,
        "_reconcile_locked",
        lambda *args: [{"job_id": "7", "kind": "search_chunk", "model": "ae", "state": "RUNNING"}],
    )
    monkeypatch.setattr(operations, "_live_search_job_ids", lambda: {"7"})

    with pytest.raises(RuntimeError, match="Live ae packed-search"):
        operations.repair_stale(root, repository, "ae", [0])

    event = json.loads(
        (root / "state" / "failures.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert event["outcome"] == "refused_or_failed"
    assert "Live ae packed-search" in event["error_signature"]


def test_reconcile_seals_successful_retrain_marker(monkeypatch, tmp_path) -> None:
    """A successful retrain array task authenticates its complete marker artifact."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    log = tmp_path / "retrain.out"
    log.write_text("done\n", encoding="utf-8")
    retrain_row = {"retrain_index": 0, "model": "ae", "trial_number": 3}
    selection = root / "selection"
    selection.mkdir()
    (selection / "pareto_retrain_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "code_commit": "a" * 40,
                "retrain_runs": [retrain_row],
            }
        ),
        encoding="utf-8",
    )
    optimized = tmp_path / "optimized.json"
    optimized.write_text("{}\n", encoding="utf-8")
    values = tmp_path / "values.csv"
    values.write_text("value\n1\n", encoding="utf-8")
    marker = root / "retraining" / "markers" / "0000.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "row": retrain_row,
                "slurm_array_job_id": "9",
                "slurm_array_task_id": "0",
                "scheduler_identity": None,
                "optimized_metric_artifact": str(optimized),
                "optimized_metric_artifact_sha256": operations._sha256(optimized),
                "value_files": [{"path": str(values), "sha256": operations._sha256(values)}],
            }
        ),
        encoding="utf-8",
    )
    from scripts import physics_background_pairing_campaign as campaign

    monkeypatch.setattr(campaign, "_validate_retrain_marker", lambda *args: None)
    operations._atomic_csv(
        root / "state" / "job_history.csv",
        [
            {
                "job_id": "9_0",
                "kind": "retrain",
                "model": "ae",
                "state": "RUNNING",
                "log_path": str(log),
            }
        ],
    )
    monkeypatch.setattr(
        operations,
        "_accounting",
        lambda ids: {"9_0": {"state": "COMPLETED", "exit_code": "0:0"}},
    )

    row = operations.reconcile(root, repository)[0]

    receipt = root / "state" / "retrain_receipts" / "9_0.json"
    assert row["success_sha256"] == operations._sha256(receipt)
    assert json.loads(receipt.read_text(encoding="utf-8"))["evidence"]["sha256"] == (
        operations._sha256(marker)
    )


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
    """Retraining maps bounded local tasks to exact sparse frozen indices."""
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
    assert "--array=0-1%2" in calls[0]
    with (root / "state" / "job_history.csv").open(encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    assert [row["job_id"] for row in ledger] == ["987_0", "987_1"]
    assert len({row["bindings_sha256"] for row in ledger}) == 1
    digest = ledger[0]["bindings_sha256"]
    task_map_path = root / "state" / "retrain_task_maps" / f"{digest}.json"
    assert operations._sha256(task_map_path) == digest
    task_map = json.loads(task_map_path.read_text(encoding="utf-8"))
    assert [entry["retrain_index"] for entry in task_map["entries"]] == [1, 3]
    assert operations._retrain_index_for_ledger_row(root, operations._design(root), ledger[1]) == 3


def test_submit_retrains_batches_at_clariden_max_array_size(monkeypatch, tmp_path) -> None:
    """Only one <=1001-local-task batch is submitted even for a 3,600-row front union."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    selection = root / "selection"
    selection.mkdir()
    rows = [
        {"retrain_index": index, "model": "ae", "trial_number": index} for index in range(3600)
    ]
    (selection / "pareto_retrain_manifest.json").write_text(
        json.dumps({"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": rows}),
        encoding="utf-8",
    )
    monkeypatch.setattr(operations, "_validate_retrain_marker", lambda *args: False)
    monkeypatch.setattr(operations.shutil, "which", lambda name: f"/bin/{name}")
    calls = []
    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(stdout="988\n"),
    )

    operations.submit_retrains(
        root, repository, 24, tmp_path / "data", tmp_path / "raw", tmp_path / "pairing"
    )

    assert "--array=0-1000%24" in calls[0]
    with (root / "state" / "job_history.csv").open(encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    assert len(ledger) == operations.MAX_RETRAIN_ARRAY_TASKS
    assert ledger[-1]["job_id"] == "988_1000"
    assert (
        operations._retrain_index_for_ledger_row(root, operations._design(root), ledger[-1])
        == 1000
    )


def test_retrain_ledger_mapping_fails_closed_on_map_tampering(monkeypatch, tmp_path) -> None:
    """A successful local task cannot be reconciled through a changed map."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    selection = root / "selection"
    selection.mkdir()
    manifest_path = selection / "pareto_retrain_manifest.json"
    manifest = {
        "schema_version": 2,
        "code_commit": "a" * 40,
        "retrain_runs": [{"retrain_index": 0, "model": "vae", "trial_number": 7}],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    task_map_path, digest, _ = operations._write_retrain_task_map(
        root, operations._design(root), manifest_path, manifest, [0]
    )
    task_map_path.write_text(task_map_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing or changed"):
        operations._retrain_index_for_ledger_row(
            root,
            operations._design(root),
            {
                "job_id": "900_0",
                "kind": "retrain",
                "model": "vae",
                "bindings_sha256": digest,
            },
        )


def test_retrain_ledger_supports_legacy_direct_index_rows(tmp_path) -> None:
    """Pre-map campaign ledgers retain their direct task-index interpretation."""
    root, _ = _campaign(tmp_path)
    assert (
        operations._retrain_index_for_ledger_row(
            root,
            operations._design(root),
            {"job_id": "900_3599", "kind": "retrain", "model": "ae", "bindings_sha256": ""},
        )
        == 3599
    )


def test_completed_retrain_reconciliation_resolves_local_task_to_actual_marker(
    monkeypatch, tmp_path
) -> None:
    """Terminal reconciliation seals the mapped frozen index, not the local array ID."""
    root, _ = _campaign(tmp_path)
    selection = root / "selection"
    selection.mkdir()
    rows = [{"retrain_index": index, "model": "vae", "trial_number": index} for index in range(4)]
    manifest = {"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": rows}
    manifest_path = selection / "pareto_retrain_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, digest, _ = operations._write_retrain_task_map(
        root, operations._design(root), manifest_path, manifest, [3]
    )
    marker = root / "retraining" / "markers" / "0003.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "row": rows[3],
                "slurm_array_job_id": "999",
                "slurm_array_task_id": "0",
                "scheduler_identity": {
                    "mode": "mapped",
                    "task_map_sha256": digest,
                    "local_task_id": 0,
                    "retrain_index": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(operations, "_validate_retrain_marker", lambda path, row: True)

    sealed = operations._completed_retrain_marker(
        root,
        operations._design(root),
        {
            "job_id": "999_0",
            "kind": "retrain",
            "model": "vae",
            "bindings_sha256": digest,
        },
    )

    receipt = root / "state" / "retrain_receipts" / "999_0.json"
    assert sealed == operations._sha256(receipt)
    assert json.loads(receipt.read_text(encoding="utf-8"))["evidence"]["sha256"] == (
        operations._sha256(marker)
    )


def test_completed_retrain_reconciliation_accepts_authenticated_pending_segment(
    monkeypatch, tmp_path
) -> None:
    """A successful 40-epoch task seals progress before a fresh array runs the next phase."""
    from scripts import physics_background_pairing_campaign as campaign

    root, _ = _campaign(tmp_path)
    row = {"retrain_index": 0, "model": "ae", "trial_number": 3}
    selection = root / "selection"
    selection.mkdir()
    (selection / "pareto_retrain_manifest.json").write_text(
        json.dumps({"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": [row]}),
        encoding="utf-8",
    )
    progress = _write_retrain_progress(root, row, ["777_0"])
    monkeypatch.setattr(campaign, "_validate_retrain_progress", lambda *args, **kwargs: progress)

    sealed = operations._completed_retrain_marker(
        root,
        operations._design(root),
        {
            "job_id": "777_0",
            "kind": "retrain",
            "model": "ae",
            "bindings_sha256": "",
        },
    )

    receipt = root / "state" / "retrain_receipts" / "777_0.json"
    assert sealed == operations._sha256(receipt)
    assert json.loads(receipt.read_text(encoding="utf-8"))["evidence"][
        "snapshot_sha256"
    ] == operations._json_sha256(json.loads(progress.read_text(encoding="utf-8")))


def test_segment_receipt_projection_accepts_emergency_requeue_attempts() -> None:
    """One Slurm task may restart repeatedly before its unique completed boundary."""
    attempts = [
        {
            "attempt_number": 1,
            "phase": "training",
            "target_epoch_boundary": 40,
            "slurm_array_job_id": "777",
            "slurm_array_task_id": "0",
            "completed_at": None,
        },
        {
            "attempt_number": 2,
            "phase": "training",
            "target_epoch_boundary": 40,
            "slurm_array_job_id": "777",
            "slurm_array_task_id": "0",
            "completed_at": "done",
        },
    ]
    state = {
        "schema_version": 1,
        "attempts": attempts,
        "segments": [{"epoch_boundary": 40, "attempt_number": 2}],
    }

    snapshot = operations._segment_snapshot_for_job(state, "777_0")

    assert snapshot["attempts"] == attempts
    assert snapshot["segments"] == state["segments"]


def test_retrain_receipts_remain_valid_after_later_segments(monkeypatch, tmp_path) -> None:
    """A later mutable restart state cannot rebind an earlier completed task."""
    from scripts import physics_background_pairing_campaign as campaign

    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    row = {"retrain_index": 0, "model": "ae", "trial_number": 3}
    selection = root / "selection"
    selection.mkdir()
    (selection / "pareto_retrain_manifest.json").write_text(
        json.dumps({"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": [row]}),
        encoding="utf-8",
    )
    progress = _write_retrain_progress(root, row, ["700_0"])
    monkeypatch.setattr(campaign, "_validate_retrain_progress", lambda *args, **kwargs: progress)
    first_row = {
        "job_id": "700_0",
        "kind": "retrain",
        "model": "ae",
        "state": "RUNNING",
        "log_path": str(tmp_path / "700_0.out"),
        "bindings_sha256": "",
    }
    Path(first_row["log_path"]).write_text("first\n", encoding="utf-8")
    operations._atomic_csv(root / "state" / "job_history.csv", [first_row])
    monkeypatch.setattr(
        operations,
        "_accounting",
        lambda ids: {job_id: {"state": "COMPLETED", "exit_code": "0:0"} for job_id in ids},
    )
    first_digest = operations.reconcile(root, repository)[0]["success_sha256"]

    _write_retrain_progress(root, row, ["700_0", "701_0"])
    second_row = {
        **first_row,
        "job_id": "701_0",
        "state": "RUNNING",
        "log_path": str(tmp_path / "701_0.out"),
        "success_sha256": "",
        "log_sha256": "",
    }
    Path(second_row["log_path"]).write_text("second\n", encoding="utf-8")
    _, ledger = operations._ledger(root)
    operations._atomic_csv(root / "state" / "job_history.csv", [*ledger, second_row])
    reconciled = operations.reconcile(root, repository)
    second_digest = reconciled[1]["success_sha256"]

    assert reconciled[0]["success_sha256"] == first_digest
    assert operations.reconcile(root, repository)[0]["success_sha256"] == first_digest
    assert (
        operations._completed_retrain_marker(root, operations._design(root), reconciled[0])
        == first_digest
    )
    assert second_digest != first_digest
    assert (
        json.loads(
            (root / "state" / "retrain_receipts" / "700_0.json").read_text(encoding="utf-8")
        )["evidence"]["snapshot"]["segments"][-1]["epoch_boundary"]
        == 40
    )
    assert (
        json.loads(
            (root / "state" / "retrain_receipts" / "701_0.json").read_text(encoding="utf-8")
        )["evidence"]["snapshot"]["segments"][-1]["epoch_boundary"]
        == 80
    )


def test_retrain_receipt_rejects_segment_and_receipt_tampering(monkeypatch, tmp_path) -> None:
    """Receipts fail closed if either sealed artifact bytes or receipt identity changes."""
    from scripts import physics_background_pairing_campaign as campaign

    root, _ = _campaign(tmp_path)
    row = {"retrain_index": 0, "model": "vae", "trial_number": 4}
    selection = root / "selection"
    selection.mkdir()
    (selection / "pareto_retrain_manifest.json").write_text(
        json.dumps({"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": [row]}),
        encoding="utf-8",
    )
    progress = _write_retrain_progress(root, row, ["800_0"])
    monkeypatch.setattr(campaign, "_validate_retrain_progress", lambda *args, **kwargs: progress)
    ledger_row = {
        "job_id": "800_0",
        "kind": "retrain",
        "model": "vae",
        "bindings_sha256": "",
    }
    operations._completed_retrain_marker(root, operations._design(root), ledger_row)
    checkpoint = root / "retraining" / "restart_state" / "0000" / "segments" / "epoch_040.ckpt"
    checkpoint.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact is missing or changed"):
        operations._completed_retrain_marker(root, operations._design(root), ledger_row)

    _write_exact_checkpoint(checkpoint, epoch=39)
    receipt = root / "state" / "retrain_receipts" / "800_0.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["model"] = "ae"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from its frozen task"):
        operations._completed_retrain_marker(root, operations._design(root), ledger_row)


def test_older_retrain_task_cannot_seal_a_later_final_marker(monkeypatch, tmp_path) -> None:
    """A final marker is scheduler-bound; an older task receives its segment snapshot."""
    root, _ = _campaign(tmp_path)
    row = {"retrain_index": 0, "model": "ae", "trial_number": 2}
    selection = root / "selection"
    selection.mkdir()
    (selection / "pareto_retrain_manifest.json").write_text(
        json.dumps({"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": [row]}),
        encoding="utf-8",
    )
    state_path = _write_retrain_progress(root, row, ["900_0"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["attempts"].append(
        {
            "attempt_number": 2,
            "phase": "evaluation",
            "slurm_array_job_id": "901",
            "slurm_array_task_id": "0",
            "completed_at": "done-evaluation",
        }
    )
    state["completed_at"] = "done-evaluation"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    marker = root / "retraining" / "markers" / "0000.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "row": row,
                "slurm_array_job_id": "901",
                "slurm_array_task_id": "0",
                "scheduler_identity": None,
                "restart_state_artifact": str(state_path.resolve()),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(operations, "_validate_retrain_marker", lambda *args: True)

    sealed = operations._completed_retrain_marker(
        root,
        operations._design(root),
        {
            "job_id": "900_0",
            "kind": "retrain",
            "model": "ae",
            "bindings_sha256": "",
        },
    )

    receipt = root / "state" / "retrain_receipts" / "900_0.json"
    assert sealed == operations._sha256(receipt)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["evidence"]["kind"] == "training_segment"
    assert "completed_at" not in payload["evidence"]["snapshot"]


def test_submit_retrains_refuses_unregistered_live_array(monkeypatch, tmp_path) -> None:
    """A post-sbatch ledger crash cannot permit a duplicate retraining array."""
    root, repository = _campaign(tmp_path)
    _authenticate(monkeypatch)
    monkeypatch.setattr(operations, "_live_retrain_job_ids", lambda: {"987"})

    with pytest.raises(RuntimeError, match="absent from the campaign ledger"):
        operations.submit_retrains(
            root,
            repository,
            24,
            tmp_path / "data",
            tmp_path / "raw",
            tmp_path / "pairing",
        )


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

    with pytest.raises(RuntimeError, match="complete artifact audit"):
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
            "success_sha256": "2" * 64,
            "bindings_sha256": "3" * 64,
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
