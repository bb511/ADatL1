#!/usr/bin/env python3
"""Prepare and operate the fixed 84-run physics confirmation campaign.

This is the control-plane entry point for ``CONFIRMATION_PLAN.md``.  It never
starts Optuna: the exact discovery-selected configurations are authenticated and
projected onto the restart-safe segmented retrainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess  # nosec B404 -- fixed read-only Git command
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import physics_background_pairing_campaign as training_campaign
from scripts import physics_background_pairing_operations as operations
from src.evaluation.physics_background_pairing_confirmation_matrix import (
    freeze_confirmation_campaign,
    freeze_reserve_extension_manifest,
    load_authenticated_confirmation_campaign,
)

DEFAULT_DISCOVERY_ROOT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_nsgaii_05b649c"
)


def _git(repository: Path, *args: str) -> str:
    """Run one fixed Git inspection command."""
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git")
    return subprocess.run(  # nosec B603 -- command is fixed by callers
        [executable, *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def prepare(root: Path, discovery_root: Path, repository: Path) -> dict[str, str]:
    """Freeze the confirmation root from a clean execution repository."""
    repository = repository.expanduser().resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Commit all implementation changes before freezing confirmation.")
    commit = _git(repository, "rev-parse", "HEAD")
    paths = freeze_confirmation_campaign(
        discovery_root,
        root,
        execution_commit=commit,
        execution_preflight=repository / "research/physics_background_pairing/preflight.json",
    )
    load_authenticated_confirmation_campaign(root)
    return {key: str(path.resolve()) for key, path in paths.items()}


def status(root: Path, repository: Path) -> dict[str, Any]:
    """Authenticate the matrix and summarize exact segmented progress."""
    campaign = load_authenticated_confirmation_campaign(root)
    operations._authenticate_execution_repository(repository.resolve(), campaign["design"])
    rows = campaign["runner_manifest"]["retrain_runs"]
    complete = []
    boundaries: dict[str, int] = {}
    invalid = []
    active_or_interrupted = []
    for row in rows:
        index = int(row["retrain_index"])
        marker = root / "retraining/markers" / f"{index:04d}.json"
        state = root / "retraining/restart_state" / f"{index:04d}/state.json"
        if marker.is_file():
            try:
                operations._validate_retrain_marker(marker, row)
            except Exception as error:  # status must expose every invalid row
                invalid.append({"retrain_index": index, "error": str(error)})
            else:
                complete.append(index)
            continue
        count = 0
        if state.is_file():
            try:
                payload = json.loads(state.read_text(encoding="utf-8"))
                observed = [int(item["epoch_boundary"]) for item in payload.get("segments", [])]
                expected = [40, 80, 120, 160, 200][: len(observed)]
                if observed != expected:
                    raise ValueError(f"noncontiguous boundaries {observed}")
                attempts = payload.get("attempts", [])
                if not attempts:
                    raise ValueError("restart state contains no allocation attempts")
                if attempts[-1].get("completed_at") is None:
                    active_or_interrupted.append(index)
                else:
                    training_campaign._validate_retrain_progress(root, index, row)
                count = 0 if not observed else observed[-1]
            except Exception as error:
                invalid.append({"retrain_index": index, "error": str(error)})
        boundaries[str(count)] = boundaries.get(str(count), 0) + 1
    return {
        "campaign_root": str(root.resolve()),
        "code_commit": campaign["design"]["code_commit"],
        "expected_runs": len(rows),
        "complete_runs": len(complete),
        "pending_runs": len(rows) - len(complete),
        "epoch_boundaries": boundaries,
        "active_or_interrupted": active_or_interrupted,
        "invalid": invalid,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canary_marker(root: Path, model: str) -> Path:
    return root / "canaries" / model / "marker.json"


def _validate_canary(root: Path, model: str, marker: dict[str, Any]) -> None:
    """Authenticate a one-epoch full-validation canary for one model family."""
    frozen = load_authenticated_confirmation_campaign(root)
    configurations = [row for row in frozen["matrix"]["configurations"] if row["model"] == model]
    if not configurations:
        raise ValueError(f"Confirmation matrix has no configuration for {model}.")
    configuration = configurations[0]
    artifact_path = Path(marker.get("optimized_metric_artifact", ""))
    config_path = Path(marker.get("resolved_config", ""))
    if (
        marker.get("schema_version") != 1
        or marker.get("model") != model
        or marker.get("code_commit") != frozen["design"]["code_commit"]
        or marker.get("configuration_index") != configuration["configuration_index"]
        or marker.get("params_sha256") != configuration["params_sha256"]
        or not artifact_path.is_file()
        or not config_path.is_file()
        or _sha256(artifact_path) != marker.get("optimized_metric_artifact_sha256")
        or _sha256(config_path) != marker.get("resolved_config_sha256")
    ):
        raise RuntimeError(f"Confirmation canary evidence differs for {model}.")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    objectives = [row["id"] for row in training_campaign._logical_objectives(model)]
    if artifact.get("objective_order") != objectives or set(artifact.get("selections", {})) != set(
        objectives
    ):
        raise RuntimeError(f"Confirmation canary objective coverage differs for {model}.")
    if any(
        not isinstance(selection.get("optimized_metric"), list)
        or len(selection["optimized_metric"]) != 2
        or not all(math.isfinite(float(value)) for value in selection["optimized_metric"])
        for selection in artifact["selections"].values()
    ):
        raise RuntimeError(f"Confirmation canary emitted nonfinite selections for {model}.")
    threshold_rows = training_campaign._audit_selected_threshold_checkpoints(artifact_path, model)
    if threshold_rows != marker.get("threshold_checkpoint_audit"):
        raise RuntimeError(f"Confirmation canary threshold evidence differs for {model}.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "trainer.min_epochs": 1,
        "trainer.max_epochs": 1,
        "trainer.limit_train_batches": 2,
        "trainer.num_sanity_val_steps": 0,
        "train": True,
        "test": False,
        "skip_evaluation": False,
        "physics_selection.cap_metric_config.n_epochs": 1,
        "seed": int(frozen["matrix"]["training_seeds"][0]),
        **configuration["params"],
    }
    for key, expected in expected_config.items():
        observed = training_campaign._config_value(config, key)
        if key in configuration["params"]:
            expected = training_campaign._resolved_parameter_value(expected)
        if observed != expected:
            raise RuntimeError(
                f"Confirmation canary config differs for {model}/{key}: "
                f"{observed!r} != {expected!r}."
            )


def run_canary(root: Path, repository: Path, model: str) -> Path:
    """Run two train batches and the complete background validation callback suite."""
    frozen = load_authenticated_confirmation_campaign(root)
    operations._authenticate_execution_repository(repository.resolve(), frozen["design"])
    marker_path = _canary_marker(root, model)
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        _validate_canary(root, model, marker)
        return marker_path
    configurations = [row for row in frozen["matrix"]["configurations"] if row["model"] == model]
    if not configurations:
        raise ValueError(f"Confirmation matrix has no configuration for {model}.")
    configuration = configurations[0]
    canary_root = root / "canaries" / model
    hydra_dir = canary_root / "hydra"
    checkpoint_dir = canary_root / "checkpoints"
    experiment = f"physics_bgpair_confirmation_canary_{model}"
    run_name = f"{model}__trial{int(configuration['trial_number']):04d}"
    command = [
        sys.executable,
        "src/train.py",
        f"experiment=physics/{model}_background_all",
        "logger=none",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=2",
        "trainer.num_sanity_val_steps=0",
        "train=true",
        "test=false",
        "skip_evaluation=false",
        "++physics_selection.cap_metric_config.n_epochs=1",
        f"seed={frozen['matrix']['training_seeds'][0]}",
        f"experiment_name={experiment}",
        f"run_name={run_name}",
        f"hydra.run.dir={hydra_dir}",
        f"paths.checkpoints_dir={checkpoint_dir}",
    ]
    command.extend(
        f"{key}={training_campaign._hydra_value(value)}"
        for key, value in configuration["params"].items()
    )
    subprocess.run(command, cwd=repository, check=True)  # nosec B603
    artifact_path = hydra_dir / "optimized_metric.json"
    config_path = hydra_dir / ".hydra/config.yaml"
    if not artifact_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"Confirmation canary did not emit its artifacts for {model}.")
    marker = {
        "schema_version": 1,
        "model": model,
        "code_commit": frozen["design"]["code_commit"],
        "configuration_index": int(configuration["configuration_index"]),
        "trial_number": int(configuration["trial_number"]),
        "params_sha256": configuration["params_sha256"],
        "command": command,
        "optimized_metric_artifact": str(artifact_path.resolve()),
        "optimized_metric_artifact_sha256": _sha256(artifact_path),
        "resolved_config": str(config_path.resolve()),
        "resolved_config_sha256": _sha256(config_path),
        "threshold_checkpoint_audit": training_campaign._audit_selected_threshold_checkpoints(
            artifact_path, model
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    training_campaign._write_json(marker_path, marker, create=True)
    _validate_canary(root, model, marker)
    return marker_path


def canary_status(root: Path) -> dict[str, Any]:
    """Return exact authenticated canary coverage without modifying artifacts."""
    complete, invalid = [], []
    for model in training_campaign.MODELS:
        marker_path = _canary_marker(root, model)
        if not marker_path.is_file():
            continue
        try:
            _validate_canary(root, model, json.loads(marker_path.read_text(encoding="utf-8")))
        except Exception as error:
            invalid.append({"model": model, "error": str(error)})
        else:
            complete.append(model)
    return {
        "complete": complete,
        "missing": sorted(set(training_campaign.MODELS) - set(complete)),
        "invalid": invalid,
    }


def submit_canaries(
    root: Path,
    repository: Path,
    *,
    data_dir: Path,
    raw_data_dir: Path,
    pairing_dir: Path,
    concurrency: int,
) -> dict[str, Any] | None:
    """Submit only missing model canaries as one bounded Slurm array."""
    if concurrency <= 0:
        raise ValueError("Canary concurrency must be positive.")
    frozen = load_authenticated_confirmation_campaign(root)
    operations._authenticate_execution_repository(repository.resolve(), frozen["design"])
    current = canary_status(root)
    if current["invalid"]:
        raise RuntimeError(f"Invalid canary artifacts must be audited: {current['invalid']}")
    indices = [
        index
        for index, model in enumerate(training_campaign.MODELS)
        if model in current["missing"]
    ]
    if not indices:
        return None
    for path in (data_dir, raw_data_dir, pairing_dir):
        if not path.resolve().is_dir():
            raise FileNotFoundError(path)
    squeue = shutil.which("squeue")
    sbatch = shutil.which("sbatch")
    if squeue is None or sbatch is None:
        raise FileNotFoundError("squeue/sbatch")
    live = subprocess.run(  # nosec B603 -- fixed read-only scheduler query
        [
            squeue,
            "--noheader",
            "--me",
            "--name=phys-bgpair-confirm-canary",
            "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED",
            "--format=%A",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if live:
        raise RuntimeError(f"Confirmation canary allocation is already live: {live}")
    script = repository / "scripts/clariden/physics_background_pairing_confirmation_canary.sbatch"
    exports = {
        "PHYSICS_BGPAIR_REPOSITORY_ROOT": repository.resolve(),
        "PHYSICS_BGPAIR_CAMPAIGN_ROOT": root.resolve(),
        "DATA_DIR": data_dir.resolve(),
        "RAW_DATA_DIR": raw_data_dir.resolve(),
        "PHYSICS_PAIRING_DIR": pairing_dir.resolve(),
    }
    export_value = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    result = subprocess.run(  # nosec B603 -- reviewed sbatch command and frozen script
        [
            sbatch,
            "--parsable",
            f"--array={','.join(map(str, indices))}%{min(concurrency, len(indices))}",
            f"--export={export_value}",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected canary sbatch result: {result.stdout!r}")
    receipt = {
        "schema_version": 1,
        "job_id": job_id,
        "code_commit": frozen["design"]["code_commit"],
        "array_indices": indices,
        "models": [training_campaign.MODELS[index] for index in indices],
        "concurrency": min(concurrency, len(indices)),
        "script": str(script.resolve()),
        "exports": {key: str(value) for key, value in exports.items()},
    }
    receipt_path = root / "canaries/submissions" / f"{job_id}.json"
    training_campaign._write_json(receipt_path, receipt, create=True)
    return receipt


def submit_restart_canary(root: Path, repository: Path, canary_root: Path) -> dict[str, Any]:
    """Submit the forced scheduler-requeue canary for the frozen execution commit."""
    frozen = load_authenticated_confirmation_campaign(root)
    operations._authenticate_execution_repository(repository.resolve(), frozen["design"])
    canary_root = canary_root.expanduser().resolve()
    if (
        not canary_root.name.startswith("retrain_restart_canary_")
        or "campaigns" in canary_root.parts
    ):
        raise ValueError(
            "Restart canary root must start with 'retrain_restart_canary_' and remain "
            "outside every campaign directory."
        )
    if canary_root.exists():
        raise FileExistsError(f"Restart canary root must be fresh: {canary_root}")
    squeue = shutil.which("squeue")
    sbatch = shutil.which("sbatch")
    if squeue is None or sbatch is None:
        raise FileNotFoundError("squeue/sbatch")
    live = subprocess.run(  # nosec B603 -- fixed read-only scheduler query
        [
            squeue,
            "--noheader",
            "--me",
            "--name=phys-bgpair-restart-canary",
            "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED",
            "--format=%A",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if live:
        raise RuntimeError(f"Restart canary allocation is already live: {live}")
    script = repository / "scripts/clariden/physics_background_pairing_restart_canary.sbatch"
    exports = {
        "PHYSICS_BGPAIR_REPOSITORY_ROOT": repository.resolve(),
        "PHYSICS_BGPAIR_RESTART_CANARY_ROOT": canary_root,
        "PHYSICS_BGPAIR_EXPECTED_COMMIT": frozen["design"]["code_commit"],
    }
    export_value = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    result = subprocess.run(  # nosec B603 -- reviewed sbatch command and frozen script
        [sbatch, "--parsable", f"--export={export_value}", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected restart-canary sbatch result: {result.stdout!r}")
    receipt = {
        "schema_version": 1,
        "job_id": job_id,
        "code_commit": frozen["design"]["code_commit"],
        "canary_root": str(canary_root),
        "script": str(script.resolve()),
    }
    path = root / "state/restart_canary_submissions" / f"{job_id}.json"
    training_campaign._write_json(path, receipt, create=True)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execution-repository", type=Path, default=REPOSITORY_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--discovery-root", type=Path, default=DEFAULT_DISCOVERY_ROOT)
    commands.add_parser("status")
    extension = commands.add_parser("freeze-reserve-extension")
    extension.add_argument("--decisions", type=Path, required=True)
    extension.add_argument("--analysis-manifest", type=Path, required=True)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--audit-sealed", action="store_true")
    submit = commands.add_parser("submit")
    submit.add_argument("--concurrency", type=int, default=24)
    submit.add_argument("--data-dir", type=Path, required=True)
    submit.add_argument("--raw-data-dir", type=Path, required=True)
    submit.add_argument("--pairing-dir", type=Path, required=True)
    canary = commands.add_parser("run-canary")
    canary.add_argument("--model", choices=training_campaign.MODELS, required=True)
    commands.add_parser("canary-status")
    submit_canary = commands.add_parser("submit-canaries")
    submit_canary.add_argument("--concurrency", type=int, default=6)
    submit_canary.add_argument("--data-dir", type=Path, required=True)
    submit_canary.add_argument("--raw-data-dir", type=Path, required=True)
    submit_canary.add_argument("--pairing-dir", type=Path, required=True)
    restart_canary = commands.add_parser("submit-restart-canary")
    restart_canary.add_argument("--canary-root", type=Path, required=True)
    validate_restart = commands.add_parser("validate-restart-canary")
    validate_restart.add_argument("--canary-root", type=Path, required=True)
    validate_restart.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    repository = args.execution_repository.expanduser().resolve()
    if args.command == "prepare":
        result: Any = prepare(root, args.discovery_root, repository)
    elif args.command == "status":
        result = status(root, repository)
    elif args.command == "freeze-reserve-extension":
        result = {
            "reserve_extension": str(
                freeze_reserve_extension_manifest(
                    root, args.decisions, args.analysis_manifest, repository
                ).resolve()
            )
        }
    elif args.command == "reconcile":
        frozen = load_authenticated_confirmation_campaign(root)
        operations._authenticate_execution_repository(repository, frozen["design"])
        result = operations.reconcile(root, repository, args.audit_sealed)
    elif args.command == "run-canary":
        result = {"marker": str(run_canary(root, repository, args.model))}
    elif args.command == "canary-status":
        result = canary_status(root)
    elif args.command == "submit-canaries":
        result = submit_canaries(
            root,
            repository,
            data_dir=args.data_dir.expanduser().resolve(),
            raw_data_dir=args.raw_data_dir.expanduser().resolve(),
            pairing_dir=args.pairing_dir.expanduser().resolve(),
            concurrency=args.concurrency,
        )
    elif args.command == "submit-restart-canary":
        result = submit_restart_canary(root, repository, args.canary_root)
    elif args.command == "validate-restart-canary":
        result = {
            "validation": str(
                operations.validate_restart_canary(root, repository, args.canary_root, args.job_id)
            )
        }
    else:
        canaries = canary_status(root)
        if canaries["missing"] or canaries["invalid"]:
            raise RuntimeError(f"All six confirmation canaries must pass: {canaries}")
        operations._require_restart_canary_gate(
            root, load_authenticated_confirmation_campaign(root)["design"]
        )
        load_authenticated_confirmation_campaign(root)
        result = operations.submit_retrains(
            root,
            repository,
            args.concurrency,
            args.data_dir.expanduser().resolve(),
            args.raw_data_dir.expanduser().resolve(),
            args.pairing_dir.expanduser().resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
