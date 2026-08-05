#!/usr/bin/env python3
"""Submit and authenticate operational jobs for a frozen pairing campaign.

This control-plane utility is deliberately separate from the commit-pinned scientific
runner.  It may evolve while a campaign is active, but every submitted job executes the
repository and commit named by ``design.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess  # nosec B404 -- fixed git, Slurm, and sbatch commands
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
TARGET_TRIALS = 600
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
LEDGER_FIELDS = (
    "job_id",
    "kind",
    "model",
    "state",
    "elapsed",
    "exit_code",
    "started",
    "ended",
    "commit",
    "log_path",
    "log_sha256",
    "success_sha256",
    "bindings_sha256",
)


def _parse_args() -> argparse.Namespace:
    """Parse one control-plane operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execution-repository", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit-search")
    submit.add_argument("--model", choices=MODELS, required=True)
    submit.add_argument("--trials", type=int, default=48)
    submit.add_argument("--jobs", type=int, default=48)
    submit.add_argument("--timeout-min", type=int, default=720)
    submit.add_argument("--data-dir", type=Path, required=True)
    submit.add_argument("--raw-data-dir", type=Path, required=True)
    submit.add_argument("--pairing-dir", type=Path, required=True)
    retrain = commands.add_parser("submit-retrains")
    retrain.add_argument("--concurrency", type=int, default=24)
    retrain.add_argument("--data-dir", type=Path, required=True)
    retrain.add_argument("--raw-data-dir", type=Path, required=True)
    retrain.add_argument("--pairing-dir", type=Path, required=True)
    commands.add_parser("missing-retrains")
    commands.add_parser("validate-canaries")
    commands.add_parser("reconcile")
    backup = commands.add_parser("backup-study")
    backup.add_argument("--model", choices=MODELS, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Replace the operator ledger atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in LEDGER_FIELDS} for row in rows
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ledger(root: Path) -> tuple[Path, list[dict[str, str]]]:
    """Load and validate the single campaign job ledger."""
    path = root / "state" / "job_history.csv"
    if not path.is_file():
        return path, []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEDGER_FIELDS:
            raise RuntimeError(f"Unexpected job-ledger columns in {path}.")
        rows = list(reader)
    if len({row["job_id"] for row in rows}) != len(rows):
        raise RuntimeError("job_history.csv contains duplicate job IDs.")
    return path, rows


def _design(root: Path) -> dict[str, Any]:
    """Load the frozen schema-2 campaign design."""
    path = root / "design.json"
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("schema_version") != 2:
        raise RuntimeError("Expected a schema-2 frozen campaign design.")
    return design


def _git(repository: Path, *args: str) -> str:
    """Run a read-only Git command with a resolved executable."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git")
    return subprocess.run(  # nosec B603 -- fixed git executable and reviewed arguments
        [git, *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authenticate_execution_repository(repository: Path, design: dict[str, Any]) -> None:
    """Require the exact clean worktree named by the campaign design."""
    repository = repository.resolve()
    if _git(repository, "rev-parse", "HEAD") != design["code_commit"]:
        raise RuntimeError("Execution repository does not match the frozen campaign commit.")
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Execution repository is dirty.")


def _normalized_state(value: str) -> str:
    """Normalize decorated Slurm states to their stable base name."""
    return value.strip().split()[0].split("+")[0].upper()


def _active_model_rows(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    """Return active operator-ledger rows for one model."""
    return [
        row
        for row in rows
        if row.get("model") == model and _normalized_state(row.get("state", "")) in ACTIVE_STATES
    ]


def _active_kind_rows(rows: list[dict[str, str]], kind: str) -> list[dict[str, str]]:
    """Return active operator-ledger rows for one operational phase."""
    return [
        row
        for row in rows
        if row.get("kind") == kind and _normalized_state(row.get("state", "")) in ACTIVE_STATES
    ]


def _assert_database_quiescent(root: Path, model: str) -> None:
    """Reject a study database with an active Optuna trial."""
    database = root / "studies" / f"{model}.db"
    if not database.exists():
        return
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        states = {
            str(row[0]).upper()
            for row in connection.execute(
                "SELECT DISTINCT state FROM trials WHERE state IN ('RUNNING', 'WAITING')"
            )
        }
    if states:
        raise RuntimeError(f"Study {model} is not quiescent: {sorted(states)}.")


def _require_canary_gate(root: Path, design: dict[str, Any]) -> None:
    """Require the authenticated terminal gate before any production search chunk."""
    path = root / "state" / "production_canary_validation.json"
    if not path.is_file():
        raise RuntimeError("Production canaries have not passed the terminal validation gate.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    studies = payload.get("studies", [])
    availability = payload.get("candidate_availability", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("code_commit") != design["code_commit"]
        or payload.get("bindings") != 24
        or payload.get("checkpoint_selections") != 224
        or len(payload.get("mlflow_runs", [])) != 24
        or len(payload.get("candidate_checkpoints", [])) != 272
        or availability.get("configured_slots") != 272
        or availability.get("required_available") != 248
        or availability.get("stable_configured") != 24
        or not 0 <= availability.get("stable_available", -1) <= 24
        or not 0 <= availability.get("stable_missing", -1) <= 24
        or availability.get("stable_available", -1) + availability.get("stable_missing", -1) != 24
        or availability.get("available", -1) != 248 + availability.get("stable_available", -1)
        or len(jobs) != len(MODELS)
        or {row.get("model") for row in jobs} != set(MODELS)
        or any(
            row.get("state") != "COMPLETED"
            or row.get("exit_code") != "0:0"
            or row.get("commit") != design["code_commit"]
            or not row.get("log_sha256")
            for row in jobs
        )
        or len(studies) != len(MODELS)
        or {row.get("model") for row in studies} != set(MODELS)
        or any(
            int(row.get("total", -1)) != 4
            or int(row.get("complete", -1)) != 4
            or int(row.get("usable_complete", -1)) != 4
            or any(int(row.get(key, -1)) != 0 for key in ("running", "waiting", "fail", "pruned"))
            for row in studies
        )
    ):
        raise RuntimeError("Production-canary validation artifact is incomplete or inconsistent.")


def _usable_search_trials(root: Path, design: dict[str, Any], model: str) -> int:
    """Count complete finite full-vector trials in one quiescent persistent study."""
    import optuna

    matches = [study for study in design["studies"] if study["model"] == model]
    if len(matches) != 1:
        raise RuntimeError(f"Campaign design does not contain exactly one {model} study.")
    database = root / "studies" / f"{model}.db"
    if not database.is_file():
        return 0
    study = matches[0]
    loaded = optuna.load_study(study_name=study["study_name"], storage=f"sqlite:///{database}")
    expected_values = 2 * len(study["objectives"])
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        and trial.values is not None
        and len(trial.values) == expected_values
        and all(math.isfinite(float(value)) for value in trial.values)
        for trial in loaded.get_trials(deepcopy=False)
    )


def submit_search(
    root: Path,
    repository: Path,
    model: str,
    trials: int,
    jobs: int,
    timeout_min: int,
    data_dir: Path,
    raw_data_dir: Path,
    pairing_dir: Path,
) -> str:
    """Submit one non-overlapping packed search chunk and register it atomically."""
    if min(trials, jobs, timeout_min) <= 0 or jobs > trials:
        raise ValueError("Require positive trials/jobs/timeout and jobs <= trials.")
    design = _design(root)
    _authenticate_execution_repository(repository, design)
    ledger_path, rows = _ledger(root)
    if _active_model_rows(rows, model):
        raise RuntimeError(f"Refusing to overlap an active {model} campaign job.")
    _assert_database_quiescent(root, model)
    _require_canary_gate(root, design)
    usable = _usable_search_trials(root, design, model)
    remaining = TARGET_TRIALS - usable
    if remaining <= 0:
        raise RuntimeError(f"Study {model} already has {usable} usable trials.")
    if trials > remaining:
        raise RuntimeError(
            f"Search chunk would exceed {TARGET_TRIALS} usable trials for {model}: "
            f"{usable} complete + {trials} requested; at most {remaining} remain."
        )
    for path in (data_dir, raw_data_dir, pairing_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    script = repository / "scripts" / "clariden" / "physics_background_pairing_sweep.sbatch"
    if not script.is_file():
        raise FileNotFoundError(script)
    exports = {
        "PHYSICS_BGPAIR_REPOSITORY_ROOT": repository.resolve(),
        "PHYSICS_BGPAIR_CAMPAIGN_ROOT": root.resolve(),
        "PROJECT_ROOT": repository.resolve(),
        "DATA_DIR": data_dir.resolve(),
        "RAW_DATA_DIR": raw_data_dir.resolve(),
        "PHYSICS_PAIRING_DIR": pairing_dir.resolve(),
        "LOG_DIR": root.resolve() / "runtime" / "logs",
        "OUTPUT_DIR": root.resolve() / "runtime" / "outputs",
        "CHECKPOINT_DIR": root.resolve() / "runtime" / "checkpoints",
        "WANDB_MODE": "offline",
        "STUDY_MODEL": model,
        "CHUNK_TRIALS": trials,
        "CHUNK_JOBS": jobs,
        "TIMEOUT_MIN": timeout_min,
    }
    export_value = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise FileNotFoundError("sbatch")
    result = subprocess.run(  # nosec B603 -- reviewed sbatch command
        [
            sbatch,
            "--parsable",
            f"--nodes={math.ceil(jobs / 4)}",
            f"--export={export_value}",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch job ID: {result.stdout!r}")
    rows.append(
        {
            "job_id": job_id,
            "kind": "search_chunk",
            "model": model,
            "state": "PENDING",
            "commit": design["code_commit"],
            "log_path": str(
                Path("/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm")
                / f"phys-bgpair-sweep-{job_id}.out"
            ),
        }
    )
    _atomic_csv(ledger_path, rows)
    return job_id


def _validate_retrain_marker(marker: Path, row: dict[str, Any]) -> bool:
    """Return whether one marker completely authenticates its frozen retrain row."""
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("row") != row:
        raise RuntimeError(f"Retraining marker differs from its frozen row: {marker}")
    artifacts = [
        {
            "path": payload.get("optimized_metric_artifact"),
            "sha256": payload.get("optimized_metric_artifact_sha256"),
        },
        *payload.get("value_files", []),
    ]
    if not payload.get("value_files"):
        raise RuntimeError(f"Retraining marker contains no downstream values: {marker}")
    for artifact in artifacts:
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise RuntimeError(f"Retraining artifact is missing or changed: {path}")
    return True


def missing_retrains(root: Path, repository: Path) -> list[int]:
    """Return exact frozen retrain indices lacking an authenticated marker."""
    design = _design(root)
    _authenticate_execution_repository(repository, design)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Freeze the Pareto fronts before submitting retraining.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("code_commit") != design["code_commit"]:
        raise RuntimeError("Retrain manifest does not match the frozen campaign.")
    rows = manifest.get("retrain_runs", [])
    if [row.get("retrain_index") for row in rows] != list(range(len(rows))):
        raise RuntimeError("Retrain manifest indices are not exact and contiguous.")
    return [
        index
        for index, row in enumerate(rows)
        if not _validate_retrain_marker(root / "retraining" / "markers" / f"{index:04d}.json", row)
    ]


def _array_spec(indices: list[int]) -> str:
    """Compress sorted unique indices into one deterministic Slurm array expression."""
    if not indices or indices != sorted(set(indices)):
        raise ValueError("Array indices must be nonempty, sorted, and unique.")
    ranges: list[str] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def submit_retrains(
    root: Path,
    repository: Path,
    concurrency: int,
    data_dir: Path,
    raw_data_dir: Path,
    pairing_dir: Path,
) -> str | None:
    """Submit only missing authenticated retrain indices and ledger every array task."""
    if concurrency <= 0:
        raise ValueError("Retrain concurrency must be positive.")
    design = _design(root)
    _authenticate_execution_repository(repository, design)
    ledger_path, ledger_rows = _ledger(root)
    if _active_kind_rows(ledger_rows, "retrain"):
        raise RuntimeError("Refusing to overlap an active retraining array.")
    missing = missing_retrains(root, repository)
    if not missing:
        return None
    for path in (data_dir, raw_data_dir, pairing_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    manifest = json.loads(
        (root / "selection" / "pareto_retrain_manifest.json").read_text(encoding="utf-8")
    )
    rows_by_index = {int(row["retrain_index"]): row for row in manifest["retrain_runs"]}
    script = repository / "scripts" / "clariden" / "physics_background_pairing_retrain.sbatch"
    if not script.is_file():
        raise FileNotFoundError(script)
    exports = {
        "PHYSICS_BGPAIR_REPOSITORY_ROOT": repository.resolve(),
        "PHYSICS_BGPAIR_CAMPAIGN_ROOT": root.resolve(),
        "PROJECT_ROOT": repository.resolve(),
        "DATA_DIR": data_dir.resolve(),
        "RAW_DATA_DIR": raw_data_dir.resolve(),
        "PHYSICS_PAIRING_DIR": pairing_dir.resolve(),
        "LOG_DIR": root.resolve() / "runtime" / "logs",
        "OUTPUT_DIR": root.resolve() / "runtime" / "outputs",
        "CHECKPOINT_DIR": root.resolve() / "runtime" / "checkpoints",
        "WANDB_MODE": "offline",
    }
    export_value = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise FileNotFoundError("sbatch")
    result = subprocess.run(  # nosec B603 -- reviewed sbatch command
        [
            sbatch,
            "--parsable",
            f"--array={_array_spec(missing)}%{min(concurrency, len(missing))}",
            f"--export={export_value}",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    array_job_id = result.stdout.strip().split(";")[0]
    if not array_job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch array job ID: {result.stdout!r}")
    log_root = root.parent.parent / "logs" / "slurm"
    for index in missing:
        ledger_rows.append(
            {
                "job_id": f"{array_job_id}_{index}",
                "kind": "retrain",
                "model": str(rows_by_index[index]["model"]),
                "state": "PENDING",
                "commit": design["code_commit"],
                "log_path": str(log_root / f"phys-bgpair-retrain-{array_job_id}_{index}.out"),
            }
        )
    _atomic_csv(ledger_path, ledger_rows)
    return array_job_id


def _primary_checkpoint_family(objective: dict[str, Any]) -> str:
    """Return the exact checkpoint-family path for one logical primary metric."""
    score = str(objective["score"])
    metric = str(objective["metric"])
    suffix = "ZB_run396102_vs_ZB_run398183"
    if metric == "cap":
        name = f"cap_{score}_{objective['strategy']}_ema_{suffix}"
        direction = "max"
    elif metric == "wasserstein":
        name = f"w1dist_{score}_ema_{suffix}"
        direction = "min"
    elif metric == "drift":
        name = f"{score}_operational_drift_ema"
        direction = "min"
    else:  # pragma: no cover - authenticated by the design loader
        raise RuntimeError(f"Unknown logical metric: {metric}")
    return f"summary/{name}/{direction}"


def _candidate_checkpoint(path: Path, model: str) -> dict[str, Any]:
    """Authenticate score-specific thresholds in one retained candidate checkpoint."""
    import torch

    payload = torch.load(  # nosec B614 -- frozen campaign-local checkpoint
        path, map_location="cpu", weights_only=False
    )
    state = payload.get("state_dict", {})
    expected = {"thres_operational"}
    if model in {"ae", "vae"}:
        expected.add("thres_operational__residual_oas")
    observed = {str(key) for key in state if str(key).startswith("thres_operational")}
    if observed != expected:
        raise RuntimeError(f"Candidate threshold keys differ in {path}: {observed}")
    thresholds = {}
    for key in sorted(expected):
        value = torch.as_tensor(state[key]).detach().cpu()
        if value.numel() != 1 or not torch.isfinite(value).item():
            raise RuntimeError(f"Candidate threshold is not finite in {path}: {key}")
        thresholds[key] = float(value.item())
    return {"path": str(path.resolve()), "sha256": _sha256(path), "thresholds": thresholds}


def validate_canaries(root: Path, repository: Path) -> Path:
    """Fail closed unless all six four-trial production canaries satisfy every gate."""
    from scripts import physics_background_pairing_report as report

    design = report._design(root)
    _authenticate_execution_repository(repository, design)
    ledger_rows = reconcile(root, repository)
    canary_jobs = [row for row in ledger_rows if row.get("kind") == "production_canary"]
    if len(canary_jobs) != len(MODELS) or {row["model"] for row in canary_jobs} != set(MODELS):
        raise RuntimeError("Canary ledger does not contain exactly one job per model.")
    for row in canary_jobs:
        if (
            row["state"] != "COMPLETED"
            or row["exit_code"] != "0:0"
            or row["commit"] != design["code_commit"]
            or not row["log_sha256"]
        ):
            raise RuntimeError(f"Production canary job is not authenticated: {row}")

    status_rows, _ = report._trial_exports(root, design)
    for row in status_rows:
        if any(
            int(row[key]) != expected
            for key, expected in (
                ("total", 4),
                ("complete", 4),
                ("usable_complete", 4),
                ("running", 0),
                ("waiting", 0),
                ("fail", 0),
                ("pruned", 0),
            )
        ):
            raise RuntimeError(f"Canary Optuna state is incomplete: {row}")
    bindings = report._binding_index(root)
    expected_bindings = {(model, index) for model in MODELS for index in range(4)}
    if set(bindings) != expected_bindings:
        raise RuntimeError("Canary bindings do not exactly cover trials 0-3 for six models.")
    report._validate_bindings(root, design, bindings)
    checkpoint_rows, mlflow_rows = report._checkpoint_rows(root, design, include_retrains=False)
    annotated = report._annotate_checkpoint_families(design, checkpoint_rows)
    if len(mlflow_rows) != 24 or len(annotated) != 224:
        raise RuntimeError("Canary MLflow/checkpoint-selection cardinality is incomplete.")
    validated_mlflow = report._validate_mlflow_runs(root, design, mlflow_rows, final=True)

    studies = {study["model"]: study for study in design["studies"]}
    candidate_rows = []
    for (model, trial_number), binding in sorted(bindings.items()):
        hydra_dir = Path(binding["hydra_run_dir"])
        config = yaml.safe_load((hydra_dir / ".hydra" / "config.yaml").read_text(encoding="utf-8"))
        if (
            config.get("test") is not False
            or config.get("data", {}).get("validation_aux_datasets") != []
            or config.get("data", {}).get("expose_zerobias_sources") is not True
            or set(config.get("paths", {}).get("zerobias", {})) != {"ZB_run396102", "ZB_run398183"}
        ):
            raise RuntimeError(
                f"Canary search config leaks or lacks the exact E/G sources: {hydra_dir}"
            )
        artifact = json.loads(
            Path(binding["optimized_metric_artifact"]).read_text(encoding="utf-8")
        )
        checkpoint_root = Path(artifact["run_provenance"]["checkpoint_run_dir"])
        required_families = ["last"] + [
            _primary_checkpoint_family(objective) for objective in studies[model]["objectives"]
        ]
        stable_family = "single/ascore_operational/stable"
        if len({*required_families, stable_family}) != len(required_families) + 1:
            raise RuntimeError(f"Duplicate expected checkpoint family for {model}.")
        for family in required_families:
            if family == "last":
                paths = [checkpoint_root / "last.ckpt"]
            else:
                paths = sorted((checkpoint_root / family).glob("*.ckpt"))
            if len(paths) != 1 or not paths[0].is_file():
                raise RuntimeError(
                    f"Expected one retained {family} checkpoint for {(model, trial_number)}."
                )
            candidate_rows.append(
                {
                    "model": model,
                    "trial_number": trial_number,
                    "family": family,
                    "available": True,
                    **_candidate_checkpoint(paths[0], model),
                }
            )
        stable_directory = checkpoint_root / stable_family
        if not stable_directory.is_dir():
            raise RuntimeError(
                f"Stable checkpoint family is not configured for {(model, trial_number)}."
            )
        stable_paths = sorted(stable_directory.glob("*.ckpt"))
        if len(stable_paths) > 1:
            raise RuntimeError(
                f"Expected at most one retained {stable_family} checkpoint for "
                f"{(model, trial_number)}."
            )
        candidate_rows.append(
            {
                "model": model,
                "trial_number": trial_number,
                "family": stable_family,
                "available": bool(stable_paths),
                **(
                    _candidate_checkpoint(stable_paths[0], model)
                    if stable_paths
                    else {"path": "", "sha256": "", "thresholds": {}}
                ),
            }
        )

    if len(candidate_rows) != 272:
        raise RuntimeError(
            f"Canary configured-candidate cardinality is incomplete: {len(candidate_rows)} != 272."
        )

    stable_rows = [
        row for row in candidate_rows if row["family"] == "single/ascore_operational/stable"
    ]
    stable_available = sum(bool(row["available"]) for row in stable_rows)
    candidate_availability = {
        "configured_slots": len(candidate_rows),
        "available": sum(bool(row["available"]) for row in candidate_rows),
        "required_available": len(candidate_rows) - len(stable_rows),
        "stable_configured": len(stable_rows),
        "stable_available": stable_available,
        "stable_missing": len(stable_rows) - stable_available,
    }

    output = root / "state" / "production_canary_validation.json"
    payload = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": design["code_commit"],
        "jobs": canary_jobs,
        "studies": status_rows,
        "bindings": len(bindings),
        "checkpoint_selections": len(annotated),
        "mlflow_runs": validated_mlflow,
        "candidate_availability": candidate_availability,
        "candidate_checkpoints": candidate_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def _accounting(job_ids: list[str]) -> dict[str, dict[str, str]]:
    """Read allocation-level accounting for exact registered job IDs."""
    if not job_ids:
        return {}
    sacct = shutil.which("sacct")
    if sacct is None:
        raise FileNotFoundError("sacct")
    query_ids = sorted({job_id.split("_", maxsplit=1)[0] for job_id in job_ids})
    result = subprocess.run(  # nosec B603 -- fixed sacct fields
        [
            sacct,
            "-X",
            "--json",
            "--jobs=" + ",".join(query_ids),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    observed: dict[str, dict[str, str]] = {}
    for job in json.loads(result.stdout).get("jobs", []):
        job_id = str(job.get("job_id", ""))
        array = job.get("array") or {}
        task = array.get("task_id", {})
        if task.get("set"):
            job_id = f"{array.get('job_id')}_{int(task['number'])}"
        if job_id not in job_ids:
            continue
        states = job.get("state", {}).get("current", [])
        if len(states) != 1:
            raise RuntimeError(f"Unexpected Slurm state for job {job_id}: {states!r}")
        timing = job.get("time", {})
        elapsed = int(timing.get("elapsed", 0))
        return_code = int(job.get("exit_code", {}).get("return_code", {}).get("number", 0))
        signal = int(job.get("exit_code", {}).get("signal", {}).get("id", {}).get("number", 0))
        observed[job_id] = {
            "state": _normalized_state(states[0]),
            "elapsed": f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}",
            "exit_code": f"{return_code}:{signal}",
            "started": _accounting_time(timing.get("start", 0)),
            "ended": _accounting_time(timing.get("end", 0)),
        }
    return observed


def _accounting_time(value: Any) -> str:
    """Convert a Slurm epoch timestamp to an unambiguous UTC ISO timestamp."""
    seconds = int(value or 0)
    return "" if seconds <= 0 else datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def reconcile(root: Path, repository: Path) -> list[dict[str, str]]:
    """Reconcile scheduler state and seal terminal Slurm logs by SHA-256."""
    design = _design(root)
    _authenticate_execution_repository(repository, design)
    ledger_path, rows = _ledger(root)
    accounting = _accounting([row["job_id"] for row in rows])
    for row in rows:
        if row["job_id"] not in accounting:
            continue
        row.update(accounting[row["job_id"]])
        state = row["state"]
        if state not in ACTIVE_STATES and state not in TERMINAL_STATES:
            raise RuntimeError(f"Unknown scheduler state {state!r} for job {row['job_id']}.")
        if state in TERMINAL_STATES:
            log_path = Path(row["log_path"])
            if not log_path.is_file():
                raise FileNotFoundError(f"Terminal job log is missing: {log_path}")
            row["log_sha256"] = _sha256(log_path)
    _atomic_csv(ledger_path, rows)
    return rows


def backup_study(root: Path, repository: Path, model: str) -> Path:
    """Create an authenticated, transactionally consistent SQLite backup."""
    design = _design(root)
    _authenticate_execution_repository(repository, design)
    _, rows = _ledger(root)
    if _active_model_rows(rows, model):
        raise RuntimeError(f"Refusing to back up active model {model}.")
    _assert_database_quiescent(root, model)
    source = root / "studies" / f"{model}.db"
    if not source.is_file():
        raise FileNotFoundError(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "studies" / "backups" / model / f"{stamp}.db"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".db.tmp")
    with sqlite3.connect(source) as source_connection, sqlite3.connect(
        temporary
    ) as destination_connection:
        source_connection.backup(destination_connection)
    temporary.replace(destination)
    digest = _sha256(destination)
    manifest = destination.with_suffix(".sha256")
    manifest.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return destination


def main() -> None:
    """Dispatch submission, reconciliation, or a quiescent database backup."""
    args = _parse_args()
    root = args.root.expanduser().resolve()
    repository = args.execution_repository.expanduser().resolve()
    if args.command == "submit-search":
        print(
            submit_search(
                root,
                repository,
                args.model,
                args.trials,
                args.jobs,
                args.timeout_min,
                args.data_dir,
                args.raw_data_dir,
                args.pairing_dir,
            )
        )
    elif args.command == "submit-retrains":
        print(
            submit_retrains(
                root,
                repository,
                args.concurrency,
                args.data_dir,
                args.raw_data_dir,
                args.pairing_dir,
            )
        )
    elif args.command == "missing-retrains":
        print(json.dumps(missing_retrains(root, repository)))
    elif args.command == "validate-canaries":
        print(validate_canaries(root, repository))
    elif args.command == "reconcile":
        print(json.dumps(reconcile(root, repository), indent=2, sort_keys=True))
    else:
        print(backup_study(root, repository, args.model))


if __name__ == "__main__":
    main()
