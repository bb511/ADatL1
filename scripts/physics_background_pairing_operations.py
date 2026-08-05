#!/usr/bin/env python3
"""Submit and authenticate operational jobs for a frozen pairing campaign.

This control-plane utility is deliberately separate from the commit-pinned scientific
runner.  It may evolve while a campaign is active, but every submitted job executes the
repository and commit named by ``design.json``.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess  # nosec B404 -- fixed git, Slurm, and sbatch commands
import sys
import tempfile
from contextlib import contextmanager
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
    repair = commands.add_parser("repair-stale")
    repair.add_argument("--model", choices=MODELS, required=True)
    repair.add_argument("--trial", type=int, action="append", required=True)
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace one machine-readable operational artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _operation_lock(root: Path):
    """Serialize scheduler checks, submissions, reconciliation, and repair."""
    path = root / "state" / ".operations.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another campaign operation holds {path}.") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _live_job_ids(name: str) -> set[str]:
    """Return user-scoped live Slurm allocation IDs for one exact job name."""
    squeue = shutil.which("squeue")
    if squeue is None:
        raise FileNotFoundError("squeue")
    result = subprocess.run(  # nosec B603 -- fixed read-only Slurm query
        [
            squeue,
            "--noheader",
            "--me",
            f"--name={name}",
            "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED",
            "--format=%A",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    job_ids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if any(not job_id.isdigit() for job_id in job_ids):
        raise RuntimeError(f"Unexpected live {name} job IDs: {sorted(job_ids)}.")
    return job_ids


def _live_search_job_ids() -> set[str]:
    """Return allocation IDs for live packed-search jobs directly from Slurm."""
    return _live_job_ids("phys-bgpair-sweep")


def _live_retrain_job_ids() -> set[str]:
    """Return allocation IDs for live retraining arrays directly from Slurm."""
    return _live_job_ids("phys-bgpair-retrain")


def _assert_no_live_search_overlap(rows: list[dict[str, str]], model: str) -> set[str]:
    """Reject live same-model or unregistered packed-search allocations."""
    live = _live_search_job_ids()
    registered = {
        row["job_id"]: row
        for row in rows
        if row.get("kind") in {"production_canary", "search_chunk"}
    }
    unknown = live - set(registered)
    if unknown:
        raise RuntimeError(
            "Live packed-search jobs are absent from the campaign ledger: "
            f"{sorted(unknown)}. Reconcile their provenance before continuing."
        )
    overlapping = sorted(job_id for job_id in live if registered[job_id].get("model") == model)
    if overlapping:
        raise RuntimeError(f"Live {model} packed-search jobs already exist: {overlapping}.")
    return live


def _assert_no_live_retrain_overlap(rows: list[dict[str, str]]) -> set[str]:
    """Reject every live retrain allocation, including unregistered arrays."""
    live = _live_retrain_job_ids()
    registered = {
        row["job_id"].split("_", maxsplit=1)[0] for row in rows if row.get("kind") == "retrain"
    }
    unknown = live - registered
    if unknown:
        raise RuntimeError(
            "Live retrain arrays are absent from the campaign ledger: "
            f"{sorted(unknown)}. Reconcile their provenance before continuing."
        )
    if live:
        raise RuntimeError(f"Live retraining arrays already exist: {sorted(live)}.")
    return live


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
            or not row.get("success_sha256")
            or not row.get("bindings_sha256")
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
    """Serialize and submit one non-overlapping packed search chunk."""
    with _operation_lock(root):
        return _submit_search_locked(
            root,
            repository,
            model,
            trials,
            jobs,
            timeout_min,
            data_dir,
            raw_data_dir,
            pairing_dir,
        )


def _submit_search_locked(
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
    if timeout_min > 720:
        raise ValueError("Search timeout cannot exceed the 720-minute Slurm limit.")
    design = _design(root)
    _authenticate_execution_repository(repository, design)
    ledger_path, rows = _ledger(root)
    _assert_no_live_search_overlap(rows, model)
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
    """Serialize and submit only missing authenticated retrain indices."""
    with _operation_lock(root):
        return _submit_retrains_locked(
            root, repository, concurrency, data_dir, raw_data_dir, pairing_dir
        )


def _submit_retrains_locked(
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
    _assert_no_live_retrain_overlap(ledger_rows)
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
            or not row["success_sha256"]
            or not row["bindings_sha256"]
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


def _validate_completed_search_bindings(
    root: Path,
    design: dict[str, Any],
    chunk_path: Path,
    bindings_path: Path,
    binding_rows: list[dict[str, Any]],
) -> Path:
    """Authenticate one terminal chunk's trials without rescanning prior chunks."""
    from scripts import physics_background_pairing_report as report

    model = str(json.loads(chunk_path.read_text(encoding="utf-8"))["model"])
    studies = [study for study in design["studies"] if study["model"] == model]
    if len(studies) != 1:
        raise RuntimeError(f"Frozen design does not contain exactly one {model} study.")
    study = studies[0]
    expected_order = [objective["id"] for objective in study["objectives"]]
    expected_values = 2 * len(expected_order)
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    mlflow_rows = []
    selection_rows = []
    chunk_root = chunk_path.parent.resolve()

    if [int(binding.get("hydra_job_num", -1)) for binding in binding_rows] != list(
        range(len(binding_rows))
    ):
        raise RuntimeError(f"Terminal Hydra binding order is not contiguous: {bindings_path}.")
    for binding in binding_rows:
        trial_number = int(binding["trial_number"])
        key = (model, trial_number)
        values = binding.get("values")
        artifact_path = Path(str(binding.get("optimized_metric_artifact", ""))).resolve()
        if (
            binding.get("model") != model
            or binding.get("study_name") != study["study_name"]
            or binding.get("state") != "COMPLETE"
            or key in indexed
            or not isinstance(values, list)
            or len(values) != expected_values
            or not all(math.isfinite(float(value)) for value in values)
            or not artifact_path.is_file()
            or artifact_path.parent.parent != chunk_root
            or _sha256(artifact_path) != binding.get("optimized_metric_artifact_sha256")
        ):
            raise RuntimeError(f"Terminal search binding is not usable: {key}.")
        indexed[key] = binding
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        provenance = artifact.get("run_provenance", {})
        loggers = [logger for logger in provenance.get("loggers", []) if logger.get("run_id")]
        selections = artifact.get("selections", {})
        if (
            artifact.get("schema_version") != 2
            or artifact.get("objective_order") != expected_order
            or artifact.get("optimized_metric") != values
            or set(selections) != set(expected_order)
            or len(loggers) != 1
        ):
            raise RuntimeError(f"Terminal objective/provenance contract differs: {artifact_path}.")
        for objective in study["objectives"]:
            selection = selections[objective["id"]]
            pair = selection.get("optimized_metric")
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(math.isfinite(float(value)) for value in pair)
            ):
                raise RuntimeError(
                    f"Invalid selected objective pair in {artifact_path}: {objective['id']}."
                )
            selection_rows.append(
                {
                    "model": model,
                    "logical_objective": objective["id"],
                    "optimized_ckpt_name": selection.get("optimized_ckpt_name"),
                }
            )
        logger = loggers[0]
        mlflow_rows.append(
            {
                "phase": "search",
                "model": model,
                "trial_number": trial_number,
                "retrain_index": "",
                "experiment_name": provenance.get("experiment_name", ""),
                "experiment_id": logger.get("experiment_id", ""),
                "run_id": logger["run_id"],
                "run_name": provenance.get("run_name", ""),
                "hydra_run_dir": provenance.get("hydra_output_dir", ""),
                "checkpoint_run_dir": provenance.get("checkpoint_run_dir", ""),
                "artifact": str(artifact_path),
            }
        )

    report._validate_bindings(root, design, indexed)
    report._annotate_checkpoint_families(design, selection_rows)
    candidates = report._checkpoint_candidate_rows(root, design, indexed, final=False)
    expected_candidates = len(binding_rows) * (len(expected_order) + 2)
    if len(candidates) != expected_candidates or any(
        candidate["required"] and not candidate["available"] for candidate in candidates
    ):
        raise RuntimeError(
            f"Terminal retained checkpoint candidates are incomplete: {chunk_root}."
        )
    validated_mlflow = report._validate_mlflow_runs(root, design, mlflow_rows, final=True)
    if len(validated_mlflow) != len(binding_rows):
        raise RuntimeError(f"Terminal MLflow identities are incomplete: {chunk_root}.")

    output = chunk_root / "chunk_validation.json"
    _atomic_json(
        output,
        {
            "schema_version": 1,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": design["code_commit"],
            "model": model,
            "chunk": str(chunk_path.resolve()),
            "chunk_sha256": _sha256(chunk_path),
            "bindings": str(bindings_path.resolve()),
            "bindings_sha256": _sha256(bindings_path),
            "trial_numbers": sorted(trial_number for _, trial_number in indexed),
            "usable_trials": len(indexed),
            "selection_rows": len(selection_rows),
            "candidate_slots": len(candidates),
            "mlflow_runs": len(validated_mlflow),
        },
    )
    return output


def _completed_search_artifacts(
    root: Path, design: dict[str, Any], row: dict[str, str]
) -> tuple[str, str]:
    """Authenticate one successful packed controller and its final trial bindings."""
    model = row["model"]
    matches = []
    for path in sorted((root / "sweeps" / model).glob("*/chunk.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("slurm_job_id")) == row["job_id"]:
            matches.append((path, payload))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one terminal chunk manifest for job {row['job_id']}, found {len(matches)}."
        )
    chunk_path, chunk = matches[0]
    if (
        chunk.get("schema_version") != 1
        or chunk.get("model") != model
        or chunk.get("controller_status") != "complete"
        or not chunk.get("completed_at")
        or int(chunk.get("requested_trials", 0)) <= 0
    ):
        raise RuntimeError(f"Terminal chunk manifest is incomplete: {chunk_path}.")
    bindings_path = chunk_path.parent / "trial_bindings.json"
    if not bindings_path.is_file():
        raise FileNotFoundError(f"Terminal trial bindings are missing: {bindings_path}.")
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    rows = bindings.get("bindings", [])
    if (
        bindings.get("schema_version") != 1
        or bindings.get("model") != model
        or len(rows) != int(chunk["requested_trials"])
        or any(binding.get("optimized_metric_artifact") is None for binding in rows)
    ):
        raise RuntimeError(f"Terminal trial bindings are incomplete: {bindings_path}.")
    validation = _validate_completed_search_bindings(root, design, chunk_path, bindings_path, rows)
    return _sha256(validation), _sha256(bindings_path)


def _completed_retrain_marker(root: Path, design: dict[str, Any], row: dict[str, str]) -> str:
    """Authenticate the result marker for one successful Slurm array task."""
    try:
        index = int(row["job_id"].rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"Invalid retrain task job ID: {row['job_id']}.") from error
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("retrain_runs", [])
    if (
        manifest.get("schema_version") != 2
        or manifest.get("code_commit") != design["code_commit"]
        or not 0 <= index < len(runs)
        or runs[index].get("retrain_index") != index
        or runs[index].get("model") != row["model"]
    ):
        raise RuntimeError(f"Retrain task does not match the frozen manifest: {row['job_id']}.")
    marker = root / "retraining" / "markers" / f"{index:04d}.json"
    if not _validate_retrain_marker(marker, runs[index]):
        raise RuntimeError(f"Successful retrain task has no valid marker: {row['job_id']}.")
    return _sha256(marker)


def reconcile(root: Path, repository: Path) -> list[dict[str, str]]:
    """Serialize reconciliation against submissions and stale-trial repair."""
    with _operation_lock(root):
        return _reconcile_locked(root, repository)


def _reconcile_locked(root: Path, repository: Path) -> list[dict[str, str]]:
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
            if (
                state == "COMPLETED"
                and row["exit_code"] == "0:0"
                and row.get("kind") in {"production_canary", "search_chunk"}
            ):
                if not row.get("success_sha256") or not row.get("bindings_sha256"):
                    row["success_sha256"], row["bindings_sha256"] = _completed_search_artifacts(
                        root, design, row
                    )
            elif (
                state == "COMPLETED" and row["exit_code"] == "0:0" and row.get("kind") == "retrain"
            ):
                row["success_sha256"] = _completed_retrain_marker(root, design, row)
    _atomic_csv(ledger_path, rows)
    return rows


def _create_study_backup(root: Path, model: str) -> Path:
    """Create one transactionally consistent SQLite backup without policy checks."""
    source = root / "studies" / f"{model}.db"
    if not source.is_file():
        raise FileNotFoundError(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
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


def backup_study(root: Path, repository: Path, model: str) -> Path:
    """Create an authenticated, transactionally consistent SQLite backup."""
    with _operation_lock(root):
        design = _design(root)
        _authenticate_execution_repository(repository, design)
        _, rows = _ledger(root)
        if _active_model_rows(rows, model):
            raise RuntimeError(f"Refusing to back up active model {model}.")
        _assert_database_quiescent(root, model)
        return _create_study_backup(root, model)


def _append_failure_event(root: Path, payload: dict[str, Any]) -> None:
    """Append one auditable recovery event while the operation lock is held."""
    path = root / "state" / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _refresh_repaired_bindings(
    root: Path, model: str, trial_numbers: set[int]
) -> list[dict[str, str]]:
    """Persist FAIL state in any existing explicit bindings for repaired trials."""
    updated = []
    seen: set[int] = set()
    for path in sorted((root / "sweeps" / model).glob("*/trial_bindings.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for row in payload.get("bindings", []):
            trial_number = int(row["trial_number"])
            if trial_number not in trial_numbers:
                continue
            if trial_number in seen:
                raise RuntimeError(f"Duplicate explicit binding for {(model, trial_number)}.")
            seen.add(trial_number)
            row["state"] = "FAIL"
            row["values"] = None
            changed = True
        if changed:
            _atomic_json(path, payload)
            updated.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return updated


def repair_stale(
    root: Path,
    repository: Path,
    model: str,
    trial_numbers: list[int],
) -> Path:
    """Fail exact stale Optuna trials only after proving their scheduler is absent."""
    requested = sorted(set(trial_numbers))
    if (
        not requested
        or len(requested) != len(trial_numbers)
        or any(value < 0 for value in requested)
    ):
        raise ValueError("Repair trials must be explicit, unique, non-negative numbers.")
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "search",
        "model": model,
        "trial_ids": requested,
        "classification": "stale_optuna_trial_repair",
        "action": "mark_exact_stale_trials_failed",
    }
    with _operation_lock(root):
        try:
            design = _design(root)
            _authenticate_execution_repository(repository, design)
            rows = _reconcile_locked(root, repository)
            live = _assert_no_live_search_overlap(rows, model)
            active_rows = _active_model_rows(rows, model)
            if active_rows:
                raise RuntimeError(f"Active ledger jobs still exist for {model}: {active_rows}.")
            studies = [study for study in design["studies"] if study["model"] == model]
            if len(studies) != 1:
                raise RuntimeError(f"Frozen design does not contain exactly one {model} study.")
            database = root / "studies" / f"{model}.db"
            if not database.is_file():
                raise FileNotFoundError(database)
            import optuna

            study = optuna.load_study(
                study_name=studies[0]["study_name"], storage=f"sqlite:///{database}"
            )
            by_number = {int(trial.number): trial for trial in study.get_trials(deepcopy=False)}
            before = []
            for number in requested:
                trial = by_number.get(number)
                if trial is None or trial.state not in {
                    optuna.trial.TrialState.RUNNING,
                    optuna.trial.TrialState.WAITING,
                }:
                    observed = "missing" if trial is None else trial.state.name
                    raise RuntimeError(
                        f"Trial {(model, number)} is not stale RUNNING/WAITING: {observed}."
                    )
                before.append({"trial_number": number, "state": trial.state.name})
            source_sha256 = _sha256(database)
            backup = _create_study_backup(root, model)
            backup_sha256 = _sha256(backup)
            for number in requested:
                study.tell(number, state=optuna.trial.TrialState.FAIL)
            observed_after = {
                int(trial.number): trial.state.name for trial in study.get_trials(deepcopy=False)
            }
            if any(observed_after.get(number) != "FAIL" for number in requested):
                raise RuntimeError("Optuna did not persist every requested stale trial as FAIL.")
            binding_artifacts = _refresh_repaired_bindings(root, model, set(requested))
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = root / "state" / "stale_repairs" / f"{stamp}__{model}.json"
            payload = {
                "schema_version": 1,
                "repaired_at": datetime.now(timezone.utc).isoformat(),
                "code_commit": design["code_commit"],
                "model": model,
                "scheduler_proof": {
                    "live_packed_search_job_ids": sorted(live),
                    "active_model_ledger_rows": [],
                },
                "database": str(database.resolve()),
                "database_sha256_before": source_sha256,
                "database_sha256_after": _sha256(database),
                "backup": str(backup.resolve()),
                "backup_sha256": backup_sha256,
                "trials_before": before,
                "trials_after": [
                    {"trial_number": number, "state": observed_after[number]}
                    for number in requested
                ],
                "updated_binding_artifacts": binding_artifacts,
            }
            _atomic_json(output, payload)
            _append_failure_event(
                root,
                {
                    **event,
                    "outcome": "repaired",
                    "artifact": str(output.resolve()),
                    "artifact_sha256": _sha256(output),
                    "backup": str(backup.resolve()),
                    "backup_sha256": backup_sha256,
                },
            )
            return output
        except Exception as error:
            _append_failure_event(
                root,
                {**event, "outcome": "refused_or_failed", "error_signature": str(error)},
            )
            raise


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
    elif args.command == "repair-stale":
        print(repair_stale(root, repository, args.model, args.trial))
    else:
        print(backup_study(root, repository, args.model))


if __name__ == "__main__":
    main()
