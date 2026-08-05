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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
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


def _accounting(job_ids: list[str]) -> dict[str, dict[str, str]]:
    """Read allocation-level accounting for exact registered job IDs."""
    if not job_ids:
        return {}
    sacct = shutil.which("sacct")
    if sacct is None:
        raise FileNotFoundError("sacct")
    result = subprocess.run(  # nosec B603 -- fixed sacct fields
        [
            sacct,
            "-X",
            "--json",
            "--jobs=" + ",".join(job_ids),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    observed: dict[str, dict[str, str]] = {}
    for job in json.loads(result.stdout).get("jobs", []):
        job_id = str(job.get("job_id", ""))
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
    elif args.command == "reconcile":
        print(json.dumps(reconcile(root, repository), indent=2, sort_keys=True))
    else:
        print(backup_study(root, repository, args.model))


if __name__ == "__main__":
    main()
