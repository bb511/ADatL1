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
import re
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

from src.utils.pairing.physics_tables import load_split
from src.utils.pairing.table import load_pair_table, sha256_tensor

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
TARGET_TRIALS = 600
NSGA_GENERATION_KEY = "nsga2:generation"
NSGAII_SCREENING_SAMPLER = {
    "name": "nsgaii",
    "_target_": "optuna.samplers.NSGAIISampler",
    "seed": 123,
    "population_size": 48,
    "mutation_prob": None,
    "crossover_prob": 0.9,
    "swapping_prob": 0.5,
    "constraints_func": None,
}
NSGAII_SEED_POLICY = {
    "name": "base_seed_plus_model_offset_plus_starting_trial",
    "formula": "base_seed + model_offsets[model] + starting_optuna_trial_number",
    "base_seed": 123,
    "model_offsets": {model: index * 100000 for index, model in enumerate(MODELS)},
    "population_size": 48,
    "generation_size": 48,
    "final_partial_population": 24,
    "target_trials": TARGET_TRIALS,
    "require_pristine_study": True,
    "canary_studies_are_separate": True,
}
MAX_RETRAIN_ARRAY_TASKS = 1001
RESTART_CANARY_SCHEMA_VERSION = 2
RESTART_CANARY_EPOCHS = 6
RESTART_CANARY_EXPERIMENT = "physics_bgpair_restart_canary"
RESTART_CANARY_RUN_NAME = "slurm_hpc_resume"
ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
}
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
    "attempt",
)
LEGACY_LEDGER_FIELDS = LEDGER_FIELDS[:-1]


def _parse_args() -> argparse.Namespace:
    """Parse one control-plane operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execution-repository", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    canary = commands.add_parser("submit-canary")
    canary.add_argument("--model", choices=MODELS, required=True)
    canary.add_argument("--data-dir", type=Path, required=True)
    canary.add_argument("--raw-data-dir", type=Path, required=True)
    canary.add_argument("--pairing-dir", type=Path, required=True)
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
    restart = commands.add_parser("validate-restart-canary")
    restart.add_argument("--canary-root", type=Path, required=True)
    restart.add_argument("--job-id", required=True)
    commands.add_parser("reconcile")
    retire = commands.add_parser("retire-unmaterialized-retrains")
    retire.add_argument("--array-job-id", required=True)
    candidate_audit = commands.add_parser("audit-retrain-candidates")
    candidate_audit.add_argument("--incident-id", required=True)
    candidate_reset = commands.add_parser("reset-affected-retrains")
    candidate_reset.add_argument("--incident-id", required=True)
    candidate_reset.add_argument("--audit-sha256", required=True)
    commands.add_parser("freeze")
    commands.add_parser("aggregate")
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


def _json_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using its canonical representation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    """Create one immutable JSON receipt, accepting only identical repetition."""
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Immutable receipt already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(encoded, encoding="utf-8")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
                raise RuntimeError(
                    f"Immutable receipt was concurrently created with different content: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


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
        fields = tuple(reader.fieldnames or ())
        if fields not in {LEDGER_FIELDS, LEGACY_LEDGER_FIELDS}:
            raise RuntimeError(f"Unexpected job-ledger columns in {path}.")
        rows = list(reader)
    if fields == LEGACY_LEDGER_FIELDS:
        for row in rows:
            row["attempt"] = ""
    if len({row["job_id"] for row in rows}) != len(rows):
        raise RuntimeError("job_history.csv contains duplicate job IDs.")
    return path, rows


def _design(root: Path) -> dict[str, Any]:
    """Load the frozen schema-2 campaign design."""
    path = root / "design.json"
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("schema_version") != 2:
        raise RuntimeError("Expected a schema-2 frozen campaign design.")
    sampler = design.get("screening_sampler")
    if sampler is not None and sampler != NSGAII_SCREENING_SAMPLER:
        raise RuntimeError("Campaign screening sampler differs from the NSGA-II contract.")
    if sampler is not None and any(
        study.get("hparams_search") != f"{study.get('model')}_shared_nsgaii"
        for study in design.get("studies", [])
    ):
        raise RuntimeError("Campaign HPO config differs from the NSGA-II contract.")
    if sampler is not None and design.get("screening_sampler_seed_policy") != NSGAII_SEED_POLICY:
        raise RuntimeError("Campaign sampler seed policy differs from the NSGA-II contract.")
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
    if design.get("campaign_kind") == "physics_background_native_confirmation":
        root = Path(str(design["runner_manifest"])).resolve().parents[1]
        extension_path = root / "selection/reserve_extension_manifest.json"
        if extension_path.is_file():
            from src.evaluation.physics_background_pairing_confirmation_matrix import (
                authenticate_reserve_execution_repository,
                load_authenticated_confirmation_campaign,
            )

            frozen = load_authenticated_confirmation_campaign(root)
            if frozen["design"] != design or frozen["reserve_extension"] is None:
                raise RuntimeError("Reserve execution identity is detached from the campaign.")
            authenticate_reserve_execution_repository(repository, frozen["reserve_extension"])
            return
    if _git(repository, "rev-parse", "HEAD") != design["code_commit"]:
        raise RuntimeError("Execution repository does not match the frozen campaign commit.")
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Execution repository is dirty.")


def _cache_root_for_submission(data_dir: Path, frozen_cache_root: Path) -> Path:
    """Resolve the configured robust cache below one supplied base data directory."""
    resolved = data_dir.expanduser().resolve(strict=True)
    frozen_parts = frozen_cache_root.parts
    try:
        marker_index = frozen_parts.index("data_2025E+G")
    except ValueError as error:
        raise RuntimeError(
            f"Frozen preflight cache has no data_2025E+G anchor: {frozen_cache_root}."
        ) from error
    suffix = Path(*frozen_parts[marker_index:])
    candidates = [resolved / suffix]
    if resolved.name == "data_2025E+G":
        candidates.append(resolved / Path(*frozen_parts[marker_index + 1 :]))
    candidates.append(resolved)
    matches = []
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if all((candidate / split / "torch_cache.pt").is_file() for split in ("valid", "test")):
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            "DATA_DIR must resolve unambiguously to the frozen robust cache suffix; "
            f"found {len(matches)} candidates below {resolved}: {matches}."
        )
    return matches[0]


def _expected_source_hashes(preflight: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Recover the one frozen ordered source-prefix identity for each split."""
    expected: dict[str, tuple[str, str]] = {}
    for row in preflight.get("tables", []):
        stage = str(row.get("stage"))
        identity = (str(row.get("source_1_sha256")), str(row.get("source_2_sha256")))
        if stage in expected and expected[stage] != identity:
            raise RuntimeError(f"Frozen preflight has inconsistent {stage} source identities.")
        expected[stage] = identity
    if set(expected) != {"validate", "test"}:
        raise RuntimeError("Frozen preflight must bind validation and test source identities.")
    return expected


def _authenticate_confirmation_submission_data(
    root: Path,
    design: dict[str, Any],
    data_dir: Path,
    raw_data_dir: Path,
    pairing_dir: Path,
) -> dict[str, Any] | None:
    """Authenticate and persist the data identity before a confirmation sbatch.

    Paths may move, but the ordered source prefixes and pair-table bytes may not.  The raw-data
    root is not read by an already materialized retrain and therefore remains path-bound.
    """
    import torch

    if design.get("campaign_kind") != "physics_background_native_confirmation":
        return None
    preflight_value = design.get("preflight")
    expected_preflight_sha256 = str(design.get("preflight_sha256", ""))
    if (
        not isinstance(preflight_value, str)
        or not preflight_value
        or len(expected_preflight_sha256) != 64
    ):
        raise RuntimeError("Frozen confirmation design has no authenticated preflight identity.")
    preflight_path = Path(preflight_value).expanduser().resolve(strict=True)
    if _sha256(preflight_path) != expected_preflight_sha256:
        raise RuntimeError("Frozen confirmation preflight changed before submission.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("schema_version") != 1:
        raise RuntimeError("Expected a schema-1 physics pairing preflight.")

    resolved_pairing = pairing_dir.expanduser().resolve(strict=True)
    resolved_raw = raw_data_dir.expanduser().resolve(strict=True)
    cache_root = _cache_root_for_submission(
        data_dir, Path(str(preflight["cache_root"])).expanduser()
    )
    source_root = resolved_pairing / "sources"
    source_files = [
        source_root / "zerobias_sources.json",
        source_root / "valid/zerobias_source_id.npy",
        source_root / "test/zerobias_source_id.npy",
    ]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    events = int(preflight["events_per_source"])
    datasets = (str(preflight["dataset_1"]), str(preflight["dataset_2"]))
    expected_sources = _expected_source_hashes(preflight)
    observed_sources: dict[str, dict[str, str]] = {}
    for stage in ("validate", "test"):
        digests = []
        for dataset in datasets:
            inputs, _, _ = load_split(
                cache_root,
                stage,
                dataset=dataset,
                limit=events,
                source_metadata_dir=source_root,
            )
            digests.append(sha256_tensor(torch.flatten(inputs, start_dim=1).contiguous()))
        if tuple(digests) != expected_sources[stage]:
            raise RuntimeError(
                f"DATA_DIR ordered {stage} background prefixes do not match the frozen "
                f"preflight: expected {expected_sources[stage]}, observed {tuple(digests)}."
            )
        observed_sources[stage] = {
            "source_1_sha256": digests[0],
            "source_2_sha256": digests[1],
        }

    table_identities = []
    for frozen_row in preflight["tables"]:
        table_path = resolved_pairing / Path(str(frozen_row["path"])).name
        if _sha256(table_path) != frozen_row["sha256"]:
            raise RuntimeError(f"Pair table differs from the frozen preflight: {table_path}.")
        stage = str(frozen_row["stage"])
        load_pair_table(
            table_path,
            expected_dataset_1=datasets[0],
            expected_dataset_2=datasets[1],
            expected_split=stage,
            n_dataset_1=events,
            n_dataset_2=events,
            source_1_sha256=observed_sources[stage]["source_1_sha256"],
            source_2_sha256=observed_sources[stage]["source_2_sha256"],
        )
        table_identities.append({"name": table_path.name, "sha256": str(frozen_row["sha256"])})

    content_identity = {
        "preflight_sha256": expected_preflight_sha256,
        "events_per_source": events,
        "datasets": list(datasets),
        "ordered_source_prefixes": observed_sources,
        "source_sidecars": {
            str(path.relative_to(source_root)): _sha256(path) for path in source_files
        },
        "pair_tables": sorted(table_identities, key=lambda row: row["name"]),
    }
    content_sha256 = _json_sha256(content_identity)
    binding_path = root / "state/submission_data_identity.json"
    supplied_paths = {
        "data_dir": str(data_dir.expanduser().resolve()),
        "cache_root": str(cache_root),
        "raw_data_dir": str(resolved_raw),
        "pairing_dir": str(resolved_pairing),
    }
    if binding_path.is_file():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            binding.get("schema_version") != 1
            or binding.get("campaign_design_sha256") != str(design.get("content_sha256", ""))
            or binding.get("content_identity_sha256") != content_sha256
            or binding.get("content_identity") != content_identity
        ):
            raise RuntimeError("Confirmation submission data content differs from its binding.")
        if binding.get("raw_data_dir") != str(resolved_raw):
            raise RuntimeError("RAW_DATA_DIR differs from the path frozen at first submission.")
        accepted_paths = list(binding.get("accepted_paths", []))
        if supplied_paths not in accepted_paths:
            accepted_paths.append(supplied_paths)
        binding["accepted_paths"] = accepted_paths
    else:
        binding = {
            "schema_version": 1,
            "campaign_design_sha256": str(design.get("content_sha256", "")),
            "content_identity": content_identity,
            "content_identity_sha256": content_sha256,
            "raw_data_dir": str(resolved_raw),
            "accepted_paths": [supplied_paths],
        }
    binding["last_authenticated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(binding_path, binding)
    return binding


def _path_within(path: Path, parent: Path) -> bool:
    """Return whether one resolved path is confined beneath a resolved directory."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _restart_canary_accounting(job_id: str) -> dict[str, Any]:
    """Return the unique allocation-level Slurm record for an isolated canary."""
    if not job_id.isdigit():
        raise ValueError("Restart-canary job ID must be numeric.")
    sacct = shutil.which("sacct")
    if sacct is None:
        raise FileNotFoundError("sacct")
    result = subprocess.run(  # nosec B603 -- fixed read-only Slurm query
        [sacct, "-X", "--json", f"--jobs={job_id}"],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = [
        row
        for row in json.loads(result.stdout).get("jobs", [])
        if str(row.get("job_id", "")) == job_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one allocation-level accounting row for restart canary {job_id}."
        )
    row = matches[0]
    states = row.get("state", {}).get("current", [])
    exit_code = row.get("exit_code", {})
    return_code = exit_code.get("return_code", {})
    signal = exit_code.get("signal", {}).get("id", {})
    if len(states) != 1:
        raise RuntimeError(f"Unexpected restart-canary scheduler state: {states!r}.")
    return {
        "job_id": job_id,
        "job_name": str(row.get("name", "")),
        "state": _normalized_state(str(states[0])),
        "exit_code": (f"{int(return_code.get('number', 0))}:" f"{int(signal.get('number', 0))}"),
        "restart_count": int(row.get("restart_cnt", 0)),
        "stdout": str(row.get("stdout_expanded", "")),
    }


def _restart_canary_metric_history(client, run_id: str, name: str) -> list[dict[str, Any]]:
    """Export one MLflow history without hiding duplicates behind set operations."""
    return [
        {"step": int(item.step), "value": float(item.value), "timestamp": int(item.timestamp)}
        for item in sorted(client.get_metric_history(run_id, name), key=lambda item: item.step)
    ]


def _restart_canary_jsonable(value: Any) -> Any:
    """Normalize checkpoint state exactly as the canary's JSON restore evidence."""
    import torch

    if isinstance(value, dict):
        return {str(key): _restart_canary_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_restart_canary_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _restart_canary_expected_tags(root: Path, code_commit: str) -> dict[str, str]:
    """Return the independently specified immutable MLflow identity contract."""
    identity = {
        "schema_version": RESTART_CANARY_SCHEMA_VERSION,
        "root": str(root.resolve()),
        "commit": code_commit,
        "experiment_name": RESTART_CANARY_EXPERIMENT,
        "run_name": RESTART_CANARY_RUN_NAME,
    }
    return {
        "mlflow.runName": RESTART_CANARY_RUN_NAME,
        "mlflow.source.git.commit": code_commit,
        "physics_bgpair_restart_canary": "true",
        "physics_bgpair_restart_canary_identity_sha256": _json_sha256(identity),
    }


def _restart_canary_scheduler_sentinels(text: str) -> dict[str, int]:
    """Authenticate append-preserved evidence from original and resumed allocations."""
    required = (
        "CANARY_ATTEMPT_START restart=0",
        "CANARY_ATTEMPT_START restart=1",
        "CANARY_LIGHTNING_CHILD restart=0",
        "CANARY_LIGHTNING_CHILD restart=1",
        "Handling auto-requeue signal",
    )
    counts = {needle: text.count(needle) for needle in required}
    if any(count < 1 for count in counts.values()):
        raise RuntimeError(f"Restart-canary scheduler log is missing sentinels: {counts}.")
    return counts


def _validate_restart_canary_evidence(
    campaign_root: Path,
    canary_root: Path,
    job_id: str,
    code_commit: str,
) -> dict[str, Any]:
    """Independently authenticate the complete forced-restart canary contract."""
    from mlflow.tracking import MlflowClient

    campaign_root = campaign_root.resolve()
    canary_root = canary_root.expanduser().resolve()
    if (
        not canary_root.name.startswith("retrain_restart_canary_")
        or "campaigns" in canary_root.parts
    ):
        raise RuntimeError("Restart canary root does not satisfy the isolated-root contract.")
    if _path_within(canary_root, campaign_root) or _path_within(campaign_root, canary_root):
        raise RuntimeError("Restart canary and scientific campaign roots must be disjoint.")
    complete_path = canary_root / "COMPLETE.json"
    state_path = canary_root / "state.json"
    if not complete_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("Restart canary is missing COMPLETE.json or state.json.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    restart_count = int(complete.get("slurm_restart_count", -1))
    expected_tracking_uri = f"file:{(canary_root / 'mlflow' / 'mlruns').resolve()}"
    mlflow = complete.get("mlflow", {})
    if (
        complete.get("schema_version") != RESTART_CANARY_SCHEMA_VERSION
        or complete.get("code_commit") != code_commit
        or str(complete.get("slurm_job_id", "")) != job_id
        or restart_count < 1
        or complete.get("final_epoch") != RESTART_CANARY_EPOCHS
        or not complete.get("cuda_device")
        or not complete.get("completed_at")
        or complete.get("initial_state_sha256") == complete.get("final_state_sha256")
        or len(str(complete.get("initial_state_sha256", ""))) != 64
        or complete.get("uninterrupted_reference_state_sha256")
        != complete.get("final_state_sha256")
        or len(str(complete.get("final_state_sha256", ""))) != 64
        or complete.get("parameters_match_uninterrupted_reference") is not True
        or state.get("schema_version") != RESTART_CANARY_SCHEMA_VERSION
        or state.get("code_commit") != code_commit
        or state.get("mlflow_run_id") != mlflow.get("run_id")
        or state.get("tracking_uri") != expected_tracking_uri
        or mlflow.get("tracking_uri") != expected_tracking_uri
        or mlflow.get("experiment_name") != RESTART_CANARY_EXPERIMENT
    ):
        raise RuntimeError("Restart-canary completion identity is inconsistent.")

    evidence_root = canary_root / "evidence"

    def authenticated_reference(
        row: Any, *, parent: Path, expected_name: str | None = None
    ) -> Path:
        if not isinstance(row, dict):
            raise RuntimeError("Restart-canary artifact reference is not a mapping.")
        path = Path(str(row.get("path", ""))).resolve()
        if (
            not path.is_file()
            or not _path_within(path, parent)
            or (expected_name is not None and path.name != expected_name)
            or row.get("sha256") != _sha256(path)
        ):
            raise RuntimeError(f"Restart-canary artifact reference is invalid: {path}.")
        return path

    hpc_path = authenticated_reference(
        complete.get("hpc_checkpoint_evidence"),
        parent=evidence_root,
        expected_name=f"hpc_input_{restart_count:03d}.ckpt",
    )
    if complete["hpc_checkpoint_evidence"].get("resume_eligible") is not False:
        raise RuntimeError("Restart-canary HPC checkpoint was incorrectly marked resumable.")
    exact_resume_path = authenticated_reference(
        complete.get("exact_resume_checkpoint"),
        parent=evidence_root,
        expected_name=f"exact_resume_{restart_count:03d}.ckpt",
    )
    reference_path = authenticated_reference(
        complete.get("uninterrupted_reference_evidence"),
        parent=evidence_root,
        expected_name="uninterrupted_reference.json",
    )
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference_payload.get("schema_version") != RESTART_CANARY_SCHEMA_VERSION
        or reference_payload.get("code_commit") != code_commit
        or reference_payload.get("parameter_sha256")
        != complete.get("uninterrupted_reference_state_sha256")
    ):
        raise RuntimeError("Restart-canary cold-process reference evidence is invalid.")
    exact_resume = complete["exact_resume_checkpoint"]
    completed_epochs = int(exact_resume.get("completed_epochs", -1))
    expected_pre_resume_epochs = list(range(completed_epochs))
    import torch

    exact_resume_payload = torch.load(  # nosec B614 -- hash-authenticated canary artifact
        exact_resume_path, map_location="cpu", weights_only=False
    )
    checkpoint_callbacks = exact_resume_payload.get("callbacks", {})
    exact_rng = checkpoint_callbacks.get("physics-background-pairing-exact-retrain-rng-v1", {})
    numpy_rng = exact_rng.get("numpy", {})
    byte_states = [
        exact_rng.get("torch_cpu"),
        exact_rng.get("datamodule_shuffler"),
        *(
            exact_rng.get("torch_cuda", [])
            if isinstance(exact_rng.get("torch_cuda"), list)
            else []
        ),
    ]
    try:
        progress = exact_resume_payload["loops"]["fit_loop"]["epoch_progress"]["total"]
        loop_counts = [int(progress[name]) for name in ("ready", "started", "processed")]
    except (KeyError, TypeError, ValueError):
        loop_counts = []
    if (
        not 0 < completed_epochs < RESTART_CANARY_EPOCHS
        or not isinstance(exact_resume_payload, dict)
        or int(exact_resume_payload.get("epoch", -1)) + 1 != completed_epochs
        or loop_counts != [completed_epochs] * 3
        or not isinstance(checkpoint_callbacks, dict)
        or set(exact_rng)
        != {
            "schema_version",
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "datamodule_shuffler",
        }
        or exact_rng.get("schema_version") != 1
        or not isinstance(exact_rng.get("python"), tuple)
        or len(exact_rng["python"]) != 3
        or set(numpy_rng) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
        or numpy_rng.get("algorithm") != "MT19937"
        or not isinstance(numpy_rng.get("keys"), torch.Tensor)
        or numpy_rng["keys"].dtype != torch.uint32
        or numpy_rng["keys"].ndim != 1
        or numpy_rng["keys"].numel() == 0
        or not isinstance(numpy_rng.get("position"), int)
        or not 0 <= numpy_rng["position"] <= numpy_rng["keys"].numel()
        or not isinstance(numpy_rng.get("has_gauss"), int)
        or numpy_rng["has_gauss"] not in {0, 1}
        or not isinstance(numpy_rng.get("cached_gaussian"), (int, float))
        or not math.isfinite(float(numpy_rng["cached_gaussian"]))
        or not isinstance(exact_rng.get("torch_cuda"), list)
        or len(exact_rng["torch_cuda"]) != 1
        or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
            for state in byte_states
        )
        or exact_resume.get("mlflow_epoch_steps") != expected_pre_resume_epochs
        or exact_resume.get("mlflow_epoch_values")
        != [float(value) for value in expected_pre_resume_epochs]
    ):
        raise RuntimeError("Restart-canary exact completed-epoch resume evidence is invalid.")
    restore_path = authenticated_reference(
        complete.get("restore_evidence"),
        parent=evidence_root,
        expected_name=f"restore_{restart_count:03d}.json",
    )
    restore = json.loads(restore_path.read_text(encoding="utf-8"))
    callback_states = restore.get("callback_states", {})
    restored_exact_rng = (
        callback_states.get("exact_rng") if isinstance(callback_states, dict) else None
    )
    restored_retained = complete.get("restored_retained_checkpoints")
    retained = complete.get("retained_checkpoints")
    if (
        restore.get("schema_version") != RESTART_CANARY_SCHEMA_VERSION
        or restore.get("restart_count") != restart_count
        or not restore.get("restored_at")
        or not isinstance(callback_states, dict)
        or set(callback_states)
        != {"exact_rng", "efficiency", "cap", "wasserstein", "drift", "stable"}
        or restored_exact_rng != _restart_canary_jsonable(exact_rng)
        or not isinstance(restored_retained, list)
        or not restored_retained
        or restore.get("retained_checkpoints") != restored_retained
        or not isinstance(retained, list)
        or not retained
    ):
        raise RuntimeError("Restart-canary callback restoration evidence is incomplete.")
    restored_retained_paths = [
        authenticated_reference(row, parent=evidence_root) for row in restored_retained
    ]
    retained_paths = [
        authenticated_reference(row, parent=canary_root / "retained") for row in retained
    ]
    if len(set(restored_retained_paths)) != len(restored_retained_paths) or len(
        set(retained_paths)
    ) != len(retained_paths):
        raise RuntimeError("Restart-canary retained checkpoint inventory contains duplicates.")

    scheduler = complete.get("scheduler_log", {})
    source_path = Path(str(scheduler.get("source_path", ""))).resolve()
    snapshot_path = Path(str(scheduler.get("snapshot_path", ""))).resolve()
    if (
        not source_path.is_file()
        or not snapshot_path.is_file()
        or not _path_within(snapshot_path, evidence_root)
        or snapshot_path.name != "scheduler_log_snapshot.txt"
        or scheduler.get("snapshot_sha256") != _sha256(snapshot_path)
    ):
        raise RuntimeError("Restart-canary append-preserved scheduler log is invalid.")
    scheduler_text = snapshot_path.read_text(encoding="utf-8")
    terminal_scheduler_text = source_path.read_text(encoding="utf-8")
    sentinels = _restart_canary_scheduler_sentinels(scheduler_text)
    if scheduler.get("sentinels") != sentinels:
        raise RuntimeError("Restart-canary scheduler sentinel counts differ.")
    allocation_indices = range(restart_count + 1)
    if (
        any(
            terminal_scheduler_text.count(f"CANARY_ATTEMPT_START restart={index} job={job_id}")
            != 1
            or terminal_scheduler_text.count(f"CANARY_LIGHTNING_CHILD restart={index}") != 1
            for index in allocation_indices
        )
        or terminal_scheduler_text.count("Handling auto-requeue signal") != restart_count
        or terminal_scheduler_text.count(
            f"CANARY_ATTEMPT_FINISH restart={restart_count} job={job_id}"
        )
        != 1
        or f"CANARY_COMPLETE path={complete_path}" not in terminal_scheduler_text
        or scheduler_text not in terminal_scheduler_text
    ):
        raise RuntimeError("Restart-canary scheduler log lacks exact terminal sentinels.")

    accounting = _restart_canary_accounting(job_id)
    if (
        accounting["job_name"] != "phys-bgpair-restart-canary"
        or accounting["state"] != "COMPLETED"
        or accounting["exit_code"] != "0:0"
        or accounting["restart_count"] < 1
        or accounting["restart_count"] != restart_count
        or not accounting["stdout"]
        or Path(accounting["stdout"]).resolve() != source_path
    ):
        raise RuntimeError(f"Restart-canary scheduler accounting did not pass: {accounting}.")

    run_id = str(mlflow.get("run_id", ""))
    client = MlflowClient(tracking_uri=expected_tracking_uri)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    expected_tags = _restart_canary_expected_tags(canary_root, code_commit)
    if (
        experiment is None
        or experiment.name != RESTART_CANARY_EXPERIMENT
        or run.info.status != "FINISHED"
        or any(run.data.tags.get(key) != value for key, value in expected_tags.items())
        or mlflow.get("tags") != expected_tags
    ):
        raise RuntimeError("Restart-canary MLflow run identity or tags differ.")
    epoch_history = _restart_canary_metric_history(client, run_id, "epoch_idx")
    allocation_history = _restart_canary_metric_history(
        client, run_id, "canary/allocation_restart_count"
    )
    expected_epochs = list(range(RESTART_CANARY_EPOCHS))
    expected_allocations = list(range(restart_count + 1))
    if (
        [row["step"] for row in epoch_history] != expected_epochs
        or [row["value"] for row in epoch_history] != [float(value) for value in expected_epochs]
        or [row["step"] for row in allocation_history] != expected_allocations
        or [row["value"] for row in allocation_history]
        != [float(value) for value in expected_allocations]
        or mlflow.get("epoch_steps") != expected_epochs
        or mlflow.get("epoch_values") != [float(value) for value in expected_epochs]
        or mlflow.get("allocation_steps") != expected_allocations
    ):
        raise RuntimeError("Restart-canary MLflow epoch/allocation history is not exact.")

    return {
        "schema_version": 1,
        "code_commit": code_commit,
        "canary_root": str(canary_root),
        "job_id": job_id,
        "complete": str(complete_path.resolve()),
        "complete_sha256": _sha256(complete_path),
        "state": str(state_path.resolve()),
        "state_sha256": _sha256(state_path),
        "hpc_checkpoint": str(hpc_path),
        "hpc_checkpoint_sha256": _sha256(hpc_path),
        "exact_resume_checkpoint": str(exact_resume_path),
        "exact_resume_checkpoint_sha256": _sha256(exact_resume_path),
        "exact_resume_completed_epochs": completed_epochs,
        "restore_evidence": str(restore_path),
        "restore_evidence_sha256": _sha256(restore_path),
        "restored_retained_checkpoints": [
            {"path": str(path), "sha256": _sha256(path)} for path in restored_retained_paths
        ],
        "retained_checkpoints": [
            {"path": str(path), "sha256": _sha256(path)} for path in retained_paths
        ],
        "scheduler_log": str(source_path),
        "scheduler_log_sha256": _sha256(source_path),
        "scheduler_sentinels": sentinels,
        "scheduler_accounting": accounting,
        "mlflow_run_id": run_id,
        "mlflow_epoch_history": epoch_history,
        "mlflow_allocation_history": allocation_history,
    }


def validate_restart_canary(root: Path, repository: Path, canary_root: Path, job_id: str) -> Path:
    """Validate and persist the forced-restart prerequisite idempotently."""
    with _operation_lock(root):
        design = _design(root)
        _authenticate_execution_repository(repository, design)
        evidence = _validate_restart_canary_evidence(
            root, canary_root, job_id, design["code_commit"]
        )
        output = root / "state" / "restart_canary_validation.json"
        if output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing.get("evidence") != evidence:
                raise RuntimeError(
                    "Existing restart-canary validation differs from current evidence."
                )
            return output
        _atomic_json(
            output,
            {
                "schema_version": 1,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "evidence": evidence,
            },
        )
        return output


def _require_restart_canary_gate(root: Path, design: dict[str, Any]) -> None:
    """Reauthenticate the forced-restart gate before a production canary."""
    path = root / "state" / "restart_canary_validation.json"
    if not path.is_file():
        raise RuntimeError("Forced-restart canary has not passed its validation gate.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = payload.get("evidence", {})
    if (
        payload.get("schema_version") != 1
        or not payload.get("validated_at")
        or evidence.get("schema_version") != 1
        or evidence.get("code_commit") != design["code_commit"]
        or not evidence.get("canary_root")
        or not evidence.get("job_id")
    ):
        raise RuntimeError("Forced-restart canary validation artifact is inconsistent.")
    observed = _validate_restart_canary_evidence(
        root,
        Path(evidence["canary_root"]),
        str(evidence["job_id"]),
        design["code_commit"],
    )
    if observed != evidence:
        raise RuntimeError("Forced-restart canary evidence changed after validation.")


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


def _live_job_ids(name: str, identifier_format: str = "%A") -> set[str]:
    """Return user-scoped live Slurm allocation IDs for one exact job name."""
    if identifier_format not in {"%A", "%F"}:
        raise ValueError(f"Unsupported Slurm job-identity format: {identifier_format}")
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
            f"--format={identifier_format}",
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
    """Return master array IDs for live retraining arrays directly from Slurm."""
    return _live_job_ids("phys-bgpair-retrain", "%F")


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


_RECOVERY_INCIDENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,79}\Z")


def _recovery_incident_id(value: str) -> str:
    """Reject incident names that could escape or alias the recovery namespace."""
    if not _RECOVERY_INCIDENT_PATTERN.fullmatch(value):
        raise RuntimeError(
            "Recovery incident IDs must contain 3..80 lowercase letters, digits, '_' or '-'."
        )
    return value


def _contained(path: Path, parent: Path, role: str) -> Path:
    """Resolve a campaign artifact path without permitting containment escapes."""
    resolved = path.expanduser().resolve()
    root = parent.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"{role} escapes {root}: {resolved}")
    return resolved


def _embedded_checkpoint_epoch(path: Path) -> int:
    """Load the epoch recorded inside one trusted campaign-local checkpoint."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError(f"Checkpoint has no valid embedded epoch: {path}")
    return epoch


def _candidate_reference(
    path_value: Any,
    *,
    checkpoint_root: Path,
    callback: str,
    role: str,
    expected_epoch: int | None,
    boundary: int,
) -> dict[str, Any]:
    """Audit one retained candidate reference without weakening path boundaries."""
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        raise RuntimeError(f"Malformed {role} path in checkpoint callback {callback}.")
    path = _contained(Path(path_value), checkpoint_root, f"{role} candidate")
    evidence: dict[str, Any] = {
        "callback": callback,
        "role": role,
        "path": str(path),
        "expected_record_epoch": expected_epoch,
        "exists": path.is_file(),
        "sha256": None,
        "embedded_epoch": None,
        "valid": False,
        "reason": None,
    }
    if not path.is_file():
        evidence["reason"] = "missing"
        return evidence
    try:
        embedded_epoch = _embedded_checkpoint_epoch(path)
    except Exception as error:
        evidence["reason"] = f"unreadable_or_invalid_epoch:{type(error).__name__}:{error}"
        return evidence
    evidence["sha256"] = _sha256(path)
    evidence["embedded_epoch"] = embedded_epoch
    if expected_epoch is not None and embedded_epoch != expected_epoch:
        evidence["reason"] = "dataset_aware_record_epoch_mismatch"
    elif expected_epoch is None and embedded_epoch >= boundary:
        evidence["reason"] = "lightning_candidate_not_before_committed_boundary"
    else:
        evidence["valid"] = True
    return evidence


def _checkpoint_candidate_evidence(
    checkpoint: Path, checkpoint_root: Path, boundary: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Audit DatasetAware and Lightning candidate references in a boundary checkpoint."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec B614
    callbacks = payload.get("callbacks") if isinstance(payload, dict) else None
    if not isinstance(callbacks, dict):
        raise RuntimeError(f"Boundary checkpoint has no callback state: {checkpoint}")
    references: list[dict[str, Any]] = []
    custom_callbacks = 0
    lightning_callbacks = 0
    lightning_references = 0
    for raw_key, state in callbacks.items():
        callback = str(raw_key)
        if not isinstance(state, dict):
            continue
        records_by_dataset = state.get("checkpoints")
        if isinstance(records_by_dataset, dict) and "criterion" in state:
            custom_callbacks += 1
            for dataset, records in sorted(
                records_by_dataset.items(), key=lambda item: str(item[0])
            ):
                if not isinstance(records, list):
                    raise RuntimeError(
                        f"DatasetAware callback {callback} has malformed records for {dataset}."
                    )
                for position, record in enumerate(records):
                    if (
                        not isinstance(record, dict)
                        or isinstance(record.get("epoch"), bool)
                        or not isinstance(record.get("epoch"), int)
                    ):
                        raise RuntimeError(
                            f"DatasetAware callback {callback} has a malformed record."
                        )
                    references.append(
                        _candidate_reference(
                            record.get("fpath"),
                            checkpoint_root=checkpoint_root,
                            callback=callback,
                            role=f"dataset_aware:{dataset}:{position}",
                            expected_epoch=int(record["epoch"]),
                            boundary=boundary,
                        )
                    )
        if any(key in state for key in ("best_model_path", "best_k_models", "last_model_path")):
            lightning_callbacks += 1
            for key in ("last_model_path", "best_model_path"):
                path = state.get(key)
                if not path:
                    continue
                lightning_references += 1
                references.append(
                    _candidate_reference(
                        path,
                        checkpoint_root=checkpoint_root,
                        callback=callback,
                        role=f"lightning:{key}",
                        expected_epoch=None,
                        boundary=boundary,
                    )
                )
            best_k = state.get("best_k_models", {})
            if not isinstance(best_k, dict):
                raise RuntimeError(f"Lightning callback {callback} has malformed best_k_models.")
            for position, path in enumerate(sorted(best_k, key=str)):
                lightning_references += 1
                references.append(
                    _candidate_reference(
                        path,
                        checkpoint_root=checkpoint_root,
                        callback=callback,
                        role=f"lightning:best_k_models:{position}",
                        expected_epoch=None,
                        boundary=boundary,
                    )
                )
    structural_failures = []
    if custom_callbacks == 0:
        structural_failures.append("no_dataset_aware_checkpoint_state")
    if lightning_callbacks == 0 or lightning_references == 0:
        structural_failures.append("no_lightning_model_checkpoint_candidate_references")
    return references, structural_failures


def _active_retrain_indices(
    root: Path, design: dict[str, Any], rows: list[dict[str, str]]
) -> set[int]:
    """Authenticate live arrays and return their exact frozen run indices."""
    live = _live_retrain_job_ids()
    retrain_rows = [row for row in rows if row.get("kind") == "retrain"]
    registered = {row["job_id"].split("_", maxsplit=1)[0] for row in retrain_rows}
    unknown = live - registered
    if unknown:
        raise RuntimeError(
            "Live retrain arrays are absent from the campaign ledger: "
            f"{sorted(unknown)}. Reconcile their provenance before continuing."
        )
    active_rows = [
        row for row in retrain_rows if _normalized_state(row.get("state", "")) in ACTIVE_STATES
    ]
    active_allocations = {row["job_id"].split("_", maxsplit=1)[0] for row in active_rows}
    stale = active_allocations - live
    if stale:
        raise RuntimeError(
            "Retrain ledger contains terminal-but-unreconciled arrays: "
            f"{sorted(stale)}. Reconcile before submitting another segment."
        )
    unrepresented = live - active_allocations
    if unrepresented:
        raise RuntimeError(
            "Live retrain arrays have no active ledger tasks: "
            f"{sorted(unrepresented)}. Reconcile before submitting another segment."
        )
    indices: set[int] = set()
    for row in active_rows:
        index = _retrain_index_for_ledger_row(root, design, row)
        if index in indices:
            raise RuntimeError(f"Retrain index {index} is active in more than one allocation.")
        indices.add(index)
    return indices


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
        or payload.get("main_studies_empty") is not True
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
        or len({(row.get("model"), row.get("attempt")) for row in jobs}) != len(MODELS)
        or any(
            row.get("state") != "COMPLETED"
            or row.get("exit_code") != "0:0"
            or row.get("commit") != design["code_commit"]
            or not row.get("log_sha256")
            or not row.get("success_sha256")
            or not row.get("bindings_sha256")
            or not str(row.get("attempt", "")).isdigit()
            or int(row["attempt"]) <= 0
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


def _study_trial_count(root: Path, design: dict[str, Any], model: str) -> int:
    """Return the total persisted trial count, including every terminal state."""
    import optuna

    matches = [study for study in design["studies"] if study["model"] == model]
    if len(matches) != 1:
        raise RuntimeError(f"Campaign design does not contain exactly one {model} study.")
    database = root / "studies" / f"{model}.db"
    if not database.is_file():
        return 0
    loaded = optuna.load_study(
        study_name=matches[0]["study_name"], storage=f"sqlite:///{database}"
    )
    return len(loaded.get_trials(deepcopy=False))


def submit_canary(
    root: Path,
    repository: Path,
    model: str,
    data_dir: Path,
    raw_data_dir: Path,
    pairing_dir: Path,
) -> str:
    """Submit one exact four-trial production canary from an empty study."""
    with _operation_lock(root):
        design = _design(root)
        _authenticate_execution_repository(repository, design)
        ledger_path, rows = _ledger(root)
        _assert_no_live_search_overlap(rows, model)
        if _active_model_rows(rows, model):
            raise RuntimeError(f"Refusing to overlap an active {model} campaign job.")
        model_attempts = [
            row
            for row in rows
            if row.get("kind") == "production_canary" and row.get("model") == model
        ]
        if any(
            _normalized_state(row.get("state", "")) == "COMPLETED"
            and row.get("exit_code") == "0:0"
            for row in model_attempts
        ):
            raise RuntimeError(
                f"Campaign ledger already contains a successful production canary for {model}."
            )
        attempt = (
            max(
                (
                    int(row["attempt"])
                    for row in model_attempts
                    if str(row.get("attempt", "")).isdigit()
                ),
                default=len(model_attempts),
            )
            + 1
        )
        _require_restart_canary_gate(root, design)
        canary_database = (
            root / "production_canaries" / model / f"attempt_{attempt:04d}" / "study.db"
        )
        if canary_database.exists():
            raise RuntimeError(
                f"Production canary study already exists for {model}: {canary_database}."
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
            "CHUNK_TRIALS": 4,
            "CHUNK_JOBS": 4,
            "TIMEOUT_MIN": 720,
            "CANARY_MODE": 1,
            "CANARY_ATTEMPT": attempt,
        }
        export_value = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
        sbatch = shutil.which("sbatch")
        if sbatch is None:
            raise FileNotFoundError("sbatch")
        result = subprocess.run(  # nosec B603 -- reviewed fixed canary shape
            [
                sbatch,
                "--parsable",
                "--nodes=1",
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
                "kind": "production_canary",
                "model": model,
                "state": "PENDING",
                "commit": design["code_commit"],
                "attempt": str(attempt),
                "log_path": str(
                    Path("/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm")
                    / f"phys-bgpair-sweep-{job_id}.out"
                ),
            }
        )
        _atomic_csv(ledger_path, rows)
        return job_id


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
    completed_in_generation = usable % NSGAII_SEED_POLICY["generation_size"]
    generation = usable // NSGAII_SEED_POLICY["generation_size"]
    generation_capacity = min(
        NSGAII_SEED_POLICY["generation_size"],
        TARGET_TRIALS - generation * NSGAII_SEED_POLICY["generation_size"],
    )
    generation_remaining = generation_capacity - completed_in_generation
    if trials > generation_remaining:
        raise RuntimeError(
            f"Search chunk crosses the NSGA-II generation boundary for {model}: "
            f"generation {generation} has {generation_remaining} slots, got {trials}."
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
        "CANARY_MODE": 0,
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
    from scripts import physics_background_pairing_campaign as campaign

    try:
        campaign._validate_retrain_marker(marker, row)
    except Exception as error:
        raise RuntimeError(
            f"Retraining marker failed the complete artifact audit: {marker}"
        ) from error
    return True


def _retrain_manifest_path(root: Path, index: int | None = None) -> Path:
    """Use base evidence for base indices and the overlay for reserve work."""
    base = root / "selection" / "pareto_retrain_manifest.json"
    effective = root / "selection" / "effective_retrain_manifest.json"
    if index is not None and base.is_file():
        base_rows = json.loads(base.read_text(encoding="utf-8")).get("retrain_runs", [])
        if 0 <= index < len(base_rows):
            return base
    return effective if effective.is_file() else base


def missing_retrains(root: Path, repository: Path) -> list[int]:
    """Return exact frozen retrain indices lacking an authenticated marker."""
    design = _design(root)
    _authenticate_execution_repository(repository, design)
    manifest_path = _retrain_manifest_path(root)
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


def _mlflow_retrain_run_directory(root: Path, run_id: str) -> Path:
    """Resolve exactly one campaign-local MLflow run directory."""
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise RuntimeError(f"Invalid retrain MLflow run ID: {run_id!r}.")
    tracking_root = (root / "runtime" / "logs" / "mlflow" / "mlruns").resolve()
    matches = [
        path.resolve()
        for path in tracking_root.glob(f"*/{run_id}")
        if path.is_dir() and path.parent.name != ".trash"
    ]
    if len(matches) != 1 or not matches[0].is_relative_to(tracking_root):
        raise RuntimeError(f"Expected exactly one campaign-local MLflow run for {run_id}.")
    return matches[0]


def _latest_committed_retrain_snapshot(
    root: Path,
    design: dict[str, Any],
    ledger_rows: list[dict[str, str]],
    index: int,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    """Authenticate the newest sealed segment despite later incomplete attempts."""
    candidates = []
    for ledger_row in ledger_rows:
        if (
            ledger_row.get("kind") != "retrain"
            or ledger_row.get("state") != "COMPLETED"
            or ledger_row.get("exit_code") != "0:0"
            or _retrain_index_for_ledger_row(root, design, ledger_row) != index
        ):
            continue
        receipt_path = _retrain_receipt_path(root, ledger_row["job_id"])
        if not receipt_path.is_file():
            raise RuntimeError(
                f"Successful retrain {ledger_row['job_id']} has no sealed segment receipt."
            )
        _validate_retrain_receipt(root, design, ledger_row, receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        snapshot = receipt.get("evidence", {}).get("snapshot")
        segments = snapshot.get("segments", []) if isinstance(snapshot, dict) else []
        if not segments:
            raise RuntimeError(f"Retrain receipt has no committed segment: {receipt_path}")
        candidates.append(
            (
                int(segments[-1].get("epoch_boundary", -1)),
                snapshot,
                {
                    "scheduler_job_id": ledger_row["job_id"],
                    "path": str(receipt_path.resolve()),
                    "sha256": _sha256(receipt_path),
                },
            )
        )
    if not candidates:
        raise RuntimeError(f"Retrain index {index} has no sealed committed segment receipt.")
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[-2][0] == candidates[-1][0]:
        raise RuntimeError(f"Retrain index {index} has duplicate latest committed boundaries.")
    _, snapshot, receipt_evidence = candidates[-1]
    state_path = root / "retraining" / "restart_state" / f"{index:04d}" / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    current = json.loads(state_path.read_text(encoding="utf-8"))
    committed_segments = snapshot["segments"]
    if (
        current.get("identity") != snapshot.get("identity")
        or current.get("mlflow_run_id") != snapshot.get("mlflow_run_id")
        or current.get("segments", [])[: len(committed_segments)] != committed_segments
    ):
        raise RuntimeError(f"Current retrain state changed its sealed prefix: {state_path}")
    return state_path, snapshot, receipt_evidence


def _candidate_audit_payload(
    root: Path,
    design: dict[str, Any],
    ledger_rows: list[dict[str, str]],
    incident_id: str,
) -> dict[str, Any]:
    """Build deterministic evidence for retained candidates at every committed boundary."""
    incident_id = _recovery_incident_id(incident_id)
    manifest_path = _retrain_manifest_path(root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("retrain_runs", [])
    if (
        manifest.get("schema_version") != 2
        or manifest.get("code_commit") != design.get("code_commit")
        or [row.get("retrain_index") for row in rows] != list(range(len(rows)))
    ):
        raise RuntimeError("Recovery requires an authenticated contiguous retrain manifest.")
    audits = []
    affected_indices = []
    for index, row in enumerate(rows):
        marker = root / "retraining" / "markers" / f"{index:04d}.json"
        if marker.exists():
            raise RuntimeError(f"Refusing to recover completed retrain index {index}: {marker}")
        state_path, state, committed_receipt = _latest_committed_retrain_snapshot(
            root, design, ledger_rows, index
        )
        segments = state.get("segments", [])
        if not segments:
            raise RuntimeError(f"Retrain index {index} has no committed boundary.")
        latest = segments[-1]
        boundary = int(latest.get("epoch_boundary", -1))
        segment_root = root / "retraining" / "restart_state" / f"{index:04d}" / "segments"
        checkpoint = _contained(
            Path(str(latest.get("continuation_checkpoint", ""))),
            segment_root,
            "latest committed boundary checkpoint",
        )
        expected_sha256 = str(latest.get("continuation_checkpoint_sha256", ""))
        if (
            boundary <= 0
            or boundary > int(design.get("retrain_epochs", -1))
            or not checkpoint.is_file()
            or _sha256(checkpoint) != expected_sha256
            or _embedded_checkpoint_epoch(checkpoint) != boundary - 1
        ):
            raise RuntimeError(f"Immutable committed boundary is invalid for retrain {index}.")
        identity = state.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeError(f"Retrain index {index} has no immutable identity.")
        checkpoint_root = _contained(
            root
            / "runtime"
            / "checkpoints"
            / str(identity.get("experiment_name", ""))
            / str(identity.get("run_name", "")),
            root / "runtime" / "checkpoints",
            "retrain checkpoint root",
        )
        hydra_dir = _contained(
            Path(str(identity.get("hydra_dir", ""))),
            root / "retraining" / "hydra",
            "retrain Hydra directory",
        )
        if hydra_dir != (root / "retraining" / "hydra" / f"{index:04d}").resolve():
            raise RuntimeError(f"Retrain index {index} has a mismatched Hydra directory.")
        mlflow_run = _mlflow_retrain_run_directory(root, str(state.get("mlflow_run_id", "")))
        references, structural_failures = _checkpoint_candidate_evidence(
            checkpoint, checkpoint_root, boundary
        )
        failures = [*structural_failures]
        failures.extend(
            f"{reference['role']}:{reference['reason']}"
            for reference in references
            if not reference["valid"]
        )
        affected = bool(failures)
        if affected:
            affected_indices.append(index)
        audits.append(
            {
                "retrain_index": index,
                "model": row["model"],
                "boundary": boundary,
                "boundary_checkpoint": str(checkpoint),
                "boundary_checkpoint_sha256": expected_sha256,
                "committed_segment_receipt": committed_receipt,
                "checkpoint_root": str(checkpoint_root),
                "hydra_dir": str(hydra_dir),
                "restart_state_dir": str(state_path.parent.resolve()),
                "mlflow_run_id": str(state["mlflow_run_id"]),
                "mlflow_run_dir": str(mlflow_run),
                "candidate_references": references,
                "failures": failures,
                "affected": affected,
            }
        )
    return {
        "schema_version": 1,
        "kind": "retrain_candidate_integrity_audit",
        "incident_id": incident_id,
        "campaign_root": str(root.resolve()),
        "code_commit": design["code_commit"],
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "audited_indices": list(range(len(rows))),
        "affected_indices": affected_indices,
        "retrain_audits": audits,
    }


def audit_retrain_candidates(root: Path, repository: Path, incident_id: str) -> Path:
    """Seal a quiescent, read-only audit of every retained checkpoint reference."""
    with _operation_lock(root):
        design = _design(root)
        _authenticate_execution_repository(repository, design)
        _, ledger_rows = _ledger(root)
        _assert_no_live_retrain_overlap(ledger_rows)
        payload = _candidate_audit_payload(root, design, ledger_rows, incident_id)
        path = root / "state" / "retrain_candidate_audits" / f"{incident_id}.json"
        _write_once_json(path, payload)
        return path


def _archive_inventory(path: Path) -> dict[str, Any]:
    """Hash a regular file tree without following aliases or special files."""
    if path.is_symlink():
        raise RuntimeError(f"Recovery source may not be a symlink: {path}")
    if path.is_file():
        files = [{"path": ".", "size": path.stat().st_size, "sha256": _sha256(path)}]
        directories: list[str] = []
        kind = "file"
    elif path.is_dir():
        files = []
        directories = []
        for child in sorted(path.rglob("*")):
            if child.is_symlink() or (not child.is_file() and not child.is_dir()):
                raise RuntimeError(f"Recovery source contains an unsupported entry: {child}")
            if child.is_file():
                files.append(
                    {
                        "path": child.relative_to(path).as_posix(),
                        "size": child.stat().st_size,
                        "sha256": _sha256(child),
                    }
                )
            elif child.is_dir():
                directories.append(child.relative_to(path).as_posix())
        kind = "directory"
    else:
        raise FileNotFoundError(path)
    content = {"directories": directories, "files": files}
    return {"kind": kind, **content, "content_sha256": _json_sha256(content)}


def _retrain_rows_for_indices(
    root: Path, design: dict[str, Any], ledger_rows: list[dict[str, str]], indices: set[int]
) -> list[dict[str, str]]:
    """Select every historical retrain ledger row owned by an affected index."""
    selected = []
    for row in ledger_rows:
        if row.get("kind") != "retrain":
            continue
        index = _retrain_index_for_ledger_row(root, design, row)
        if index in indices:
            selected.append(dict(row))
    return selected


def _inventory_contract(artifact: dict[str, Any]) -> dict[str, Any]:
    """Project an archive row to its immutable content inventory."""
    return {key: artifact.get(key) for key in ("kind", "directories", "files", "content_sha256")}


def _validate_reset_intent_sources(
    root: Path,
    design: dict[str, Any],
    archive_root: Path,
    audit_rows: list[dict[str, Any]],
    affected: set[int],
    original_ledger_rows: list[dict[str, str]],
    sources: Any,
) -> list[dict[str, Any]]:
    """Authenticate exact source/target identities before starting or resuming moves."""
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("Candidate reset intent has no archive sources.")
    expected_base = {}
    for row in audit_rows:
        if not row.get("affected"):
            continue
        index = int(row["retrain_index"])
        expected_base.update(
            {
                (index, "restart_state"): (
                    root / "retraining" / "restart_state" / f"{index:04d}"
                ).resolve(),
                (index, "hydra"): (root / "retraining" / "hydra" / f"{index:04d}").resolve(),
                (index, "checkpoints"): Path(str(row["checkpoint_root"])).resolve(),
                (index, "mlflow_run"): Path(str(row["mlflow_run_dir"])).resolve(),
            }
        )
    originals = {row["job_id"]: row for row in original_ledger_rows}
    seen: set[tuple[int, str]] = set()
    successful_receipts = {
        job_id
        for job_id, row in originals.items()
        if row.get("state") == "COMPLETED" and row.get("exit_code") == "0:0"
    }
    observed_receipts = set()
    for artifact in sources:
        if not isinstance(artifact, dict):
            raise RuntimeError("Candidate reset intent contains a malformed archive row.")
        index = artifact.get("retrain_index")
        role = artifact.get("role")
        if isinstance(index, bool) or not isinstance(index, int) or index not in affected:
            raise RuntimeError("Candidate reset intent names an unaudited retrain index.")
        if not isinstance(role, str) or (index, role) in seen:
            raise RuntimeError("Candidate reset intent contains a duplicate or invalid role.")
        seen.add((index, role))
        source = _contained(Path(str(artifact.get("source", ""))), root, "recovery source")
        if (index, role) in expected_base:
            if source != expected_base[(index, role)]:
                raise RuntimeError(f"Candidate reset intent changed its {role} source.")
        elif role.startswith("retrain_receipt:"):
            job_id = role.removeprefix("retrain_receipt:")
            if job_id not in originals or source != _retrain_receipt_path(root, job_id).resolve():
                raise RuntimeError("Candidate reset intent names an unrelated retrain receipt.")
            if _retrain_index_for_ledger_row(root, design, originals[job_id]) != index:
                raise RuntimeError("Candidate reset receipt belongs to another retrain index.")
            if job_id in successful_receipts:
                observed_receipts.add(job_id)
        else:
            raise RuntimeError(f"Candidate reset intent has an unsupported role: {role!r}.")
        target = _contained(
            Path(str(artifact.get("target", ""))), archive_root, "recovery archive target"
        )
        digest = artifact.get("content_sha256")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest[:20] not in target.name
        ):
            raise RuntimeError("Candidate reset target is not content-addressed.")
        inventory = _inventory_contract(artifact)
        if (
            inventory["kind"] not in {"file", "directory"}
            or not isinstance(inventory["directories"], list)
            or not isinstance(inventory["files"], list)
        ):
            raise RuntimeError("Candidate reset intent has a malformed content inventory.")
    if not set(expected_base).issubset(seen) or observed_receipts != successful_receipts:
        raise RuntimeError("Candidate reset intent does not cover its exact required sources.")
    if len({row["source"] for row in sources}) != len(sources) or len(
        {row["target"] for row in sources}
    ) != len(sources):
        raise RuntimeError("Candidate reset intent aliases archive sources or targets.")
    return sources


def reset_affected_retrains(
    root: Path, repository: Path, incident_id: str, audit_sha256: str
) -> Path:
    """Archive only audited corruption and authorize exact frozen-identity reruns."""
    with _operation_lock(root):
        incident_id = _recovery_incident_id(incident_id)
        if not re.fullmatch(r"[0-9a-f]{64}", audit_sha256):
            raise RuntimeError("A candidate reset requires the exact audit SHA-256.")
        design = _design(root)
        _authenticate_execution_repository(repository, design)
        ledger_path, ledger_rows = _ledger(root)
        _assert_no_live_retrain_overlap(ledger_rows)
        reset_path = root / "state" / "retrain_candidate_resets" / f"{incident_id}.json"
        if reset_path.exists():
            raise RuntimeError(f"Recovery incident was already reset: {reset_path}")
        audit_path = root / "state" / "retrain_candidate_audits" / f"{incident_id}.json"
        if not audit_path.is_file() or _sha256(audit_path) != audit_sha256:
            raise RuntimeError("Candidate reset audit is missing or has the wrong SHA-256.")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        affected = audit.get("affected_indices")
        audit_rows = audit.get("retrain_audits")
        if (
            audit.get("schema_version") != 1
            or audit.get("kind") != "retrain_candidate_integrity_audit"
            or audit.get("incident_id") != incident_id
            or audit.get("campaign_root") != str(root.resolve())
            or audit.get("code_commit") != design["code_commit"]
            or not isinstance(affected, list)
            or not affected
            or len(set(affected)) != len(affected)
            or not isinstance(audit_rows, list)
            or audit.get("audited_indices") != list(range(len(audit_rows)))
            or [row.get("retrain_index") for row in audit_rows] != list(range(len(audit_rows)))
            or [row.get("retrain_index") for row in audit_rows if row.get("affected")] != affected
        ):
            raise RuntimeError("Candidate reset audit has an invalid affected-index contract.")
        manifest_path = _retrain_manifest_path(root).resolve()
        if audit.get("manifest") != str(manifest_path) or audit.get("manifest_sha256") != _sha256(
            manifest_path
        ):
            raise RuntimeError("Candidate reset audit is detached from the retrain manifest.")
        affected_set = {int(index) for index in affected}
        original_ledger_rows = _retrain_rows_for_indices(root, design, ledger_rows, affected_set)
        if not original_ledger_rows:
            raise RuntimeError("No retrain ledger rows map to the affected indices.")
        if any(
            _normalized_state(row.get("state", "")) not in TERMINAL_STATES
            or not row.get("log_sha256")
            for row in original_ledger_rows
        ):
            raise RuntimeError("Affected retrain ledger rows must be reconciled and terminal.")
        archive_root = (
            root / "state" / "retrain_candidate_recovery_archives" / incident_id
        ).resolve()
        intent_path = root / "state" / "retrain_candidate_reset_intents" / f"{incident_id}.json"
        if intent_path.is_file():
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            if (
                intent.get("schema_version") != 1
                or intent.get("kind") != "retrain_candidate_reset_intent"
                or intent.get("incident_id") != incident_id
                or intent.get("campaign_root") != str(root.resolve())
                or intent.get("audit") != str(audit_path.resolve())
                or intent.get("audit_sha256") != audit_sha256
                or intent.get("affected_indices") != affected
                or intent.get("original_ledger_rows") != original_ledger_rows
            ):
                raise RuntimeError("Existing candidate reset intent differs from current state.")
            sources = intent.get("sources")
        else:
            for row in original_ledger_rows:
                successful = row.get("state") == "COMPLETED" and row.get("exit_code") == "0:0"
                if successful and not _retrain_receipt_path(root, row["job_id"]).is_file():
                    raise RuntimeError(
                        f"Successful affected retrain has no sealed receipt: {row['job_id']}."
                    )
            sources = []
            for row in audit_rows:
                if not row.get("affected"):
                    continue
                index = int(row["retrain_index"])
                expected = {
                    "restart_state": root / "retraining" / "restart_state" / f"{index:04d}",
                    "hydra": root / "retraining" / "hydra" / f"{index:04d}",
                    "checkpoints": Path(str(row["checkpoint_root"])),
                    "mlflow_run": Path(str(row["mlflow_run_dir"])),
                }
                for role, raw_source in expected.items():
                    source = _contained(raw_source, root, f"recovery {role}")
                    inventory = _archive_inventory(source)
                    target = (
                        archive_root
                        / f"{index:04d}"
                        / (f"{role}__{inventory['content_sha256'][:20]}")
                    )
                    sources.append(
                        {
                            "retrain_index": index,
                            "role": role,
                            "source": str(source),
                            "target": str(target),
                            **inventory,
                        }
                    )
            for ledger_row in original_ledger_rows:
                receipt = _retrain_receipt_path(root, ledger_row["job_id"])
                if not receipt.exists():
                    continue
                inventory = _archive_inventory(receipt)
                index = _retrain_index_for_ledger_row(root, design, ledger_row)
                target = (
                    archive_root
                    / f"{index:04d}"
                    / (
                        f"retrain_receipt_{ledger_row['job_id']}__"
                        f"{inventory['content_sha256'][:20]}.json"
                    )
                )
                sources.append(
                    {
                        "retrain_index": index,
                        "role": f"retrain_receipt:{ledger_row['job_id']}",
                        "source": str(receipt.resolve()),
                        "target": str(target),
                        **inventory,
                    }
                )
            intent = {
                "schema_version": 1,
                "kind": "retrain_candidate_reset_intent",
                "incident_id": incident_id,
                "campaign_root": str(root.resolve()),
                "audit": str(audit_path.resolve()),
                "audit_sha256": audit_sha256,
                "affected_indices": affected,
                "sources": sources,
                "original_ledger_rows": original_ledger_rows,
            }
        sources = _validate_reset_intent_sources(
            root,
            design,
            archive_root,
            audit_rows,
            affected_set,
            original_ledger_rows,
            sources,
        )
        _write_once_json(intent_path, intent)
        for artifact in sources:
            source = Path(artifact["source"])
            target = Path(artifact["target"])
            source_exists = source.exists()
            target_exists = target.exists()
            if source_exists == target_exists:
                raise RuntimeError(
                    f"Recovery transition requires exactly one source/archive copy: {source}"
                )
            expected_inventory = _inventory_contract(artifact)
            if source_exists:
                if _archive_inventory(source) != expected_inventory:
                    raise RuntimeError(f"Recovery source changed after intent: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
            if _archive_inventory(target) != expected_inventory:
                raise RuntimeError(f"Archived recovery artifact changed while moving: {target}")
        reset = {
            "schema_version": 1,
            "kind": "retrain_candidate_reset",
            "incident_id": incident_id,
            "campaign_root": str(root.resolve()),
            "code_commit": design["code_commit"],
            "audit": str(audit_path.resolve()),
            "audit_sha256": audit_sha256,
            "intent": str(intent_path.resolve()),
            "intent_sha256": _sha256(intent_path),
            "affected_indices": affected,
            "archives": [
                {**row, "source_absent": not Path(row["source"]).exists()} for row in sources
            ],
            "original_ledger_rows": original_ledger_rows,
            "superseded_job_ids": [row["job_id"] for row in original_ledger_rows],
            "rerun_contract": "from_scratch_with_unchanged_frozen_manifest_row_and_seed",
        }
        _write_once_json(reset_path, reset)
        reset_sha256 = _sha256(reset_path)
        superseded_ids = set(reset["superseded_job_ids"])
        for ledger_row in ledger_rows:
            if ledger_row["job_id"] in superseded_ids:
                ledger_row["kind"] = "retrain_superseded"
                ledger_row["success_sha256"] = reset_sha256
        _atomic_csv(ledger_path, ledger_rows)
        return reset_path


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


def _write_retrain_task_map(
    root: Path,
    design: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    actual_indices: list[int],
) -> tuple[Path, str, dict[str, Any]]:
    """Persist one content-addressed local-to-frozen retrain-index map."""
    if (
        not actual_indices
        or actual_indices != sorted(set(actual_indices))
        or len(actual_indices) > MAX_RETRAIN_ARRAY_TASKS
    ):
        raise ValueError(
            f"A retrain batch must contain 1..{MAX_RETRAIN_ARRAY_TASKS} sorted unique indices."
        )
    runs = manifest.get("retrain_runs", [])
    entries = []
    for local_task_id, retrain_index in enumerate(actual_indices):
        if not 0 <= retrain_index < len(runs):
            raise RuntimeError(
                f"Retrain index {retrain_index} is absent from the frozen manifest."
            )
        row = runs[retrain_index]
        if int(row.get("retrain_index", -1)) != retrain_index:
            raise RuntimeError(f"Frozen retrain row {retrain_index} has a mismatched index.")
        entries.append(
            {
                "local_task_id": local_task_id,
                "retrain_index": retrain_index,
                "model": str(row["model"]),
                "row_sha256": _json_sha256(row),
            }
        )
    payload = {
        "schema_version": 1,
        "code_commit": design["code_commit"],
        "campaign_root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "batch_size": len(entries),
        "entries": entries,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    path = root / "state" / "retrain_task_maps" / f"{digest}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload or _sha256(path) != digest:
            raise RuntimeError(f"Existing retrain task map is not authentic: {path}")
    else:
        _atomic_json(path, payload)
        if _sha256(path) != digest:
            raise RuntimeError(f"Persisted retrain task-map digest differs: {path}")
    return path, digest, payload


def _load_retrain_task_map(root: Path, design: dict[str, Any], digest: str) -> dict[str, Any]:
    """Authenticate a ledger-bound retrain task map against the frozen manifest."""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"Invalid retrain task-map SHA-256: {digest!r}.")
    path = root / "state" / "retrain_task_maps" / f"{digest}.json"
    if not path.is_file() or _sha256(path) != digest:
        raise RuntimeError(f"Retrain task map is missing or changed: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_value = payload.get("manifest")
    allowed_manifests = {
        (root / "selection" / "pareto_retrain_manifest.json").resolve(),
        (root / "selection" / "effective_retrain_manifest.json").resolve(),
    }
    manifest_path = Path(str(manifest_value)).expanduser().resolve()
    if manifest_path not in allowed_manifests or not manifest_path.is_file():
        raise RuntimeError(f"Retrain task map names an invalid manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if (
        payload.get("schema_version") != 1
        or payload.get("code_commit") != design["code_commit"]
        or payload.get("campaign_root") != str(root.resolve())
        or manifest_value != str(manifest_path)
        or payload.get("manifest_sha256") != _sha256(manifest_path)
        or payload.get("batch_size") != len(entries)
        or not 1 <= len(entries) <= MAX_RETRAIN_ARRAY_TASKS
        or [entry.get("local_task_id") for entry in entries] != list(range(len(entries)))
    ):
        raise RuntimeError(f"Retrain task map does not match the frozen campaign: {path}")
    runs = manifest.get("retrain_runs", [])
    actual_indices = [entry.get("retrain_index") for entry in entries]
    if actual_indices != sorted(set(actual_indices)):
        raise RuntimeError(f"Retrain task map has non-exact actual indices: {path}")
    for entry in entries:
        index = int(entry["retrain_index"])
        if (
            not 0 <= index < len(runs)
            or runs[index].get("retrain_index") != index
            or entry.get("model") != runs[index].get("model")
            or entry.get("row_sha256") != _json_sha256(runs[index])
        ):
            raise RuntimeError(f"Retrain task-map row differs from the frozen manifest: {path}")
    return payload


def _retrain_index_for_ledger_row(root: Path, design: dict[str, Any], row: dict[str, str]) -> int:
    """Resolve a task ledger row, retaining compatibility with legacy direct arrays."""
    try:
        local_task_id = int(row["job_id"].rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"Invalid retrain task job ID: {row['job_id']}.") from error
    digest = row.get("bindings_sha256", "")
    if not digest:
        return local_task_id
    payload = _load_retrain_task_map(root, design, digest)
    entries = payload["entries"]
    if not 0 <= local_task_id < len(entries):
        raise RuntimeError(f"Retrain local task is absent from its map: {row['job_id']}.")
    entry = entries[local_task_id]
    if entry["local_task_id"] != local_task_id or entry["model"] != row.get("model"):
        raise RuntimeError(f"Retrain ledger row differs from its task map: {row['job_id']}.")
    return int(entry["retrain_index"])


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
    active_indices = _active_retrain_indices(root, design, ledger_rows)
    missing = [
        index for index in missing_retrains(root, repository) if index not in active_indices
    ]
    if not missing:
        return None
    for path in (data_dir, raw_data_dir, pairing_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    _authenticate_confirmation_submission_data(root, design, data_dir, raw_data_dir, pairing_dir)
    manifest_path = _retrain_manifest_path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_by_index = {int(row["retrain_index"]): row for row in manifest["retrain_runs"]}
    batch = missing[:MAX_RETRAIN_ARRAY_TASKS]
    task_map_path, task_map_sha256, task_map = _write_retrain_task_map(
        root, design, manifest_path, manifest, batch
    )
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
        # Lightning resumes these trusted, locally produced full-state checkpoints. PyTorch
        # otherwise defaults to weights-only loading and rejects the serialized optimizer.
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        # The shared uv environment may have been created from another clean
        # worktree. Always resolve project modules from the authenticated
        # execution repository used for this submission.
        "PYTHONPATH": repository.resolve(),
        "PHYSICS_BGPAIR_RETRAIN_TASK_MAP": task_map_path.resolve(),
        "PHYSICS_BGPAIR_RETRAIN_TASK_MAP_SHA256": task_map_sha256,
    }
    export_value = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise FileNotFoundError("sbatch")
    result = subprocess.run(  # nosec B603 -- reviewed sbatch command
        [
            sbatch,
            "--parsable",
            f"--array=0-{len(batch) - 1}%{min(concurrency, len(batch))}",
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
    for entry in task_map["entries"]:
        local_task_id = int(entry["local_task_id"])
        index = int(entry["retrain_index"])
        ledger_rows.append(
            {
                "job_id": f"{array_job_id}_{local_task_id}",
                "kind": "retrain",
                "model": str(rows_by_index[index]["model"]),
                "state": "PENDING",
                "commit": design["code_commit"],
                "log_path": str(
                    log_root / f"phys-bgpair-retrain-{array_job_id}_{local_task_id}.out"
                ),
                "bindings_sha256": task_map_sha256,
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


def _canary_trial_evidence(
    root: Path, design: dict[str, Any], accepted_jobs: list[dict[str, str]]
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Authenticate isolated canary studies, bindings, selections, and MLflow identities."""
    import optuna

    status_rows = []
    bindings = {}
    selection_rows = []
    mlflow_rows = []
    accepted_by_model = {row["model"]: row for row in accepted_jobs}
    for study in design["studies"]:
        model = study["model"]
        attempt = int(accepted_by_model[model]["attempt"])
        canary_root = root / "production_canaries" / model / f"attempt_{attempt:04d}"
        database = canary_root / "study.db"
        study_name = (
            f"physics_{model}_production_canary_{design['code_commit'][:12]}_"
            f"attempt_{attempt:04d}"
        )
        loaded = optuna.load_study(study_name=study_name, storage=f"sqlite:///{database}")
        trials = list(loaded.get_trials(deepcopy=False))
        expected_values = 2 * len(study["objectives"])
        usable = [
            trial
            for trial in trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.values is not None
            and len(trial.values) == expected_values
            and all(math.isfinite(float(value)) for value in trial.values)
        ]
        states = {state.name.lower(): 0 for state in optuna.trial.TrialState}
        for trial in trials:
            states[trial.state.name.lower()] += 1
        status_rows.append(
            {
                "model": model,
                "total": len(trials),
                "usable_complete": len(usable),
                **states,
            }
        )
        payload = json.loads((canary_root / "trial_bindings.json").read_text(encoding="utf-8"))
        trial_index = {int(trial.number): trial for trial in trials}
        for binding in payload["bindings"]:
            trial_number = int(binding["trial_number"])
            trial = trial_index[trial_number]
            values = [float(value) for value in trial.values]
            if (
                binding["state"] != trial.state.name
                or binding["params"] != trial.params
                or binding["values"] != values
                or binding.get("nsga2_generation") != trial.system_attrs.get(NSGA_GENERATION_KEY)
            ):
                raise RuntimeError(
                    f"Canary binding differs from Optuna trial {(model, trial_number)}."
                )
            key = (model, trial_number)
            bindings[key] = binding
            artifact_path = Path(binding["optimized_metric_artifact"])
            if _sha256(artifact_path) != binding["optimized_metric_artifact_sha256"]:
                raise RuntimeError(f"Changed canary objective artifact: {artifact_path}.")
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact.get("optimized_metric") != values:
                raise RuntimeError(f"Canary artifact differs from Optuna trial {key}.")
            for objective in study["objectives"]:
                selection_rows.append(
                    {
                        "model": model,
                        "logical_objective": objective["id"],
                        "optimized_ckpt_name": artifact["selections"][objective["id"]][
                            "optimized_ckpt_name"
                        ],
                    }
                )
            provenance = artifact["run_provenance"]
            loggers = [logger for logger in provenance.get("loggers", []) if logger.get("run_id")]
            if len(loggers) != 1:
                raise RuntimeError(f"Canary run has no unique MLflow identity: {key}.")
            mlflow_rows.append(
                {
                    "phase": "search",
                    "model": model,
                    "trial_number": trial_number,
                    "retrain_index": "",
                    "experiment_name": provenance.get("experiment_name", ""),
                    "experiment_id": loggers[0].get("experiment_id", ""),
                    "run_id": loggers[0]["run_id"],
                    "run_name": provenance.get("run_name", ""),
                    "hydra_run_dir": provenance.get("hydra_output_dir", ""),
                    "checkpoint_run_dir": provenance.get("checkpoint_run_dir", ""),
                    "artifact": str(artifact_path),
                }
            )
    return status_rows, bindings, selection_rows, mlflow_rows


def validate_canaries(root: Path, repository: Path) -> Path:
    """Fail closed unless all six four-trial production canaries satisfy every gate."""
    from scripts import physics_background_pairing_report as report

    design = report._design(root)
    _authenticate_execution_repository(repository, design)
    ledger_rows = reconcile(root, repository, True)
    canary_attempts = [row for row in ledger_rows if row.get("kind") == "production_canary"]
    if any(
        not str(row.get("attempt", "")).isdigit()
        or int(row["attempt"]) <= 0
        or _normalized_state(row.get("state", "")) not in TERMINAL_STATES
        for row in canary_attempts
    ):
        raise RuntimeError("Canary ledger contains an active or unidentified attempt.")
    if len({(row["model"], row["attempt"]) for row in canary_attempts}) != len(canary_attempts):
        raise RuntimeError("Canary ledger contains duplicate attempt identities.")
    for model in MODELS:
        attempts = sorted(
            (row for row in canary_attempts if row["model"] == model),
            key=lambda row: int(row["attempt"]),
        )
        if [int(row["attempt"]) for row in attempts] != list(range(1, len(attempts) + 1)):
            raise RuntimeError(f"Canary attempt history is not contiguous for {model}.")
    canary_jobs = [
        row
        for row in canary_attempts
        if _normalized_state(row["state"]) == "COMPLETED" and row["exit_code"] == "0:0"
    ]
    if len(canary_jobs) != len(MODELS) or {row["model"] for row in canary_jobs} != set(MODELS):
        raise RuntimeError(
            "Canary ledger must contain exactly one successful accepted attempt per model."
        )
    if any(
        int(row["attempt"])
        != max(
            int(attempt["attempt"])
            for attempt in canary_attempts
            if attempt["model"] == row["model"]
        )
        for row in canary_jobs
    ):
        raise RuntimeError("A successful canary is not the final accepted model attempt.")
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
        current_hashes = _completed_canary_artifacts(root, design, row)
        if current_hashes != (row["success_sha256"], row["bindings_sha256"]):
            raise RuntimeError(f"Production canary changed after ledger sealing: {row}")

    status_rows, bindings, selection_rows, mlflow_rows = _canary_trial_evidence(
        root, design, canary_jobs
    )
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
    expected_bindings = {(model, index) for model in MODELS for index in range(4)}
    if set(bindings) != expected_bindings:
        raise RuntimeError("Canary bindings do not exactly cover trials 0-3 for six models.")
    annotated = report._annotate_checkpoint_families(design, selection_rows)
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
            or config.get("trainer", {}).get("min_epochs") != 50
            or config.get("trainer", {}).get("max_epochs") != 50
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
    main_studies_empty = all(_study_trial_count(root, design, model) == 0 for model in MODELS)
    if not main_studies_empty:
        raise RuntimeError("Production canaries contaminated one or more final Optuna studies.")

    output = root / "state" / "production_canary_validation.json"
    payload = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": design["code_commit"],
        "main_studies_empty": main_studies_empty,
        "jobs": canary_jobs,
        "attempt_history": canary_attempts,
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
    sampler = design.get("screening_sampler")
    expected_hparams = study.get("hparams_search", f"{model}_shared_optuna")
    chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
    binding_payload = json.loads(bindings_path.read_text(encoding="utf-8"))
    invocation = chunk.get("sampler_invocation")
    resolved_sampler = (
        None
        if sampler is None or not isinstance(invocation, dict)
        else {**sampler, "seed": invocation.get("resolved_seed")}
    )
    if sampler is not None and not isinstance(invocation, dict):
        raise RuntimeError(f"Terminal chunk has no sampler invocation: {chunk_path}.")
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
            or binding.get("hparams_search", expected_hparams) != expected_hparams
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
        if sampler is not None and (
            binding.get("screening_sampler") != resolved_sampler
            or binding.get("nsga2_trial_phase") != "sampled"
            or not isinstance(binding.get("nsga2_generation"), int)
            or int(binding["nsga2_generation"]) < 0
            or binding.get("nsga2_generation") != invocation.get("expected_generation")
            or binding.get("resolved_sampler_seed") != invocation.get("resolved_seed")
        ):
            raise RuntimeError(
                f"Terminal search binding has unauthenticated {NSGA_GENERATION_KEY}: {key}."
            )
        indexed[key] = {
            **binding,
            "binding_artifact": str(bindings_path.resolve()),
            "_binding_payload": binding_payload,
        }
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
    payload = {
        "schema_version": 1,
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
        "hparams_search": expected_hparams,
        "screening_sampler_base": sampler,
        "screening_sampler_base_sha256": (None if sampler is None else _json_sha256(sampler)),
        "screening_sampler": resolved_sampler,
        "screening_sampler_sha256": (
            None if resolved_sampler is None else _json_sha256(resolved_sampler)
        ),
        "screening_sampler_seed_policy": design.get("screening_sampler_seed_policy"),
        "screening_sampler_seed_policy_sha256": (
            None if sampler is None else _json_sha256(design["screening_sampler_seed_policy"])
        ),
        "sampler_invocation": invocation,
        "nsga2_generations": (
            None
            if sampler is None
            else {
                str(binding["trial_number"]): int(binding["nsga2_generation"])
                for binding in binding_rows
            }
        ),
    }
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validated_at = existing.pop("validated_at", None)
        if not isinstance(validated_at, str) or not validated_at or existing != payload:
            raise RuntimeError(f"Terminal chunk validation receipt differs: {output}.")
        return output
    _atomic_json(
        output,
        {**payload, "validated_at": datetime.now(timezone.utc).isoformat()},
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
    sampler = design.get("screening_sampler")
    studies = [study for study in design.get("studies", []) if study.get("model") == model]
    if sampler is not None and len(studies) != 1:
        raise RuntimeError(f"Frozen design does not contain exactly one {model} study.")
    study = studies[0] if studies else {"model": model, "study_name": chunk.get("study_name")}
    expected_hparams = study.get("hparams_search", f"{model}_shared_optuna")
    expected_schema = 2 if sampler is not None else 1
    invocation = chunk.get("sampler_invocation")
    resolved_sampler = (
        None
        if sampler is None or not isinstance(invocation, dict)
        else {**sampler, "seed": invocation.get("resolved_seed")}
    )
    if sampler is not None and not isinstance(invocation, dict):
        raise RuntimeError(f"Terminal chunk has no sampler invocation: {chunk_path}.")
    if (
        chunk.get("schema_version") != expected_schema
        or chunk.get("model") != model
        or chunk.get("controller_status") != "complete"
        or not chunk.get("completed_at")
        or int(chunk.get("requested_trials", 0)) <= 0
    ):
        raise RuntimeError(f"Terminal chunk manifest is incomplete: {chunk_path}.")
    if sampler is not None and (
        chunk.get("study_name") != study["study_name"]
        or chunk.get("hparams_search") != expected_hparams
        or chunk.get("screening_sampler_base") != sampler
        or chunk.get("screening_sampler_base_sha256") != _json_sha256(sampler)
        or chunk.get("screening_sampler") != resolved_sampler
        or chunk.get("screening_sampler_sha256") != _json_sha256(resolved_sampler)
        or chunk.get("screening_sampler_seed_policy") != NSGAII_SEED_POLICY
        or chunk.get("screening_sampler_seed_policy_sha256") != _json_sha256(NSGAII_SEED_POLICY)
        or not isinstance(invocation.get("expected_generation"), int)
        or not isinstance(invocation.get("starting_optuna_trial_number"), int)
        or invocation.get("resolved_seed")
        != NSGAII_SEED_POLICY["base_seed"]
        + NSGAII_SEED_POLICY["model_offsets"][model]
        + invocation["starting_optuna_trial_number"]
        or f"hparams_search={expected_hparams}" not in chunk.get("command", [])
        or f"hydra.sweeper.sampler.seed={invocation['resolved_seed']}"
        not in chunk.get("command", [])
    ):
        raise RuntimeError(f"Terminal chunk sampler provenance differs: {chunk_path}.")
    bindings_path = chunk_path.parent / "trial_bindings.json"
    if not bindings_path.is_file():
        raise FileNotFoundError(f"Terminal trial bindings are missing: {bindings_path}.")
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    rows = bindings.get("bindings", [])
    if (
        bindings.get("schema_version") != expected_schema
        or bindings.get("model") != model
        or len(rows) != int(chunk["requested_trials"])
        or any(binding.get("optimized_metric_artifact") is None for binding in rows)
    ):
        raise RuntimeError(f"Terminal trial bindings are incomplete: {bindings_path}.")
    if sampler is not None and (
        bindings.get("study_name") != study["study_name"]
        or bindings.get("hparams_search") != expected_hparams
        or bindings.get("screening_sampler_base") != sampler
        or bindings.get("screening_sampler") != resolved_sampler
        or bindings.get("screening_sampler_sha256") != _json_sha256(resolved_sampler)
        or bindings.get("screening_sampler_seed_policy") != NSGAII_SEED_POLICY
        or bindings.get("sampler_invocation") != invocation
    ):
        raise RuntimeError(f"Terminal binding sampler provenance differs: {bindings_path}.")
    validation = _validate_completed_search_bindings(root, design, chunk_path, bindings_path, rows)
    return _sha256(validation), _sha256(bindings_path)


def _completed_canary_artifacts(
    root: Path, design: dict[str, Any], row: dict[str, str]
) -> tuple[str, str]:
    """Authenticate one disposable four-trial packed canary."""
    model = row["model"]
    if not str(row.get("attempt", "")).isdigit() or int(row["attempt"]) <= 0:
        raise RuntimeError(f"Production canary has no valid attempt identity: {row}.")
    attempt = int(row["attempt"])
    canary_root = root / "production_canaries" / model / f"attempt_{attempt:04d}"
    success_path = canary_root / "success.json"
    bindings_path = canary_root / "trial_bindings.json"
    if not success_path.is_file() or not bindings_path.is_file():
        raise FileNotFoundError(f"Packed canary artifacts are missing for {model}.")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    resolved_seed = NSGAII_SEED_POLICY["base_seed"] + NSGAII_SEED_POLICY["model_offsets"][model]
    resolved_sampler = {**NSGAII_SCREENING_SAMPLER, "seed": resolved_seed}
    invocation = {
        "starting_optuna_trial_number": 0,
        "expected_generation": 0,
        "resolved_seed": resolved_seed,
    }
    expected_study_name = (
        f"physics_{model}_production_canary_{design['code_commit'][:12]}_attempt_{attempt:04d}"
    )
    if (
        success.get("schema_version") != 2
        or str(success.get("slurm_job_id")) != row["job_id"]
        or success.get("model") != model
        or success.get("attempt") != attempt
        or success.get("trials") != 4
        or success.get("epochs") != 50
        or success.get("code_commit") != design["code_commit"]
        or success.get("bindings") != str(bindings_path.resolve())
        or success.get("bindings_sha256") != _sha256(bindings_path)
        or success.get("screening_sampler_base") != NSGAII_SCREENING_SAMPLER
        or success.get("screening_sampler_base_sha256") != _json_sha256(NSGAII_SCREENING_SAMPLER)
        or success.get("screening_sampler") != resolved_sampler
        or success.get("screening_sampler_sha256") != _json_sha256(resolved_sampler)
        or success.get("screening_sampler_seed_policy") != NSGAII_SEED_POLICY
        or success.get("screening_sampler_seed_policy_sha256") != _json_sha256(NSGAII_SEED_POLICY)
        or success.get("sampler_invocation") != invocation
        or "trainer.min_epochs=50" not in success.get("command", [])
        or "trainer.max_epochs=50" not in success.get("command", [])
        or any("limit_train_batches" in item for item in success.get("command", []))
        or bindings.get("schema_version") != 2
        or bindings.get("model") != model
        or bindings.get("study_name") != expected_study_name
        or bindings.get("screening_sampler_base") != NSGAII_SCREENING_SAMPLER
        or bindings.get("screening_sampler_base_sha256") != _json_sha256(NSGAII_SCREENING_SAMPLER)
        or bindings.get("screening_sampler") != resolved_sampler
        or bindings.get("screening_sampler_sha256") != _json_sha256(resolved_sampler)
        or bindings.get("screening_sampler_seed_policy") != NSGAII_SEED_POLICY
        or bindings.get("screening_sampler_seed_policy_sha256") != _json_sha256(NSGAII_SEED_POLICY)
        or bindings.get("sampler_invocation") != invocation
        or len(bindings.get("bindings", [])) != 4
        or any(
            binding.get("state") != "COMPLETE"
            or binding.get("nsga2_generation") != 0
            or binding.get("resolved_sampler_seed") != resolved_seed
            or binding.get("optimized_metric_artifact") is None
            for binding in bindings.get("bindings", [])
        )
    ):
        raise RuntimeError(f"Packed canary provenance is incomplete for {model}.")
    return _sha256(success_path), _sha256(bindings_path)


def _retrain_receipt_path(root: Path, scheduler_job_id: str) -> Path:
    """Return the immutable receipt path for one exact Slurm array task."""
    parts = scheduler_job_id.split("_")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise RuntimeError(f"Invalid retrain task job ID: {scheduler_job_id}.")
    return root / "state" / "retrain_receipts" / f"{scheduler_job_id}.json"


def _receipt_segment_artifacts(
    root: Path, index: int, snapshot: dict[str, Any]
) -> list[dict[str, str]]:
    """Authenticate and enumerate immutable files named by a state snapshot."""
    segment_root = (root / "retraining" / "restart_state" / f"{index:04d}" / "segments").resolve()
    artifacts: list[dict[str, str]] = []
    for segment in snapshot.get("segments", []):
        for key in ("continuation_checkpoint", "resolved_config"):
            path = Path(str(segment.get(key, ""))).resolve()
            try:
                path.relative_to(segment_root)
            except ValueError as error:
                raise RuntimeError(
                    f"Retrain receipt artifact escapes {segment_root}: {path}"
                ) from error
            expected = str(segment.get(f"{key}_sha256", ""))
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"Retrain receipt artifact is missing or changed: {path}")
            artifacts.append({"role": key, "path": str(path), "sha256": expected})
    return artifacts


def _attempt_scheduler_job_id(attempt: dict[str, Any]) -> str | None:
    """Return the exact Slurm array-task identity recorded by one attempt."""
    array_job_id = attempt.get("slurm_array_job_id")
    array_task_id = attempt.get("slurm_array_task_id")
    if array_job_id is None or array_task_id is None:
        return None
    return f"{array_job_id}_{array_task_id}"


def _segment_snapshot_for_job(state: dict[str, Any], scheduler_job_id: str) -> dict[str, Any]:
    """Project later mutable state back to the exact boundary owned by one task."""
    attempts = state.get("attempts", [])
    task_attempts = [
        (position, attempt)
        for position, attempt in enumerate(attempts)
        if _attempt_scheduler_job_id(attempt) == scheduler_job_id
    ]
    matches = [
        (position, attempt)
        for position, attempt in task_attempts
        if attempt.get("phase") == "training" and attempt.get("completed_at") is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Restart state has {len(matches)} completed segments for scheduler task "
            f"{scheduler_job_id}."
        )
    position, attempt = matches[0]
    attempt_number = int(attempt.get("attempt_number", -1))
    boundary = int(attempt.get("target_epoch_boundary", -1))
    segments = [
        segment
        for segment in state.get("segments", [])
        if int(segment.get("epoch_boundary", -1)) <= boundary
    ]
    if (
        not segments
        or int(segments[-1].get("epoch_boundary", -1)) != boundary
        or int(segments[-1].get("attempt_number", -2)) != attempt_number
    ):
        raise RuntimeError(f"Scheduler task {scheduler_job_id} has no exact segment boundary.")
    snapshot = json.loads(json.dumps(state, allow_nan=False))
    snapshot["attempts"] = snapshot["attempts"][: position + 1]
    snapshot["segments"] = snapshot["segments"][: len(segments)]
    snapshot.pop("completed_at", None)
    return snapshot


def _scheduler_identity_matches(
    scheduler_identity: Any,
    task_map_sha256: str,
    local_task_id: int,
    retrain_index: int,
) -> bool:
    """Authenticate an attempt's optional mapped-array identity."""
    if not task_map_sha256:
        return scheduler_identity is None
    return (
        isinstance(scheduler_identity, dict)
        and scheduler_identity.get("mode") == "mapped"
        and scheduler_identity.get("task_map_sha256") == task_map_sha256
        and scheduler_identity.get("local_task_id") == local_task_id
        and scheduler_identity.get("retrain_index") == retrain_index
    )


def _validate_retrain_receipt(
    root: Path, design: dict[str, Any], ledger_row: dict[str, str], receipt_path: Path
) -> None:
    """Fail closed unless one immutable receipt seals its original task boundary."""
    expected_path = _retrain_receipt_path(root, ledger_row["job_id"])
    if receipt_path.resolve() != expected_path.resolve() or not receipt_path.is_file():
        raise RuntimeError(f"Retrain receipt has the wrong path: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_sha256 = _sha256(receipt_path)
    if (ledger_row.get("success_sha256") or "") not in {"", receipt_sha256}:
        raise RuntimeError(f"Retrain receipt changed after ledger sealing: {receipt_path}")
    index = _retrain_index_for_ledger_row(root, design, ledger_row)
    manifest_path = _retrain_manifest_path(root, index)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("retrain_runs", [])
    expected_row = runs[index] if 0 <= index < len(runs) else None
    local_task_id = int(ledger_row["job_id"].rsplit("_", maxsplit=1)[1])
    if (
        receipt.get("schema_version") != 1
        or receipt.get("code_commit") != design["code_commit"]
        or receipt.get("campaign_root") != str(root.resolve())
        or receipt.get("scheduler_job_id") != ledger_row["job_id"]
        or receipt.get("local_task_id") != local_task_id
        or receipt.get("task_map_sha256") != ledger_row.get("bindings_sha256", "")
        or receipt.get("retrain_index") != index
        or receipt.get("model") != ledger_row.get("model")
        or receipt.get("manifest") != str(manifest_path.resolve())
        or receipt.get("manifest_sha256") != _sha256(manifest_path)
        or expected_row is None
        or receipt.get("row") != expected_row
        or receipt.get("row_sha256") != _json_sha256(expected_row)
    ):
        raise RuntimeError(f"Retrain receipt differs from its frozen task: {receipt_path}")

    evidence = receipt.get("evidence", {})
    if evidence.get("kind") == "final_marker":
        marker = root / "retraining" / "markers" / f"{index:04d}.json"
        marker_snapshot = (
            json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        )
        marker_job_id = _attempt_scheduler_job_id(marker_snapshot)
        if (
            evidence.get("path") != str(marker.resolve())
            or not marker.is_file()
            or evidence.get("sha256") != _sha256(marker)
            or evidence.get("snapshot") != marker_snapshot
            or marker_job_id != ledger_row["job_id"]
            or not _scheduler_identity_matches(
                marker_snapshot.get("scheduler_identity"),
                ledger_row.get("bindings_sha256", ""),
                local_task_id,
                index,
            )
            or not _validate_retrain_marker(marker, expected_row)
        ):
            raise RuntimeError(f"Retrain final-marker receipt is invalid: {receipt_path}")
        return
    if evidence.get("kind") != "training_segment":
        raise RuntimeError(f"Retrain receipt has an unknown evidence kind: {receipt_path}")

    snapshot = evidence.get("snapshot")
    if not isinstance(snapshot, dict) or evidence.get("snapshot_sha256") != _json_sha256(snapshot):
        raise RuntimeError(f"Retrain receipt state snapshot changed: {receipt_path}")
    identity = snapshot.get("identity", {})
    segments = snapshot.get("segments", [])
    attempts = snapshot.get("attempts", [])
    boundaries = [int(segment.get("epoch_boundary", -1)) for segment in segments]
    from scripts import physics_background_pairing_campaign as campaign

    expected_boundaries = list(
        range(
            campaign.RETRAIN_SEGMENT_EPOCHS,
            campaign.RETRAIN_EPOCHS + 1,
            campaign.RETRAIN_SEGMENT_EPOCHS,
        )
    )[: len(boundaries)]
    latest = attempts[-1] if attempts else {}
    observed_job_id = _attempt_scheduler_job_id(latest)
    scheduler_identity = latest.get("scheduler_identity")
    task_map_sha256 = ledger_row.get("bindings_sha256", "")
    scheduler_identity_valid = _scheduler_identity_matches(
        scheduler_identity, task_map_sha256, local_task_id, index
    )
    if (
        snapshot.get("schema_version") != 1
        or identity.get("code_commit") != design["code_commit"]
        or identity.get("manifest") != str(manifest_path.resolve())
        or identity.get("manifest_sha256") != _sha256(manifest_path)
        or identity.get("retrain_index") != index
        or identity.get("row") != expected_row
        or identity.get("row_sha256") != _json_sha256(expected_row)
        or snapshot.get("completed_at") is not None
        or not boundaries
        or boundaries != expected_boundaries
        or latest.get("phase") != "training"
        or latest.get("completed_at") is None
        or int(latest.get("target_epoch_boundary", -1)) != boundaries[-1]
        or int(segments[-1].get("attempt_number", -1)) != int(latest.get("attempt_number", -2))
        or observed_job_id != ledger_row["job_id"]
        or not scheduler_identity_valid
    ):
        raise RuntimeError(f"Retrain receipt state boundary is invalid: {receipt_path}")
    artifacts = _receipt_segment_artifacts(root, index, snapshot)
    if evidence.get("segment_artifacts") != artifacts:
        raise RuntimeError(f"Retrain receipt artifact inventory differs: {receipt_path}")


def _completed_retrain_marker(root: Path, design: dict[str, Any], row: dict[str, str]) -> str:
    """Seal or revalidate one task's immutable final/segment receipt."""
    receipt_path = _retrain_receipt_path(root, row["job_id"])
    if receipt_path.is_file():
        _validate_retrain_receipt(root, design, row, receipt_path)
        return _sha256(receipt_path)
    if row.get("success_sha256"):
        raise RuntimeError(
            f"Ledger names a retrain success receipt that is missing: {receipt_path}"
        )

    index = _retrain_index_for_ledger_row(root, design, row)
    manifest_path = _retrain_manifest_path(root, index)
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
    if marker.is_file():
        if not _validate_retrain_marker(marker, runs[index]):
            raise RuntimeError(f"Successful retrain task has an invalid marker: {row['job_id']}.")
        marker_snapshot = json.loads(marker.read_text(encoding="utf-8"))
        local_task_id = int(row["job_id"].rsplit("_", maxsplit=1)[1])
        if _attempt_scheduler_job_id(marker_snapshot) == row["job_id"] and (
            _scheduler_identity_matches(
                marker_snapshot.get("scheduler_identity"),
                row.get("bindings_sha256", ""),
                local_task_id,
                index,
            )
        ):
            evidence = {
                "kind": "final_marker",
                "path": str(marker.resolve()),
                "sha256": _sha256(marker),
                "snapshot": marker_snapshot,
            }
        else:
            state_path = Path(str(marker_snapshot.get("restart_state_artifact", "")))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            snapshot = _segment_snapshot_for_job(state, row["job_id"])
            evidence = {
                "kind": "training_segment",
                "source_state": str(state_path.resolve()),
                "snapshot": snapshot,
                "snapshot_sha256": _json_sha256(snapshot),
                "segment_artifacts": _receipt_segment_artifacts(root, index, snapshot),
            }
    else:
        from scripts import physics_background_pairing_campaign as campaign

        try:
            progress = campaign._validate_retrain_progress(root, index, runs[index])
        except Exception as error:
            raise RuntimeError(
                f"Successful retrain task has neither a valid marker nor segment: {row['job_id']}."
            ) from error
        state = json.loads(progress.read_text(encoding="utf-8"))
        snapshot = _segment_snapshot_for_job(state, row["job_id"])
        evidence = {
            "kind": "training_segment",
            "source_state": str(progress.resolve()),
            "snapshot": snapshot,
            "snapshot_sha256": _json_sha256(snapshot),
            "segment_artifacts": _receipt_segment_artifacts(root, index, snapshot),
        }
    receipt = {
        "schema_version": 1,
        "code_commit": design["code_commit"],
        "campaign_root": str(root.resolve()),
        "scheduler_job_id": row["job_id"],
        "local_task_id": int(row["job_id"].rsplit("_", maxsplit=1)[1]),
        "task_map_sha256": row.get("bindings_sha256", ""),
        "retrain_index": index,
        "model": row["model"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "row": runs[index],
        "row_sha256": _json_sha256(runs[index]),
        "evidence": evidence,
    }
    _atomic_json(receipt_path, receipt)
    _validate_retrain_receipt(root, design, row, receipt_path)
    return _sha256(receipt_path)


def reconcile(root: Path, repository: Path, audit_sealed: bool = False) -> list[dict[str, str]]:
    """Serialize reconciliation against submissions and stale-trial repair."""
    with _operation_lock(root):
        return _reconcile_locked(root, repository, audit_sealed)


def retire_unmaterialized_retrains(root: Path, repository: Path, array_job_id: str) -> Path:
    """Seal canceled array placeholders that Slurm never materialized.

    Slurm may omit accounting rows for array elements canceled before they obtain an allocation.
    The submission ledger intentionally records every requested element, so those placeholders need
    explicit, hash-bound retirement before another array can be submitted.
    """
    with _operation_lock(root):
        if not array_job_id.isdigit():
            raise ValueError("Retrain array job ID must be numeric.")
        design = _design(root)
        _authenticate_execution_repository(repository, design)
        if array_job_id in _live_retrain_job_ids():
            raise RuntimeError(f"Retrain array {array_job_id} is still live.")
        ledger_path, rows = _ledger(root)
        array_rows = [
            row
            for row in rows
            if row.get("kind") == "retrain"
            and row["job_id"].split("_", maxsplit=1)[0] == array_job_id
        ]
        if not array_rows:
            raise RuntimeError(f"Retrain array {array_job_id} is absent from the ledger.")
        scheduler_accounting = _accounting([array_job_id, *(row["job_id"] for row in array_rows)])
        parent_accounting = scheduler_accounting.get(array_job_id)
        accounting = {
            job_id: observed
            for job_id, observed in scheduler_accounting.items()
            if job_id != array_job_id
        }
        terminal_parent = (
            parent_accounting is not None
            and _normalized_state(parent_accounting["state"]) in TERMINAL_STATES
        )
        if (not accounting and not terminal_parent) or any(
            _normalized_state(row["state"]) not in TERMINAL_STATES for row in accounting.values()
        ):
            raise RuntimeError(f"Retrain array {array_job_id} lacks terminal accounting evidence.")
        ledger_by_job = {row["job_id"]: row for row in array_rows}
        if any(
            _normalized_state(ledger_by_job[job_id].get("state", ""))
            != _normalized_state(observed["state"])
            for job_id, observed in accounting.items()
        ):
            raise RuntimeError(
                f"Retrain array {array_job_id} must be reconciled before retirement."
            )
        targets = [
            row
            for row in array_rows
            if _normalized_state(row.get("state", "")) in ACTIVE_STATES
            and row["job_id"] not in accounting
        ]
        receipt_path = (
            root / "state" / "unmaterialized_retrain_retirements" / f"{array_job_id}.json"
        )
        if not targets:
            if not receipt_path.is_file():
                raise RuntimeError(
                    f"Retrain array {array_job_id} has no unmaterialized active placeholders."
                )
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            digest = existing.pop("content_sha256", None)
            if (
                digest != _json_sha256(existing)
                or existing.get("array_job_id") != array_job_id
                or existing.get("campaign_root") != str(root.resolve())
                or existing.get("code_commit") != design["code_commit"]
            ):
                raise RuntimeError(f"Retirement receipt changed: {receipt_path}")
            return receipt_path
        materialized_missing = [row["job_id"] for row in targets if Path(row["log_path"]).exists()]
        if materialized_missing:
            raise RuntimeError(
                "Refusing to retire retrain tasks with scheduler logs but no accounting: "
                f"{materialized_missing}."
            )
        retired_at = datetime.now(timezone.utc).isoformat()
        receipt = {
            "schema_version": 1,
            "kind": "unmaterialized_retrain_array_retirement",
            "campaign_root": str(root.resolve()),
            "code_commit": design["code_commit"],
            "array_job_id": array_job_id,
            "retired_at": retired_at,
            "terminal_accounting": scheduler_accounting,
            "retired_tasks": [
                {
                    "job_id": row["job_id"],
                    "original_log_path": row["log_path"],
                    "bindings_sha256": row.get("bindings_sha256", ""),
                }
                for row in targets
            ],
        }
        receipt["content_sha256"] = _json_sha256(receipt)
        if receipt_path.exists():
            raise FileExistsError(f"Refusing to replace retirement receipt: {receipt_path}")
        else:
            _atomic_json(receipt_path, receipt)
        receipt_sha256 = _sha256(receipt_path)
        for row in targets:
            row.update(
                {
                    "state": "CANCELLED",
                    "elapsed": "",
                    "exit_code": "",
                    "started": "",
                    "ended": retired_at,
                    "log_path": str(receipt_path.resolve()),
                    "log_sha256": receipt_sha256,
                    "success_sha256": "",
                }
            )
        _atomic_csv(ledger_path, rows)
        return receipt_path


def _terminal_row_is_sealed(row: dict[str, str]) -> bool:
    """Return whether routine reconciliation can trust one immutable terminal record."""
    state = _normalized_state(row.get("state", ""))
    if state not in TERMINAL_STATES or not row.get("log_sha256"):
        return False
    successful = state == "COMPLETED" and row.get("exit_code") == "0:0"
    if not successful:
        return True
    if row.get("kind") in {"production_canary", "search_chunk"}:
        return bool(row.get("success_sha256") and row.get("bindings_sha256"))
    if row.get("kind") == "retrain":
        return bool(row.get("success_sha256"))
    if row.get("kind") == "retrain_superseded":
        return bool(row.get("success_sha256") and row.get("bindings_sha256"))
    return True


def _validate_superseded_retrain_row(root: Path, row: dict[str, str]) -> None:
    """Bind one superseded ledger row to its immutable reset and archived evidence."""
    expected_sha256 = row.get("success_sha256", "")
    matches = [
        path
        for path in (root / "state" / "retrain_candidate_resets").glob("*.json")
        if _sha256(path) == expected_sha256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Superseded retrain row has no unique reset receipt: {row['job_id']}.")
    reset = json.loads(matches[0].read_text(encoding="utf-8"))
    originals = reset.get("original_ledger_rows", [])
    original = next((item for item in originals if item.get("job_id") == row["job_id"]), None)
    if (
        reset.get("schema_version") != 1
        or reset.get("kind") != "retrain_candidate_reset"
        or row["job_id"] not in reset.get("superseded_job_ids", [])
        or original is None
        or original.get("kind") != "retrain"
        or any(
            row.get(field) != original.get(field)
            for field in LEDGER_FIELDS
            if field not in {"kind", "success_sha256"}
        )
    ):
        raise RuntimeError(f"Superseded retrain ledger provenance differs: {row['job_id']}.")
    incident_id = _recovery_incident_id(str(reset.get("incident_id", "")))
    audit = Path(str(reset.get("audit", "")))
    intent = Path(str(reset.get("intent", "")))
    if (
        reset.get("campaign_root") != str(root.resolve())
        or audit != (root / "state" / "retrain_candidate_audits" / f"{incident_id}.json").resolve()
        or not audit.is_file()
        or _sha256(audit) != reset.get("audit_sha256")
        or intent
        != (root / "state" / "retrain_candidate_reset_intents" / f"{incident_id}.json").resolve()
        or not intent.is_file()
        or _sha256(intent) != reset.get("intent_sha256")
    ):
        raise RuntimeError(f"Superseded retrain reset provenance differs: {row['job_id']}.")
    archived_receipt = False
    for archive in reset.get("archives", []):
        source = _contained(Path(str(archive.get("source", ""))), root, "recovery source")
        target = Path(str(archive.get("target", "")))
        _contained(
            target,
            root / "state" / "retrain_candidate_recovery_archives" / incident_id,
            "recovery archive",
        )
        observed = _archive_inventory(target)
        expected = {
            key: archive.get(key) for key in ("kind", "directories", "files", "content_sha256")
        }
        if source.exists() or archive.get("source_absent") is not True or observed != expected:
            raise RuntimeError(f"Superseded retrain receipt archive changed: {target}")
        archived_receipt |= archive.get("role") == f"retrain_receipt:{row['job_id']}"
    successful = original.get("state") == "COMPLETED" and original.get("exit_code") == "0:0"
    if successful and not archived_receipt:
        raise RuntimeError(f"Superseded successful retrain receipt is absent: {row['job_id']}.")


def _reconcile_locked(
    root: Path, repository: Path, audit_sealed: bool = False
) -> list[dict[str, str]]:
    """Reconcile scheduler state and seal terminal Slurm logs by SHA-256."""
    design = _design(root)
    _authenticate_execution_repository(repository, design)
    ledger_path, rows = _ledger(root)
    query_rows = (
        rows if audit_sealed else [row for row in rows if not _terminal_row_is_sealed(row)]
    )
    accounting = _accounting([row["job_id"] for row in query_rows])
    for row in rows:
        sealed = _terminal_row_is_sealed(row)
        if row["job_id"] in accounting:
            row.update(accounting[row["job_id"]])
        elif not (audit_sealed and sealed):
            continue
        state = row["state"]
        if state not in ACTIVE_STATES and state not in TERMINAL_STATES:
            raise RuntimeError(f"Unknown scheduler state {state!r} for job {row['job_id']}.")
        if state in TERMINAL_STATES:
            log_path = Path(row["log_path"])
            if not log_path.is_file():
                raise FileNotFoundError(f"Terminal job log is missing: {log_path}")
            log_sha256 = _sha256(log_path)
            if row.get("log_sha256") and row["log_sha256"] != log_sha256:
                raise RuntimeError(f"Terminal job log changed after sealing: {log_path}")
            row["log_sha256"] = log_sha256
            if (
                state == "COMPLETED"
                and row["exit_code"] == "0:0"
                and row.get("kind") in {"production_canary", "search_chunk"}
            ):
                if audit_sealed or not row.get("success_sha256") or not row.get("bindings_sha256"):
                    if row.get("kind") == "production_canary":
                        evidence = _completed_canary_artifacts(root, design, row)
                    else:
                        evidence = _completed_search_artifacts(root, design, row)
                    sealed_evidence = (row.get("success_sha256"), row.get("bindings_sha256"))
                    if audit_sealed and all(sealed_evidence) and evidence != sealed_evidence:
                        raise RuntimeError(
                            f"Terminal {row.get('kind')} artifacts changed after sealing: "
                            f"{row['job_id']}."
                        )
                    row["success_sha256"], row["bindings_sha256"] = evidence
            elif (
                state == "COMPLETED" and row["exit_code"] == "0:0" and row.get("kind") == "retrain"
            ):
                if audit_sealed or not row.get("success_sha256"):
                    row["success_sha256"] = _completed_retrain_marker(root, design, row)
            elif row.get("kind") == "retrain_superseded":
                _validate_superseded_retrain_row(root, row)
            if not sealed:
                # Keep new authentication work durable if a later row fails or is interrupted.
                _atomic_csv(ledger_path, rows)
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


def _validate_frozen_manifest_contract(
    root: Path, design: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Authenticate the immutable 56-front manifest against the frozen studies."""
    from scripts import physics_background_pairing_report as report

    if (
        manifest.get("schema_version") != 2
        or manifest.get("code_commit") != design["code_commit"]
        or manifest.get("selection_uses_downstream_anomalies") is not False
    ):
        raise RuntimeError("Frozen Pareto manifest violates the campaign contract.")
    expected_fronts = {
        (str(study["model"]), str(objective["id"]))
        for study in design["studies"]
        for objective in study["objectives"]
    }
    if len(expected_fronts) != 56:
        raise RuntimeError("Frozen campaign design does not contain exactly 56 logical fronts.")
    memberships = manifest.get("front_memberships", [])
    observed_fronts = {
        (str(row.get("model")), str(row.get("logical_objective"))) for row in memberships
    }
    if observed_fronts != expected_fronts:
        raise RuntimeError("Frozen manifest does not contain every logical front.")
    if [row.get("membership_index") for row in memberships] != list(range(len(memberships))):
        raise RuntimeError("Frozen front membership indices are not exact and contiguous.")
    retrains = manifest.get("retrain_runs", [])
    if [row.get("retrain_index") for row in retrains] != list(range(len(retrains))):
        raise RuntimeError("Frozen retrain indices are not exact and contiguous.")
    if any(
        int(row.get("seed", -1)) != 123
        or int(row.get("epochs", -1)) != int(design["retrain_epochs"])
        or int(row.get("epochs", -1)) != 200
        for row in retrains
    ):
        raise RuntimeError("Frozen retrains do not use seed 123 and exactly 200 epochs.")
    retrain_indices = {int(row["retrain_index"]) for row in retrains}
    if any(int(row.get("retrain_index", -1)) not in retrain_indices for row in memberships):
        raise RuntimeError("A frozen front membership has no retrain row.")
    report._validate_manifest_against_studies(root, design, manifest)


def freeze_campaign(root: Path, repository: Path) -> Path:
    """Freeze only an authenticated, exact, scheduler- and database-quiescent search."""
    with _operation_lock(root):
        return _freeze_campaign_locked(root, repository)


def _freeze_campaign_locked(root: Path, repository: Path) -> Path:
    """Run the frozen front builder and seal its complete transition evidence."""
    from scripts import physics_background_pairing_report as report

    design = report._design(root)
    _authenticate_execution_repository(repository, design)
    ledger_rows = _reconcile_locked(root, repository, True)
    for model in MODELS:
        _assert_no_live_search_overlap(ledger_rows, model)
        _assert_database_quiescent(root, model)
    _assert_no_live_retrain_overlap(ledger_rows)
    active = [
        row for row in ledger_rows if _normalized_state(row.get("state", "")) in ACTIVE_STATES
    ]
    if active:
        raise RuntimeError(f"Campaign jobs remain active at freeze: {active}.")

    status_rows, trial_rows = report._trial_exports(root, design)
    target = int(design["target_trials_per_study"])
    if target != TARGET_TRIALS:
        raise RuntimeError(f"Frozen design target differs from {TARGET_TRIALS} trials.")
    if len(status_rows) != len(MODELS) or {str(row["model"]) for row in status_rows} != set(
        MODELS
    ):
        raise RuntimeError("Freeze status does not exactly cover the six model studies.")
    for row in status_rows:
        if (
            int(row["usable_complete"]) != target
            or int(row["selected_for_freeze"]) != target
            or int(row["running"]) != 0
            or int(row["waiting"]) != 0
        ):
            raise RuntimeError(f"Study is not exact and quiescent at freeze: {row}.")
    selected = {
        (str(row["model"]), int(row["trial_number"]))
        for row in trial_rows
        if bool(row["selected_for_freeze"])
    }
    expected_selected = len(MODELS) * target
    if len(selected) != expected_selected or any(
        sum(model == selected_model for selected_model, _ in selected) != target
        for model in MODELS
    ):
        raise RuntimeError(
            f"Freeze selected {len(selected)} trials, expected {expected_selected}."
        )
    bindings = report._binding_index(root)
    selected_bindings = {key: bindings[key] for key in selected if key in bindings}
    if set(selected_bindings) != selected or any(
        not row.get("optimized_metric_artifact") for row in selected_bindings.values()
    ):
        missing = sorted(selected - set(selected_bindings))
        raise RuntimeError(f"Selected trials lack complete explicit bindings: {missing[:10]}.")
    report._validate_bindings(root, design, selected_bindings)
    checkpoint_rows, mlflow_rows = report._checkpoint_rows(root, design, include_retrains=False)
    selected_checkpoint_rows = [
        row for row in checkpoint_rows if (str(row["model"]), int(row["trial_number"])) in selected
    ]
    expected_selections = sum(len(study["objectives"]) * target for study in design["studies"])
    if len(selected_checkpoint_rows) != expected_selections:
        raise RuntimeError(
            "Selected optimized-metric artifacts do not contain every logical selection."
        )
    report._annotate_checkpoint_families(design, selected_checkpoint_rows)
    selected_mlflow_rows = [
        row for row in mlflow_rows if (str(row["model"]), int(row["trial_number"])) in selected
    ]
    validated_mlflow = report._validate_mlflow_runs(root, design, selected_mlflow_rows, final=True)
    if len(validated_mlflow) != expected_selected:
        raise RuntimeError("Selected trials do not have exact authenticated MLflow identities.")
    candidate_rows = report._checkpoint_candidate_rows(root, design, selected_bindings, final=True)
    if any(bool(row["required"]) and not bool(row["available"]) for row in candidate_rows):
        raise RuntimeError("A required first-600 checkpoint candidate is unavailable.")

    database_evidence = []
    source_hashes = {}
    for model in MODELS:
        source = root / "studies" / f"{model}.db"
        source_hashes[model] = _sha256(source)
        backup = _create_study_backup(root, model)
        database_evidence.append(
            {
                "model": model,
                "source": str(source.resolve()),
                "source_sha256": source_hashes[model],
                "backup": str(backup.resolve()),
                "backup_sha256": _sha256(backup),
                "backup_manifest": str(backup.with_suffix(".sha256").resolve()),
                "backup_manifest_sha256": _sha256(backup.with_suffix(".sha256")),
            }
        )
    if any(
        _sha256(root / "studies" / f"{model}.db") != digest
        for model, digest in source_hashes.items()
    ):
        raise RuntimeError("A study database changed while freeze backups were created.")

    runner = repository / "scripts" / "physics_background_pairing_campaign.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    command = [sys.executable, str(runner), "--root", str(root), "freeze"]
    subprocess.run(command, cwd=repository, check=True)  # nosec B603
    manifest_path = _retrain_manifest_path(root)
    if not manifest_path.is_file():
        raise RuntimeError("Frozen campaign runner emitted no Pareto manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_frozen_manifest_contract(root, design, manifest)
    if any(
        _sha256(root / "studies" / f"{model}.db") != digest
        for model, digest in source_hashes.items()
    ):
        raise RuntimeError("A study database changed while Pareto fronts were frozen.")

    stable_rows = [row for row in candidate_rows if row["slot"] == "stable_native"]
    output = root / "state" / "freeze_validation.json"
    _atomic_json(
        output,
        {
            "schema_version": 1,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": design["code_commit"],
            "command": command,
            "target_trials_per_study": target,
            "selected_trials": len(selected),
            "logical_fronts": 56,
            "front_memberships": len(manifest["front_memberships"]),
            "unique_retrains": len(manifest["retrain_runs"]),
            "checkpoint_selections": len(selected_checkpoint_rows),
            "mlflow_runs": len(validated_mlflow),
            "candidate_slots": len(candidate_rows),
            "required_candidates_available": sum(
                bool(row["required"]) and bool(row["available"]) for row in candidate_rows
            ),
            "stable_candidates_configured": len(stable_rows),
            "stable_candidates_available": sum(bool(row["available"]) for row in stable_rows),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "study_databases": database_evidence,
        },
    )
    return output


def _validate_aggregate_oracle(
    design: dict[str, Any], manifest: dict[str, Any], oracle: dict[str, Any]
) -> None:
    """Authenticate the exact oracle output against every frozen front membership."""
    if set(oracle) != {
        "schema_version",
        "created_at",
        "oracle_uses_downstream_anomalies",
        "winners",
        "all_front_memberships",
    } or (
        oracle.get("schema_version") != 2
        or oracle.get("oracle_uses_downstream_anomalies") is not True
    ):
        raise RuntimeError("Oracle result does not have the exact schema-2 contract.")

    expected_fronts = [
        (str(study["model"]), str(objective["id"]))
        for study in design["studies"]
        for objective in study["objectives"]
    ]
    if len(expected_fronts) != 56 or len(set(expected_fronts)) != 56:
        raise RuntimeError("Campaign design does not define exactly 56 unique logical fronts.")
    expected_memberships = manifest["front_memberships"]
    memberships = oracle.get("all_front_memberships")
    if not isinstance(memberships, list) or len(memberships) != len(expected_memberships):
        raise RuntimeError("Oracle membership cardinality differs from the frozen manifest.")
    retrains = {int(row["retrain_index"]): row for row in manifest["retrain_runs"]}
    membership_by_index: dict[int, dict[str, Any]] = {}
    for expected, observed in zip(expected_memberships, memberships, strict=True):
        if not isinstance(observed, dict) or any(
            observed.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError("Oracle membership identity differs from the frozen manifest.")
        index = int(expected["membership_index"])
        if (
            index in membership_by_index
            or observed.get("params") != retrains[int(expected["retrain_index"])]["params"]
        ):
            raise RuntimeError("Oracle membership has duplicate identity or changed parameters.")
        values = observed.get("retrained_objective_values")
        efficiencies = observed.get("downstream_efficiencies")
        mean = observed.get("mean_downstream_efficiency")
        if (
            not observed.get("optimized_ckpt_name")
            or not isinstance(values, list)
            or len(values) != 2
            or not all(math.isfinite(float(value)) for value in values)
            or not isinstance(efficiencies, dict)
            or len(efficiencies) != 20
            or not all(
                math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
                for value in efficiencies.values()
            )
            or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
            or not math.isclose(
                float(mean), sum(float(value) for value in efficiencies.values()) / 20
            )
        ):
            raise RuntimeError(f"Oracle membership {index} has invalid downstream values.")
        membership_by_index[index] = observed
    if list(membership_by_index) != list(range(len(expected_memberships))):
        raise RuntimeError("Oracle membership indices are not exact and contiguous.")

    winners = oracle.get("winners")
    if not isinstance(winners, list) or len(winners) != 56:
        raise RuntimeError("Oracle output does not contain exactly 56 winners.")
    observed_fronts = [
        (str(row.get("model")), str(row.get("logical_objective")))
        for row in winners
        if isinstance(row, dict)
    ]
    if observed_fronts != expected_fronts:
        raise RuntimeError("Oracle winners do not exactly follow the 56 campaign fronts.")
    for front, winner in zip(expected_fronts, winners, strict=True):
        candidates = [
            row
            for row in memberships
            if (str(row["model"]), str(row["logical_objective"])) == front
        ]
        if not candidates:
            raise RuntimeError(f"Oracle front has no frozen membership: {front}.")
        expected_winner = max(
            candidates,
            key=lambda row: (
                float(row["mean_downstream_efficiency"]),
                -int(row["membership_index"]),
            ),
        )
        if winner != expected_winner:
            raise RuntimeError(f"Oracle winner does not maximize mean performance: {front}.")


def _validate_aggregate_receipt(
    path: Path,
    design: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    marker_evidence: list[dict[str, Any]],
    oracle_path: Path,
    oracle: dict[str, Any],
    command: list[str],
) -> None:
    """Authenticate an existing completed transition without rerunning aggregation."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "code_commit": design["code_commit"],
        "command": command,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "logical_fronts": 56,
        "front_memberships": len(manifest["front_memberships"]),
        "unique_retrains": len(manifest["retrain_runs"]),
        "retrain_markers": marker_evidence,
        "retrain_markers_sha256": _json_sha256(marker_evidence),
        "oracle": str(oracle_path.resolve()),
        "oracle_sha256": _sha256(oracle_path),
        "oracle_winners": len(oracle["winners"]),
        "oracle_memberships": len(oracle["all_front_memberships"]),
    }
    if set(receipt) != {*expected, "validated_at"} or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Existing aggregate validation receipt is changed or inconsistent.")


def aggregate_campaign(root: Path, repository: Path) -> Path:
    """Aggregate only a complete, quiescent, authenticated retraining campaign."""
    with _operation_lock(root):
        design = _design(root)
        _authenticate_execution_repository(repository, design)
        ledger_rows = _reconcile_locked(root, repository, True)
        _assert_no_live_retrain_overlap(ledger_rows)
        active_retrains = _active_kind_rows(ledger_rows, "retrain")
        if active_retrains:
            raise RuntimeError(f"Retraining jobs remain active at aggregation: {active_retrains}.")

        manifest_path = _retrain_manifest_path(root)
        if not manifest_path.is_file():
            raise FileNotFoundError("Freeze the Pareto fronts before aggregation.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_frozen_manifest_contract(root, design, manifest)
        freeze_path = root / "state" / "freeze_validation.json"
        if not freeze_path.is_file():
            raise RuntimeError("The authenticated freeze transition receipt is missing.")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (
            freeze.get("schema_version") != 1
            or freeze.get("code_commit") != design["code_commit"]
            or freeze.get("logical_fronts") != 56
            or freeze.get("front_memberships") != len(manifest["front_memberships"])
            or freeze.get("unique_retrains") != len(manifest["retrain_runs"])
            or freeze.get("manifest") != str(manifest_path.resolve())
            or freeze.get("manifest_sha256") != _sha256(manifest_path)
        ):
            raise RuntimeError("Freeze receipt does not authenticate the current Pareto manifest.")

        marker_root = root / "retraining" / "markers"
        expected_marker_paths = [
            marker_root / f"{int(row['retrain_index']):04d}.json"
            for row in manifest["retrain_runs"]
        ]
        observed_marker_paths = sorted(marker_root.glob("*.json")) if marker_root.is_dir() else []
        if observed_marker_paths != expected_marker_paths:
            raise RuntimeError("Retraining markers do not exactly cover the frozen retrain rows.")
        marker_evidence = []
        for row, marker_path in zip(manifest["retrain_runs"], expected_marker_paths, strict=True):
            if not _validate_retrain_marker(marker_path, row):
                raise RuntimeError(f"Retraining marker failed validation: {marker_path}.")
            marker_evidence.append(
                {
                    "retrain_index": int(row["retrain_index"]),
                    "path": str(marker_path.resolve()),
                    "sha256": _sha256(marker_path),
                }
            )

        runner = repository / "scripts" / "physics_background_pairing_campaign.py"
        if not runner.is_file():
            raise FileNotFoundError(runner)
        command = [sys.executable, str(runner), "--root", str(root), "aggregate"]
        output = root / "state" / "aggregate_validation.json"
        oracle_path = root / "results" / "oracle_winners.json"
        if output.exists():
            if not oracle_path.is_file():
                raise RuntimeError("Aggregate receipt exists but its oracle result is missing.")
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
            _validate_aggregate_oracle(design, manifest, oracle)
            _validate_aggregate_receipt(
                output,
                design,
                manifest_path,
                manifest,
                marker_evidence,
                oracle_path,
                oracle,
                command,
            )
            return output
        subprocess.run(command, cwd=repository, check=True)  # nosec B603
        if not oracle_path.is_file():
            raise RuntimeError("Campaign runner emitted no oracle result.")
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        _validate_aggregate_oracle(design, manifest, oracle)

        _atomic_json(
            output,
            {
                "schema_version": 1,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "code_commit": design["code_commit"],
                "command": command,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "logical_fronts": 56,
                "front_memberships": len(manifest["front_memberships"]),
                "unique_retrains": len(manifest["retrain_runs"]),
                "retrain_markers": marker_evidence,
                "retrain_markers_sha256": _json_sha256(marker_evidence),
                "oracle": str(oracle_path.resolve()),
                "oracle_sha256": _sha256(oracle_path),
                "oracle_winners": len(oracle["winners"]),
                "oracle_memberships": len(oracle["all_front_memberships"]),
            },
        )
        return output


def _append_failure_event(root: Path, payload: dict[str, Any]) -> None:
    """Append one auditable recovery event while the operation lock is held."""
    path = root / "state" / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _refresh_repaired_bindings(
    root: Path, model: str, repaired: dict[int, dict[str, Any]]
) -> list[dict[str, str]]:
    """Persist FAIL state in any existing explicit bindings for repaired trials."""
    updated = []
    seen: set[int] = set()
    for path in sorted((root / "sweeps" / model).glob("*/trial_bindings.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for row in payload.get("bindings", []):
            trial_number = int(row["trial_number"])
            if trial_number not in repaired:
                continue
            if trial_number in seen:
                raise RuntimeError(f"Duplicate explicit binding for {(model, trial_number)}.")
            seen.add(trial_number)
            row["state"] = "FAIL"
            row["values"] = None
            phase = repaired[trial_number]["nsga2_trial_phase"]
            row["nsga2_trial_phase"] = phase
            if phase == "pre_sampling_failure":
                row["nsga2_generation"] = None
                row["pre_sampling_failure_evidence"] = "stale_repair"
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
                generation = trial.system_attrs.get(NSGA_GENERATION_KEY)
                if type(generation) is int and generation >= 0:
                    phase = "sampled"
                elif (
                    NSGA_GENERATION_KEY not in trial.system_attrs
                    and not trial.params
                    and trial.values is None
                ):
                    phase = "pre_sampling_failure"
                else:
                    raise RuntimeError(
                        f"Trial {(model, number)} has an ambiguous NSGA-II sampling state."
                    )
                before.append(
                    {
                        "trial_number": number,
                        "state": trial.state.name,
                        "params": trial.params,
                        "values": trial.values,
                        "system_attrs": trial.system_attrs,
                        "nsga2_trial_phase": phase,
                        "nsga2_generation": generation,
                    }
                )
            source_sha256 = _sha256(database)
            backup = _create_study_backup(root, model)
            backup_sha256 = _sha256(backup)
            for number in requested:
                study.tell(number, state=optuna.trial.TrialState.FAIL)
            after_trials = {int(trial.number): trial for trial in study.get_trials(deepcopy=False)}
            if any(after_trials[number].state.name != "FAIL" for number in requested):
                raise RuntimeError("Optuna did not persist every requested stale trial as FAIL.")
            before_by_number = {int(row["trial_number"]): row for row in before}
            binding_artifacts = _refresh_repaired_bindings(root, model, before_by_number)
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
                    {
                        "trial_number": number,
                        "state": after_trials[number].state.name,
                        "params": after_trials[number].params,
                        "values": after_trials[number].values,
                        "system_attrs": after_trials[number].system_attrs,
                        "nsga2_trial_phase": before_by_number[number]["nsga2_trial_phase"],
                        "nsga2_generation": before_by_number[number]["nsga2_generation"],
                    }
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
    if args.command == "submit-canary":
        print(
            submit_canary(
                root,
                repository,
                args.model,
                args.data_dir,
                args.raw_data_dir,
                args.pairing_dir,
            )
        )
    elif args.command == "submit-search":
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
    elif args.command == "validate-restart-canary":
        print(validate_restart_canary(root, repository, args.canary_root, args.job_id))
    elif args.command == "reconcile":
        print(json.dumps(reconcile(root, repository), indent=2, sort_keys=True))
    elif args.command == "retire-unmaterialized-retrains":
        print(retire_unmaterialized_retrains(root, repository, args.array_job_id))
    elif args.command == "audit-retrain-candidates":
        print(audit_retrain_candidates(root, repository, args.incident_id))
    elif args.command == "reset-affected-retrains":
        print(reset_affected_retrains(root, repository, args.incident_id, args.audit_sha256))
    elif args.command == "freeze":
        print(freeze_campaign(root, repository))
    elif args.command == "aggregate":
        print(aggregate_campaign(root, repository))
    elif args.command == "repair-stale":
        print(repair_stale(root, repository, args.model, args.trial))
    else:
        print(backup_study(root, repository, args.model))


if __name__ == "__main__":
    main()
