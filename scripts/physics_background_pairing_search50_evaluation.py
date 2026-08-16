#!/usr/bin/env python3
"""Authenticate and run provisional downstream evaluation of frozen search checkpoints.

This runner deliberately produces inputs for later analysis, not scientific conclusions.  It
evaluates the exact search-50 checkpoint selected by every frozen Pareto membership and keeps the
results separate from the final 200-epoch retraining campaign.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404 -- authenticated executable and fixed argv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_ANOMALIES = 20
MODEL_ORDER = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
OAS_MODELS = {"ae", "vae"}
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value with canonical serialization."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting non-object roots."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    """Atomically write one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write a nonempty list of uniform records as CSV."""
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    """Atomically write a text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _git(repository: Path, *args: str) -> str:
    """Run a read-only Git query in a repository."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required for evaluation provenance.")
    result = subprocess.run(  # nosec B603
        [git, "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _authenticate_freeze(root: Path, frozen_repository: Path) -> tuple[dict, dict, dict]:
    """Authenticate the frozen signal-blind selection universe."""
    design_path = root / "design.json"
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    receipt_path = root / "state" / "freeze_validation.json"
    design, manifest, receipt = map(_read_json, (design_path, manifest_path, receipt_path))
    commit = str(design["code_commit"])
    if receipt.get("code_commit") != commit or manifest.get("code_commit") != commit:
        raise RuntimeError("Frozen design, manifest, and receipt disagree on the code commit.")
    if receipt.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("Frozen retrain manifest differs from its freeze receipt.")
    if int(receipt.get("front_memberships", -1)) != len(manifest["front_memberships"]):
        raise RuntimeError("Freeze receipt membership count differs.")
    if int(receipt.get("unique_retrains", -1)) != len(manifest["retrain_runs"]):
        raise RuntimeError("Freeze receipt retrain count differs.")
    frozen_repository = frozen_repository.resolve(strict=True)
    if _git(frozen_repository, "rev-parse", "HEAD") != commit:
        raise RuntimeError("Frozen science repository is not at the campaign commit.")
    if _git(frozen_repository, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Frozen science repository has tracked modifications.")
    return design, manifest, receipt


def _authenticate_execution(
    repository: Path, expected_commit: str | None = None
) -> tuple[Path, str]:
    """Require a clean execution repository with checkpoint-only support."""
    repository = repository.resolve(strict=True)
    commit = _git(repository, "rev-parse", "HEAD")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            "Evaluation execution repository commit differs from the prepared manifest."
        )
    if _git(repository, "status", "--porcelain"):
        raise RuntimeError("Evaluation execution repository must be completely clean.")
    train_source = (repository / "src" / "train.py").read_text(encoding="utf-8")
    if (
        "skip_validation_evaluation" not in train_source
        or "evaluation_callback_allowlist" not in train_source
    ):
        raise RuntimeError(
            "Evaluation repository lacks the required test-only allowlist controls."
        )
    return repository, commit


def _authenticate_inputs(
    design: dict, data_dir: Path, raw_data_dir: Path, pairing_dir: Path
) -> dict[str, Any]:
    """Authenticate immutable data aliases and pairing tables."""
    data_dir, raw_data_dir, pairing_dir = (
        path.resolve(strict=True) for path in (data_dir, raw_data_dir, pairing_dir)
    )
    preflight_path = Path(design["preflight"]).resolve(strict=True)
    if _sha256(preflight_path) != design["preflight_sha256"]:
        raise RuntimeError("Frozen pairing preflight differs from the campaign design.")
    preflight = _read_json(preflight_path)
    if Path(preflight["artifact_dir"]).resolve(strict=True) != pairing_dir:
        raise RuntimeError("Requested pairing directory differs from the frozen preflight.")
    table_hashes = {}
    for row in preflight["tables"]:
        path = Path(row["path"]).resolve(strict=True)
        if not path.is_relative_to(pairing_dir) or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"Pairing table differs from frozen preflight: {path}")
        table_hashes[str(path)] = row["sha256"]
    alias_links = {
        str(path.relative_to(data_dir)): str(path.resolve(strict=True))
        for path in sorted(data_dir.rglob("*"))
        if path.is_symlink()
    }
    return {
        "data_dir": str(data_dir),
        "raw_data_dir": str(raw_data_dir),
        "pairing_dir": str(pairing_dir),
        "data_alias_symlinks": alias_links,
        "pairing_preflight": str(preflight_path),
        "pairing_preflight_sha256": design["preflight_sha256"],
        "pairing_table_hashes": table_hashes,
    }


def _resolve_checkpoint(root: Path, artifact: dict, checkpoint_name: str) -> Path:
    """Resolve exactly one selected checkpoint inside the campaign root."""
    trusted = (root / "runtime" / "checkpoints").resolve(strict=True)
    run_dir = Path(artifact["run_provenance"]["checkpoint_run_dir"]).resolve(strict=True)
    if not run_dir.is_relative_to(trusted):
        raise RuntimeError(f"Checkpoint root escapes campaign runtime: {run_dir}")
    if checkpoint_name == "last":
        paths = [run_dir / "last.ckpt"]
    else:
        paths = sorted((run_dir / checkpoint_name).glob("*.ckpt"))
    if len(paths) != 1 or not paths[0].is_file():
        raise RuntimeError(f"Expected one checkpoint for {checkpoint_name!r} in {run_dir}.")
    return paths[0].resolve(strict=True)


def build_manifest(
    root: Path,
    frozen_repository: Path,
    execution_repository: Path,
    data_dir: Path,
    raw_data_dir: Path,
    pairing_dir: Path,
) -> dict[str, Any]:
    """Build the deterministic authenticated task manifest without writing it."""
    design, frozen, receipt = _authenticate_freeze(root, frozen_repository)
    execution_repository, execution_commit = _authenticate_execution(execution_repository)
    selection_path = root / "results" / "checkpoint_selections.csv"
    with selection_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selection_index: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        if row["phase"] != "search":
            continue
        key = (row["model"], int(row["trial_number"]), row["logical_objective"])
        if key in selection_index:
            raise RuntimeError(f"Duplicate search selection: {key}")
        selection_index[key] = row

    memberships_by_retrain: dict[int, list[dict]] = defaultdict(list)
    for membership in frozen["front_memberships"]:
        memberships_by_retrain[int(membership["retrain_index"])].append(membership)
    tasks = []
    selected_global: set[tuple[str, int, str]] = set()
    for run in sorted(frozen["retrain_runs"], key=lambda row: int(row["retrain_index"])):
        retrain_index = int(run["retrain_index"])
        memberships = sorted(
            memberships_by_retrain[retrain_index], key=lambda row: int(row["membership_index"])
        )
        if {(m["model"], int(m["trial_number"])) for m in memberships} != {
            (run["model"], int(run["trial_number"]))
        }:
            raise RuntimeError(f"Membership/run mismatch at retrain index {retrain_index}.")
        checkpoints: dict[tuple[str, str], dict] = {}
        membership_rows = []
        for membership in memberships:
            key = (
                membership["model"],
                int(membership["trial_number"]),
                membership["logical_objective"],
            )
            selection = selection_index.get(key)
            if selection is None:
                raise RuntimeError(f"Missing checkpoint selection: {key}")
            artifact_path = (root / selection["artifact"]).resolve(strict=True)
            if _sha256(artifact_path) != selection["artifact_sha256"]:
                raise RuntimeError(f"Changed optimized-metric artifact: {artifact_path}")
            artifact = _read_json(artifact_path)
            selected = artifact["selections"][membership["logical_objective"]]
            if selected["optimized_ckpt_name"] != selection["optimized_ckpt_name"]:
                raise RuntimeError(f"Selection/artifact checkpoint mismatch: {key}")
            source = _resolve_checkpoint(root, artifact, selection["optimized_ckpt_name"])
            source_sha = _sha256(source)
            physical_key = (str(source), source_sha)
            if physical_key not in checkpoints:
                checkpoint_id = f"selected_{len(checkpoints):03d}"
                checkpoints[physical_key] = {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_family": selection["checkpoint_family"],
                    "optimized_ckpt_name": selection["optimized_ckpt_name"],
                    "source_path": str(source),
                    "source_sha256": source_sha,
                    "logical_objectives": [],
                }
            checkpoint = checkpoints[physical_key]
            checkpoint["logical_objectives"].append(membership["logical_objective"])
            selected_global.add(
                (membership["model"], int(membership["trial_number"]), str(source))
            )
            membership_rows.append(
                {
                    **membership,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "optimized_ckpt_name": selection["optimized_ckpt_name"],
                    "checkpoint_family": selection["checkpoint_family"],
                    "optimized_metric": [float(x) for x in selected["optimized_metric"]],
                    "optimized_metric_artifact": str(artifact_path),
                    "optimized_metric_artifact_sha256": selection["artifact_sha256"],
                }
            )
        task_index = len(tasks)
        if task_index != retrain_index:
            raise RuntimeError("Retrain indices are not contiguous task indices.")
        model = str(run["model"])
        tasks.append(
            {
                "task_index": task_index,
                "model": model,
                "trial_number": int(run["trial_number"]),
                "params": run["params"],
                "scores": ["native", "residual_oas"] if model in OAS_MODELS else ["native"],
                "checkpoints": list(checkpoints.values()),
                "memberships": membership_rows,
            }
        )
    if len(tasks) != 346 or len(frozen["front_memberships"]) != 591 or len(selected_global) != 421:
        raise RuntimeError(
            "Frozen campaign no longer has the authenticated 346/591/421 cardinality."
        )

    inputs = _authenticate_inputs(design, data_dir, raw_data_dir, pairing_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "provisional_search50_downstream_evaluation_inputs",
        "campaign_root": str(root.resolve(strict=True)),
        "code_commit": design["code_commit"],
        "frozen_science_repository": str(frozen_repository.resolve(strict=True)),
        "evaluation_execution_repository": str(execution_repository),
        "evaluation_execution_commit": execution_commit,
        "frozen_manifest": str((root / "selection" / "pareto_retrain_manifest.json").resolve()),
        "frozen_manifest_sha256": receipt["manifest_sha256"],
        "freeze_receipt": str((root / "state" / "freeze_validation.json").resolve()),
        "freeze_receipt_sha256": _sha256(root / "state" / "freeze_validation.json"),
        "checkpoint_selection_index": str(selection_path.resolve()),
        "checkpoint_selection_index_sha256": _sha256(selection_path),
        "inputs": inputs,
        "expected_anomalies_per_score_checkpoint": EXPECTED_ANOMALIES,
        "tasks": tasks,
    }
    payload["content_sha256"] = _canonical_hash(payload)
    return payload


def _screening_dir(root: Path) -> Path:
    """Return the isolated provisional-evaluation workspace."""
    return root / "screening_evaluation"


def prepare(args: argparse.Namespace) -> None:
    """Persist the immutable checkpoint-evaluation task manifest."""
    manifest = build_manifest(
        args.root.resolve(strict=True),
        args.frozen_repository,
        args.execution_repository,
        args.data_dir,
        args.raw_data_dir,
        args.pairing_dir,
    )
    path = _screening_dir(args.root) / "manifest.json"
    if path.exists():
        if _read_json(path) != manifest:
            raise RuntimeError(f"Refusing to replace a different screening manifest: {path}")
    else:
        _write_json(path, manifest)
    print(
        json.dumps(
            {"manifest": str(path), "sha256": _sha256(path), "tasks": len(manifest["tasks"])},
            sort_keys=True,
        )
    )


def _load_prepared(root: Path) -> dict:
    """Load and reauthenticate the prepared evaluation manifest."""
    path = _screening_dir(root) / "manifest.json"
    manifest = _read_json(path)
    content_hash = manifest.pop("content_sha256", None)
    if content_hash != _canonical_hash(manifest):
        raise RuntimeError("Prepared screening manifest content hash differs.")
    manifest["content_sha256"] = content_hash
    design, _, _ = _authenticate_freeze(root, Path(manifest["frozen_science_repository"]))
    _authenticate_execution(
        Path(manifest["evaluation_execution_repository"]), manifest["evaluation_execution_commit"]
    )
    observed_inputs = _authenticate_inputs(
        design,
        Path(manifest["inputs"]["data_dir"]),
        Path(manifest["inputs"]["raw_data_dir"]),
        Path(manifest["inputs"]["pairing_dir"]),
    )
    if observed_inputs != manifest["inputs"]:
        raise RuntimeError("Prepared immutable input identity differs.")
    return manifest


def _stage_task(root: Path, manifest: dict, task: dict) -> tuple[Path, str, str]:
    """Stage authenticated checkpoint symlinks for one evaluation task."""
    task_dir = _screening_dir(root) / "tasks" / f"{int(task['task_index']):03d}"
    experiment = f"physics_bgpair_search50_eval_{task['model']}"
    run_name = f"trial_{int(task['trial_number']):04d}"
    criterion = (
        task_dir
        / "checkpoint_store"
        / experiment
        / run_name
        / "candidate"
        / "checkpoint"
        / "selected"
    )
    criterion.mkdir(parents=True, exist_ok=True)
    expected_names = set()
    for checkpoint in task["checkpoints"]:
        source = Path(checkpoint["source_path"])
        if _sha256(source) != checkpoint["source_sha256"]:
            raise RuntimeError(f"Selected checkpoint changed: {source}")
        name = f"ds={checkpoint['checkpoint_id']}__selected.ckpt"
        expected_names.add(name)
        target = criterion / name
        if target.is_symlink():
            if target.resolve(strict=True) != source.resolve(strict=True):
                raise RuntimeError(f"Staged symlink points to another checkpoint: {target}")
        elif target.exists():
            raise RuntimeError(f"Staged checkpoint is not a symlink: {target}")
        else:
            target.symlink_to(source.resolve(strict=True))
    observed = {path.name for path in criterion.glob("*.ckpt")}
    if observed != expected_names:
        raise RuntimeError(f"Unexpected staged checkpoints in {criterion}.")
    return task_dir, experiment, run_name


def _values_rows(attempt_dir: Path, task: dict) -> tuple[list[dict], list[dict]]:
    """Read and authenticate exact downstream-efficiency coverage."""
    expected = {
        (checkpoint["checkpoint_id"], score)
        for checkpoint in task["checkpoints"]
        for score in task["scores"]
    }
    found: dict[tuple[str, str], list[dict]] = {}
    files = []
    score_by_callback = {"eff_native": "native", "eff_residual_oas": "residual_oas"}
    for path in sorted(attempt_dir.glob("checkpoint_store/**/plots/test/*/eff_*/values.csv")):
        callback = path.parent.name
        if callback not in score_by_callback:
            raise RuntimeError(f"Unexpected efficiency callback output: {path}")
        checkpoint_name = path.parent.parent.name
        checkpoint_id = checkpoint_name.split("ds=", 1)[-1].split("__", 1)[0]
        key = (checkpoint_id, score_by_callback[callback])
        if key in found:
            raise RuntimeError(f"Duplicate efficiency values for {key}.")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != EXPECTED_ANOMALIES:
            raise RuntimeError(
                f"Expected exactly 20 downstream efficiencies for {key}, got {len(rows)}."
            )
        if len({row["intervention"] for row in rows}) != EXPECTED_ANOMALIES:
            raise RuntimeError(f"Duplicate downstream intervention rows for {key}.")
        if {row["metric"] for row in rows} != {"efficiency_operational"}:
            raise RuntimeError(f"Unexpected efficiency metric rows for {key}.")
        normalized = sorted(
            (
                {
                    "intervention": row["intervention"],
                    "metric": row["metric"],
                    "value": float(row["value"]),
                }
                for row in rows
            ),
            key=lambda row: row["intervention"],
        )
        found[key] = normalized
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "checkpoint_id": checkpoint_id,
                "score": key[1],
                "values_sha256": _canonical_hash(normalized),
            }
        )
    if set(found) != expected:
        raise RuntimeError(
            f"Efficiency output coverage differs: missing={sorted(expected-set(found))}, extra={sorted(set(found)-expected)}"
        )
    values = [
        {
            "checkpoint_id": key[0],
            "score": key[1],
            "rows": found[key],
            "values_sha256": _canonical_hash(found[key]),
        }
        for key in sorted(found)
    ]
    return values, files


def _marker_valid(root: Path, manifest: dict, task: dict, marker: dict) -> bool:
    """Return whether a task marker and referenced values remain valid."""
    if marker.get("manifest_content_sha256") != manifest["content_sha256"] or marker.get(
        "task_sha256"
    ) != _canonical_hash(task):
        return False
    attempt_dir = Path(marker.get("attempt_dir", ""))
    try:
        values, files = _values_rows(attempt_dir, task)
    except (OSError, ValueError, RuntimeError):
        return False
    return marker.get("values") == values and marker.get("value_files") == files


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate all selected physical checkpoints for one trial."""
    root = args.root.resolve(strict=True)
    manifest = _load_prepared(root)
    task = manifest["tasks"][args.task_index]
    if int(task["task_index"]) != args.task_index:
        raise RuntimeError("Task index mismatch.")
    task_dir, experiment, run_name = _stage_task(root, manifest, task)
    marker_path = task_dir / "marker.json"
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker_path.exists() and _marker_valid(root, manifest, task, _read_json(marker_path)):
            print(f"task {args.task_index}: already complete")
            return
        attempt_id = args.attempt_id or f"pid-{os.getpid()}"
        attempt_dir = task_dir / "attempts" / attempt_id
        if attempt_dir.exists() and any(attempt_dir.iterdir()):
            try:
                values, files = _values_rows(task_dir, task)
            except (OSError, ValueError, RuntimeError) as error:
                raise RuntimeError(
                    f"Attempt directory is nonempty but incomplete; choose a new --attempt-id: {attempt_dir}"
                ) from error
        else:
            attempt_dir.mkdir(parents=True, exist_ok=True)
            # The evaluator derives its checkpoint run path from these two names.
            staged_store = task_dir / "checkpoint_store"
            suite = (
                "physics_background_native_oas"
                if task["model"] in OAS_MODELS
                else "physics_background_native"
            )
            command = [
                sys.executable,
                "src/train.py",
                f"experiment=physics/{task['model']}_background_all",
                f"+retrain_suite={suite}",
                "logger=none",
                "trainer=gpu",
                "trainer.devices=[0]",
                "train=false",
                "test=true",
                "skip_validation_evaluation=true",
                "optimized_metric_config=null",
                "optimized_metric_configs=null",
                "evaluation.evaluator.ckpts.last=false",
                "evaluation.evaluator.ckpts.summary=false",
                "evaluation.evaluator.ckpts.single=false",
                "++evaluation.evaluator.ckpts={candidate:{checkpoint:[selected]}}",
                f"paths.checkpoints_dir={staged_store}",
                f"experiment_name={experiment}",
                f"run_name={run_name}",
                f"hydra.run.dir={attempt_dir / 'hydra'}",
            ]
            callback_allowlist = (
                "[anomaly_efficiency,anomaly_efficiency_oas]"
                if task["model"] in OAS_MODELS
                else "[anomaly_efficiency]"
            )
            command.append(f"evaluation_callback_allowlist={callback_allowlist}")
            command.extend(f"{key}={value}" for key, value in sorted(task["params"].items()))
            env = os.environ.copy()
            env.update(
                {
                    "DATA_DIR": manifest["inputs"]["data_dir"],
                    "RAW_DATA_DIR": manifest["inputs"]["raw_data_dir"],
                    "PHYSICS_PAIRING_DIR": manifest["inputs"]["pairing_dir"],
                    "CHECKPOINT_DIR": str(staged_store),
                    "OUTPUT_DIR": str(attempt_dir / "outputs"),
                    "LOG_DIR": str(attempt_dir / "logs"),
                    "HYDRA_FULL_ERROR": "1",
                    "MLFLOW_DISABLE_TELEMETRY": "true",
                }
            )
            subprocess.run(
                command, cwd=manifest["evaluation_execution_repository"], env=env, check=True
            )  # nosec B603
            values, files = _values_rows(task_dir, task)
        marker = {
            "schema_version": SCHEMA_VERSION,
            "task_index": args.task_index,
            "manifest_content_sha256": manifest["content_sha256"],
            "task_sha256": _canonical_hash(task),
            "attempt_dir": str(task_dir.resolve()),
            "hydra_attempt_dir": str(attempt_dir.resolve()),
            "values": values,
            "value_files": files,
            "selected_checkpoint_hashes": {
                row["checkpoint_id"]: row["source_sha256"] for row in task["checkpoints"]
            },
        }
        _write_json(marker_path, marker)
        print(f"task {args.task_index}: complete")


def status(args: argparse.Namespace) -> None:
    """Print authenticated completion status for the task universe."""
    root = args.root.resolve(strict=True)
    manifest = _load_prepared(root)
    complete, invalid = [], []
    for task in manifest["tasks"]:
        marker_path = _screening_dir(root) / "tasks" / f"{task['task_index']:03d}" / "marker.json"
        if not marker_path.exists():
            continue
        if _marker_valid(root, manifest, task, _read_json(marker_path)):
            complete.append(task["task_index"])
        else:
            invalid.append(task["task_index"])
    pending = sorted(set(range(len(manifest["tasks"]))) - set(complete))
    print(
        json.dumps(
            {
                "complete": len(complete),
                "pending": len(pending),
                "invalid": invalid,
                "pending_indices": pending,
            },
            sort_keys=True,
        )
    )


def aggregate_input(args: argparse.Namespace) -> None:
    """Aggregate authenticated task outputs without selecting winners."""
    root = args.root.resolve(strict=True)
    manifest = _load_prepared(root)
    rows = []
    candidate_rows = []
    for task in manifest["tasks"]:
        marker_path = _screening_dir(root) / "tasks" / f"{task['task_index']:03d}" / "marker.json"
        marker = _read_json(marker_path)
        if not _marker_valid(root, manifest, task, marker):
            raise RuntimeError(f"Missing or invalid marker for task {task['task_index']}.")
        values = {(row["checkpoint_id"], row["score"]): row for row in marker["values"]}
        checkpoints = {row["checkpoint_id"]: row for row in task["checkpoints"]}
        for membership in task["memberships"]:
            for score in task["scores"]:
                value = values[(membership["checkpoint_id"], score)]
                for efficiency in value["rows"]:
                    rows.append(
                        {
                            "task_index": task["task_index"],
                            "model": task["model"],
                            "trial_number": task["trial_number"],
                            "membership_index": membership["membership_index"],
                            "logical_objective": membership["logical_objective"],
                            "selection_score": membership["score"],
                            "evaluation_score": score,
                            "metric": membership["metric"],
                            "strategy": membership["strategy"],
                            "checkpoint_id": membership["checkpoint_id"],
                            "optimized_ckpt_name": membership["optimized_ckpt_name"],
                            "checkpoint_family": membership["checkpoint_family"],
                            "intervention": efficiency["intervention"],
                            "efficiency_metric": efficiency["metric"],
                            "efficiency": efficiency["value"],
                            "values_sha256": value["values_sha256"],
                        }
                    )
            checkpoint = checkpoints[membership["checkpoint_id"]]
            selected_values = values[(membership["checkpoint_id"], membership["score"])]
            for efficiency in selected_values["rows"]:
                candidate_rows.append(
                    {
                        "membership_index": membership["membership_index"],
                        "model": task["model"],
                        "logical_objective": membership["logical_objective"],
                        "score": membership["score"],
                        "metric": membership["metric"],
                        "strategy": membership["strategy"],
                        "trial_number": task["trial_number"],
                        "checkpoint": checkpoint["source_path"],
                        "checkpoint_sha256": checkpoint["source_sha256"],
                        "anomaly": efficiency["intervention"],
                        "efficiency_operational": efficiency["value"],
                    }
                )
    output = _screening_dir(root) / "aggregate_input.json"
    _write_json(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "manifest_content_sha256": manifest["content_sha256"],
            "cross_score_rows": rows,
            "cross_score_rows_sha256": _canonical_hash(rows),
            "candidate_efficiencies": candidate_rows,
            "candidate_efficiencies_sha256": _canonical_hash(candidate_rows),
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "cross_score_rows": len(rows),
                "candidate_efficiencies": len(candidate_rows),
                "sha256": _sha256(output),
            },
            sort_keys=True,
        )
    )


def analyze(args: argparse.Namespace) -> None:
    """Build all provisional result tables and the caveated Markdown report."""
    from src.evaluation.physics_background_pairing_paper_comparison import (
        load_authenticated_paper_reference,
    )
    from src.evaluation.physics_background_pairing_search50_analysis import (
        build_provisional_search50_analysis,
        render_provisional_search50_report,
    )

    root = args.root.resolve(strict=True)
    manifest = _load_prepared(root)
    aggregate_path = _screening_dir(root) / "aggregate_input.json"
    aggregate = _read_json(aggregate_path)
    if aggregate.get("manifest_content_sha256") != manifest["content_sha256"]:
        raise RuntimeError("Aggregate input belongs to another evaluation manifest.")
    candidate_rows = aggregate.get("candidate_efficiencies", [])
    if aggregate.get("candidate_efficiencies_sha256") != _canonical_hash(candidate_rows):
        raise RuntimeError("Candidate-efficiency aggregate hash differs.")
    frozen_manifest = _read_json(Path(manifest["frozen_manifest"]))
    paper_reference, paper_reference_sha256 = load_authenticated_paper_reference(
        Path(manifest["evaluation_execution_repository"])
    )
    result = build_provisional_search50_analysis(frozen_manifest, candidate_rows, paper_reference)
    result["method"].update(
        {
            "evaluation_manifest_sha256": _sha256(_screening_dir(root) / "manifest.json"),
            "aggregate_input_sha256": _sha256(aggregate_path),
            "paper_reference_sha256": paper_reference_sha256,
        }
    )

    result_dir = root / "results" / "provisional_search50"
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result_dir / "evaluation_manifest.json", manifest)
    _write_json(result_dir / "analysis.json", result)
    table_names = (
        "candidate_efficiencies",
        "candidate_summary",
        "front_winners",
        "cross_combination_rankings",
        "cross_combination_choices",
        "paper_comparison",
    )
    for name in table_names:
        _write_csv(result_dir / f"{name}.csv", result[name])
    _write_text(
        result_dir / "PROVISIONAL_REPORT.md",
        render_provisional_search50_report(result),
    )
    artifact_paths = sorted(
        path for path in result_dir.iterdir() if path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "namespace": "provisional_search50",
        "artifacts": [
            {"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        ],
    }
    _write_json(result_dir / "artifact_manifest.json", artifact_manifest)
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "front_winners": len(result["front_winners"]),
                "cross_combination_choices": len(result["cross_combination_choices"]),
                "artifact_manifest_sha256": _sha256(result_dir / "artifact_manifest.json"),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse the checkpoint-evaluation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--frozen-repository", type=Path, required=True)
    prep.add_argument("--execution-repository", type=Path, required=True)
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--raw-data-dir", type=Path, required=True)
    prep.add_argument("--pairing-dir", type=Path, required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--task-index", type=int, required=True, choices=range(346))
    run.add_argument("--attempt-id")
    commands.add_parser("status")
    commands.add_parser("aggregate-input")
    commands.add_parser("analyze")
    return parser.parse_args()


def main() -> None:
    """Dispatch one checkpoint-evaluation operation."""
    args = parse_args()
    {
        "prepare": prepare,
        "evaluate": evaluate,
        "status": status,
        "aggregate-input": aggregate_input,
        "analyze": analyze,
    }[args.command](args)


if __name__ == "__main__":
    main()
