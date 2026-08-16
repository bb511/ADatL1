#!/usr/bin/env python3
"""Evaluate AUPRC on the exact frozen search-50 checkpoint universe.

This is intentionally isolated from ``screening_evaluation``: it reads that authenticated
manifest as its scientific source but writes only below ``auprc_evaluation``.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess  # nosec B404 -- fixed executable/argv from authenticated repository
import sys
from pathlib import Path
from typing import Any

if __package__:
    from scripts.physics_background_pairing_search50_evaluation import (
        EXPECTED_ANOMALIES,
        OAS_MODELS,
        _authenticate_execution,
        _canonical_hash,
        _load_prepared,
        _read_json,
        _sha256,
        _write_json,
    )
else:
    from physics_background_pairing_search50_evaluation import (  # type: ignore
        EXPECTED_ANOMALIES,
        OAS_MODELS,
        _authenticate_execution,
        _canonical_hash,
        _load_prepared,
        _read_json,
        _sha256,
        _write_json,
    )

SCHEMA_VERSION = 1
NAMESPACE = "auprc_evaluation"


def _workspace(root: Path) -> Path:
    """Return the namespace isolated from existing efficiency artifacts."""
    return root / NAMESPACE


def build_manifest(root: Path, execution_repository: Path) -> dict[str, Any]:
    """Bind the exact authenticated search50 tasks to the AUPRC execution revision."""
    source = _load_prepared(root)
    repository, commit = _authenticate_execution(execution_repository)
    tasks = source["tasks"]
    if (
        len(tasks) != 346
        or sum(len(task["checkpoints"]) for task in tasks) != 421
        or sum(len(task["checkpoints"]) * len(task["scores"]) for task in tasks) != 601
    ):
        raise RuntimeError("Source manifest differs from the authenticated 346/421/601 universe.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "search50_downstream_auprc_only",
        "campaign_root": str(root.resolve(strict=True)),
        "source_manifest": str(
            (_workspace(root).parent / "screening_evaluation/manifest.json").resolve()
        ),
        "source_manifest_sha256": _sha256(root / "screening_evaluation/manifest.json"),
        "source_manifest_content_sha256": source["content_sha256"],
        "evaluation_execution_repository": str(repository),
        "evaluation_execution_commit": commit,
        "inputs": source["inputs"],
        "expected_anomalies_per_score_checkpoint": EXPECTED_ANOMALIES,
        "tasks": tasks,
    }
    payload["content_sha256"] = _canonical_hash(payload)
    return payload


def prepare(args: argparse.Namespace) -> None:
    """Persist the immutable AUPRC task manifest without changing efficiency state."""
    root = args.root.resolve(strict=True)
    manifest = build_manifest(root, args.execution_repository)
    path = _workspace(root) / "manifest.json"
    if path.exists() and _read_json(path) != manifest:
        raise RuntimeError(f"Refusing to replace a different AUPRC manifest: {path}")
    if not path.exists():
        _write_json(path, manifest)
    print(
        json.dumps({"manifest": str(path), "tasks": 346, "sha256": _sha256(path)}, sort_keys=True)
    )


def _load_manifest(root: Path) -> dict[str, Any]:
    """Load and reauthenticate the AUPRC manifest and its exact source."""
    manifest = _read_json(_workspace(root) / "manifest.json")
    digest = manifest.pop("content_sha256", None)
    if digest != _canonical_hash(manifest):
        raise RuntimeError("AUPRC manifest content hash differs.")
    manifest["content_sha256"] = digest
    source_path = Path(manifest["source_manifest"])
    if _sha256(source_path) != manifest["source_manifest_sha256"]:
        raise RuntimeError("Source search50 manifest changed after AUPRC preparation.")
    source = _load_prepared(root)
    if source["content_sha256"] != manifest["source_manifest_content_sha256"]:
        raise RuntimeError("Source search50 manifest identity differs.")
    if source["tasks"] != manifest["tasks"] or source["inputs"] != manifest["inputs"]:
        raise RuntimeError("AUPRC scientific tasks differ from the frozen search50 source.")
    _authenticate_execution(
        Path(manifest["evaluation_execution_repository"]),
        manifest["evaluation_execution_commit"],
    )
    return manifest


def _stage(root: Path, task: dict[str, Any]) -> tuple[Path, str, str]:
    """Create hash-checked checkpoint symlinks only inside the AUPRC workspace."""
    task_dir = _workspace(root) / "tasks" / f"{int(task['task_index']):03d}"
    experiment = f"physics_bgpair_search50_auprc_{task['model']}"
    run_name = f"trial_{int(task['trial_number']):04d}"
    criterion = (
        task_dir / "checkpoint_store" / experiment / run_name / "candidate/checkpoint/selected"
    )
    criterion.mkdir(parents=True, exist_ok=True)
    expected = set()
    for checkpoint in task["checkpoints"]:
        source = Path(checkpoint["source_path"]).resolve(strict=True)
        if _sha256(source) != checkpoint["source_sha256"]:
            raise RuntimeError(f"Frozen checkpoint changed: {source}")
        name = f"ds={checkpoint['checkpoint_id']}__selected.ckpt"
        expected.add(name)
        target = criterion / name
        if target.is_symlink():
            if target.resolve(strict=True) != source:
                raise RuntimeError(f"Staged AUPRC symlink changed target: {target}")
        elif target.exists():
            raise RuntimeError(f"AUPRC staged checkpoint is not a symlink: {target}")
        else:
            target.symlink_to(source)
    if {path.name for path in criterion.glob("*.ckpt")} != expected:
        raise RuntimeError("AUPRC staged checkpoint coverage differs.")
    return task_dir, experiment, run_name


def _value_rows(task_dir: Path, task: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Authenticate exactly 20 AUPRC values for every checkpoint/score pair."""
    expected = {
        (checkpoint["checkpoint_id"], score)
        for checkpoint in task["checkpoints"]
        for score in task["scores"]
    }
    callback_scores = {"auprc_native": "native", "auprc_residual_oas": "residual_oas"}
    found: dict[tuple[str, str], list[dict]] = {}
    files = []
    for path in sorted(task_dir.glob("checkpoint_store/**/plots/test/*/auprc_*/values.csv")):
        if path.parent.name not in callback_scores:
            raise RuntimeError(f"Unexpected AUPRC callback output: {path}")
        checkpoint_id = path.parent.parent.name.split("ds=", 1)[-1].split("__", 1)[0]
        key = (checkpoint_id, callback_scores[path.parent.name])
        if key in found:
            raise RuntimeError(f"Duplicate AUPRC output for {key}.")
        with path.open(newline="", encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
        if (
            len(raw) != EXPECTED_ANOMALIES
            or len({row["intervention"] for row in raw}) != EXPECTED_ANOMALIES
            or {row["metric"] for row in raw} != {"auprc"}
        ):
            raise RuntimeError(f"AUPRC coverage differs for {key}.")
        rows = sorted(
            (
                {
                    "intervention": row["intervention"],
                    "metric": "auprc",
                    "value": float(row["value"]),
                }
                for row in raw
            ),
            key=lambda row: row["intervention"],
        )
        found[key] = rows
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "checkpoint_id": key[0],
                "score": key[1],
                "values_sha256": _canonical_hash(rows),
            }
        )
    if set(found) != expected:
        raise RuntimeError(
            f"AUPRC output coverage differs: missing={sorted(expected-set(found))}, extra={sorted(set(found)-expected)}"
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


def _marker_valid(manifest: dict, task: dict, marker: dict) -> bool:
    """Reauthenticate a marker and every AUPRC values file it references."""
    if marker.get("manifest_content_sha256") != manifest["content_sha256"] or marker.get(
        "task_sha256"
    ) != _canonical_hash(task):
        return False
    try:
        values, files = _value_rows(Path(marker.get("task_dir", "")), task)
    except (OSError, ValueError, RuntimeError):
        return False
    return marker.get("values") == values and marker.get("value_files") == files


def evaluate(args: argparse.Namespace) -> None:
    """Run AUPRC-only downstream evaluation for one frozen task."""
    root = args.root.resolve(strict=True)
    manifest = _load_manifest(root)
    task = manifest["tasks"][args.task_index]
    if int(task["task_index"]) != args.task_index:
        raise RuntimeError("Task index mismatch.")
    task_dir, experiment, run_name = _stage(root, task)
    marker_path = task_dir / "marker.json"
    with (task_dir / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker_path.exists() and _marker_valid(manifest, task, _read_json(marker_path)):
            print(f"task {args.task_index}: already complete")
            return
        attempt_id = args.attempt_id or f"pid-{os.getpid()}"
        attempt_dir = task_dir / "attempts" / attempt_id
        if attempt_dir.exists() and any(attempt_dir.iterdir()):
            raise RuntimeError(f"Refusing to reuse nonempty AUPRC attempt: {attempt_dir}")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        staged_store = task_dir / "checkpoint_store"
        suite = (
            "physics_background_native_oas"
            if task["model"] in OAS_MODELS
            else "physics_background_native"
        )
        allowlist = (
            "[anomaly_auprc,anomaly_auprc_oas]"
            if task["model"] in OAS_MODELS
            else "[anomaly_auprc]"
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
            "evaluation.evaluator.ckpts.last=false",
            "evaluation.evaluator.ckpts.summary=false",
            "evaluation.evaluator.ckpts.single=false",
            "++evaluation.evaluator.ckpts={candidate:{checkpoint:[selected]}}",
            f"evaluation_callback_allowlist={allowlist}",
            f"paths.checkpoints_dir={staged_store}",
            f"experiment_name={experiment}",
            f"run_name={run_name}",
            f"hydra.run.dir={attempt_dir / 'hydra'}",
        ]
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
        values, files = _value_rows(task_dir, task)
        marker = {
            "schema_version": SCHEMA_VERSION,
            "task_index": args.task_index,
            "manifest_content_sha256": manifest["content_sha256"],
            "task_sha256": _canonical_hash(task),
            "task_dir": str(task_dir.resolve()),
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
    """Print exact authenticated completion, pending, and invalid task counts."""
    root = args.root.resolve(strict=True)
    manifest = _load_manifest(root)
    complete, invalid = [], []
    for task in manifest["tasks"]:
        path = _workspace(root) / "tasks" / f"{int(task['task_index']):03d}" / "marker.json"
        if path.exists():
            (complete if _marker_valid(manifest, task, _read_json(path)) else invalid).append(
                task["task_index"]
            )
    pending = sorted(set(range(346)) - set(complete))
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


def aggregate(args: argparse.Namespace) -> None:
    """Persist authenticated per-membership and per-checkpoint AUPRC rows."""
    root = args.root.resolve(strict=True)
    manifest = _load_manifest(root)
    cross_rows, candidate_rows = [], []
    for task in manifest["tasks"]:
        path = _workspace(root) / "tasks" / f"{int(task['task_index']):03d}" / "marker.json"
        marker = _read_json(path)
        if not _marker_valid(manifest, task, marker):
            raise RuntimeError(f"Missing or invalid AUPRC marker for task {task['task_index']}.")
        values = {(row["checkpoint_id"], row["score"]): row for row in marker["values"]}
        checkpoints = {row["checkpoint_id"]: row for row in task["checkpoints"]}
        for membership in task["memberships"]:
            for score in task["scores"]:
                for result in values[(membership["checkpoint_id"], score)]["rows"]:
                    cross_rows.append(
                        {
                            "membership_index": membership["membership_index"],
                            "model": task["model"],
                            "trial_number": task["trial_number"],
                            "logical_objective": membership["logical_objective"],
                            "selection_score": membership["score"],
                            "evaluation_score": score,
                            "metric": membership["metric"],
                            "strategy": membership["strategy"],
                            "checkpoint_id": membership["checkpoint_id"],
                            "anomaly": result["intervention"],
                            "auprc": result["value"],
                        }
                    )
            selected = values[(membership["checkpoint_id"], membership["score"])]
            checkpoint = checkpoints[membership["checkpoint_id"]]
            for result in selected["rows"]:
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
                        "anomaly": result["intervention"],
                        "auprc": result["value"],
                    }
                )
    output = _workspace(root) / "aggregate.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_content_sha256": manifest["content_sha256"],
        "cross_score_rows": cross_rows,
        "cross_score_rows_sha256": _canonical_hash(cross_rows),
        "candidate_auprc": candidate_rows,
        "candidate_auprc_sha256": _canonical_hash(candidate_rows),
    }
    _write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "cross_score_rows": len(cross_rows),
                "candidate_auprc": len(candidate_rows),
                "sha256": _sha256(output),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse isolated AUPRC evaluation operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--execution-repository", type=Path, required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--task-index", type=int, choices=range(346), required=True)
    run.add_argument("--attempt-id")
    commands.add_parser("status")
    commands.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    """Dispatch one isolated AUPRC evaluation operation."""
    args = parse_args()
    {"prepare": prepare, "evaluate": evaluate, "status": status, "aggregate": aggregate}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
