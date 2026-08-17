#!/usr/bin/env python3
"""Authenticate and evaluate random-bijection CAP on sealed physics checkpoints."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import statistics
import subprocess  # nosec B404 -- authenticated repository and fixed argv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.physics_background_pairing_search50_evaluation import (
    OAS_MODELS,
    _authenticate_execution,
    _canonical_hash,
    _load_prepared,
    _read_json,
    _sha256,
    _write_json,
)
from src.evaluation.physics_background_pairing_confirmation_matrix import (
    load_authenticated_confirmation_campaign,
)
from src.utils.pairing.random_tables import validate_frozen_random_pairing_tables

SCHEMA_VERSION = 1
MODES = ("discovery", "confirmation")
SPLITS = ("validate", "test")
EXPECTED_REPLICATES = 5


def _workspace(root: Path, mode: str) -> Path:
    """Return the mode-specific namespace isolated from endpoint artifacts."""
    if mode not in MODES:
        raise ValueError(f"Unknown random-CAP mode: {mode}")
    return root / "random_cap_evaluation" / mode


def _table_identity(root: Path) -> dict[str, Any]:
    """Authenticate and summarize the frozen ten-table control universe."""
    manifest = validate_frozen_random_pairing_tables(root)
    path = root / "random_pairing/manifest.json"
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": _sha256(path),
        "content_sha256": manifest["content_sha256"],
        "seed_contract_sha256": manifest["seed_contract"]["sha256"],
        "table_file_sha256": {str(row["path"]): row["file_sha256"] for row in manifest["tables"]},
        "tables": [
            {
                "split": row["split"],
                "replicate": int(row["replicate"]),
                "seed": int(row["seed"]),
                "file_sha256": row["file_sha256"],
                "permutation_sha256": row["permutation_sha256"],
            }
            for row in manifest["tables"]
        ],
    }


def build_discovery_manifest(root: Path, execution_repository: Path) -> dict[str, Any]:
    """Bind the exact frozen 346/421/601 discovery checkpoint universe."""
    source = _load_prepared(root)
    repository, commit = _authenticate_execution(execution_repository)
    tasks = source["tasks"]
    if len(tasks) != 346 or sum(len(row["checkpoints"]) for row in tasks) != 421:
        raise RuntimeError("Discovery random-CAP source differs from the frozen 346/421 universe.")
    if sum(len(row["checkpoints"]) * len(row["scores"]) for row in tasks) != 601:
        raise RuntimeError("Discovery random-CAP score/checkpoint coverage is not 601.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "discovery",
        "purpose": "offline_random_cap_diagnostic_only",
        "campaign_root": str(root.resolve(strict=True)),
        "source_manifest": str((root / "screening_evaluation/manifest.json").resolve(strict=True)),
        "source_manifest_sha256": _sha256(root / "screening_evaluation/manifest.json"),
        "source_manifest_content_sha256": source["content_sha256"],
        "execution_repository": str(repository),
        "execution_commit": commit,
        "inputs": source["inputs"],
        "random_pairing": _table_identity(root),
        "tasks": tasks,
    }
    payload["content_sha256"] = _canonical_hash(payload)
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV while requiring a unique nonempty header."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid CSV header: {path}")
        return list(reader)


def build_confirmation_manifest(
    root: Path,
    execution_repository: Path,
    checkpoint_selections: Path,
    endpoint_receipt: Path,
) -> dict[str, Any]:
    """Bind every sealed confirmation cell/seed checkpoint selection."""
    frozen = load_authenticated_confirmation_campaign(root)
    matrix = frozen["matrix"]
    runner = frozen["runner_manifest"]
    seed_count = len(matrix["training_seeds"])
    expected_tasks = 28 * seed_count
    expected_memberships = 42 * seed_count
    repository, commit = _authenticate_execution(execution_repository)
    if commit != frozen["design"]["code_commit"]:
        raise RuntimeError("Random-CAP execution commit differs from sealed confirmation.")
    checkpoint_selections = checkpoint_selections.resolve(strict=True)
    endpoint_receipt = endpoint_receipt.resolve(strict=True)
    receipt = _read_json(endpoint_receipt)
    if Path(receipt.get("campaign_root", "")).resolve(strict=True) != root.resolve(
        strict=True
    ) or receipt.get("artifacts", {}).get("checkpoint_selections.csv") != _sha256(
        checkpoint_selections
    ):
        raise RuntimeError("Confirmation checkpoint-selection receipt differs.")
    rows = _read_csv(checkpoint_selections)
    if len(rows) != expected_memberships:
        raise RuntimeError(
            f"Expected {expected_memberships} sealed confirmation selections, "
            f"found {len(rows)}."
        )
    row_index: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = (int(row["cell_index"]), int(row["seed"]))
        if key in row_index:
            raise RuntimeError(f"Duplicate sealed confirmation selection: {key}")
        path = Path(row["checkpoint_path"]).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or _sha256(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Sealed confirmation checkpoint changed: {path}")
        row_index[key] = row

    discovery_source = _load_prepared(Path(matrix["discovery_root"]))
    configurations = {int(row["configuration_index"]): row for row in matrix["configurations"]}
    cells = {int(row["cell_index"]): row for row in matrix["logical_cells"]}
    memberships_by_run: dict[int, list[dict]] = defaultdict(list)
    for membership in runner["front_memberships"]:
        memberships_by_run[int(membership["retrain_index"])].append(membership)
    tasks = []
    seen_memberships = set()
    for run in sorted(runner["retrain_runs"], key=lambda row: int(row["retrain_index"])):
        run_index = int(run["retrain_index"])
        checkpoints: dict[tuple[str, str], dict[str, Any]] = {}
        task_memberships = []
        for membership in sorted(
            memberships_by_run[run_index], key=lambda row: int(row["membership_index"])
        ):
            key = (int(membership["cell_index"]), int(membership["seed"]))
            selection = row_index.get(key)
            if selection is None:
                raise RuntimeError(f"Missing sealed checkpoint selection: {key}")
            cell = cells[key[0]]
            expected = {
                "model": cell["model"],
                "trial_number": str(cell["trial_number"]),
                "seed": str(key[1]),
                "logical_objective": cell["logical_objective"],
            }
            if any(str(selection[name]) != str(value) for name, value in expected.items()):
                raise RuntimeError(f"Sealed selection metadata differs for {key}.")
            physical = (selection["checkpoint_path"], selection["checkpoint_sha256"])
            if physical not in checkpoints:
                checkpoints[physical] = {
                    "checkpoint_id": f"selected_{len(checkpoints):03d}",
                    "source_path": physical[0],
                    "source_sha256": physical[1],
                    "logical_objectives": [],
                }
            checkpoint = checkpoints[physical]
            checkpoint["logical_objectives"].append(cell["logical_objective"])
            task_memberships.append(
                {
                    **membership,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_name": selection["checkpoint_family_id"],
                }
            )
            seen_memberships.add(key)
        configuration = configurations[int(run["configuration_index"])]
        model = str(run["model"])
        tasks.append(
            {
                "task_index": len(tasks),
                "run_index": run_index,
                "model": model,
                "trial_number": int(run["trial_number"]),
                "training_seed": int(run["seed"]),
                "params": configuration["params"],
                "scores": ["native", "residual_oas"] if model in OAS_MODELS else ["native"],
                "checkpoints": list(checkpoints.values()),
                "memberships": task_memberships,
            }
        )
    if (
        len(tasks) != expected_tasks
        or len(seen_memberships) != expected_memberships
        or set(row_index) != seen_memberships
    ):
        raise RuntimeError(
            "Confirmation random-CAP task coverage differs: "
            f"tasks={len(tasks)}/{expected_tasks}, "
            f"memberships={len(seen_memberships)}/{expected_memberships}."
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "confirmation",
        "purpose": "offline_random_cap_diagnostic_only",
        "campaign_root": str(root.resolve(strict=True)),
        "confirmation_matrix_content_sha256": matrix["content_sha256"],
        "runner_manifest_content_sha256": runner["content_sha256"],
        "checkpoint_selections": str(checkpoint_selections),
        "checkpoint_selections_sha256": _sha256(checkpoint_selections),
        "endpoint_receipt": str(endpoint_receipt),
        "endpoint_receipt_sha256": _sha256(endpoint_receipt),
        "execution_repository": str(repository),
        "execution_commit": commit,
        "source_discovery_evaluation_manifest_content_sha256": discovery_source["content_sha256"],
        "inputs": discovery_source["inputs"],
        "random_pairing": _table_identity(root),
        "tasks": tasks,
    }
    payload["content_sha256"] = _canonical_hash(payload)
    return payload


def prepare(args: argparse.Namespace) -> None:
    """Persist one immutable mode-specific task manifest."""
    root = args.root.resolve(strict=True)
    if args.command == "prepare-discovery":
        manifest = build_discovery_manifest(root, args.execution_repository)
        mode = "discovery"
    else:
        manifest = build_confirmation_manifest(
            root,
            args.execution_repository,
            args.checkpoint_selections,
            args.endpoint_receipt,
        )
        mode = "confirmation"
    path = _workspace(root, mode) / "manifest.json"
    if path.exists() and _read_json(path) != manifest:
        raise RuntimeError(f"Refusing to replace a different random-CAP manifest: {path}")
    if not path.exists():
        _write_json(path, manifest)
    print(
        json.dumps(
            {"manifest": str(path), "tasks": len(manifest["tasks"]), "sha256": _sha256(path)},
            sort_keys=True,
        )
    )


def _load_manifest(root: Path, mode: str) -> dict[str, Any]:
    """Reauthenticate a prepared manifest, tables, execution code, and checkpoints."""
    path = _workspace(root, mode) / "manifest.json"
    manifest = _read_json(path)
    digest = manifest.pop("content_sha256", None)
    if digest != _canonical_hash(manifest):
        raise RuntimeError("Prepared random-CAP manifest content hash differs.")
    manifest["content_sha256"] = digest
    _authenticate_execution(Path(manifest["execution_repository"]), manifest["execution_commit"])
    if mode == "discovery":
        source = _load_prepared(root)
        if (
            _sha256(Path(manifest["source_manifest"])) != manifest["source_manifest_sha256"]
            or source["content_sha256"] != manifest["source_manifest_content_sha256"]
            or source["tasks"] != manifest["tasks"]
        ):
            raise RuntimeError("Prepared discovery source manifest identity differs.")
    else:
        frozen = load_authenticated_confirmation_campaign(root)
        if (
            frozen["matrix"]["content_sha256"] != manifest["confirmation_matrix_content_sha256"]
            or frozen["runner_manifest"]["content_sha256"]
            != manifest["runner_manifest_content_sha256"]
            or _sha256(Path(manifest["checkpoint_selections"]))
            != manifest["checkpoint_selections_sha256"]
            or _sha256(Path(manifest["endpoint_receipt"])) != manifest["endpoint_receipt_sha256"]
        ):
            raise RuntimeError("Prepared sealed confirmation identity differs.")
    tables = _table_identity(root)
    if tables != manifest["random_pairing"]:
        raise RuntimeError("Prepared random-CAP table identity differs.")
    for task in manifest["tasks"]:
        for checkpoint in task["checkpoints"]:
            path = Path(checkpoint["source_path"])
            if _sha256(path) != checkpoint["source_sha256"]:
                raise RuntimeError(f"Prepared random-CAP checkpoint changed: {path}")
    return manifest


def _stage(root: Path, mode: str, task: dict) -> tuple[Path, str, str]:
    """Stage only authenticated symlinks below the random-CAP namespace."""
    task_dir = _workspace(root, mode) / "tasks" / f"{int(task['task_index']):03d}"
    experiment = f"physics_bgpair_random_cap_{mode}_{task['model']}"
    suffix = (
        f"trial_{task['trial_number']:04d}__seed_{task['training_seed']}"
        if mode == "confirmation"
        else f"trial_{task['trial_number']:04d}"
    )
    run_name = suffix
    criterion = (
        task_dir / "checkpoint_store" / experiment / run_name / "candidate/checkpoint/selected"
    )
    criterion.mkdir(parents=True, exist_ok=True)
    expected = set()
    for checkpoint in task["checkpoints"]:
        source = Path(checkpoint["source_path"]).resolve(strict=True)
        if _sha256(source) != checkpoint["source_sha256"]:
            raise RuntimeError(f"Random-CAP checkpoint changed: {source}")
        name = f"ds={checkpoint['checkpoint_id']}__selected.ckpt"
        expected.add(name)
        target = criterion / name
        if target.is_symlink():
            if target.resolve(strict=True) != source:
                raise RuntimeError(f"Random-CAP staged symlink target differs: {target}")
        elif target.exists():
            raise RuntimeError(f"Random-CAP staged checkpoint is not a symlink: {target}")
        else:
            target.symlink_to(source)
    if {path.name for path in criterion.glob("*.ckpt")} != expected:
        raise RuntimeError("Random-CAP staged checkpoint coverage differs.")
    return task_dir, experiment, run_name


def _value_rows(task_dir: Path, task: dict, random_pairing: dict) -> tuple[list[dict], list[dict]]:
    """Authenticate exact checkpoint/score/split/replicate random-CAP coverage."""
    expected = {
        (checkpoint["checkpoint_id"], score, split, replicate)
        for checkpoint in task["checkpoints"]
        for score in task["scores"]
        for split in SPLITS
        for replicate in range(EXPECTED_REPLICATES)
    }
    found = {}
    files = []
    expected_tables = {
        (row["split"], int(row["replicate"])): row for row in random_pairing.get("tables", [])
    }
    if len(expected_tables) != len(SPLITS) * EXPECTED_REPLICATES:
        raise RuntimeError("Random-CAP manifest does not bind ten split/replicate tables.")
    callback_score = {"random_cap_native": "native", "random_cap_residual_oas": "residual_oas"}
    for path in sorted(task_dir.glob("checkpoint_store/**/plots/*/*/random_cap_*/values.csv")):
        callback = path.parent.name
        if callback not in callback_score:
            raise RuntimeError(f"Unexpected random-CAP callback output: {path}")
        expected_split = "validate" if path.parents[2].name == "val" else path.parents[2].name
        if expected_split not in SPLITS:
            raise RuntimeError(f"Unexpected random-CAP split directory: {path}")
        checkpoint_stem = path.parent.parent.name
        checkpoint_id = path.parent.parent.name.split("ds=", 1)[-1].split("__", 1)[0]
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != EXPECTED_REPLICATES:
            raise RuntimeError(f"Random-CAP values do not contain five seeds: {path}")
        normalized = []
        for row in rows:
            key = (checkpoint_id, callback_score[callback], row["split"], int(row["replicate"]))
            value = float(row["value"])
            expected_table = expected_tables.get((key[2], key[3]))
            if (
                key in found
                or key not in expected
                or row.get("checkpoint") != checkpoint_stem
                or row.get("score") != callback_score[callback]
                or row.get("split") != expected_split
                or row["metric"] != "cap"
                or not math.isfinite(value)
                or expected_table is None
                or int(row["seed"]) != int(expected_table["seed"])
                or row["pair_table_sha256"] != expected_table["file_sha256"]
                or row["permutation_sha256"] != expected_table["permutation_sha256"]
            ):
                raise RuntimeError(f"Invalid or duplicate random-CAP value: {key}")
            normalized_row = {
                "checkpoint_id": checkpoint_id,
                "score": key[1],
                "split": key[2],
                "replicate": key[3],
                "seed": int(row["seed"]),
                "value": value,
                "pair_table_sha256": row["pair_table_sha256"],
                "permutation_sha256": row["permutation_sha256"],
            }
            found[key] = normalized_row
            normalized.append(normalized_row)
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows_sha256": _canonical_hash(normalized),
            }
        )
    if set(found) != expected:
        raise RuntimeError(
            f"Random-CAP output coverage differs: missing={len(expected-set(found))}, extra={len(set(found)-expected)}"
        )
    return [found[key] for key in sorted(found)], files


def _marker_valid(manifest: dict, task: dict, marker: dict) -> bool:
    """Reauthenticate a marker and every seed-level CSV it references."""
    if marker.get("manifest_content_sha256") != manifest["content_sha256"] or marker.get(
        "task_sha256"
    ) != _canonical_hash(task):
        return False
    try:
        rows, files = _value_rows(
            Path(marker.get("task_dir", "")), task, manifest["random_pairing"]
        )
    except (OSError, ValueError, RuntimeError):
        return False
    return marker.get("values") == rows and marker.get("value_files") == files


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate all five fixed random controls for one checkpoint task."""
    root = args.root.resolve(strict=True)
    manifest = _load_manifest(root, args.mode)
    task = manifest["tasks"][args.task_index]
    if int(task["task_index"]) != args.task_index:
        raise RuntimeError("Random-CAP task index mismatch.")
    task_dir, experiment, run_name = _stage(root, args.mode, task)
    marker_path = task_dir / "marker.json"
    with (task_dir / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker_path.exists() and _marker_valid(manifest, task, _read_json(marker_path)):
            print(f"task {args.task_index}: already complete")
            return
        attempt = task_dir / "attempts" / (args.attempt_id or f"pid-{os.getpid()}")
        if attempt.exists() and any(attempt.iterdir()):
            raise RuntimeError(f"Refusing to reuse nonempty random-CAP attempt: {attempt}")
        attempt.mkdir(parents=True, exist_ok=True)
        staged = task_dir / "checkpoint_store"
        suite = (
            "physics_background_random_cap_native_oas"
            if task["model"] in OAS_MODELS
            else "physics_background_random_cap_native"
        )
        allowlist = (
            "[random_cap_native,random_cap_residual_oas]"
            if task["model"] in OAS_MODELS
            else "[random_cap_native]"
        )
        command = [
            sys.executable,
            "src/train.py",
            f"experiment=physics/{task['model']}_background_all",
            f"+diagnostic_suite={suite}",
            "logger=none",
            "trainer=gpu",
            "trainer.devices=[0]",
            "train=false",
            "test=true",
            "skip_validation_evaluation=false",
            "data.validation_aux_datasets=[]",
            "+data.test_aux_datasets=[]",
            "data.max_normal_eval_batches=null",
            "evaluation.evaluator.ckpts.last=false",
            "evaluation.evaluator.ckpts.summary=false",
            "evaluation.evaluator.ckpts.single=false",
            "++evaluation.evaluator.ckpts={candidate:{checkpoint:[selected]}}",
            f"evaluation_callback_allowlist={allowlist}",
            f"paths.checkpoints_dir={staged}",
            f"experiment_name={experiment}",
            f"run_name={run_name}",
            f"hydra.run.dir={attempt / 'hydra'}",
        ]
        command.extend(f"{key}={value}" for key, value in sorted(task["params"].items()))
        env = os.environ.copy()
        env.update(
            {
                "DATA_DIR": manifest["inputs"]["data_dir"],
                "RAW_DATA_DIR": manifest["inputs"]["raw_data_dir"],
                "PHYSICS_PAIRING_DIR": manifest["inputs"]["pairing_dir"],
                "PHYSICS_RANDOM_PAIRING_MANIFEST": manifest["random_pairing"]["path"],
                "CHECKPOINT_DIR": str(staged),
                "OUTPUT_DIR": str(attempt / "outputs"),
                "LOG_DIR": str(attempt / "logs"),
                "HYDRA_FULL_ERROR": "1",
                "MLFLOW_DISABLE_TELEMETRY": "true",
            }
        )
        subprocess.run(
            command, cwd=manifest["execution_repository"], env=env, check=True
        )  # nosec B603
        values, files = _value_rows(task_dir, task, manifest["random_pairing"])
        _write_json(
            marker_path,
            {
                "schema_version": SCHEMA_VERSION,
                "task_index": args.task_index,
                "manifest_content_sha256": manifest["content_sha256"],
                "task_sha256": _canonical_hash(task),
                "task_dir": str(task_dir.resolve()),
                "values": values,
                "value_files": files,
                "checkpoint_hashes": {
                    row["checkpoint_id"]: row["source_sha256"] for row in task["checkpoints"]
                },
                "random_pairing_content_sha256": manifest["random_pairing"]["content_sha256"],
            },
        )
        print(f"task {args.task_index}: complete")


def status(args: argparse.Namespace) -> None:
    """Report authenticated complete, pending, and invalid task counts."""
    root = args.root.resolve(strict=True)
    manifest = _load_manifest(root, args.mode)
    complete, invalid = [], []
    for task in manifest["tasks"]:
        path = _workspace(root, args.mode) / "tasks" / f"{int(task['task_index']):03d}/marker.json"
        if path.exists():
            (complete if _marker_valid(manifest, task, _read_json(path)) else invalid).append(
                task["task_index"]
            )
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


def aggregate(args: argparse.Namespace) -> None:
    """Emit seed-level CAP rows plus descriptive five-pairing mean and variance."""
    root = args.root.resolve(strict=True)
    manifest = _load_manifest(root, args.mode)
    rows = []
    for task in manifest["tasks"]:
        marker_path = (
            _workspace(root, args.mode) / "tasks" / f"{int(task['task_index']):03d}/marker.json"
        )
        marker = _read_json(marker_path)
        if not _marker_valid(manifest, task, marker):
            raise RuntimeError(
                f"Missing or invalid random-CAP marker for task {task['task_index']}."
            )
        values = {
            (row["checkpoint_id"], row["score"], row["split"]): [] for row in marker["values"]
        }
        for row in marker["values"]:
            values[(row["checkpoint_id"], row["score"], row["split"])].append(row)
        for membership in task["memberships"]:
            for score in task["scores"]:
                for split in SPLITS:
                    for value in sorted(
                        values[(membership["checkpoint_id"], score, split)],
                        key=lambda row: row["replicate"],
                    ):
                        rows.append(
                            {
                                "mode": args.mode,
                                "task_index": task["task_index"],
                                "model": task["model"],
                                "trial_number": task["trial_number"],
                                "training_seed": task.get("training_seed", ""),
                                "membership_index": membership["membership_index"],
                                "logical_objective": membership["logical_objective"],
                                "selection_score": membership.get("score", "native"),
                                "selection_metric": membership.get("metric", "cap"),
                                "selection_strategy": membership.get("strategy", ""),
                                "evaluation_score": score,
                                "split": split,
                                "replicate": value["replicate"],
                                "pairing_seed": value["seed"],
                                "cap": value["value"],
                                "checkpoint_id": membership["checkpoint_id"],
                                "pair_table_sha256": value["pair_table_sha256"],
                                "checkpoint_sha256": next(
                                    row["source_sha256"]
                                    for row in task["checkpoints"]
                                    if row["checkpoint_id"] == membership["checkpoint_id"]
                                ),
                            }
                        )
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(
            row[name]
            for name in ("mode", "task_index", "membership_index", "evaluation_score", "split")
        )
        grouped[key].append(row["cap"])
    summary = []
    for key, values in sorted(grouped.items()):
        if len(values) != EXPECTED_REPLICATES:
            raise RuntimeError(f"Random-CAP aggregate group does not contain five seeds: {key}")
        source_rows = [
            row
            for row in rows
            if tuple(
                row[name]
                for name in (
                    "mode",
                    "task_index",
                    "membership_index",
                    "evaluation_score",
                    "split",
                )
            )
            == key
        ]
        first = source_rows[0]
        summary.append(
            {
                "mode": key[0],
                "task_index": key[1],
                "membership_index": key[2],
                "model": first["model"],
                "trial_number": first["trial_number"],
                "training_seed": first["training_seed"],
                "logical_objective": first["logical_objective"],
                "selection_score": first["selection_score"],
                "selection_metric": first["selection_metric"],
                "selection_strategy": first["selection_strategy"],
                "evaluation_score": key[3],
                "split": key[4],
                "checkpoint_id": first["checkpoint_id"],
                "checkpoint_sha256": first["checkpoint_sha256"],
                "n_pairings": len(values),
                "cap_mean": statistics.fmean(values),
                "cap_sample_variance": statistics.variance(values),
            }
        )
    output = _workspace(root, args.mode) / "aggregate.json"
    _write_json(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "manifest_content_sha256": manifest["content_sha256"],
            "seed_split_values": rows,
            "seed_split_values_sha256": _canonical_hash(rows),
            "summary": summary,
            "summary_sha256": _canonical_hash(summary),
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "summary": len(summary),
                "sha256": _sha256(output),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse preparation and authenticated evaluation operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("prepare-discovery")
    discovery.add_argument("--execution-repository", type=Path, required=True)
    confirmation = commands.add_parser("prepare-confirmation")
    confirmation.add_argument("--execution-repository", type=Path, required=True)
    confirmation.add_argument("--checkpoint-selections", type=Path, required=True)
    confirmation.add_argument("--endpoint-receipt", type=Path, required=True)
    for name in ("evaluate", "status", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--mode", choices=MODES, required=True)
        if name == "evaluate":
            command.add_argument("--task-index", type=int, required=True)
            command.add_argument("--attempt-id")
    return parser.parse_args()


def main() -> None:
    """Dispatch one isolated random-CAP operation."""
    args = parse_args()
    if args.command.startswith("prepare-"):
        prepare(args)
    else:
        {"evaluate": evaluate, "status": status, "aggregate": aggregate}[args.command](args)


if __name__ == "__main__":
    main()
