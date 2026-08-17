"""Fail-closed analysis of offline random-bijection CAP diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.physics_background_pairing_analysis import _spearman

SCHEMA_VERSION = 1
SPLITS = ("validate", "test")
REPLICATES = tuple(range(5))
DISCOVERY_NAMESPACE = "discovery_5293a29"
DISCOVERY_MANIFEST_SHA256 = "cd755201c36b036a2b5b9b83fffd28dbb0153614de42b3d7e695b6b137fd463d"
DISCOVERY_MANIFEST_CONTENT_SHA256 = (
    "8327effabdaaefc68ed0dd2ee0b7504c2e505390635c5fbdc7ca25478c57a70d"
)
DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256 = (
    "d767bff5f5fb21eba668455f85fc02504159783426a91e3d00b68cf382a5d67a"
)
DISCOVERY_TASKS = 346
DISCOVERY_MEMBERSHIPS = 591
DISCOVERY_MEMBERSHIP_SCORES = 859


def _content_hash(value: Any) -> str:
    """Hash one JSON-compatible value using the campaign encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow payload copy without its self-authentication field."""
    return {key: value for key, value in payload.items() if key != "content_sha256"}


def _finite(value: Any, label: str) -> float:
    """Parse one finite floating-point value or fail closed."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return result


def _authenticate_manifest(manifest: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Authenticate task, membership, checkpoint, and random-table identities."""
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("purpose") != "offline_random_cap_diagnostic_only"
        or manifest.get("mode") not in {"discovery", "confirmation"}
        or manifest.get("content_sha256") != _content_hash(_without_hash(manifest))
    ):
        raise ValueError("Random-CAP manifest identity or content hash differs.")
    tables = manifest.get("random_pairing", {}).get("tables", [])
    table_keys = {(row.get("split"), int(row.get("replicate", -1))) for row in tables}
    if len(tables) != 10 or table_keys != {(split, rep) for split in SPLITS for rep in REPLICATES}:
        raise ValueError("Random-CAP manifest must bind exactly five tables per split.")
    if len({int(row["seed"]) for row in tables}) != 10 or any(
        not isinstance(row.get("file_sha256"), str) or len(row["file_sha256"]) != 64
        for row in tables
    ):
        raise ValueError("Random-CAP table seeds or file hashes are not unique and complete.")
    tasks = manifest.get("tasks", [])
    if {int(task["task_index"]) for task in tasks} != set(range(len(tasks))):
        raise ValueError("Random-CAP task indices are not contiguous and unique.")
    memberships: dict[tuple[int, int], dict[str, Any]] = {}
    for task in tasks:
        task_index = int(task["task_index"])
        checkpoints = {row["checkpoint_id"]: row for row in task.get("checkpoints", [])}
        if len(checkpoints) != len(task.get("checkpoints", [])):
            raise ValueError(f"Duplicate checkpoint ID in random-CAP task {task_index}.")
        scores = task.get("scores", [])
        if not scores or len(scores) != len(set(scores)):
            raise ValueError(f"Invalid score streams in random-CAP task {task_index}.")
        for membership in task.get("memberships", []):
            key = (task_index, int(membership["membership_index"]))
            if key in memberships or membership.get("checkpoint_id") not in checkpoints:
                raise ValueError(f"Invalid or duplicate random-CAP membership {key}.")
            memberships[key] = {
                "task": task,
                "membership": membership,
                "checkpoint": checkpoints[membership["checkpoint_id"]],
            }
    if not memberships:
        raise ValueError("Random-CAP manifest contains no logical memberships.")
    membership_scores = sum(
        len(task.get("memberships", [])) * len(task.get("scores", [])) for task in tasks
    )
    if (
        manifest["mode"] == "discovery"
        and manifest["content_sha256"] == (DISCOVERY_MANIFEST_CONTENT_SHA256)
        and (len(tasks), len(memberships), membership_scores)
        != (DISCOVERY_TASKS, DISCOVERY_MEMBERSHIPS, DISCOVERY_MEMBERSHIP_SCORES)
    ):
        raise ValueError("Discovery random-CAP frozen 346/591/859 cardinality differs.")
    return memberships


def _normalize(
    manifest: dict[str, Any], aggregate: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    """Validate exact seed rows and independently recompute every summary."""
    memberships = _authenticate_manifest(manifest)
    if (
        aggregate.get("schema_version") != SCHEMA_VERSION
        or aggregate.get("manifest_content_sha256") != manifest["content_sha256"]
        or aggregate.get("seed_split_values_sha256")
        != _content_hash(aggregate.get("seed_split_values", []))
        or aggregate.get("summary_sha256") != _content_hash(aggregate.get("summary", []))
    ):
        raise ValueError("Random-CAP aggregate identity or child hash differs.")
    table_by_key = {
        (row["split"], int(row["replicate"])): row for row in manifest["random_pairing"]["tables"]
    }
    expected = set()
    for (task_index, membership_index), identity in memberships.items():
        for score in identity["task"]["scores"]:
            for split in SPLITS:
                for replicate in REPLICATES:
                    expected.add((task_index, membership_index, score, split, replicate))
    normalized, seen = [], set()
    for supplied in aggregate.get("seed_split_values", []):
        row = dict(supplied)
        key = (
            int(row["task_index"]),
            int(row["membership_index"]),
            row["evaluation_score"],
            row["split"],
            int(row["replicate"]),
        )
        identity = memberships.get(key[:2])
        table = table_by_key.get(key[3:])
        if key in seen or key not in expected or identity is None or table is None:
            raise ValueError(f"Unexpected or duplicate random-CAP seed row: {key}.")
        task, membership, checkpoint = (
            identity["task"],
            identity["membership"],
            identity["checkpoint"],
        )
        expected_fields = {
            "mode": manifest["mode"],
            "model": task["model"],
            "trial_number": int(task["trial_number"]),
            "logical_objective": membership["logical_objective"],
            "selection_score": membership.get("score", "native"),
            "selection_metric": membership.get("metric", "cap"),
            "selection_strategy": membership.get("strategy", ""),
            "checkpoint_id": membership["checkpoint_id"],
            "checkpoint_sha256": checkpoint["source_sha256"],
            "pairing_seed": int(table["seed"]),
            "pair_table_sha256": table["file_sha256"],
        }
        for field, value in expected_fields.items():
            actual = row.get(field)
            if field in {"trial_number", "pairing_seed"}:
                actual = int(actual)
            if actual != value:
                raise ValueError(f"Random-CAP seed row {field} differs for {key}.")
        expected_training_seed = task.get("training_seed", "")
        if str(row.get("training_seed", "")) != str(expected_training_seed):
            raise ValueError(f"Random-CAP training seed differs for {key}.")
        row.update(
            {
                "task_index": key[0],
                "membership_index": key[1],
                "trial_number": int(row["trial_number"]),
                "replicate": key[4],
                "pairing_seed": int(row["pairing_seed"]),
                "cap": _finite(row["cap"], f"CAP {key}"),
            }
        )
        seen.add(key)
        normalized.append(row)
    if seen != expected:
        raise ValueError(
            f"Random-CAP seed coverage differs: expected {len(expected)}, found {len(seen)}."
        )

    groups: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        groups[
            (
                row["task_index"],
                row["membership_index"],
                row["evaluation_score"],
                row["split"],
            )
        ].append(row)
    supplied_summaries = {
        (
            int(row["task_index"]),
            int(row["membership_index"]),
            row["evaluation_score"],
            row["split"],
        ): row
        for row in aggregate.get("summary", [])
    }
    if len(supplied_summaries) != len(aggregate.get("summary", [])) or set(
        supplied_summaries
    ) != set(groups):
        raise ValueError("Random-CAP summary keys differ from seed-level coverage.")
    summaries = []
    for key, group in sorted(groups.items()):
        values = [row["cap"] for row in sorted(group, key=lambda item: item["replicate"])]
        supplied = supplied_summaries[key]
        mean = statistics.fmean(values)
        variance = statistics.variance(values)
        if (
            int(supplied.get("n_pairings", -1)) != 5
            or not math.isclose(_finite(supplied["cap_mean"], "CAP mean"), mean, abs_tol=1e-12)
            or not math.isclose(
                _finite(supplied["cap_sample_variance"], "CAP variance"),
                variance,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"Random-CAP summary was not reproduced for {key}.")
        first = group[0]
        summaries.append(
            {
                **{
                    field: first[field]
                    for field in (
                        "mode",
                        "task_index",
                        "membership_index",
                        "model",
                        "trial_number",
                        "training_seed",
                        "logical_objective",
                        "selection_score",
                        "selection_metric",
                        "selection_strategy",
                        "evaluation_score",
                        "split",
                        "checkpoint_id",
                        "checkpoint_sha256",
                    )
                },
                "n_pairings": 5,
                "cap_mean": mean,
                "cap_sample_variance": variance,
                "cap_sample_std": math.sqrt(variance),
                "cap_min": min(values),
                "cap_max": max(values),
                "cap_range": max(values) - min(values),
            }
        )
    return normalized, summaries, memberships


def _rank_associations(
    rows: list[dict[str, Any]], memberships: dict[tuple[int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Relate structured-CAP search ranks to random-CAP ranks without pooling panels."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["selection_metric"] == "cap" and row["evaluation_score"] == row["selection_score"]:
            groups[(row["model"], row["logical_objective"], row["split"])].append(row)
    expected_panels = {
        (identity["task"]["model"], membership["logical_objective"], split)
        for identity in memberships.values()
        if (membership := identity["membership"]).get("metric") == "cap"
        for split in SPLITS
    }
    if set(groups) != expected_panels:
        raise ValueError("Structured-CAP rank-panel coverage differs from the frozen manifest.")
    output = []
    for group_key, group in sorted(groups.items()):
        by_membership: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_membership[(row["task_index"], row["membership_index"])].append(row)
        ordered = sorted(by_membership)
        structured = []
        for key in ordered:
            metric = memberships[key]["membership"].get("optimized_metric")
            if not isinstance(metric, list) or not metric:
                raise ValueError(f"Structured CAP metric is missing for rank membership {key}.")
            objective_values = memberships[key]["membership"].get("objective_values")
            if objective_values is not None and objective_values != metric:
                raise ValueError(f"Structured CAP metric vectors differ for membership {key}.")
            structured.append(_finite(metric[0], f"structured CAP {key}"))
        random_columns = {
            "five_pairing_mean": [
                statistics.fmean(row["cap"] for row in by_membership[key]) for key in ordered
            ]
        }
        for replicate in REPLICATES:
            random_columns[f"pairing_replicate_{replicate}"] = [
                next(row["cap"] for row in by_membership[key] if row["replicate"] == replicate)
                for key in ordered
            ]
        correlations = {}
        for source, random_values in random_columns.items():
            correlations[source] = (
                _spearman(structured, random_values) if len(ordered) >= 3 else (None, False, False)
            )
        replicate_rhos = [correlations[f"pairing_replicate_{rep}"][0] for rep in REPLICATES]
        finite_replicate_rhos = [float(value) for value in replicate_rhos if value is not None]
        for source, (rho, constant, ties) in correlations.items():
            output.append(
                {
                    "model": group_key[0],
                    "logical_objective": group_key[1],
                    "split": group_key[2],
                    "random_source": source,
                    "structured_objective_index": 0,
                    "structured_direction": "maximize",
                    "n_checkpoints": len(ordered),
                    "spearman_rho": rho,
                    "constant": constant,
                    "ties": ties,
                    "insufficient_n": len(ordered) < 3,
                    "pairing_seed_rho_finite_count": len(finite_replicate_rhos),
                    "pairing_seed_rho_undefined_count": 5 - len(finite_replicate_rhos),
                    "pairing_seed_rho_min": (
                        min(finite_replicate_rhos) if finite_replicate_rhos else None
                    ),
                    "pairing_seed_rho_max": (
                        max(finite_replicate_rhos) if finite_replicate_rhos else None
                    ),
                    "pairing_seed_rho_range": (
                        max(finite_replicate_rhos) - min(finite_replicate_rhos)
                        if finite_replicate_rhos
                        else None
                    ),
                }
            )
    if len(output) != len(expected_panels) * 6:
        raise ValueError("Structured/random rank-association cardinality differs.")
    return output


def build_random_cap_analysis(
    manifest: dict[str, Any], aggregate: dict[str, Any]
) -> dict[str, Any]:
    """Return authenticated random-CAP values, variability, and rank diagnostics."""
    rows, variability, memberships = _normalize(manifest, aggregate)
    associations = _rank_associations(rows, memberships) if manifest["mode"] == "discovery" else []
    membership_scores = sum(
        len(task.get("memberships", [])) * len(task.get("scores", []))
        for task in manifest["tasks"]
    )
    return {
        "method": {
            "schema_version": SCHEMA_VERSION,
            "mode": manifest["mode"],
            "purpose": "negative_control_diagnostic_only",
            "pairing_replicates_per_split": 5,
            "tasks": len(manifest["tasks"]),
            "memberships": len(memberships),
            "membership_scores": membership_scores,
            "seed_rows": len(rows),
            "variability_rows": len(variability),
            "rank_association_rows": len(associations),
            "rank_panels": len(associations) // 6,
            "rank_panels_are_unpooled": True,
            "random_cap_did_not_select_hyperparameters_or_checkpoints": True,
        },
        "random_cap_values": rows,
        "random_cap_variability": variability,
        "structured_random_rank_associations": associations,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one rectangular table atomically without replacing different evidence."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"Refusing to replace different analysis table: {path}")
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_random_cap_analysis(
    manifest_path: Path, aggregate_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Write a portable, hash-bound random-CAP diagnostic package."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    result = build_random_cap_analysis(manifest, aggregate)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    namespace = manifest_path.parent.name
    if manifest["mode"] == "discovery" and (
        namespace != DISCOVERY_NAMESPACE
        or manifest_sha256 != DISCOVERY_MANIFEST_SHA256
        or manifest["content_sha256"] != DISCOVERY_MANIFEST_CONTENT_SHA256
        or manifest.get("random_pairing", {}).get("content_sha256")
        != DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256
    ):
        raise ValueError("Discovery random-CAP analysis is not bound to discovery_5293a29.")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in (
        "random_cap_values",
        "random_cap_variability",
        "structured_random_rank_associations",
    ):
        rows = result[name]
        if rows:
            path = output_dir / f"{name}.csv"
            _write_csv(path, rows)
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "method": result["method"],
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_namespace": namespace,
            "random_pairing_content_sha256": manifest.get("random_pairing", {}).get(
                "content_sha256"
            ),
            "execution_commit": manifest.get("execution_commit"),
            "campaign_root": manifest.get("campaign_root"),
            "confirmation_matrix_content_sha256": manifest.get(
                "confirmation_matrix_content_sha256"
            ),
            "runner_manifest_content_sha256": manifest.get("runner_manifest_content_sha256"),
            "reserve_extension_content_sha256": manifest.get("reserve_extension_content_sha256"),
            "checkpoint_selections_sha256": manifest.get("checkpoint_selections_sha256"),
            "endpoint_receipt_sha256": manifest.get("endpoint_receipt_sha256"),
            "aggregate": str(aggregate_path.resolve()),
            "aggregate_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
        },
        "artifacts": artifacts,
    }
    receipt["content_sha256"] = _content_hash(receipt)
    receipt_path = output_dir / "random_cap_analysis.json"
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    _write_text(receipt_path, serialized)
    return result


def _write_text(path: Path, text: str) -> None:
    """Write one text artifact atomically and idempotently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"Refusing to replace different analysis receipt: {path}")
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


__all__ = ["build_random_cap_analysis", "write_random_cap_analysis"]
