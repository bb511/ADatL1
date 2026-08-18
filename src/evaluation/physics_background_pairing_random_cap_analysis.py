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

from src.evaluation.physics_background_pairing_analysis import (
    DOWNSTREAM_DATASETS,
    _spearman,
)

SCHEMA_VERSION = 1
SPLITS = ("validate", "test")
REPLICATES = tuple(range(5))
DOWNSTREAM_ENDPOINTS = ("efficiency_250hz", "auprc")
DISCOVERY_NAMESPACE = "discovery_immutable_presort_20260818"
DISCOVERY_MANIFEST_SHA256 = "ed5b59cbc4eaec05c2f384e18659767b794656182d53e6ea2ac363efdc80b3bd"
DISCOVERY_MANIFEST_CONTENT_SHA256 = (
    "8371c822f16e2eeaf2f65f6d1ee9e96c9c664c03f0b86804c86bdec2bb4a3f66"
)
DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256 = (
    "4ed193bf05b453bbfaf95de9ffea2ab618956b235d388c39e1aafcb6400f6870"
)
DISCOVERY_ENDPOINT_ANALYSIS_SHA256 = (
    "b8baede48d8da59aecd6acdd816120ef754cda9edfb25e4c15d89c92afec9d1d"
)
DISCOVERY_COMBINED_LONG_SHA256 = "51018c9993ed01d1b8c612ca7f1ccfbdb82b3480e189e0995677c7ac975c4f36"
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


def _sha256(path: Path) -> str:
    """Return the physical SHA-256 digest of one input artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _load_downstream_rows(
    endpoint_analysis_path: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Authenticate and load the endpoint long table named by its analysis receipt."""
    receipt = json.loads(endpoint_analysis_path.read_text(encoding="utf-8"))
    method = receipt.get("method", {})
    artifact_hashes = receipt.get("artifact_hashes", {})
    if "combined_long.csv" in artifact_hashes:
        artifact_names = ("combined_long.csv",)
        analysis_kind = "discovery"
        expected_columns = [
            "membership_index",
            "model",
            "logical_objective",
            "score",
            "trial_number",
            "checkpoint",
            "checkpoint_sha256",
            "anomaly",
            "selector",
            "pairing",
            "endpoint",
            "value",
        ]
    elif "endpoints_long.csv" in artifact_hashes:
        artifact_names = ("endpoints_long.csv",)
        analysis_kind = "confirmation"
        expected_columns = [
            "cell_index",
            "configuration_index",
            "model",
            "trial_number",
            "seed",
            "score",
            "selector",
            "pairing",
            "logical_objective",
            "checkpoint_path",
            "checkpoint_sha256",
            "anomaly",
            "endpoint",
            "value",
        ]
    else:
        raise ValueError("Endpoint analysis receipt lacks a supported downstream table.")
    expected_hashes = {name: artifact_hashes.get(name) for name in artifact_names}
    if method.get("schema_version") != SCHEMA_VERSION or any(
        not isinstance(digest, str) or len(digest) != 64 for digest in expected_hashes.values()
    ):
        raise ValueError("Endpoint analysis receipt schema or downstream-table hash differs.")
    table_paths = {name: endpoint_analysis_path.parent / name for name in artifact_names}
    rows = []
    for name, table_path in table_paths.items():
        if not table_path.is_file() or _sha256(table_path) != expected_hashes[name]:
            raise ValueError("Endpoint downstream table does not match its analysis receipt.")
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_columns:
                raise ValueError("Endpoint downstream-table schema differs.")
            rows.extend(reader)
    if analysis_kind == "confirmation":
        membership_indices = {}
        for task in manifest.get("tasks", []):
            for membership in task.get("memberships", []):
                key = (int(membership["cell_index"]), int(membership["seed"]))
                if key in membership_indices:
                    raise ValueError(f"Duplicate confirmation endpoint membership key {key}.")
                membership_indices[key] = int(membership["membership_index"])
        normalized = []
        for row in rows:
            try:
                key = (int(row["cell_index"]), int(row["seed"]))
                membership_index = membership_indices[key]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Confirmation endpoint membership join differs.") from error
            normalized.append(
                {
                    "membership_index": membership_index,
                    "model": row["model"],
                    "logical_objective": row["logical_objective"],
                    "score": row["score"],
                    "trial_number": row["trial_number"],
                    "checkpoint": row["checkpoint_path"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "anomaly": row["anomaly"],
                    "selector": row["selector"],
                    "pairing": row["pairing"],
                    "endpoint": row["endpoint"],
                    "value": row["value"],
                }
            )
        rows = normalized
    provenance = {
        "endpoint_analysis": str(endpoint_analysis_path.resolve()),
        "endpoint_analysis_sha256": _sha256(endpoint_analysis_path),
        "endpoint_analysis_kind": analysis_kind,
        "downstream_tables": {
            name: {"path": str(table_paths[name].resolve()), "sha256": expected_hashes[name]}
            for name in artifact_names
        },
        "endpoint_method": method,
    }
    if analysis_kind == "discovery":
        table_path = table_paths["combined_long.csv"]
        provenance.update(
            {
                "combined_long": str(table_path.resolve()),
                "combined_long_sha256": expected_hashes["combined_long.csv"],
            }
        )
    else:
        provenance["upstream_endpoint_receipt_sha256"] = receipt.get("inputs", {}).get(
            "endpoint_receipt_sha256"
        )
    return rows, provenance


def _normalize_downstream(
    rows: list[dict[str, Any]],
    memberships: dict[tuple[int, int], dict[str, Any]],
) -> tuple[dict[tuple[int, int, str], float], dict[str, int]]:
    """Exact-join and average the two downstream endpoints over 20 anomalies."""
    by_membership_index: dict[int, tuple[tuple[int, int], dict[str, Any]]] = {}
    for key, identity in memberships.items():
        membership_index = key[1]
        if membership_index in by_membership_index:
            raise ValueError("Endpoint join requires globally unique membership indices.")
        by_membership_index[membership_index] = (key, identity)

    expected_anomalies = set(DOWNSTREAM_DATASETS)
    expected_keys = {
        (membership_index, endpoint, anomaly)
        for membership_index in by_membership_index
        for endpoint in DOWNSTREAM_ENDPOINTS
        for anomaly in DOWNSTREAM_DATASETS
    }
    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    seen = set()
    for supplied in rows:
        row = dict(supplied)
        try:
            membership_index = int(row["membership_index"])
            trial_number = int(row["trial_number"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Endpoint membership or trial identity is invalid.") from error
        joined = by_membership_index.get(membership_index)
        if joined is None:
            raise ValueError(f"Unexpected endpoint membership {membership_index}.")
        key, identity = joined
        task, membership, checkpoint = (
            identity["task"],
            identity["membership"],
            identity["checkpoint"],
        )
        endpoint = row.get("endpoint")
        anomaly = row.get("anomaly")
        row_key = (membership_index, endpoint, anomaly)
        if (
            row_key in seen
            or endpoint not in DOWNSTREAM_ENDPOINTS
            or anomaly not in expected_anomalies
        ):
            raise ValueError(f"Unexpected or duplicate downstream row {row_key}.")
        expected_pairing = membership.get("strategy") or "not_applicable"
        expected_fields = {
            "model": task["model"],
            "logical_objective": membership["logical_objective"],
            "score": membership.get("score", "native"),
            "checkpoint_sha256": checkpoint["source_sha256"],
            "selector": membership.get("metric"),
            "pairing": expected_pairing,
        }
        for field, value in expected_fields.items():
            if row.get(field) != value:
                raise ValueError(f"Endpoint {field} differs for membership {membership_index}.")
        if trial_number != int(task["trial_number"]):
            raise ValueError(f"Endpoint trial differs for membership {membership_index}.")
        source_path = checkpoint.get("source_path")
        if source_path is not None and row.get("checkpoint") != source_path:
            raise ValueError(
                f"Endpoint checkpoint path differs for membership {membership_index}."
            )
        value = _finite(row.get("value"), f"downstream {row_key}")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Downstream value lies outside [0, 1] for {row_key}.")
        seen.add(row_key)
        grouped[(key[0], key[1], endpoint)].append(value)
    if seen != expected_keys:
        raise ValueError(
            f"Endpoint coverage differs: expected {len(expected_keys)}, found {len(seen)}."
        )
    means = {}
    for key, values in grouped.items():
        if len(values) != len(DOWNSTREAM_DATASETS):
            raise ValueError(f"Endpoint anomaly count differs for membership panel {key}.")
        means[key] = statistics.fmean(values)
    expected_means = len(memberships) * len(DOWNSTREAM_ENDPOINTS)
    if len(means) != expected_means:
        raise ValueError("Endpoint mean coverage differs after the exact membership join.")
    return means, {
        "downstream_rows": len(rows),
        "downstream_memberships": len(memberships),
        "downstream_anomalies": len(DOWNSTREAM_DATASETS),
        "downstream_endpoints": len(DOWNSTREAM_ENDPOINTS),
    }


def _structured_cap(
    key: tuple[int, int], memberships: dict[tuple[int, int], dict[str, Any]]
) -> float:
    """Return one authenticated, maximize-direction structured CAP value."""
    membership = memberships[key]["membership"]
    metric = membership.get("optimized_metric")
    if not isinstance(metric, list) or not metric:
        raise ValueError(f"Structured CAP metric is missing for rank membership {key}.")
    objective_values = membership.get("objective_values")
    if objective_values is not None and objective_values != metric:
        raise ValueError(f"Structured CAP metric vectors differ for membership {key}.")
    objective_index = int(membership.get("selection_metric_objective_index", 0))
    if (
        membership.get("metric") != "cap"
        or objective_index != 0
        or objective_index >= len(metric)
        or membership.get("selection_metric_name", "cap") != "cap"
        or membership.get("selection_metric_direction", "maximize") != "maximize"
    ):
        raise ValueError(f"Structured CAP objective identity differs for membership {key}.")
    return _finite(metric[objective_index], f"structured CAP {key}")


def _downstream_rank_associations(
    rows: list[dict[str, Any]],
    memberships: dict[tuple[int, int], dict[str, Any]],
    downstream: dict[tuple[int, int, str], float],
) -> list[dict[str, Any]]:
    """Relate structured and random CAP ranks to downstream endpoints per panel."""
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["selection_metric"] == "cap" and row["evaluation_score"] == row["selection_score"]:
            groups[
                (
                    row["model"],
                    row["selection_score"],
                    row["logical_objective"],
                    row["selection_strategy"],
                    row["split"],
                )
            ].append(row)
    expected_panels = {
        (
            identity["task"]["model"],
            membership.get("score", "native"),
            membership["logical_objective"],
            membership.get("strategy", ""),
            split,
        )
        for identity in memberships.values()
        if (membership := identity["membership"]).get("metric") == "cap"
        for split in SPLITS
    }
    if set(groups) != expected_panels:
        raise ValueError("Downstream rank-panel coverage differs from the manifest.")

    output = []
    for panel, group in sorted(groups.items()):
        by_membership: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_membership[(row["task_index"], row["membership_index"])].append(row)
        ordered = sorted(by_membership)
        structured = [_structured_cap(key, memberships) for key in ordered]
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
        cap_sources = {"structured_cap": structured, **random_columns}
        for endpoint in DOWNSTREAM_ENDPOINTS:
            endpoint_values = [downstream[(key[0], key[1], endpoint)] for key in ordered]
            for source, cap_values in cap_sources.items():
                rho, constant, ties = (
                    _spearman(cap_values, endpoint_values)
                    if len(ordered) >= 3
                    else (None, False, False)
                )
                output.append(
                    {
                        "model": panel[0],
                        "selection_score": panel[1],
                        "logical_objective": panel[2],
                        "selection_strategy": panel[3],
                        "split": panel[4],
                        "downstream_endpoint": endpoint,
                        "cap_source": source,
                        "cap_direction": "maximize",
                        "downstream_direction": "maximize",
                        "n_checkpoints": len(ordered),
                        "n_anomalies_per_checkpoint": len(DOWNSTREAM_DATASETS),
                        "spearman_rho": rho,
                        "spearman_defined": rho is not None,
                        "constant": constant,
                        "ties": ties,
                        "insufficient_n": len(ordered) < 3,
                    }
                )
    expected_rows = len(expected_panels) * len(DOWNSTREAM_ENDPOINTS) * (2 + len(REPLICATES))
    if len(output) != expected_rows:
        raise ValueError("Downstream rank-association cardinality differs.")
    return output


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
        structured = [_structured_cap(key, memberships) for key in ordered]
        for key in ordered:
            membership = memberships[key]["membership"]
            if (
                memberships[key]["task"]["model"] != group_key[0]
                or membership.get("logical_objective") != group_key[1]
            ):
                raise ValueError(f"Structured CAP panel identity differs for membership {key}.")
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
    manifest: dict[str, Any],
    aggregate: dict[str, Any],
    downstream_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return authenticated random-CAP values, variability, and rank diagnostics."""
    rows, variability, memberships = _normalize(manifest, aggregate)
    associations = _rank_associations(rows, memberships)
    membership_scores = sum(
        len(task.get("memberships", [])) * len(task.get("scores", []))
        for task in manifest["tasks"]
    )
    method = {
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
    }
    result = {
        "method": method,
        "random_cap_values": rows,
        "random_cap_variability": variability,
        "structured_random_rank_associations": associations,
    }
    if downstream_rows is not None:
        downstream, downstream_method = _normalize_downstream(downstream_rows, memberships)
        downstream_associations = _downstream_rank_associations(rows, memberships, downstream)
        defined = sum(row["spearman_defined"] for row in downstream_associations)
        method.update(
            {
                **downstream_method,
                "downstream_association_rows": len(downstream_associations),
                "downstream_association_panels": len(downstream_associations)
                // (2 + len(REPLICATES)),
                "downstream_association_sources": 2 + len(REPLICATES),
                "downstream_association_defined_rows": defined,
                "downstream_association_undefined_rows": len(downstream_associations) - defined,
                "downstream_panels_are_unpooled": True,
            }
        )
        result["cap_downstream_rank_associations"] = downstream_associations
    return result


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


def _guard_output_package(output_dir: Path, includes_downstream: bool) -> None:
    """Refuse to extend an existing immutable package with a different table set."""
    if not output_dir.exists():
        return
    receipt_path = output_dir / "random_cap_analysis.json"
    if not receipt_path.is_file():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to reuse nonempty analysis directory without a receipt: {output_dir}"
            )
        return
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    existing_downstream = "cap_downstream_rank_associations.csv" in receipt.get("artifacts", {})
    if existing_downstream != includes_downstream:
        raise FileExistsError(
            "Refusing to change an existing analysis package's downstream table set; "
            "write a new versioned output directory."
        )


def write_random_cap_analysis(
    manifest_path: Path,
    aggregate_path: Path,
    output_dir: Path,
    endpoint_analysis_path: Path | None = None,
) -> dict[str, Any]:
    """Write a portable, hash-bound random-CAP diagnostic package."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    downstream_rows = None
    downstream_provenance = None
    if endpoint_analysis_path is not None:
        downstream_rows, downstream_provenance = _load_downstream_rows(
            endpoint_analysis_path, manifest
        )
    result = build_random_cap_analysis(manifest, aggregate, downstream_rows)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    namespace = manifest_path.parent.name
    if manifest["mode"] == "discovery" and (
        namespace != DISCOVERY_NAMESPACE
        or manifest_sha256 != DISCOVERY_MANIFEST_SHA256
        or manifest["content_sha256"] != DISCOVERY_MANIFEST_CONTENT_SHA256
        or manifest.get("random_pairing", {}).get("content_sha256")
        != DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256
    ):
        raise ValueError(
            "Discovery random-CAP analysis is not bound to the authenticated "
            "immutable-presort namespace."
        )
    if downstream_provenance is not None:
        endpoint_method = downstream_provenance.pop("endpoint_method")
        analysis_kind = downstream_provenance["endpoint_analysis_kind"]
        if analysis_kind != manifest["mode"]:
            raise ValueError("Endpoint-analysis mode differs from the random-CAP manifest.")
        if analysis_kind == "discovery":
            expected_method = {
                "schema_version": SCHEMA_VERSION,
                "anomalies": result["method"]["downstream_anomalies"],
                "candidate_memberships": result["method"]["downstream_memberships"],
                "combined_rows": result["method"]["downstream_rows"],
                "rows_per_endpoint": result["method"]["downstream_rows"]
                // len(DOWNSTREAM_ENDPOINTS),
            }
        else:
            expected_method = {
                "schema_version": SCHEMA_VERSION,
                "anomalies": result["method"]["downstream_anomalies"],
                "endpoint_rows": result["method"]["downstream_rows"],
            }
        for field, value in expected_method.items():
            if endpoint_method.get(field) != value:
                raise ValueError(f"Endpoint analysis method field {field} differs.")
        bound_receipt = manifest.get("endpoint_receipt_sha256")
        if analysis_kind == "discovery":
            if (
                bound_receipt is not None
                and bound_receipt != downstream_provenance["endpoint_analysis_sha256"]
            ):
                raise ValueError("Manifest endpoint receipt binding differs.")
            if (
                downstream_provenance["endpoint_analysis_sha256"]
                != DISCOVERY_ENDPOINT_ANALYSIS_SHA256
                or downstream_provenance["combined_long_sha256"] != DISCOVERY_COMBINED_LONG_SHA256
            ):
                raise ValueError("Discovery endpoint-analysis identity differs.")
        elif (
            bound_receipt is None
            or bound_receipt != downstream_provenance["upstream_endpoint_receipt_sha256"]
        ):
            raise ValueError("Confirmation endpoint-analysis provenance differs.")
    _guard_output_package(output_dir, downstream_rows is not None)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    artifact_names = [
        "random_cap_values",
        "random_cap_variability",
        "structured_random_rank_associations",
    ]
    if downstream_rows is not None:
        artifact_names.append("cap_downstream_rank_associations")
    for name in artifact_names:
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
            **(downstream_provenance or {}),
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
