"""Contracts for authenticated random-CAP negative-control analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import pytest

from src.evaluation import (
    physics_background_pairing_random_cap_analysis as analysis_module,
)
from src.evaluation.physics_background_pairing_random_cap_analysis import (
    DOWNSTREAM_DATASETS,
    _content_hash,
    build_random_cap_analysis,
    write_random_cap_analysis,
)


def _inputs() -> tuple[dict, dict]:
    """Build a minimal three-checkpoint, two-split authenticated fixture."""
    tables = [
        {
            "split": split,
            "replicate": replicate,
            "seed": 1000 + split_index * 10 + replicate,
            "file_sha256": f"{split_index * 10 + replicate + 1:064x}",
        }
        for split_index, split in enumerate(("validate", "test"))
        for replicate in range(5)
    ]
    tasks = []
    for index in range(3):
        tasks.append(
            {
                "task_index": index,
                "model": "ae",
                "trial_number": 10 + index,
                "scores": ["native"],
                "checkpoints": [
                    {
                        "checkpoint_id": "selected_000",
                        "source_sha256": f"{index + 100:064x}",
                    }
                ],
                "memberships": [
                    {
                        "membership_index": index,
                        "checkpoint_id": "selected_000",
                        "logical_objective": "native__cap__cdf",
                        "score": "native",
                        "metric": "cap",
                        "strategy": "cdf",
                        "optimized_metric": [float(index), 0.1],
                    }
                ],
            }
        )
    manifest = {
        "schema_version": 1,
        "mode": "discovery",
        "purpose": "offline_random_cap_diagnostic_only",
        "runner_manifest_content_sha256": "b" * 64,
        "reserve_extension_content_sha256": "c" * 64,
        "checkpoint_selections_sha256": "d" * 64,
        "endpoint_receipt_sha256": "e" * 64,
        "random_pairing": {"tables": tables},
        "tasks": tasks,
    }
    manifest["content_sha256"] = _content_hash(manifest)
    rows = []
    for task in tasks:
        membership = task["memberships"][0]
        checkpoint = task["checkpoints"][0]
        for split in ("validate", "test"):
            split_tables = [row for row in tables if row["split"] == split]
            for table in split_tables:
                rows.append(
                    {
                        "mode": "discovery",
                        "task_index": task["task_index"],
                        "model": "ae",
                        "trial_number": task["trial_number"],
                        "training_seed": "",
                        "membership_index": membership["membership_index"],
                        "logical_objective": membership["logical_objective"],
                        "selection_score": "native",
                        "selection_metric": "cap",
                        "selection_strategy": "cdf",
                        "evaluation_score": "native",
                        "split": split,
                        "replicate": table["replicate"],
                        "pairing_seed": table["seed"],
                        "cap": task["task_index"] + table["replicate"] / 10,
                        "checkpoint_id": "selected_000",
                        "pair_table_sha256": table["file_sha256"],
                        "checkpoint_sha256": checkpoint["source_sha256"],
                    }
                )
    summaries = []
    for task in tasks:
        for split in ("validate", "test"):
            selected = [
                row
                for row in rows
                if row["task_index"] == task["task_index"] and row["split"] == split
            ]
            values = [row["cap"] for row in selected]
            first = selected[0]
            summaries.append(
                {
                    "mode": "discovery",
                    "task_index": task["task_index"],
                    "membership_index": task["task_index"],
                    "model": "ae",
                    "trial_number": task["trial_number"],
                    "training_seed": "",
                    "logical_objective": "native__cap__cdf",
                    "selection_score": "native",
                    "selection_metric": "cap",
                    "selection_strategy": "cdf",
                    "evaluation_score": "native",
                    "split": split,
                    "checkpoint_id": "selected_000",
                    "checkpoint_sha256": first["checkpoint_sha256"],
                    "n_pairings": 5,
                    "cap_mean": statistics.fmean(values),
                    "cap_sample_variance": statistics.variance(values),
                }
            )
    aggregate = {
        "schema_version": 1,
        "manifest_content_sha256": manifest["content_sha256"],
        "seed_split_values": rows,
        "seed_split_values_sha256": _content_hash(rows),
        "summary": summaries,
        "summary_sha256": _content_hash(summaries),
    }
    return manifest, aggregate


def _downstream_rows(manifest: dict) -> list[dict]:
    """Build exact 20-anomaly endpoint coverage for every logical membership."""
    rows = []
    for task in manifest["tasks"]:
        checkpoints = {row["checkpoint_id"]: row for row in task["checkpoints"]}
        for membership in task["memberships"]:
            checkpoint = checkpoints[membership["checkpoint_id"]]
            for anomaly in DOWNSTREAM_DATASETS:
                for endpoint in ("efficiency_250hz", "auprc"):
                    index = int(membership["membership_index"])
                    value = (index + 1) / 4 if endpoint == "efficiency_250hz" else (3 - index) / 4
                    rows.append(
                        {
                            "membership_index": str(index),
                            "model": task["model"],
                            "logical_objective": membership["logical_objective"],
                            "score": membership["score"],
                            "trial_number": str(task["trial_number"]),
                            "checkpoint": "unused-in-fixture",
                            "checkpoint_sha256": checkpoint["source_sha256"],
                            "anomaly": anomaly,
                            "selector": membership["metric"],
                            "pairing": membership["strategy"],
                            "endpoint": endpoint,
                            "value": str(value),
                        }
                    )
    return rows


def _add_structured_cap_values(manifest: dict, aggregate: dict) -> None:
    """Bind same-checkpoint structured CAP for every confirmation score and split."""
    rows = []
    for task in manifest["tasks"]:
        checkpoints = {row["checkpoint_id"]: row for row in task["checkpoints"]}
        for membership in task["memberships"]:
            if membership["metric"] != "cap":
                continue
            checkpoint = checkpoints[membership["checkpoint_id"]]
            for score_index, score in enumerate(task["scores"]):
                for split_index, split in enumerate(("validate", "test")):
                    rows.append(
                        {
                            "mode": "confirmation",
                            "task_index": task["task_index"],
                            "membership_index": membership["membership_index"],
                            "model": task["model"],
                            "training_seed": task.get("training_seed", ""),
                            "logical_objective": membership["logical_objective"],
                            "selection_score": membership["score"],
                            "selection_metric": "cap",
                            "selection_strategy": membership["strategy"],
                            "evaluation_score": score,
                            "split": split,
                            "checkpoint_id": membership["checkpoint_id"],
                            "checkpoint_sha256": checkpoint["source_sha256"],
                            "cap": task["task_index"] + score_index / 10 + split_index / 100,
                        }
                    )
    aggregate["structured_cap_values"] = rows
    aggregate["structured_cap_values_sha256"] = _content_hash(rows)
    aggregate["structured_cap_values_count"] = len(rows)


def test_analysis_recomputes_variability_and_unpooled_rank_association() -> None:
    """Seed variability and rank panels are recomputed without cross-panel pooling."""
    manifest, aggregate = _inputs()
    result = build_random_cap_analysis(manifest, aggregate)
    assert result["method"]["seed_rows"] == 30
    assert result["method"]["variability_rows"] == 6
    assert result["method"]["rank_association_rows"] == 12
    assert {row["split"] for row in result["structured_random_rank_associations"]} == {
        "validate",
        "test",
    }
    assert all(
        row["spearman_rho"] == pytest.approx(1.0)
        for row in result["structured_random_rank_associations"]
    )


def test_confirmation_analysis_requires_and_uses_structured_metric_identity() -> None:
    """Confirmation mode emits the same authenticated unpooled rank evidence."""
    manifest, aggregate = _inputs()
    manifest["mode"] = "confirmation"
    for task in manifest["tasks"]:
        membership = task["memberships"][0]
        membership.update(
            {
                "selection_metric_name": "cap",
                "selection_metric_direction": "maximize",
                "selection_metric_objective_index": 0,
                "objective_values": membership["optimized_metric"],
            }
        )
    manifest["content_sha256"] = _content_hash(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    for row in aggregate["seed_split_values"]:
        row["mode"] = "confirmation"
    for row in aggregate["summary"]:
        row["mode"] = "confirmation"
    aggregate["manifest_content_sha256"] = manifest["content_sha256"]
    aggregate["seed_split_values_sha256"] = _content_hash(aggregate["seed_split_values"])
    aggregate["summary_sha256"] = _content_hash(aggregate["summary"])
    _add_structured_cap_values(manifest, aggregate)
    result = build_random_cap_analysis(manifest, aggregate)
    assert result["method"]["rank_panels"] == 2
    assert result["method"]["rank_association_rows"] == 12

    manifest["tasks"][0]["memberships"][0]["selection_metric_name"] = "wasserstein"
    manifest["content_sha256"] = _content_hash(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    aggregate["manifest_content_sha256"] = manifest["content_sha256"]
    with pytest.raises(ValueError, match="objective identity differs"):
        build_random_cap_analysis(manifest, aggregate)


@pytest.mark.parametrize("target", ["seed_hash", "summary_value", "pair_seed"])
def test_analysis_rejects_tampered_aggregate(target: str) -> None:
    """Child hashes, summaries, and random-table identities are fail-closed."""
    manifest, aggregate = _inputs()
    if target == "seed_hash":
        aggregate["seed_split_values_sha256"] = "0" * 64
    elif target == "summary_value":
        aggregate["summary"][0]["cap_mean"] += 1
        aggregate["summary_sha256"] = _content_hash(aggregate["summary"])
    else:
        aggregate["seed_split_values"][0]["pairing_seed"] += 1
        aggregate["seed_split_values_sha256"] = _content_hash(aggregate["seed_split_values"])
    with pytest.raises(ValueError):
        build_random_cap_analysis(manifest, aggregate)


def test_writer_persists_hash_bound_tables_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Portable tables and their receipt can be reproduced byte-for-byte."""
    manifest, aggregate = _inputs()
    manifest_path = tmp_path / "manifest.json"
    aggregate_path = tmp_path / "aggregate.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    monkeypatch.setattr(analysis_module, "DISCOVERY_NAMESPACE", manifest_path.parent.name)
    monkeypatch.setattr(
        analysis_module,
        "DISCOVERY_MANIFEST_SHA256",
        analysis_module.hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        analysis_module, "DISCOVERY_MANIFEST_CONTENT_SHA256", manifest["content_sha256"]
    )
    monkeypatch.setattr(
        analysis_module,
        "DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256",
        manifest["random_pairing"].get("content_sha256"),
    )
    monkeypatch.setattr(analysis_module, "DISCOVERY_TASKS", 3)
    monkeypatch.setattr(analysis_module, "DISCOVERY_MEMBERSHIPS", 3)
    monkeypatch.setattr(analysis_module, "DISCOVERY_MEMBERSHIP_SCORES", 3)
    output = tmp_path / "results"
    write_random_cap_analysis(manifest_path, aggregate_path, output)
    first = (output / "random_cap_analysis.json").read_bytes()
    receipt = json.loads(first)
    assert receipt["inputs"]["runner_manifest_content_sha256"] == "b" * 64
    assert receipt["inputs"]["reserve_extension_content_sha256"] == "c" * 64
    assert receipt["inputs"]["checkpoint_selections_sha256"] == "d" * 64
    assert receipt["inputs"]["endpoint_receipt_sha256"] == "e" * 64
    write_random_cap_analysis(manifest_path, aggregate_path, output)
    assert (output / "random_cap_analysis.json").read_bytes() == first
    assert (output / "random_cap_values.csv").exists()
    assert (output / "structured_random_rank_associations.csv").exists()


def test_rank_analysis_handles_constant_and_tied_panels() -> None:
    """Constant inputs are undefined while ties use deterministic average ranks."""
    manifest, aggregate = _inputs()
    for task in manifest["tasks"]:
        task["memberships"][0]["optimized_metric"][0] = 1.0
    manifest["content_sha256"] = _content_hash(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    aggregate["manifest_content_sha256"] = manifest["content_sha256"]
    constant = build_random_cap_analysis(manifest, aggregate)
    assert all(
        row["spearman_rho"] is None for row in constant["structured_random_rank_associations"]
    )
    assert all(
        row["pairing_seed_rho_undefined_count"] == 5
        for row in constant["structured_random_rank_associations"]
    )

    manifest, aggregate = _inputs()
    manifest["tasks"][1]["memberships"][0]["optimized_metric"][0] = 0.0
    manifest["content_sha256"] = _content_hash(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    aggregate["manifest_content_sha256"] = manifest["content_sha256"]
    tied = build_random_cap_analysis(manifest, aggregate)
    assert all(row["ties"] for row in tied["structured_random_rank_associations"])
    assert all(
        row["spearman_rho"] == pytest.approx(0.8660254037844387)
        for row in tied["structured_random_rank_associations"]
    )


def test_rank_analysis_rejects_missing_structured_panel_metric() -> None:
    """A malformed membership cannot silently remove a structured-CAP rank panel."""
    manifest, aggregate = _inputs()
    manifest["tasks"][0]["memberships"][0].pop("optimized_metric")
    manifest["content_sha256"] = _content_hash(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    aggregate["manifest_content_sha256"] = manifest["content_sha256"]
    with pytest.raises(ValueError, match="Structured CAP metric is missing"):
        build_random_cap_analysis(manifest, aggregate)


def test_downstream_associations_preserve_endpoint_score_and_split_panels() -> None:
    """Structured and six random sources are correlated with both endpoints per split."""
    manifest, aggregate = _inputs()
    result = build_random_cap_analysis(manifest, aggregate, _downstream_rows(manifest))
    rows = result["cap_downstream_rank_associations"]
    assert len(rows) == 28
    assert result["method"]["downstream_association_panels"] == 4
    assert result["method"]["downstream_association_sources"] == 7
    assert result["method"]["downstream_association_undefined_rows"] == 0
    assert {row["split"] for row in rows} == {"validate", "test"}
    assert {row["downstream_endpoint"] for row in rows} == {
        "efficiency_250hz",
        "auprc",
    }
    assert {row["cap_source"] for row in rows} == {
        "structured_cap",
        "five_pairing_mean",
        *(f"pairing_replicate_{replicate}" for replicate in range(5)),
    }
    assert all(row["n_checkpoints"] == 3 for row in rows)
    assert all(row["n_anomalies_per_checkpoint"] == 20 for row in rows)
    assert all(
        row["spearman_rho"] == pytest.approx(1.0)
        for row in rows
        if row["downstream_endpoint"] == "efficiency_250hz"
    )
    assert all(
        row["spearman_rho"] == pytest.approx(-1.0)
        for row in rows
        if row["downstream_endpoint"] == "auprc"
    )


@pytest.mark.parametrize("target", ["identity", "coverage"])
def test_downstream_exact_join_fails_closed(target: str) -> None:
    """Endpoint identity mismatches and incomplete anomaly coverage cannot be analyzed."""
    manifest, aggregate = _inputs()
    rows = _downstream_rows(manifest)
    if target == "identity":
        rows[0]["checkpoint_sha256"] = "f" * 64
        match = "checkpoint_sha256 differs"
    else:
        rows.pop()
        match = "coverage differs"
    with pytest.raises(ValueError, match=match):
        build_random_cap_analysis(manifest, aggregate, rows)


def test_downstream_constant_panels_remain_explicit_and_undefined() -> None:
    """Undefined endpoint associations retain their panel cardinality and reason flags."""
    manifest, aggregate = _inputs()
    rows = _downstream_rows(manifest)
    for row in rows:
        row["value"] = "0.5"
    result = build_random_cap_analysis(manifest, aggregate, rows)
    associations = result["cap_downstream_rank_associations"]
    assert len(associations) == 28
    assert all(not row["spearman_defined"] for row in associations)
    assert all(row["constant"] for row in associations)
    assert result["method"]["downstream_association_undefined_rows"] == 28


def test_downstream_panels_do_not_pool_models_or_score_streams() -> None:
    """Model and native/OAS identities remain distinct even for undersized panels."""
    manifest, aggregate = _inputs()
    task = manifest["tasks"][2]
    membership = task["memberships"][0]
    task["model"] = "vae"
    task["scores"] = ["residual_oas"]
    membership["score"] = "residual_oas"
    membership["logical_objective"] = "residual_oas__cap__cdf"
    for collection in (aggregate["seed_split_values"], aggregate["summary"]):
        for row in collection:
            if row["task_index"] == 2:
                row["model"] = "vae"
                row["selection_score"] = "residual_oas"
                row["evaluation_score"] = "residual_oas"
                row["logical_objective"] = "residual_oas__cap__cdf"
    manifest["content_sha256"] = _content_hash(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    aggregate["manifest_content_sha256"] = manifest["content_sha256"]
    aggregate["seed_split_values_sha256"] = _content_hash(aggregate["seed_split_values"])
    aggregate["summary_sha256"] = _content_hash(aggregate["summary"])
    result = build_random_cap_analysis(manifest, aggregate, _downstream_rows(manifest))
    rows = result["cap_downstream_rank_associations"]
    assert {(row["model"], row["selection_score"]) for row in rows} == {
        ("ae", "native"),
        ("vae", "residual_oas"),
    }
    assert {(row["model"], row["selection_score"], row["n_checkpoints"]) for row in rows} == {
        ("ae", "native", 2),
        ("vae", "residual_oas", 1),
    }
    assert all(row["insufficient_n"] and not row["spearman_defined"] for row in rows)


def test_writer_authenticates_endpoint_receipt_and_persists_downstream_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The downstream table and receipt are hash-bound in a fresh analysis version."""
    manifest, aggregate = _inputs()
    endpoint_dir = tmp_path / "endpoint"
    endpoint_dir.mkdir()
    combined_path = endpoint_dir / "combined_long.csv"
    downstream_rows = _downstream_rows(manifest)
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(downstream_rows[0]))
        writer.writeheader()
        writer.writerows(downstream_rows)
    endpoint_receipt = {
        "artifact_hashes": {
            "combined_long.csv": hashlib.sha256(combined_path.read_bytes()).hexdigest()
        },
        "method": {
            "schema_version": 1,
            "anomalies": 20,
            "candidate_memberships": 3,
            "combined_rows": 120,
            "rows_per_endpoint": 60,
        },
    }
    endpoint_path = endpoint_dir / "endpoint_analysis.json"
    endpoint_path.write_text(json.dumps(endpoint_receipt), encoding="utf-8")
    manifest["endpoint_receipt_sha256"] = hashlib.sha256(endpoint_path.read_bytes()).hexdigest()
    manifest["content_sha256"] = _content_hash(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    aggregate["manifest_content_sha256"] = manifest["content_sha256"]
    manifest_path = tmp_path / "manifest.json"
    aggregate_path = tmp_path / "aggregate.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    monkeypatch.setattr(analysis_module, "DISCOVERY_NAMESPACE", manifest_path.parent.name)
    monkeypatch.setattr(
        analysis_module,
        "DISCOVERY_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        analysis_module, "DISCOVERY_MANIFEST_CONTENT_SHA256", manifest["content_sha256"]
    )
    monkeypatch.setattr(
        analysis_module,
        "DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256",
        manifest["random_pairing"].get("content_sha256"),
    )
    monkeypatch.setattr(analysis_module, "DISCOVERY_TASKS", 3)
    monkeypatch.setattr(analysis_module, "DISCOVERY_MEMBERSHIPS", 3)
    monkeypatch.setattr(analysis_module, "DISCOVERY_MEMBERSHIP_SCORES", 3)
    monkeypatch.setattr(
        analysis_module,
        "DISCOVERY_ENDPOINT_ANALYSIS_SHA256",
        hashlib.sha256(endpoint_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        analysis_module,
        "DISCOVERY_COMBINED_LONG_SHA256",
        hashlib.sha256(combined_path.read_bytes()).hexdigest(),
    )
    output = tmp_path / "analysis_versions" / "downstream_v1"
    write_random_cap_analysis(
        manifest_path, aggregate_path, output, endpoint_analysis_path=endpoint_path
    )
    receipt = json.loads((output / "random_cap_analysis.json").read_text())
    artifact = output / "cap_downstream_rank_associations.csv"
    assert artifact.exists()
    assert receipt["artifacts"][artifact.name] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert (
        receipt["inputs"]["combined_long_sha256"]
        == endpoint_receipt["artifact_hashes"]["combined_long.csv"]
    )
    with pytest.raises(FileExistsError, match="new versioned output directory"):
        write_random_cap_analysis(manifest_path, aggregate_path, output)

    combined_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        write_random_cap_analysis(
            manifest_path,
            aggregate_path,
            tmp_path / "analysis_versions" / "downstream_v2",
            endpoint_analysis_path=endpoint_path,
        )


def test_confirmation_writer_joins_cell_seed_endpoint_analysis(tmp_path: Path) -> None:
    """Confirmation endpoints_long rows join to random memberships by cell and seed."""
    manifest, aggregate = _inputs()
    manifest["mode"] = "confirmation"
    manifest["endpoint_receipt_sha256"] = "f" * 64
    endpoint_rows = []
    for task in manifest["tasks"]:
        task["training_seed"] = 100 + task["task_index"]
        checkpoint = task["checkpoints"][0]
        checkpoint["source_path"] = f"/sealed/checkpoint_{task['task_index']}.ckpt"
        membership = task["memberships"][0]
        membership["cell_index"] = task["task_index"]
        membership["seed"] = task["training_seed"]
        membership["selection_metric_name"] = "cap"
        membership["selection_metric_direction"] = "maximize"
        membership["selection_metric_objective_index"] = 0
        membership["objective_values"] = membership["optimized_metric"]
        for anomaly in DOWNSTREAM_DATASETS:
            for endpoint in ("efficiency_250hz", "auprc"):
                endpoint_rows.append(
                    {
                        "cell_index": task["task_index"],
                        "configuration_index": task["task_index"],
                        "model": task["model"],
                        "trial_number": task["trial_number"],
                        "seed": task["training_seed"],
                        "score": "native",
                        "selector": "cap",
                        "pairing": "cdf",
                        "logical_objective": membership["logical_objective"],
                        "checkpoint_path": checkpoint["source_path"],
                        "checkpoint_sha256": checkpoint["source_sha256"],
                        "anomaly": anomaly,
                        "endpoint": endpoint,
                        "value": (task["task_index"] + 1) / 4,
                    }
                )
    for row in aggregate["seed_split_values"]:
        row["mode"] = "confirmation"
        row["training_seed"] = 100 + row["task_index"]
    for row in aggregate["summary"]:
        row["mode"] = "confirmation"
        row["training_seed"] = 100 + row["task_index"]
    for task in manifest["tasks"]:
        task["scores"].append("residual_oas")
    aggregate["seed_split_values"].extend(
        {**row, "evaluation_score": "residual_oas"} for row in list(aggregate["seed_split_values"])
    )
    aggregate["summary"].extend(
        {**row, "evaluation_score": "residual_oas"} for row in list(aggregate["summary"])
    )
    oas_endpoint_rows = [{**row, "score": "residual_oas"} for row in endpoint_rows]
    manifest["content_sha256"] = _content_hash(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    aggregate["manifest_content_sha256"] = manifest["content_sha256"]
    aggregate["seed_split_values_sha256"] = _content_hash(aggregate["seed_split_values"])
    aggregate["summary_sha256"] = _content_hash(aggregate["summary"])
    _add_structured_cap_values(manifest, aggregate)
    endpoint_dir = tmp_path / "confirmation_endpoint"
    endpoint_dir.mkdir()
    endpoints_path = endpoint_dir / "endpoints_long.csv"
    with endpoints_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    oas_endpoints_path = endpoint_dir / "oas_endpoints_long.csv"
    with oas_endpoints_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oas_endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(oas_endpoint_rows)
    endpoint_receipt = {
        "method": {
            "schema_version": 1,
            "anomalies": 20,
            "endpoint_rows": 120,
            "oas_endpoint_rows": 120,
        },
        "inputs": {"endpoint_receipt_sha256": "f" * 64},
        "artifacts": {
            "endpoints_long.csv": hashlib.sha256(endpoints_path.read_bytes()).hexdigest(),
            "oas_endpoints_long.csv": hashlib.sha256(oas_endpoints_path.read_bytes()).hexdigest(),
        },
    }
    endpoint_analysis_path = endpoint_dir / "confirmation_endpoint_analysis.json"
    endpoint_analysis_path.write_text(json.dumps(endpoint_receipt), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    aggregate_path = tmp_path / "aggregate.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    output = tmp_path / "confirmation_downstream_v1"
    result = write_random_cap_analysis(
        manifest_path,
        aggregate_path,
        output,
        endpoint_analysis_path=endpoint_analysis_path,
    )
    assert result["method"]["downstream_rows"] == 240
    assert result["method"]["downstream_association_rows"] == 56
    receipt = json.loads((output / "random_cap_analysis.json").read_text())
    assert receipt["inputs"]["endpoint_analysis_kind"] == "confirmation"
    assert receipt["inputs"]["upstream_endpoint_receipt_sha256"] == "f" * 64
