"""Render the final physics background-pairing confirmation report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from src.evaluation.physics_background_pairing_confirmation_analysis import (
    _content_hash,
)
from src.evaluation.physics_background_pairing_paper_comparison import (
    build_paper_comparison,
    load_authenticated_paper_reference,
)
from src.evaluation.physics_background_pairing_random_cap_analysis import (
    DISCOVERY_MANIFEST_CONTENT_SHA256,
    DISCOVERY_MANIFEST_SHA256,
    DISCOVERY_MEMBERSHIP_SCORES,
    DISCOVERY_MEMBERSHIPS,
    DISCOVERY_NAMESPACE,
    DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256,
    DISCOVERY_TASKS,
)

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
SELECTORS = (
    ("cap", "flat_physical", "Flat CAP"),
    ("cap", "physics_summary", "Summary CAP"),
    ("cap", "typed_sliced_wasserstein", "Typed CAP"),
    ("cap", "jetclr", "JetCLR CAP"),
    ("cap", "cdf", "CDF CAP"),
    ("wasserstein", "not_applicable", "W1"),
    ("drift", "not_applicable", "Drift"),
)


def _sha256(path: Path) -> str:
    """Hash one artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one nonempty rectangular CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid report input header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty report input: {path}")
    return rows


def _load_analysis(analysis_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Authenticate the analysis manifest and every table it declares."""
    manifest_path = analysis_dir / "confirmation_endpoint_analysis.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.pop("content_sha256", None)
    if digest != _content_hash(manifest):
        raise ValueError("Confirmation analysis manifest content hash differs.")
    manifest["content_sha256"] = digest
    tables = {}
    for name, expected_hash in manifest.get("artifacts", {}).items():
        path = (analysis_dir / name).resolve(strict=True)
        if not path.is_relative_to(analysis_dir.resolve()) or _sha256(path) != expected_hash:
            raise ValueError(f"Confirmation analysis artifact changed: {path}")
        tables[name] = _read_csv(path)
    required = {
        "endpoints_across_seeds.csv",
        "comparison_summary.csv",
        "oas_vs_native_summary.csv",
        "reserve_seed_decisions.csv",
    }
    if not required <= set(tables):
        raise ValueError("Confirmation analysis lacks final-report tables.")
    return manifest, tables


def _load_random_analysis(
    analysis_dir: Path,
    expected_mode: str,
    *,
    confirmation_provenance: dict[str, str | None] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Authenticate one portable random-CAP analysis package and its tables."""
    analysis_dir = analysis_dir.resolve(strict=True)
    manifest_path = analysis_dir / "random_cap_analysis.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.pop("content_sha256", None)
    if digest != _content_hash(manifest):
        raise ValueError("Random-CAP analysis receipt content hash differs.")
    manifest["content_sha256"] = digest
    if (
        manifest.get("schema_version") != 1
        or manifest.get("method", {}).get("mode") != expected_mode
        or manifest.get("method", {}).get("purpose") != "negative_control_diagnostic_only"
        or not manifest.get("method", {}).get(
            "random_cap_did_not_select_hyperparameters_or_checkpoints"
        )
    ):
        raise ValueError(f"Random-CAP {expected_mode} analysis identity differs.")
    inputs = manifest.get("inputs", {})
    if expected_mode == "discovery":
        expected_identity = {
            "manifest_namespace": DISCOVERY_NAMESPACE,
            "manifest_sha256": DISCOVERY_MANIFEST_SHA256,
            "manifest_content_sha256": DISCOVERY_MANIFEST_CONTENT_SHA256,
            "random_pairing_content_sha256": DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256,
        }
        expected_counts = {
            "tasks": DISCOVERY_TASKS,
            "memberships": DISCOVERY_MEMBERSHIPS,
            "membership_scores": DISCOVERY_MEMBERSHIP_SCORES,
            "seed_rows": DISCOVERY_MEMBERSHIP_SCORES * 10,
            "variability_rows": DISCOVERY_MEMBERSHIP_SCORES * 2,
        }
        if any(inputs.get(key) != value for key, value in expected_identity.items()) or any(
            int(manifest["method"].get(key, -1)) != value for key, value in expected_counts.items()
        ):
            raise ValueError(
                "Discovery random-CAP authoritative provenance or cardinality differs."
            )
    elif confirmation_provenance is None or any(
        inputs.get(key) != value for key, value in confirmation_provenance.items()
    ):
        raise ValueError(
            "Confirmation random-CAP analysis is not bound to the endpoint provenance."
        )
    tables = {}
    for name, expected_hash in manifest.get("artifacts", {}).items():
        path = (analysis_dir / name).resolve(strict=True)
        if not path.is_relative_to(analysis_dir) or _sha256(path) != expected_hash:
            raise ValueError(f"Random-CAP analysis artifact changed: {path}")
        tables[name] = _read_csv(path)
    if "random_cap_variability.csv" not in tables or "random_cap_values.csv" not in tables:
        raise ValueError("Random-CAP analysis lacks values or variability evidence.")
    if "structured_random_rank_associations.csv" not in tables:
        raise ValueError(f"{expected_mode.title()} random-CAP analysis lacks rank associations.")
    if "cap_downstream_rank_associations.csv" not in tables:
        raise ValueError(
            f"{expected_mode.title()} random-CAP analysis lacks downstream rank associations."
        )
    values = tables["random_cap_values.csv"]
    variability = tables["random_cap_variability.csv"]
    required_value_fields = {
        "task_index",
        "membership_index",
        "evaluation_score",
        "split",
        "replicate",
        "pairing_seed",
        "checkpoint_sha256",
        "pair_table_sha256",
        "cap",
    }
    required_variability_fields = {
        "task_index",
        "membership_index",
        "evaluation_score",
        "split",
        "n_pairings",
        "cap_mean",
        "cap_sample_variance",
        "cap_sample_std",
        "cap_range",
    }
    if (
        any(not required_value_fields <= set(row) for row in values)
        or any(not required_variability_fields <= set(row) for row in variability)
        or len(values) != int(manifest["method"].get("seed_rows", -1))
        or len(variability) != int(manifest["method"].get("variability_rows", -1))
    ):
        raise ValueError("Random-CAP analysis table schema or cardinality differs.")
    value_keys = {
        (
            row["task_index"],
            row["membership_index"],
            row["evaluation_score"],
            row["split"],
            row["replicate"],
        )
        for row in values
    }
    variability_keys = {
        (row["task_index"], row["membership_index"], row["evaluation_score"], row["split"])
        for row in variability
    }
    if len(value_keys) != len(values) or len(variability_keys) != len(variability):
        raise ValueError("Random-CAP analysis contains duplicate value or variability keys.")
    for row in values:
        _number(row, "cap")
    for row in variability:
        if int(row["n_pairings"]) != 5:
            raise ValueError("Random-CAP variability row does not contain five pairings.")
        for field in ("cap_mean", "cap_sample_variance", "cap_sample_std", "cap_range"):
            _number(row, field)
    associations = tables["structured_random_rank_associations.csv"]
    if expected_mode in {"discovery", "confirmation"}:
        required_association_fields = {
            "model",
            "selection_score",
            "evaluation_score",
            "logical_objective",
            "selection_strategy",
            "split",
            "random_source",
            "n_checkpoints",
            "spearman_rho",
            "pairing_seed_rho_finite_count",
            "pairing_seed_rho_undefined_count",
            "pairing_seed_rho_min",
            "pairing_seed_rho_max",
            "pairing_seed_rho_range",
        }
        if len(associations) != int(manifest["method"].get("rank_association_rows", -1)) or any(
            not required_association_fields <= set(row) for row in associations
        ):
            raise ValueError("Random-CAP rank-association schema or cardinality differs.")
        panels: dict[tuple[str, str, str, str, str, str], set[str]] = {}
        for row in associations:
            key = (
                row["model"],
                row["selection_score"],
                row["evaluation_score"],
                row["logical_objective"],
                row["selection_strategy"],
                row["split"],
            )
            sources = panels.setdefault(key, set())
            if row["random_source"] in sources:
                raise ValueError(f"Duplicate random-CAP association source for {key}.")
            sources.add(row["random_source"])
            if row["spearman_rho"]:
                rho = _number(row, "spearman_rho")
                if not -1.0 <= rho <= 1.0:
                    raise ValueError("Random-CAP Spearman rho is outside [-1, 1].")
            finite_count = int(row["pairing_seed_rho_finite_count"])
            undefined_count = int(row["pairing_seed_rho_undefined_count"])
            if finite_count + undefined_count != 5:
                raise ValueError("Random-CAP pairing-seed rho coverage is not five.")
            if finite_count:
                low = _number(row, "pairing_seed_rho_min")
                high = _number(row, "pairing_seed_rho_max")
                span = _number(row, "pairing_seed_rho_range")
                if low > high or not math.isclose(span, high - low, abs_tol=1e-12):
                    raise ValueError("Random-CAP pairing-seed rho range differs.")
        expected_sources = {"five_pairing_mean", *[f"pairing_replicate_{i}" for i in range(5)]}
        if len(panels) != int(manifest["method"].get("rank_panels", -1)) or any(
            sources != expected_sources for sources in panels.values()
        ):
            raise ValueError("Random-CAP rank-panel coverage differs.")
    downstream = tables["cap_downstream_rank_associations.csv"]
    required_downstream_fields = {
        "model",
        "selection_score",
        "evaluation_score",
        "logical_objective",
        "selection_strategy",
        "split",
        "downstream_endpoint",
        "cap_source",
        "cap_direction",
        "downstream_direction",
        "n_checkpoints",
        "n_anomalies_per_checkpoint",
        "spearman_rho",
        "spearman_defined",
        "constant",
        "ties",
        "insufficient_n",
    }
    method = manifest["method"]
    expected_downstream_sources = {
        "structured_cap",
        "five_pairing_mean",
        *[f"pairing_replicate_{index}" for index in range(5)],
    }
    if any(not required_downstream_fields <= set(row) for row in downstream):
        raise ValueError("Random-CAP downstream-association schema differs.")
    downstream_panels: dict[tuple[str, str, str, str, str, str, str], set[str]] = {}
    defined_rows = 0
    rank_panel_n = {
        (
            row["model"],
            row["selection_score"],
            row["evaluation_score"],
            row["logical_objective"],
            row["selection_strategy"],
            row["split"],
        ): int(row["n_checkpoints"])
        for row in associations
    }

    def boolean(row: dict[str, str], field: str) -> bool:
        if row[field] not in {"True", "False"}:
            raise ValueError(f"Random-CAP downstream {field} flag differs.")
        return row[field] == "True"

    for row in downstream:
        key = (
            row["model"],
            row["selection_score"],
            row["evaluation_score"],
            row["logical_objective"],
            row["selection_strategy"],
            row["split"],
            row["downstream_endpoint"],
        )
        sources = downstream_panels.setdefault(key, set())
        if row["cap_source"] in sources:
            raise ValueError(f"Duplicate random-CAP downstream source for {key}.")
        sources.add(row["cap_source"])
        n_checkpoints = int(row["n_checkpoints"])
        defined = boolean(row, "spearman_defined")
        boolean(row, "constant")
        boolean(row, "ties")
        insufficient = boolean(row, "insufficient_n")
        if (
            row["split"] not in {"validate", "test"}
            or row["downstream_endpoint"] not in {"efficiency_250hz", "auprc"}
            or row["cap_direction"] != "maximize"
            or row["downstream_direction"] != "maximize"
            or int(row["n_anomalies_per_checkpoint"]) != 20
            or n_checkpoints < 1
            or insufficient != (n_checkpoints < 3)
            or rank_panel_n.get(
                (
                    row["model"],
                    row["selection_score"],
                    row["evaluation_score"],
                    row["logical_objective"],
                    row["selection_strategy"],
                    row["split"],
                )
            )
            != n_checkpoints
        ):
            raise ValueError("Random-CAP downstream panel identity or cardinality differs.")
        if defined:
            rho = _number(row, "spearman_rho")
            if not -1.0 <= rho <= 1.0:
                raise ValueError("Random-CAP downstream Spearman rho is outside [-1, 1].")
            defined_rows += 1
        elif row["spearman_rho"]:
            raise ValueError("Undefined random-CAP downstream rho is not empty.")
    base_downstream_panels = {
        (model, selection_score, evaluation_score, objective, strategy, split)
        for model, selection_score, evaluation_score, objective, strategy, split, _ in downstream_panels
    }
    if (
        len(downstream) != int(method.get("downstream_association_rows", -1))
        or len(downstream_panels) != int(method.get("downstream_association_panels", -1))
        or int(method.get("downstream_association_sources", -1)) != 7
        or int(method.get("downstream_association_defined_rows", -1)) != defined_rows
        or int(method.get("downstream_association_undefined_rows", -1))
        != len(downstream) - defined_rows
        or int(method.get("downstream_anomalies", -1)) != 20
        or int(method.get("downstream_endpoints", -1)) != 2
        or int(method.get("downstream_memberships", -1)) != int(method.get("memberships", -2))
        or int(method.get("downstream_rows", -1))
        != int(method.get("downstream_membership_scores", -2)) * 20 * 2
        or method.get("downstream_panels_are_unpooled") is not True
        or any(sources != expected_downstream_sources for sources in downstream_panels.values())
        or base_downstream_panels != set(rank_panel_n)
    ):
        raise ValueError("Random-CAP downstream-association method cardinality differs.")
    return manifest, tables


def _confirmation_provenance(
    analysis_manifest: dict[str, Any],
) -> dict[str, str | None]:
    """Authenticate and expose the endpoint analysis's upstream identities."""
    inputs = analysis_manifest.get("inputs", {})
    matrix_path = Path(inputs.get("matrix", "")).resolve(strict=True)
    if _sha256(matrix_path) != inputs.get("matrix_sha256"):
        raise ValueError("Confirmation endpoint matrix file hash differs.")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    digest = matrix.get("content_sha256")
    if digest != _content_hash(
        {key: value for key, value in matrix.items() if key != "content_sha256"}
    ):
        raise ValueError("Confirmation endpoint matrix content hash differs.")
    endpoint_path = Path(inputs.get("endpoint_receipt", "")).resolve(strict=True)
    endpoint_sha256 = inputs.get("endpoint_receipt_sha256")
    if _sha256(endpoint_path) != endpoint_sha256:
        raise ValueError("Confirmation endpoint receipt file hash differs.")
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    if endpoint.get("matrix_content_sha256") != digest:
        raise ValueError("Confirmation endpoint receipt matrix identity differs.")
    return {
        "confirmation_matrix_content_sha256": str(digest),
        "runner_manifest_content_sha256": endpoint.get("effective_runner_content_sha256"),
        "reserve_extension_content_sha256": endpoint.get("reserve_extension_content_sha256"),
        "checkpoint_selections_sha256": endpoint.get("artifacts", {}).get(
            "checkpoint_selections.csv"
        ),
        "endpoint_receipt_sha256": endpoint_sha256,
    }


def _completion_summary(analysis_manifest: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the endpoint receipt's complete-run marker inventory."""
    receipt_path = Path(analysis_manifest["inputs"]["endpoint_receipt"]).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    markers = receipt.get("source_markers")
    cardinality = receipt.get("cardinality", {})
    sealed_runs = int(cardinality.get("sealed_runs", -1))
    if (
        not isinstance(markers, list)
        or sealed_runs <= 0
        or len(markers) != sealed_runs
        or receipt.get("source_markers_content_sha256") != _content_hash(markers)
    ):
        raise ValueError("Endpoint receipt lacks an authenticated complete-run marker inventory.")
    run_ids: set[str] = set()
    for row in markers:
        marker_path = Path(str(row.get("marker", ""))).resolve(strict=True)
        metric_path = Path(str(row.get("optimized_metric_artifact", ""))).resolve(strict=True)
        if (
            row.get("marker_sha256") != _sha256(marker_path)
            or row.get("optimized_metric_artifact_sha256") != _sha256(metric_path)
            or not str(row.get("run_id", ""))
            or str(row["run_id"]) in run_ids
        ):
            raise ValueError("Endpoint receipt complete-run marker identity differs.")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("run_name") != row["run_id"]
            or marker.get("optimized_metric_artifact") != str(metric_path)
            or marker.get("optimized_metric_artifact_sha256")
            != row["optimized_metric_artifact_sha256"]
        ):
            raise ValueError("Endpoint receipt marker contents differ from its inventory.")
        run_ids.add(str(row["run_id"]))
    return {
        "sealed_runs": sealed_runs,
        "checkpoint_selections": int(cardinality.get("checkpoint_selections", -1)),
        "primary_endpoint_rows": int(cardinality.get("primary_endpoint_rows", -1)),
        "oas_endpoint_rows": int(cardinality.get("oas_endpoint_rows", -1)),
        "marker_inventory_sha256": receipt["source_markers_content_sha256"],
        "endpoint_receipt_sha256": _sha256(receipt_path),
    }


def _campaign_attempt_summary(analysis_manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarize scheduler attempts while requiring a quiescent final campaign ledger."""
    receipt_path = Path(analysis_manifest["inputs"]["endpoint_receipt"]).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    campaign_root = receipt.get("campaign_root")
    if not campaign_root:
        return {"available": False}
    ledger_path = Path(campaign_root).resolve(strict=True) / "state/job_history.csv"
    rows = _read_csv(ledger_path)
    retrains = [row for row in rows if row.get("kind") in {"retrain", "retrain_superseded"}]
    active = [
        row
        for row in retrains
        if row.get("state") in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
    ]
    if active:
        raise ValueError("Final report requires a quiescent retrain ledger.")
    counts = {
        state: sum(row.get("state") == state for row in retrains)
        for state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")
    }
    return {
        "available": True,
        "ledger": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "attempt_rows": len(retrains),
        "states": counts,
        "terminal_noncompleted_attempts": len(retrains) - counts["COMPLETED"],
        "missing_final_endpoints": 0,
    }


def _load_confirmation_context(
    context_dir: Path, analysis_manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Authenticate the dedicated MLflow/discovery context package."""
    context_dir = context_dir.resolve(strict=True)
    path = context_dir / "confirmation_context.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    digest = receipt.pop("content_sha256", None)
    if digest != _content_hash(receipt):
        raise ValueError("Confirmation context receipt content hash differs.")
    receipt["content_sha256"] = digest
    inputs = receipt.get("inputs", {})
    if (
        receipt.get("kind") != "physics_background_pairing_confirmation_context"
        or inputs.get("confirmation_analysis_content_sha256")
        != analysis_manifest["content_sha256"]
        or inputs.get("endpoint_receipt_sha256")
        != analysis_manifest["inputs"]["endpoint_receipt_sha256"]
    ):
        raise ValueError("Confirmation context is not bound to the final endpoint analysis.")
    tables = {}
    for name, expected_hash in receipt.get("artifacts", {}).items():
        artifact = (context_dir / name).resolve(strict=True)
        if not artifact.is_relative_to(context_dir) or _sha256(artifact) != expected_hash:
            raise ValueError(f"Confirmation context artifact changed: {artifact}")
        tables[name] = _read_csv(artifact)
    required = {
        "persistence.csv",
        "hypotheses.csv",
        "training_curves.csv",
        "training_curve_summary.csv",
    }
    if set(tables) != required or any(
        len(tables[name]) != int(receipt["cardinality"].get(name.removesuffix(".csv"), -1))
        for name in required
    ):
        raise ValueError("Confirmation context table coverage differs.")
    if len(tables["persistence.csv"]) != 84 or len(tables["hypotheses.csv"]) != 72:
        raise ValueError("Confirmation 50-to-200 persistence cardinality differs.")
    return receipt, tables


def _number(row: dict[str, str], field: str) -> float:
    """Read one finite numeric result."""
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite report value for {field}.")
    return value


def _fmt(value: float, *, percent: bool = False, bold: bool = False) -> str:
    """Format an endpoint value and optional row winner."""
    text = f"{100.0 * value:.2f}%" if percent else f"{value:.3f}"
    return f"**{text}**" if bold else text


def _downstream_random_comparisons(
    tables: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Compare structured CAP with five-random-mean downstream rank alignment."""
    grouped: dict[tuple[str, str, str, str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in tables["cap_downstream_rank_associations.csv"]:
        key = (
            row["model"],
            row["selection_score"],
            row["evaluation_score"],
            row["logical_objective"],
            row["selection_strategy"],
            row["split"],
            row["downstream_endpoint"],
        )
        grouped.setdefault(key, {})[row["cap_source"]] = row
    output = []
    for key, sources in sorted(grouped.items()):
        structured = sources["structured_cap"]
        random_mean = sources["five_pairing_mean"]
        structured_rho = (
            _number(structured, "spearman_rho")
            if structured["spearman_defined"] == "True"
            else None
        )
        random_rho = (
            _number(random_mean, "spearman_rho")
            if random_mean["spearman_defined"] == "True"
            else None
        )
        if structured_rho is None or random_rho is None:
            stronger = "undefined"
            delta = None
        else:
            delta = structured_rho - random_rho
            stronger = "structured" if delta > 0 else "random" if delta < 0 else "tie"
        output.append(
            {
                "model": key[0],
                "selection_score": key[1],
                "score": key[2],
                "logical_objective": key[3],
                "pairing": key[4],
                "split": key[5],
                "endpoint": key[6],
                "n_checkpoints": int(structured["n_checkpoints"]),
                "structured_rho": structured_rho,
                "random_mean_rho": random_rho,
                "delta_rho": delta,
                "stronger": stronger,
            }
        )
    return output


def _append_downstream_random_summary(
    lines: list[str], mode: str, rows: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    """Render every unpooled downstream comparison and return endpoint counts."""
    counts: dict[str, dict[str, int]] = {}
    for endpoint in ("efficiency_250hz", "auprc"):
        endpoint_rows = [row for row in rows if row["endpoint"] == endpoint]
        counts[endpoint] = {
            verdict: sum(row["stronger"] == verdict for row in endpoint_rows)
            for verdict in ("structured", "random", "tie", "undefined")
        }
    lines.extend(
        [
            "",
            f"### {mode.title()} CAP-to-downstream rank alignment",
            "",
            (
                f"Higher rho is better because CAP and both downstream endpoints are maximized. "
                f"For efficiency, structured/random/tie/undefined panels are "
                f"{counts['efficiency_250hz']['structured']}/"
                f"{counts['efficiency_250hz']['random']}/"
                f"{counts['efficiency_250hz']['tie']}/"
                f"{counts['efficiency_250hz']['undefined']}; for AUPRC they are "
                f"{counts['auprc']['structured']}/"
                f"{counts['auprc']['random']}/"
                f"{counts['auprc']['tie']}/"
                f"{counts['auprc']['undefined']}."
            ),
            "",
            "| Model | Score | Pairing | Split | Endpoint | N | Structured rho | Five-random-mean rho | Stronger |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        structured = (
            "undefined" if row["structured_rho"] is None else f"{row['structured_rho']:.3f}"
        )
        random_mean = (
            "undefined" if row["random_mean_rho"] is None else f"{row['random_mean_rho']:.3f}"
        )
        lines.append(
            f"| {row['model'].upper()} | {row['score']} | {row['pairing']} | "
            f"{row['split']} | {row['endpoint']} | {row['n_checkpoints']} | "
            f"{structured} | {random_mean} | {row['stronger']} |"
        )
    return counts


def _cell_table(rows: list[dict[str, str]]) -> tuple[list[str], dict[tuple[str, str, str], float]]:
    """Render all 42 cells with independent efficiency/AUPRC winners."""
    index = {
        (row["model"], row["selector"], row["pairing"], row["endpoint"]): _number(row, "mean")
        for row in rows
    }
    expected = {
        (model, selector, pairing, endpoint)
        for model in MODELS
        for selector, pairing, _ in SELECTORS
        for endpoint in ("efficiency_250hz", "auprc")
    }
    if set(index) != expected:
        raise ValueError("Across-seed summaries do not contain the exact 42-cell matrix.")
    lines = [
        "| Model | " + " | ".join(label for _, _, label in SELECTORS) + " |",
        "|---|" + "---:|" * len(SELECTORS),
    ]
    compact = {}
    for model in MODELS:
        efficiencies = [
            index[(model, selector, pairing, "efficiency_250hz")]
            for selector, pairing, _ in SELECTORS
        ]
        auprcs = [index[(model, selector, pairing, "auprc")] for selector, pairing, _ in SELECTORS]
        best_efficiency, best_auprc = max(efficiencies), max(auprcs)
        cells = []
        for (selector, pairing, _), efficiency, auprc in zip(SELECTORS, efficiencies, auprcs):
            cells.append(
                f"{_fmt(efficiency, percent=True, bold=efficiency == best_efficiency)} / "
                f"{_fmt(auprc, bold=auprc == best_auprc)}"
            )
            compact[(model, selector, pairing, "efficiency_250hz")] = efficiency
            compact[(model, selector, pairing, "auprc")] = auprc
        lines.append(f"| {model.upper()} | " + " | ".join(cells) + " |")
    return lines, compact


def _validated_reserve_decisions(
    rows: list[dict[str, str]], seeds_by_model: dict[str, list[int]]
) -> dict[str, dict[str, str]]:
    """Require one terminal reserve-seed decision for every model."""
    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row.get("model", "")
        if model not in MODELS or model in decisions:
            raise ValueError("Reserve-seed decisions must name each model exactly once.")
        decision = row.get("decision")
        if decision not in {"no_extension", "extension_complete"}:
            raise RuntimeError(
                f"Final report blocked: reserve-seed extension is incomplete for {model}: "
                f"{decision!r}."
            )
        expected = 3 if decision == "no_extension" else 5
        observed = int(row.get("observed_seed_count", expected))
        if observed != expected or observed != len(seeds_by_model.get(model, [])):
            raise ValueError(
                f"Reserve decision/count mismatch for {model}: {decision} with {observed} seeds."
            )
        decisions[model] = row
    if set(decisions) != set(MODELS):
        raise ValueError("Reserve-seed decisions must cover all six models.")
    return decisions


def _contrast_counts(rows: list[dict[str, str]], endpoint: str) -> tuple[int, int, int]:
    """Return directional-positive, adjusted-improvement, and total counts."""
    selected = [row for row in rows if row["endpoint"] == endpoint]
    return (
        sum(_number(row, "mean_delta") > 0.0 for row in selected),
        sum(row["classification"] == "meaningful_improvement" for row in selected),
        len(selected),
    )


def _paper_rows(
    repository: Path, cell_values: dict[tuple[str, str, str, str], float]
) -> tuple[list[dict[str, Any]], str]:
    """Build semantically matched paper comparisons from confirmation means."""
    reference, reference_hash = load_authenticated_paper_reference(repository)
    winners = []
    for model in MODELS:
        for metric, pairing in (
            ("cap", "cdf"),
            ("drift", "not_applicable"),
            ("wasserstein", "not_applicable"),
        ):
            winners.append(
                {
                    "model": model,
                    "score": "native",
                    "metric": metric,
                    "strategy": "cdf" if metric == "cap" else "",
                    "mean_efficiency": cell_values[(model, metric, pairing, "efficiency_250hz")],
                }
            )
    return build_paper_comparison(reference, winners), reference_hash


def build_confirmation_report(
    analysis_dir: Path,
    repository: Path,
    *,
    discovery_random_analysis: Path | None = None,
    confirmation_random_analysis: Path | None = None,
    confirmation_context: Path | None = None,
    report_mode: str = "final",
) -> tuple[str, dict[str, Any]]:
    """Build an information-dense report and machine-readable result summary."""
    if report_mode not in {"final", "provisional"}:
        raise ValueError(f"Unknown confirmation report mode: {report_mode!r}.")
    if report_mode == "final" and (
        discovery_random_analysis is None
        or confirmation_random_analysis is None
        or confirmation_context is None
    ):
        raise RuntimeError(
            "Final report blocked: authenticated discovery and confirmation random-CAP "
            "analyses and the authenticated 50-to-200/training-curve context are mandatory. "
            "Use report_mode='provisional' for an explicitly "
            "nonfinal report."
        )
    manifest, tables = _load_analysis(analysis_dir.resolve(strict=True))
    completion = _completion_summary(manifest)
    attempt_summary = _campaign_attempt_summary(manifest)
    context_receipt = None
    context_tables = None
    if confirmation_context is not None:
        context_receipt, context_tables = _load_confirmation_context(
            confirmation_context, manifest
        )
    seeds_by_model = manifest["method"].get("training_seeds_by_model") or {
        model: manifest["method"]["seeds"] for model in MODELS
    }
    reserve_decisions = _validated_reserve_decisions(
        tables["reserve_seed_decisions.csv"], seeds_by_model
    )
    cell_lines, cell_values = _cell_table(tables["endpoints_across_seeds.csv"])
    paper_rows, paper_reference_hash = _paper_rows(repository.resolve(strict=True), cell_values)
    comparisons = tables["comparison_summary.csv"]
    controlled = [
        row for row in comparisons if row["scope"] == "model" and row["family"] == "controlled"
    ]
    pairing = [
        row for row in comparisons if row["scope"] == "model" and row["family"] == "pairing"
    ]
    macro_controlled = [
        row for row in comparisons if row["scope"] == "macro" and row["family"] == "controlled"
    ]
    macro_pairing = [
        row for row in comparisons if row["scope"] == "macro" and row["family"] == "pairing"
    ]
    if len(controlled) != 6 * 2 * 2 or len(pairing) != 6 * 4 * 2:
        raise ValueError("Model-level comparison summary cardinality differs.")
    if len(macro_controlled) != 2 * 2 or len(macro_pairing) != 4 * 2:
        raise ValueError("Macro comparison summary cardinality differs.")

    comparable_paper = [row for row in paper_rows if row["comparison_status"] == "comparable"]
    paper_better = sum(float(row["delta_percentage_points"]) > 0 for row in comparable_paper)
    cap_eff = _contrast_counts(controlled, "efficiency_250hz")
    cap_auprc = _contrast_counts(controlled, "auprc")
    pairing_eff = _contrast_counts(pairing, "efficiency_250hz")
    pairing_auprc = _contrast_counts(pairing, "auprc")

    lines = [
        (
            "# Physics background-pairing confirmation results"
            if report_mode == "final"
            else "# PROVISIONAL physics background-pairing confirmation results"
        ),
        "",
        "## Direct answers",
        "",
        f"Completion: **{completion['sealed_runs']}/{completion['sealed_runs']} authenticated retraining runs sealed**; {completion['checkpoint_selections']} logical checkpoint selections and {completion['primary_endpoint_rows']} native endpoint rows were verified.",
        "",
        f"1. **Background0/background1 system versus the paper:** {paper_better}/{len(comparable_paper)} method-matched efficiency means are directionally higher. This is a cross-system benchmark, not a causal isolation of the changed validation datasets; the paper has no matching AUPRC.",
        f"2. **Does pairing improve CAP?** Efficiency: {pairing_eff[0]}/{pairing_eff[2]} directional gains and {pairing_eff[1]} multiplicity-adjusted meaningful improvements. AUPRC: {pairing_auprc[0]}/{pairing_auprc[2]} directional gains and {pairing_auprc[1]} adjusted meaningful improvements.",
        f"3. **Does CDF CAP beat W1/drift?** Efficiency: {cap_eff[0]}/{cap_eff[2]} directional gains and {cap_eff[1]} adjusted meaningful improvements. AUPRC: {cap_auprc[0]}/{cap_auprc[2]} directional gains and {cap_auprc[1]} adjusted meaningful improvements. W1 and drift remain separate contrasts below.",
        "",
        "All values below are means over the fixed seeds and the same 20 downstream anomalies. Efficiency is at 250 Hz. Each cell is `efficiency / AUPRC`; bold marks the independent winner within that model and endpoint.",
        "",
        "## Complete confirmation matrix",
        "",
        *cell_lines,
        "",
        "## Robustness of the mean-winning cells",
        "",
        "The table reports anomaly-level robustness after averaging seeds first. Worst is the lowest of the 20 anomaly means; bottom-quartile CVaR is the mean of the lowest five.",
        "",
        "| Model | Endpoint | Mean-winning method | Mean | Median | Worst | Bottom-five CVaR | Seed-mean SD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    across_rows = tables["endpoints_across_seeds.csv"]
    for model in MODELS:
        for endpoint in ("efficiency_250hz", "auprc"):
            candidates = [
                row for row in across_rows if row["model"] == model and row["endpoint"] == endpoint
            ]
            winner = max(candidates, key=lambda row: _number(row, "mean"))
            method = winner["pairing"] if winner["selector"] == "cap" else winner["selector"]
            scale = 100.0 if endpoint == "efficiency_250hz" else 1.0
            lines.append(
                f"| {model.upper()} | {endpoint} | {method} | "
                f"{scale * _number(winner, 'mean'):.3f} | "
                f"{scale * _number(winner, 'median'):.3f} | "
                f"{scale * _number(winner, 'worst'):.3f} | "
                f"{scale * _number(winner, 'bottom_quartile_cvar'):.3f} | "
                f"{scale * _number(winner, 'seed_mean_std'):.3f} |"
            )
    lines.extend(
        [
            "",
            "## CDF CAP versus W1 and drift",
            "",
            "| Model | Endpoint | Contrast | Mean delta (pp or AUPRC) | Adjusted CI | Classification |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in controlled:
        scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
        lines.append(
            f"| {row['model'].upper()} | {row['endpoint']} | {row['contrast']} | "
            f"{scale * _number(row, 'mean_delta'):+.3f} | "
            f"[{scale * _number(row, 'adjusted_ci_low'):+.3f}, {scale * _number(row, 'adjusted_ci_high'):+.3f}] | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Learned pairing versus CDF CAP",
            "",
            "| Model | Pairing contrast | Endpoint | Mean delta (pp or AUPRC) | Adjusted CI | Classification |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in pairing:
        scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
        lines.append(
            f"| {row['model'].upper()} | {row['contrast']} | {row['endpoint']} | "
            f"{scale * _number(row, 'mean_delta'):+.3f} | "
            f"[{scale * _number(row, 'adjusted_ci_low'):+.3f}, {scale * _number(row, 'adjusted_ci_high'):+.3f}] | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Equal-model macro contrasts",
            "",
            "| Family | Contrast | Endpoint | Mean delta (pp or AUPRC) | Adjusted CI | Classification |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in (*macro_controlled, *macro_pairing):
        scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
        lines.append(
            f"| {row['family']} | {row['contrast']} | {row['endpoint']} | "
            f"{scale * _number(row, 'mean_delta'):+.3f} | "
            f"[{scale * _number(row, 'adjusted_ci_low'):+.3f}, "
            f"{scale * _number(row, 'adjusted_ci_high'):+.3f}] | "
            f"{row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Matched comparison with the paper",
            "",
            "| Model | Method | Paper efficiency | Confirmation efficiency | Delta (pp) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparable_paper:
        lines.append(
            f"| {row['model'].upper()} | {row['paper_method']} | "
            f"{float(row['paper_mean_efficiency_percent']):.2f}% | "
            f"{float(row['campaign_mean_efficiency_percent']):.2f}% | "
            f"{float(row['delta_percentage_points']):+.2f} |"
        )
    lines.extend(
        [
            "",
            "These are method-matched native-score cross-system comparisons: CDF CAP to paper CAP, forward drift to paper Stable, and W1 to paper W1. They test the full changed validation/selection system, not dataset choice in isolation.",
            "",
            "## AE/VAE masked-OAS diagnostic",
            "",
            "OAS is evaluated on the same native-selected checkpoints; it did not select configurations or checkpoints.",
            "",
            "| Model | Selector/pairing | Endpoint | OAS − native | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in tables["oas_vs_native_summary.csv"]:
        scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
        label = row["pairing"] if row["selector"] == "cap" else row["selector"]
        lines.append(
            f"| {row['model'].upper()} | {label} | {row['endpoint']} | "
            f"{scale * _number(row, 'mean_delta'):+.3f} | "
            f"[{scale * _number(row, 'ci95_low'):+.3f}, {scale * _number(row, 'ci95_high'):+.3f}] |"
        )
    oas_rows = tables["oas_vs_native_summary.csv"]
    oas_eff = [row for row in oas_rows if row["endpoint"] == "efficiency_250hz"]
    oas_auprc = [row for row in oas_rows if row["endpoint"] == "auprc"]
    lines.extend(
        [
            "",
            f"Masked OAS is directionally higher than native scoring in "
            f"{sum(_number(row, 'mean_delta') > 0 for row in oas_eff)}/{len(oas_eff)} "
            f"efficiency cells and {sum(_number(row, 'mean_delta') > 0 for row in oas_auprc)}/"
            f"{len(oas_auprc)} AUPRC cells. Interpret the confidence intervals above; "
            "these are same-checkpoint score effects, not independently optimized selectors.",
            "",
            "## Reserve-seed decisions",
            "",
            "| Model | Observed seeds | Decision |",
            "|---|---:|---|",
            *[
                f"| {model.upper()} | "
                f"{reserve_decisions[model].get('observed_seed_count', '3')} | "
                f"{reserve_decisions[model]['decision']} |"
                for model in MODELS
            ],
        ]
    )

    lines.extend(["", "## 50-to-200-epoch persistence and training curves", ""])
    if context_tables is None:
        lines.append(
            "Not supplied in this provisional report. No convergence or persistence claim is made."
        )
    else:
        hypotheses = context_tables["hypotheses.csv"]
        reversed_rows = [row for row in hypotheses if row["direction_outcome"] == "reversed"]
        survived_rows = [
            row for row in hypotheses if row["direction_outcome"] == "survived_directionally"
        ]
        curve_rows = context_tables["training_curve_summary.csv"]
        favorable_curves = sum(
            _number(row, "favorable_change_50_to_200") > 0 for row in curve_rows
        )
        lines.extend(
            [
                f"Across the 72 predeclared selector/pairing endpoint contrasts, {len(survived_rows)} retained their discovery direction and {len(reversed_rows)} reversed between the selected 50-epoch checkpoint and the multi-seed 200-epoch mean. This is persistence evidence, not a fresh untouched selection test.",
                f"The authenticated validation-objective histories show favorable mean movement from epoch 50 to 200 in {favorable_curves}/{len(curve_rows)} model/selector panels.",
                "",
                "| Model | Objective | Direction | Runs | Epoch 50 mean | Epoch 200 mean | Favorable change | Epoch 160→200 raw change |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in curve_rows:
            lines.append(
                f"| {row['model'].upper()} | {row['logical_objective']} | {row['direction']} | "
                f"{row['runs']} | {_number(row, 'mean_epoch_50'):.5g} | "
                f"{_number(row, 'mean_epoch_200'):.5g} | "
                f"{_number(row, 'favorable_change_50_to_200'):+.5g} | "
                f"{_number(row, 'mean_last_40_change'):+.5g} |"
            )
        lines.extend(
            [
                "",
                "Discovery-direction reversals (all non-reversed hypotheses remain in the packaged CSV):",
                "",
                "| Model | Endpoint | Contrast | 50-epoch delta | 200-epoch delta |",
                "|---|---|---|---:|---:|",
            ]
        )
        if reversed_rows:
            for row in reversed_rows:
                scale = 100.0 if row["endpoint"] == "efficiency_250hz" else 1.0
                lines.append(
                    f"| {row['model'].upper()} | {row['endpoint']} | {row['contrast']} | "
                    f"{scale * _number(row, 'delta_50_epoch'):+.3f} | "
                    f"{scale * _number(row, 'delta_200_epoch'):+.3f} |"
                )
        else:
            lines.append("| — | — | No directional reversals | — | — |")

    random_status = "not_supplied"
    random_receipts: dict[str, str] = {}
    random_downstream_summary: dict[str, dict[str, dict[str, int]]] = {}
    lines.extend(["", "## Random-pairing negative control", ""])
    if discovery_random_analysis is not None:
        discovery_receipt, discovery_tables = _load_random_analysis(
            discovery_random_analysis, "discovery"
        )
        random_receipts["discovery"] = discovery_receipt["content_sha256"]
        mean_associations = [
            row
            for row in discovery_tables["structured_random_rank_associations.csv"]
            if row["random_source"] == "five_pairing_mean"
        ]
        association_index: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
        for row in mean_associations:
            key = (
                row["model"],
                row["selection_score"],
                row["evaluation_score"],
                row["logical_objective"],
            )
            if row["split"] in association_index.setdefault(key, {}):
                raise ValueError(f"Duplicate random-CAP rank association: {key}.")
            association_index[key][row["split"]] = row
        if any(
            set(split_rows) != {"validate", "test"} for split_rows in association_index.values()
        ):
            raise ValueError("Random-CAP rank association lacks one analysis split.")
        lines.extend(
            [
                "Structured-CAP versus mean-random-CAP checkpoint-rank association on the frozen discovery universe:",
                "",
                "| Model | Selection score | Evaluation score | Structured objective | N | Validation rho | Validation seed-rho range | Test rho | Test seed-rho range |",
                "|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for (model, selection_score, evaluation_score, objective), split_rows in sorted(
            association_index.items()
        ):
            validate, test = split_rows["validate"], split_rows["test"]
            if validate["n_checkpoints"] != test["n_checkpoints"]:
                raise ValueError("Random-CAP rank-association split cardinalities differ.")

            def seed_range(row: dict[str, str]) -> str:
                if int(row["pairing_seed_rho_finite_count"]) == 0:
                    return f"undefined ({row['pairing_seed_rho_undefined_count']}/5)"
                return (
                    f"[{float(row['pairing_seed_rho_min']):.3f}, "
                    f"{float(row['pairing_seed_rho_max']):.3f}] "
                    f"({row['pairing_seed_rho_undefined_count']}/5 undefined)"
                )

            lines.append(
                f"| {model.upper()} | {selection_score} | {evaluation_score} | {objective} | "
                f"{validate['n_checkpoints']} | "
                f"{validate['spearman_rho'] or 'undefined'} | "
                f"{seed_range(validate)} | {test['spearman_rho'] or 'undefined'} | "
                f"{seed_range(test)} |"
            )
        discovery_downstream = _downstream_random_comparisons(discovery_tables)
        random_downstream_summary["discovery"] = _append_downstream_random_summary(
            lines, "discovery", discovery_downstream
        )
        random_status = (
            f"discovery authenticated ({discovery_receipt['method']['seed_rows']} seed rows)"
        )
    if confirmation_random_analysis is not None:
        confirmation_receipt, confirmation_tables = _load_random_analysis(
            confirmation_random_analysis,
            "confirmation",
            confirmation_provenance=_confirmation_provenance(manifest),
        )
        random_receipts["confirmation"] = confirmation_receipt["content_sha256"]
        confirmation_mean_associations = [
            row
            for row in confirmation_tables["structured_random_rank_associations.csv"]
            if row["random_source"] == "five_pairing_mean"
        ]
        variability_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in confirmation_tables["random_cap_variability.csv"]:
            variability_groups.setdefault(
                (row["model"], row["evaluation_score"], row["split"]), []
            ).append(row)
        lines.extend(
            [
                "",
                "Five-random-bijection variability on sealed confirmation checkpoints:",
                "",
                "| Model | Score | Split | Checkpoint memberships | Median random-CAP SD | Maximum range |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for (model, score, split), rows in sorted(variability_groups.items()):
            standard_deviations = [_number(row, "cap_sample_std") for row in rows]
            ranges = [_number(row, "cap_range") for row in rows]
            lines.append(
                f"| {model.upper()} | {score} | {split} | {len(rows)} | "
                f"{statistics.median(standard_deviations):.4g} | {max(ranges):.4g} |"
            )
        lines.extend(
            [
                "",
                "Structured-versus-random CAP rank association on the sealed confirmation checkpoints (unpooled by model, selector, split, and score):",
                "",
                "| Model | Selection score | Evaluation score | Objective | Split | N | Five-pairing-mean rho |",
                "|---|---|---|---|---|---:|---:|",
            ]
        )
        for row in sorted(
            confirmation_mean_associations,
            key=lambda item: (
                item["model"],
                item["selection_score"],
                item["evaluation_score"],
                item["logical_objective"],
                item["split"],
            ),
        ):
            rho = row["spearman_rho"] or "undefined"
            lines.append(
                f"| {row['model'].upper()} | {row['selection_score']} | "
                f"{row['evaluation_score']} | {row['logical_objective']} | {row['split']} | "
                f"{row['n_checkpoints']} | {rho} |"
            )
        confirmation_downstream = _downstream_random_comparisons(confirmation_tables)
        random_downstream_summary["confirmation"] = _append_downstream_random_summary(
            lines, "confirmation", confirmation_downstream
        )
        random_status = (
            f"{random_status}; confirmation authenticated "
            f"({confirmation_receipt['method']['seed_rows']} seed rows)"
        )
    lines.extend(
        [
            "",
            f"Status: {random_status}. Random CAP is diagnostic only and never selected a hyperparameter or checkpoint.",
            "The downstream-rank comparison is post-hoc on the fixed structured-selected checkpoint universes. It does not estimate the efficiency or AUPRC of a random-CAP-selected model and does not turn random CAP into a selector.",
            "",
            "## Provenance and limitations",
            "",
            f"- Confirmation analysis SHA-256: `{manifest['content_sha256']}`.",
            f"- Paper-reference SHA-256: `{paper_reference_hash}`.",
            f"- Seeds: {', '.join(map(str, manifest['method']['seeds']))}; anomalies: {manifest['method']['anomalies']}.",
            (
                f"- Scheduler attempt ledger: {attempt_summary['attempt_rows']} terminal retrain "
                f"attempt rows, including {attempt_summary['terminal_noncompleted_attempts']} "
                "non-completed scheduler attempts; all required final endpoints are "
                "present."
                if attempt_summary["available"]
                else "- Scheduler-attempt ledger was unavailable in this portable fixture."
            ),
            "- Downstream anomalies were already inspected during discovery. These results establish 200-epoch and seed stability, not a fully untouched external holdout.",
            "- Paper AUPRC is unavailable and is not inferred.",
            "",
        ]
    )
    summary = {
        "schema_version": 1,
        "report_mode": report_mode,
        "scientifically_final": report_mode == "final",
        "analysis_content_sha256": manifest["content_sha256"],
        "completion": completion,
        "scheduler_attempts": attempt_summary,
        "paper_reference_sha256": paper_reference_hash,
        "paper_comparison": paper_rows,
        "paper_matched_positive": paper_better,
        "paper_matched_total": len(comparable_paper),
        "cdf_cap_efficiency_directional_positive": cap_eff[0],
        "cdf_cap_efficiency_meaningful": cap_eff[1],
        "cdf_cap_efficiency_total": cap_eff[2],
        "cdf_cap_auprc_directional_positive": cap_auprc[0],
        "cdf_cap_auprc_meaningful": cap_auprc[1],
        "cdf_cap_auprc_total": cap_auprc[2],
        "learned_pairing_efficiency_directional_positive": pairing_eff[0],
        "learned_pairing_efficiency_meaningful": pairing_eff[1],
        "learned_pairing_efficiency_total": pairing_eff[2],
        "learned_pairing_auprc_directional_positive": pairing_auprc[0],
        "learned_pairing_auprc_meaningful": pairing_auprc[1],
        "learned_pairing_auprc_total": pairing_auprc[2],
        "reserve_seed_decisions": {
            model: reserve_decisions[model]["decision"] for model in MODELS
        },
        "random_cap_status": random_status,
        "random_cap_analysis_receipts": random_receipts,
        "random_cap_downstream_rank_comparison": random_downstream_summary,
        "confirmation_context_receipt": (
            None if context_receipt is None else context_receipt["content_sha256"]
        ),
    }
    summary["content_sha256"] = _content_hash(summary)
    return "\n".join(lines), summary


def write_confirmation_report(
    analysis_dir: Path,
    repository: Path,
    output_dir: Path,
    *,
    discovery_random_analysis: Path | None = None,
    confirmation_random_analysis: Path | None = None,
    confirmation_context: Path | None = None,
    report_mode: str = "final",
    analysis_stage: str = "final",
    generator: dict[str, Any],
) -> dict[str, Any]:
    """Persist an immutable compact report bundle with authenticated evidence."""
    if analysis_stage not in {"primary", "final"}:
        raise ValueError(f"Unknown confirmation analysis stage: {analysis_stage!r}.")
    if report_mode == "final" and analysis_stage != "final":
        raise RuntimeError("A final report must consume the final endpoint/analysis namespace.")
    report, summary = build_confirmation_report(
        analysis_dir,
        repository,
        discovery_random_analysis=discovery_random_analysis,
        confirmation_random_analysis=confirmation_random_analysis,
        confirmation_context=confirmation_context,
        report_mode=report_mode,
    )
    summary.pop("content_sha256", None)
    summary["generator"] = generator
    summary["content_sha256"] = _content_hash(summary)
    analysis_manifest, _ = _load_analysis(analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "REPORT.md": report + "\n",
        "report_summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
    }
    for name, content in artifacts.items():
        path = output_dir / name
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary = Path(temporary_name)
            if path.exists():
                if path.read_bytes() != temporary.read_bytes():
                    raise FileExistsError(f"Refusing to replace different final report: {path}")
            else:
                os.replace(temporary, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    evidence_sources: dict[str, Path] = {}
    for name in (
        "endpoints_long.csv",
        "endpoints_summary.csv",
        "endpoints_across_seeds.csv",
        "selector_comparison_paired.csv",
        "pairing_vs_cdf_paired.csv",
        "comparison_summary.csv",
        "oas_endpoints_long.csv",
        "oas_vs_native_paired.csv",
        "oas_vs_native_summary.csv",
        "reserve_seed_decisions.csv",
        "confirmation_endpoint_analysis.json",
    ):
        evidence_sources[f"confirmation_{name}"] = analysis_dir / name
    analysis_inputs = analysis_manifest.get("inputs", {})
    evidence_sources["confirmation_matrix.json"] = Path(analysis_inputs["matrix"])
    endpoint_receipt_path = Path(analysis_inputs["endpoint_receipt"])
    evidence_sources["confirmation_endpoint_extraction.json"] = endpoint_receipt_path
    endpoint_receipt = json.loads(endpoint_receipt_path.read_text(encoding="utf-8"))
    checkpoint_selections = endpoint_receipt_path.parent / "checkpoint_selections.csv"
    if not checkpoint_selections.is_file() or _sha256(
        checkpoint_selections
    ) != endpoint_receipt.get("artifacts", {}).get("checkpoint_selections.csv"):
        raise ValueError("Confirmation checkpoint-selection evidence differs.")
    evidence_sources["confirmation_checkpoint_selections.csv"] = checkpoint_selections
    campaign_root_value = endpoint_receipt.get("campaign_root")
    if campaign_root_value:
        campaign_root = Path(campaign_root_value).resolve(strict=True)
        campaign_evidence = {
            "confirmation_design.json": campaign_root / "design.json",
            "confirmation_runner_manifest.json": campaign_root
            / "selection/pareto_retrain_manifest.json",
            "confirmation_job_history.csv": campaign_root / "state/job_history.csv",
        }
        failures = campaign_root / "state/failures.jsonl"
        if failures.is_file():
            campaign_evidence["confirmation_failures.jsonl"] = failures
        evidence_sources.update(campaign_evidence)
    for prefix, source in (
        ("discovery_random", discovery_random_analysis),
        ("confirmation_random", confirmation_random_analysis),
    ):
        if source is None:
            continue
        for name in (
            "random_cap_analysis.json",
            "random_cap_values.csv",
            "random_cap_variability.csv",
            "structured_cap_values.csv",
            "structured_random_rank_associations.csv",
            "cap_downstream_rank_associations.csv",
        ):
            evidence_sources[f"{prefix}_{name}"] = source / name
    if confirmation_context is not None:
        for name in (
            "confirmation_context.json",
            "persistence.csv",
            "hypotheses.csv",
            "training_curves.csv",
            "training_curve_summary.csv",
        ):
            evidence_sources[f"context_{name}"] = confirmation_context / name
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, source in evidence_sources.items():
        source = source.resolve(strict=True)
        target = evidence_dir / name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise FileExistsError(f"Refusing to replace different report evidence: {target}")
        else:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=evidence_dir)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    paper_comparison_path = evidence_dir / "paper_comparison.csv"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{paper_comparison_path.name}.", dir=evidence_dir
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            paper_rows = summary["paper_comparison"]
            writer = csv.DictWriter(handle, fieldnames=list(paper_rows[0]))
            writer.writeheader()
            writer.writerows(paper_rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        if paper_comparison_path.exists():
            if paper_comparison_path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(
                    f"Refusing to replace different report evidence: {paper_comparison_path}"
                )
        else:
            os.replace(temporary, paper_comparison_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _atomic_report_text(
        figures_dir / "mean_endpoint_matrix.svg",
        _endpoint_matrix_svg(_read_csv(analysis_dir / "endpoints_across_seeds.csv")),
    )

    prior_report = repository / "REPORT.md"
    prior_target = output_dir / "prior_repository_REPORT.md"
    if (
        prior_report.is_file()
        and prior_report.read_bytes() != (output_dir / "REPORT.md").read_bytes()
    ):
        if prior_target.exists() and prior_target.read_bytes() != prior_report.read_bytes():
            raise FileExistsError(f"Refusing to replace a different prior report: {prior_target}")
        if not prior_target.exists():
            shutil.copyfile(prior_report, prior_target)

    provenance = {
        "schema_version": 1,
        "kind": "physics_background_pairing_confirmation_report_bundle",
        "generator": generator,
        "report_mode": report_mode,
        "analysis_stage": analysis_stage,
        "scientifically_final": summary["scientifically_final"],
        "analysis_content_sha256": summary["analysis_content_sha256"],
        "random_cap_analysis_receipts": summary["random_cap_analysis_receipts"],
        "paper_reference_sha256": summary["paper_reference_sha256"],
        "completion": summary["completion"],
        "evidence": {
            str((Path("evidence") / name).as_posix()): _sha256(evidence_dir / name)
            for name in sorted(evidence_sources)
        },
    }
    provenance["evidence"]["evidence/paper_comparison.csv"] = _sha256(paper_comparison_path)
    provenance["content_sha256"] = _content_hash(provenance)
    _atomic_report_text(
        output_dir / "provenance.json",
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    )
    manifest_rows = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "manifest.sha256":
            continue
        manifest_rows.append(f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    _atomic_report_text(output_dir / "manifest.sha256", "\n".join(manifest_rows) + "\n")
    verify_confirmation_report_bundle(output_dir)
    return summary


def _atomic_report_text(path: Path, content: str) -> None:
    """Write one report-package text file idempotently."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"Refusing to replace different report artifact: {path}")
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _endpoint_matrix_svg(rows: list[dict[str, str]]) -> str:
    """Render one dependency-free compact mean endpoint matrix."""
    values = {
        (row["model"], row["selector"], row["pairing"], row["endpoint"]): _number(row, "mean")
        for row in rows
    }
    width, height = 1120, 330
    left, top, cell_width, cell_height = 100, 55, 142, 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.h{font-size:12px;font-weight:700}.c{font-size:11px}.m{font-size:12px;font-weight:700}</style>",
        '<text x="15" y="22" class="h">Mean efficiency at 250 Hz / AUPRC</text>',
    ]
    for column, (_, _, label) in enumerate(SELECTORS):
        parts.append(
            f'<text x="{left + column * cell_width + cell_width / 2:.0f}" y="44" text-anchor="middle" class="h">{label}</text>'
        )
    for row_index, model in enumerate(MODELS):
        y = top + row_index * cell_height
        parts.append(
            f'<text x="88" y="{y + 25}" text-anchor="end" class="m">{model.upper()}</text>'
        )
        for column, (selector, pairing, _) in enumerate(SELECTORS):
            efficiency = values[(model, selector, pairing, "efficiency_250hz")]
            auprc = values[(model, selector, pairing, "auprc")]
            intensity = max(0, min(255, round(245 - efficiency * 1200)))
            fill = f"rgb({intensity},{min(250, intensity + 18)},250)"
            x = left + column * cell_width
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_width - 4}" height="{cell_height - 4}" rx="3" fill="{fill}" stroke="#c8d0dc"/>',
                    f'<text x="{x + (cell_width - 4) / 2:.0f}" y="{y + 24}" text-anchor="middle" class="c">{100 * efficiency:.2f}% / {auprc:.3f}</text>',
                ]
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def verify_confirmation_report_bundle(bundle: Path) -> dict[str, Any]:
    """Verify every declared compact-package hash and its final/provisional identity."""
    bundle = bundle.resolve(strict=True)
    manifest = bundle / "manifest.sha256"
    declared: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        path = (bundle / relative).resolve(strict=True)
        if not path.is_relative_to(bundle) or relative in declared or _sha256(path) != digest:
            raise ValueError(f"Report bundle manifest differs for {relative}.")
        declared.add(relative)
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    }
    if declared != actual:
        raise ValueError("Report bundle manifest coverage differs.")
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    digest = provenance.pop("content_sha256", None)
    if digest != _content_hash(provenance):
        raise ValueError("Report bundle provenance content hash differs.")
    provenance["content_sha256"] = digest
    return provenance


def promote_confirmation_report(
    bundle: Path, repository: Path, results_dir: Path
) -> dict[str, str]:
    """Atomically install one verified final report and ignored download mirror."""
    provenance = verify_confirmation_report_bundle(bundle)
    if provenance.get("scientifically_final") is not True:
        raise RuntimeError("Only a scientifically final report bundle may be promoted.")
    repository = repository.resolve(strict=True)
    root_report = repository / "REPORT.md"
    source_report = bundle.resolve(strict=True) / "REPORT.md"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".REPORT.md.", dir=repository)
    try:
        os.close(descriptor)
        temporary = Path(temporary_name)
        shutil.copyfile(source_report, temporary)
        os.replace(temporary, root_report)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    results_dir = results_dir.expanduser().resolve()
    if results_dir.exists():
        if not results_dir.is_dir():
            raise FileExistsError(f"Report mirror is not a directory: {results_dir}")
        existing_manifest = results_dir / "manifest.sha256"
        if (
            not existing_manifest.exists()
            or existing_manifest.read_bytes() != (bundle / "manifest.sha256").read_bytes()
        ):
            raise FileExistsError(f"Refusing to replace a different report mirror: {results_dir}")
    else:
        results_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{results_dir.name}.", dir=results_dir.parent)
        )
        try:
            shutil.copytree(bundle, temporary_dir, dirs_exist_ok=True)
            os.replace(temporary_dir, results_dir)
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)
    verify_confirmation_report_bundle(results_dir)
    return {
        "repository_report": str(root_report),
        "results_mirror": str(results_dir),
        "manifest_sha256": _sha256(results_dir / "manifest.sha256"),
    }


__all__ = [
    "build_confirmation_report",
    "promote_confirmation_report",
    "verify_confirmation_report_bundle",
    "write_confirmation_report",
]
