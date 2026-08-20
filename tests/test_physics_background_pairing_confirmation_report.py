"""Contracts for the final physics confirmation report."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import pytest

from src.evaluation import (
    physics_background_pairing_confirmation_report as report_module,
)
from src.evaluation.physics_background_pairing_confirmation_analysis import (
    _content_hash,
)
from src.evaluation.physics_background_pairing_confirmation_report import (
    MODELS,
    SELECTORS,
    build_confirmation_report,
    promote_confirmation_report,
    verify_confirmation_report_bundle,
    write_confirmation_report,
)

TEST_RUNNER_CONTENT_SHA256 = "b" * 64
TEST_RESERVE_CONTENT_SHA256 = "c" * 64
TEST_CHECKPOINT_SELECTIONS_SHA256 = "d" * 64


def _write(path: Path, rows: list[dict]) -> None:
    """Write a synthetic rectangular result table."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _analysis(tmp_path: Path) -> Path:
    """Create a minimal hash-authenticated complete analysis package."""
    across = []
    for model_index, model in enumerate(MODELS):
        for selector_index, (selector, pairing, _) in enumerate(SELECTORS):
            for endpoint in ("efficiency_250hz", "auprc"):
                across.append(
                    {
                        "model": model,
                        "selector": selector,
                        "pairing": pairing,
                        "endpoint": endpoint,
                        "mean": 0.1
                        + 0.01 * selector_index
                        + 0.001 * model_index
                        + (0.4 if endpoint == "auprc" else 0.0),
                        "median": 0.09,
                        "worst": 0.01,
                        "bottom_quartile_cvar": 0.03,
                        "seed_mean_std": 0.002,
                    }
                )
    comparisons = []
    for model in MODELS:
        for endpoint in ("efficiency_250hz", "auprc"):
            for contrast in ("cdf_cap_minus_wasserstein", "cdf_cap_minus_drift"):
                comparisons.append(
                    {
                        "scope": "model",
                        "model": model,
                        "family": "controlled",
                        "endpoint": endpoint,
                        "contrast": contrast,
                        "mean_delta": 0.01,
                        "adjusted_ci_low": 0.005,
                        "adjusted_ci_high": 0.015,
                        "classification": "positive_but_not_meaningful",
                    }
                )
            for pairing in (
                "flat_physical",
                "physics_summary",
                "typed_sliced_wasserstein",
                "jetclr",
            ):
                comparisons.append(
                    {
                        "scope": "model",
                        "model": model,
                        "family": "pairing",
                        "endpoint": endpoint,
                        "contrast": f"{pairing}_minus_cdf",
                        "mean_delta": 0.02,
                        "adjusted_ci_low": 0.01,
                        "adjusted_ci_high": 0.03,
                        "classification": "meaningful_improvement",
                    }
                )
    for endpoint in ("efficiency_250hz", "auprc"):
        for family, contrasts in (
            ("controlled", ("cdf_cap_minus_wasserstein", "cdf_cap_minus_drift")),
            (
                "pairing",
                tuple(
                    f"{pairing}_minus_cdf"
                    for pairing in (
                        "flat_physical",
                        "physics_summary",
                        "typed_sliced_wasserstein",
                        "jetclr",
                    )
                ),
            ),
        ):
            for contrast in contrasts:
                comparisons.append(
                    {
                        "scope": "macro",
                        "model": "macro_equal_model",
                        "family": family,
                        "endpoint": endpoint,
                        "contrast": contrast,
                        "mean_delta": 0.01,
                        "adjusted_ci_low": 0.005,
                        "adjusted_ci_high": 0.015,
                        "classification": "positive_but_not_meaningful",
                    }
                )
    oas = []
    for model in ("ae", "vae"):
        for selector, pairing, _ in SELECTORS:
            for endpoint in ("efficiency_250hz", "auprc"):
                oas.append(
                    {
                        "model": model,
                        "selector": selector,
                        "pairing": pairing,
                        "endpoint": endpoint,
                        "mean_delta": -0.01,
                        "ci95_low": -0.02,
                        "ci95_high": 0.0,
                    }
                )
    reserve = [{"model": model, "decision": "no_extension"} for model in MODELS]
    tables = {
        "endpoints_long.csv": [{"synthetic": "endpoint"}],
        "endpoints_summary.csv": [{"synthetic": "summary"}],
        "endpoints_across_seeds.csv": across,
        "selector_comparison_paired.csv": [{"synthetic": "selector"}],
        "pairing_vs_cdf_paired.csv": [{"synthetic": "pairing"}],
        "comparison_summary.csv": comparisons,
        "oas_endpoints_long.csv": [{"synthetic": "oas_endpoint"}],
        "oas_vs_native_paired.csv": [{"synthetic": "oas_pair"}],
        "oas_vs_native_summary.csv": oas,
        "reserve_seed_decisions.csv": reserve,
    }
    for name, rows in tables.items():
        _write(tmp_path / name, rows)
    matrix = {"schema_version": 1, "cells": ["synthetic"]}
    matrix["content_sha256"] = _content_hash(matrix)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    optimized = tmp_path / "optimized_metric.json"
    optimized.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    marker = tmp_path / "complete_marker.json"
    marker.write_text(
        json.dumps(
            {
                "run_name": "synthetic_run",
                "optimized_metric_artifact": str(optimized.resolve()),
                "optimized_metric_artifact_sha256": hashlib.sha256(
                    optimized.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    source_markers = [
        {
            "retrain_index": 0,
            "run_id": "synthetic_run",
            "marker": str(marker.resolve()),
            "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
            "optimized_metric_artifact": str(optimized.resolve()),
            "optimized_metric_artifact_sha256": hashlib.sha256(optimized.read_bytes()).hexdigest(),
        }
    ]
    endpoint_receipt = {
        "matrix_content_sha256": matrix["content_sha256"],
        "effective_runner_content_sha256": TEST_RUNNER_CONTENT_SHA256,
        "reserve_extension_content_sha256": TEST_RESERVE_CONTENT_SHA256,
        "artifacts": {
            "checkpoint_selections.csv": TEST_CHECKPOINT_SELECTIONS_SHA256,
        },
        "cardinality": {
            "sealed_runs": 1,
            "checkpoint_selections": 42,
            "primary_endpoint_rows": 5040,
            "oas_endpoint_rows": 1680,
        },
        "source_markers": source_markers,
        "source_markers_content_sha256": _content_hash(source_markers),
    }
    endpoint_receipt_path = tmp_path / "endpoint_extraction.json"
    endpoint_receipt_path.write_text(json.dumps(endpoint_receipt), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": {"seeds": [271828, 314159, 161803], "anomalies": 20},
        "artifacts": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in tables
        },
        "inputs": {
            "matrix": str(matrix_path.resolve()),
            "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "endpoint_receipt": str(endpoint_receipt_path.resolve()),
            "endpoint_receipt_sha256": hashlib.sha256(
                endpoint_receipt_path.read_bytes()
            ).hexdigest(),
        },
    }
    manifest["content_sha256"] = _content_hash(manifest)
    (tmp_path / "confirmation_endpoint_analysis.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return tmp_path


def _random_analysis(
    tmp_path: Path,
    mode: str,
    *,
    confirmation_provenance: dict[str, str | None] | None = None,
) -> Path:
    """Create a minimal portable random-CAP analysis receipt for report tests."""
    tmp_path.mkdir()
    values = []
    variability = []
    for split_index, split in enumerate(("validate", "test")):
        split_values = []
        for replicate in range(5):
            value = -0.2 + replicate / 100
            split_values.append(value)
            values.append(
                {
                    "task_index": 0,
                    "membership_index": 0,
                    "evaluation_score": "native",
                    "split": split,
                    "replicate": replicate,
                    "pairing_seed": 100 + split_index * 10 + replicate,
                    "checkpoint_sha256": "a" * 64,
                    "pair_table_sha256": f"{split_index * 10 + replicate + 1:064x}",
                    "cap": value,
                }
            )
        variability.append(
            {
                "task_index": 0,
                "membership_index": 0,
                "model": "ae",
                "evaluation_score": "native",
                "split": split,
                "n_pairings": 5,
                "cap_mean": statistics.fmean(split_values),
                "cap_sample_variance": statistics.variance(split_values),
                "cap_sample_std": statistics.stdev(split_values),
                "cap_range": max(split_values) - min(split_values),
            }
        )
    tables = {"random_cap_values.csv": values, "random_cap_variability.csv": variability}
    tables["structured_cap_values.csv"] = [
        {
            "model": "ae",
            "selection_score": "native",
            "evaluation_score": "native",
            "selection_strategy": "cdf",
            "split": split,
            "checkpoint_sha256": "a" * 64,
            "cap": 0.5,
        }
        for split in ("validate", "test")
    ]
    tables["structured_random_rank_associations.csv"] = [
        {
            "model": "ae",
            "selection_score": "native",
            "evaluation_score": "native",
            "logical_objective": "native__cap__cdf",
            "selection_strategy": "cdf",
            "split": split,
            "random_source": source,
            "n_checkpoints": 20,
            "spearman_rho": 0.25,
            "pairing_seed_rho_finite_count": 5,
            "pairing_seed_rho_undefined_count": 0,
            "pairing_seed_rho_min": 0.1,
            "pairing_seed_rho_max": 0.4,
            "pairing_seed_rho_range": 0.3,
        }
        for split in ("validate", "test")
        for source in ("five_pairing_mean", *[f"pairing_replicate_{i}" for i in range(5)])
    ]
    downstream_rhos = {
        ("validate", "efficiency_250hz"): (0.7, 0.2),
        ("validate", "auprc"): (0.1, 0.4),
        ("test", "efficiency_250hz"): (0.5, 0.5),
        ("test", "auprc"): (None, None),
    }
    downstream = []
    for split in ("validate", "test"):
        for endpoint in ("efficiency_250hz", "auprc"):
            structured_rho, random_mean_rho = downstream_rhos[(split, endpoint)]
            for source in (
                "structured_cap",
                "five_pairing_mean",
                *[f"pairing_replicate_{index}" for index in range(5)],
            ):
                rho = (
                    structured_rho
                    if source == "structured_cap"
                    else random_mean_rho
                    if source == "five_pairing_mean"
                    else random_mean_rho
                )
                downstream.append(
                    {
                        "model": "ae",
                        "selection_score": "native",
                        "evaluation_score": "native",
                        "logical_objective": "native__cap__cdf",
                        "selection_strategy": "cdf",
                        "split": split,
                        "downstream_endpoint": endpoint,
                        "cap_source": source,
                        "cap_direction": "maximize",
                        "downstream_direction": "maximize",
                        "n_checkpoints": 20,
                        "n_anomalies_per_checkpoint": 20,
                        "spearman_rho": "" if rho is None else rho,
                        "spearman_defined": rho is not None,
                        "constant": rho is None,
                        "ties": False,
                        "insufficient_n": False,
                    }
                )
    tables["cap_downstream_rank_associations.csv"] = downstream
    for name, rows in tables.items():
        _write(tmp_path / name, rows)
    receipt = {
        "schema_version": 1,
        "method": {
            "mode": mode,
            "purpose": "negative_control_diagnostic_only",
            "tasks": 1,
            "memberships": 1,
            "membership_scores": 1,
            "seed_rows": 10,
            "variability_rows": 2,
            "rank_association_rows": 12,
            "rank_panels": 2,
            "downstream_rows": 40,
            "downstream_memberships": 1,
            "downstream_membership_scores": 1,
            "downstream_anomalies": 20,
            "downstream_endpoints": 2,
            "downstream_association_rows": 28,
            "downstream_association_panels": 4,
            "downstream_association_sources": 7,
            "downstream_association_defined_rows": 21,
            "downstream_association_undefined_rows": 7,
            "downstream_panels_are_unpooled": True,
            "random_cap_did_not_select_hyperparameters_or_checkpoints": True,
        },
        "inputs": {
            "manifest_namespace": report_module.DISCOVERY_NAMESPACE,
            "manifest_sha256": report_module.DISCOVERY_MANIFEST_SHA256,
            "manifest_content_sha256": report_module.DISCOVERY_MANIFEST_CONTENT_SHA256,
            "random_pairing_content_sha256": report_module.DISCOVERY_RANDOM_PAIRING_CONTENT_SHA256,
            **(confirmation_provenance or {}),
        },
        "artifacts": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in tables
        },
    }
    receipt["content_sha256"] = _content_hash(receipt)
    (tmp_path / "random_cap_analysis.json").write_text(json.dumps(receipt), encoding="utf-8")
    return tmp_path


def _context(tmp_path: Path, analysis: Path) -> Path:
    """Create a minimal authenticated persistence/curve package for report tests."""
    tmp_path.mkdir()
    analysis_manifest = json.loads((analysis / "confirmation_endpoint_analysis.json").read_text())
    persistence = [
        {
            "model": model,
            "selector": selector,
            "pairing": pairing,
            "endpoint": endpoint,
            "mean_50_epoch": 0.1,
            "mean_200_epoch": 0.2,
            "delta_200_minus_50": 0.1,
        }
        for model in MODELS
        for selector, pairing, _ in SELECTORS
        for endpoint in ("efficiency_250hz", "auprc")
    ]
    hypotheses = [
        {
            "model": model,
            "endpoint": endpoint,
            "contrast": contrast,
            "delta_50_epoch": -0.1 if contrast.endswith("minus_cdf") else 0.1,
            "delta_200_epoch": 0.1,
            "direction_outcome": (
                "reversed" if contrast.endswith("minus_cdf") else "survived_directionally"
            ),
        }
        for model in MODELS
        for endpoint in ("efficiency_250hz", "auprc")
        for contrast in (
            "cdf_cap_minus_wasserstein",
            "cdf_cap_minus_drift",
            "flat_physical_minus_cdf",
            "physics_summary_minus_cdf",
            "typed_sliced_wasserstein_minus_cdf",
            "jetclr_minus_cdf",
        )
    ]
    curves = [
        {
            "model": "ae",
            "trial_number": 1,
            "training_seed": 2,
            "run_id": "synthetic",
            "logical_objective": "native__cap__cdf",
            "direction": "maximize",
            "epoch": epoch,
            "mlflow_step": epoch,
            "value": epoch / 200,
        }
        for epoch in range(1, 201)
    ]
    curve_summary = [
        {
            "model": "ae",
            "logical_objective": "native__cap__cdf",
            "direction": "maximize",
            "runs": 1,
            "mean_epoch_50": 0.25,
            "mean_epoch_200": 1.0,
            "favorable_change_50_to_200": 0.75,
            "mean_last_40_change": 0.2,
        }
    ]
    tables = {
        "persistence.csv": persistence,
        "hypotheses.csv": hypotheses,
        "training_curves.csv": curves,
        "training_curve_summary.csv": curve_summary,
    }
    for name, rows in tables.items():
        _write(tmp_path / name, rows)
    receipt = {
        "schema_version": 1,
        "kind": "physics_background_pairing_confirmation_context",
        "inputs": {
            "confirmation_analysis_content_sha256": analysis_manifest["content_sha256"],
            "endpoint_receipt_sha256": analysis_manifest["inputs"]["endpoint_receipt_sha256"],
        },
        "cardinality": {name.removesuffix(".csv"): len(rows) for name, rows in tables.items()},
        "artifacts": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in tables
        },
    }
    receipt["content_sha256"] = _content_hash(receipt)
    (tmp_path / "confirmation_context.json").write_text(json.dumps(receipt), encoding="utf-8")
    return tmp_path


def test_final_report_answers_three_questions_and_shows_every_cell(tmp_path: Path) -> None:
    """The report is complete, winner-aware, and explicit about paper AUPRC."""
    repository = Path(__file__).resolve().parents[1]
    report, summary = build_confirmation_report(
        _analysis(tmp_path), repository, report_mode="provisional"
    )
    assert "## Direct answers" in report
    assert "## Complete confirmation matrix" in report
    assert "## CDF CAP versus W1 and drift" in report
    assert "## Learned pairing versus CDF CAP" in report
    assert "Paper AUPRC is unavailable" in report
    assert report.count("| AE |") >= 1
    assert "**" in report
    assert summary["paper_matched_total"] == 18
    assert summary["cdf_cap_efficiency_total"] == 12
    assert summary["cdf_cap_auprc_total"] == 12
    assert summary["learned_pairing_efficiency_total"] == 24
    assert summary["learned_pairing_auprc_total"] == 24


def test_final_mode_requires_both_random_control_analyses(tmp_path: Path) -> None:
    """A missing mandatory negative-control package cannot produce a final report."""
    with pytest.raises(RuntimeError, match="context are mandatory"):
        build_confirmation_report(_analysis(tmp_path), Path(__file__).resolve().parents[1])


def test_final_report_refuses_pending_reserve_extension(tmp_path: Path) -> None:
    """A triggered but incomplete reserve extension cannot be called final."""
    analysis = _analysis(tmp_path)
    reserve_path = analysis / "reserve_seed_decisions.csv"
    rows = list(csv.DictReader(reserve_path.open(encoding="utf-8", newline="")))
    rows[0]["decision"] = "add_both_reserve_seeds"
    _write(reserve_path, rows)
    manifest_path = analysis / "confirmation_endpoint_analysis.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("content_sha256")
    manifest["artifacts"]["reserve_seed_decisions.csv"] = hashlib.sha256(
        reserve_path.read_bytes()
    ).hexdigest()
    manifest["content_sha256"] = _content_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="extension is incomplete"):
        build_confirmation_report(
            analysis, Path(__file__).resolve().parents[1], report_mode="provisional"
        )


def test_final_report_authenticates_random_control_analysis(tmp_path: Path, monkeypatch) -> None:
    """The final report renders discovery rank and confirmation variability evidence."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    analysis = _analysis(analysis)
    analysis_manifest = json.loads((analysis / "confirmation_endpoint_analysis.json").read_text())
    confirmation_provenance = report_module._confirmation_provenance(analysis_manifest)
    monkeypatch.setattr(report_module, "DISCOVERY_TASKS", 1)
    monkeypatch.setattr(report_module, "DISCOVERY_MEMBERSHIPS", 1)
    monkeypatch.setattr(report_module, "DISCOVERY_MEMBERSHIP_SCORES", 1)
    report, summary = build_confirmation_report(
        analysis,
        Path(__file__).resolve().parents[1],
        discovery_random_analysis=_random_analysis(tmp_path / "random_discovery", "discovery"),
        confirmation_random_analysis=_random_analysis(
            tmp_path / "random_confirmation",
            "confirmation",
            confirmation_provenance=confirmation_provenance,
        ),
        confirmation_context=_context(tmp_path / "context", analysis),
    )
    assert "Structured-CAP versus mean-random-CAP" in report
    assert "Five-random-bijection variability" in report
    assert "Validation seed-rho range" in report
    assert "Discovery CAP-to-downstream rank alignment" in report
    assert "Confirmation CAP-to-downstream rank alignment" in report
    assert "structured/random/tie/undefined panels are 1/0/1/0" in report
    assert "for AUPRC they are 0/1/0/1" in report
    assert "post-hoc on the fixed structured-selected checkpoint universes" in report
    assert set(summary["random_cap_analysis_receipts"]) == {"discovery", "confirmation"}
    assert summary["random_cap_downstream_rank_comparison"]["discovery"]["auprc"] == {
        "structured": 0,
        "random": 1,
        "tie": 0,
        "undefined": 1,
    }


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("provenance", "authoritative provenance"),
        ("method_count", "authoritative provenance or cardinality"),
        ("missing_rank_source", "rank-panel coverage"),
        ("missing_downstream", "lacks downstream rank associations"),
        ("downstream_method_count", "downstream-association method cardinality"),
    ),
)
def test_final_report_rejects_random_analysis_tampering(
    tmp_path: Path, monkeypatch, tamper: str, message: str
) -> None:
    """Wrong provenance, method counts, or a missing rank source cannot reach the report."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    analysis = _analysis(analysis)
    monkeypatch.setattr(report_module, "DISCOVERY_TASKS", 1)
    monkeypatch.setattr(report_module, "DISCOVERY_MEMBERSHIPS", 1)
    monkeypatch.setattr(report_module, "DISCOVERY_MEMBERSHIP_SCORES", 1)
    random_dir = _random_analysis(tmp_path / "random_discovery", "discovery")
    receipt_path = random_dir / "random_cap_analysis.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("content_sha256")
    if tamper == "provenance":
        receipt["inputs"]["manifest_namespace"] = "failed_pilot"
    elif tamper == "method_count":
        receipt["method"]["seed_rows"] = 11
    elif tamper == "missing_rank_source":
        associations_path = random_dir / "structured_random_rank_associations.csv"
        associations = list(csv.DictReader(associations_path.open(encoding="utf-8")))
        _write(associations_path, associations[1:])
        receipt["method"]["rank_association_rows"] = 11
        receipt["artifacts"][associations_path.name] = hashlib.sha256(
            associations_path.read_bytes()
        ).hexdigest()
    elif tamper == "missing_downstream":
        receipt["artifacts"].pop("cap_downstream_rank_associations.csv")
    else:
        receipt["method"]["downstream_association_rows"] = 29
    receipt["content_sha256"] = _content_hash(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_confirmation_report(
            analysis,
            Path(__file__).resolve().parents[1],
            discovery_random_analysis=random_dir,
            report_mode="provisional",
        )


def test_final_report_rejects_mixed_confirmation_endpoint_provenance(
    tmp_path: Path,
) -> None:
    """A random-CAP package from another effective runner cannot enter the report."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    analysis = _analysis(analysis)
    analysis_manifest = json.loads((analysis / "confirmation_endpoint_analysis.json").read_text())
    provenance = report_module._confirmation_provenance(analysis_manifest)
    random_dir = _random_analysis(
        tmp_path / "random_confirmation",
        "confirmation",
        confirmation_provenance=provenance,
    )
    receipt_path = random_dir / "random_cap_analysis.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("content_sha256")
    receipt["inputs"]["runner_manifest_content_sha256"] = "f" * 64
    receipt["content_sha256"] = _content_hash(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="endpoint provenance"):
        build_confirmation_report(
            analysis,
            Path(__file__).resolve().parents[1],
            confirmation_random_analysis=random_dir,
            report_mode="provisional",
        )


def test_final_report_accepts_completed_dsvae_only_extension(tmp_path: Path) -> None:
    """A mixed three/five-seed package is final only after the trigger is complete."""
    analysis = _analysis(tmp_path)
    reserve_path = analysis / "reserve_seed_decisions.csv"
    rows = list(csv.DictReader(reserve_path.open(encoding="utf-8", newline="")))
    for row in rows:
        row["observed_seed_count"] = "5" if row["model"] == "dsvae" else "3"
        if row["model"] == "dsvae":
            row["decision"] = "extension_complete"
    _write(reserve_path, rows)
    manifest_path = analysis / "confirmation_endpoint_analysis.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("content_sha256")
    primary = [271828, 314159, 161803]
    manifest["method"]["training_seeds_by_model"] = {
        model: primary + ([141421, 173205] if model == "dsvae" else []) for model in MODELS
    }
    manifest["artifacts"][reserve_path.name] = hashlib.sha256(
        reserve_path.read_bytes()
    ).hexdigest()
    manifest["content_sha256"] = _content_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report, _ = build_confirmation_report(
        analysis, Path(__file__).resolve().parents[1], report_mode="provisional"
    )
    assert "| DSVAE | 5 | extension_complete |" in report


def test_report_bundle_manifest_and_explicit_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compact bundle is complete, tamper-evident, and explicitly promotable."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    analysis = _analysis(analysis)
    provenance = report_module._confirmation_provenance(
        json.loads((analysis / "confirmation_endpoint_analysis.json").read_text())
    )
    monkeypatch.setattr(report_module, "DISCOVERY_TASKS", 1)
    monkeypatch.setattr(report_module, "DISCOVERY_MEMBERSHIPS", 1)
    monkeypatch.setattr(report_module, "DISCOVERY_MEMBERSHIP_SCORES", 1)
    discovery = _random_analysis(tmp_path / "random_discovery", "discovery")
    confirmation = _random_analysis(
        tmp_path / "random_confirmation",
        "confirmation",
        confirmation_provenance=provenance,
    )
    repository = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "bundle"
    summary = write_confirmation_report(
        analysis,
        repository,
        bundle,
        discovery_random_analysis=discovery,
        confirmation_random_analysis=confirmation,
        confirmation_context=_context(tmp_path / "context", analysis),
    )
    assert summary["scientifically_final"] is True
    verified = verify_confirmation_report_bundle(bundle)
    assert verified["scientifically_final"] is True
    assert (bundle / "prior_repository_REPORT.md").exists()
    assert (bundle / "evidence/discovery_random_cap_downstream_rank_associations.csv").exists()
    assert (bundle / "evidence/confirmation_random_cap_downstream_rank_associations.csv").exists()
    assert (bundle / "evidence/confirmation_endpoints_long.csv").exists()
    assert (bundle / "evidence/confirmation_selector_comparison_paired.csv").exists()
    assert (bundle / "evidence/discovery_random_random_cap_values.csv").exists()
    assert (bundle / "evidence/confirmation_random_random_cap_values.csv").exists()
    assert (bundle / "evidence/context_training_curves.csv").exists()
    assert (bundle / "evidence/paper_comparison.csv").exists()

    install_repository = tmp_path / "install_repository"
    install_repository.mkdir()
    mirror = install_repository / "results/physics_confirmation/final_v1"
    installed = promote_confirmation_report(bundle, install_repository, mirror)
    assert Path(installed["repository_report"]).read_bytes() == (bundle / "REPORT.md").read_bytes()
    assert (
        verify_confirmation_report_bundle(mirror)["content_sha256"] == verified["content_sha256"]
    )

    evidence = next((bundle / "evidence").glob("*.csv"))
    evidence.write_text(evidence.read_text() + "tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        verify_confirmation_report_bundle(bundle)
