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
        "endpoints_across_seeds.csv": across,
        "comparison_summary.csv": comparisons,
        "oas_vs_native_summary.csv": oas,
        "reserve_seed_decisions.csv": reserve,
    }
    for name, rows in tables.items():
        _write(tmp_path / name, rows)
    matrix = {"schema_version": 1, "cells": ["synthetic"]}
    matrix["content_sha256"] = _content_hash(matrix)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    endpoint_receipt = {
        "matrix_content_sha256": matrix["content_sha256"],
        "effective_runner_content_sha256": TEST_RUNNER_CONTENT_SHA256,
        "reserve_extension_content_sha256": TEST_RESERVE_CONTENT_SHA256,
        "artifacts": {
            "checkpoint_selections.csv": TEST_CHECKPOINT_SELECTIONS_SHA256,
        },
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
    if mode == "discovery":
        tables["structured_random_rank_associations.csv"] = [
            {
                "model": "ae",
                "logical_objective": "native__cap__cdf",
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
            "rank_association_rows": 12 if mode == "discovery" else 0,
            "rank_panels": 2 if mode == "discovery" else 0,
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


def test_final_report_answers_three_questions_and_shows_every_cell(tmp_path: Path) -> None:
    """The report is complete, winner-aware, and explicit about paper AUPRC."""
    repository = Path(__file__).resolve().parents[1]
    report, summary = build_confirmation_report(_analysis(tmp_path), repository)
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
        build_confirmation_report(analysis, Path(__file__).resolve().parents[1])


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
    )
    assert "Structured-CAP versus mean-random-CAP" in report
    assert "Five-random-bijection variability" in report
    assert "Validation seed-rho range" in report
    assert set(summary["random_cap_analysis_receipts"]) == {"discovery", "confirmation"}


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("provenance", "authoritative provenance"),
        ("method_count", "authoritative provenance or cardinality"),
        ("missing_rank_source", "rank-panel coverage"),
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
    else:
        associations_path = random_dir / "structured_random_rank_associations.csv"
        associations = list(csv.DictReader(associations_path.open(encoding="utf-8")))
        _write(associations_path, associations[1:])
        receipt["method"]["rank_association_rows"] = 11
        receipt["artifacts"][associations_path.name] = hashlib.sha256(
            associations_path.read_bytes()
        ).hexdigest()
    receipt["content_sha256"] = _content_hash(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_confirmation_report(
            analysis,
            Path(__file__).resolve().parents[1],
            discovery_random_analysis=random_dir,
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
    report, _ = build_confirmation_report(analysis, Path(__file__).resolve().parents[1])
    assert "| DSVAE | 5 | extension_complete |" in report
