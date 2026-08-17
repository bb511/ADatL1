"""Contracts for the final physics confirmation report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.physics_background_pairing_confirmation_analysis import (
    _content_hash,
)
from src.evaluation.physics_background_pairing_confirmation_report import (
    MODELS,
    SELECTORS,
    build_confirmation_report,
)


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
    manifest = {
        "schema_version": 1,
        "method": {"seeds": [271828, 314159, 161803], "anomalies": 20},
        "artifacts": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in tables
        },
    }
    manifest["content_sha256"] = _content_hash(manifest)
    (tmp_path / "confirmation_endpoint_analysis.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
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
