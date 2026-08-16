"""Tests for authenticated physics campaign-to-paper comparisons."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.physics_background_pairing_paper_comparison import (
    build_paper_comparison,
    load_authenticated_paper_reference,
    paper_comparison_markdown,
    paper_reference_markdown,
)

ROOT = Path(__file__).parents[1]


def _winners() -> list[dict]:
    """Return one semantically matched native winner per model and method."""
    rows = []
    for model in ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp"):
        rows.extend(
            [
                {
                    "model": model,
                    "score": "native",
                    "metric": "cap",
                    "strategy": "cdf",
                    "mean_efficiency": 0.10,
                },
                {
                    "model": model,
                    "score": "native",
                    "metric": "drift",
                    "strategy": "",
                    "mean_efficiency": 0.08,
                },
                {
                    "model": model,
                    "score": "native",
                    "metric": "wasserstein",
                    "strategy": "",
                    "mean_efficiency": 0.06,
                },
            ]
        )
    return rows


def test_authenticated_reference_renders_both_baselines_and_correlations() -> None:
    """The human reference must expose both endpoints and signed correlations."""
    reference, digest = load_authenticated_paper_reference(ROOT)
    lines = paper_reference_markdown(reference, digest)
    text = "\n".join(lines)
    assert "Paper mean efficiency at 250 Hz" in text
    assert "Paper mean efficiency at q99" in text
    assert "Paper epoch-level metric/downstream correlations" in text
    assert "| DSVAE | 0.413 | -0.216 | -0.315 | 0.280 |" in text
    assert digest in text


def test_search_only_comparison_is_explicitly_provisional() -> None:
    """A frozen search without downstream values must never produce deltas."""
    reference, _ = load_authenticated_paper_reference(ROOT)
    rows = build_paper_comparison(reference)
    assert len(rows) == 48
    assert not any(row["comparison_status"] == "comparable" for row in rows)
    assert all(row["delta_percentage_points"] is None for row in rows)
    text = "\n".join(paper_comparison_markdown(rows, provisional=True))
    assert "cannot establish downstream improvement" in text
    assert "downstream_results_not_available" in text


def test_downstream_comparison_matches_only_native_cdf_drift_and_w1() -> None:
    """Authenticated final winners yield 250 Hz deltas but never inferred q99 deltas."""
    reference, _ = load_authenticated_paper_reference(ROOT)
    rows = build_paper_comparison(reference, _winners())
    comparable = [row for row in rows if row["comparison_status"] == "comparable"]
    assert len(comparable) == 18
    ae_cap = next(
        row for row in comparable if row["model"] == "ae" and row["paper_method"] == "cap"
    )
    assert ae_cap["paper_mean_efficiency_percent"] == 5.73
    assert ae_cap["campaign_mean_efficiency_percent"] == 10.0
    assert ae_cap["delta_percentage_points"] == pytest.approx(4.27)
    q99 = [row for row in rows if row["operating_point"] == "q99"]
    assert not any(row["delta_percentage_points"] is not None for row in q99)
    assert any(row["comparison_status"] == "campaign_q99_not_exported" for row in q99)
    text = "\n".join(paper_comparison_markdown(rows, provisional=False))
    assert "Campaign 250 Hz (%)" in text
    assert "campaign_q99_not_exported" in text


def test_duplicate_or_missing_matched_winner_fails_closed() -> None:
    """A final comparison requires exactly one winner for each matching method."""
    reference, _ = load_authenticated_paper_reference(ROOT)
    winners = _winners()
    with pytest.raises(ValueError, match="Duplicate"):
        build_paper_comparison(reference, [*winners, winners[0]])
    with pytest.raises(ValueError, match="Missing"):
        build_paper_comparison(reference, winners[1:])
