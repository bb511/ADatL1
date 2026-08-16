"""Tests for provisional 50-epoch downstream analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.physics_background_pairing_analysis import DOWNSTREAM_DATASETS
from src.evaluation.physics_background_pairing_paper_comparison import (
    load_authenticated_paper_reference,
)
from src.evaluation.physics_background_pairing_search50_analysis import (
    PROVISIONAL_NAMESPACE,
    build_provisional_search50_analysis,
    render_provisional_search50_report,
)

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
STRATEGIES = (
    ("cap", "flat_physical"),
    ("cap", "physics_summary"),
    ("cap", "typed_sliced_wasserstein"),
    ("cap", "jetclr"),
    ("cap", "cdf"),
    ("wasserstein", ""),
    ("drift", ""),
)
REPOSITORY_ROOT = Path(__file__).parents[1]


def _fixture() -> tuple[dict, list[dict]]:
    """Build the exact 591-member, 56-front synthetic campaign shape."""
    fronts = [
        (model, score, metric, strategy)
        for model in MODELS
        for score in (("native", "residual_oas") if model in {"ae", "vae"} else ("native",))
        for metric, strategy in STRATEGIES
    ]
    memberships, efficiencies = [], []
    membership_index = 0
    for front_index, (model, score, metric, strategy) in enumerate(fronts):
        count = 11 if front_index < 31 else 10
        strategy_index = STRATEGIES.index((metric, strategy))
        desired = [0.30 + 0.03 * strategy_index] * len(DOWNSTREAM_DATASETS)
        if (model, score, metric, strategy) == ("ae", "native", "wasserstein", ""):
            desired = [0.75] * len(DOWNSTREAM_DATASETS)
        elif (model, score, metric, strategy) == ("ae", "native", "drift", ""):
            desired = [0.99] * 15 + [0.21] * 5
        for local_index in range(count):
            objective = f"{score}__{metric}" + (f"__{strategy}" if strategy else "")
            membership = {
                "membership_index": membership_index,
                "model": model,
                "logical_objective": objective,
                "score": score,
                "metric": metric,
                "strategy": strategy,
                "trial_number": front_index * 20 + local_index,
                "objective_values": [0.5, 0.5],
                "retrain_index": front_index,
            }
            memberships.append(membership)
            is_first_front_tie = front_index == 0 and local_index in {0, 1}
            is_winner = local_index == count - 1 or is_first_front_tie
            values = desired if is_winner else [max(0.0, value - 0.10) for value in desired]
            for anomaly, value in zip(DOWNSTREAM_DATASETS, values):
                efficiencies.append(
                    {
                        **{
                            key: membership[key]
                            for key in (
                                "membership_index",
                                "model",
                                "logical_objective",
                                "score",
                                "metric",
                                "strategy",
                                "trial_number",
                            )
                        },
                        "checkpoint": f"checkpoints/{membership_index}.ckpt",
                        "checkpoint_sha256": f"{membership_index:064x}",
                        "anomaly": anomaly,
                        "efficiency_operational": value,
                    }
                )
            membership_index += 1
    assert len(memberships) == 591
    return {
        "schema_version": 2,
        "selection_uses_downstream_anomalies": False,
        "front_memberships": memberships,
    }, efficiencies


def _analysis() -> dict:
    """Build one authenticated-reference synthetic analysis."""
    manifest, rows = _fixture()
    reference, _ = load_authenticated_paper_reference(REPOSITORY_ROOT)
    return build_provisional_search50_analysis(manifest, rows, reference)


def test_search50_analysis_has_exact_contract_and_deterministic_winners() -> None:
    """All requested tables have exact coverage and deterministic tie handling."""
    analysis = _analysis()
    assert analysis["method"] == {
        "schema_version": 1,
        "namespace": PROVISIONAL_NAMESPACE,
        "training_epochs": 50,
        "candidate_universe_selected_without_downstream": True,
        "reported_choices_use_downstream_anomalies": True,
        "memberships": 591,
        "logical_fronts": 56,
        "cross_combination_choices": 16,
    }
    assert len(analysis["candidate_efficiencies"]) == 591 * 20
    assert len(analysis["candidate_summary"]) == 591
    assert len(analysis["front_winners"]) == 56
    assert len(analysis["cross_combination_rankings"]) == 56
    assert len(analysis["cross_combination_choices"]) == 16
    first_front = analysis["front_winners"][0]
    assert first_front["logical_objective"] == "native__cap__flat_physical"
    assert first_front["membership_index"] == 0


def test_cross_combination_mean_and_tail_choices_are_distinct_and_predeclared() -> None:
    """Mean and lower-tail selectors apply their fixed, distinct rules."""
    choices = {
        (row["model"], row["score"], row["selector"]): row
        for row in _analysis()["cross_combination_choices"]
    }
    assert choices[("ae", "native", "mean_oracle")]["logical_objective"] == "native__drift"
    assert choices[("ae", "native", "tail_oracle")]["logical_objective"] == ("native__wasserstein")
    assert "max mean" in choices[("ae", "native", "mean_oracle")]["selection_rule"]
    assert "bottom-quartile CVaR" in choices[("ae", "native", "tail_oracle")]["selection_rule"]


def test_paper_comparison_and_report_are_explicitly_provisional() -> None:
    """Only 18 native 250-Hz deltas are populated and caveats are prominent."""
    analysis = _analysis()
    paper = analysis["paper_comparison"]
    assert len(paper) == 48
    comparable = [
        row for row in paper if row["comparison_status"] == "provisional_50epoch_oracle_delta"
    ]
    assert len(comparable) == 18
    assert all(row["operating_point"] == "250hz" for row in comparable)
    assert all(row["campaign_method"].startswith("native__") for row in comparable)
    assert all(
        row["delta_percentage_points"] is None for row in paper if row["operating_point"] == "q99"
    )
    report = render_provisional_search50_report(analysis)
    assert "does not replace the 200-epoch retraining result" in report
    assert "second selection layer" in report
    assert "target-selection biased" in report
    assert "q99 is not exported and is not inferred" in report


def test_missing_candidate_efficiency_fails_closed() -> None:
    """Incomplete anomaly coverage cannot produce provisional selections."""
    manifest, rows = _fixture()
    reference, _ = load_authenticated_paper_reference(REPOSITORY_ROOT)
    with pytest.raises(ValueError, match="exact 20-anomaly coverage"):
        build_provisional_search50_analysis(manifest, rows[:-1], reference)
