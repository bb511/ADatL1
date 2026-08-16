"""Contracts for joined physics search-50 endpoint analysis."""

from __future__ import annotations

import copy

import pytest

from src.evaluation.physics_background_pairing_analysis import DOWNSTREAM_DATASETS
from src.evaluation.physics_background_pairing_endpoint_analysis import (
    _canonical_hash,
    build_endpoint_analysis,
)
from src.evaluation.physics_background_pairing_search50_analysis import STRATEGIES

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")


def _fixture() -> tuple[list[dict], list[dict], dict]:
    """Construct the exact 591-membership campaign with two complete endpoints."""
    families = [
        (model, score)
        for model in MODELS
        for score in (("native", "residual_oas") if model in {"ae", "vae"} else ("native",))
    ]
    fronts = [
        (model, score, metric, strategy)
        for model, score in families
        for metric, strategy in STRATEGIES
    ]
    efficiency_rows, auprc_rows, winners = [], [], []
    membership_index = 0
    for front_index, (model, score, metric, strategy) in enumerate(fronts):
        count = 11 if front_index < 31 else 10
        winner_index = membership_index + count - 1
        for local_index in range(count):
            objective = f"{score}__{metric}" + (f"__{strategy}" if strategy else "")
            efficiency = 0.10 + 0.01 * STRATEGIES.index((metric, strategy))
            auprc = 0.20 + 0.01 * STRATEGIES.index((metric, strategy))
            if local_index == count - 1:
                efficiency += 0.10
                auprc += 0.05
            common = {
                "membership_index": membership_index,
                "model": model,
                "logical_objective": objective,
                "score": score,
                "metric": metric,
                "strategy": strategy,
                "trial_number": front_index * 20 + local_index,
                "checkpoint": f"checkpoints/{membership_index}.ckpt",
                "checkpoint_sha256": f"{membership_index:064x}",
            }
            for anomaly in DOWNSTREAM_DATASETS:
                efficiency_rows.append(
                    {**common, "anomaly": anomaly, "efficiency_operational": efficiency}
                )
                auprc_rows.append({**common, "anomaly": anomaly, "auprc": auprc})
            if membership_index == winner_index:
                winners.append({**common, "mean_efficiency": efficiency})
            membership_index += 1
    assert membership_index == 591 and len(winners) == 56
    aggregate = {
        "schema_version": 1,
        "manifest_content_sha256": "manifest",
        "candidate_auprc": auprc_rows,
        "candidate_auprc_sha256": _canonical_hash(auprc_rows),
    }
    return efficiency_rows, winners, aggregate


def test_endpoint_analysis_has_exact_join_and_table_cardinalities() -> None:
    """Complete inputs yield every required endpoint and comparison table."""
    efficiency, winners, aggregate = _fixture()
    result = build_endpoint_analysis(efficiency, winners, aggregate)
    assert result["method"] == {
        "schema_version": 1,
        "primary_selection_endpoint": "efficiency_250hz",
        "candidate_memberships": 591,
        "logical_fronts": 56,
        "anomalies": 20,
        "rows_per_endpoint": 11820,
        "combined_rows": 23640,
        "auprc_aggregate_manifest_content_sha256": "manifest",
    }
    assert len(result["combined_long"]) == 23640
    assert len(result["primary_front_summary"]) == 56
    assert len(result["controlled_selector_comparison"]) == 24
    assert len(result["fixed_pairing_vs_cdf"]) == 32
    assert len(result["best_learned_comparison"]) == 24
    assert len(result["oas_panel"]) == 14
    assert {row["endpoint"] for row in result["combined_long"]} == {
        "efficiency_250hz",
        "auprc",
    }


def test_comparisons_keep_w1_drift_unpaired_and_oas_separate() -> None:
    """Selector and score semantics cannot silently turn into extra pairings."""
    result = build_endpoint_analysis(*_fixture())
    controlled = result["controlled_selector_comparison"]
    assert all(
        row["pairing"] == "not_applicable"
        for row in controlled
        if row["selector"] in {"wasserstein", "drift"}
    )
    native = [row for row in controlled if row["score"] == "native"]
    oas = [row for row in controlled if row["score"] == "residual_oas"]
    assert len(native) == 18 and len(oas) == 6
    assert {(row["model"], row["selector"]) for row in result["oas_panel"]} == {
        (model, metric) for model in ("ae", "vae") for metric, _ in STRATEGIES
    }


def test_incomplete_or_changed_auprc_aggregate_fails_closed() -> None:
    """Missing rows, stale hashes, and exact-key mismatches cannot publish tables."""
    efficiency, winners, aggregate = _fixture()
    incomplete = copy.deepcopy(aggregate)
    incomplete["candidate_auprc"].pop()
    incomplete["candidate_auprc_sha256"] = _canonical_hash(incomplete["candidate_auprc"])
    with pytest.raises(ValueError, match="AUPRC aggregate incomplete"):
        build_endpoint_analysis(efficiency, winners, incomplete)

    stale = copy.deepcopy(aggregate)
    stale["candidate_auprc"][0]["auprc"] += 0.01
    with pytest.raises(ValueError, match="hash differs"):
        build_endpoint_analysis(efficiency, winners, stale)

    mismatched = copy.deepcopy(aggregate)
    mismatched["candidate_auprc"][0]["checkpoint_sha256"] = "f" * 64
    mismatched["candidate_auprc_sha256"] = _canonical_hash(mismatched["candidate_auprc"])
    with pytest.raises(ValueError, match="key join differs"):
        build_endpoint_analysis(efficiency, winners, mismatched)


def test_front_winner_must_match_efficiency_selected_candidate() -> None:
    """A different downstream winner cannot be introduced during endpoint joining."""
    efficiency, winners, aggregate = _fixture()
    winners[0]["membership_index"] -= 1
    with pytest.raises(ValueError, match="front winners differ"):
        build_endpoint_analysis(efficiency, winners, aggregate)
