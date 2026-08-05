"""Golden tests for pure physics background-pairing scientific analysis."""

from __future__ import annotations

import copy

import pytest

from src.evaluation.physics_background_pairing_analysis import (
    CAP_STRATEGIES,
    DOWNSTREAM_DATASETS,
    MODEL_ORDER,
    build_physics_background_pairing_analysis,
)


def _fixture():
    """Build a complete nonuniform synthetic campaign with known effects."""
    studies = []
    memberships = []
    candidates = []
    winners = []
    checkpoint_rows = []
    primary_rows = []
    retrains = []
    retrain_index = {}
    for model_index, model in enumerate(MODEL_ORDER):
        scores = ("native", "residual_oas") if model in {"ae", "vae"} else ("native",)
        objectives = []
        for score in scores:
            methods = [("cap", strategy) for strategy in CAP_STRATEGIES] + [
                ("wasserstein", None),
                ("drift", None),
            ]
            for metric, strategy in methods:
                objective_id = f"{score}__{metric}" + (f"__{strategy}" if strategy else "")
                directions = (
                    ["maximize", "minimize"] if metric == "cap" else ["minimize", "minimize"]
                )
                objectives.append(
                    {
                        "id": objective_id,
                        "score": score,
                        "metric": metric,
                        "strategy": strategy,
                        "directions": directions,
                        "value_indices": [2 * len(objectives), 2 * len(objectives) + 1],
                    }
                )
                front_candidates = []
                for candidate_offset, trial in enumerate((0, 1)):
                    key = (model, trial)
                    if key not in retrain_index:
                        retrain_index[key] = len(retrains)
                        retrains.append(
                            {
                                "model": model,
                                "trial_number": trial,
                                "params": {},
                                "logical_objectives": [],
                                "seed": 123,
                                "epochs": 200,
                                "retrain_index": retrain_index[key],
                            }
                        )
                    retrains[retrain_index[key]]["logical_objectives"].append(objective_id)
                    membership = {
                        "membership_index": len(memberships),
                        "model": model,
                        "logical_objective": objective_id,
                        "score": score,
                        "metric": metric,
                        "strategy": strategy,
                        "trial_number": trial,
                        "objective_values": [0.2 + candidate_offset, 1.2 - candidate_offset],
                        "retrain_index": retrain_index[key],
                    }
                    memberships.append(membership)
                    strategy_index = (
                        CAP_STRATEGIES.index(strategy) if strategy else 5 + (metric == "drift")
                    )
                    score_bonus = 0.03 if score == "residual_oas" else 0.0
                    mapping_bonus = 0.01 * strategy_index
                    candidate_bonus = 0.04 if candidate_offset == 1 else 0.0
                    efficiencies = {
                        anomaly: min(
                            0.95,
                            0.2
                            + 0.01 * model_index
                            + score_bonus
                            + mapping_bonus
                            + candidate_bonus
                            + 0.001 * anomaly_index,
                        )
                        for anomaly_index, anomaly in enumerate(DOWNSTREAM_DATASETS)
                    }
                    checkpoint_family = ("last", "stable_native", "primary_ema")[
                        (model_index + strategy_index + candidate_offset) % 3
                    ]
                    checkpoint = {
                        "last": "last",
                        "stable_native": "single/ascore_operational/stable",
                        "primary_ema": (
                            f"summary/{objective_id}/"
                            f"{'max' if directions[0] == 'maximize' else 'min'}"
                        ),
                    }[checkpoint_family]
                    candidate = {
                        **membership,
                        "params": {},
                        "optimized_ckpt_name": checkpoint,
                        "retrained_objective_values": [0.3 + candidate_offset, 1.1],
                        "mean_downstream_efficiency": sum(efficiencies.values()) / 20,
                        "downstream_efficiencies": efficiencies,
                    }
                    candidates.append(candidate)
                    front_candidates.append(candidate)
                    checkpoint_rows.append(
                        {
                            "phase": "search",
                            "model": model,
                            "score": score,
                            "metric": metric,
                            "strategy": strategy or "",
                            "optimized_ckpt_name": checkpoint,
                            "checkpoint_family": checkpoint_family,
                        }
                    )
                    checkpoint_rows.append(
                        {
                            "phase": "retrain",
                            "model": model,
                            "retrain_index": retrain_index[key],
                            "logical_objective": objective_id,
                            "score": score,
                            "metric": metric,
                            "strategy": strategy or "",
                            "optimized_ckpt_name": checkpoint,
                            "checkpoint_family": checkpoint_family,
                        }
                    )
                winner = max(front_candidates, key=lambda row: row["mean_downstream_efficiency"])
                winners.append(winner)
                if metric == "drift":
                    primary_rows.extend(
                        (
                            {
                                "model": model,
                                "score": score,
                                "metric_name": "drift_test",
                                "value": 0.2 + 0.01 * model_index,
                            },
                            {
                                "model": model,
                                "score": score,
                                "metric_name": "reverse_drift_posthoc",
                                "value": 0.3 + 0.01 * model_index,
                            },
                        )
                    )
        studies.append(
            {
                "model": model,
                "study_name": f"study_{model}",
                "objectives": objectives,
            }
        )
    for row in retrains:
        row["logical_objectives"] = sorted(set(row["logical_objectives"]))
    design = {"models": list(MODEL_ORDER), "studies": studies}
    manifest = {"front_memberships": memberships, "retrain_runs": retrains}
    oracle = {"all_front_memberships": candidates, "winners": winners}
    checkpoint_rows.append(
        {
            "phase": "retrain",
            "model": "ae",
            "retrain_index": retrain_index[("ae", 0)],
            "logical_objective": "extra_nonmembership_objective",
            "score": "native",
            "metric": "cap",
            "strategy": "cdf",
            "optimized_ckpt_name": "last",
            "checkpoint_family": "last",
        }
    )
    return design, manifest, oracle, checkpoint_rows, primary_rows


def _analysis():
    """Run the analysis over the complete synthetic campaign."""
    return build_physics_background_pairing_analysis(*_fixture())


def test_analysis_has_complete_stable_row_sets() -> None:
    """All specified comparison tables have exact deterministic coverage."""
    result = _analysis()

    assert len(result["front_candidates"]) == 112
    assert len(result["front_summaries"]) == 56
    assert len(result["mapping_comparisons"]) == 32
    assert len(result["mapping_anomaly_deltas"]) == 640
    assert len(result["metric_methods"]) == 56
    assert len(result["metric_contrasts"]) == 88
    assert len(result["metric_anomaly_deltas"]) == 1760
    assert len(result["score_comparisons"]) == 14
    assert len(result["score_anomaly_deltas"]) == 280
    assert len(result["model_native_summary"]) == 6
    assert len(result["model_anomaly_performance"]) == 120
    assert len(result["proxy_associations"]) == 224
    assert len(result["drift_directionality"]) == 8
    assert result["analysis_summary"]["unique_retrains"] == 12
    assert result == _analysis()


def test_mapping_score_and_metric_deltas_have_known_signs() -> None:
    """Matched comparisons preserve the fixture's known effect directions."""
    result = _analysis()
    flat = result["mapping_comparisons"][0]

    assert flat["model"] == "ae"
    assert flat["score"] == "native"
    assert flat["mapping_strategy"] == "flat_physical"
    assert flat["front_intersection"] == 2
    assert flat["front_union"] == 2
    assert flat["front_jaccard"] == pytest.approx(1.0)
    assert flat["delta_mean"] == pytest.approx(-0.04)
    assert flat["anomalies_worse"] == 20
    assert result["mapping_summary"][0]["groups_worse"] == 8
    assert all(row["delta_mean"] == pytest.approx(0.03) for row in result["score_comparisons"])
    assert result["metric_contrasts"][0]["left_method"] == "cap__flat_physical"
    assert result["metric_contrasts"][0]["right_method"] == "wasserstein"
    assert result["metric_contrasts"][0]["delta_mean"] == pytest.approx(-0.05)


def test_model_comparison_is_native_only_and_anomaly_complete() -> None:
    """Six-model summaries remain native-only and cover every anomaly."""
    result = _analysis()

    assert {row["model"] for row in result["model_native_summary"]} == set(MODEL_ORDER)
    assert {row["fronts"] for row in result["model_native_summary"]} == {7}
    for model in MODEL_ORDER:
        rows = [row for row in result["model_anomaly_performance"] if row["model"] == model]
        assert len(rows) == 20
        assert {row["anomaly"] for row in rows} == set(DOWNSTREAM_DATASETS)


def test_proxy_associations_are_direction_aware_and_report_ties() -> None:
    """Proxy correlations respect objective direction and expose constants."""
    result = _analysis()
    rows = result["proxy_associations"][:4]

    assert rows[0]["source"] == "search_q_prime"
    assert rows[0]["spearman_rho"] == pytest.approx(1.0)
    assert rows[1]["source"] == "search_q_double_prime"
    assert rows[1]["spearman_rho"] == pytest.approx(1.0)
    assert rows[2]["spearman_rho"] == pytest.approx(1.0)
    assert rows[3]["constant"] is True
    assert rows[3]["spearman_rho"] is None


def test_checkpoint_families_and_directional_drift_are_complete() -> None:
    """Checkpoint and directional-drift outputs cover their populations."""
    result = _analysis()
    populations = {row["population"] for row in result["checkpoint_family_summary"]}

    assert populations == {"search", "retrain_candidate", "winner"}
    assert {row["family"] for row in result["checkpoint_family_summary"]} == {
        "last",
        "primary_ema",
        "stable_native",
    }
    assert all(
        row["reverse_minus_forward"] == pytest.approx(0.1)
        for row in result["drift_directionality"]
    )


def test_analysis_rejects_incomplete_or_nonfinite_anomaly_data() -> None:
    """Incomplete and nonfinite downstream records fail closed."""
    inputs = list(_fixture())
    broken = copy.deepcopy(inputs[2])
    broken["all_front_memberships"][0]["downstream_efficiencies"].pop(DOWNSTREAM_DATASETS[0])
    inputs[2] = broken
    with pytest.raises(ValueError, match="exact 20-anomaly"):
        build_physics_background_pairing_analysis(*inputs)

    inputs = list(_fixture())
    broken = copy.deepcopy(inputs[2])
    broken["all_front_memberships"][0]["downstream_efficiencies"][DOWNSTREAM_DATASETS[0]] = float(
        "nan"
    )
    inputs[2] = broken
    with pytest.raises(ValueError, match="Nonfinite"):
        build_physics_background_pairing_analysis(*inputs)


def test_analysis_rejects_unauthenticated_checkpoint_identity() -> None:
    """Unknown checkpoint names cannot be silently classified as primary EMA."""
    inputs = list(_fixture())
    broken = copy.deepcopy(inputs[3])
    broken[0]["optimized_ckpt_name"] = "unknown"
    inputs[3] = broken
    with pytest.raises(ValueError, match="not the last checkpoint"):
        build_physics_background_pairing_analysis(*inputs)
