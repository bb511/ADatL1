"""Focused tests for the matched-marginal Causal Chamber theorem bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import cchamber_theorem_bridge as bridge


def test_common_quantile_calibration_matches_marginals_exactly() -> None:
    """Different continuous marginals become one identical quantile multiset."""
    rng = np.random.default_rng(11)
    first = rng.lognormal(size=500)
    second = rng.standard_t(df=4, size=500)
    calibrated_first = bridge.common_quantile_calibration(first)
    calibrated_second = bridge.common_quantile_calibration(second)

    np.testing.assert_array_equal(
        np.sort(calibrated_first),
        np.sort(calibrated_second),
    )
    assert np.array_equal(np.argsort(first), np.argsort(calibrated_first))
    assert np.array_equal(np.argsort(second), np.argsort(calibrated_second))


def test_matched_marginal_metrics_leave_only_paired_cap_discriminative() -> None:
    """Wasserstein and drift tie despite a large paired-dependence difference."""
    rng = np.random.default_rng(22)
    latent = rng.normal(size=1000)
    high_first = latent + 0.1 * rng.normal(size=1000)
    high_second = latent + 0.1 * rng.normal(size=1000)
    low_first = rng.lognormal(size=1000)
    low_second = rng.standard_t(df=3, size=1000)

    high = bridge.matched_marginal_metrics(high_first, high_second)
    low = bridge.matched_marginal_metrics(low_first, low_second)

    assert high["wasserstein"] == low["wasserstein"] == 0.0
    assert high["threshold_drift"] == low["threshold_drift"]
    assert high["cap"] > low["cap"] + 0.2
    assert high["paired_spearman"] > 0.95


def test_common_quantile_calibration_handles_rare_atoms_without_reordering() -> None:
    """The distributional transform only randomizes observations tied exactly."""
    scores = np.asarray([0.0, 1.0, 1.0, 2.0] * 30)
    calibrated = bridge.common_quantile_calibration(scores, tie_seed=7)

    assert np.max(calibrated[scores == 0.0]) < np.min(calibrated[scores == 1.0])
    assert np.max(calibrated[scores == 1.0]) < np.min(calibrated[scores == 2.0])
    assert np.unique(calibrated).size == len(scores)


def test_candidate_summary_uses_seed_first_cap_ranks() -> None:
    """Candidate aggregation keeps model seeds paired before averaging."""
    score_rows = []
    outcome_rows = []
    trajectory = 0
    for candidate, cap_values, outcomes in (
        ("a", (0.1, 0.9), (0.4, 0.8)),
        ("b", (0.2, 0.8), (0.5, 0.7)),
    ):
        for seed_index, seed in enumerate((1001, 1002)):
            score_rows.append(
                {
                    "trajectory_index": trajectory,
                    "model": "realnvp",
                    "candidate_id": candidate,
                    "reporting_seed": seed,
                    "validation_cap": cap_values[seed_index],
                    "test_cap": cap_values[seed_index],
                    "validation_random_pair_cap": 0.0,
                    "test_random_pair_cap": 0.0,
                }
            )
            for physical_class in ("process_or_actuator", "measurement_chain"):
                outcome_rows.append(
                    {
                        "trajectory_index": trajectory,
                        "model": "realnvp",
                        "candidate_id": candidate,
                        "reporting_seed": seed,
                        "physical_class": physical_class,
                        "value": outcomes[seed_index],
                    }
                )
            trajectory += 1

    ranked, candidate = bridge.candidate_summary(
        pd.DataFrame(score_rows),
        pd.DataFrame(outcome_rows),
    )

    assert len(ranked) == 4
    assert len(candidate) == 2
    assert candidate["validation_cap_mean_rank"].eq(0.5).all()
    assert np.allclose(candidate["all_auprc"], 0.6)
