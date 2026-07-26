"""Focused tests for the empirical Causal Chamber alignment bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import cchamber_empirical_alignment_bridge as bridge


def test_feature_audit_removes_only_normal_degenerate_dimensions() -> None:
    """The frozen normal-only rule rejects constant or low-cardinality features."""
    rows = 100
    reference = pd.DataFrame(
        {
            feature: (np.zeros(rows) if index == 0 else np.arange(rows, dtype=float) + index)
            for index, feature in enumerate(bridge.READOUT_FEATURES)
        }
    )
    contract = {
        "center": [0.0] * len(bridge.READOUT_FEATURES),
        "scale": [1.0e-6] + [1.0] * (len(bridge.READOUT_FEATURES) - 1),
    }
    audit, keep = bridge.feature_audit(reference, contract, bridge.SEED)

    assert not keep[0]
    assert keep[1:].all()
    assert audit.loc[0, "exclusion_reason"]
    assert audit.loc[1:, "exclusion_reason"].eq("").all()


def test_cap_direction_recovers_non_gaussian_paired_reliability() -> None:
    """Direct CAP optimization finds a shared Laplace direction without Gaussianization."""
    rng = np.random.default_rng(123)

    def pairs(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        shared = rng.laplace(size=n_rows)
        first = np.column_stack(
            [
                shared + 0.15 * rng.normal(size=n_rows),
                rng.laplace(size=n_rows),
                rng.standard_t(df=4, size=n_rows),
            ]
        )
        second = np.column_stack(
            [
                shared + 0.15 * rng.normal(size=n_rows),
                rng.laplace(size=n_rows),
                rng.standard_t(df=4, size=n_rows),
            ]
        )
        return first, second

    validation_1, validation_2 = pairs(240)
    test_1, test_2 = pairs(240)
    direction, _, summary = bridge.optimize_cap_direction(
        validation_1,
        validation_2,
        test_1,
        test_2,
        seed=123,
        n_fit_pairs=160,
        n_restarts=16,
        n_steps=150,
        learning_rate=0.04,
    )

    assert abs(direction[0]) > 0.9
    assert summary["test_cap"] > summary["test_random_pair_cap"] + 0.05
    assert summary["test_paired_spearman"] > 0.9


def test_shared_candidate_bank_applies_all_three_normal_only_selectors() -> None:
    """CAP, drift, and Wasserstein select exactly one score from one bank."""
    rng = np.random.default_rng(321)

    def paired(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        shared = rng.laplace(size=n_rows)
        return (
            np.column_stack([shared + 0.1 * rng.normal(size=n_rows), rng.standard_t(4, n_rows)]),
            np.column_stack([shared + 0.1 * rng.normal(size=n_rows), rng.standard_t(4, n_rows)]),
        )

    train = rng.standard_t(4, size=(400, 2))
    validation_1, validation_2 = paired(200)
    test_1, test_2 = paired(200)
    directions, candidates, summary = bridge.select_normal_score_directions(
        train,
        validation_1,
        validation_2,
        test_1,
        test_2,
        seed=321,
        n_candidates=128,
        n_drift_splits=4,
    )

    assert set(directions) == set(bridge.SELECTORS)
    assert set(summary["selected"]) == set(bridge.SELECTORS)
    assert all(candidates[f"selected_by_{selector}"].sum() == 1 for selector in bridge.SELECTORS)
    assert abs(directions["cap"][0]) > 0.8


def test_fixed_norm_alignment_sweep_separates_aligned_and_orthogonal() -> None:
    """A frozen direction detects aligned but not orthogonal fixed-norm shifts."""
    rng = np.random.default_rng(456)
    validation = rng.standard_t(df=5, size=(1_000, 3))
    normal = rng.standard_t(df=5, size=(1_000, 3))
    background = rng.standard_t(df=5, size=(400, 3))
    direction = np.asarray([1.0, 0.0, 0.0])
    sweep = bridge.synthetic_alignment_sweep(
        direction,
        validation,
        normal,
        background,
        shift_norm=3.0,
        angles=np.asarray([0.0, 90.0]),
    )
    aligned = sweep[sweep["angle_degrees"] == 0.0]
    orthogonal = sweep[sweep["angle_degrees"] == 90.0]

    assert aligned["selected_auprc"].mean() > orthogonal["selected_auprc"].mean() + 0.4
    assert aligned["selected_efficiency"].mean() > orthogonal["selected_efficiency"].mean() + 0.10
    assert orthogonal["oracle_auprc"].mean() > orthogonal["selected_auprc"].mean() + 0.4
