"""Focused tests for the exploratory Causal Chamber Gaussian bridge."""

from __future__ import annotations

import numpy as np

from scripts import cchamber_gaussian_bridge


def test_frozen_split_helpers_are_deterministic_and_disjoint() -> None:
    """Normal and signal helpers reproduce stable held-out splits."""
    train, validation, test = cchamber_gaussian_bridge._split_indices(100, 314159)
    repeated = cchamber_gaussian_bridge._split_indices(100, 314159)
    signal = cchamber_gaussian_bridge._signal_test_indices(100, 314159)

    assert all(
        np.array_equal(left, right) for left, right in zip(repeated, (train, validation, test))
    )
    assert (len(train), len(validation), len(test), len(signal)) == (60, 20, 20, 40)
    assert not (set(train) & set(validation))
    assert not (set(train) & set(test))
    assert not (set(validation) & set(test))


def test_directional_metrics_recover_an_aligned_gaussian_shift() -> None:
    """A mean shift is detected along its direction rather than orthogonally."""
    generator = np.random.default_rng(123)
    normal = generator.normal(size=(2_000, 2))
    signal = generator.normal(size=(800, 2))
    signal[:, 0] += 3.0
    weights = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    metrics = cchamber_gaussian_bridge._directional_signal_metrics(
        normal,
        signal,
        weights,
        0.01,
    )

    assert metrics["auprc"][0] > 0.9
    assert metrics["auprc"][0] > metrics["auprc"][1]
    assert metrics["efficiency"][0] > metrics["efficiency"][1]
    assert metrics["standardized_mean_shift"][0] > 2.5
    assert metrics["gaussian_tpr"][0] > metrics["gaussian_tpr"][1]
