from unittest.mock import Mock

import numpy as np
import pytest

import src.evaluation.leakage_probe.mlp as leakage_probe
from src.evaluation.leakage_probe import (
    AllMLPProbeCandidatesFailed,
    MLPProbeCandidateResult,
    PROBE_INITIALIZATION_SEEDS,
    ProbeFitError,
    make_probe_inner_partition,
    select_mlp_probe_seed,
)


def make_candidate(
    seed: int,
    inner_r2_raw: float,
) -> MLPProbeCandidateResult:
    return MLPProbeCandidateResult(
        seed=seed,
        inner_r2_raw=inner_r2_raw,
        inner_mae_gev=10.0,
        convergence_warnings=(),
        n_iter=5,
        final_loss=0.1,
        feature_scaler=Mock(),
        target_scaler=Mock(),
        estimator=Mock(),
    )


def make_data() -> tuple[np.ndarray, np.ndarray]:
    features = np.arange(
        60,
        dtype=np.float64,
    ).reshape(20, 3)

    target = np.linspace(
        80.0,
        180.0,
        num=20,
    )

    return features, target


def test_all_frozen_seeds_are_fitted_in_order(
    monkeypatch,
) -> None:
    features, target = make_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    calls = []

    scores = {
        10: 0.2,
        123: 0.7,
        500: 0.4,
    }

    def fake_fit(
        received_features,
        received_target,
        received_partition,
        *,
        seed,
    ):
        assert received_features is features
        assert received_target is target
        assert received_partition is partition

        calls.append(seed)
        return make_candidate(seed, scores[seed])

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe_candidate",
        fake_fit,
    )

    selection = select_mlp_probe_seed(
        features,
        target,
        partition,
    )

    assert calls == list(PROBE_INITIALIZATION_SEEDS)
    assert selection.selected_seed == 123
    assert selection.selected_candidate.seed == 123

    assert [
        candidate.seed
        for candidate in selection.successful_candidates
    ] == list(PROBE_INITIALIZATION_SEEDS)

    assert selection.failed_candidates == ()


def test_seed_selection_uses_highest_raw_r2(
    monkeypatch,
) -> None:
    features, target = make_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    scores = {
        10: -0.8,
        123: -0.2,
        500: -0.5,
    }

    def fake_fit(
        features,
        target,
        partition,
        *,
        seed,
    ):
        return make_candidate(seed, scores[seed])

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe_candidate",
        fake_fit,
    )

    selection = select_mlp_probe_seed(
        features,
        target,
        partition,
    )

    # -0.2 is the highest raw R2. Negative scores must not be
    # clipped to zero before seed selection.
    assert selection.selected_seed == 123
    assert (
        selection.selected_candidate.inner_r2_raw
        == pytest.approx(-0.2)
    )


def test_exact_tie_selects_first_seed_in_frozen_order(
    monkeypatch,
) -> None:
    features, target = make_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    scores = {
        10: 0.5,
        123: 0.5,
        500: 0.5,
    }

    def fake_fit(
        features,
        target,
        partition,
        *,
        seed,
    ):
        return make_candidate(seed, scores[seed])

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe_candidate",
        fake_fit,
    )

    selection = select_mlp_probe_seed(
        features,
        target,
        partition,
    )

    assert selection.selected_seed == 10


def test_failed_candidate_is_recorded_and_others_continue(
    monkeypatch,
) -> None:
    features, target = make_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    def fake_fit(
        features,
        target,
        partition,
        *,
        seed,
    ):
        if seed == 123:
            raise ProbeFitError(
                "mlp_fit_failed",
                "Synthetic failure for seed 123.",
            )

        score = {
            10: 0.3,
            500: 0.8,
        }[seed]

        return make_candidate(seed, score)

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe_candidate",
        fake_fit,
    )

    selection = select_mlp_probe_seed(
        features,
        target,
        partition,
    )

    assert selection.selected_seed == 500

    assert [
        candidate.seed
        for candidate in selection.successful_candidates
    ] == [10, 500]

    assert len(selection.failed_candidates) == 1

    failure = selection.failed_candidates[0]
    assert failure.seed == 123
    assert failure.reason == "mlp_fit_failed"
    assert "Synthetic failure" in failure.message


def test_all_candidate_failures_invalidate_selection(
    monkeypatch,
) -> None:
    features, target = make_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    def fake_fit(
        features,
        target,
        partition,
        *,
        seed,
    ):
        raise ProbeFitError(
            "mlp_fit_failed",
            f"Synthetic failure for seed {seed}.",
        )

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe_candidate",
        fake_fit,
    )

    with pytest.raises(
        AllMLPProbeCandidatesFailed
    ) as error:
        select_mlp_probe_seed(
            features,
            target,
            partition,
        )

    assert (
        error.value.reason
        == "all_mlp_candidates_failed"
    )
    assert len(error.value.failed_candidates) == 3

    assert [
        failure.seed
        for failure in error.value.failed_candidates
    ] == list(PROBE_INITIALIZATION_SEEDS)


def test_non_finite_candidate_is_recorded_as_failure(
    monkeypatch,
) -> None:
    features, target = make_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    scores = {
        10: 0.3,
        123: np.nan,
        500: 0.6,
    }

    def fake_fit(
        features,
        target,
        partition,
        *,
        seed,
    ):
        return make_candidate(seed, scores[seed])

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe_candidate",
        fake_fit,
    )

    selection = select_mlp_probe_seed(
        features,
        target,
        partition,
    )

    assert selection.selected_seed == 500
    assert len(selection.failed_candidates) == 1
    assert selection.failed_candidates[0].seed == 123
    assert (
        selection.failed_candidates[0].reason
        == "non_finite_inner_r2"
    )


def test_unexpected_programming_error_is_not_suppressed(
    monkeypatch,
) -> None:
    features, target = make_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    def fake_fit(
        features,
        target,
        partition,
        *,
        seed,
    ):
        raise AssertionError("synthetic programming error")

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe_candidate",
        fake_fit,
    )

    with pytest.raises(
        AssertionError,
        match="synthetic programming error",
    ):
        select_mlp_probe_seed(
            features,
            target,
            partition,
        )


def test_candidate_seed_mismatch_is_rejected(
    monkeypatch,
) -> None:
    features, target = make_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    def fake_fit(
        features,
        target,
        partition,
        *,
        seed,
    ):
        return make_candidate(999, 0.5)

    monkeypatch.setattr(
        leakage_probe,
        "fit_mlp_probe_candidate",
        fake_fit,
    )

    with pytest.raises(
        RuntimeError,
        match="different seed",
    ):
        select_mlp_probe_seed(
            features,
            target,
            partition,
        )
