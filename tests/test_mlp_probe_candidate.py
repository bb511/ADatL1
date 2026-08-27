import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

import src.evaluation.leakage_probe as leakage_probe
from src.evaluation.leakage_probe import (
    MLP_PROBE_CONFIG,
    PROBE_INITIALIZATION_SEEDS,
    ProbeFitError,
    fit_mlp_probe_candidate,
    make_probe_inner_partition,
)


class RecordingRegressor:
    """Fast estimator double used to inspect preprocessing and config."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ):
        self.fit_features = features.copy()
        self.fit_target = target.copy()

        self.coefs_ = [
            np.ones(
                (features.shape[1], 1),
                dtype=np.float64,
            )
        ]
        self.intercepts_ = [
            np.zeros(1, dtype=np.float64)
        ]
        self.n_iter_ = 5
        self.loss_ = 0.01

        return self

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        # The synthetic target is an affine transformation of feature 0.
        # After separate standardization, both have identical values.
        return features[:, 0]


class WarningRegressor(RecordingRegressor):
    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ):
        result = super().fit(features, target)
        warnings.warn(
            "Maximum iterations reached.",
            ConvergenceWarning,
        )
        return result


class FailingRegressor(RecordingRegressor):
    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ):
        raise RuntimeError("synthetic optimizer failure")


def make_linear_data(
    n_events: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    feature_zero = np.arange(
        n_events,
        dtype=np.float64,
    )

    features = np.column_stack(
        [
            feature_zero,
            feature_zero**2,
            np.sin(feature_zero),
        ]
    )

    target = 100.0 + 10.0 * feature_zero

    return features, target


def test_candidate_uses_frozen_mlp_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        RecordingRegressor,
    )

    features, target = make_linear_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    result = fit_mlp_probe_candidate(
        features,
        target,
        partition,
        seed=10,
    )

    assert result.seed == 10
    assert result.estimator.kwargs == {
        **dict(MLP_PROBE_CONFIG),
        "random_state": 10,
    }
    assert (
        result.estimator.kwargs["hidden_layer_sizes"]
        == (64, 32)
    )
    assert result.estimator.kwargs["early_stopping"] is True
    assert result.estimator.kwargs["max_iter"] == 500


def test_scalers_are_fitted_only_on_probe_fit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        RecordingRegressor,
    )

    features, target = make_linear_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    result = fit_mlp_probe_candidate(
        features,
        target,
        partition,
        seed=123,
    )

    expected_feature_mean = features[
        partition.fit_indices
    ].mean(axis=0)

    expected_target_mean = target[
        partition.fit_indices
    ].mean()

    np.testing.assert_allclose(
        result.feature_scaler.mean_,
        expected_feature_mean,
    )
    np.testing.assert_allclose(
        result.target_scaler.mean_,
        np.array([expected_target_mean]),
    )

    # These assertions would normally fail if validation data had been
    # included while fitting either scaler.
    np.testing.assert_allclose(
        result.estimator.fit_features.mean(axis=0),
        np.zeros(features.shape[1]),
        atol=1e-12,
    )
    assert result.estimator.fit_target.mean() == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_candidate_scores_predictions_in_physical_units(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        RecordingRegressor,
    )

    features, target = make_linear_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    result = fit_mlp_probe_candidate(
        features,
        target,
        partition,
        seed=500,
    )

    assert result.inner_r2_raw == pytest.approx(1.0)
    assert result.inner_mae_gev == pytest.approx(
        0.0,
        abs=1e-10,
    )
    assert result.n_iter == 5
    assert result.final_loss == pytest.approx(0.01)


def test_convergence_warning_is_recorded_but_not_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        WarningRegressor,
    )

    features, target = make_linear_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    result = fit_mlp_probe_candidate(
        features,
        target,
        partition,
        seed=10,
    )

    assert len(result.convergence_warnings) == 1
    assert (
        "Maximum iterations reached"
        in result.convergence_warnings[0]
    )


def test_candidate_fit_exception_is_wrapped(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        FailingRegressor,
    )

    features, target = make_linear_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe_candidate(
            features,
            target,
            partition,
            seed=10,
        )

    assert error.value.reason == "mlp_fit_failed"


@pytest.mark.parametrize(
    "seed",
    [0, 42, 999],
)
def test_unregistered_probe_seed_is_rejected(
    seed: int,
) -> None:
    features, target = make_linear_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe_candidate(
            features,
            target,
            partition,
            seed=seed,
        )

    assert error.value.reason == "invalid_probe_seed"
    assert seed not in PROBE_INITIALIZATION_SEEDS


def test_feature_target_row_mismatch_is_rejected() -> None:
    features, target = make_linear_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe_candidate(
            features,
            target[:-1],
            partition,
            seed=10,
        )

    assert (
        error.value.reason
        == "probe_feature_target_row_mismatch"
    )


def test_non_finite_features_are_rejected() -> None:
    features, target = make_linear_data()
    features[0, 0] = np.nan

    partition = make_probe_inner_partition(
        features.shape[0]
    )

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe_candidate(
            features,
            target,
            partition,
            seed=10,
        )

    assert error.value.reason == "non_finite_probe_features"


def test_constant_inner_validation_target_is_rejected() -> None:
    features, target = make_linear_data()
    partition = make_probe_inner_partition(
        features.shape[0]
    )

    target[partition.validation_indices] = 100.0

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe_candidate(
            features,
            target,
            partition,
            seed=10,
        )

    assert (
        error.value.reason
        == "constant_probe_inner_validation_target"
    )


def test_real_mlp_candidate_smoke_test() -> None:
    random_state = np.random.RandomState(7)

    features = random_state.normal(
        size=(120, 4)
    )
    target = (
        100.0
        + 12.0 * features[:, 0]
        - 4.0 * features[:, 1]
        + random_state.normal(
            scale=0.5,
            size=features.shape[0],
        )
    )

    partition = make_probe_inner_partition(
        features.shape[0]
    )

    result = fit_mlp_probe_candidate(
        features,
        target,
        partition,
        seed=10,
    )

    assert np.isfinite(result.inner_r2_raw)
    assert np.isfinite(result.inner_mae_gev)
    assert np.isfinite(result.final_loss)
    assert result.n_iter > 0