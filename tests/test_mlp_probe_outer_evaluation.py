import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

import src.evaluation.leakage_probe.mlp as leakage_probe
from src.evaluation.leakage_probe import (
    MLP_PROBE_CONFIG,
    PROBE_INITIALIZATION_SEED,
    ProbeFitError,
    fit_mlp_probe,
)


class RecordingRegressor:
    """Fast estimator double for checking probe-fit behavior."""

    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)

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
        self.n_iter_ = 7
        self.loss_ = 0.02
        self.loss_curve_ = [0.8, 0.5, 0.3, 0.2, 0.1, 0.04, 0.02]
        self.validation_scores_ = [-0.2, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7]

        return self

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return features[:, 0]


class MeanPredictionRegressor(RecordingRegressor):
    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        # Zero in standardized target space corresponds to the
        # complete training-target mean.
        return np.zeros(
            features.shape[0],
            dtype=np.float64,
        )


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
        raise RuntimeError("synthetic fit failure")


def make_data():
    train_feature_zero = np.arange(
        20,
        dtype=np.float64,
    )
    validation_feature_zero = np.arange(
        20,
        26,
        dtype=np.float64,
    )

    train_features = np.column_stack(
        [
            train_feature_zero,
            train_feature_zero**2,
        ]
    )
    validation_features = np.column_stack(
        [
            validation_feature_zero,
            validation_feature_zero**2,
        ]
    )

    train_target = (
        100.0 + 10.0 * train_feature_zero
    )
    validation_target = (
        100.0 + 10.0 * validation_feature_zero
    )

    return (
        train_features,
        train_target,
        validation_features,
        validation_target,
    )


def test_fit_creates_fresh_scalers_and_estimator(
    monkeypatch,
) -> None:
    RecordingRegressor.instances.clear()

    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        RecordingRegressor,
    )

    result = fit_mlp_probe(*make_data())

    assert len(RecordingRegressor.instances) == 1
    assert result.estimator is RecordingRegressor.instances[0]
    assert result.feature_scaler is not result.target_scaler

    assert result.estimator.kwargs == {
        **dict(MLP_PROBE_CONFIG),
        "random_state": 123,
        "verbose": True,
    }


def test_scalers_use_complete_development_pool_only(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        RecordingRegressor,
    )

    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_data()

    result = fit_mlp_probe(
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    np.testing.assert_allclose(
        result.feature_scaler.mean_,
        train_features.mean(axis=0),
    )
    np.testing.assert_allclose(
        result.target_scaler.mean_,
        np.array([train_target.mean()]),
    )

    np.testing.assert_allclose(
        result.estimator.fit_features.mean(axis=0),
        np.zeros(train_features.shape[1]),
        atol=1e-12,
    )
    assert result.estimator.fit_target.mean() == pytest.approx(
        0.0,
        abs=1e-12,
    )

    combined_feature_mean = np.concatenate(
        [train_features, validation_features],
        axis=0,
    ).mean(axis=0)

    assert not np.allclose(
        result.feature_scaler.mean_,
        combined_feature_mean,
    )


def test_outer_metrics_are_reported_in_physical_units(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        RecordingRegressor,
    )

    result = fit_mlp_probe(*make_data())

    assert result.seed == PROBE_INITIALIZATION_SEED
    assert result.outer_r2_raw == pytest.approx(1.0)
    assert result.outer_r2_clipped == pytest.approx(1.0)
    assert result.outer_mae_gev == pytest.approx(
        0.0,
        abs=1e-10,
    )
    assert result.n_train == 20
    assert result.n_validation == 6
    assert result.n_iter == 7
    assert result.final_loss == pytest.approx(0.02)
    assert result.loss_curve == tuple(result.estimator.loss_curve_)
    assert result.early_stopping_validation_scores == tuple(result.estimator.validation_scores_)


def test_negative_outer_r2_is_preserved_and_clipped(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        MeanPredictionRegressor,
    )

    result = fit_mlp_probe(*make_data())

    assert result.outer_r2_raw < 0.0
    assert result.outer_r2_clipped == 0.0
    assert result.outer_mae_gev > 0.0


def test_convergence_warning_is_recorded(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        WarningRegressor,
    )

    result = fit_mlp_probe(*make_data())

    assert len(result.convergence_warnings) == 1
    assert (
        "Maximum iterations reached"
        in result.convergence_warnings[0]
    )


def test_fit_failure_is_wrapped(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        FailingRegressor,
    )

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe(*make_data())

    assert error.value.reason == "mlp_fit_failed"


def test_train_validation_feature_dimension_mismatch_is_rejected() -> None:
    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_data()

    validation_features = validation_features[:, :1]

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe(
            train_features,
            train_target,
            validation_features,
            validation_target,
        )

    assert (
        error.value.reason
        == "probe_feature_dimension_mismatch"
    )


def test_constant_outer_validation_target_is_rejected() -> None:
    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_data()

    validation_target = np.ones_like(
        validation_target
    )

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe(
            train_features,
            train_target,
            validation_features,
            validation_target,
        )

    assert (
        error.value.reason
        == "constant_outer_validation_target"
    )


@pytest.mark.parametrize(
    "seed",
    [0, 10, 42, 500, 999],
)
def test_unregistered_probe_seed_is_rejected(seed: int) -> None:
    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe(*make_data(), seed=seed)

    assert error.value.reason == "invalid_probe_seed"
    assert seed != PROBE_INITIALIZATION_SEED


def test_feature_target_row_mismatch_is_rejected() -> None:
    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_data()

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe(
            train_features,
            train_target[:-1],
            validation_features,
            validation_target,
        )

    assert (
        error.value.reason
        == "full_train_feature_target_row_mismatch"
    )


def test_non_finite_features_are_rejected() -> None:
    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_data()

    train_features = train_features.copy()
    train_features[0, 0] = np.nan

    with pytest.raises(ProbeFitError) as error:
        fit_mlp_probe(
            train_features,
            train_target,
            validation_features,
            validation_target,
        )

    assert (
        error.value.reason
        == "non_finite_full_train_features"
    )


def test_real_fit_smoke_test() -> None:
    random_state = np.random.RandomState(17)

    train_features = random_state.normal(
        size=(120, 4)
    )
    validation_features = random_state.normal(
        size=(40, 4)
    )

    train_target = (
        100.0
        + 8.0 * train_features[:, 0]
        - 3.0 * train_features[:, 1]
        + random_state.normal(
            scale=0.5,
            size=train_features.shape[0],
        )
    )
    validation_target = (
        100.0
        + 8.0 * validation_features[:, 0]
        - 3.0 * validation_features[:, 1]
        + random_state.normal(
            scale=0.5,
            size=validation_features.shape[0],
        )
    )

    result = fit_mlp_probe(
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    assert np.isfinite(result.outer_r2_raw)
    assert 0.0 <= result.outer_r2_clipped
    assert np.isfinite(result.outer_mae_gev)
    assert np.isfinite(result.final_loss)
    assert result.n_iter > 0
