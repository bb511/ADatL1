import warnings
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

import src.evaluation.leakage_probe.mlp as leakage_probe
from src.evaluation.leakage_probe import (
    MLP_PROBE_CONFIG,
    MLPProbeCandidateResult,
    MLPProbeSeedSelection,
    ProbeFitError,
    refit_selected_mlp_probe,
)


class RecordingRegressor:
    """Fast estimator double for checking refit behavior."""

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
        raise RuntimeError("synthetic refit failure")


def make_selection(
    seed: int = 123,
) -> MLPProbeSeedSelection:
    candidate = MLPProbeCandidateResult(
        seed=seed,
        inner_r2_raw=0.5,
        inner_mae_gev=5.0,
        convergence_warnings=(),
        n_iter=4,
        final_loss=0.1,
        feature_scaler=Mock(name="candidate_feature_scaler"),
        target_scaler=Mock(name="candidate_target_scaler"),
        estimator=Mock(name="candidate_estimator"),
    )

    return MLPProbeSeedSelection(
        selected_seed=seed,
        selected_candidate=candidate,
        successful_candidates=(candidate,),
        failed_candidates=(),
    )


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


def test_refit_creates_fresh_scalers_and_estimator(
    monkeypatch,
) -> None:
    RecordingRegressor.instances.clear()

    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        RecordingRegressor,
    )

    data = make_data()
    selection = make_selection(seed=123)

    result = refit_selected_mlp_probe(
        *data,
        selection,
    )

    assert len(RecordingRegressor.instances) == 1
    assert result.estimator is RecordingRegressor.instances[0]

    assert (
        result.estimator
        is not selection.selected_candidate.estimator
    )
    assert (
        result.feature_scaler
        is not selection.selected_candidate.feature_scaler
    )
    assert (
        result.target_scaler
        is not selection.selected_candidate.target_scaler
    )

    assert result.estimator.kwargs == {
        **dict(MLP_PROBE_CONFIG),
        "random_state": 123,
    }


def test_refit_scalers_use_complete_training_only(
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

    result = refit_selected_mlp_probe(
        train_features,
        train_target,
        validation_features,
        validation_target,
        make_selection(),
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

    result = refit_selected_mlp_probe(
        *make_data(),
        make_selection(seed=500),
    )

    assert result.selected_seed == 500
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


def test_negative_outer_r2_is_preserved_and_clipped(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        MeanPredictionRegressor,
    )

    result = refit_selected_mlp_probe(
        *make_data(),
        make_selection(),
    )

    assert result.outer_r2_raw < 0.0
    assert result.outer_r2_clipped == 0.0
    assert result.outer_mae_gev > 0.0


def test_refit_convergence_warning_is_recorded(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        WarningRegressor,
    )

    result = refit_selected_mlp_probe(
        *make_data(),
        make_selection(),
    )

    assert len(result.convergence_warnings) == 1
    assert (
        "Maximum iterations reached"
        in result.convergence_warnings[0]
    )


def test_refit_failure_is_wrapped(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        leakage_probe,
        "MLPRegressor",
        FailingRegressor,
    )

    with pytest.raises(ProbeFitError) as error:
        refit_selected_mlp_probe(
            *make_data(),
            make_selection(),
        )

    assert error.value.reason == "mlp_refit_failed"


def test_train_validation_feature_dimension_mismatch_is_rejected() -> None:
    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_data()

    validation_features = validation_features[:, :1]

    with pytest.raises(ProbeFitError) as error:
        refit_selected_mlp_probe(
            train_features,
            train_target,
            validation_features,
            validation_target,
            make_selection(),
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
        refit_selected_mlp_probe(
            train_features,
            train_target,
            validation_features,
            validation_target,
            make_selection(),
        )

    assert (
        error.value.reason
        == "constant_outer_validation_target"
    )


def test_selected_candidate_seed_mismatch_is_rejected() -> None:
    selection = make_selection(seed=10)

    inconsistent_selection = MLPProbeSeedSelection(
        selected_seed=123,
        selected_candidate=selection.selected_candidate,
        successful_candidates=(
            selection.selected_candidate,
        ),
        failed_candidates=(),
    )

    with pytest.raises(ProbeFitError) as error:
        refit_selected_mlp_probe(
            *make_data(),
            inconsistent_selection,
        )

    assert (
        error.value.reason
        == "selected_candidate_seed_mismatch"
    )


def test_real_refit_smoke_test() -> None:
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

    result = refit_selected_mlp_probe(
        train_features,
        train_target,
        validation_features,
        validation_target,
        make_selection(seed=10),
    )

    assert np.isfinite(result.outer_r2_raw)
    assert 0.0 <= result.outer_r2_clipped
    assert np.isfinite(result.outer_mae_gev)
    assert np.isfinite(result.final_loss)
    assert result.n_iter > 0
