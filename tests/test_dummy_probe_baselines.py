import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.evaluation.leakage_probe import (
    PrimaryDummyBaselineResult,
    ProbeFitError,
    ProbeRepresentationSet,
    evaluate_dummy_baseline_representation,
    evaluate_primary_dummy_baselines,
    fit_dummy_baseline,
)


def make_representation_set(
    split: str,
    target: np.ndarray,
) -> ProbeRepresentationSet:
    target = np.asarray(
        target,
        dtype=np.float64,
    )

    latent_logits = np.column_stack(
        [
            np.arange(target.size, dtype=np.float64),
            np.linspace(0.0, 1.0, target.size),
        ]
    )

    reconstructed_data = np.column_stack(
        [
            np.linspace(-1.0, 1.0, target.size),
            np.arange(target.size, dtype=np.float64) ** 2,
            np.ones(target.size, dtype=np.float64),
        ]
    )

    return ProbeRepresentationSet(
        split=split,
        latent_logits=latent_logits,
        latent_sample=(
            latent_logits > 0.5
        ).astype(np.float64),
        reconstructed_data=reconstructed_data,
        sensitive_target=target,
        n_events=int(target.size),
        sample_seed=12345,
        max_samples=None,
        manifest_hash=f"{split}-manifest",
    )


def test_dummy_baseline_uses_training_target_mean() -> None:
    train_features = np.arange(
        12,
        dtype=np.float64,
    ).reshape(6, 2)

    validation_features = np.arange(
        8,
        dtype=np.float64,
    ).reshape(4, 2)

    train_target = np.asarray(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        dtype=np.float64,
    )
    validation_target = np.asarray(
        [70.0, 80.0, 90.0, 100.0],
        dtype=np.float64,
    )

    result = fit_dummy_baseline(
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    expected_mean = float(train_target.mean())
    expected_predictions = np.full(
        validation_target.shape,
        expected_mean,
        dtype=np.float64,
    )

    assert isinstance(result.estimator, DummyRegressor)
    assert result.estimator.strategy == "mean"
    assert result.train_mean_gev == pytest.approx(
        expected_mean
    )

    np.testing.assert_allclose(
        np.asarray(
            result.estimator.constant_
        ).reshape(-1),
        [expected_mean],
    )

    assert result.outer_r2_raw == pytest.approx(
        r2_score(
            validation_target,
            expected_predictions,
        )
    )
    assert result.outer_r2_clipped == pytest.approx(
        max(
            0.0,
            r2_score(
                validation_target,
                expected_predictions,
            ),
        )
    )
    assert result.outer_mae_gev == pytest.approx(
        mean_absolute_error(
            validation_target,
            expected_predictions,
        )
    )
    assert result.n_train == 6
    assert result.n_validation == 4

    # The validation target must not affect the fitted constant.
    combined_mean = float(
        np.concatenate(
            [train_target, validation_target]
        ).mean()
    )
    assert result.train_mean_gev != pytest.approx(
        combined_mean
    )


def test_negative_dummy_r2_is_preserved_and_clipped() -> None:
    train_features = np.arange(
        12,
        dtype=np.float64,
    ).reshape(6, 2)
    validation_features = np.arange(
        8,
        dtype=np.float64,
    ).reshape(4, 2)

    train_target = np.asarray(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        dtype=np.float64,
    )
    validation_target = np.asarray(
        [100.0, 101.0, 102.0, 103.0],
        dtype=np.float64,
    )

    result = fit_dummy_baseline(
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    assert result.outer_r2_raw < 0.0
    assert result.outer_r2_clipped == 0.0


def test_dummy_feature_dimension_mismatch_is_rejected() -> None:
    train_features = np.ones(
        (6, 2),
        dtype=np.float64,
    )
    validation_features = np.ones(
        (4, 3),
        dtype=np.float64,
    )
    train_target = np.arange(
        6,
        dtype=np.float64,
    )
    validation_target = np.arange(
        4,
        dtype=np.float64,
    )

    with pytest.raises(ProbeFitError) as error:
        fit_dummy_baseline(
            train_features,
            train_target,
            validation_features,
            validation_target,
        )

    assert (
        error.value.reason
        == "dummy_feature_dimension_mismatch"
    )


def test_named_dummy_baseline_uses_requested_representation() -> None:
    train = make_representation_set(
        "train",
        np.linspace(50.0, 150.0, 20),
    )
    validation = make_representation_set(
        "valid",
        np.linspace(60.0, 140.0, 10),
    )

    result = evaluate_dummy_baseline_representation(
        train,
        validation,
        representation_name="latent_logits",
    )

    assert result.representation_name == "latent_logits"
    assert result.metric_name == "z_logits"
    assert result.feature_dimension == 2


def test_latent_sample_is_not_a_primary_dummy_baseline() -> None:
    train = make_representation_set(
        "train",
        np.linspace(50.0, 150.0, 20),
    )
    validation = make_representation_set(
        "valid",
        np.linspace(60.0, 140.0, 10),
    )

    with pytest.raises(ProbeFitError) as error:
        evaluate_dummy_baseline_representation(
            train,
            validation,
            representation_name="latent_sample",
        )

    assert (
        error.value.reason
        == "unknown_dummy_baseline_representation"
    )


def test_primary_dummy_baselines_use_independent_estimators() -> None:
    train = make_representation_set(
        "train",
        np.linspace(50.0, 150.0, 20),
    )
    validation = make_representation_set(
        "valid",
        np.linspace(60.0, 140.0, 10),
    )

    result = evaluate_primary_dummy_baselines(
        train,
        validation,
    )

    assert isinstance(
        result,
        PrimaryDummyBaselineResult,
    )

    latent = result.latent_logits
    reconstruction = result.reconstructed_data

    assert latent.representation_name == "latent_logits"
    assert (
        reconstruction.representation_name
        == "reconstructed_data"
    )

    assert (
        latent.outer_result.estimator
        is not reconstruction.outer_result.estimator
    )

    # DummyRegressor ignores features, so both independently fitted
    # controls must produce identical metrics for the same target.
    assert (
        latent.outer_result.outer_r2_raw
        == pytest.approx(
            reconstruction.outer_result.outer_r2_raw
        )
    )
    assert (
        latent.outer_result.outer_mae_gev
        == pytest.approx(
            reconstruction.outer_result.outer_mae_gev
        )
    )


@pytest.mark.parametrize(
    ("train_split", "validation_split", "reason"),
    [
        (
            "valid",
            "valid",
            "invalid_dummy_baseline_training_split",
        ),
        (
            "train",
            "test",
            "invalid_dummy_baseline_outer_split",
        ),
    ],
)
def test_primary_dummy_baseline_split_protocol(
    train_split: str,
    validation_split: str,
    reason: str,
) -> None:
    train = make_representation_set(
        train_split,
        np.linspace(50.0, 150.0, 20),
    )
    validation = make_representation_set(
        validation_split,
        np.linspace(60.0, 140.0, 10),
    )

    with pytest.raises(ProbeFitError) as error:
        evaluate_primary_dummy_baselines(
            train,
            validation,
        )

    assert error.value.reason == reason