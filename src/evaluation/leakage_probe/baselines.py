"""Target-only dummy baselines for leakage evaluation."""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .constants import (
    PRIMARY_PROBE_REPRESENTATIONS,
    PROBE_REPRESENTATION_METRIC_NAMES,
)
from .errors import ProbeFitError
from .mlp import _validate_probe_dataset
from .types import (
    DummyBaselineOuterResult,
    NamedDummyBaselineResult,
    PrimaryDummyBaselineResult,
    ProbeRepresentationSet,
)

def fit_dummy_baseline(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
) -> DummyBaselineOuterResult:
    """Fit a training-mean control and score outer validation."""

    train_features, train_target = _validate_probe_dataset(
        train_features,
        train_target,
        split_name="dummy_full_train",
    )
    validation_features, validation_target = (
        _validate_probe_dataset(
            validation_features,
            validation_target,
            split_name="dummy_outer_validation",
        )
    )

    if (
        train_features.shape[1]
        != validation_features.shape[1]
    ):
        raise ProbeFitError(
            "dummy_feature_dimension_mismatch",
            "Dummy-baseline training and validation feature "
            f"dimensions differ: {train_features.shape[1]} != "
            f"{validation_features.shape[1]}.",
        )

    estimator = DummyRegressor(
        strategy="mean"
    )

    try:
        estimator.fit(
            train_features,
            train_target,
        )
    except Exception as error:
        raise ProbeFitError(
            "dummy_baseline_fit_failed",
            f"Dummy baseline fitting failed: {error}",
        ) from error

    constant = np.asarray(
        estimator.constant_,
        dtype=np.float64,
    ).reshape(-1)

    if (
        constant.size != 1
        or not np.isfinite(constant).all()
    ):
        raise ProbeFitError(
            "invalid_dummy_baseline_constant",
            "The fitted dummy baseline has an invalid "
            "training-target mean.",
        )

    try:
        predictions_gev = np.asarray(
            estimator.predict(
                validation_features
            ),
            dtype=np.float64,
        ).reshape(-1)
    except Exception as error:
        raise ProbeFitError(
            "dummy_baseline_prediction_failed",
            f"Dummy baseline prediction failed: {error}",
        ) from error

    if (
        predictions_gev.shape[0]
        != validation_target.shape[0]
    ):
        raise ProbeFitError(
            "dummy_baseline_prediction_row_mismatch",
            "Dummy baseline predictions and validation "
            "targets have different event counts.",
        )

    if not np.isfinite(predictions_gev).all():
        raise ProbeFitError(
            "non_finite_dummy_baseline_predictions",
            "The dummy baseline produced non-finite predictions.",
        )

    outer_r2_raw = float(
        r2_score(
            validation_target,
            predictions_gev,
        )
    )
    outer_mae_gev = float(
        mean_absolute_error(
            validation_target,
            predictions_gev,
        )
    )

    if not np.isfinite(outer_r2_raw):
        raise ProbeFitError(
            "non_finite_dummy_baseline_r2",
            "The dummy baseline produced non-finite R2.",
        )

    if not np.isfinite(outer_mae_gev):
        raise ProbeFitError(
            "non_finite_dummy_baseline_mae",
            "The dummy baseline produced non-finite MAE.",
        )

    return DummyBaselineOuterResult(
        outer_r2_raw=outer_r2_raw,
        outer_r2_clipped=max(
            0.0,
            outer_r2_raw,
        ),
        outer_mae_gev=outer_mae_gev,
        train_mean_gev=float(constant[0]),
        n_train=int(train_features.shape[0]),
        n_validation=int(
            validation_features.shape[0]
        ),
        estimator=estimator,
    )

def evaluate_dummy_baseline_representation(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
    *,
    representation_name: str,
) -> NamedDummyBaselineResult:
    """Evaluate the target-mean control for one representation."""

    if representation_name not in PRIMARY_PROBE_REPRESENTATIONS:
        raise ProbeFitError(
            "unknown_dummy_baseline_representation",
            "Dummy baselines are defined only for "
            f"{PRIMARY_PROBE_REPRESENTATIONS}, got "
            f"{representation_name!r}.",
        )

    train_features = getattr(
        train_representations,
        representation_name,
    )
    validation_features = getattr(
        validation_representations,
        representation_name,
    )

    outer_result = fit_dummy_baseline(
        train_features,
        train_representations.sensitive_target,
        validation_features,
        validation_representations.sensitive_target,
    )

    return NamedDummyBaselineResult(
        representation_name=representation_name,
        metric_name=PROBE_REPRESENTATION_METRIC_NAMES[
            representation_name
        ],
        feature_dimension=int(
            train_features.shape[1]
        ),
        outer_result=outer_result,
    )

def evaluate_primary_dummy_baselines(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
) -> PrimaryDummyBaselineResult:
    """Evaluate separate target-mean controls for both inputs."""

    if train_representations.split != "train":
        raise ProbeFitError(
            "invalid_dummy_baseline_training_split",
            "Dummy baselines require the AE train split, "
            f"got {train_representations.split!r}.",
        )

    if validation_representations.split != "valid":
        raise ProbeFitError(
            "invalid_dummy_baseline_outer_split",
            "Dummy baselines require the held-out AE valid "
            f"split, got "
            f"{validation_representations.split!r}.",
        )

    latent_result = (
        evaluate_dummy_baseline_representation(
            train_representations,
            validation_representations,
            representation_name="latent_logits",
        )
    )

    reconstruction_result = (
        evaluate_dummy_baseline_representation(
            train_representations,
            validation_representations,
            representation_name="reconstructed_data",
        )
    )

    if (
        latent_result.outer_result.estimator
        is reconstruction_result.outer_result.estimator
    ):
        raise ProbeFitError(
            "dummy_baseline_state_shared",
            "Latent and reconstruction dummy baselines "
            "unexpectedly share one fitted estimator.",
        )

    return PrimaryDummyBaselineResult(
        latent_logits=latent_result,
        reconstructed_data=reconstruction_result,
    )

