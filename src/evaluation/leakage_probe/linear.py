"""Linear leakage probes for latent and reconstructed features."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .constants import (
    PRIMARY_PROBE_REPRESENTATIONS,
    PROBE_REPRESENTATION_METRIC_NAMES,
)
from .errors import ProbeFitError
from .mlp import _validate_probe_dataset
from .types import (
    LinearProbeOuterResult,
    NamedLinearProbeResult,
    PrimaryLinearProbeResult,
    ProbeRepresentationSet,
)

def fit_linear_probe(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
) -> LinearProbeOuterResult:
    """Fit a scaled linear probe on AE train and score AE validation."""

    train_features, train_target = _validate_probe_dataset(
        train_features,
        train_target,
        split_name="linear_full_train",
    )
    validation_features, validation_target = (
        _validate_probe_dataset(
            validation_features,
            validation_target,
            split_name="linear_outer_validation",
        )
    )

    if (
        train_features.shape[1]
        != validation_features.shape[1]
    ):
        raise ProbeFitError(
            "linear_feature_dimension_mismatch",
            "Linear-probe training and validation feature "
            f"dimensions differ: {train_features.shape[1]} != "
            f"{validation_features.shape[1]}.",
        )

    # Only features need scaling for LinearRegression. The target
    # remains in physical GeV.
    feature_scaler = StandardScaler()

    scaled_train_features = feature_scaler.fit_transform(
        train_features
    )
    scaled_validation_features = feature_scaler.transform(
        validation_features
    )

    estimator = LinearRegression()

    try:
        estimator.fit(
            scaled_train_features,
            train_target,
        )
    except Exception as error:
        raise ProbeFitError(
            "linear_fit_failed",
            f"Linear probe fitting failed: {error}",
        ) from error

    coefficients = np.asarray(estimator.coef_)
    intercept = np.asarray(estimator.intercept_)

    if (
        not np.isfinite(coefficients).all()
        or not np.isfinite(intercept).all()
    ):
        raise ProbeFitError(
            "non_finite_linear_parameters",
            "The fitted linear probe contains non-finite "
            "coefficients or intercepts.",
        )

    try:
        predictions_gev = np.asarray(
            estimator.predict(
                scaled_validation_features
            )
        ).reshape(-1)
    except Exception as error:
        raise ProbeFitError(
            "linear_outer_prediction_failed",
            f"Linear outer prediction failed: {error}",
        ) from error

    if (
        predictions_gev.shape[0]
        != validation_target.shape[0]
    ):
        raise ProbeFitError(
            "linear_outer_prediction_row_mismatch",
            "Linear predictions and validation targets have "
            "different event counts.",
        )

    if not np.isfinite(predictions_gev).all():
        raise ProbeFitError(
            "non_finite_linear_predictions",
            "The linear probe produced non-finite predictions.",
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
            "non_finite_linear_outer_r2",
            "Linear outer-validation R2 is non-finite.",
        )

    if not np.isfinite(outer_mae_gev):
        raise ProbeFitError(
            "non_finite_linear_outer_mae",
            "Linear outer-validation MAE is non-finite.",
        )

    return LinearProbeOuterResult(
        outer_r2_raw=outer_r2_raw,
        outer_r2_clipped=max(0.0, outer_r2_raw),
        outer_mae_gev=outer_mae_gev,
        n_train=int(train_features.shape[0]),
        n_validation=int(validation_features.shape[0]),
        feature_scaler=feature_scaler,
        estimator=estimator,
    )


def evaluate_linear_probe_representation(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
    *,
    representation_name: str,
) -> NamedLinearProbeResult:
    """Fit one of the two allowed linear representation probes."""

    if representation_name not in PRIMARY_PROBE_REPRESENTATIONS:
        raise ProbeFitError(
            "unknown_linear_probe_representation",
            "Linear probes are defined only for "
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

    outer_result = fit_linear_probe(
        train_features,
        train_representations.sensitive_target,
        validation_features,
        validation_representations.sensitive_target,
    )

    return NamedLinearProbeResult(
        representation_name=representation_name,
        metric_name=PROBE_REPRESENTATION_METRIC_NAMES[
            representation_name
        ],
        feature_dimension=int(train_features.shape[1]),
        outer_result=outer_result,
    )


def evaluate_primary_linear_probes(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
) -> PrimaryLinearProbeResult:
    """Evaluate independent latent and reconstruction linear probes."""

    if train_representations.split != "train":
        raise ProbeFitError(
            "invalid_linear_probe_training_split",
            "Linear probes require the AE train split, got "
            f"{train_representations.split!r}.",
        )

    if validation_representations.split != "valid":
        raise ProbeFitError(
            "invalid_linear_probe_outer_split",
            "Linear probes require the held-out AE valid split, "
            f"got {validation_representations.split!r}.",
        )

    latent_result = evaluate_linear_probe_representation(
        train_representations,
        validation_representations,
        representation_name="latent_logits",
    )

    reconstruction_result = (
        evaluate_linear_probe_representation(
            train_representations,
            validation_representations,
            representation_name="reconstructed_data",
        )
    )

    latent_outer = latent_result.outer_result
    reconstruction_outer = reconstruction_result.outer_result

    if (
        latent_outer.estimator
        is reconstruction_outer.estimator
        or latent_outer.feature_scaler
        is reconstruction_outer.feature_scaler
    ):
        raise ProbeFitError(
            "linear_probe_state_shared",
            "Latent and reconstruction linear probes share "
            "fitted state.",
        )

    return PrimaryLinearProbeResult(
        latent_logits=latent_result,
        reconstructed_data=reconstruction_result,
    )

