"""Fixed-capacity MLP leakage probes and seed selection."""

from __future__ import annotations

import warnings
import logging
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .constants import (
    MLP_PROBE_CONFIG,
    PROBE_INITIALIZATION_SEED,
    PROBE_REPRESENTATION_METRIC_NAMES,
)
from .errors import ProbeFitError
from .types import (
    MLPProbeOuterResult,
    NamedMLPProbeResult,
    PrimaryMLPLeakageResult,
    ProbeRepresentationSet,
)

log = logging.getLogger(__name__)


def _record_mlp_history(estimator: MLPRegressor) -> dict[str, tuple[float, ...]]:
    """Copy sklearn's diagnostics without changing its optimizer/early stopping."""
    return {
        "loss_curve": tuple(float(value) for value in getattr(estimator, "loss_curve_", ())),
        "early_stopping_validation_scores": tuple(
            float(value) for value in (getattr(estimator, "validation_scores_", None) or ())
        ),
    }


def _log_mlp_fit_result(label: str, estimator, elapsed: float, fit_warnings) -> None:
    log.info(
        "%s finished in %.1fs: epochs=%d, last training loss=%.6g.",
        label, elapsed, estimator.n_iter_, estimator.loss_,
    )
    for message in fit_warnings:
        log.warning("%s: %s", label, message)

def _validate_fitted_mlp(estimator: MLPRegressor) -> None:
    parameter_arrays = [
        *getattr(estimator, "coefs_", []),
        *getattr(estimator, "intercepts_", []),
    ]

    if not parameter_arrays:
        raise ProbeFitError(
            "missing_mlp_parameters",
            "The fitted MLP does not expose weights and biases.",
        )

    if not all(
        np.isfinite(parameter).all()
        for parameter in parameter_arrays
    ):
        raise ProbeFitError(
            "non_finite_mlp_parameters",
            "The fitted MLP contains non-finite parameters.",
        )

    final_loss = float(
        getattr(estimator, "loss_", np.nan)
    )

    if not np.isfinite(final_loss):
        raise ProbeFitError(
            "non_finite_mlp_loss",
            "The fitted MLP has a non-finite final loss.",
        )


def _validate_probe_dataset(
    features: np.ndarray,
    target: np.ndarray,
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one complete train or outer-validation dataset."""

    features = np.asarray(features)
    target = np.asarray(target)

    if features.ndim != 2 or features.shape[1] == 0:
        raise ProbeFitError(
            f"invalid_{split_name}_feature_shape",
            f"{split_name} features must have shape "
            f"[events, nonzero features], got {features.shape}.",
        )

    if target.ndim == 2 and target.shape[1] == 1:
        target = target.reshape(-1)
    elif target.ndim != 1:
        raise ProbeFitError(
            f"invalid_{split_name}_target_shape",
            f"{split_name} target must have shape [events], "
            f"got {target.shape}.",
        )

    if features.shape[0] != target.shape[0]:
        raise ProbeFitError(
            f"{split_name}_feature_target_row_mismatch",
            f"{split_name} features and target have different "
            f"event counts: {features.shape[0]} != "
            f"{target.shape[0]}.",
        )

    if features.shape[0] < 2:
        raise ProbeFitError(
            f"{split_name}_too_small",
            f"{split_name} requires at least two events.",
        )

    if not np.isfinite(features).all():
        raise ProbeFitError(
            f"non_finite_{split_name}_features",
            f"{split_name} features contain NaN or infinity.",
        )

    if not np.isfinite(target).all():
        raise ProbeFitError(
            f"non_finite_{split_name}_target",
            f"{split_name} target contains NaN or infinity.",
        )

    if np.unique(target).size < 2:
        raise ProbeFitError(
            f"constant_{split_name}_target",
            f"{split_name} target is constant.",
        )

    return features, target

def fit_mlp_probe(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    *,
    seed: int = PROBE_INITIALIZATION_SEED,
) -> MLPProbeOuterResult:
    """Fit the frozen-seed MLP on the development pool and score held-out data.

    This mirrors ``linear.fit_linear_probe``: one fit on the complete
    development pool and one held-out score. The protocol fits a single
    predeclared initialization, so there is no candidate stage.
    """

    if seed != PROBE_INITIALIZATION_SEED:
        raise ProbeFitError(
            "invalid_probe_seed",
            f"Probe seed must be {PROBE_INITIALIZATION_SEED}, got {seed}.",
        )

    train_features, train_target = _validate_probe_dataset(
        train_features,
        train_target,
        split_name="full_train",
    )
    validation_features, validation_target = (
        _validate_probe_dataset(
            validation_features,
            validation_target,
            split_name="outer_validation",
        )
    )

    if (
        train_features.shape[1]
        != validation_features.shape[1]
    ):
        raise ProbeFitError(
            "probe_feature_dimension_mismatch",
            "Training and outer-validation feature dimensions "
            f"differ: {train_features.shape[1]} != "
            f"{validation_features.shape[1]}.",
        )

    # Each probe owns its scalers, fitted on the development pool only.
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    scaled_train_features = feature_scaler.fit_transform(
        train_features
    )
    scaled_train_target = target_scaler.fit_transform(
        train_target.reshape(-1, 1)
    ).reshape(-1)

    scaled_validation_features = feature_scaler.transform(
        validation_features
    )

    estimator = MLPRegressor(
        **MLP_PROBE_CONFIG,
        random_state=seed,
        verbose=True,
    )

    label = f"MLP probe seed={seed}"
    log.info(
        "%s starting: development=%d, held-out=%d, features=%d.",
        label, len(train_target), len(validation_target), train_features.shape[1],
    )
    started = perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )
            estimator.fit(
                scaled_train_features,
                scaled_train_target,
            )
    except Exception as error:
        raise ProbeFitError(
            "mlp_fit_failed",
            f"MLP probe fitting failed for seed {seed}: {error}",
        ) from error

    convergence_warnings = tuple(
        str(item.message)
        for item in caught
        if issubclass(
            item.category,
            ConvergenceWarning,
        )
    )

    _validate_fitted_mlp(estimator)
    _log_mlp_fit_result(label, estimator, perf_counter() - started, convergence_warnings)

    try:
        scaled_predictions = np.asarray(
            estimator.predict(
                scaled_validation_features
            )
        ).reshape(-1)
    except Exception as error:
        raise ProbeFitError(
            "mlp_outer_prediction_failed",
            f"Outer-validation prediction failed: {error}",
        ) from error

    if (
        scaled_predictions.shape[0]
        != validation_target.shape[0]
    ):
        raise ProbeFitError(
            "mlp_outer_prediction_row_mismatch",
            "Outer predictions and targets have different "
            "event counts.",
        )

    if not np.isfinite(scaled_predictions).all():
        raise ProbeFitError(
            "non_finite_mlp_outer_predictions",
            "The refitted MLP produced non-finite outer "
            "predictions.",
        )

    predictions_gev = target_scaler.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).reshape(-1)

    if not np.isfinite(predictions_gev).all():
        raise ProbeFitError(
            "non_finite_mlp_outer_predictions_gev",
            "Inverse-transformed outer predictions are non-finite.",
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
            "non_finite_outer_r2",
            "Outer-validation R2 is non-finite.",
        )

    if not np.isfinite(outer_mae_gev):
        raise ProbeFitError(
            "non_finite_outer_mae",
            "Outer-validation MAE is non-finite.",
        )

    log.info(
        "%s held-out score: R2=%.6f, clipped R2=%.6f, MAE=%.6g GeV.",
        label, outer_r2_raw, max(0.0, outer_r2_raw), outer_mae_gev,
    )
    return MLPProbeOuterResult(
        seed=seed,
        outer_r2_raw=outer_r2_raw,
        outer_r2_clipped=max(0.0, outer_r2_raw),
        outer_mae_gev=outer_mae_gev,
        convergence_warnings=convergence_warnings,
        n_iter=int(estimator.n_iter_),
        final_loss=float(estimator.loss_),
        n_train=int(train_features.shape[0]),
        n_validation=int(validation_features.shape[0]),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        estimator=estimator,
        **_record_mlp_history(estimator),
    )


def evaluate_mlp_probe_representation(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
    *,
    representation_name: str,
) -> NamedMLPProbeResult:
    """Fit and score the frozen-seed MLP probe for one representation."""

    if (
        representation_name
        not in PROBE_REPRESENTATION_METRIC_NAMES
    ):
        raise ProbeFitError(
            "unknown_probe_representation",
            f"Unknown probe representation "
            f"{representation_name!r}. Expected one of "
            f"{tuple(PROBE_REPRESENTATION_METRIC_NAMES)}.",
        )

    train_features = getattr(
        train_representations,
        representation_name,
    )
    validation_features = getattr(
        validation_representations,
        representation_name,
    )

    log.info(
        "Starting MLP probe for %s: %d features, development=%s (%d), held-out=%s (%d).",
        representation_name, train_features.shape[1], train_representations.split,
        train_representations.n_events, validation_representations.split,
        validation_representations.n_events,
    )
    outer_result = fit_mlp_probe(
        train_features,
        train_representations.sensitive_target,
        validation_features,
        validation_representations.sensitive_target,
    )

    return NamedMLPProbeResult(
        representation_name=representation_name,
        metric_name=PROBE_REPRESENTATION_METRIC_NAMES[
            representation_name
        ],
        feature_dimension=int(train_features.shape[1]),
        outer_result=outer_result,
    )

def evaluate_primary_mlp_probes(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
) -> PrimaryMLPLeakageResult:
    """Evaluate both primary probes for hyperparameter selection."""

    if train_representations.split != "train":
        raise ProbeFitError(
            "invalid_probe_training_split",
            "Primary probe fitting requires the AE train split, "
            f"got {train_representations.split!r}.",
        )

    if validation_representations.split != "valid":
        raise ProbeFitError(
            "invalid_probe_outer_split",
            "Hyperparameter-selection leakage must use the AE "
            f"valid split, got "
            f"{validation_representations.split!r}.",
        )

    if (
        train_representations.n_events
        != train_representations.sensitive_target.shape[0]
    ):
        raise ProbeFitError(
            "train_event_count_mismatch",
            "Recorded training event count does not match the "
            "training target.",
        )

    if (
        validation_representations.n_events
        != validation_representations.sensitive_target.shape[0]
    ):
        raise ProbeFitError(
            "validation_event_count_mismatch",
            "Recorded validation event count does not match the "
            "validation target.",
        )

    latent_result = evaluate_mlp_probe_representation(
        train_representations,
        validation_representations,
        representation_name="latent_logits",
    )

    reconstruction_result = (
        evaluate_mlp_probe_representation(
            train_representations,
            validation_representations,
            representation_name="reconstructed_data",
        )
    )

    # Defensive runtime check: these are scientifically independent
    # probes and must never share fitted preprocessing or estimators.
    latent_outer = latent_result.outer_result
    reconstruction_outer = reconstruction_result.outer_result

    shared_objects = (
        latent_outer.estimator
        is reconstruction_outer.estimator
        or latent_outer.feature_scaler
        is reconstruction_outer.feature_scaler
        or latent_outer.target_scaler
        is reconstruction_outer.target_scaler
    )

    if shared_objects:
        raise ProbeFitError(
            "primary_probe_state_shared",
            "Latent and reconstruction probes unexpectedly share "
            "an estimator or fitted scaler.",
        )

    leakage_worst = max(
        latent_outer.outer_r2_clipped,
        reconstruction_outer.outer_r2_clipped,
    )

    return PrimaryMLPLeakageResult(
        latent_logits=latent_result,
        reconstructed_data=reconstruction_result,
        leakage_worst=float(leakage_worst),
    )
