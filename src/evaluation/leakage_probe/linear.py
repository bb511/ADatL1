"""Linear leakage probes for latent and reconstructed features."""

import logging
from time import perf_counter

import numpy as np
from scipy.linalg import lstsq, qr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .constants import (
    PRIMARY_PROBE_REPRESENTATIONS,
    PROBE_REPRESENTATION_METRIC_NAMES,
    PROBE_STREAM_CHUNK_ROWS,
)
from .errors import ProbeFitError
from .mlp import _validate_probe_dataset
from .types import (
    LinearProbeOuterResult,
    NamedLinearProbeResult,
    PrimaryLinearProbeResult,
    ProbeRepresentationSet,
)

log = logging.getLogger(__name__)


def _streamed_training_mse(
    coefficients: np.ndarray,
    intercept: float,
    feature_scaler: StandardScaler,
    features: np.ndarray,
    target: np.ndarray,
) -> float:
    """Training MSE without materialising the scaled development matrix."""

    squared_error = 0.0
    for start in range(0, len(target), PROBE_STREAM_CHUNK_ROWS):
        stop = min(start + PROBE_STREAM_CHUNK_ROWS, len(target))
        chunk = feature_scaler.transform(
            features[start:stop]
        ).astype(np.float64)
        residuals = (
            chunk @ coefficients
            + intercept
            - target[start:stop]
        )
        squared_error += float(np.dot(residuals, residuals))
    return squared_error / len(target)


def _streamed_predictions(
    coefficients: np.ndarray,
    intercept: float,
    feature_scaler: StandardScaler,
    features: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Predict in row blocks so the scaled held-out matrix never exists."""

    predictions = np.empty(n_rows, dtype=np.float64)
    for start in range(0, n_rows, PROBE_STREAM_CHUNK_ROWS):
        stop = min(start + PROBE_STREAM_CHUNK_ROWS, n_rows)
        chunk = feature_scaler.transform(
            features[start:stop]
        ).astype(np.float64)
        predictions[start:stop] = (
            chunk @ coefficients + intercept
        )
    return predictions


def _solve_streamed_least_squares(
    feature_scaler: StandardScaler,
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """Least squares on a development matrix that is never materialised.

    A blocked Householder QR reduces ``[X | 1 | y]`` to a small upper triangle
    one row block at a time.  That triangle carries exactly the singular values
    of the augmented development matrix, so the SVD solve below returns the same
    minimum-norm least-squares solution ``scipy.linalg.lstsq`` gives for the
    whole matrix.  Normal equations would be cheaper still but square the
    condition number; measured on decoder-shaped features they disagreed with
    the reference solution by 8e-4 in R^2, which is the size of the effects this
    study has to resolve.

    The accumulation is float64 on purpose.  Reconstruction features are a
    deterministic function of an 8-bit latent and therefore strongly collinear
    (kappa ~ 6e7 in simulation).  Against float32 epsilon that leaves no correct
    digits in the smallest singular directions, which is what the previous
    float32 ``LinearRegression`` fit was solving in.
    """

    n_features = int(features.shape[1])
    width = n_features + 2
    triangle: np.ndarray | None = None

    for start in range(0, len(target), PROBE_STREAM_CHUNK_ROWS):
        stop = min(start + PROBE_STREAM_CHUNK_ROWS, len(target))
        block = np.empty(
            (stop - start, width),
            dtype=np.float64,
        )
        block[:, :n_features] = feature_scaler.transform(
            features[start:stop]
        )
        block[:, n_features] = 1.0
        block[:, n_features + 1] = target[start:stop]
        stacked = (
            block
            if triangle is None
            else np.vstack((triangle, block))
        )
        triangle = qr(
            stacked,
            mode="r",
            check_finite=False,
        )[0][:width]

    if triangle is None:
        raise ProbeFitError(
            "linear_empty_development_pool",
            "The linear probe received no development rows.",
        )

    solution, _, rank, singular_values = lstsq(
        triangle[:, : n_features + 1],
        triangle[:, n_features + 1],
        cond=None,
        check_finite=False,
    )

    return (
        np.asarray(solution[:n_features], dtype=np.float64),
        float(solution[n_features]),
        int(rank),
        np.asarray(singular_values, dtype=np.float64),
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

    # Only features need scaling for the linear probe. The target remains in
    # physical GeV. ``fit`` computes the statistics without producing a scaled
    # copy; every later pass transforms one row block at a time.
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_features)

    log.info(
        "Linear least-squares fit starting: development=%d, held-out=%d, features=%d "
        "(streamed direct solve; no training epochs).",
        len(train_target), len(validation_target), train_features.shape[1],
    )
    started = perf_counter()
    try:
        (
            coefficients,
            intercept,
            rank,
            singular_values,
        ) = _solve_streamed_least_squares(
            feature_scaler,
            train_features,
            train_target,
        )
    except ProbeFitError:
        raise
    except Exception as error:
        raise ProbeFitError(
            "linear_fit_failed",
            f"Linear probe fitting failed: {error}",
        ) from error

    # The effective rank is a diagnostic worth having in the log: the
    # reconstruction columns are generated from an 8-bit latent, so a rank well
    # below the feature count is expected and is not by itself a failure.
    positive_singular_values = singular_values[singular_values > 0.0]
    condition_number = (
        float(
            singular_values.max()
            / positive_singular_values.min()
        )
        if positive_singular_values.size
        else float("inf")
    )
    log.info(
        "Linear solve conditioning: effective rank=%d of %d, condition number=%.4g.",
        rank, coefficients.shape[0] + 1, condition_number,
    )

    if (
        not np.isfinite(coefficients).all()
        or not np.isfinite(intercept)
    ):
        raise ProbeFitError(
            "non_finite_linear_parameters",
            "The fitted linear probe contains non-finite "
            "coefficients or intercepts.",
        )

    # A real LinearRegression carries the solution so that everything
    # downstream -- the four-probe independence check, predict(), serialization
    # -- keeps working unchanged.
    estimator = LinearRegression()
    estimator.coef_ = coefficients
    estimator.intercept_ = intercept
    estimator.n_features_in_ = int(coefficients.shape[0])
    estimator.rank_ = rank
    estimator.singular_ = singular_values

    try:
        predictions_gev = _streamed_predictions(
            coefficients,
            intercept,
            feature_scaler,
            validation_features,
            len(validation_target),
        )
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

    train_mse = _streamed_training_mse(
        coefficients,
        intercept,
        feature_scaler,
        train_features,
        train_target,
    )
    residuals = predictions_gev - validation_target
    outer_mse = float(np.dot(residuals, residuals) / len(residuals))
    log.info(
        "Linear probe finished in %.1fs: held-out R2=%.6f, clipped R2=%.6f, "
        "MAE=%.6g GeV; MSE development=%.6g, held-out=%.6g GeV^2.",
        perf_counter() - started, outer_r2_raw, max(0.0, outer_r2_raw),
        outer_mae_gev, train_mse, outer_mse,
    )
    return LinearProbeOuterResult(
        outer_r2_raw=outer_r2_raw,
        outer_r2_clipped=max(0.0, outer_r2_raw),
        outer_mae_gev=outer_mae_gev,
        n_train=int(train_features.shape[0]),
        n_validation=int(validation_features.shape[0]),
        feature_scaler=feature_scaler,
        estimator=estimator,
        train_mse_gev2=train_mse,
        outer_mse_gev2=outer_mse,
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

    log.info("Starting linear probe for %s.", representation_name)
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
