"""Extraction of frozen autoencoder representations for leakage probes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.data.utils import unpack_batch

import warnings
from types import MappingProxyType

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PROBE_INNER_SPLIT_SEED = 12345
PROBE_INNER_VALIDATION_FRACTION = 0.2
PROBE_INITIALIZATION_SEEDS = (10, 123, 500)
MLP_PROBE_CONFIG = MappingProxyType(
    {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "learning_rate": "constant",
        "learning_rate_init": 1e-3,
        "max_iter": 500,
        "shuffle": True,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "tol": 1e-4,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
    }
)
PROBE_REPRESENTATION_METRIC_NAMES = MappingProxyType(
    {
        "latent_logits": "z_logits",
        "reconstructed_data": "reconstruction",
        "latent_sample": "z_sample",
    }
)

PRIMARY_PROBE_REPRESENTATIONS = (
    "latent_logits",
    "reconstructed_data",
)


class ProbeExtractionError(RuntimeError):
    """Failure to construct a scientifically valid probe dataset."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class ProbeRepresentationSet:
    """Representations and physical sensitive targets for one AE split."""

    split: str
    latent_logits: np.ndarray
    latent_sample: np.ndarray
    reconstructed_data: np.ndarray
    sensitive_target: np.ndarray
    n_events: int
    sample_seed: int
    max_samples: int | None
    manifest_hash: str


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors inside a supported batch structure to one device."""

    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)

    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]

    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }

    # This preserves None and any non-tensor metadata.
    return value


def _as_feature_matrix(
    tensor: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Convert one representation to shape [events, features]."""

    if not isinstance(tensor, torch.Tensor):
        raise ProbeExtractionError(
            f"invalid_{name}",
            f"{name} must be a torch.Tensor.",
        )

    if tensor.ndim == 0:
        raise ProbeExtractionError(
            f"invalid_{name}",
            f"{name} cannot be a scalar.",
        )

    if tensor.ndim == 1:
        return tensor.unsqueeze(1)

    return torch.flatten(tensor, start_dim=1)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor, transfer it to CPU, and own the resulting memory."""

    return tensor.detach().cpu().numpy().copy()


def _validate_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ProbeExtractionError(
            f"non_finite_{name}",
            f"{name} contains NaN or infinity.",
        )


def extract_probe_split(
    model,
    datamodule,
    split: str,
    *,
    device: torch.device | str = "cpu",
) -> ProbeRepresentationSet:
    """Extract frozen AE representations and physical FET.Et for one split.

    The datamodule owns the raw split tensors. This function guarantees that
    the split is released, including when extraction or validation fails.
    """

    datamodule.setup_probe_split(split)

    try:
        object_feature_map = getattr(
            datamodule,
            "object_feature_map",
            None,
        )
        control_object_feature_map = getattr(
            datamodule,
            "control_object_feature_map",
            None,
        )

        if (
            object_feature_map is None
            or control_object_feature_map is None
        ):
            raise ProbeExtractionError(
                "feature_map_missing",
                "Both model-input and control feature maps are required.",
            )

        normalizer = getattr(datamodule, "normalizer", None)

        if normalizer is None:
            raise ProbeExtractionError(
                "normalizer_missing",
                "A normalizer is required to extract physical FET.Et.",
            )

        # These maps are not checkpoint parameters. They come from the data
        # cache and must therefore be attached explicitly.
        model.object_feature_map = object_feature_map
        model.control_object_feature_map = control_object_feature_map

        try:
            model._assert_sensitive_not_in_model_input()
        except RuntimeError as error:
            raise ProbeExtractionError(
                "sensitive_feature_in_input",
                str(error),
            ) from error

        resolved_device = torch.device(device)

        model.to(resolved_device)
        model.eval()
        model.requires_grad_(False)

        latent_logits_parts: list[np.ndarray] = []
        latent_sample_parts: list[np.ndarray] = []
        reconstruction_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []

        first_batch = True

        with torch.inference_mode():
            for batch in datamodule.probe_dataloader():
                device_batch = _move_to_device(
                    batch,
                    resolved_device,
                )
                batch_view = unpack_batch(device_batch)

                # This is the exact input layout used in AE.model_step().
                model_input = torch.flatten(
                    batch_view.x,
                    start_dim=1,
                )

                representations = (
                    model.forward_with_representations(model_input)
                )

                required_names = {
                    "latent_logits",
                    "latent_sample",
                    "reconstructed_data",
                }
                missing_names = (
                    required_names - representations.keys()
                )

                if missing_names:
                    raise ProbeExtractionError(
                        "representation_missing",
                        "Missing representations: "
                        f"{sorted(missing_names)}.",
                    )

                latent_logits = _as_feature_matrix(
                    representations["latent_logits"],
                    "latent_logits",
                )
                latent_sample = _as_feature_matrix(
                    representations["latent_sample"],
                    "latent_sample",
                )
                reconstructed_data = _as_feature_matrix(
                    representations["reconstructed_data"],
                    "reconstructed_data",
                )

                sensitive_target = model.extract_sensitive_values(
                    device_batch,
                    use_denormalized=True,
                    normalizer=normalizer,
                ).reshape(-1)

                batch_size = model_input.shape[0]

                row_counts = {
                    "model_input": batch_size,
                    "latent_logits": latent_logits.shape[0],
                    "latent_sample": latent_sample.shape[0],
                    "reconstructed_data": reconstructed_data.shape[0],
                    "sensitive_target": sensitive_target.shape[0],
                }

                if len(set(row_counts.values())) != 1:
                    raise ProbeExtractionError(
                        "representation_target_row_mismatch",
                        f"Batch row counts differ: {row_counts}.",
                    )

                if reconstructed_data.shape[1] != model_input.shape[1]:
                    raise ProbeExtractionError(
                        "reconstruction_shape_mismatch",
                        "Reconstruction width does not match model-input "
                        f"width: {reconstructed_data.shape[1]} != "
                        f"{model_input.shape[1]}.",
                    )

                is_binary = torch.all(
                    (latent_sample == 0)
                    | (latent_sample == 1)
                )

                if not bool(is_binary.item()):
                    raise ProbeExtractionError(
                        "latent_sample_not_binary",
                        "Evaluation-time latent_sample contains values "
                        "other than zero and one.",
                    )

                # Repeat only the first batch. This verifies that eval mode
                # disabled stochastic Bernoulli draws without doubling the
                # cost of the complete extraction.
                if first_batch:
                    repeated = model.forward_with_representations(
                        model_input
                    )
                    repeated_sample = _as_feature_matrix(
                        repeated["latent_sample"],
                        "latent_sample",
                    )

                    if not torch.equal(
                        latent_sample,
                        repeated_sample,
                    ):
                        raise ProbeExtractionError(
                            "latent_sample_not_deterministic",
                            "Repeated evaluation of the first batch "
                            "produced a different hard latent sample.",
                        )

                    first_batch = False

                latent_logits_parts.append(
                    _to_numpy(latent_logits)
                )
                latent_sample_parts.append(
                    _to_numpy(latent_sample)
                )
                reconstruction_parts.append(
                    _to_numpy(reconstructed_data)
                )
                target_parts.append(
                    _to_numpy(sensitive_target)
                )

        if not target_parts:
            raise ProbeExtractionError(
                "empty_split",
                f"Probe split {split!r} did not yield any events.",
            )

        latent_logits_array = np.concatenate(
            latent_logits_parts,
            axis=0,
        )
        latent_sample_array = np.concatenate(
            latent_sample_parts,
            axis=0,
        )
        reconstruction_array = np.concatenate(
            reconstruction_parts,
            axis=0,
        )
        target_array = np.concatenate(
            target_parts,
            axis=0,
        )

        _validate_finite(
            "latent_logits",
            latent_logits_array,
        )
        _validate_finite(
            "latent_sample",
            latent_sample_array,
        )
        _validate_finite(
            "reconstructed_data",
            reconstruction_array,
        )
        _validate_finite(
            "sensitive_target",
            target_array,
        )

        if np.unique(target_array).size < 2:
            raise ProbeExtractionError(
                "constant_target",
                "The sensitive target has fewer than two distinct values.",
            )

        n_events = int(target_array.shape[0])

        event_positions = np.arange(
            n_events,
            dtype="<i8",
        )
        manifest_hash = hashlib.sha256(
            event_positions.tobytes()
        ).hexdigest()

        return ProbeRepresentationSet(
            split=split,
            latent_logits=latent_logits_array,
            latent_sample=latent_sample_array,
            reconstructed_data=reconstruction_array,
            sensitive_target=target_array,
            n_events=n_events,
            sample_seed=12345,
            max_samples=None,
            manifest_hash=manifest_hash,
        )

    finally:
        datamodule.release_probe_split()

class ProbePartitionError(ValueError):
    """Failure to construct the fixed inner probe partition."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class ProbeInnerPartition:
    """Indices dividing AE training events into probe fit and validation."""

    fit_indices: np.ndarray
    validation_indices: np.ndarray
    seed: int
    validation_fraction: float
    manifest_hash: str


def make_probe_inner_partition(
    n_events: int,
    *,
    seed: int = PROBE_INNER_SPLIT_SEED,
    validation_fraction: float = PROBE_INNER_VALIDATION_FRACTION,
) -> ProbeInnerPartition:
    """Create the deterministic 80/20 split used for probe-seed selection."""

    if (
        isinstance(n_events, bool)
        or not isinstance(n_events, (int, np.integer))
        or int(n_events) <= 0
    ):
        raise ProbePartitionError(
            "invalid_event_count",
            f"n_events must be a positive integer, got {n_events!r}.",
        )

    n_events = int(n_events)

    if (
        not np.isfinite(validation_fraction)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ProbePartitionError(
            "invalid_inner_validation_fraction",
            "validation_fraction must be finite and strictly between "
            f"zero and one, got {validation_fraction!r}.",
        )

    if isinstance(seed, bool) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ProbePartitionError(
            "invalid_inner_split_seed",
            f"seed must be an integer, got {seed!r}.",
        )

    seed = int(seed)

    # Ceil matches the conventional interpretation of a 20% held-out
    # partition while ensuring that the requested fraction is not reduced.
    n_validation = int(
        np.ceil(n_events * validation_fraction)
    )
    n_fit = n_events - n_validation

    # R² requires at least two observations. We enforce that minimum in
    # both partitions before attempting to train any probe.
    if n_fit < 2 or n_validation < 2:
        raise ProbePartitionError(
            "inner_partition_too_small",
            "Both probe_fit and probe_inner_validation require at "
            f"least two events, got {n_fit} and {n_validation}.",
        )

    # RandomState fixes the MT19937 permutation algorithm. The seed and
    # algorithm together make partition membership reproducible.
    random_state = np.random.RandomState(seed)
    permutation = random_state.permutation(n_events)

    validation_indices = np.sort(
        permutation[:n_validation]
    ).astype(np.int64, copy=False)

    fit_indices = np.sort(
        permutation[n_validation:]
    ).astype(np.int64, copy=False)

    manifest = hashlib.sha256()
    manifest.update(b"probe_fit\\0")
    manifest.update(
        np.asarray(
            fit_indices,
            dtype="<i8",
        ).tobytes()
    )
    manifest.update(b"probe_inner_validation\\0")
    manifest.update(
        np.asarray(
            validation_indices,
            dtype="<i8",
        ).tobytes()
    )

    # The dataclass is frozen, but NumPy arrays are mutable separately.
    # Marking them read-only prevents accidental partition changes.
    fit_indices.setflags(write=False)
    validation_indices.setflags(write=False)

    return ProbeInnerPartition(
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        seed=seed,
        validation_fraction=float(validation_fraction),
        manifest_hash=manifest.hexdigest(),
    )

class ProbeFitError(RuntimeError):
    """Failure while fitting or evaluating one probe candidate."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class MLPProbeCandidateResult:
    """Fitted MLP and its inner-validation diagnostics."""

    seed: int
    inner_r2_raw: float
    inner_mae_gev: float
    convergence_warnings: tuple[str, ...]
    n_iter: int
    final_loss: float
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    estimator: MLPRegressor


def _validate_candidate_inputs(
    features: np.ndarray,
    target: np.ndarray,
    partition: ProbeInnerPartition,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features)
    target = np.asarray(target)

    if features.ndim != 2:
        raise ProbeFitError(
            "invalid_probe_feature_shape",
            "Probe features must have shape [events, features], "
            f"got {features.shape}.",
        )

    if target.ndim == 2 and target.shape[1] == 1:
        target = target.reshape(-1)
    elif target.ndim != 1:
        raise ProbeFitError(
            "invalid_probe_target_shape",
            "Probe target must have shape [events], "
            f"got {target.shape}.",
        )

    if features.shape[0] != target.shape[0]:
        raise ProbeFitError(
            "probe_feature_target_row_mismatch",
            "Probe features and target have different event counts: "
            f"{features.shape[0]} != {target.shape[0]}.",
        )

    if not np.isfinite(features).all():
        raise ProbeFitError(
            "non_finite_probe_features",
            "Probe features contain NaN or infinity.",
        )

    if not np.isfinite(target).all():
        raise ProbeFitError(
            "non_finite_probe_target",
            "Probe target contains NaN or infinity.",
        )

    combined_indices = np.concatenate(
        [
            partition.fit_indices,
            partition.validation_indices,
        ]
    )

    expected_indices = np.arange(
        features.shape[0],
        dtype=np.int64,
    )

    if (
        combined_indices.size != expected_indices.size
        or not np.array_equal(
            np.sort(combined_indices),
            expected_indices,
        )
    ):
        raise ProbeFitError(
            "invalid_inner_partition",
            "The inner partition must contain every event exactly once.",
        )

    return features, target


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


def fit_mlp_probe_candidate(
    features: np.ndarray,
    target: np.ndarray,
    partition: ProbeInnerPartition,
    *,
    seed: int,
) -> MLPProbeCandidateResult:
    """Fit one fixed MLP candidate and score inner validation.

    Feature and target preprocessing are fitted only on probe_fit.
    """

    if seed not in PROBE_INITIALIZATION_SEEDS:
        raise ProbeFitError(
            "invalid_probe_seed",
            f"Probe seed must be one of "
            f"{PROBE_INITIALIZATION_SEEDS}, got {seed}.",
        )

    features, target = _validate_candidate_inputs(
        features,
        target,
        partition,
    )

    features_fit = features[partition.fit_indices]
    target_fit = target[partition.fit_indices]

    features_validation = features[
        partition.validation_indices
    ]
    target_validation = target[
        partition.validation_indices
    ]

    if np.unique(target_fit).size < 2:
        raise ProbeFitError(
            "constant_probe_fit_target",
            "The probe-fit target is constant.",
        )

    if np.unique(target_validation).size < 2:
        raise ProbeFitError(
            "constant_probe_inner_validation_target",
            "The inner-validation target is constant.",
        )

    # These are fitted only on probe_fit. Validation information must not
    # influence either scaler.
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    scaled_features_fit = feature_scaler.fit_transform(
        features_fit
    )
    scaled_target_fit = target_scaler.fit_transform(
        target_fit.reshape(-1, 1)
    ).reshape(-1)

    scaled_features_validation = feature_scaler.transform(
        features_validation
    )

    estimator = MLPRegressor(
        **MLP_PROBE_CONFIG,
        random_state=seed,
    )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )
            estimator.fit(
                scaled_features_fit,
                scaled_target_fit,
            )
    except Exception as error:
        raise ProbeFitError(
            "mlp_fit_failed",
            f"MLP candidate with seed {seed} failed: {error}",
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

    scaled_predictions = np.asarray(
        estimator.predict(
            scaled_features_validation
        )
    ).reshape(-1)

    if (
        scaled_predictions.shape[0]
        != target_validation.shape[0]
    ):
        raise ProbeFitError(
            "mlp_prediction_row_mismatch",
            "MLP prediction and validation target counts differ.",
        )

    if not np.isfinite(scaled_predictions).all():
        raise ProbeFitError(
            "non_finite_mlp_predictions",
            "The MLP produced non-finite predictions.",
        )

    predictions_gev = target_scaler.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).reshape(-1)

    inner_r2_raw = float(
        r2_score(
            target_validation,
            predictions_gev,
        )
    )
    inner_mae_gev = float(
        mean_absolute_error(
            target_validation,
            predictions_gev,
        )
    )

    if not np.isfinite(inner_r2_raw):
        raise ProbeFitError(
            "non_finite_inner_r2",
            "The MLP produced a non-finite inner-validation R2.",
        )

    if not np.isfinite(inner_mae_gev):
        raise ProbeFitError(
            "non_finite_inner_mae",
            "The MLP produced a non-finite inner-validation MAE.",
        )

    return MLPProbeCandidateResult(
        seed=seed,
        inner_r2_raw=inner_r2_raw,
        inner_mae_gev=inner_mae_gev,
        convergence_warnings=convergence_warnings,
        n_iter=int(estimator.n_iter_),
        final_loss=float(estimator.loss_),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        estimator=estimator,
    )

@dataclass(frozen=True)
class MLPProbeCandidateFailure:
    """Recorded failure of one probe initialization."""

    seed: int
    reason: str
    message: str


@dataclass(frozen=True)
class MLPProbeSeedSelection:
    """Result of selecting one MLP initialization using inner R2."""

    selected_seed: int
    selected_candidate: MLPProbeCandidateResult
    successful_candidates: tuple[MLPProbeCandidateResult, ...]
    failed_candidates: tuple[MLPProbeCandidateFailure, ...]


class AllMLPProbeCandidatesFailed(ProbeFitError):
    """Raised when none of the frozen initialization seeds succeeds."""

    def __init__(
        self,
        failed_candidates: tuple[
            MLPProbeCandidateFailure,
            ...,
        ],
    ) -> None:
        self.failed_candidates = failed_candidates

        failure_summary = "; ".join(
            f"seed={failure.seed}: "
            f"{failure.reason}: {failure.message}"
            for failure in failed_candidates
        )

        super().__init__(
            "all_mlp_candidates_failed",
            "Every frozen MLP initialization failed. "
            f"{failure_summary}",
        )

def select_mlp_probe_seed(
    features: np.ndarray,
    target: np.ndarray,
    partition: ProbeInnerPartition,
) -> MLPProbeSeedSelection:
    """Fit every frozen seed and select the highest raw inner R2.

    Exact ties are resolved by the order in
    PROBE_INITIALIZATION_SEEDS.
    """

    successful_candidates: list[
        MLPProbeCandidateResult
    ] = []
    failed_candidates: list[
        MLPProbeCandidateFailure
    ] = []

    for seed in PROBE_INITIALIZATION_SEEDS:
        try:
            candidate = fit_mlp_probe_candidate(
                features,
                target,
                partition,
                seed=seed,
            )
        except ProbeFitError as error:
            failed_candidates.append(
                MLPProbeCandidateFailure(
                    seed=seed,
                    reason=error.reason,
                    message=str(error),
                )
            )
            continue

        if candidate.seed != seed:
            raise RuntimeError(
                "MLP candidate returned a different seed than "
                f"requested: requested {seed}, got "
                f"{candidate.seed}."
            )

        if not np.isfinite(candidate.inner_r2_raw):
            failed_candidates.append(
                MLPProbeCandidateFailure(
                    seed=seed,
                    reason="non_finite_inner_r2",
                    message=(
                        "Candidate returned a non-finite "
                        "inner-validation R2."
                    ),
                )
            )
            continue

        successful_candidates.append(candidate)

    failed_tuple = tuple(failed_candidates)

    if not successful_candidates:
        raise AllMLPProbeCandidatesFailed(
            failed_tuple
        )

    # Start with the first successful candidate. Replacing it only when
    # another score is strictly greater gives a deterministic tie rule.
    selected_candidate = successful_candidates[0]

    for candidate in successful_candidates[1:]:
        if (
            candidate.inner_r2_raw
            > selected_candidate.inner_r2_raw
        ):
            selected_candidate = candidate

    return MLPProbeSeedSelection(
        selected_seed=selected_candidate.seed,
        selected_candidate=selected_candidate,
        successful_candidates=tuple(
            successful_candidates
        ),
        failed_candidates=failed_tuple,
    )


@dataclass(frozen=True)
class MLPProbeOuterResult:
    """Fresh selected-seed MLP evaluated on outer validation."""

    selected_seed: int
    outer_r2_raw: float
    outer_r2_clipped: float
    outer_mae_gev: float
    convergence_warnings: tuple[str, ...]
    n_iter: int
    final_loss: float
    n_train: int
    n_validation: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    estimator: MLPRegressor

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

def refit_selected_mlp_probe(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    selection: MLPProbeSeedSelection,
) -> MLPProbeOuterResult:
    """Refit the selected seed on full AE train and score outer validation."""

    selected_seed = selection.selected_seed

    if selected_seed not in PROBE_INITIALIZATION_SEEDS:
        raise ProbeFitError(
            "invalid_selected_probe_seed",
            f"Selected seed must be one of "
            f"{PROBE_INITIALIZATION_SEEDS}, got {selected_seed}.",
        )

    if selection.selected_candidate.seed != selected_seed:
        raise ProbeFitError(
            "selected_candidate_seed_mismatch",
            "Seed selection and selected candidate disagree: "
            f"{selected_seed} != "
            f"{selection.selected_candidate.seed}.",
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

    # These must be new objects. The candidate scalers were fitted on
    # only probe_fit and must not be reused.
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

    # This must also be a fresh estimator. Only the selected random seed
    # is reused from the candidate-selection stage.
    estimator = MLPRegressor(
        **MLP_PROBE_CONFIG,
        random_state=selected_seed,
    )

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
            "mlp_refit_failed",
            "Fresh MLP refit failed for selected seed "
            f"{selected_seed}: {error}",
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

    return MLPProbeOuterResult(
        selected_seed=selected_seed,
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
    )

@dataclass(frozen=True)
class NamedMLPProbeResult:
    """Complete MLP procedure for one named representation."""

    representation_name: str
    metric_name: str
    feature_dimension: int
    seed_selection: MLPProbeSeedSelection
    outer_result: MLPProbeOuterResult


@dataclass(frozen=True)
class PrimaryMLPLeakageResult:
    """Primary latent and reconstruction leakage measurement."""

    latent_logits: NamedMLPProbeResult
    reconstructed_data: NamedMLPProbeResult
    inner_partition: ProbeInnerPartition
    leakage_worst: float

def evaluate_mlp_probe_representation(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
    partition: ProbeInnerPartition,
    *,
    representation_name: str,
) -> NamedMLPProbeResult:
    """Run seed selection and outer scoring for one representation."""

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

    selection = select_mlp_probe_seed(
        train_features,
        train_representations.sensitive_target,
        partition,
    )

    outer_result = refit_selected_mlp_probe(
        train_features,
        train_representations.sensitive_target,
        validation_features,
        validation_representations.sensitive_target,
        selection,
    )

    return NamedMLPProbeResult(
        representation_name=representation_name,
        metric_name=PROBE_REPRESENTATION_METRIC_NAMES[
            representation_name
        ],
        feature_dimension=int(train_features.shape[1]),
        seed_selection=selection,
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

    # Construct this once. Both primary representations must reuse
    # precisely the same probe-fit and inner-validation memberships.
    partition = make_probe_inner_partition(
        train_representations.n_events
    )

    latent_result = evaluate_mlp_probe_representation(
        train_representations,
        validation_representations,
        partition,
        representation_name="latent_logits",
    )

    reconstruction_result = (
        evaluate_mlp_probe_representation(
            train_representations,
            validation_representations,
            partition,
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
        inner_partition=partition,
        leakage_worst=float(leakage_worst),
    )