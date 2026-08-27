"""Extraction of frozen autoencoder representations for leakage probes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.data.utils import unpack_batch


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


PROBE_INNER_SPLIT_SEED = 12345
PROBE_INNER_VALIDATION_FRACTION = 0.2


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