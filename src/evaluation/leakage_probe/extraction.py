"""Extract frozen autoencoder representations for leakage probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.utils import unpack_batch

from .errors import ProbeExtractionError
from .types import ProbeRepresentationSet

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


def _probe_cache_identity(
    datamodule: Any,
    control_object_feature_map: dict[str, Any],
) -> tuple[str, str]:
    """Return a stable identity for the concrete mlready cache."""

    cache_folder = getattr(
        datamodule,
        "main_cache_folder",
        None,
    )
    if cache_folder is None:
        raise ProbeExtractionError(
            "probe_cache_identity_missing",
            "The datamodule does not expose main_cache_folder.",
        )

    cache_path = str(
        Path(cache_folder).expanduser().resolve()
    )
    descriptor = {
        "cache_path": cache_path,
        "control_object_feature_map": (
            control_object_feature_map
        ),
    }
    canonical = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), cache_path


def _update_event_manifest(
    manifests: dict[str, Any],
    layouts: dict[str, tuple[Any, ...]],
    name: str,
    tensor: torch.Tensor | None,
) -> None:
    """Hash one cached tensor component independently of batching."""

    manifest = manifests.setdefault(name, hashlib.sha256())
    if tensor is None:
        layout = ("none",)
        if name in layouts and layouts[name] != layout:
            raise ProbeExtractionError(
                "probe_event_manifest_layout_changed",
                f"Event identity component {name!r} changed layout.",
            )
        if name not in layouts:
            layouts[name] = layout
            manifest.update(b"none")
        return

    values = tensor.detach().cpu().contiguous().numpy()
    layout = (
        str(values.dtype),
        *values.shape[1:],
    )
    if name in layouts and layouts[name] != layout:
        raise ProbeExtractionError(
            "probe_event_manifest_layout_changed",
            f"Event identity component {name!r} changed layout.",
        )
    if name not in layouts:
        layouts[name] = layout
        manifest.update(str(values.dtype).encode("ascii"))
        manifest.update(
            np.asarray(values.shape[1:], dtype="<i8").tobytes()
        )
    manifest.update(values.tobytes(order="C"))


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

        data_cache_id, data_cache_path = _probe_cache_identity(
            datamodule,
            control_object_feature_map,
        )

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
        event_component_manifests: dict[str, Any] = {}
        event_component_layouts: dict[
            str,
            tuple[Any, ...],
        ] = {}

        first_batch = True

        with torch.inference_mode():
            for batch in datamodule.probe_dataloader():
                identity_batch_view = unpack_batch(batch)
                device_batch = _move_to_device(
                    batch,
                    resolved_device,
                )
                batch_view = unpack_batch(device_batch)

                identity_data = (
                    identity_batch_view.control_x
                    if identity_batch_view.control_x is not None
                    else identity_batch_view.x
                )
                identity_mask = (
                    identity_batch_view.control_mask
                    if identity_batch_view.control_mask is not None
                    else identity_batch_view.mask
                )
                _update_event_manifest(
                    event_component_manifests,
                    event_component_layouts,
                    "cached_data",
                    identity_data,
                )
                _update_event_manifest(
                    event_component_manifests,
                    event_component_layouts,
                    "cached_mask",
                    identity_mask,
                )
                _update_event_manifest(
                    event_component_manifests,
                    event_component_layouts,
                    "l1bit",
                    identity_batch_view.l1bit,
                )

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
        event_manifest = hashlib.sha256()
        event_manifest.update(split.encode("utf-8"))
        event_manifest.update(b"\0")
        event_manifest.update(
            np.asarray([n_events], dtype="<i8").tobytes()
        )
        for component_name in sorted(
            event_component_manifests
        ):
            event_manifest.update(
                component_name.encode("utf-8")
            )
            event_manifest.update(b"\0")
            event_manifest.update(
                event_component_manifests[
                    component_name
                ].digest()
            )

        return ProbeRepresentationSet(
            split=split,
            latent_logits=latent_logits_array,
            latent_sample=latent_sample_array,
            reconstructed_data=reconstruction_array,
            sensitive_target=target_array,
            n_events=n_events,
            sample_seed=12345,
            max_samples=None,
            manifest_hash=event_manifest.hexdigest(),
            data_cache_id=data_cache_id,
            data_cache_path=data_cache_path,
            source_splits=(split,),
        )

    finally:
        datamodule.release_probe_split()
