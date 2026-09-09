"""Shared provenance contract for joinable Pareto scientific artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.utils import unpack_batch


ARTIFACT_PROVENANCE_SCHEMA_VERSION = 1


def normalize_artifact_identity(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the run fields that identify one Pareto candidate execution."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        try:
            value = dict(value)
        except (TypeError, ValueError) as error:
            raise TypeError("artifact_provenance must be a mapping or None.") from error

    protocol_version = value.get("protocol_version")
    configuration_id = value.get("configuration_id")
    autoencoder_seed = value.get("autoencoder_seed")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise ValueError("artifact_provenance.protocol_version must be non-empty.")
    if not isinstance(configuration_id, str) or not configuration_id:
        raise ValueError("artifact_provenance.configuration_id must be non-empty.")
    if isinstance(autoencoder_seed, bool) or not isinstance(
        autoencoder_seed,
        Integral,
    ):
        raise ValueError("artifact_provenance.autoencoder_seed must be an integer.")

    return {
        "protocol_version": protocol_version,
        "configuration_id": configuration_id,
        "autoencoder_seed": int(autoencoder_seed),
    }


def evaluation_mode_from_trainer_split(split: str) -> str:
    """Map evaluator split labels to the frozen scientific reporting modes."""
    modes = {"val": "validation", "test": "final_test"}
    try:
        return modes[str(split)]
    except KeyError as error:
        raise ValueError(
            "Artifact provenance requires trainer.split to be 'val' or 'test', got "
            f"{split!r}."
        ) from error


def data_cache_identity(datamodule: Any) -> dict[str, str]:
    """Return the stable cache identity shared with leakage-probe extraction."""
    cache_folder = getattr(datamodule, "main_cache_folder", None)
    control_feature_map = getattr(datamodule, "control_object_feature_map", None)
    if cache_folder is None or control_feature_map is None:
        raise RuntimeError(
            "Artifact provenance requires datamodule.main_cache_folder and "
            "datamodule.control_object_feature_map."
        )

    cache_path = str(Path(cache_folder).expanduser().resolve())
    descriptor = {
        "cache_path": cache_path,
        "control_object_feature_map": control_feature_map,
    }
    canonical = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "id": hashlib.sha256(canonical).hexdigest(),
        "path": cache_path,
    }


def data_cache_identity_from_trainer(trainer: Any) -> dict[str, str]:
    """Get cache identity from the datamodule explicitly attached to the evaluator."""
    datamodule = getattr(trainer, "artifact_provenance_datamodule", None)
    if datamodule is None:
        raise RuntimeError(
            "Artifact provenance datamodule is missing from the evaluator trainer."
        )
    return data_cache_identity(datamodule)


def checkpoint_identity(ckpt_path: str | Path) -> dict[str, Any]:
    """Fingerprint the exact selected checkpoint without relying on its filename."""
    resolved_path = Path(ckpt_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Artifact provenance checkpoint does not exist: {resolved_path}."
        )

    digest = hashlib.sha256()
    with resolved_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return {
        "name": resolved_path.name,
        "path": str(resolved_path),
        "sha256": digest.hexdigest(),
        "size_bytes": resolved_path.stat().st_size,
        "selection_metric": "val/loss_total",
    }


def checkpoint_identity_from_trainer(
    trainer: Any,
    ckpt_path: str | Path,
) -> dict[str, Any]:
    """Cache a checkpoint digest on the evaluator trainer for sibling callbacks."""
    resolved_path = str(Path(ckpt_path).expanduser().resolve())
    identities = getattr(trainer, "_artifact_checkpoint_identities", None)
    if identities is None:
        identities = {}
        trainer._artifact_checkpoint_identities = identities
    if resolved_path not in identities:
        identities[resolved_path] = checkpoint_identity(resolved_path)
    return dict(identities[resolved_path])


class EventManifest:
    """Stream a stable event identity for the exact rows an artifact consumes."""

    def __init__(self, source_split: str) -> None:
        self.source_split = source_split
        self._components: dict[str, dict[str, Any]] = defaultdict(dict)
        self._layouts: dict[str, dict[str, tuple[Any, ...]]] = defaultdict(dict)
        self._event_counts: dict[str, int] = defaultdict(int)

    def update_batch(
        self,
        dataset: str,
        batch: Any,
        *,
        rows: int | torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Add a complete or selected prefix/mask of a batch to one dataset manifest."""
        view = unpack_batch(batch)
        identity_data = view.control_x if view.control_x is not None else view.x
        identity_mask = (
            view.control_mask if view.control_mask is not None else view.mask
        )
        selected_data = self._select_rows(identity_data, rows)
        if selected_data.shape[0] == 0:
            return

        components = {
            "cached_data": selected_data,
            "cached_mask": self._select_rows(identity_mask, rows),
            "l1bit": self._select_rows(view.l1bit, rows),
            "label": self._select_rows(view.y, rows),
        }
        for name, tensor in components.items():
            self._update_component(dataset, name, tensor)
        self._event_counts[dataset] += int(selected_data.shape[0])

    @staticmethod
    def _select_rows(
        tensor: torch.Tensor | None,
        rows: int | torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor | None:
        if tensor is None or rows is None:
            return tensor
        if isinstance(rows, Integral) and not isinstance(rows, bool):
            return tensor[: int(rows)]
        if isinstance(rows, np.ndarray):
            rows = torch.as_tensor(rows, device=tensor.device)
        if isinstance(rows, torch.Tensor):
            return tensor[rows]
        raise TypeError("Event-manifest row selector must be an integer or array.")

    def _update_component(
        self,
        dataset: str,
        name: str,
        tensor: torch.Tensor | None,
    ) -> None:
        if tensor is None:
            layout = ("none",)
            values = None
        else:
            values = tensor.detach().cpu().contiguous().numpy()
            layout = (str(values.dtype), *values.shape[1:])

        layouts = self._layouts[dataset]
        previous_layout = layouts.get(name)
        if previous_layout is not None and previous_layout != layout:
            raise RuntimeError(
                f"Event identity component {dataset}/{name} changed layout."
            )

        manifest = self._components[dataset].setdefault(name, hashlib.sha256())
        if previous_layout is None:
            layouts[name] = layout
            if values is None:
                manifest.update(b"none")
            else:
                manifest.update(str(values.dtype).encode("ascii"))
                manifest.update(
                    np.asarray(values.shape[1:], dtype="<i8").tobytes()
                )
        if values is not None:
            manifest.update(values.tobytes(order="C"))

    def payload(self) -> dict[str, Any]:
        """Return the combined manifest and independently inspectable dataset parts."""
        if not self._components:
            raise RuntimeError("Artifact event manifest has no evaluated events.")

        root = hashlib.sha256()
        root.update(b"pareto-artifact-event-manifest-v1\0")
        root.update(self.source_split.encode("utf-8"))
        root.update(b"\0")
        datasets: dict[str, Any] = {}
        for dataset in sorted(self._components):
            component_manifests = self._components[dataset]
            dataset_digest = hashlib.sha256()
            dataset_digest.update(dataset.encode("utf-8"))
            dataset_digest.update(b"\0")
            dataset_digest.update(
                np.asarray([self._event_counts[dataset]], dtype="<i8").tobytes()
            )
            components: dict[str, str] = {}
            for name in sorted(component_manifests):
                digest = component_manifests[name].digest()
                components[name] = digest.hex()
                dataset_digest.update(name.encode("utf-8"))
                dataset_digest.update(b"\0")
                dataset_digest.update(digest)
            digest = dataset_digest.digest()
            root.update(dataset.encode("utf-8"))
            root.update(b"\0")
            root.update(digest)
            datasets[dataset] = {
                "n_events": self._event_counts[dataset],
                "event_manifest_hash": digest.hex(),
                "event_manifest_components": components,
            }

        return {
            "source_split": self.source_split,
            "event_manifest_hash": root.hexdigest(),
            "datasets": datasets,
        }


def make_artifact_provenance(
    identity: Mapping[str, Any] | None,
    *,
    checkpoint: Mapping[str, Any],
    evaluation_mode: str,
    data_cache: Mapping[str, str],
    event_manifest: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build the canonical provenance envelope shared by Pareto artifacts."""
    normalized = normalize_artifact_identity(identity)
    if normalized is None:
        return None
    if evaluation_mode not in {"validation", "final_test"}:
        raise ValueError(f"Unsupported artifact evaluation mode {evaluation_mode!r}.")

    return {
        "schema_version": ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        **normalized,
        "checkpoint": dict(checkpoint),
        "evaluation_mode": evaluation_mode,
        "data": {
            "cache": dict(data_cache),
            "event_manifest": (
                dict(event_manifest) if event_manifest is not None else None
            ),
        },
    }
