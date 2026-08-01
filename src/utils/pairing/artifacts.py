from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.utils.pairing.table import atomic_torch_save

FULL_PAIRING_ARTIFACT_TYPE = "full_pairing"
FULL_PAIRING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FullPairingTensors:
    """Full-size deterministic pairing maps.

    ``target_to_reference`` and the target-aligned fields always have one entry
    per target event. Unmatched targets use ``-1`` for the reference index,
    ``+inf`` for distance, rank zero, and ``False`` in ``valid``.
    ``reference_to_target`` is the inverse map and uses ``-1`` for unused
    reference events. ``caliper_valid`` is an independent acceptance mask: a
    target rejected by the caliper retains its deterministic assignment.
    """

    target_to_reference: torch.Tensor
    reference_to_target: torch.Tensor
    distance: torch.Tensor
    valid: torch.Tensor
    caliper_valid: torch.Tensor
    candidate_rank: torch.Tensor

    @property
    def n_target(self) -> int:
        """Return the number of target events."""
        return int(self.target_to_reference.numel())

    @property
    def n_reference(self) -> int:
        """Return the number of reference events."""
        return int(self.reference_to_target.numel())

    @property
    def n_pairs(self) -> int:
        """Return the number of assigned target events."""
        return int(self.valid.sum().item())

    @property
    def n_accepted(self) -> int:
        """Return the number of caliper-accepted target events."""
        return int(self.caliper_valid.sum().item())

    @property
    def coverage(self) -> float:
        """Return the fraction of target events with assignments."""
        return self.n_pairs / self.n_target if self.n_target else 0.0

    @property
    def acceptance(self) -> float:
        """Return the fraction of targets accepted by the caliper."""
        return self.n_accepted / self.n_target if self.n_target else 0.0

    def validate(self) -> None:
        """Validate shapes, types, bounds, masks, and inverse consistency."""
        _validate_full_pairing_tensors(self)

    def sparse_indices(self, *, accepted_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned sparse target and reference indices."""
        mask = self.caliper_valid if accepted_only else self.valid
        target = torch.nonzero(mask, as_tuple=False).flatten()
        return target, self.target_to_reference[target]


def full_pairing_artifact(
    tensors: FullPairingTensors,
    *,
    target_dataset: str,
    reference_dataset: str,
    split: str,
    strategy: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a serializable, versioned full-pairing artifact."""
    tensors.validate()
    for name, value in (
        ("target_dataset", target_dataset),
        ("reference_dataset", reference_dataset),
        ("split", split),
        ("strategy", strategy),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        n_target=tensors.n_target,
        n_reference=tensors.n_reference,
        n_pairs=tensors.n_pairs,
        n_accepted=tensors.n_accepted,
        coverage=tensors.coverage,
        acceptance=tensors.acceptance,
    )
    return {
        "artifact_type": FULL_PAIRING_ARTIFACT_TYPE,
        "schema_version": FULL_PAIRING_SCHEMA_VERSION,
        "target_to_reference": tensors.target_to_reference.cpu(),
        "reference_to_target": tensors.reference_to_target.cpu(),
        "distance": tensors.distance.cpu(),
        "valid": tensors.valid.cpu(),
        "caliper_valid": tensors.caliper_valid.cpu(),
        "candidate_rank": tensors.candidate_rank.cpu(),
        "target_dataset": target_dataset,
        "reference_dataset": reference_dataset,
        "split": split,
        "strategy": strategy,
        "metadata": artifact_metadata,
    }


def pairing_tensors_from_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_target_dataset: str | None = None,
    expected_reference_dataset: str | None = None,
    expected_split: str | None = None,
    expected_n_target: int | None = None,
    expected_n_reference: int | None = None,
) -> FullPairingTensors:
    """Validate an artifact and return its full-size tensors."""
    if not isinstance(artifact, Mapping):
        raise TypeError("Full pairing artifact must be a mapping.")
    if artifact.get("artifact_type") != FULL_PAIRING_ARTIFACT_TYPE:
        raise ValueError("Artifact is not a full-pairing tensor artifact.")
    if artifact.get("schema_version") != FULL_PAIRING_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported full-pairing schema version: " f"{artifact.get('schema_version')!r}."
        )

    tensors = FullPairingTensors(
        target_to_reference=_tensor(artifact, "target_to_reference"),
        reference_to_target=_tensor(artifact, "reference_to_target"),
        distance=_tensor(artifact, "distance"),
        valid=_tensor(artifact, "valid"),
        caliper_valid=_tensor(artifact, "caliper_valid"),
        candidate_rank=_tensor(artifact, "candidate_rank"),
    )
    tensors.validate()

    for key, expected in (
        ("target_dataset", expected_target_dataset),
        ("reference_dataset", expected_reference_dataset),
        ("split", expected_split),
    ):
        value = artifact.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Full pairing artifact {key} must be a non-empty string.")
        if expected is not None and value != expected:
            raise ValueError(f"Artifact {key} is {value!r}, expected {expected!r}.")
    strategy = artifact.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("Full pairing artifact strategy must be a non-empty string.")

    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("Full pairing artifact metadata must be a mapping.")
    expected_counts = {
        "n_target": tensors.n_target,
        "n_reference": tensors.n_reference,
        "n_pairs": tensors.n_pairs,
        "n_accepted": tensors.n_accepted,
    }
    for key, observed in expected_counts.items():
        if metadata.get(key) != observed:
            raise ValueError(
                f"Artifact metadata.{key} is {metadata.get(key)!r}, expected {observed}."
            )
    if expected_n_target is not None and tensors.n_target != int(expected_n_target):
        raise ValueError(f"Artifact has {tensors.n_target} targets, expected {expected_n_target}.")
    if expected_n_reference is not None and tensors.n_reference != int(expected_n_reference):
        raise ValueError(
            "Artifact has " f"{tensors.n_reference} references, expected {expected_n_reference}."
        )
    return tensors


def load_full_pairing_artifact(
    path: str | Path,
    **expectations: Any,
) -> tuple[dict[str, Any], FullPairingTensors]:
    """Load and validate a full-pairing artifact from disk."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Full pairing artifact does not exist: {resolved}")
    artifact = torch.load(resolved, map_location="cpu", weights_only=False)
    tensors = pairing_tensors_from_artifact(artifact, **expectations)
    return artifact, tensors


def save_full_pairing_artifact(
    artifact: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically save a full-pairing artifact."""
    pairing_tensors_from_artifact(artifact)
    return atomic_torch_save(dict(artifact), path, overwrite=overwrite)


def _validate_full_pairing_tensors(tensors: FullPairingTensors) -> None:
    """Validate the internal consistency of full-size pairing tensors."""
    target = tensors.target_to_reference
    inverse = tensors.reference_to_target
    distance = tensors.distance
    valid = tensors.valid
    caliper_valid = tensors.caliper_valid
    rank = tensors.candidate_rank
    if target.ndim != 1 or target.dtype != torch.long:
        raise ValueError("target_to_reference must be a one-dimensional LongTensor.")
    if inverse.ndim != 1 or inverse.dtype != torch.long:
        raise ValueError("reference_to_target must be a one-dimensional LongTensor.")
    if target.numel() == 0 or inverse.numel() == 0:
        raise ValueError("Full pairing tensors require non-empty source datasets.")
    for name, value in (
        ("distance", distance),
        ("valid", valid),
        ("caliper_valid", caliper_valid),
        ("candidate_rank", rank),
    ):
        if value.ndim != 1 or value.numel() != target.numel():
            raise ValueError(f"{name} must have one entry per target event.")
    if not torch.is_floating_point(distance):
        raise ValueError("distance must be a floating-point tensor.")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be a BoolTensor.")
    if caliper_valid.dtype != torch.bool:
        raise ValueError("caliper_valid must be a BoolTensor.")
    if rank.dtype != torch.long:
        raise ValueError("candidate_rank must be a LongTensor.")
    if not torch.equal(valid, target >= 0):
        raise ValueError("valid must be exactly equivalent to target_to_reference >= 0.")
    if torch.any(caliper_valid & ~valid):
        raise ValueError("caliper_valid may only accept assigned target events.")
    if torch.any(target < -1) or torch.any(target >= inverse.numel()):
        raise ValueError("target_to_reference contains an out-of-bounds index.")
    if torch.any(inverse < -1) or torch.any(inverse >= target.numel()):
        raise ValueError("reference_to_target contains an out-of-bounds index.")
    if torch.any(~torch.isfinite(distance[valid])) or torch.any(distance[valid] < 0):
        raise ValueError("Matched distances must be finite and non-negative.")
    if torch.any(~torch.isinf(distance[~valid])) or torch.any(distance[~valid] < 0):
        raise ValueError("Unmatched distances must be +inf.")
    if torch.any(rank[valid] <= 0) or torch.any(rank[~valid] != 0):
        raise ValueError("Candidate rank must be positive for matches and zero otherwise.")
    matched_target = torch.nonzero(valid, as_tuple=False).flatten()
    matched_reference = target[valid]
    if torch.unique(matched_reference).numel() != matched_reference.numel():
        raise ValueError("A reference event may be paired to at most one target event.")
    if not torch.equal(inverse[matched_reference], matched_target):
        raise ValueError("reference_to_target is not the inverse of target_to_reference.")
    unused_reference = torch.ones(inverse.numel(), dtype=torch.bool)
    unused_reference[matched_reference] = False
    if torch.any(inverse[unused_reference] != -1):
        raise ValueError("Unused reference events must map to -1.")


def _tensor(artifact: Mapping[str, Any], name: str) -> torch.Tensor:
    """Extract one artifact tensor on the CPU."""
    value = artifact.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"Full pairing artifact {name} must be a tensor.")
    return value.detach().cpu()
