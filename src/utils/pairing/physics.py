"""Deterministic, physics-defined representations for L1 event pairing.

The representations in this module are controls for learned event embeddings.  They
operate in physical L1 units, treat azimuth as a periodic coordinate, retain padding
information explicitly, and fit all metric scales on a reference sample only.

Two complementary controls are provided:

``flat_physical``
    A deliberately simple data-space baseline.  It keeps object slots, but replaces
    each phi value by sine/cosine and adds an activity bit per slot.

``physics_summary``
    A permutation- and global-rotation-invariant event descriptor.  Each object type
    is represented by multiplicity, transverse-energy, eta, and recoil summaries.
    The FET direction is used as the event's azimuthal reference when available.

``typed_sliced_wasserstein``
    A scalable linearized optimal-transport control.  Energy-weighted quantiles of
    deterministic projections of each typed eta-phi point cloud form a Euclidean
    sliced-Wasserstein embedding suitable for exact FAISS L2 retrieval.

The fitted state is made only of tensors and JSON-compatible values so it can be
stored beside a pair table and replayed exactly during another experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

PHYSICS_DESCRIPTOR_STATE_VERSION = 1
_KINEMATIC_FEATURES = ("Et", "eta", "phi")
_DESCRIPTOR_KINDS = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
)
_TRANSPORT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
_TRANSPORT_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.577350269, 0.577350269, 0.577350269),
    (0.577350269, 0.577350269, -0.577350269),
    (0.577350269, -0.577350269, 0.577350269),
    (-0.577350269, 0.577350269, 0.577350269),
)


@dataclass(frozen=True)
class PhysicsFeatureSchema:
    """Mapping and calibration needed to recover physical L1 quantities.

    ``normalization_shift`` and ``normalization_scale`` undo the preprocessing
    normalization.  ``l1_scales`` then converts integer hardware coordinates to
    GeV/radians (and eta units).  Features absent from ``l1_scales`` are regarded as
    structural padding; this is how the dummy FET eta column is excluded.
    """

    object_feature_map: Mapping[str, Mapping[str, Sequence[int]]]
    l1_scales: Mapping[str, Mapping[str, float]]
    normalization_shift: torch.Tensor | None = None
    normalization_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Canonicalize and validate all schema inputs."""
        feature_map = _canonical_feature_map(self.object_feature_map)
        scales = _canonical_l1_scales(self.l1_scales)
        n_features = _mapped_feature_count(feature_map)
        shift = _calibration_tensor(self.normalization_shift, n_features, fill=0.0)
        scale = _calibration_tensor(self.normalization_scale, n_features, fill=1.0)
        if not torch.isfinite(shift).all() or not torch.isfinite(scale).all():
            raise ValueError("Normalization calibration must be finite.")
        if torch.any(scale <= 0):
            raise ValueError("Normalization scales must be strictly positive.")

        object.__setattr__(self, "object_feature_map", feature_map)
        object.__setattr__(self, "l1_scales", scales)
        object.__setattr__(self, "normalization_shift", shift)
        object.__setattr__(self, "normalization_scale", scale)

    @property
    def n_features(self) -> int:
        """Return the required flattened input width."""
        return int(self.normalization_shift.numel())

    @property
    def object_types(self) -> tuple[str, ...]:
        """Return object families in deterministic order."""
        # Sorting makes the descriptor independent of JSON/dict insertion order.
        return tuple(sorted(self.object_feature_map))

    @classmethod
    def from_normalizer(
        cls,
        object_feature_map: Mapping[str, Mapping[str, Sequence[int]]],
        l1_scales: Mapping[str, Mapping[str, float]],
        normalizer: Any,
    ) -> PhysicsFeatureSchema:
        """Construct a schema from the repository's ``L1DataNormalizer``."""
        normalizer.setup_1d_denorm(object_feature_map)
        return cls(
            object_feature_map=object_feature_map,
            l1_scales=l1_scales,
            normalization_shift=normalizer.shift_tensor.detach().cpu().clone(),
            normalization_scale=normalizer.scale_tensor.detach().cpu().clone(),
        )

    def active_features(self, object_type: str) -> tuple[str, ...]:
        """Return mapped kinematics with a defined physical calibration."""
        mapping = self.object_feature_map[object_type]
        scales = self.l1_scales.get(object_type, {})
        return tuple(name for name in _KINEMATIC_FEATURES if name in mapping and name in scales)

    def physical_feature(
        self,
        flat_x: torch.Tensor,
        object_type: str,
        feature: str,
    ) -> torch.Tensor:
        """Return one ``[event, slot]`` feature array in physical units."""
        if feature not in self.active_features(object_type):
            raise KeyError(f"No calibrated {object_type}/{feature} feature in schema.")
        indices = torch.as_tensor(
            self.object_feature_map[object_type][feature], dtype=torch.long, device=flat_x.device
        )
        shift = self.normalization_shift.to(device=flat_x.device, dtype=flat_x.dtype)[indices]
        scale = self.normalization_scale.to(device=flat_x.device, dtype=flat_x.dtype)[indices]
        l1_scale = float(self.l1_scales[object_type][feature])
        return (flat_x[:, indices] * scale + shift) * l1_scale

    def signature(self) -> str:
        """Stable digest protecting fitted state against a different data schema."""
        payload = {
            "object_feature_map": self.object_feature_map,
            "l1_scales": self.l1_scales,
            "normalization_shift": self.normalization_shift.tolist(),
            "normalization_scale": self.normalization_scale.tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PhysicsDescriptorState:
    """Serializable robust-scaling state fitted on reference background only."""

    kind: str
    schema_signature: str
    feature_names: tuple[str, ...]
    feature_blocks: tuple[str, ...]
    center: torch.Tensor
    scale: torch.Tensor
    block_weights: Mapping[str, float]
    canonicalize_flat: bool = False
    quantile_range: tuple[float, float] = (0.25, 0.75)
    version: int = PHYSICS_DESCRIPTOR_STATE_VERSION

    def __post_init__(self) -> None:
        """Validate state tensors and copy them to contiguous CPU storage."""
        if self.version != PHYSICS_DESCRIPTOR_STATE_VERSION:
            raise ValueError(f"Unsupported physics descriptor state version {self.version}.")
        if self.kind not in _DESCRIPTOR_KINDS:
            raise ValueError(f"Unknown physics descriptor kind {self.kind!r}.")
        n_features = len(self.feature_names)
        if len(self.feature_blocks) != n_features:
            raise ValueError("Every descriptor feature must have an object block.")
        if self.center.shape != (n_features,) or self.scale.shape != (n_features,):
            raise ValueError("Descriptor center and scale must have one value per feature.")
        if not torch.isfinite(self.center).all() or not torch.isfinite(self.scale).all():
            raise ValueError("Descriptor center and scale must be finite.")
        if torch.any(self.scale <= 0):
            raise ValueError("Descriptor scales must be strictly positive.")
        qlow, qhigh = self.quantile_range
        if not 0.0 <= qlow < qhigh <= 1.0:
            raise ValueError("quantile_range must satisfy 0 <= low < high <= 1.")
        for block in set(self.feature_blocks):
            weight = float(self.block_weights.get(block, 1.0))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"Block weight for {block!r} must be finite and positive.")

        object.__setattr__(self, "center", self.center.detach().cpu().float().contiguous())
        object.__setattr__(self, "scale", self.scale.detach().cpu().float().contiguous())
        object.__setattr__(
            self, "block_weights", {str(k): float(v) for k, v in self.block_weights.items()}
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a tensor-and-primitive representation for ``torch.save``."""
        return {
            "version": self.version,
            "kind": self.kind,
            "schema_signature": self.schema_signature,
            "feature_names": list(self.feature_names),
            "feature_blocks": list(self.feature_blocks),
            "center": self.center.clone(),
            "scale": self.scale.clone(),
            "block_weights": dict(self.block_weights),
            "canonicalize_flat": self.canonicalize_flat,
            "quantile_range": list(self.quantile_range),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> PhysicsDescriptorState:
        """Reconstruct and validate fitted state from its serialized form."""
        return cls(
            version=int(state["version"]),
            kind=str(state["kind"]),
            schema_signature=str(state["schema_signature"]),
            feature_names=tuple(state["feature_names"]),
            feature_blocks=tuple(state["feature_blocks"]),
            center=torch.as_tensor(state["center"]),
            scale=torch.as_tensor(state["scale"]),
            block_weights=dict(state["block_weights"]),
            canonicalize_flat=bool(state.get("canonicalize_flat", False)),
            quantile_range=tuple(float(q) for q in state.get("quantile_range", (0.25, 0.75))),
        )


class PhysicsPairingDescriptor:
    """Fit and apply a deterministic Euclidean physics pairing metric."""

    def __init__(
        self,
        schema: PhysicsFeatureSchema,
        *,
        kind: str = "physics_summary",
        block_weights: Mapping[str, float] | None = None,
        canonicalize_flat: bool = False,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        fit_max_events: int | None = 200_000,
    ) -> None:
        if kind not in _DESCRIPTOR_KINDS:
            raise ValueError(f"kind must be one of {_DESCRIPTOR_KINDS}, got {kind!r}.")
        if fit_max_events is not None and int(fit_max_events) <= 0:
            raise ValueError("fit_max_events must be positive or None.")
        qlow, qhigh = quantile_range
        if not 0.0 <= qlow < qhigh <= 1.0:
            raise ValueError("quantile_range must satisfy 0 <= low < high <= 1.")
        weights = {str(k): float(v) for k, v in (block_weights or {}).items()}
        if any(not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError("All block weights must be finite and positive.")

        self.schema = schema
        self.kind = kind
        self.block_weights = weights
        self.canonicalize_flat = bool(canonicalize_flat)
        self.quantile_range = (float(qlow), float(qhigh))
        self.fit_max_events = None if fit_max_events is None else int(fit_max_events)
        self.state: PhysicsDescriptorState | None = None

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None) -> PhysicsPairingDescriptor:
        """Fit robust metric scales using a deterministic subset of reference rows."""
        if not torch.is_tensor(x) or x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("x must be a non-empty tensor with an event dimension.")
        indices = _evenly_spaced_indices(x.shape[0], self.fit_max_events, x.device)
        selected_mask = None if mask is None else mask[indices]
        # Select before flattening, casting, and the finite scan.  Production train
        # caches contain O(10M) events, whereas a reproducible 200k-event reference
        # sample is sufficient to fix this low-dimensional metric calibration.
        flat_x, flat_mask = self._prepare_inputs(x[indices], selected_mask)
        raw, names, blocks = self._raw_transform(flat_x, flat_mask)
        raw = raw.detach().cpu().float()
        if not torch.isfinite(raw).all():
            raise ValueError("Cannot fit descriptor scaling on non-finite values.")

        qlow, qhigh = torch.quantile(
            raw,
            torch.tensor(self.quantile_range, dtype=raw.dtype),
            dim=0,
        )
        center = torch.quantile(raw, 0.5, dim=0)
        scale = qhigh - qlow
        # Sparse lower-rank slots can have a zero IQR.  Standard deviation is a
        # deterministic, data-derived fallback and avoids assigning them GeV^-1.
        fallback = torch.std(raw, dim=0, correction=0)
        scale = torch.where(scale > torch.finfo(raw.dtype).eps, scale, fallback)
        scale = torch.where(scale > torch.finfo(raw.dtype).eps, scale, torch.ones_like(scale))

        self.state = PhysicsDescriptorState(
            kind=self.kind,
            schema_signature=self.schema.signature(),
            feature_names=names,
            feature_blocks=blocks,
            center=center,
            scale=scale,
            block_weights=self.block_weights,
            canonicalize_flat=self.canonicalize_flat,
            quantile_range=self.quantile_range,
        )
        return self

    def transform(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Return the fitted, block-balanced descriptor used by L2 retrieval."""
        if self.state is None:
            raise RuntimeError("Call fit or load_state_dict before transform.")
        if self.state.schema_signature != self.schema.signature():
            raise ValueError("Fitted descriptor state belongs to a different physics schema.")
        flat_x, flat_mask = self._prepare_inputs(x, mask)
        raw, names, blocks = self._raw_transform(flat_x, flat_mask)
        if names != self.state.feature_names or blocks != self.state.feature_blocks:
            raise ValueError("Fitted descriptor feature layout changed since fit.")
        center = self.state.center.to(device=raw.device, dtype=raw.dtype)
        scale = self.state.scale.to(device=raw.device, dtype=raw.dtype)
        result = (raw - center) / scale

        # Equal block weight means equal expected influence per object family, not
        # equal weight per coordinate.  This prevents high-multiplicity collections
        # from dominating merely because they have more padded slots.
        for block in sorted(set(blocks)):
            columns = torch.tensor(
                [i for i, value in enumerate(blocks) if value == block],
                dtype=torch.long,
                device=result.device,
            )
            weight = float(self.state.block_weights.get(block, 1.0))
            result[:, columns] *= weight / math.sqrt(columns.numel())
        return result

    def fit_transform(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Fit reference-only scaling and transform the same events."""
        return self.fit(x, mask).transform(x, mask)

    def state_dict(self) -> dict[str, Any]:
        """Serialize the fitted descriptor state."""
        if self.state is None:
            raise RuntimeError("Cannot serialize an unfitted descriptor.")
        return self.state.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load fitted state after strict strategy and schema checks."""
        loaded = PhysicsDescriptorState.from_state_dict(state)
        if loaded.schema_signature != self.schema.signature():
            raise ValueError("Descriptor state was fitted with a different physics schema.")
        if loaded.kind != self.kind:
            raise ValueError(f"Descriptor state kind is {loaded.kind!r}, expected {self.kind!r}.")
        if loaded.canonicalize_flat != self.canonicalize_flat:
            raise ValueError("Descriptor state canonicalize_flat setting does not match.")
        self.state = loaded

    def raw_transform(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Expose interpretable physical values for diagnostics (without fitted scaling)."""
        flat_x, flat_mask = self._prepare_inputs(x, mask)
        values, names, _ = self._raw_transform(flat_x, flat_mask)
        return values, names

    def _prepare_inputs(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten, cast, and validate an event batch and its mask."""
        if not torch.is_tensor(x) or x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("x must be a non-empty tensor with an event dimension.")
        flat_x = torch.flatten(x, start_dim=1).float()
        if flat_x.shape[1] != self.schema.n_features:
            raise ValueError(
                f"Input has {flat_x.shape[1]} flattened features, "
                f"schema requires {self.schema.n_features}."
            )
        if mask is None:
            flat_mask = torch.ones_like(flat_x, dtype=torch.bool)
        else:
            flat_mask = torch.flatten(mask, start_dim=1).bool()
            if flat_mask.shape != flat_x.shape:
                raise ValueError("mask must have the same flattened shape as x.")
        if not torch.isfinite(flat_x).all():
            raise ValueError("x must contain only finite values.")
        return flat_x, flat_mask

    def _raw_transform(
        self, flat_x: torch.Tensor, flat_mask: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[str, ...], tuple[str, ...]]:
        """Dispatch to the selected unscaled physical representation."""
        if self.kind == "flat_physical":
            return self._flat_physical(flat_x, flat_mask)
        if self.kind == "physics_summary":
            return self._physics_summary(flat_x, flat_mask)
        return self._typed_sliced_wasserstein(flat_x, flat_mask)

    def _flat_physical(
        self, flat_x: torch.Tensor, flat_mask: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[str, ...], tuple[str, ...]]:
        """Build the masked slot-level data-space control."""
        columns: list[torch.Tensor] = []
        names: list[str] = []
        blocks: list[str] = []
        for object_type in self.schema.object_types:
            features = self.schema.active_features(object_type)
            if not features:
                continue
            arrays = {
                feature: self.schema.physical_feature(flat_x, object_type, feature)
                for feature in features
            }
            feature_masks = {
                feature: _feature_mask(flat_mask, self.schema, object_type, feature)
                for feature in features
            }
            presence = feature_masks.get("Et", torch.stack(tuple(feature_masks.values())).any(0))
            if self.canonicalize_flat:
                sort_values = arrays.get("Et", torch.zeros_like(presence, dtype=flat_x.dtype))
                order = torch.argsort(
                    torch.where(presence, -sort_values, torch.full_like(sort_values, torch.inf)),
                    dim=1,
                    stable=True,
                )
                presence = torch.gather(presence, 1, order)
                arrays = {key: torch.gather(value, 1, order) for key, value in arrays.items()}
                feature_masks = {
                    key: torch.gather(value, 1, order) for key, value in feature_masks.items()
                }

            n_slots = presence.shape[1]
            for slot in range(n_slots):
                columns.append(presence[:, slot].to(flat_x.dtype))
                names.append(f"{object_type}[{slot}].present")
                blocks.append(object_type)
                for feature in features:
                    value = arrays[feature][:, slot]
                    active = feature_masks[feature][:, slot]
                    if feature == "phi":
                        columns.extend(
                            (
                                torch.where(active, torch.sin(value), torch.zeros_like(value)),
                                torch.where(active, torch.cos(value), torch.zeros_like(value)),
                            )
                        )
                        names.extend(
                            (f"{object_type}[{slot}].sin_phi", f"{object_type}[{slot}].cos_phi")
                        )
                        blocks.extend((object_type, object_type))
                    else:
                        columns.append(torch.where(active, value, torch.zeros_like(value)))
                        names.append(f"{object_type}[{slot}].{feature}")
                        blocks.append(object_type)
        return torch.stack(columns, dim=1), tuple(names), tuple(blocks)

    def _physics_summary(
        self, flat_x: torch.Tensor, flat_mask: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[str, ...], tuple[str, ...]]:
        """Build typed event-level activity and recoil summaries."""
        columns: list[torch.Tensor] = []
        names: list[str] = []
        blocks: list[str] = []

        fet_phi: torch.Tensor | None = None
        fet_active: torch.Tensor | None = None
        if "FET" in self.schema.object_feature_map and "phi" in self.schema.active_features("FET"):
            fet_phi = self.schema.physical_feature(flat_x, "FET", "phi")[:, 0]
            fet_active = _feature_mask(flat_mask, self.schema, "FET", "phi")[:, 0]

        def add(block: str, name: str, value: torch.Tensor) -> None:
            columns.append(value)
            names.append(f"{block}.{name}")
            blocks.append(block)

        for object_type in self.schema.object_types:
            features = self.schema.active_features(object_type)
            if "Et" not in features:
                continue
            et = self.schema.physical_feature(flat_x, object_type, "Et")
            active = _feature_mask(flat_mask, self.schema, object_type, "Et")
            et = torch.where(active, et.clamp_min(0), torch.zeros_like(et))
            count = active.sum(dim=1).to(flat_x.dtype)
            sum_et = et.sum(dim=1)
            # FET is one event-level missing-energy vector, not an object
            # collection.  Its direction enters the recoil terms below; repeating
            # its Et as scalar/leading/mean would triple-count the same observable.
            if object_type == "FET":
                add(object_type, "et", sum_et)
                continue
            safe_count = count.clamp_min(1)
            mean_et = sum_et / safe_count
            variance_et = ((et - mean_et[:, None]).square() * active).sum(dim=1) / safe_count
            ranked_et = torch.sort(et, dim=1, descending=True, stable=True).values

            add(object_type, "multiplicity", count)
            add(object_type, "scalar_et", sum_et)
            add(object_type, "leading_et", ranked_et[:, 0])
            if ranked_et.shape[1] > 1:
                add(object_type, "subleading_et", ranked_et[:, 1])
            add(object_type, "mean_et", mean_et)
            add(object_type, "std_et", torch.sqrt(variance_et.clamp_min(0)))

            if "eta" in features:
                eta = self.schema.physical_feature(flat_x, object_type, "eta")
                eta_active = active & _feature_mask(flat_mask, self.schema, object_type, "eta")
                weights = torch.where(eta_active, et, torch.zeros_like(et))
                weight_sum = weights.sum(dim=1).clamp_min(torch.finfo(flat_x.dtype).eps)
                mean_eta = (weights * eta).sum(dim=1) / weight_sum
                var_eta = (weights * (eta - mean_eta[:, None]).square()).sum(dim=1) / weight_sum
                add(object_type, "et_weighted_eta", mean_eta)
                add(object_type, "et_weighted_abs_eta", (weights * eta.abs()).sum(1) / weight_sum)
                add(object_type, "et_weighted_eta_std", torch.sqrt(var_eta.clamp_min(0)))
                central_et = (weights * (eta.abs() < 1.5)).sum(dim=1)
                add(object_type, "central_et_fraction", central_et / weight_sum)

            if "phi" in features and object_type != "FET":
                phi = self.schema.physical_feature(flat_x, object_type, "phi")
                phi_active = active & _feature_mask(flat_mask, self.schema, object_type, "phi")
                weights = torch.where(phi_active, et, torch.zeros_like(et))
                px = (weights * torch.cos(phi)).sum(dim=1)
                py = (weights * torch.sin(phi)).sum(dim=1)
                resultant = torch.sqrt(px.square() + py.square())
                add(object_type, "vector_et", resultant)
                add(object_type, "vector_et_fraction", resultant / sum_et.clamp_min(1e-12))
                if fet_phi is not None and fet_active is not None:
                    delta_phi = phi - fet_phi[:, None]
                    reference_ok = phi_active & fet_active[:, None]
                    ref_weights = torch.where(reference_ok, et, torch.zeros_like(et))
                    add(
                        object_type, "et_parallel_fet", (ref_weights * torch.cos(delta_phi)).sum(1)
                    )
                    add(object_type, "et_cross_fet", (ref_weights * torch.sin(delta_phi)).sum(1))

        return torch.stack(columns, dim=1), tuple(names), tuple(blocks)

    def _typed_sliced_wasserstein(
        self, flat_x: torch.Tensor, flat_mask: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[str, ...], tuple[str, ...]]:
        """Build a linearized transport descriptor with deterministic projections.

        The ground coordinates are ``eta / 5``, ``cos(delta_phi)``, and
        ``sin(delta_phi)``.  The first puts the full L1 eta acceptance on a scale
        comparable to the unit-circle chord distance.  Quantile features represent
        each normalized transverse-energy measure, while multiplicity and scalar Et
        retain information intentionally removed by mass normalization.
        """
        if (
            "FET" not in self.schema.object_feature_map
            or "phi" not in self.schema.active_features("FET")
        ):
            raise ValueError("typed_sliced_wasserstein requires calibrated FET phi.")
        fet_phi = self.schema.physical_feature(flat_x, "FET", "phi")[:, 0]
        fet_active = _feature_mask(flat_mask, self.schema, "FET", "phi")[:, 0]
        directions = torch.tensor(_TRANSPORT_DIRECTIONS, dtype=flat_x.dtype, device=flat_x.device)
        quantiles = torch.tensor(_TRANSPORT_QUANTILES, dtype=flat_x.dtype, device=flat_x.device)

        columns: list[torch.Tensor] = []
        names: list[str] = []
        blocks: list[str] = []

        def add(block: str, name: str, value: torch.Tensor) -> None:
            columns.append(value)
            names.append(f"{block}.{name}")
            blocks.append(block)

        for object_type in self.schema.object_types:
            features = self.schema.active_features(object_type)
            if "Et" not in features:
                continue
            et = self.schema.physical_feature(flat_x, object_type, "Et")
            active = _feature_mask(flat_mask, self.schema, object_type, "Et")
            et = torch.where(active, et.clamp_min(0), torch.zeros_like(et))
            scalar_et = et.sum(dim=1)
            if object_type == "FET":
                add(object_type, "et", scalar_et)
                continue

            add(object_type, "multiplicity", active.sum(dim=1).to(flat_x.dtype))
            add(object_type, "scalar_et", scalar_et)
            if "eta" not in features or "phi" not in features:
                continue
            eta = self.schema.physical_feature(flat_x, object_type, "eta")
            phi = self.schema.physical_feature(flat_x, object_type, "phi")
            coordinate_mask = (
                active
                & fet_active[:, None]
                & _feature_mask(flat_mask, self.schema, object_type, "eta")
                & _feature_mask(flat_mask, self.schema, object_type, "phi")
            )
            weights = torch.where(coordinate_mask, et, torch.zeros_like(et))
            total_weight = weights.sum(dim=1, keepdim=True)
            normalized_weights = weights / total_weight.clamp_min(torch.finfo(flat_x.dtype).eps)
            delta_phi = phi - fet_phi[:, None]
            coordinates = torch.stack(
                (eta / 5.0, torch.cos(delta_phi), torch.sin(delta_phi)), dim=2
            )
            # [event, slot, direction]
            projected = coordinates @ directions.T
            for direction_index in range(directions.shape[0]):
                values = projected[:, :, direction_index]
                ordered_values, order = torch.sort(values, dim=1, stable=True)
                ordered_weights = torch.gather(normalized_weights, 1, order)
                cdf = torch.cumsum(ordered_weights, dim=1)
                query = quantiles[None, :].expand(flat_x.shape[0], -1).contiguous()
                indices = torch.searchsorted(cdf.contiguous(), query, right=False)
                indices = indices.clamp_max(values.shape[1] - 1)
                weighted_quantiles = torch.gather(ordered_values, 1, indices)
                weighted_quantiles = torch.where(
                    total_weight > torch.finfo(flat_x.dtype).eps,
                    weighted_quantiles,
                    torch.zeros_like(weighted_quantiles),
                )
                for quantile_index, quantile in enumerate(_TRANSPORT_QUANTILES):
                    add(
                        object_type,
                        f"sw_p{direction_index}_q{quantile:g}",
                        weighted_quantiles[:, quantile_index],
                    )

        return torch.stack(columns, dim=1), tuple(names), tuple(blocks)


def _canonical_feature_map(
    feature_map: Mapping[str, Mapping[str, Sequence[int]]],
) -> dict[str, dict[str, list[int]]]:
    """Copy a feature map and enforce a contiguous, unique flat layout."""
    if not feature_map:
        raise ValueError("object_feature_map cannot be empty.")
    result: dict[str, dict[str, list[int]]] = {}
    seen: list[int] = []
    for object_type in sorted(feature_map):
        if not feature_map[object_type]:
            raise ValueError(f"Feature map for {object_type!r} cannot be empty.")
        result[str(object_type)] = {}
        for feature in sorted(feature_map[object_type]):
            indices = [int(index) for index in feature_map[object_type][feature]]
            if not indices or any(index < 0 for index in indices):
                raise ValueError(f"Feature indices for {object_type}/{feature} are invalid.")
            result[str(object_type)][str(feature)] = indices
            seen.extend(indices)
    if sorted(seen) != list(range(len(seen))):
        raise ValueError(
            "object_feature_map indices must cover each flattened column exactly once."
        )
    return result


def _canonical_l1_scales(
    scales: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Copy physical scale metadata while validating every defined value."""
    result: dict[str, dict[str, float]] = {}
    for object_type in sorted(scales):
        result[str(object_type)] = {}
        for feature in sorted(scales[object_type]):
            value = float(scales[object_type][feature])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"L1 scale for {object_type}/{feature} must be finite and positive."
                )
            result[str(object_type)][str(feature)] = value
    return result


def _mapped_feature_count(feature_map: Mapping[str, Mapping[str, Sequence[int]]]) -> int:
    """Count flattened columns in a validated feature map."""
    return sum(len(indices) for mapping in feature_map.values() for indices in mapping.values())


def _calibration_tensor(value: torch.Tensor | None, size: int, *, fill: float) -> torch.Tensor:
    """Return an owned one-dimensional calibration tensor."""
    if value is None:
        return torch.full((size,), fill, dtype=torch.float32)
    tensor = torch.as_tensor(value).detach().cpu().float().contiguous()
    if tensor.shape != (size,):
        raise ValueError(
            f"Calibration tensor has shape {tuple(tensor.shape)}, expected {(size,)}."
        )
    return tensor


def _feature_mask(
    flat_mask: torch.Tensor,
    schema: PhysicsFeatureSchema,
    object_type: str,
    feature: str,
) -> torch.Tensor:
    """Gather a typed feature's activity mask."""
    indices = torch.as_tensor(
        schema.object_feature_map[object_type][feature], dtype=torch.long, device=flat_mask.device
    )
    return flat_mask[:, indices]


def _evenly_spaced_indices(
    n_events: int,
    maximum: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Select a deterministic sample spanning an ordered cache."""
    if maximum is None or n_events <= maximum:
        return torch.arange(n_events, device=device)
    # Integer arithmetic is bitwise reproducible and includes the full split rather
    # than fitting only the first events from a shuffled cache.
    return torch.div(
        torch.arange(maximum, device=device, dtype=torch.long) * n_events,
        maximum,
        rounding_mode="floor",
    )
