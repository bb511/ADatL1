"""L1 layout helpers for converting flattened hardware features to physical units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

FeatureMap = Mapping[str, Mapping[str, Sequence[int]]]
ScaleMap = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class L1PhysicalLayout:
    """Hardware LSBs aligned with a flattened L1 feature tensor.

    ``defined`` distinguishes physical features from structural columns introduced
    by schema unification, such as the empty FET eta column.  ``scale`` converts
    denormalized integer hardware values to GeV or angular coordinates; callers must
    undo dataset normalization before applying it.
    """

    scale: torch.Tensor
    defined: torch.Tensor

    def apply(self, hardware_values: torch.Tensor) -> torch.Tensor:
        """Convert already-denormalized hardware values to physical units."""
        if hardware_values.shape[-1] != self.scale.numel():
            raise ValueError(
                "Last tensor dimension does not match the flattened L1 layout: "
                f"{hardware_values.shape[-1]} != {self.scale.numel()}."
            )
        return hardware_values * self.scale.to(
            device=hardware_values.device, dtype=hardware_values.dtype
        )


def build_l1_physical_layout(
    object_feature_map: FeatureMap,
    l1_scales: ScaleMap,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> L1PhysicalLayout:
    """Align configured hardware LSBs with flattened feature indices.

    Features present only to unify array shapes are left at scale one and marked undefined.  All
    mapped indices must be unique, non-negative integers.
    """
    indices = [
        int(idx)
        for feature_map in object_feature_map.values()
        for feature_indices in feature_map.values()
        for idx in feature_indices
    ]
    if not indices:
        return L1PhysicalLayout(
            scale=torch.empty(0, dtype=dtype, device=device),
            defined=torch.empty(0, dtype=torch.bool, device=device),
        )
    if min(indices) < 0:
        raise ValueError("L1 feature indices must be non-negative.")
    if len(indices) != len(set(indices)):
        raise ValueError("L1 feature indices must be unique.")

    size = max(indices) + 1
    scale = torch.ones(size, dtype=dtype, device=device)
    defined = torch.zeros(size, dtype=torch.bool, device=device)

    for object_name, feature_map in object_feature_map.items():
        object_scales = l1_scales.get(object_name, {})
        for feature_name, feature_indices in feature_map.items():
            if feature_name not in object_scales:
                continue
            idx = torch.as_tensor(feature_indices, dtype=torch.long, device=device)
            scale[idx] = float(object_scales[feature_name])
            defined[idx] = True

    return L1PhysicalLayout(scale=scale, defined=defined)
