from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FixedQuantileSensitiveBinner:
    """Fixed quantile binner for MI-sensitive variables.

    Computes bin edges once on the training split and reuses the same edges
    during training, validation, and testing.

    Example:
        variable="FET.Et"

    This means:
        object  = "FET"
        feature = "Et"
    """

    variable: str
    num_bins: int = 10
    reduction: str = "first"
    use_denormalized: bool = False

    def __post_init__(self) -> None:
        if "." not in self.variable:
            raise ValueError(
                f"Sensitive variable must have format '<object>.<feature>', "
                f"for example 'FET.Et'. Got {self.variable!r}."
            )

        if self.num_bins < 2:
            raise ValueError(f"num_bins must be at least 2. Got {self.num_bins}.")

        self.reduction = self.reduction.lower()

        allowed_reductions = {"sum", "mean", "first", "last", "min", "max"}
        if self.reduction not in allowed_reductions:
            raise ValueError(
                f"Unsupported reduction {self.reduction!r}. "
                f"Allowed values are: {sorted(allowed_reductions)}."
            )

        self.object_name, self.feature_name = self.variable.split(".", maxsplit=1)
        self.bin_edges: torch.Tensor | None = None
        self.fit_stats: dict[
            str,
            float | int | list[float] | list[int],
        ] = {}

    @property
    def is_fitted(self) -> bool:
        return self.bin_edges is not None

    def fit(
        self,
        x: torch.Tensor,
        object_feature_map: dict,
        normalizer=None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute fixed quantile bin edges from the full training tensor."""
        values = self.extract_values(
            x=x,
            object_feature_map=object_feature_map,
            normalizer=normalizer,
            mask=mask,
        )

        values = values.detach().flatten()
        values = values[torch.isfinite(values)]

        if values.numel() == 0:
            raise RuntimeError(
                f"No finite values found for sensitive variable {self.variable!r}."
            )

        values_for_quantile = values.float().cpu()

        quantiles = torch.linspace(
            0.0,
            1.0,
            steps=self.num_bins + 1,
            dtype=values_for_quantile.dtype,
        )[1:-1]

        edges = torch.quantile(values_for_quantile, quantiles).contiguous()
        edges = torch.sort(edges).values

        unique_edges = torch.unique_consecutive(edges)

        if unique_edges.numel() != edges.numel():
            print(
                f"[MI][warning] Requested {self.num_bins} bins for {self.variable}, "
                f"but quantile edges contain duplicates. Effective bins: "
                f"{unique_edges.numel() + 1}."
            )

        self.bin_edges = unique_edges.detach().cpu()
        num_effective_bins = int(self.bin_edges.numel() + 1)
        num_unique_values = int(torch.unique(values_for_quantile).numel())

        labels = torch.bucketize(values_for_quantile, self.bin_edges)
        counts = torch.bincount(labels, minlength=num_effective_bins)
        value_min = float(values_for_quantile.min().item())
        value_max = float(values_for_quantile.max().item())
        histogram_min = value_min
        histogram_max = value_max
        if histogram_min == histogram_max:
            half_width = max(abs(histogram_min) * 1e-3, 1e-6)
            histogram_min -= half_width
            histogram_max += half_width

        raw_histogram_counts = torch.histc(
            values_for_quantile,
            bins=num_effective_bins,
            min=histogram_min,
            max=histogram_max,
        )
        raw_histogram_edges = torch.linspace(
            histogram_min,
            histogram_max,
            steps=num_effective_bins + 1,
            dtype=values_for_quantile.dtype,
        )
        unique_value_histogram_counts = torch.histc(
            values_for_quantile,
            bins=num_unique_values,
            min=histogram_min,
            max=histogram_max,
        )
        unique_value_histogram_edges = torch.linspace(
            histogram_min,
            histogram_max,
            steps=num_unique_values + 1,
            dtype=values_for_quantile.dtype,
        )

        self.fit_stats = {
            "num_values": int(values_for_quantile.numel()),
            "num_unique_values": num_unique_values,
            "num_bins_requested": int(self.num_bins),
            "num_bins_effective": num_effective_bins,
            "min": value_min,
            "max": value_max,
            "mean": float(values_for_quantile.mean().item()),
            "std": float(values_for_quantile.std(unbiased=False).item()),
            "edges": [float(v) for v in self.bin_edges.tolist()],
            "counts": [int(v) for v in counts.tolist()],
            "raw_histogram_bins": num_effective_bins,
            "raw_histogram_counts": [
                int(v) for v in raw_histogram_counts.tolist()
            ],
            "raw_histogram_edges": [
                float(v) for v in raw_histogram_edges.tolist()
            ],
            "unique_value_histogram_bins": num_unique_values,
            "unique_value_histogram_counts": [
                int(v) for v in unique_value_histogram_counts.tolist()
            ],
            "unique_value_histogram_edges": [
                float(v) for v in unique_value_histogram_edges.tolist()
            ],
        }

        return self.bin_edges

    def transform(
        self,
        x: torch.Tensor,
        object_feature_map: dict,
        normalizer=None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert a batch into fixed-bin sensitive labels.

        Returns:
            Long tensor with shape [batch, 1].
        """
        if self.bin_edges is None:
            raise RuntimeError(
                f"FixedQuantileSensitiveBinner for {self.variable!r} is not fitted. "
                "Call fit(...) once on the training split before training."
            )

        values = self.extract_values(
            x=x,
            object_feature_map=object_feature_map,
            normalizer=normalizer,
            mask=mask,
        )

        return self.transform_values(values)

    def transform_values(self, values: torch.Tensor) -> torch.Tensor:
        """Convert already extracted sensitive values using the fixed bin edges."""
        if self.bin_edges is None:
            raise RuntimeError(
                f"FixedQuantileSensitiveBinner for {self.variable!r} is not fitted. "
                "Call fit(...) once on the training split before training."
            )

        edges = self.bin_edges.to(device=values.device, dtype=values.dtype)
        labels = torch.bucketize(values.detach().flatten(), edges)

        return labels.to(dtype=torch.long).unsqueeze(1)

    def extract_values(
        self,
        x: torch.Tensor,
        object_feature_map: dict,
        normalizer=None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract one scalar sensitive value per event."""
        x_flat = torch.flatten(x, start_dim=1)

        if mask is not None:
            mask_flat = torch.flatten(mask, start_dim=1).to(device=x_flat.device)
        else:
            mask_flat = None

        x_values = self._maybe_denormalize(
            x=x_flat,
            normalizer=normalizer,
            object_feature_map=object_feature_map,
        )

        indices = self._resolve_indices(object_feature_map)
        index_tensor = torch.as_tensor(indices, device=x_values.device, dtype=torch.long)

        selected = x_values.index_select(dim=1, index=index_tensor)

        if mask_flat is not None:
            selected_mask = mask_flat.index_select(dim=1, index=index_tensor)
            selected = selected * selected_mask.to(dtype=selected.dtype)

        return self._reduce(selected)

    def _maybe_denormalize(
        self,
        x: torch.Tensor,
        normalizer,
        object_feature_map: dict,
    ) -> torch.Tensor:
        if not self.use_denormalized:
            return x.detach()

        if normalizer is None or not hasattr(normalizer, "denorm_1d_tensor"):
            return x.detach()

        x_phys = x.detach().clone()

        if (
            getattr(normalizer, "scale_tensor", None) is None
            or getattr(normalizer, "shift_tensor", None) is None
        ):
            normalizer.setup_1d_denorm(object_feature_map)

        return normalizer.denorm_1d_tensor(x_phys)

    def _resolve_indices(self, object_feature_map: dict) -> list[int]:
        object_key = self._find_case_insensitive_key(
            object_feature_map,
            self.object_name,
            "object",
        )

        feature_map = object_feature_map[object_key]

        feature_key = self._find_case_insensitive_key(
            feature_map,
            self.feature_name,
            f"feature for object {object_key!r}",
        )

        indices = [int(i) for i in feature_map[feature_key]]

        if len(indices) == 0:
            raise RuntimeError(
                f"No indices found for sensitive variable {self.variable!r}."
            )

        return indices

    def _find_case_insensitive_key(
        self,
        mapping: dict,
        requested_key: str,
        kind: str,
    ) -> str:
        for key in mapping.keys():
            if str(key).lower() == requested_key.lower():
                return key

        raise KeyError(
            f"Could not find {kind} {requested_key!r}. "
            f"Available keys: {list(mapping.keys())}"
        )

    def _reduce(self, selected: torch.Tensor) -> torch.Tensor:
        if selected.ndim != 2:
            raise ValueError(
                f"Expected selected sensitive tensor with shape [batch, n_columns], "
                f"got {tuple(selected.shape)}."
            )

        if self.reduction == "first":
            return selected[:, 0]

        if self.reduction == "last":
            return selected[:, -1]

        if self.reduction == "sum":
            return selected.sum(dim=1)

        if self.reduction == "mean":
            return selected.mean(dim=1)

        if self.reduction == "min":
            return selected.min(dim=1).values

        if self.reduction == "max":
            return selected.max(dim=1).values

        raise RuntimeError(f"Unexpected reduction: {self.reduction!r}")
