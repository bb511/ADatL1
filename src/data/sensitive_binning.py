from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class FixedQuantileSensitiveBinner:
    """Fixed quantile binner for MI-sensitive variables.

    Purpose
    -------
    Compute bin edges once on the training split and then reuse the same edges
    during training, validation, and testing.

    Example
    -------
    variable = "FET.Et"

    This means:
        object  = "FET"
        feature = "Et"

    The class uses object_feature_map to find the flattened column index of FET.Et.
    """
    variable: str
    num_bins: int = 10
    reduction: str = "sum"
    use_denormalized: bool = True

    def __post_init__(self) -> None:
        if "." not in self.variable:
            raise ValueError(
                f"Sensitive variable must have format '<object>.<feature>', "
                f"for example 'FET.Et'. Got '{self.variable!r}'.")
        
        if self.num_bins < 2:
            raise ValueError(f"num_bins must be at least 2. Got {self.num_bins}.")
        
        allowed_reductions = {"sum", "mean", "first", "last", "min", "max"}
        if self.reduction not in allowed_reductions:
            raise ValueError(
                f"Unsupported reduction '{self.reduction}'. Allowed values are: {allowed_reductions}."
            )
        
        self.object_name, self.feature_name = self.variable.split(".")
        self.bin_edges: torch.Tensor | None = None

    @property
    def is_fitted(self) -> bool:
        return self.bin_edges is not None
    
    def fit(self, x: torch.Tensor, object_feature_map: dict, normalizer = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute fixed quantile bin edges from a full training tensor.

        Args:
            x:
                Full training tensor, already flattened or flattenable.
            object_feature_map:
                Mapping {object: {feature: [flattened_indices]}}.
            normalizer:
                Optional L1DataNormalizer. If available and use_denormalized=True,
                the sensitive variable is computed in physical units.
            mask:
                Optional full training mask.

        Returns:
            Tensor of shape [num_bins - 1] containing fixed bin edges.
        """

        values = self.extract_values(x=x, object_feature_map=object_feature_map, normalizer=normalizer, mask=mask)
        values = values[torch.isfinite(values)]  # Remove NaNs and Infs

        if values.numel() == 0:
            raise RunetimeError(
                f"No finite values found for sensitive variable {self.vatiable!r}."
            )
        
        quantiles = torch.linspace(0.0, 1.0, setps = self.num_bins + 1, device=values.device, dtype=values.dtype)[1:-1]

        edges = torch.quantile(values, quantiles).continguous()

        # torch.bucketize assumes monotonically increasing boundaries.
        # Quantiles can repeat if the variable has many identical values.
        edges = torch.sort(edges).values

        self.bin_edges = edges.detach.cpu()

        return self.bin_edges
    
    def transform(self, x: torch.Tensor, object_feature_map: dict, normalizer = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Convert a batch into discrete sensitive labels using fixed edges.

        Returns:
            Long tensor with shape [batch, 1].
        """
        if self.bin_edges is None:
            raise RuntimeError(
                f"FixedQuantileSensitiveBinner for {self.variable!r} is not fitted. "
                "Call fit(...) once on the training split before training."
            )
        
        values = self.extract_values(x=x, object_feature_map = object_feature_map, normalizer = normalizer, mask = mask)
        
        edges = self.bin_edges.to(device = values.device, dtype = values.dfype)
        labels = torch.bucketize(values.detach().flatten(), edges)

        return labels.to(dtype = torch.long).unsqueeze(1)
    
    def extract_values(self, x: torch.Tensor, object_feature_map: dict, normalizer = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Extract one scalar sensitive value per event."""
        x_flat = torch.flatten(x, start_dim=1)

        if mask is not None:
            mask_flat = torch.flatten(mask, start_dim=1).to(device=x_flat.device)
        else:
            mask_flat = None


        x_values = self._maybe_denormalize(x=x_flat, normalizer = normalizer, object_feature_map = object_feature_map)

        indices = self._resolve_indices(object_feature_map)
        index_tensor = torch.as_tensor(indices, device=x_flat.device, dtype=torch.long)

        selected = x_values.index_selected(dim=1, index = index_tensor)

        if mask_flat is not None:
            selected_mask = mask_flat.index_select(dim=1, index=index_tensor)
            selected = selected * selected_mask.to(dtype=selected.dtype)

        return self._reduce(selected)
    
    def _maybe_denormalize(self, x: torch.Tensor, normalizer, object_feature_map: dict) -> torch.Tensor:
        if not self.use_denormalized:
            return x.detach()
        
        if normalizer is None or not hasattr(normalizer, "denorm_1d_tensor"):
            return x.detach()
        
        x_phys = x.detach().clone()

        if (getattr(normalizer, "scale_tensor", None) is None or getattr(normalizer, "shift_tensor", None) is None):
            normalizer.setup_1d_denorm(object_feature_map)

        return normalizer.denorm_1d_tensor(x_phys)
    
    def _resolve_indices(self, object_feature_map: dict) -> list[int]:
        object_key = self._find_case_insensitive_key(object_feature_map, self.object_name, "object")

        feature_map = object_feature_map[object_key]

        feature_key = self._find_case_insensitive_key(feature_map, self.feature_name, f"feature for object {object_key!r}")

        indices = [int(i) for i in feature_map[feature_key]]

        if len(indices) == 0:
            raise RuntimeError(
                f"No indices found for sensitive variable {self.variable!r}."
            )
        
        return indices
    
    def _find_case_insensitive_key(self, mapping: dict, requested_key: str, kind: str) -> str:
        for key in mapping.keys():
            if str(key).lower() == requested_key.lower():
                return key
        
        raise KeyError(
            f"Could notfind {kind} {requested_key!r}. "
            f"Available keys: {list(mapping.keys())}"
        )
    
    def __reduce(self, selected: torch.Tensor) -> torch.Tensor:
        if selected.ndim != 2:
            raise ValueError(
                f"Expected selected sensitive tensor with shape [batch, n_columns], "
                f"got {tuple(selected.shape)}."
            )

        if self.reduction == "first":
            return selected[:, 0]

        if self.reduction == "sum":
            return selected.sum(dim=1)

        if self.reduction == "mean":
            return selected.mean(dim=1)

        if self.reduction == "max":
            return selected.max(dim=1).values

        raise RuntimeError(f"Unexpected reduction: {self.reduction!r}")