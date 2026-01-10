from pathlib import Path
from typing import Dict, List, Optional
import json

import torch
import torch.nn as nn

from src.data.utils import load_object_feature_map


class ParticleMasking(nn.Module):
    """
    Column-wise masking:
      - Input x: (batch_size, n_features)
      - For each object in object_probs:
          with prob p per sample, all columns of that object are set to mask_value.
    """

    def __init__(
        self,
        object_feature_map_path: str,
        object_probs: Optional[Dict[str, float]] = None,
        mask_value: float = 0.0,
        training_only: bool = True,
    ):
        super().__init__()
        self.feature_map = load_object_feature_map(object_feature_map_path)
        self.object_probs = dict(object_probs or {})
        self.mask_value = float(mask_value)
        self.training_only = bool(training_only)

        # Precompute list of column indices per object
        self.object_columns: Dict[str, List[int]] = {}
        for obj, feats in self.feature_map.items():
            cols: List[int] = []
            for indices in feats.values():
                cols.extend(indices)
            self.object_columns[obj] = sorted(set(cols))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training_only and not self.training:
            return x

        if x.dim() != 2:
            raise ValueError(
                f"ParticleMasking expects (batch, features), got {tuple(x.shape)}"
            )

        if not self.object_probs:
            return x

        batch_size, _ = x.shape
        x_masked = x.clone()
        device = x.device

        for obj, prob in self.object_probs.items():
            if prob <= 0.0:
                continue

            cols = self.object_columns.get(obj)
            if not cols:
                continue

            cols_t = torch.tensor(cols, dtype=torch.long, device=device)
            row_mask = torch.rand(batch_size, device=device) < prob
            if not row_mask.any():
                continue

            row_idx = row_mask.nonzero(as_tuple=False).squeeze(1)
            x_masked[row_idx[:, None], cols_t] = self.mask_value

        return x_masked


class MultiplicityMasking(nn.Module):
    """
    Row-wise masking based on MET.Et:
      - Input x: (batch_size, n_features)
      - Take MET.Et column from object_feature_map["MET"]["Et"][0]
      - Compute percentile over the batch.
      - Rows with MET.Et > threshold are set entirely to mask_value.
    """

    def __init__(
        self,
        object_feature_map_path: str,
        percentile: float = 75.0,
        mask_value: float = 0.0,
        training_only: bool = True,
    ):
        super().__init__()
        self.feature_map = load_object_feature_map(object_feature_map_path)
        self.percentile = float(percentile)
        self.mask_value = float(mask_value)
        self.training_only = bool(training_only)

        # MET.Et is a single-column list in the JSON
        self.met_et_col = int(self.feature_map["MET"]["Et"][0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training_only and not self.training:
            return x

        if x.dim() != 2:
            raise ValueError(
                f"MultiplicityMasking expects (batch, features), got {tuple(x.shape)}"
            )

        if self.percentile <= 0.0:
            return x

        met_et = x[:, self.met_et_col]
        if met_et.numel() <= 1:
            return x

        threshold = torch.quantile(met_et, self.percentile / 100.0)
        row_mask = met_et > threshold
        if not row_mask.any():
            return x

        x_masked = x.clone()
        x_masked[row_mask] = self.mask_value
        return x_masked
