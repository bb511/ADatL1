from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path
import torch
import json


def load_object_feature_map(path: str) -> Dict[str, Dict[str, List[int]]]:
    """Return {object: {feature: [col_indices...]}} as plain Python lists."""
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    return {
        obj: {feat: list(indices) for feat, indices in feats.items()}
        for obj, feats in data.items()
    }


@dataclass
class BatchView:
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor | None = None
    l1bit: torch.Tensor | None = None


def unpack_batch(batch: Any) -> BatchView:
    if isinstance(batch, (tuple, list)):
        if len(batch) == 4:
            x, mask, l1bit, y = batch
            return BatchView(x=x, y=y, mask=mask, l1bit=l1bit)
        if len(batch) == 2:
            x, y = batch
            return BatchView(x=x, y=y)
    if isinstance(batch, dict):
        return BatchView(
            x=batch["x"],
            y=batch["y"],
            mask=batch.get("mask"),
            l1bit=batch.get("l1bit"),
        )
    raise ValueError(f"Unsupported batch format: {type(batch)}")
