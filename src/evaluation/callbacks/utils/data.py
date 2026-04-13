# Handing of different data sets at the level of the callbacks.
from dataclasses import dataclass
from typing import Any
import torch


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
