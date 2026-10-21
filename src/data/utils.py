"""Helpers independent of any one dataset, shared by datamodules, models and callbacks.

Batch unpacking, archive download and extraction, seeded permutations, device transfer.
"""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
from colorama import Back

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


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


def seeded_permutation(n: int, seed: int) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def check_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"MD5 of {path} is {digest}, expected {expected}. Delete the file to "
            "download it again."
        )


def download(url: str, archive: Path) -> None:
    """Download to a process-specific name and rename it into place once complete.

    Runs started together on a cold data directory download concurrently, and the rename
    keeps any of them from reading another's partial file.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    log.info(Back.GREEN + f"Downloading {url}...")
    partial = archive.with_name(f"{archive.name}.{os.getpid()}")
    urlretrieve(url, partial)  # nosec B310, URL from the data config
    partial.replace(archive)


def extract(archive: Path, target: Path) -> None:
    """Extract into a staging directory and move the dataset into place whole.

    The archive holds a single top-level directory named like ``target``. When a
    concurrent run has moved its copy first, the rename fails and that copy is kept.
    """
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        with ZipFile(archive) as zf:
            zf.extractall(staging)
        try:
            (Path(staging) / target.name).rename(target)
        except OSError:
            if not target.exists():
                raise


def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Pin CPU tensors before an asynchronous CUDA copy."""
    if device.type != "cuda":
        return t.to(device)
    if t.device.type == "cpu" and not t.is_pinned():
        t = t.pin_memory()

    return t.to(device, non_blocking=True)
