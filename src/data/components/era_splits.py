"""Derive the frozen splits of an era-only cache tree from those of the merged tree.

Usage, from a Python session at the repository root::

    from src.data.components.era_splits import plan_era_splits, write_era_splits
    zerobias, aux = plan_era_splits(merged_splits, era_splits, "E", 10363545)
    write_era_splits(era_splits, zerobias, aux)

Here ``merged_splits`` and ``era_splits`` are the ``splits/`` directories of the merged
2025E+G tree and of the era tree, as ``Path`` objects. ``plan_era_splits`` writes
nothing, so calling it alone is a dry run.
"""

import shutil
from pathlib import Path

import numpy as np

SPLITS = ("itrain", "ivalid", "itest")


def restrict(indices: np.ndarray, era: str, n_first: int) -> np.ndarray:
    """Keep the merged rows of one era, renumbered from 0 within it."""
    if era == "E":
        return indices[indices < n_first]
    return indices[indices >= n_first] - n_first


def era_zerobias(merged_file: Path, era: str, n_first: int) -> dict[str, np.ndarray]:
    with np.load(merged_file) as merged:
        return {k: restrict(merged[k], era, n_first) for k in SPLITS}


def plan_era_splits(
    merged_dir: Path, era_dir: Path, era: str, n_first: int
) -> tuple[dict[str, np.ndarray], list[Path]]:
    """Return the era zero-bias splits and the aux files to copy, writing nothing.

    Raises:
        FileExistsError: if any target in ``era_dir`` exists already.
    """
    merged = merged_dir / "zerobias.npz"
    zerobias = era_zerobias(merged, era, n_first)
    aux = sorted(merged_dir.glob("aux__*.npz"))
    targets = [era_dir / f.name for f in (merged, *aux)]
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    return zerobias, aux


def write_era_splits(
    era_dir: Path, zerobias: dict[str, np.ndarray], aux: list[Path]
) -> None:
    era_dir.mkdir(parents=True, exist_ok=True)
    np.savez(era_dir / "zerobias.npz", **zerobias)
    for source in aux:
        shutil.copy2(source, era_dir / source.name)
