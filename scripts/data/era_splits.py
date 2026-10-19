#!/usr/bin/env python3
"""Derive the frozen splits of an era-only cache tree from those of the merged tree.

Why this exists
---------------
A fresh cache tree draws every split from one seeded stream (``mlready.py``), so an
era-only tree would redraw each auxiliary split differently from the merged 2025E+G
tree, and the haa-4b-ma15 split, reconstructed from the published record, cannot be
drawn at all. Writing the era tree's ``splits/`` before its first run avoids both: every
``aux__*.npz`` (21 for 2025E+G) is copied unchanged, and the era's ``zerobias.npz`` is
the merged ``itrain``/``ivalid``/``itest`` restricted to that era's rows and renumbered
from 0, each split keeping its merged order.

Row order
---------
``L1DataMLReady._prepare_maindata`` concatenates the zero-bias runs in sorted path order
(``_get_files_per_object`` sorts the files, and a pyarrow dataset built from a list of
files keeps the list's order), so the merged rows follow the sorted keys of
``paths.zerobias``: run 396102 (2025E) holds rows [0, n_first) and run 398183 (2025G)
rows [n_first, n). ``--n-first-run`` is n_first, 10,363,545 for 2025E+G. A wrong value
is caught when the era tree is first prepared, since ``_frozen_split`` refuses a split
whose total differs from the processed event count.

Not covered
-----------
The other era's zero-bias run as an auxiliary set needs a frozen split of its own,
``aux__ZB_run<run>.npz``, which this script does not write; without one, ``mlready.py``
draws and freezes it from its seeded stream.

Usage
-----
    python scripts/data/era_splits.py \\
        --merged-splits data/data_2025E+G/mlready/<name>/splits \\
        --era-splits data/data_2025E/mlready/<name>/splits \\
        --era E --n-first-run 10363545 --dry-run
"""

import argparse
import shutil
import sys
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


def report(
    era: str, zerobias: dict[str, np.ndarray], targets: list[Path], dry: bool
) -> None:
    sizes = ", ".join(f"{k[1:]} {len(i):,}" for k, i in zerobias.items())
    total = sum(len(i) for i in zerobias.values())
    print(f"era {era} zerobias: {sizes} ({total:,} events)")
    verb = "would write" if dry else "writing"
    for target in targets:
        print(f"{verb} {target}")


def write(era_dir: Path, zerobias: dict[str, np.ndarray], aux: list[Path]) -> None:
    era_dir.mkdir(parents=True, exist_ok=True)
    np.savez(era_dir / "zerobias.npz", **zerobias)
    for source in aux:
        shutil.copy2(source, era_dir / source.name)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--merged-splits",
        type=Path,
        required=True,
        help="splits/ of the merged 2025E+G tree",
    )
    ap.add_argument(
        "--era-splits",
        type=Path,
        required=True,
        help="splits/ of the era tree, written before its first run",
    )
    ap.add_argument("--era", choices=("E", "G"), required=True)
    ap.add_argument(
        "--n-first-run",
        type=int,
        required=True,
        help="zero-bias events of run 396102, the first in key order",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    merged = args.merged_splits / "zerobias.npz"
    zerobias = era_zerobias(merged, args.era, args.n_first_run)
    aux = sorted(args.merged_splits.glob("aux__*.npz"))
    targets = [args.era_splits / f.name for f in (merged, *aux)]
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        raise SystemExit("refusing to overwrite: " + ", ".join(existing))

    report(args.era, zerobias, targets, args.dry_run)
    if not args.dry_run:
        write(args.era_splits, zerobias, aux)
    return 0


if __name__ == "__main__":
    sys.exit(main())
