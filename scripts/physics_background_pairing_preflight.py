#!/usr/bin/env python3
"""Authenticate physics background-pairing inputs before launching searches."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.utils.pairing.physics_tables import load_split
from src.utils.pairing.table import load_pair_table, sha256_file, sha256_tensor

DATASET_1 = "ZB_run396102"
DATASET_2 = "ZB_run398183"
STRATEGIES = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
    "jetclr",
)


def parse_args() -> argparse.Namespace:
    """Parse cache, artifact, sample-size, and report-output arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "/iopsstor/scratch/cscs/podagiu/data/data_2025E+G/mlready/"
            "eminimalTauFET_pdefaultTauFET_default/robust"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "PHYSICS_PAIRING_DIR",
                "/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G/"
                "pairing/ZB_run396102_to_ZB_run398183",
            )
        ),
    )
    parser.add_argument("--events", type=int, default=163_840)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _source_counts(source_dir: Path) -> dict[str, dict[str, int]]:
    """Count each recorded source within every deterministic main-data split."""
    counts = {}
    for split in ("train", "valid", "test"):
        values = np.load(source_dir / split / "zerobias_source_id.npy", mmap_mode="r")
        unique, frequency = np.unique(values, return_counts=True)
        counts[split] = {str(int(key)): int(value) for key, value in zip(unique, frequency)}
    return counts


def run(cache_root: Path, artifact_dir: Path, events: int) -> dict:
    """Authenticate every pair table against the exact ordered input prefixes."""
    source_dir = artifact_dir / "sources"
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "events_per_source": int(events),
        "dataset_1": DATASET_1,
        "dataset_2": DATASET_2,
        "source_counts": _source_counts(source_dir),
        "tables": [],
    }
    for stage in ("validate", "test"):
        x1, _, _ = load_split(
            cache_root,
            stage,
            dataset=DATASET_1,
            limit=events,
            source_metadata_dir=source_dir,
        )
        x2, _, _ = load_split(
            cache_root,
            stage,
            dataset=DATASET_2,
            limit=events,
            source_metadata_dir=source_dir,
        )
        x1 = torch.flatten(x1, start_dim=1).contiguous()
        x2 = torch.flatten(x2, start_dim=1).contiguous()
        digest_1 = sha256_tensor(x1)
        digest_2 = sha256_tensor(x2)
        for strategy in STRATEGIES:
            path = artifact_dir / f"{stage}_{strategy}_cap_n{events}.pt"
            table = load_pair_table(
                path,
                expected_dataset_1=DATASET_1,
                expected_dataset_2=DATASET_2,
                expected_split=stage,
                n_dataset_1=events,
                n_dataset_2=events,
                source_1_sha256=digest_1,
                source_2_sha256=digest_2,
            )
            report["tables"].append(
                {
                    "stage": stage,
                    "strategy": strategy,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "pairs": int(table["idx_1"].numel()),
                    "source_1_sha256": digest_1,
                    "source_2_sha256": digest_2,
                    "authenticated": True,
                }
            )
        del x1, x2
    return report


def main() -> None:
    """Run the preflight and optionally persist its JSON report atomically."""
    args = parse_args()
    report = run(args.cache_root, args.artifact_dir, args.events)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
