#!/usr/bin/env python3
"""Produce paired confirmation endpoint tables from authenticated results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.physics_background_pairing_confirmation_analysis import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    write_confirmation_endpoint_analysis,
)


def main() -> None:
    """Parse paths and write the fail-closed portable result package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--endpoint-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    result = write_confirmation_endpoint_analysis(
        args.matrix.resolve(strict=True),
        args.endpoint_rows.resolve(strict=True),
        args.output_dir.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result["method"], sort_keys=True))


if __name__ == "__main__":
    main()
