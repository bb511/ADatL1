#!/usr/bin/env python3
"""Produce authenticated random-CAP negative-control result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.physics_background_pairing_random_cap_analysis import (
    write_random_cap_analysis,
)


def main() -> None:
    """Parse inputs and write authenticated random-CAP analysis artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_random_cap_analysis(
        args.manifest.resolve(strict=True),
        args.aggregate.resolve(strict=True),
        args.output_dir.resolve(),
    )
    print(json.dumps(result["method"], sort_keys=True))


if __name__ == "__main__":
    main()
