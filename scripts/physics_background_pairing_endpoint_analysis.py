#!/usr/bin/env python3
"""Join complete search-50 AUPRC and efficiency evidence into report tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.physics_background_pairing_endpoint_analysis import (
    write_endpoint_analysis,
)


def main() -> None:
    """Parse artifact paths, validate the exact join, and write the table bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provisional-dir", type=Path, required=True)
    parser.add_argument("--auprc-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = write_endpoint_analysis(
        args.provisional_dir.resolve(strict=True),
        args.auprc_aggregate.resolve(strict=True),
        args.output_dir.resolve(),
    )
    print(json.dumps(analysis["method"], sort_keys=True))


if __name__ == "__main__":
    main()
