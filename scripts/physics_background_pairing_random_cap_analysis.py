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
    parser.add_argument(
        "--output-version",
        help="Optional single-component child directory for a new immutable analysis version.",
    )
    parser.add_argument(
        "--endpoint-analysis-receipt",
        type=Path,
        help="Hash-authenticated endpoint_analysis.json used for downstream rank associations.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.output_version is not None:
        version = Path(args.output_version)
        if version.name != args.output_version or args.output_version in {"", ".", ".."}:
            parser.error("--output-version must be one non-special path component")
        output_dir /= args.output_version
    result = write_random_cap_analysis(
        args.manifest.resolve(strict=True),
        args.aggregate.resolve(strict=True),
        output_dir,
        (
            args.endpoint_analysis_receipt.resolve(strict=True)
            if args.endpoint_analysis_receipt is not None
            else None
        ),
    )
    print(json.dumps(result["method"], sort_keys=True))


if __name__ == "__main__":
    main()
