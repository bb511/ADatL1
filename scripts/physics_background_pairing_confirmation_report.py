#!/usr/bin/env python3
"""Render the authenticated final physics confirmation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.physics_background_pairing_confirmation_report import (
    write_confirmation_report,
)


def main() -> None:
    """Parse authenticated result paths and write the final report package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--discovery-random-analysis", type=Path)
    parser.add_argument("--confirmation-random-analysis", type=Path)
    args = parser.parse_args()
    summary = write_confirmation_report(
        args.analysis_dir.resolve(strict=True),
        args.repository.resolve(strict=True),
        args.output_dir.resolve(),
        discovery_random_analysis=(
            args.discovery_random_analysis.resolve(strict=True)
            if args.discovery_random_analysis
            else None
        ),
        confirmation_random_analysis=(
            args.confirmation_random_analysis.resolve(strict=True)
            if args.confirmation_random_analysis
            else None
        ),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
