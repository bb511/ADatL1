#!/usr/bin/env python3
"""Render the authenticated final physics confirmation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.physics_background_pairing_confirmation_report import (
    promote_confirmation_report,
    write_confirmation_report,
)
from src.utils.execution_provenance import authenticate_generator


def main() -> None:
    """Parse authenticated result paths and write the final report package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--execution-repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--discovery-random-analysis", type=Path)
    parser.add_argument("--confirmation-random-analysis", type=Path)
    parser.add_argument(
        "--confirmation-context",
        type=Path,
        help="Authenticated 50-to-200 and MLflow training-curve context package.",
    )
    parser.add_argument(
        "--report-mode",
        choices=("final", "provisional"),
        default="final",
        help="Final fails closed on all mandatory controls; provisional is visibly nonfinal.",
    )
    parser.add_argument(
        "--analysis-stage",
        choices=("primary", "final"),
        default="final",
        help="Bind the report to the primary reserve-decision or final effective-seed namespace.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="After verification, atomically install a final bundle as root REPORT.md.",
    )
    parser.add_argument(
        "--results-mirror",
        type=Path,
        help="Ignored download mirror required with --promote.",
    )
    args = parser.parse_args()
    generator = authenticate_generator(args.execution_repository, Path(__file__))
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
        confirmation_context=(
            args.confirmation_context.resolve(strict=True) if args.confirmation_context else None
        ),
        report_mode=args.report_mode,
        analysis_stage=args.analysis_stage,
        generator=generator,
    )
    promotion = None
    if args.promote:
        if args.report_mode != "final" or args.results_mirror is None:
            parser.error("--promote requires --report-mode final and --results-mirror")
        promotion = promote_confirmation_report(
            args.output_dir.resolve(strict=True),
            args.repository.resolve(strict=True),
            args.results_mirror.resolve(),
        )
    print(json.dumps({"summary": summary, "promotion": promotion}, sort_keys=True))


if __name__ == "__main__":
    main()
