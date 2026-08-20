#!/usr/bin/env python3
"""Extract authenticated 50-to-200 and MLflow training-curve evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.physics_background_pairing_confirmation_context import (
    write_confirmation_context,
)
from src.utils.execution_provenance import authenticate_generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-analysis-dir", type=Path, required=True)
    parser.add_argument("--discovery-analysis-dir", type=Path, required=True)
    parser.add_argument("--mlflow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-repository", type=Path, required=True)
    args = parser.parse_args()
    generator = authenticate_generator(args.execution_repository, Path(__file__))
    print(
        json.dumps(
            write_confirmation_context(
                args.confirmation_analysis_dir,
                args.discovery_analysis_dir,
                args.mlflow_root,
                args.output_dir,
                generator=generator,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
