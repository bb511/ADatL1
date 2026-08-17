#!/usr/bin/env python3
"""Freeze or validate deterministic random-bijection CAP control tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.pairing.random_tables import (
    freeze_random_pairing_tables,
    validate_frozen_random_pairing_tables,
)


def parse_args() -> argparse.Namespace:
    """Parse the campaign root and fail-closed operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--namespace", default="random_pairing")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("command", choices=("freeze", "validate"))
    return parser.parse_args()


def main() -> None:
    """Freeze new tables or reauthenticate an existing frozen directory."""
    args = parse_args()
    if args.command == "freeze":
        manifest = freeze_random_pairing_tables(
            args.root, cache_root=args.cache_root, namespace=args.namespace
        )
    else:
        manifest = validate_frozen_random_pairing_tables(
            args.root, manifest_path=args.root / args.namespace / "manifest.json"
        )
    print(
        json.dumps(
            {
                "random_pairing": str(args.root / args.namespace),
                "tables": len(manifest["tables"]),
                "content_sha256": manifest["content_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
