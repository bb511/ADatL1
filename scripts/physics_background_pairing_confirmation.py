#!/usr/bin/env python3
"""Preview or explicitly freeze physics background-pairing confirmation targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.physics_background_pairing_confirmation_manifest import (
    build_freeze_payload,
    freeze_confirmation_manifest,
)


def main() -> None:
    """Run the confirmation manifest command-line interface."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "freeze"):
        command = subparsers.add_parser(name)
        command.add_argument("--campaign-root", type=Path, required=True)
        command.add_argument("--scope", choices=("native", "all"), required=True)
        if name == "freeze":
            command.add_argument("--expected-preview-sha256", required=True)
    args = parser.parse_args()
    if args.command == "preview":
        print(
            json.dumps(
                build_freeze_payload(args.campaign_root, scope=args.scope),
                indent=2,
                sort_keys=True,
            )
        )
        return
    output = freeze_confirmation_manifest(
        args.campaign_root,
        scope=args.scope,
        expected_preview_sha256=args.expected_preview_sha256,
    )
    print(output)


if __name__ == "__main__":
    main()
