#!/usr/bin/env python3
"""Evaluate a frozen VAE trajectory without creating inference-mode buffers.

The frozen campaign's score helper moves the model inside ``torch.inference_mode``.
That turns residual-state buffers into inference tensors after the first branch and
prevents the next branch checkpoint from being loaded.  Moving the model once before
entering that helper preserves ordinary buffers while leaving score computation fully
inference-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.cchamber_vae_reporting_campaign import _frozen_module


def evaluate(root: Path, trajectory_index: int) -> Path:
    """Run one authenticated frozen evaluation with a safe device transition."""
    frozen = _frozen_module()
    inference_scores = frozen._scores

    def safe_scores(model, loader, score_name, device):
        """Move ordinary model state before the decorated inference helper runs."""
        model.eval().to(device)
        return inference_scores(model, loader, score_name, device)

    frozen._scores = safe_scores
    return frozen.evaluate(root.expanduser().resolve(), int(trajectory_index))


def main() -> None:
    """Parse one trajectory and run its safe frozen evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, required=True)
    args = parser.parse_args()
    print(evaluate(args.root, args.trajectory_index))


if __name__ == "__main__":
    main()
