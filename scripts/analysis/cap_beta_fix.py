#!/usr/bin/env python3
"""Print the fixed beta of every evidence map of the CAP bank.

For every run directory (one holding ``meta.pt``) of the given lattices, the scores of
validation ``normal`` in the last ``epoch_*.pt`` are mapped to evidence z by every map
of ``configs/cap/bank.yaml``, and q, the 0.95 quantile of abs(z), is taken. Per map, the
fixed beta is 2 artanh(0.9) / q at the median q over the runs, so that
tanh(beta q / 2) = 0.9; runs whose q is not finite are left out of the median and
counted. The last line is the ``beta_fix`` entry of ``bank.yaml``. No label and no
efficiency is read.

Usage::

    python scripts/analysis/cap_beta_fix.py logs/scores/<lattice> [...]
"""
import argparse
import math
import statistics
from pathlib import Path

import rootutils
import torch
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks.components.cap.bank import MAPS

BANK = ROOT / "configs/cap/bank.yaml"


def beta_fix(q: float) -> float:
    return 2.0 * math.atanh(0.9) / q


def evidence_quantile(scores: torch.Tensor, name: str, params) -> float:
    """Return the 0.95 quantile of abs(z), z the evidence of the scores under name."""
    z = MAPS[name](scores, **params)
    return torch.quantile(z.abs(), 0.95).item()


def _runs(lattices: list[Path]) -> list[Path]:
    metas = (meta for lattice in lattices for meta in lattice.glob("*/meta.pt"))
    return sorted(meta.parent for meta in metas if any(meta.parent.glob("epoch_*.pt")))


def _last_scores(run: Path) -> torch.Tensor:
    record = torch.load(sorted(run.glob("epoch_*.pt"))[-1], weights_only=True)
    return record["val"]["normal"]["ascore/full"]


def quantiles(runs: list[Path], maps) -> dict[str, list[float]]:
    """Return q of every run, per map."""
    values = {name: [] for name in maps}
    for run in runs:
        scores = _last_scores(run)
        for name, params in maps.items():
            values[name].append(evidence_quantile(scores, name, params))
    return values


def _summary(name: str, qs: list[float]) -> tuple[str, float]:
    finite = [q for q in qs if math.isfinite(q)]
    beta = beta_fix(statistics.median(finite))
    line = (
        f"{name}: {len(finite)} runs ({len(qs) - len(finite)} non-finite left out), "
        f"median q = {statistics.median(finite):.4f}, beta_fix = {beta:.4f}"
    )
    return line, beta


def report(values: dict[str, list[float]]) -> str:
    """Return one line per map and the YAML line of the fixed betas."""
    summaries = {name: _summary(name, qs) for name, qs in values.items()}
    entries = ", ".join(f"{name}: {beta:.4f}" for name, (_, beta) in summaries.items())
    lines = [line for line, _ in summaries.values()]
    return "\n".join(lines + [f"beta_fix: {{ {entries} }}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the CAP bank's fixed betas.")
    parser.add_argument("lattices", type=Path, nargs="+", help="logs/scores/<lattice>")
    runs = _runs(parser.parse_args().lattices)
    print(report(quantiles(runs, OmegaConf.load(BANK)["maps"])))


if __name__ == "__main__":
    main()
