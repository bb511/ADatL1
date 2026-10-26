#!/usr/bin/env python3
"""Print the fixed beta of every probe of the threshold posterior.

For every run directory (one holding ``meta.pt``) of the given lattices, the last
``epoch_*.pt`` gives the stored scores of validation ``normal`` (X1), of every X2 set and
of every probe of ``configs/cap/posterior.yaml`` (the ``aux`` sets). Per X2 set and
probe, the half-entropy beta is the beta at which the posterior p_1 over the grid of
normal and X2 has the entropy (1/2) log|T|, +inf where it keeps more as beta grows
(``half_entropy_beta``). Per probe, the fixed beta is the median over the runs and the
X2 sets pooled, +inf entering as +inf; a value whose scores are not all finite is left
out and counted. The last line is the ``beta_fix`` entry of ``posterior.yaml``, with
YAML's ``.inf`` for an infinite median. No label and no efficiency is read.

Usage::

    python scripts/analysis/posterior_beta_fix.py lattices/cchamber/<lattice> [...] \
        [--x2 sensor_weak sensor_mid sensor_strong]
"""
import argparse
import math
import statistics
from pathlib import Path

import rootutils
import torch
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks.threshold_posterior import costs, half_entropy_beta, threshold_grid

POSTERIOR = ROOT / "configs/cap/posterior.yaml"
X2_SETS = ["sensor_weak", "sensor_mid", "sensor_strong"]
SCORE = "ascore/full"


def _runs(lattices: list[Path]) -> list[Path]:
    metas = (meta for lattice in lattices for meta in lattice.glob("*/meta.pt"))
    return sorted(meta.parent for meta in metas if any(meta.parent.glob("epoch_*.pt")))


def half_entropy(scores_1, scores_2, probe, levels: int) -> float:
    """Return the half-entropy beta of p_1, NaN if a score is not finite."""
    if not all(bool(torch.isfinite(s).all()) for s in (scores_1, scores_2, probe)):
        return math.nan
    grid = threshold_grid(scores_1, scores_2, levels)
    return half_entropy_beta(costs(scores_1, probe, grid))


def epoch_betas(path: Path, x2_sets: list[str], config) -> dict[str, list[float]]:
    """Return per probe the half-entropy beta of every X2 set at one epoch file."""
    record = torch.load(path, weights_only=True)
    val, aux, levels = record["val"], record["aux"], config["levels"]
    normal = val["normal"][SCORE]
    return {
        probe: [
            half_entropy(normal, val[x2][SCORE], aux[probe][SCORE], levels)
            for x2 in x2_sets
        ]
        for probe in config["probes"]
    }


def betas(runs: list[Path], x2_sets: list[str], config) -> dict[str, list[float]]:
    """Return per probe the half-entropy betas of every run's last epoch, pooled."""
    values = {probe: [] for probe in config["probes"]}
    for run in runs:
        last = sorted(run.glob("epoch_*.pt"))[-1]
        for probe, run_values in epoch_betas(last, x2_sets, config).items():
            values[probe] += run_values
    return values


def _summary(probe: str, values: list[float]) -> tuple[str, float]:
    kept = [value for value in values if not math.isnan(value)]
    median = statistics.median(kept)
    infinite = sum(math.isinf(value) for value in kept)
    line = (
        f"{probe}: {len(kept)} values ({len(values) - len(kept)} non-finite left out, "
        f"{infinite} +inf), median beta = {median:.6g}"
    )
    return line, median


def _yaml(beta: float) -> str:
    return ".inf" if math.isinf(beta) else f"{beta:.6g}"


def report(values: dict[str, list[float]]) -> str:
    """Return one line per probe and the YAML line of the fixed betas."""
    summaries = {probe: _summary(probe, v) for probe, v in values.items()}
    entries = ", ".join(f"{p}: {_yaml(beta)}" for p, (_, beta) in summaries.items())
    lines = [line for line, _ in summaries.values()]
    return "\n".join(lines + [f"beta_fix: {{ {entries} }}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the posterior's fixed betas.")
    parser.add_argument("lattices", type=Path, nargs="+", help="lattice directories")
    parser.add_argument("--x2", nargs="+", default=X2_SETS, help="the X2 sets")
    args = parser.parse_args()
    runs = _runs(args.lattices)
    print(f"{len(runs)} runs, X2 sets {args.x2}")
    print(report(betas(runs, args.x2, OmegaConf.load(POSTERIOR))))


if __name__ == "__main__":
    main()
