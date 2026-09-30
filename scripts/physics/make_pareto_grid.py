#!/usr/bin/env python3
"""Expand the Pareto grid into one line per training run.

The grid lives in configs/pareto_study/fet_et.yaml and nowhere else. This script
reads it and writes the job list that batch/runae_pareto.sub consumes, so a
single condor_submit fans the whole study out as independent stage-1 jobs.

Generating the list rather than maintaining it by hand is not tidiness: Phase 2
validates every run against the manifest, so a hand-written list that drifts from
fet_et.yaml does not fail loudly, it makes the collector reject every run.

Run names are

    Seed<seed>_Gamma_<gamma>_Bins_<bins>_architecture_<arch>_Run<attempt>

The study is single-seed: every configuration is trained once, with the seed
in pareto_study.candidate.autoencoder_seed. A trailing attempt counter lets a
failed configuration be retrained as _Run02 without colliding with the remains
of the first attempt.

Output columns, comma separated for HTCondor's `queue ... from`:

    SEED,GAMMA,BINS,ARCH,NODES,RUN_NAME

NODES uses underscores (64_32_8) because a comma inside a field would break the
queue statement; batch/runae_pareto.sh turns it back into [64,32,8].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import argparse
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID = REPO_ROOT / "configs" / "pareto_study" / "fet_et.yaml"


def _format_gamma(value: float) -> str:
    """Render gamma the same way everywhere, so 0 and 0.0 cannot diverge."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def iter_runs(grid: Dict[str, Any], seed: int) -> Iterator[Tuple[Any, ...]]:
    """Yield (seed, gamma, bins, architecture_id, nodes) for every run."""
    search = grid["search_space"]
    baseline = search["gamma_zero_baseline"]

    # One gamma-zero baseline per architecture. Binning is irrelevant at
    # gamma=0, so the study fixes it at the canonical 50 rather than training
    # three identical baselines.
    for architecture_id, nodes in baseline["architectures"].items():
        yield (
            seed,
            float(baseline["mi_gamma"]),
            int(baseline["mi_sensitive_num_bins"]),
            architecture_id,
            list(nodes),
        )

    # `regularized` first, then the optional `refinement` block. The order keeps
    # the lines (and so the HTCondor ProcIds) of earlier submissions stable when
    # a refinement is appended. A grid point listed in both is emitted once.
    seen = set()
    for group in ("regularized", "refinement"):
        block = search.get(group)
        if not block:
            continue
        for architecture_id, nodes in block["architectures"].items():
            for gamma in block["mi_gamma"]:
                for bins in block["mi_sensitive_num_bins"]:
                    key = (float(gamma), int(bins), architecture_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield (seed, float(gamma), int(bins), architecture_id, list(nodes))


def run_name(seed: int, gamma: float, bins: int, architecture_id: str, attempt: str) -> str:
    return (
        f"Seed{seed}"
        f"_Gamma_{_format_gamma(gamma)}"
        f"_Bins_{bins}"
        f"_architecture_{architecture_id}"
        f"_Run{attempt}"
    )


def build_rows(grid_path: Path, attempt: str) -> List[Dict[str, Any]]:
    grid = OmegaConf.to_container(OmegaConf.load(grid_path), resolve=False)

    seed = grid.get("candidate", {}).get("autoencoder_seed")
    if seed is None:
        raise SystemExit(f"{grid_path} declares no candidate.autoencoder_seed.")
    seed = int(seed)

    contract = grid["search_space"].get("fixed_latent_width")
    rows: List[Dict[str, Any]] = []
    for seed, gamma, bins, architecture_id, nodes in iter_runs(grid, seed):
        if contract is not None and int(nodes[-1]) != int(contract):
            # The study fixes the latent width; a candidate that violates it is
            # rejected downstream, so refuse to generate it in the first place.
            raise SystemExit(
                f"architecture {architecture_id} ends in {nodes[-1]}, "
                f"but the study fixes the latent width at {contract}."
            )
        rows.append(
            {
                "seed": seed,
                "gamma": _format_gamma(gamma),
                "bins": bins,
                "architecture_id": architecture_id,
                "nodes": "_".join(str(int(n)) for n in nodes),
                "run_name": run_name(seed, gamma, bins, architecture_id, attempt),
            }
        )

    names = [row["run_name"] for row in rows]
    if len(set(names)) != len(names):
        raise SystemExit("Generated duplicate run names; the grid is malformed.")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument(
        "--attempt",
        default="01",
        help="Attempt counter in the run name. Use 02 to retrain failures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "batch" / "pareto_runs.txt",
        help="Job list for batch/runae_pareto.sub.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Substring filter on the run name, for retraining a subset. "
            "E.g. --only Gamma_0.1_."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help=(
            "Existing job list; its run names are left out. E.g. write only the "
            "runs a refinement added: --exclude batch/pareto_runs.txt."
        ),
    )
    args = parser.parse_args(argv)

    rows = build_rows(args.grid, args.attempt)
    if args.exclude:
        existing = {
            line.rsplit(",", 1)[-1].strip()
            for line in args.exclude.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if row["run_name"] not in existing]
        if not rows:
            raise SystemExit(f"Every run is already listed in {args.exclude}.")
    if args.only:
        rows = [row for row in rows if args.only in row["run_name"]]
        if not rows:
            raise SystemExit(f"No run matches --only {args.only!r}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            f"{r['seed']},{r['gamma']},{r['bins']},{r['architecture_id']},"
            f"{r['nodes']},{r['run_name']}\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    print(f"runs           : {len(rows)} (one per configuration)")
    print(f"seed           : {rows[0]['seed']}")
    print(f"wrote          : {args.output}")
    print()
    print("first and last lines:")
    lines = args.output.read_text().splitlines()
    for line in lines[:2] + ["  ..."] + lines[-2:]:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
