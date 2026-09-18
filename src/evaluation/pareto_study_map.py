"""Build a Phase 2 study map by reading an experiment directory.

The Pareto collector consumes a study map: an explicit list of every
configuration and seed, each pointing at its resolved manifest and its
checkpoint directory. The study runner writes that list up front, from the grid
it is about to launch.

This module produces the same file the other way round, from what is actually on
disk. Stage 1 leaves a ``run_manifest.yaml`` in every run's checkpoint directory
(src/utils/run_manifest.py), so an experiment directory describes itself:

    checkpoints/<experiment_name>/
        <run_a>/run_manifest.yaml
        <run_b>/run_manifest.yaml
        ...

That is what lets autoencoders be trained one at a time, whenever and wherever
there is capacity, collected into one directory afterwards, and have the front
computed over whatever is there.

The collector is left untouched. It still validates every claim in the map
against the run's own resolved manifest and artifacts; this module only decides
which runs to put in front of it.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import argparse
import sys

from omegaconf import OmegaConf

#: Matches docs/evaluation/pareto_study_map.example.yaml.
STUDY_MAP_SCHEMA_VERSION = 1

RUN_MANIFEST_FILENAME = "run_manifest.yaml"


class StudyMapError(RuntimeError):
    """Raised when an experiment directory cannot be turned into a study map."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _resolved_config_path(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Locate the resolved config the collector must validate against.

    Prefer the copy stage 1 left beside the checkpoint: the Hydra run directory
    recorded in the manifest is a timestamped folder under logs/ that does not
    travel with the checkpoint when a batch job transfers its outputs.
    """
    recorded = (manifest.get("paths") or {}).get("resolved_config")
    candidates = []
    if recorded:
        candidates.append(Path(recorded))
    candidates.append(run_dir / "resolved_config.yaml")

    hydra_dir = (manifest.get("paths") or {}).get("hydra_output_dir")
    if hydra_dir and hydra_dir != "None":
        candidates.append(Path(hydra_dir) / "pareto_manifest.resolved.yaml")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise StudyMapError(
        f"{run_dir.name}: no resolved config found. Looked at "
        + ", ".join(str(c) for c in candidates)
    )


def discover_runs(experiment_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Return every run in the directory that stage 1 finished, sorted by name."""
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise StudyMapError(f"Not a directory: {experiment_dir}")

    found: List[Tuple[Path, Dict[str, Any]]] = []
    for manifest_path in sorted(experiment_dir.glob(f"*/{RUN_MANIFEST_FILENAME}")):
        found.append((manifest_path.parent, _load_yaml(manifest_path)))
    return found


def build_study_map(
    experiment_dir: Path,
    *,
    require_checkpoint: bool = True,
) -> Dict[str, Any]:
    """Assemble the study map for every completed run in one experiment directory."""
    experiment_dir = Path(experiment_dir).resolve()
    discovered = discover_runs(experiment_dir)
    if not discovered:
        raise StudyMapError(
            f"No {RUN_MANIFEST_FILENAME} found under {experiment_dir}. Runs trained "
            "before per-run manifests existed do not have one; retrain, or write "
            "the study map by hand."
        )

    runs: List[Dict[str, Any]] = []
    skipped: List[str] = []
    study_ids: set[str] = set()
    protocol_versions: set[str] = set()
    expected_seeds: set[Tuple[int, ...]] = set()
    seeds_by_configuration: Dict[str, List[int]] = defaultdict(list)

    for run_dir, manifest in discovered:
        run_name = str(manifest.get("run_name") or run_dir.name)

        if require_checkpoint and not (run_dir / "loss_total.ckpt").is_file():
            skipped.append(f"{run_name}: no loss_total.ckpt")
            continue

        resolved_path = _resolved_config_path(run_dir, manifest)
        resolved = _load_yaml(resolved_path)
        study = resolved.get("pareto_study")
        if not isinstance(study, Mapping) or study.get("enabled") is not True:
            # An honest skip rather than a silent one: the collector would reject
            # this run anyway, and saying so here points at the cause.
            skipped.append(
                f"{run_name}: resolved config carries no enabled pareto_study block"
            )
            continue

        configuration_id = str(study.get("configuration_id"))
        seed = manifest.get("autoencoder_seed")
        if seed is None:
            skipped.append(f"{run_name}: manifest records no autoencoder_seed")
            continue
        seed = int(seed)

        study_ids.add(str(study.get("study_id")))
        protocol_versions.add(str(study.get("protocol_version")))
        paired = study.get("paired_autoencoder_seeds") or []
        expected_seeds.add(tuple(int(s) for s in paired))
        seeds_by_configuration[configuration_id].append(seed)

        runs.append(
            {
                "configuration_id": configuration_id,
                "autoencoder_seed": seed,
                "manifest_path": str(resolved_path),
                "checkpoint_run_dir": str(run_dir),
            }
        )

    if not runs:
        raise StudyMapError(
            "No usable runs in "
            f"{experiment_dir}.\n  " + "\n  ".join(skipped)
        )

    # A study map has one identity. Mixing protocol versions in one directory is
    # a real mistake -- the results are not comparable -- so refuse rather than
    # silently picking one.
    if len(study_ids) != 1:
        raise StudyMapError(f"Runs disagree on study_id: {sorted(study_ids)}")
    if len(protocol_versions) != 1:
        raise StudyMapError(
            f"Runs disagree on protocol_version: {sorted(protocol_versions)}"
        )
    if len(expected_seeds) != 1:
        raise StudyMapError(
            "Runs disagree on paired_autoencoder_seeds: "
            f"{sorted(expected_seeds)}"
        )

    study_map = {
        "schema_version": STUDY_MAP_SCHEMA_VERSION,
        "study_id": study_ids.pop(),
        "protocol_version": protocol_versions.pop(),
        "expected_autoencoder_seeds": list(expected_seeds.pop()),
        "runs": sorted(
            runs, key=lambda r: (r["configuration_id"], r["autoencoder_seed"])
        ),
    }
    return {
        "study_map": study_map,
        "skipped": skipped,
        "seeds_by_configuration": dict(seeds_by_configuration),
    }


def report(result: Mapping[str, Any], *, stream=sys.stdout) -> None:
    """Print what went into the map, and what will happen to what did not."""
    study_map = result["study_map"]
    expected = study_map["expected_autoencoder_seeds"]
    seeds_by_configuration = result["seeds_by_configuration"]

    print(f"study_id          : {study_map['study_id']}", file=stream)
    print(f"protocol_version  : {study_map['protocol_version']}", file=stream)
    print(f"expected seeds    : {expected}", file=stream)
    print(f"runs in the map   : {len(study_map['runs'])}", file=stream)
    print(f"configurations    : {len(seeds_by_configuration)}", file=stream)

    # The aggregation needs at least two seeds per configuration to report a mean
    # and a confidence interval, and the feasibility rule requires every expected
    # seed. Saying so here turns a silent downstream rejection into a to-do list.
    incomplete = {
        configuration: sorted(seeds)
        for configuration, seeds in seeds_by_configuration.items()
        if set(seeds) != set(expected)
    }
    if incomplete:
        print("\nconfigurations missing an expected seed (they will be "
              "rejected by Phase 2):", file=stream)
        for configuration, seeds in sorted(incomplete.items()):
            missing = sorted(set(expected) - set(seeds))
            print(f"  {configuration}: have {seeds}, missing {missing}", file=stream)

    if result["skipped"]:
        print("\nskipped:", file=stream)
        for line in result["skipped"]:
            print(f"  {line}", file=stream)


def write_study_map(result: Mapping[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    OmegaConf.save(config=OmegaConf.create(result["study_map"]), f=tmp)
    tmp.replace(output_path)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Phase 2 study map from an experiment directory of trained "
            "runs, using the run_manifest.yaml each one carries."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        type=Path,
        help="checkpoints/<experiment_name>, containing one directory per run.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write study_map.yaml.",
    )
    parser.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help=(
            "Include runs whose loss_total.ckpt is absent. Off by default: such a "
            "run cannot have produced any artifact, so it only adds a rejection."
        ),
    )
    args = parser.parse_args(argv)

    try:
        result = build_study_map(
            args.experiment_dir,
            require_checkpoint=not args.allow_missing_checkpoints,
        )
    except StudyMapError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report(result)
    path = write_study_map(result, args.output)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
