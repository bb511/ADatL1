#!/usr/bin/env python3
"""Redraw the loss PNGs of finished runs (``plots/losses/``) with the current layout.

``LossesCallback`` writes ``training_losses.png``, ``training_losses_unnormalized.png``
and ``mi_vs_permuted_null.png`` at the end of training from the MLflow metric
histories. This script redraws them later from the local MLflow file store, with
the same plotting code (``src.plot.loss_curves``), and writes them

* over the PNGs in ``<checkpoint run>/plots/losses/``, and
* over the copies in the MLflow run's ``artifacts/losses/`` (only if that folder
  exists, i.e. the callback logged them).

Each positional argument is a checkpoint run folder or a folder of them (e.g.
``checkpoints/Pareto-Front-261002``). Only run folders that already have
``plots/losses/`` are redrawn.

MLflow run of a checkpoint folder, first hit wins:

* ``links.yaml`` (``mlflow.experiment_id`` / ``mlflow.run_id``);
* otherwise the one active run named like the folder, in the active experiment(s)
  named like the parent folder, that has ``train/loss`` and is not tagged
  ``link.role = other training attempt``. No match or several: reported, skipped.

γ (title and ``Loss_mi * gamma``) comes from ``resolved_config.yaml``
(``algorithm.mi_gamma``) via ``read_mi_hyperparameters``; unknown γ: skipped.
Metric files are read directly (``<timestamp> <value> <step>`` per line), so neither
mlflow nor torch is needed.

The PNGs are drawn in a temporary folder first and copied over the old files only
when all of a run's figures were drawn.

Examples::

    python src/analysis/scripts/redo_loss_plots.py checkpoints/Pareto-Front-261002 --dry-run
    python src/analysis/scripts/redo_loss_plots.py checkpoints/Pareto-Front-261002 \\
        checkpoints/Pareto-Front-261015 --jobs 4
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.analysis.run_mi_hyperparameters import read_mi_hyperparameters  # noqa: E402
from src.plot import loss_curves  # noqa: E402


DEFAULT_MLRUNS_ROOT = REPO_ROOT / "logs" / "mlflow" / "mlruns"
LOSSES_FOLDER = "losses"  # LossesCallback(name="losses"): plot and artifact folder
OTHER_ATTEMPT = "other training attempt"


class MetricPoint(NamedTuple):
    """One line of an MLflow FileStore metric file."""

    timestamp: int
    value: float
    step: int


@dataclass
class RunResult:
    checkpoint_dir: Path
    status: str  # "ok", "dry-run" or "skipped"
    message: str
    written: list[Path] = field(default_factory=list)


# --------------------------------------------------------------------------- store


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _read_tag(run_dir: Path, key: str) -> str | None:
    path = run_dir / "tags" / key
    return path.read_text().strip() if path.is_file() else None


def read_metric_points(run_dir: Path, key: str) -> list[MetricPoint]:
    """All logged points of metric ``key`` (e.g. ``train/loss_mi``) of a run."""
    path = run_dir / "metrics" / key
    if not path.is_file():
        return []
    points = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        step = int(fields[2]) if len(fields) > 2 else 0
        points.append(MetricPoint(int(fields[0]), float(fields[1]), step))
    return points


def read_history(run_dir: Path, key: str) -> dict[int, float] | None:
    """Step-aligned history as the callback builds it; None if absent or not finite."""
    points = read_metric_points(run_dir, key)
    if not points:
        return None
    try:
        return loss_curves.history_by_step(points)
    except ValueError:
        return None


@lru_cache(maxsize=None)
def _experiments_by_name(mlruns_root: Path) -> dict[str, tuple[Path, ...]]:
    """Active experiment folders of the store, grouped by experiment name."""
    found: dict[str, list[Path]] = {}
    for meta in sorted(mlruns_root.glob("*/meta.yaml")):
        content = _read_yaml(meta)
        if content.get("lifecycle_stage", "active") != "active":
            continue
        found.setdefault(str(content.get("name")), []).append(meta.parent)
    return {name: tuple(dirs) for name, dirs in found.items()}


def find_mlflow_run(checkpoint_dir: Path, mlruns_root: Path) -> tuple[Path | None, str]:
    """MLflow run folder of a checkpoint run folder, and how it was found."""
    links = _read_yaml(checkpoint_dir / "links.yaml").get("mlflow") or {}
    experiment_id, run_id = links.get("experiment_id"), links.get("run_id")
    if experiment_id and run_id:
        run_dir = mlruns_root / str(experiment_id) / str(run_id)
        if run_dir.is_dir():
            return run_dir, "links.yaml"
        return None, f"links.yaml names run {run_id}, which is not in the store"

    experiment = checkpoint_dir.parent.name
    candidates = []
    for experiment_dir in _experiments_by_name(mlruns_root).get(experiment, ()):
        for meta in experiment_dir.glob("*/meta.yaml"):
            content = _read_yaml(meta)
            run_dir = meta.parent
            if (
                str(content.get("run_name")) == checkpoint_dir.name
                and content.get("lifecycle_stage", "active") == "active"
                and _read_tag(run_dir, "link.role") != OTHER_ATTEMPT
                and (run_dir / "metrics" / "train" / "loss").is_file()
            ):
                candidates.append(run_dir)
    if len(candidates) == 1:
        return candidates[0], "run name"
    if not candidates:
        return None, f"no active run named like the folder in experiment {experiment!r}"
    names = ", ".join(sorted(run.name for run in candidates))
    return None, f"{len(candidates)} active runs named like the folder: {names}"


# --------------------------------------------------------------------------- redraw


def discover_checkpoint_dirs(roots: Iterable[Path]) -> list[Path]:
    """Run folders with ``plots/losses/``: each root itself or its direct children."""
    found = []
    for root in roots:
        root = Path(root)
        candidates = [root] if (root / "plots" / LOSSES_FOLDER).is_dir() else sorted(
            child for child in root.iterdir() if child.is_dir()
        )
        found.extend(
            run for run in candidates if (run / "plots" / LOSSES_FOLDER).is_dir()
        )
    return found


def redraw_run(
    checkpoint_dir: Path,
    mlruns_root: Path = DEFAULT_MLRUNS_ROOT,
    dry_run: bool = False,
    preview_dir: Path | None = None,
) -> RunResult:
    """Redraw the loss PNGs of one checkpoint run folder.

    With ``preview_dir`` the PNGs go to ``preview_dir/<experiment>/<run>/`` and
    nothing in the checkpoint folder or the store is touched.
    """
    checkpoint_dir = Path(checkpoint_dir)

    def skipped(reason: str) -> RunResult:
        return RunResult(checkpoint_dir, "skipped", reason)

    run_dir, found_by = find_mlflow_run(checkpoint_dir, Path(mlruns_root))
    if run_dir is None:
        return skipped(found_by)

    gamma = read_mi_hyperparameters(checkpoint_dir).gamma
    if gamma is None:
        return skipped("gamma not found in resolved_config.yaml / MLflow params")

    histories = {
        name: read_history(run_dir, f"train/{name}")
        for name, _ in loss_curves.TRAINING_LOSSES
    }
    missing = [f"train/{name}" for name, history in histories.items() if history is None]
    if missing:
        return skipped(f"run {run_dir.name}: no finite history for {', '.join(missing)}")
    epochs = {len(history) for history in histories.values()}

    targets = [checkpoint_dir / "plots" / LOSSES_FOLDER]
    artifact_dir = run_dir / "artifacts" / LOSSES_FOLDER
    if preview_dir is not None:
        targets = [Path(preview_dir) / checkpoint_dir.parent.name / checkpoint_dir.name]
    elif artifact_dir.is_dir():
        targets.append(artifact_dir)

    summary = (
        f"gamma={gamma:g} epochs={'/'.join(map(str, sorted(epochs)))} "
        f"run={run_dir.parent.name}/{run_dir.name} ({found_by})"
    )
    if preview_dir is None and not artifact_dir.is_dir():
        summary += " [no MLflow artifacts/losses]"
    if dry_run:
        return RunResult(checkpoint_dir, "dry-run", summary)

    with tempfile.TemporaryDirectory(prefix="redo_loss_plots_") as tmp:
        drawn = loss_curves.plot_training_losses(histories, gamma, Path(tmp))
        mi_null = loss_curves.plot_mi_null(
            lambda key: read_history(run_dir, key), Path(tmp)
        )
        if mi_null is not None:
            drawn.append(mi_null)

        written = []
        for target in targets:
            target.mkdir(parents=True, exist_ok=True)
            for png in drawn:
                shutil.copyfile(png, target / png.name)
                written.append(target / png.name)

    existing = {path.name for path in (checkpoint_dir / "plots" / LOSSES_FOLDER).glob("*.png")}
    not_redrawn = sorted(existing - {png.name for png in drawn})
    if not_redrawn:
        summary += f" [old PNGs not redrawn: {', '.join(not_redrawn)}]"
    return RunResult(checkpoint_dir, "ok", f"{len(drawn)} PNGs, {summary}", written)


def _redraw_star(arguments: tuple) -> RunResult:
    try:
        return redraw_run(*arguments)
    except Exception as error:  # report and keep going with the other runs
        return RunResult(Path(arguments[0]), "skipped", f"error: {error!r}")


def redraw_all(
    checkpoint_dirs: Sequence[Path],
    mlruns_root: Path = DEFAULT_MLRUNS_ROOT,
    dry_run: bool = False,
    preview_dir: Path | None = None,
    jobs: int = 1,
) -> list[RunResult]:
    arguments = [(path, mlruns_root, dry_run, preview_dir) for path in checkpoint_dirs]
    if jobs <= 1 or dry_run:
        results = map(_redraw_star, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        results = executor.map(_redraw_star, arguments)
    collected = []
    for result in results:
        collected.append(result)
        name = f"{result.checkpoint_dir.parent.name}/{result.checkpoint_dir.name}"
        print(f"[{result.status}] {name}: {result.message}", flush=True)
    if jobs > 1 and not dry_run:
        executor.shutdown()
    return collected


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("roots", nargs="+", type=Path, help="checkpoint run folders or folders of them")
    parser.add_argument("--mlruns-root", type=Path, default=DEFAULT_MLRUNS_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="only resolve runs and histories")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="write the PNGs here (<experiment>/<run>/) instead of over the old ones",
    )
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint_dirs = discover_checkpoint_dirs(args.roots)
    print(f"{len(checkpoint_dirs)} run folders with plots/{LOSSES_FOLDER}/", flush=True)
    results = redraw_all(
        checkpoint_dirs,
        mlruns_root=args.mlruns_root.resolve(),
        dry_run=args.dry_run,
        preview_dir=args.preview_dir,
        jobs=args.jobs,
    )
    counts = Counter(result.status for result in results)
    written = sum(len(result.written) for result in results)
    print(f"done: {dict(counts)}; {written} PNG files written", flush=True)
    return 1 if counts.get("skipped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
