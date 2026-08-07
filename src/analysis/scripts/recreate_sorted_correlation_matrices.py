#!/usr/bin/env python3
"""Recreate sorted correlation-change matrices for every run in an MLflow experiment."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.callbacks.correlation_matrix import CorrelationMatrixCallback


DEFAULT_MLRUNS_ROOT = REPO_ROOT / "logs" / "mlflow" / "mlruns"
DEFAULT_CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class MlflowRun:
    """Minimal local MLflow run metadata needed to locate checkpoint artifacts."""

    run_id: str
    run_name: str


@dataclass(frozen=True)
class RecreationSummary:
    """Counters returned by :func:`recreate_experiment`."""

    discovered_runs: int
    unique_run_names: int
    planned_targets: int
    recreated_targets: int
    existing_targets: int
    duplicate_targets: int
    missing_targets: int
    failed_targets: int


def _read_yaml_scalar(path: Path, key: str) -> str | None:
    """Read one top-level scalar from the simple metadata YAML files MLflow writes."""
    prefix = f"{key}:"
    for line in path.read_text().splitlines():
        if not line.startswith(prefix):
            continue

        value = line[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value or None

    return None


def _read_run_name(run_dir: Path) -> str | None:
    """Resolve the MLflow display name from its tag file, then from ``meta.yaml``."""
    run_name_tag = run_dir / "tags" / "mlflow.runName"
    if run_name_tag.is_file():
        run_name = run_name_tag.read_text().strip()
        if run_name:
            return run_name

    meta_path = run_dir / "meta.yaml"
    if meta_path.is_file():
        return _read_yaml_scalar(meta_path, "run_name")

    return None


def discover_mlflow_runs(experiment_dir: Path) -> list[MlflowRun]:
    """Return local MLflow runs with usable run names, ordered by run ID."""
    runs = []
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir() or not RUN_ID_PATTERN.fullmatch(run_dir.name):
            continue

        run_name = _read_run_name(run_dir)
        if run_name is None:
            continue
        if Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError(
                f"Unsafe MLflow run name {run_name!r} in run {run_dir.name}."
            )

        runs.append(MlflowRun(run_id=run_dir.name, run_name=run_name))

    return runs


def resolve_experiment_name(experiment_dir: Path, override: str | None = None) -> str:
    """Resolve the checkpoint experiment folder name from MLflow metadata."""
    experiment_name = override
    if experiment_name is None:
        meta_path = experiment_dir / "meta.yaml"
        if not meta_path.is_file():
            raise FileNotFoundError(f"MLflow experiment metadata not found: {meta_path}")
        experiment_name = _read_yaml_scalar(meta_path, "name")

    if not experiment_name:
        raise ValueError(
            "Could not resolve the experiment name. Pass --experiment-name explicitly."
        )
    if Path(experiment_name).name != experiment_name or experiment_name in {".", ".."}:
        raise ValueError(f"Unsafe experiment name: {experiment_name!r}.")

    return experiment_name


def _load_correlation_matrix(path: Path) -> pd.DataFrame:
    """Load and validate one labelled square correlation matrix CSV."""
    corr = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="raise")
    if corr.empty:
        raise ValueError(f"Correlation matrix is empty: {path}")
    if not corr.index.is_unique or not corr.columns.is_unique:
        raise ValueError(f"Correlation matrix contains duplicate labels: {path}")
    if list(corr.index) != list(corr.columns):
        raise ValueError(
            f"Correlation matrix row and column labels do not match: {path}"
        )
    return corr


def load_correlation_change(matrix_dir: Path, method: str) -> pd.DataFrame:
    """Recompute ``|corr_reconstruction| - |corr_input|`` from saved CSV files."""
    input_path = matrix_dir / f"input_{method}_correlation_matrix.csv"
    reconstruction_path = matrix_dir / f"reconstruction_{method}_correlation_matrix.csv"
    corr_before = _load_correlation_matrix(input_path)
    corr_after = _load_correlation_matrix(reconstruction_path)

    common_labels = [label for label in corr_before.index if label in corr_after.index]
    if not common_labels:
        raise ValueError(
            f"Input and reconstruction matrices have no common variables in {matrix_dir}."
        )

    correlation_change = corr_after.loc[common_labels, common_labels].abs()
    correlation_change -= corr_before.loc[common_labels, common_labels].abs()
    correlation_change = CorrelationMatrixCallback._exclude_nan_variables(
        correlation_change
    )
    if correlation_change.empty:
        raise ValueError(f"Correlation-change matrix is empty in {matrix_dir}.")

    return correlation_change


def expected_output_paths(matrix_dir: Path, method: str) -> tuple[Path, ...]:
    """Return the full and ``Et``-only CSV/PNG paths produced for both sort orders."""
    stem = f"abs_reconstruction_minus_input_{method}_correlation_matrix"
    return tuple(
        matrix_dir / f"{stem}_sorted_by_{direction}{suffix}.{extension}"
        for direction in ("increase", "decrease")
        for suffix in ("", "_et_only")
        for extension in ("csv", "png")
    )


def recreate_target(matrix_dir: Path, method: str) -> tuple[Path, ...]:
    """Write sorted increase/decrease matrices into one callback output directory."""
    correlation_change = load_correlation_change(matrix_dir, method)
    callback = CorrelationMatrixCallback(correlation_methods=[method])
    change_stem = f"abs_reconstruction_minus_input_{method}_correlation_matrix"
    method_name = method.capitalize()

    for direction, ascending in (("increase", False), ("decrease", True)):
        callback._write_correlation_matrix_variants(
            corr=correlation_change,
            plot_folder=matrix_dir,
            stem=f"{change_stem}_sorted_by_{direction}",
            title=(
                f"Change in {method_name} correlation: "
                f"variables sorted by mean {direction}"
            ),
            sort_ascending=ascending,
        )

    return expected_output_paths(matrix_dir, method)


def recreate_experiment(
    experiment_id: str,
    mlruns_root: Path = DEFAULT_MLRUNS_ROOT,
    checkpoints_root: Path = DEFAULT_CHECKPOINTS_ROOT,
    experiment_name: str | None = None,
    splits: Sequence[str] = ("val", "test"),
    datasets: Sequence[str] = ("normal",),
    checkpoint_name: str = "last",
    callback_name: str = "correlation_matrix",
    method: str = "pearson",
    skip_existing: bool = False,
    dry_run: bool = False,
    strict: bool = False,
) -> RecreationSummary:
    """Recreate sorted matrices for all locally available runs in an experiment."""
    experiment_dir = Path(mlruns_root) / str(experiment_id)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"MLflow experiment directory not found: {experiment_dir}")

    resolved_experiment_name = resolve_experiment_name(experiment_dir, experiment_name)
    runs = discover_mlflow_runs(experiment_dir)
    if not runs:
        raise RuntimeError(f"No named MLflow runs found in {experiment_dir}.")

    checkpoint_experiment_dir = Path(checkpoints_root) / resolved_experiment_name
    seen_targets: set[Path] = set()
    planned_targets = 0
    recreated_targets = 0
    existing_targets = 0
    duplicate_targets = 0
    missing_targets = 0
    failed_targets = 0

    print(
        f"Experiment {resolved_experiment_name!r} ({experiment_id}): "
        f"{len(runs)} named MLflow runs"
    )

    for run in runs:
        for split in splits:
            for dataset in datasets:
                matrix_dir = (
                    checkpoint_experiment_dir
                    / run.run_name
                    / "plots"
                    / split
                    / checkpoint_name
                    / callback_name
                    / dataset
                )
                matrix_dir = matrix_dir.resolve()

                if matrix_dir in seen_targets:
                    duplicate_targets += 1
                    continue
                seen_targets.add(matrix_dir)

                input_path = matrix_dir / f"input_{method}_correlation_matrix.csv"
                reconstruction_path = (
                    matrix_dir / f"reconstruction_{method}_correlation_matrix.csv"
                )
                missing_sources = [
                    path
                    for path in (input_path, reconstruction_path)
                    if not path.is_file()
                ]
                if missing_sources:
                    missing_targets += 1
                    print(
                        f"SKIP {run.run_name} [{split}/{dataset}]: missing "
                        + ", ".join(path.name for path in missing_sources)
                    )
                    continue

                output_paths = expected_output_paths(matrix_dir, method)
                if skip_existing and all(path.is_file() for path in output_paths):
                    existing_targets += 1
                    print(f"EXISTS {run.run_name} [{split}/{dataset}]")
                    continue

                if dry_run:
                    planned_targets += 1
                    print(f"WOULD RECREATE {run.run_name} [{split}/{dataset}]")
                    continue

                try:
                    recreate_target(matrix_dir, method)
                except Exception as error:
                    failed_targets += 1
                    print(f"FAILED {run.run_name} [{split}/{dataset}]: {error}")
                    if strict:
                        raise
                else:
                    recreated_targets += 1
                    print(f"RECREATED {run.run_name} [{split}/{dataset}]")

    summary = RecreationSummary(
        discovered_runs=len(runs),
        unique_run_names=len({run.run_name for run in runs}),
        planned_targets=planned_targets,
        recreated_targets=recreated_targets,
        existing_targets=existing_targets,
        duplicate_targets=duplicate_targets,
        missing_targets=missing_targets,
        failed_targets=failed_targets,
    )
    print(
        "Summary: "
        f"runs={summary.discovered_runs}, "
        f"unique_run_names={summary.unique_run_names}, "
        f"planned={summary.planned_targets}, "
        f"recreated={summary.recreated_targets}, "
        f"existing={summary.existing_targets}, "
        f"duplicate_targets={summary.duplicate_targets}, "
        f"missing={summary.missing_targets}, "
        f"failed={summary.failed_targets}"
    )

    if strict and (summary.missing_targets or summary.failed_targets):
        raise RuntimeError(
            "Strict mode failed because one or more correlation targets were missing "
            "or invalid."
        )

    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate sorted absolute-correlation change matrices for every local "
            "MLflow run in an experiment."
        )
    )
    parser.add_argument("experiment_id", help="Local MLflow experiment ID.")
    parser.add_argument(
        "--mlruns-root",
        type=Path,
        default=DEFAULT_MLRUNS_ROOT,
        help="Directory containing MLflow experiment folders.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=DEFAULT_CHECKPOINTS_ROOT,
        help="Root containing checkpoint experiment folders.",
    )
    parser.add_argument(
        "--experiment-name",
        help="Checkpoint experiment folder name; defaults to the MLflow metadata name.",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--datasets", nargs="+", default=["normal"])
    parser.add_argument("--checkpoint-name", default="last")
    parser.add_argument("--callback-name", default="correlation_matrix")
    parser.add_argument("--method", default="pearson")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite targets for which all expected files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show targets without writing files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested target is missing or invalid.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    summary = recreate_experiment(
        experiment_id=args.experiment_id,
        mlruns_root=args.mlruns_root,
        checkpoints_root=args.checkpoints_root,
        experiment_name=args.experiment_name,
        splits=args.splits,
        datasets=args.datasets,
        checkpoint_name=args.checkpoint_name,
        callback_name=args.callback_name,
        method=args.method,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        strict=args.strict,
    )
    return 1 if summary.failed_targets else 0


if __name__ == "__main__":
    raise SystemExit(main())
