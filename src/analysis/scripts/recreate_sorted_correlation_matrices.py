#!/usr/bin/env python3
"""Recreate sorted matrices and rebuild galleries for an MLflow experiment."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.callbacks.correlation_matrix import (
    CORRELATION_SOURCE_FILENAMES,
    CorrelationMatrixCallback,
)
from src.evaluation.callbacks.utils.mlflow import build_gallery_html


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
    planned_galleries: int
    updated_galleries: int
    failed_galleries: int


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


def _correlate_variables(variables: pd.DataFrame, method: str) -> pd.DataFrame:
    """Recompute one clean correlation matrix from an event-level variable table."""
    clean_variables = variables.replace([np.inf, -np.inf], np.nan)
    clean_variables = clean_variables.dropna(axis=0)
    if clean_variables.empty:
        raise ValueError("Event-level correlation variable table has no complete rows.")

    corr = clean_variables.corr(method=method)
    corr = CorrelationMatrixCallback._exclude_nan_variables(corr)
    if corr.empty:
        raise ValueError("Recomputed correlation matrix is empty.")
    return corr


def _load_combined_variable_spaces(matrix_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load input/reconstruction tables from the superseded combined source CSV."""
    source_path = matrix_dir / "correlation_variables.csv"
    source_table = pd.read_csv(source_path, header=[0, 1])
    available_spaces = set(source_table.columns.get_level_values(0))
    required_spaces = {"input", "reconstruction"}
    missing_spaces = sorted(required_spaces - available_spaces)
    if missing_spaces:
        raise ValueError(
            f"{source_path} is missing source spaces: {missing_spaces}."
        )

    return (
        source_table.xs("input", axis=1, level=0),
        source_table.xs("reconstruction", axis=1, level=0),
    )


def _load_variable_spaces(matrix_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two source CSVs, with compatibility for the combined CSV format."""
    input_path = matrix_dir / CORRELATION_SOURCE_FILENAMES["input"]
    reconstruction_path = matrix_dir / CORRELATION_SOURCE_FILENAMES["reconstruction"]
    if input_path.is_file() and reconstruction_path.is_file():
        return pd.read_csv(input_path), pd.read_csv(reconstruction_path)

    combined_source_path = matrix_dir / "correlation_variables.csv"
    if combined_source_path.is_file():
        return _load_combined_variable_spaces(matrix_dir)

    raise FileNotFoundError("No event-level correlation variable source is available.")


def correlation_source_description(matrix_dir: Path, method: str) -> str | None:
    """Describe the best available source from which matrices can be recreated."""
    variable_paths = (
        matrix_dir / CORRELATION_SOURCE_FILENAMES["input"],
        matrix_dir / CORRELATION_SOURCE_FILENAMES["reconstruction"],
    )
    if all(path.is_file() for path in variable_paths):
        return "input_variables.csv and reconstruction_variables.csv"

    if (matrix_dir / "correlation_variables.csv").is_file():
        return "combined correlation_variables.csv"

    legacy_matrix_paths = (
        matrix_dir / f"input_{method}_correlation_matrix.csv",
        matrix_dir / f"reconstruction_{method}_correlation_matrix.csv",
    )
    if all(path.is_file() for path in legacy_matrix_paths):
        return "legacy input/reconstruction correlation CSVs"

    return None


def load_correlation_change(matrix_dir: Path, method: str) -> pd.DataFrame:
    """Recompute ``|corr_reconstruction| - |corr_input|`` from available sources."""
    try:
        input_variables, reconstruction_variables = _load_variable_spaces(matrix_dir)
    except FileNotFoundError:
        input_path = matrix_dir / f"input_{method}_correlation_matrix.csv"
        reconstruction_path = (
            matrix_dir / f"reconstruction_{method}_correlation_matrix.csv"
        )
        corr_before = _load_correlation_matrix(input_path)
        corr_after = _load_correlation_matrix(reconstruction_path)
    else:
        corr_before = _correlate_variables(input_variables, method)
        corr_after = _correlate_variables(reconstruction_variables, method)

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
    """Return the full and ``Et``-only PNG paths produced for both sort orders."""
    stem = f"abs_reconstruction_minus_input_{method}_correlation_matrix"
    return tuple(
        matrix_dir / f"{stem}_sorted_by_{direction}{suffix}.png"
        for direction in ("increase", "decrease")
        for suffix in ("", "_et_only")
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


def gallery_artifact_path(
    experiment_dir: Path,
    run_id: str,
    split: str,
    checkpoint_name: str,
    dataset: str,
    callback_name: str,
) -> Path:
    """Return the local MLflow HTML artifact path matching the evaluator layout."""
    return (
        experiment_dir
        / run_id
        / "artifacts"
        / split
        / checkpoint_name
        / f"{dataset}_{callback_name}.html"
    )


def update_local_mlflow_gallery(
    matrix_dir: Path,
    gallery_path: Path,
    section_name: str,
) -> None:
    """Rebuild one local MLflow gallery from every current matrix image."""
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    gallery_path.write_text(
        build_gallery_html(matrix_dir, section_name),
        encoding="utf-8",
    )


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
    planned_galleries = 0
    updated_galleries = 0
    failed_galleries = 0

    print(
        f"Experiment {resolved_experiment_name!r} ({experiment_id}): "
        f"{len(runs)} named MLflow runs"
    )

    runs_by_name: dict[str, list[MlflowRun]] = defaultdict(list)
    for run in runs:
        runs_by_name[run.run_name].append(run)

    for run_name, matching_runs in sorted(runs_by_name.items()):
        for split in splits:
            for dataset in datasets:
                matrix_dir = (
                    checkpoint_experiment_dir
                    / run_name
                    / "plots"
                    / split
                    / checkpoint_name
                    / callback_name
                    / dataset
                )
                matrix_dir = matrix_dir.resolve()

                duplicate_targets += len(matching_runs) - 1
                if matrix_dir in seen_targets:
                    raise RuntimeError(f"Unexpected duplicate matrix target: {matrix_dir}")
                seen_targets.add(matrix_dir)

                output_paths = expected_output_paths(matrix_dir, method)
                target_ready = False
                if skip_existing and all(path.is_file() for path in output_paths):
                    existing_targets += 1
                    target_ready = True
                    print(f"EXISTS {run_name} [{split}/{dataset}]")
                else:
                    source_description = correlation_source_description(matrix_dir, method)
                    if source_description is None:
                        missing_targets += 1
                        print(
                            f"SKIP {run_name} [{split}/{dataset}]: no supported "
                            "correlation source CSVs"
                        )
                        continue

                    if dry_run:
                        planned_targets += 1
                        target_ready = True
                        print(
                            f"WOULD RECREATE {run_name} [{split}/{dataset}] "
                            f"from {source_description}"
                        )
                    else:
                        try:
                            recreate_target(matrix_dir, method)
                        except Exception as error:
                            failed_targets += 1
                            print(f"FAILED {run_name} [{split}/{dataset}]: {error}")
                            if strict:
                                raise
                        else:
                            recreated_targets += 1
                            target_ready = True
                            print(f"RECREATED {run_name} [{split}/{dataset}]")

                if not target_ready:
                    continue

                for run in matching_runs:
                    gallery_path = gallery_artifact_path(
                        experiment_dir=experiment_dir,
                        run_id=run.run_id,
                        split=split,
                        checkpoint_name=checkpoint_name,
                        dataset=dataset,
                        callback_name=callback_name,
                    )
                    if dry_run:
                        planned_galleries += 1
                        print(f"WOULD UPDATE GALLERY {run.run_id}: {gallery_path}")
                        continue

                    try:
                        update_local_mlflow_gallery(
                            matrix_dir=matrix_dir,
                            gallery_path=gallery_path,
                            section_name=checkpoint_name,
                        )
                    except Exception as error:
                        failed_galleries += 1
                        print(f"FAILED GALLERY {run.run_id}: {error}")
                        if strict:
                            raise
                    else:
                        updated_galleries += 1
                        print(f"UPDATED GALLERY {run.run_id}: {gallery_path}")

    summary = RecreationSummary(
        discovered_runs=len(runs),
        unique_run_names=len({run.run_name for run in runs}),
        planned_targets=planned_targets,
        recreated_targets=recreated_targets,
        existing_targets=existing_targets,
        duplicate_targets=duplicate_targets,
        missing_targets=missing_targets,
        failed_targets=failed_targets,
        planned_galleries=planned_galleries,
        updated_galleries=updated_galleries,
        failed_galleries=failed_galleries,
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
        f"failed={summary.failed_targets}, "
        f"planned_galleries={summary.planned_galleries}, "
        f"updated_galleries={summary.updated_galleries}, "
        f"failed_galleries={summary.failed_galleries}"
    )

    if strict and (
        summary.missing_targets
        or summary.failed_targets
        or summary.failed_galleries
    ):
        raise RuntimeError(
            "Strict mode failed because one or more correlation targets were missing "
            "or invalid, or an MLflow gallery could not be updated."
        )

    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate sorted absolute-correlation change matrices for every local "
            "MLflow run in an experiment and rebuild each HTML gallery."
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
    return 1 if summary.failed_targets or summary.failed_galleries else 0


if __name__ == "__main__":
    raise SystemExit(main())
