"""Analyse reconstructed ``FET.Et`` correlations across MI configurations.

For every checkpoint run matching a glob pattern, this module loads the
reconstructed correlation matrix and computes

    sum(abs(corr(FET.Et, other_variable)))

where the ``FET.Et`` self-correlation is explicitly excluded. The effective bin
count is read from the latest MI bin-width diagnostic and gamma is resolved from
the corresponding local MLflow run.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "physics_ae_models"
MLRUNS_ROOT_CANDIDATES = (
    REPO_ROOT / "logs" / "mlflow" / "mlruns",
    REPO_ROOT / "deploy" / "logs" / "mlflow" / "mlruns",
    REPO_ROOT / "data" / "mlflow" / "mllogs",
    REPO_ROOT / "logs" / "mlflow" / "mlready",
)
OUTPUT_DIR_CANDIDATES = (
    REPO_ROOT / "logs" / "plots",
    REPO_ROOT / "deploy" / "logs" / "plots",
)
DEFAULT_MLRUNS_ROOT = next(
    (path for path in MLRUNS_ROOT_CANDIDATES if path.is_dir()),
    MLRUNS_ROOT_CANDIDATES[0],
)
DEFAULT_OUTPUT_DIR = next(
    (path for path in OUTPUT_DIR_CANDIDATES if path.is_dir()),
    OUTPUT_DIR_CANDIDATES[0],
)
DEFAULT_RUN_PATTERN = "Bernoulli_MI_No_FET_Run_*"
DEFAULT_MATRIX_FILENAME = "reconstruction_pearson_correlation_matrix.csv"
DEFAULT_VARIABLES_FILENAME = "reconstruction_variables.csv"

EFFECTIVE_BIN_NUMBER = "effective_bin_number"
FET_ET_SUM = "FET.Et_abs_correlation_sum"
GAMMA = "gamma"
DATA_COLUMNS = (EFFECTIVE_BIN_NUMBER, FET_ET_SUM, GAMMA)


@dataclass(frozen=True)
class MLflowRunMetadata:
    """Metadata needed from one local MLflow run."""

    run_name: str
    gamma: float
    status: str
    start_time: int
    run_dir: Path


def load_correlation_matrix(path: str | Path) -> pd.DataFrame:
    """Load and validate a labelled square correlation matrix."""
    matrix_path = Path(path)
    corr = pd.read_csv(matrix_path, index_col=0)
    corr = corr.apply(pd.to_numeric, errors="raise")

    if corr.empty:
        raise ValueError(f"Correlation matrix is empty: {matrix_path}")
    if not corr.index.is_unique or not corr.columns.is_unique:
        raise ValueError(
            f"Correlation matrix contains duplicate labels: {matrix_path}"
        )
    if list(corr.index) != list(corr.columns):
        raise ValueError(
            "Correlation matrix row and column labels do not match in "
            f"{matrix_path}."
        )

    return corr


def fet_et_abs_correlation_sum_from_variables(
    path: str | Path,
    variable: str = "FET.Et",
    chunksize: int = 250_000,
) -> float:
    """Compute the required Pearson-row sum from an event table in chunks."""
    variables_path = Path(path)
    columns = list(pd.read_csv(variables_path, nrows=0).columns)
    if variable not in columns:
        raise KeyError(
            f"Variable {variable!r} is absent from {variables_path}. "
            f"Available columns: {columns}"
        )

    target_index = columns.index(variable)
    count = 0
    sums = np.zeros(len(columns), dtype=np.float64)
    sum_squares = np.zeros(len(columns), dtype=np.float64)
    target_products = np.zeros(len(columns), dtype=np.float64)

    for chunk in pd.read_csv(variables_path, chunksize=chunksize):
        chunk = chunk.apply(pd.to_numeric, errors="raise")
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        if chunk.empty:
            continue

        values = chunk.loc[:, columns].to_numpy(dtype=np.float64)
        target = values[:, target_index]
        count += len(values)
        sums += values.sum(axis=0)
        sum_squares += np.einsum("ij,ij->j", values, values)
        target_products += np.einsum("i,ij->j", target, values)

    if count == 0:
        raise ValueError(
            f"Reconstruction variable table has no complete rows: {variables_path}"
        )

    centered_sum_squares = sum_squares - np.square(sums) / count
    centered_sum_squares = np.maximum(centered_sum_squares, 0.0)
    target_sum_squares = centered_sum_squares[target_index]
    if target_sum_squares <= 0.0:
        raise ValueError(f"Variable {variable!r} is constant in {variables_path}.")

    cross_products = target_products - sums[target_index] * sums / count
    denominators = np.sqrt(target_sum_squares * centered_sum_squares)
    valid = np.isfinite(denominators) & (denominators > 0.0)
    valid[target_index] = False
    if not valid.any():
        raise ValueError(
            f"No finite off-diagonal {variable} correlations in {variables_path}."
        )

    correlations = cross_products[valid] / denominators[valid]
    return float(np.abs(correlations).sum())


def fet_et_abs_correlation_sum(
    corr: pd.DataFrame,
    variable: str = "FET.Et",
) -> float:
    """Sum absolute correlations in the ``variable`` row, excluding itself."""
    if variable not in corr.index or variable not in corr.columns:
        raise KeyError(
            f"Variable {variable!r} is not present on both axes. "
            f"Available labels: {list(corr.columns)}"
        )

    correlations = pd.to_numeric(corr.loc[variable], errors="raise").drop(
        labels=variable
    )
    values = correlations.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        invalid_labels = correlations.index[~np.isfinite(values)].tolist()
        raise ValueError(
            f"Non-finite {variable} correlations found for: {invalid_labels}"
        )

    return float(np.abs(values).sum())


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def _read_meta_value(meta_path: Path, key: str) -> str | None:
    if not meta_path.is_file():
        return None

    prefix = f"{key}:"
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip("'\"")
    return None


def _read_parameter(run_dir: Path, name: str) -> str | None:
    """Read a flattened Hydra parameter, with its nested-name fallback."""
    for relative_path in (Path("params") / name, Path("params/algorithm") / name):
        value = _read_text(run_dir / relative_path)
        if value is not None:
            return value
    return None


def _parse_gamma(value: str, run_dir: Path) -> float:
    gamma = float(value)
    if not math.isfinite(gamma):
        raise ValueError(f"Invalid gamma {value!r} in {run_dir}.")
    return gamma


def _is_finished(status: str) -> bool:
    # MLflow's file store serialises FINISHED as enum value 3.
    return status.upper() == "FINISHED" or status == "3"


def load_mlflow_metadata(
    mlruns_root: str | Path = DEFAULT_MLRUNS_ROOT,
) -> dict[str, MLflowRunMetadata]:
    """Index local, active MLflow records by run name.

    Incomplete records without gamma are ignored. If multiple active records share a
    name, the latest finished record is selected deterministically.
    """
    root = Path(mlruns_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MLflow root does not exist: {root}")

    candidates: dict[str, list[MLflowRunMetadata]] = {}
    for tag_path in root.glob("*/*/tags/mlflow.runName"):
        run_dir = tag_path.parents[1]
        meta_path = run_dir / "meta.yaml"
        if _read_meta_value(meta_path, "lifecycle_stage") == "deleted":
            continue

        run_name = _read_text(tag_path)
        gamma_value = _read_parameter(run_dir, "mi_gamma")
        if not run_name or gamma_value is None:
            continue

        try:
            gamma = _parse_gamma(gamma_value, run_dir)
        except ValueError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            continue

        status = _read_meta_value(meta_path, "status") or ""
        start_value = _read_meta_value(meta_path, "start_time") or "0"
        try:
            start_time = int(start_value)
        except ValueError:
            start_time = 0

        metadata = MLflowRunMetadata(
            run_name=run_name,
            gamma=gamma,
            status=status,
            start_time=start_time,
            run_dir=run_dir,
        )
        candidates.setdefault(run_name, []).append(metadata)

    selected: dict[str, MLflowRunMetadata] = {}
    for run_name, run_candidates in candidates.items():
        selected[run_name] = max(
            run_candidates,
            key=lambda item: (
                _is_finished(item.status),
                item.start_time,
                item.run_dir.name,
            ),
        )
    return selected


def load_effective_bin_count(run_dir: str | Path) -> int:
    """Load the effective bin count from the latest bin-width diagnostic CSV."""
    diagnostics_dir = Path(run_dir) / "plots" / "mi_diagnostics" / "data"
    width_paths = sorted(
        diagnostics_dir.glob("epoch_*/mi_bin_widths_epoch*.csv")
    )
    if not width_paths:
        raise FileNotFoundError(
            f"No MI bin-width diagnostics found in {diagnostics_dir}."
        )

    latest_path = width_paths[-1]
    widths = pd.read_csv(latest_path)
    if "bin_id" not in widths.columns or widths.empty:
        raise ValueError(
            f"Effective-bin diagnostic is empty or lacks bin_id: {latest_path}"
        )

    bin_ids = pd.to_numeric(widths["bin_id"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(bin_ids).all() or not np.equal(bin_ids, np.floor(bin_ids)).all():
        raise ValueError(f"Invalid bin_id values in {latest_path}.")

    bin_ids = bin_ids.astype(int)
    expected_ids = np.arange(len(bin_ids))
    if not np.array_equal(bin_ids, expected_ids):
        raise ValueError(
            f"Expected contiguous bin_id values 0..{len(bin_ids) - 1} in "
            f"{latest_path}."
        )

    return len(bin_ids)


def collect_decorrelation_data(
    run_pattern: str = DEFAULT_RUN_PATTERN,
    checkpoints_root: str | Path = DEFAULT_CHECKPOINTS_ROOT,
    mlruns_root: str | Path = DEFAULT_MLRUNS_ROOT,
    split: str = "val",
    checkpoint_name: str = "last",
    dataset: str = "normal",
    matrix_filename: str = DEFAULT_MATRIX_FILENAME,
    variables_filename: str = DEFAULT_VARIABLES_FILENAME,
    variable: str = "FET.Et",
    strict: bool = False,
) -> dict[str, list[int | float]]:
    """Collect effective bins, total absolute ``FET.Et`` correlation, and gamma.

    Runs with a missing matrix, bin diagnostic, or MLflow metadata are skipped with a
    warning. Set ``strict=True`` to raise on the first missing artifact instead.
    """
    checkpoint_root = Path(checkpoints_root)
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    run_dirs = sorted(
        path for path in checkpoint_root.glob(run_pattern) if path.is_dir()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No checkpoint runs in {checkpoint_root} match {run_pattern!r}."
        )

    mlflow_by_name = load_mlflow_metadata(mlruns_root)
    records: list[dict[str, int | float]] = []
    relative_matrix_dir = (
        Path("plots")
        / split
        / checkpoint_name
        / "correlation_matrix"
        / dataset
    )

    for run_dir in run_dirs:
        matrix_dir = run_dir / relative_matrix_dir
        matrix_path = matrix_dir / matrix_filename
        variables_path = matrix_dir / variables_filename
        if matrix_path.is_file():
            corr = load_correlation_matrix(matrix_path)
            fet_et_sum = fet_et_abs_correlation_sum(corr, variable=variable)
        elif variables_path.is_file():
            fet_et_sum = fet_et_abs_correlation_sum_from_variables(
                variables_path,
                variable=variable,
            )
        else:
            message = (
                f"Missing reconstructed correlation data for {run_dir.name}; "
                f"expected {matrix_path} or {variables_path}."
            )
            if strict:
                raise FileNotFoundError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue

        metadata = mlflow_by_name.get(run_dir.name)
        if metadata is None:
            message = f"Missing MLflow bin/gamma metadata for {run_dir.name}."
            if strict:
                raise KeyError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue

        try:
            effective_bin_number = load_effective_bin_count(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            if strict:
                raise
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            continue

        records.append(
            {
                EFFECTIVE_BIN_NUMBER: effective_bin_number,
                FET_ET_SUM: fet_et_sum,
                GAMMA: metadata.gamma,
            }
        )

    if not records:
        raise ValueError(
            f"No complete checkpoint/MLflow records matched {run_pattern!r}."
        )

    records.sort(
        key=lambda record: (record[GAMMA], record[EFFECTIVE_BIN_NUMBER])
    )
    return {column: [record[column] for record in records] for column in DATA_COLUMNS}


def plot_decorrelation(
    data: dict[str, list[int | float]],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Plot total absolute ``FET.Et`` correlation against effective MI bins."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(data)
    missing_columns = [column for column in DATA_COLUMNS if column not in frame]
    if missing_columns:
        raise KeyError(f"Missing decorrelation data columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Cannot plot empty decorrelation data.")

    frame = frame.loc[:, list(DATA_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Decorrelation plot data contains non-finite values.")

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    scatter = ax.scatter(
        frame[EFFECTIVE_BIN_NUMBER],
        frame[FET_ET_SUM],
        c=frame[GAMMA],
        cmap="hot",
        s=72,
        edgecolors="black",
        linewidths=0.45,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(r"$\gamma$")

    ax.set_title(title or r"Total absolute reconstructed $FET.Et$ correlation")
    ax.set_xlabel("Number of effective MI bins")
    ax.set_ylabel(r"$\sum_{x \ne FET.Et} |\rho(FET.Et, x)|$")
    ax.grid(alpha=0.25)

    minimum = frame.loc[frame[FET_ET_SUM].idxmin()]
    minimum_x = float(minimum[EFFECTIVE_BIN_NUMBER])
    minimum_y = float(minimum[FET_ET_SUM])
    ax.scatter(
        [minimum_x],
        [minimum_y],
        s=180,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=1.8,
        zorder=4,
    )
    ax.annotate(
        f"FET.Et sum = {minimum_y:.4f}",
        xy=(minimum_x, minimum_y),
        xytext=(14, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.9},
        fontsize=9,
        zorder=5,
    )

    unique_bins = np.sort(frame[EFFECTIVE_BIN_NUMBER].unique())
    if len(unique_bins) <= 15:
        ax.set_xticks(unique_bins)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_decorrelation_analysis(
    data: dict[str, list[int | float]],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_stem: str = "fet_et_decorrelation",
    title: str | None = None,
) -> tuple[Path, Path]:
    """Persist the collected values as CSV and render the corresponding plot."""
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{output_stem}.csv"
    png_path = save_dir / f"{output_stem}.png"

    pd.DataFrame(data).to_csv(csv_path, index=False)
    plot_decorrelation(data=data, output_path=png_path, title=title)
    return csv_path, png_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the total absolute reconstructed FET.Et correlation by effective "
            "MI bin count, with gamma represented by colour."
        )
    )
    parser.add_argument("--run-pattern", default=DEFAULT_RUN_PATTERN)
    parser.add_argument(
        "--checkpoints-root", type=Path, default=DEFAULT_CHECKPOINTS_ROOT
    )
    parser.add_argument("--mlruns-root", type=Path, default=DEFAULT_MLRUNS_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint-name", default="last")
    parser.add_argument("--dataset", default="normal")
    parser.add_argument("--matrix-filename", default=DEFAULT_MATRIX_FILENAME)
    parser.add_argument("--variables-filename", default=DEFAULT_VARIABLES_FILENAME)
    parser.add_argument("--variable", default="FET.Et")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="fet_et_decorrelation")
    parser.add_argument("--title")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping runs with missing matrices or metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = collect_decorrelation_data(
        run_pattern=args.run_pattern,
        checkpoints_root=args.checkpoints_root,
        mlruns_root=args.mlruns_root,
        split=args.split,
        checkpoint_name=args.checkpoint_name,
        dataset=args.dataset,
        matrix_filename=args.matrix_filename,
        variables_filename=args.variables_filename,
        variable=args.variable,
        strict=args.strict,
    )
    csv_path, png_path = save_decorrelation_analysis(
        data=data,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        title=args.title,
    )

    print(f"Collected {len(data[EFFECTIVE_BIN_NUMBER])} run(s).")
    print(f"Saved decorrelation values to {csv_path}.")
    print(f"Saved decorrelation plot to {png_path}.")


if __name__ == "__main__":
    main()
