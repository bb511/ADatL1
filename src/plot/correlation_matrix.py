"""Correlation-matrix heatmaps shared by the evaluator and the re-plot scripts.

Every correlation-matrix figure (before/after training, the ``|after| - |before|``
change, its sorted orderings and each ``*.Et``-only crop) is drawn through
:func:`plot_correlation_matrix`, so all variants share one layout:

* the row of the highlighted variable (``FET.Et``, the MI-sensitive variable) is
  framed by a bold black border;
* an entry of that row is printed in bold green when the pair is decorrelated,
  ``|r| <= DECORRELATED_ABS_THRESHOLD``, in the *decorrelation reference*: the
  matrix itself for the before/after correlation matrices, and the reconstruction
  (after-training) matrix for the ``|after| - |before|`` change matrices;
* in the ``comparison_gamma0/`` matrices (``|reco(run)| - |reco(gamma = 0)|``) an
  entry of that row is green if and only if the run's reconstructed correlation is
  strictly closer to 0 than the gamma = 0 run's (:func:`closer_to_zero_columns`,
  passed as ``green_columns``), i.e. where the plotted difference is negative.

The module imports neither torch nor Lightning, so the plots can be redrawn from
the saved CSV matrices on any machine
(``src/analysis/scripts/redo_correlation_matrix_plots.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.plot import matrix


#: Row framed in every correlation-matrix plot.
HIGHLIGHT_VARIABLE = "FET.Et"
#: Highlighted-row entries whose |r| in the decorrelation reference is at or below
#: this are printed in :data:`DECORRELATED_TEXT_COLOR`.
DECORRELATED_ABS_THRESHOLD = 0.1
DECORRELATED_TEXT_COLOR = "#008000"
#: Border width of the highlighted row, in points.
HIGHLIGHT_LINEWIDTH = 3.0

ET_ONLY_SUFFIX = "_et_only"

# Folder layout below <callback>/<dataset>/<Method>/ (e.g. correlation_matrix/normal/
# Pearson/): the input and reconstruction matrices sit in the method folder itself,
# |reconstruction| - |input| ("self improvement") in SELF_IMPROVEMENT_DIR, and
# |reconstruction| - |reconstruction of the gamma = 0 run| in COMPARISON_GAMMA0_DIR
# (written afterwards by src/analysis/correlation_gamma0_comparison.py).
SELF_IMPROVEMENT_DIR = "self_improvement"
COMPARISON_GAMMA0_DIR = "comparison_gamma0"
SUBFOLDERS = (SELF_IMPROVEMENT_DIR, COMPARISON_GAMMA0_DIR)
#: Filename suffix -> figure scale of each plotted variant.
VARIANT_FIGURE_SCALES = {"": 1.0, ET_ONLY_SUFFIX: 0.6}
SORT_DIRECTIONS = {"increase": False, "decrease": True}

CMAP = "coolwarm"
VMIN = -1.0
VMAX = 1.0


def correlation_matrix_stem(space: str, method: str) -> str:
    """File stem of the correlation matrix of one variable space."""
    return f"{space}_{method}_correlation_matrix"


def correlation_change_stem(method: str, direction: str | None = None) -> str:
    """File stem of the ``|after| - |before|`` matrix, optionally sorted."""
    stem = f"abs_reconstruction_minus_input_{method}_correlation_matrix"
    if direction is not None:
        stem = f"{stem}_sorted_by_{direction}"
    return stem


def gamma0_comparison_stem(method: str, direction: str | None = None) -> str:
    """File stem of the ``|after| - |after of the gamma = 0 run|`` matrix."""
    stem = f"abs_reconstruction_minus_gamma0_reconstruction_{method}_correlation_matrix"
    if direction is not None:
        stem = f"{stem}_sorted_by_{direction}"
    return stem


def gamma0_reference_csv_name(method: str) -> str:
    """Copy of the gamma = 0 run's reconstruction matrix inside COMPARISON_GAMMA0_DIR."""
    return f"gamma0_reconstruction_{method}_correlation_matrix.csv"


def gallery_name(dataset: str, callback: str, method: str, subfolder: str | None = None) -> str:
    """MLflow gallery file stem of a method folder or one of its SUBFOLDERS."""
    name = f"{dataset}_{callback}_{method}"
    return f"{name}_{subfolder}" if subfolder else name


def correlation_matrix_title(space: str, method: str) -> str:
    """Plot title of the correlation matrix of one variable space."""
    method_name = method.capitalize()
    return {
        "input": f"{method_name} correlation matrix before training",
        "reconstruction": f"{method_name} correlation matrix after training",
    }.get(space, f"{method_name} correlation matrix: {space}")


def correlation_change_title(method: str, direction: str | None = None) -> str:
    """Plot title of the ``|after| - |before|`` matrix, optionally sorted."""
    method_name = method.capitalize()
    if direction is None:
        return f"Change in {method_name} correlation: |corr_after| - |corr_before|"
    return (
        f"Change in {method_name} correlation: "
        f"variables sorted by mean {direction}"
    )


def gamma0_comparison_title(method: str, direction: str | None = None) -> str:
    """Plot title of the comparison with the gamma = 0 run, optionally sorted."""
    method_name = method.capitalize()
    if direction is None:
        return (
            f"Change vs γ = 0 in {method_name} correlation: "
            "|corr_after| - |corr_after(γ = 0)|"
        )
    return (
        f"Change vs γ = 0 in {method_name} correlation: "
        f"variables sorted by mean {direction}"
    )


def et_only(corr: pd.DataFrame) -> pd.DataFrame:
    """Restrict a labelled matrix to the variables whose label ends in ``.Et``."""
    et_labels = [label for label in corr.columns if str(label).endswith(".Et")]
    if not et_labels:
        raise RuntimeError(
            "Cannot create the required *.Et-only correlation matrix because "
            "the configured correlation variables contain no labels ending in '.Et'."
        )
    return corr.loc[et_labels, et_labels]


def select_variant(corr: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Return the matrix shown in the variant with filename ``suffix``."""
    if suffix == "":
        return corr
    if suffix == ET_ONLY_SUFFIX:
        return et_only(corr)
    raise ValueError(f"Unknown correlation-matrix variant suffix {suffix!r}.")


def load_correlation_matrix_csv(path: Path) -> pd.DataFrame:
    """Load a labelled square correlation matrix written by the callback."""
    corr = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="raise")
    if corr.empty or list(corr.index) != list(corr.columns):
        raise ValueError(f"Not a labelled square matrix: {path}")
    return corr


def exclude_nan_variables(corr: pd.DataFrame) -> pd.DataFrame:
    """Same rule as ``CorrelationMatrixCallback._exclude_nan_variables``."""
    corr = corr.replace([float("inf"), float("-inf")], float("nan"))
    while corr.isna().to_numpy().any():
        nan_counts = corr.isna().sum(axis=0) + corr.isna().sum(axis=1)
        label = nan_counts.idxmax()
        corr = corr.drop(index=label, columns=label)
    return corr


def abs_correlation_change(corr_before: pd.DataFrame, corr_after: pd.DataFrame) -> pd.DataFrame:
    """``|after| - |before|`` on the common variables, as in the callback."""
    common = [label for label in corr_before.index if label in corr_after.index]
    if not common:
        raise ValueError("The two correlation matrices share no variables.")
    change = corr_after.loc[common, common].abs() - corr_before.loc[common, common].abs()
    change = exclude_nan_variables(change)
    if change.empty:
        raise ValueError("Correlation-change matrix is empty.")
    return change


def sort_correlation_change_matrix(
    corr: pd.DataFrame,
    ascending: bool,
) -> pd.DataFrame:
    """Order both axes by each variable's mean off-diagonal correlation change.

    Positive scores mean that a variable became more strongly correlated on
    average after reconstruction; negative scores mean that it became less
    strongly correlated. The diagonal is excluded because self-correlation does
    not describe a relationship between variables.
    """
    if list(corr.index) != list(corr.columns):
        raise ValueError(
            "Cannot sort a correlation-change matrix whose row and column "
            "labels differ."
        )

    off_diagonal = corr.copy()
    np.fill_diagonal(off_diagonal.values, np.nan)
    mean_change = off_diagonal.mean(axis=1).fillna(0.0)
    ordered_labels = mean_change.sort_values(
        ascending=ascending,
        kind="stable",
    ).index

    return corr.loc[ordered_labels, ordered_labels]


def _row_values(corr: pd.DataFrame | None, variable: str | None) -> pd.Series | None:
    """Numeric row ``variable`` of ``corr``, keyed by column label as ``str``.

    The row label matches exactly, else case-insensitively; a missing row (or no
    matrix) gives ``None``.
    """
    if corr is None or variable is None:
        return None
    labels = [str(label) for label in corr.index]
    lowered = [label.lower() for label in labels]
    if variable in labels:
        row = corr.index[labels.index(variable)]
    elif variable.lower() in lowered:
        row = corr.index[lowered.index(variable.lower())]
    else:
        return None
    values = pd.to_numeric(corr.loc[row], errors="coerce")
    values.index = [str(column) for column in values.index]
    return values


def decorrelated_columns(
    reference: pd.DataFrame | None,
    variable: str | None = HIGHLIGHT_VARIABLE,
    threshold: float = DECORRELATED_ABS_THRESHOLD,
) -> list[str]:
    """Columns ``c`` with a finite ``|reference[variable, c]| <= threshold``.

    The row label matches exactly, else case-insensitively; a missing row (or no
    reference) gives no columns.
    """
    values = _row_values(reference, variable)
    if values is None:
        return []
    return [
        str(column)
        for column, value in values.items()
        if np.isfinite(value) and abs(value) <= threshold
    ]


def closer_to_zero_columns(
    run_corr: pd.DataFrame | None,
    reference_corr: pd.DataFrame | None,
    variable: str | None = HIGHLIGHT_VARIABLE,
) -> list[str]:
    """Columns ``c`` with ``|run_corr[variable, c]| < |reference_corr[variable, c]|``.

    The green rule of the ``comparison_gamma0/`` plots: ``run_corr`` is the run's
    reconstruction matrix, ``reference_corr`` the gamma = 0 run's. "Closer to 0"
    is strict, so equal values are not listed; columns missing from either matrix
    or non-finite in either are skipped.
    """
    run_values = _row_values(run_corr, variable)
    reference_values = _row_values(reference_corr, variable)
    if run_values is None or reference_values is None:
        return []
    columns = []
    for column, value in run_values.items():
        if column not in reference_values.index:
            continue
        reference = reference_values[column]
        if np.isfinite(value) and np.isfinite(reference) and abs(value) < abs(reference):
            columns.append(column)
    return columns


def plot_correlation_matrix(
    corr: pd.DataFrame,
    save_dir: Path,
    filename: str,
    title: str,
    *,
    figure_scale: float = 1.0,
    highlight_variable: str | None = HIGHLIGHT_VARIABLE,
    decorrelation_reference: pd.DataFrame | None = None,
    green_columns: Sequence[str] | None = None,
    subtitle: str | None = None,
) -> None:
    """Draw one correlation matrix with the highlighted-variable row framed.

    :param subtitle: Line below the title, e.g. the MI hyperparameters
        (``MiHyperparameters.text()`` in ``src/analysis/run_mi_hyperparameters.py``).

    :param decorrelation_reference: Correlation matrix that decides which entries of
        the highlighted row are printed in green (``|r| <= DECORRELATED_ABS_THRESHOLD``):
        ``corr`` itself for a before/after matrix, the reconstruction matrix for a
        change matrix. ``None`` colours nothing.
    :param green_columns: Columns of the highlighted row to print in green, used
        instead of ``decorrelation_reference`` when given (the ``comparison_gamma0/``
        rule, :func:`closer_to_zero_columns`).
    """
    if green_columns is None:
        green_columns = decorrelated_columns(decorrelation_reference, highlight_variable)
    matrix.plot(
        data=corr.to_dict(orient="index"),
        value_name=title,
        save_dir=save_dir,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        filename=filename,
        figure_scale=figure_scale,
        outline_row=highlight_variable,
        outline_linewidth=HIGHLIGHT_LINEWIDTH,
        text_highlight_columns=list(green_columns),
        text_highlight_color=DECORRELATED_TEXT_COLOR,
        subtitle=subtitle,
    )


def write_correlation_matrix_variants(
    corr: pd.DataFrame,
    plot_folder: Path,
    stem: str,
    title: str,
    *,
    sort_ascending: bool | None = None,
    highlight_variable: str | None = HIGHLIGHT_VARIABLE,
    decorrelation_reference: pd.DataFrame | None = None,
    green_columns: Sequence[str] | None = None,
    subtitle: str | None = None,
) -> None:
    """Save the full-variable and ``*.Et``-only PNG of one correlation matrix.

    ``decorrelation_reference``, ``green_columns`` and ``subtitle`` are passed on
    to :func:`plot_correlation_matrix`.

    With ``sort_ascending`` set, each variant is ordered by
    :func:`sort_correlation_change_matrix` after the ``*.Et`` selection.
    """
    # Build every variant first so a missing *.Et label fails before any plot.
    variants = [
        (suffix, select_variant(corr, suffix), scale)
        for suffix, scale in VARIANT_FIGURE_SCALES.items()
    ]

    for suffix, variant, figure_scale in variants:
        if sort_ascending is not None:
            variant = sort_correlation_change_matrix(variant, ascending=sort_ascending)

        plot_correlation_matrix(
            variant,
            save_dir=plot_folder,
            filename=f"{stem}{suffix}.png",
            title=title,
            figure_scale=figure_scale,
            highlight_variable=highlight_variable,
            decorrelation_reference=decorrelation_reference,
            green_columns=green_columns,
            subtitle=subtitle,
        )
