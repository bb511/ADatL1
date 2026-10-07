"""Correlation-matrix heatmaps shared by the evaluator and the re-plot scripts.

Every correlation-matrix figure (before/after training, the ``|after| - |before|``
change, its sorted orderings and each ``*.Et``-only crop) is drawn through
:func:`plot_correlation_matrix`, so all variants share one layout:

* the row of the highlighted variable (``FET.Et``, the MI-sensitive variable) is
  framed by a bold black border;
* an entry of that row is printed in bold green when the pair is decorrelated,
  ``|r| <= DECORRELATED_ABS_THRESHOLD``, in the *decorrelation reference*: the
  matrix itself for the before/after correlation matrices, and the reconstruction
  (after-training) matrix for the ``|after| - |before|`` change matrices.

The module imports neither torch nor Lightning, so the plots can be redrawn from
the saved CSV matrices on any machine
(``src/analysis/scripts/redo_correlation_matrix_plots.py``).
"""

from __future__ import annotations

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


def decorrelated_columns(
    reference: pd.DataFrame | None,
    variable: str | None = HIGHLIGHT_VARIABLE,
    threshold: float = DECORRELATED_ABS_THRESHOLD,
) -> list[str]:
    """Columns ``c`` with a finite ``|reference[variable, c]| <= threshold``.

    The row label matches exactly, else case-insensitively; a missing row (or no
    reference) gives no columns.
    """
    if reference is None or variable is None:
        return []
    labels = [str(label) for label in reference.index]
    if variable in labels:
        row = reference.index[labels.index(variable)]
    elif variable.lower() in [label.lower() for label in labels]:
        row = reference.index[[label.lower() for label in labels].index(variable.lower())]
    else:
        return []
    values = pd.to_numeric(reference.loc[row], errors="coerce")
    return [
        str(column)
        for column, value in values.items()
        if np.isfinite(value) and abs(value) <= threshold
    ]


def plot_correlation_matrix(
    corr: pd.DataFrame,
    save_dir: Path,
    filename: str,
    title: str,
    *,
    figure_scale: float = 1.0,
    highlight_variable: str | None = HIGHLIGHT_VARIABLE,
    decorrelation_reference: pd.DataFrame | None = None,
    subtitle: str | None = None,
) -> None:
    """Draw one correlation matrix with the highlighted-variable row framed.

    :param subtitle: Line below the title, e.g. the MI hyperparameters
        (``MiHyperparameters.text()`` in ``src/analysis/run_mi_hyperparameters.py``).

    :param decorrelation_reference: Correlation matrix that decides which entries of
        the highlighted row are printed in green (``|r| <= DECORRELATED_ABS_THRESHOLD``):
        ``corr`` itself for a before/after matrix, the reconstruction matrix for a
        change matrix. ``None`` colours nothing.
    """
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
        text_highlight_columns=decorrelated_columns(
            decorrelation_reference,
            highlight_variable,
        ),
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
    subtitle: str | None = None,
) -> None:
    """Save the full-variable and ``*.Et``-only PNG of one correlation matrix.

    ``decorrelation_reference`` and ``subtitle`` are passed on to
    :func:`plot_correlation_matrix`.

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
            subtitle=subtitle,
        )
