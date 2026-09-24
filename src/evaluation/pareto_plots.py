"""Phase 4: the figures of the validation Pareto study.

Phase 3 selects and writes tables; this module draws and writes nothing else.
Keeping them apart means a figure can be restyled without re-running selection,
and selection cannot be silently changed by a plotting edit.

Every figure reads the two Phase 3 tables and nothing else, so what is drawn is
exactly what was selected.

Colour follows the data-viz palette. Gamma is an ordered factor with five
regularised levels, so it takes a validated five-step single-hue ordinal ramp
(all four ordinal checks pass on the light surface). Gamma zero is deliberately
NOT the lightest step of that ramp: it is the unregularised baseline that every
feasibility constraint is measured against, a reference rather than a little bit
of regularisation, so it is drawn as a neutral open marker with its own legend
entry. Residual correlation is continuous magnitude and takes the sequential
blue ramp, labelled on its colourbar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .pareto_selection import (
    CONFIGURATION_ID_COLUMN,
    CORRELATION_COLUMN,
    EFFICIENCY_COLUMN,
    FEASIBLE_COLUMN,
    LEAKAGE_COLUMN,
    VALID_COLUMN,
)

#: Panels read in capacity order rather than alphabetically, which would put
#: h128_64 before h64_32. Anything unrecognised is appended, sorted.
ARCHITECTURE_ORDER = ("h64_32", "h128_64", "h64_64_32")

GAMMA_COLUMN = "mi_gamma"
BINS_COLUMN = "mi_sensitive_num_bins"
ARCHITECTURE_COLUMN = "architecture_id"
FRONT_COLUMN = "is_pareto_front"

#: Validated ordinal ramp for the five regularised gamma levels: one hue,
#: monotone lightness, every adjacent gap >= 0.06, light end 2.06:1 on #fcfcfb.
GAMMA_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
#: Sequential blue, 100 -> 700, for continuous residual correlation.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
BASELINE_INK = "#52514e"
EXCLUDED_INK = "#bdbcb5"
GRID_INK = "#e8e7e3"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

#: Bins is a three-level ordered factor; area is a weak channel, so the steps are
#: widely separated. The label says "requested" on purpose: quantile edges on
#: FET.Et collapse, so the effective bin count is lower than the requested one.
BINS_AREA = {40: 34.0, 50: 86.0, 60: 170.0}
_DEFAULT_AREA = 86.0

AXIS_LABELS = {
    LEAKAGE_COLUMN: "Leakage $L$ (worst of four probes, held-out $R^2$)",
    EFFICIENCY_COLUMN: r"Signal efficiency $\epsilon_{\mathrm{median}}$",
    CORRELATION_COLUMN: "Residual correlation $E$",
}


class ParetoPlotError(ValueError):
    """The Phase 3 tables cannot be plotted."""


def _as_bool(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ParetoPlotError(f"{label} is not boolean.")
    return mapped.astype(bool)


def _ci_bounds(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Asymmetric error-bar offsets from the stored interval."""

    low = frame[column.replace("_mean", "_ci95_low")]
    high = frame[column.replace("_mean", "_ci95_high")]
    return np.vstack(((frame[column] - low).to_numpy(), (high - frame[column]).to_numpy()))


def _gamma_levels(frame: pd.DataFrame) -> list[float]:
    return sorted({float(value) for value in frame[GAMMA_COLUMN].dropna() if float(value) > 0.0})


def _gamma_colors(frame: pd.DataFrame) -> dict[float, str]:
    """Map regularised gamma levels onto the ordinal ramp, darkest = strongest."""

    levels = _gamma_levels(frame)
    if len(levels) > len(GAMMA_RAMP):
        raise ParetoPlotError(
            f"{len(levels)} regularised gamma levels exceed the {len(GAMMA_RAMP)}-step "
            "validated ordinal ramp. Re-validate a longer ramp rather than cycling hues."
        )
    if len(levels) == 1:
        return {levels[0]: GAMMA_RAMP[-1]}
    indices = np.linspace(0, len(GAMMA_RAMP) - 1, num=len(levels)).round().astype(int)
    return {level: GAMMA_RAMP[index] for level, index in zip(levels, indices)}


def _areas(frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [BINS_AREA.get(int(value), _DEFAULT_AREA) if pd.notna(value) else _DEFAULT_AREA
         for value in frame[BINS_COLUMN]],
        dtype=float,
    )


def _style_axis(axis, *, x_column: str, y_column: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_xlabel(AXIS_LABELS[x_column], color=TEXT_PRIMARY)
    axis.set_ylabel(AXIS_LABELS[y_column], color=TEXT_PRIMARY)
    axis.grid(True, color=GRID_INK, linewidth=0.6, linestyle="-", zorder=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(GRID_INK)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _split(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eligible configurations and the ones excluded before the comparison."""

    eligible_mask = (
        _as_bool(candidates[VALID_COLUMN], label=VALID_COLUMN)
        & _as_bool(candidates[FEASIBLE_COLUMN], label=FEASIBLE_COLUMN)
    )
    return candidates.loc[eligible_mask].copy(), candidates.loc[~eligible_mask].copy()


def _draw_excluded(axis, excluded: pd.DataFrame, x_column: str, y_column: str) -> bool:
    """Invalid and infeasible configurations, visibly set apart from the cloud.

    They are drawn hollow and pale so a latent-collapsed or unpaired run is never
    mistaken for a configuration that actually competed. Rows whose objectives
    did not survive aggregation simply have nothing to plot.
    """

    finite = excluded.loc[excluded[x_column].notna() & excluded[y_column].notna()]
    if finite.empty:
        return False
    axis.scatter(
        finite[x_column], finite[y_column],
        s=28.0, facecolors="none", edgecolors=EXCLUDED_INK, linewidths=0.9,
        marker="o", zorder=1,
    )
    return True


def _draw_front_rings(axis, front: pd.DataFrame, x_column: str, y_column: str) -> None:
    """Open rings plus the 95% intervals: the front is a claim with uncertainty."""

    if front.empty:
        return
    axis.errorbar(
        front[x_column], front[y_column],
        xerr=_ci_bounds(front, x_column), yerr=_ci_bounds(front, y_column),
        fmt="none", ecolor=TEXT_SECONDARY, elinewidth=0.9, capsize=2, zorder=3,
    )
    axis.scatter(
        front[x_column], front[y_column],
        s=_areas(front) + 150.0, facecolors="none", edgecolors=TEXT_PRIMARY,
        linewidths=1.5, zorder=4,
    )


def _gamma_legend_handles(
    gamma_colors: Mapping[float, str], *, has_baseline: bool, baseline_filled: bool = False
):
    """Swatches for the ramp, with the baseline set apart rather than ramped.

    ``baseline_filled`` matches the swatch to how the baseline is actually
    drawn: hollow where the marks are hollow, solid where it is a line.
    """

    from matplotlib.lines import Line2D

    handles = []
    if has_baseline:
        handles.append(
            Line2D([], [], marker="s", linestyle="none", markersize=8,
                   markerfacecolor=BASELINE_INK if baseline_filled else "none",
                   markeredgecolor=BASELINE_INK, markeredgewidth=1.4,
                   label=r"$\gamma = 0$  (baseline)")
        )
    handles.extend(
        Line2D([], [], marker="s", linestyle="none", markersize=8,
               markerfacecolor=color, markeredgecolor="none",
               label=rf"$\gamma = {level:g}$")
        for level, color in sorted(gamma_colors.items())
    )
    return handles


def _bins_legend_handles(frame: pd.DataFrame):
    from matplotlib.lines import Line2D

    present = sorted({int(value) for value in frame[BINS_COLUMN].dropna()})
    return [
        Line2D([], [], marker="o", linestyle="none",
               markersize=float(np.sqrt(BINS_AREA.get(bins, _DEFAULT_AREA))),
               markerfacecolor=BASELINE_INK, markeredgecolor="none",
               label=f"{bins} bins")
        for bins in present
    ]


def _scatter_by_gamma(axis, frame: pd.DataFrame, x_column: str, y_column: str,
                      gamma_colors: Mapping[float, str]) -> None:
    baseline = frame.loc[frame[GAMMA_COLUMN] == 0.0]
    if not baseline.empty:
        axis.scatter(
            baseline[x_column], baseline[y_column], s=_areas(baseline),
            facecolors="none", edgecolors=BASELINE_INK, linewidths=1.4, zorder=2,
        )
    regularised = frame.loc[frame[GAMMA_COLUMN] != 0.0]
    if not regularised.empty:
        axis.scatter(
            regularised[x_column], regularised[y_column], s=_areas(regularised),
            c=[gamma_colors[float(value)] for value in regularised[GAMMA_COLUMN]],
            edgecolors=SURFACE, linewidths=0.8, zorder=2,
        )


def faceted_figure(candidates: pd.DataFrame, *, y_column: str, output_path: Path) -> Path:
    """L against one objective, one panel per architecture, shared axes.

    Faceting rather than a marker-shape channel: position, colour and area are
    already three encodings, and three interleaved symbol shapes on top of them
    is more than a reader can decode at once. Side-by-side panels answer "does
    the architecture matter" directly.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible, excluded = _split(candidates)
    present = {str(value) for value in candidates[ARCHITECTURE_COLUMN].dropna()}
    architectures = [name for name in ARCHITECTURE_ORDER if name in present]
    architectures += sorted(present - set(ARCHITECTURE_ORDER))
    if not architectures:
        raise ParetoPlotError("No architecture_id values to facet on.")
    gamma_colors = _gamma_colors(eligible if not eligible.empty else candidates)

    figure, axes = plt.subplots(
        1, len(architectures), figsize=(5.2 * len(architectures), 5.0),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    figure.patch.set_facecolor(SURFACE)
    any_excluded = False

    for axis, architecture in zip(axes, architectures):
        panel = eligible.loc[eligible[ARCHITECTURE_COLUMN] == architecture]
        panel_excluded = excluded.loc[excluded[ARCHITECTURE_COLUMN] == architecture]
        any_excluded |= _draw_excluded(axis, panel_excluded, LEAKAGE_COLUMN, y_column)
        _scatter_by_gamma(axis, panel, LEAKAGE_COLUMN, y_column, gamma_colors)
        _draw_front_rings(
            axis, panel.loc[_as_bool(panel[FRONT_COLUMN], label=FRONT_COLUMN)],
            LEAKAGE_COLUMN, y_column,
        )
        _style_axis(axis, x_column=LEAKAGE_COLUMN, y_column=y_column)
        axis.set_title(architecture, color=TEXT_PRIMARY, fontsize=11)

    for axis in axes[1:]:
        axis.set_ylabel("")

    handles = _gamma_legend_handles(
        gamma_colors, has_baseline=bool((eligible[GAMMA_COLUMN] == 0.0).any())
    )
    handles += _bins_legend_handles(eligible)
    handles += _ring_legend_handle()
    if any_excluded:
        handles += _excluded_legend_handle()
    figure.legend(
        handles=handles, loc="outside right upper", frameon=False,
        labelcolor=TEXT_SECONDARY, fontsize=9,
    )
    figure.suptitle(
        f"Validation Pareto study: $L$ against {_plain(y_column)}",
        color=TEXT_PRIMARY, fontsize=13,
    )
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return output_path


def _plain(column: str) -> str:
    return {
        EFFICIENCY_COLUMN: r"$\epsilon_{\mathrm{median}}$",
        CORRELATION_COLUMN: "$E$",
        LEAKAGE_COLUMN: "$L$",
    }[column]


def _ring_legend_handle():
    from matplotlib.lines import Line2D

    return [Line2D([], [], marker="o", linestyle="none", markersize=11,
                   markerfacecolor="none", markeredgecolor=TEXT_PRIMARY,
                   markeredgewidth=1.5, label="On the Pareto front")]


def _excluded_legend_handle():
    from matplotlib.lines import Line2D

    return [Line2D([], [], marker="o", linestyle="none", markersize=6,
                   markerfacecolor="none", markeredgecolor=EXCLUDED_INK,
                   markeredgewidth=0.9, label="Invalid or infeasible")]


def third_objective_figure(candidates: pd.DataFrame, *, output_path: Path) -> Path:
    """L against efficiency with residual correlation as the colour.

    The companion to the faceted figures and, for reading the front, the more
    honest of the two: dominance is decided in three dimensions, so a ringed
    point can sit inside the cloud of a two-dimensional projection purely
    because it wins on the axis that projection dropped. Here that axis is the
    colour.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    eligible, excluded = _split(candidates)
    # The 100-150 steps of the sequential ramp are specified to recede toward
    # the surface, which suits a heatmap and not an 8px mark. Marks start at
    # step 200 so the lowest-E points stay visible.
    colormap = LinearSegmentedColormap.from_list(
        "sequential_blue_marks", list(SEQUENTIAL_BLUE[2:])
    )

    figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)
    any_excluded = _draw_excluded(axis, excluded, LEAKAGE_COLUMN, EFFICIENCY_COLUMN)
    points = axis.scatter(
        eligible[LEAKAGE_COLUMN], eligible[EFFICIENCY_COLUMN],
        c=eligible[CORRELATION_COLUMN], cmap=colormap,
        s=90.0, edgecolors=SURFACE, linewidths=0.8, zorder=2,
    )
    _draw_front_rings(
        axis, eligible.loc[_as_bool(eligible[FRONT_COLUMN], label=FRONT_COLUMN)],
        LEAKAGE_COLUMN, EFFICIENCY_COLUMN,
    )
    _style_axis(axis, x_column=LEAKAGE_COLUMN, y_column=EFFICIENCY_COLUMN)

    colorbar = figure.colorbar(points, ax=axis)
    colorbar.set_label(AXIS_LABELS[CORRELATION_COLUMN] + "  (lower is better)",
                       color=TEXT_PRIMARY)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    colorbar.outline.set_edgecolor(GRID_INK)

    handles = _ring_legend_handle() + (_excluded_legend_handle() if any_excluded else [])
    axis.legend(handles=handles, frameon=False, labelcolor=TEXT_SECONDARY, fontsize=9)
    axis.set_title("All three objectives at once", color=TEXT_PRIMARY, fontsize=12)
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return output_path


#: Above this many front members, direct-labelling every polyline collides with
#: itself. The rest stay drawn but recede, and only the best-ranked are named.
PARALLEL_LABEL_LIMIT = 8
#: Minimum vertical separation between two right-hand labels, in axis fraction.
_LABEL_SPACING = 0.045


def _spread_labels(values: Sequence[float], *, spacing: float = _LABEL_SPACING) -> list[float]:
    """Push overlapping label anchors apart, preserving their order.

    One upward sweep, then one downward sweep to pull anything shoved past the
    top back inside. Enough for the handful of labels this figure draws.
    """

    order = sorted(range(len(values)), key=lambda index: values[index])
    placed = list(values)
    for position, index in enumerate(order):
        if position and placed[index] - placed[order[position - 1]] < spacing:
            placed[index] = placed[order[position - 1]] + spacing
    for position in range(len(order) - 2, -1, -1):
        index, above = order[position], order[position + 1]
        if placed[above] - placed[index] < spacing:
            placed[index] = placed[above] - spacing
    return placed


def _short_id(configuration_id: str) -> str:
    return str(configuration_id).split("__", 1)[-1] if "__" in str(configuration_id) else str(configuration_id)


def front_parallel_coordinates(front: pd.DataFrame, *, output_path: Path) -> Path:
    """The front itself, on all three objectives at once.

    A scatter of the front alone is a handful of dots on rescaled axes, which
    invites a false comparison with the figures above. Parallel coordinates is
    what actually answers the question the front poses -- what does each member
    cost on the objectives it loses -- because every member is one polyline
    across all three.

    Each axis is oriented so up is better and annotated with its own raw range,
    since the three objectives share no units. When the front is larger than
    PARALLEL_LABEL_LIMIT, only the best-ranked members are emphasised and named;
    the rest recede rather than being dropped, so the shape of the whole front
    is still visible without the labels colliding into noise.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if front.empty:
        raise ParetoPlotError("The Pareto front is empty; there is nothing to draw.")

    axes_spec = (
        (LEAKAGE_COLUMN, "lower"),
        (CORRELATION_COLUMN, "lower"),
        (EFFICIENCY_COLUMN, "higher"),
    )
    gamma_colors = _gamma_colors(front)

    ordered = front.copy()
    if "pareto_rank" in ordered.columns and ordered["pareto_rank"].notna().any():
        ordered = ordered.sort_values("pareto_rank", kind="stable")
    emphasised = min(len(ordered), PARALLEL_LABEL_LIMIT)

    normalised = np.empty((len(ordered), len(axes_spec)), dtype=float)
    for position, (column, better) in enumerate(axes_spec):
        values = ordered[column].to_numpy(dtype=float)
        low, high = float(values.min()), float(values.max())
        span = high - low
        # A degenerate axis (every member equal) is drawn flat down the middle
        # rather than amplified into a fake spread by dividing by ~zero.
        scaled = np.full_like(values, 0.5) if span <= 0.0 else (values - low) / span
        normalised[:, position] = 1.0 - scaled if better == "lower" else scaled

    figure, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    positions = np.arange(len(axes_spec), dtype=float)

    for position in positions:
        axis.axvline(position, color=GRID_INK, linewidth=1.0, zorder=0)

    # Background first, so the ranked members are never buried under the rest.
    for row in range(emphasised, len(ordered)):
        gamma = float(ordered.iloc[row][GAMMA_COLUMN])
        axis.plot(
            positions, normalised[row],
            color=BASELINE_INK if gamma == 0.0 else gamma_colors[gamma],
            linewidth=1.0, alpha=0.35, zorder=1, solid_capstyle="round",
        )

    for row in range(emphasised):
        configuration = ordered.iloc[row]
        gamma = float(configuration[GAMMA_COLUMN])
        axis.plot(
            positions, normalised[row],
            color=BASELINE_INK if gamma == 0.0 else gamma_colors[gamma],
            linewidth=2.2, marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=0.8, zorder=3, solid_capstyle="round",
        )

    anchors = _spread_labels([float(normalised[row, -1]) for row in range(emphasised)])
    for row, anchor in enumerate(anchors):
        configuration = ordered.iloc[row]
        rank = configuration.get("pareto_rank")
        prefix = "" if pd.isna(rank) else f"#{int(rank)}  "
        axis.annotate(
            f"{prefix}{_short_id(configuration[CONFIGURATION_ID_COLUMN])}",
            xy=(positions[-1], normalised[row, -1]),
            xytext=(positions[-1] + 0.10, anchor),
            textcoords="data", va="center", fontsize=8.5, color=TEXT_SECONDARY,
            arrowprops=dict(arrowstyle="-", color=GRID_INK, linewidth=0.8,
                            shrinkA=0, shrinkB=2),
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{_plain(column)}\n({better} is better)" for column, better in axes_spec],
        color=TEXT_PRIMARY, fontsize=10,
    )
    # Range annotations hug their own axis line and sit inside the plot, so the
    # legend (outside, below) can never land on top of them.
    for position, (column, _) in zip(positions, axes_spec):
        values = ordered[column]
        align = "left" if position == 0 else "center"
        offset = 4 if position == 0 else 0
        axis.annotate(f"{values.max():.4g}", xy=(position, 1.0), xytext=(offset, 9),
                      textcoords="offset points", ha=align, fontsize=8,
                      color=TEXT_SECONDARY)
        axis.annotate(f"{values.min():.4g}", xy=(position, 0.0), xytext=(offset, -15),
                      textcoords="offset points", ha=align, fontsize=8,
                      color=TEXT_SECONDARY)

    axis.set_yticks([])
    axis.set_ylim(-0.14, 1.14)
    axis.set_xlim(-0.25, len(axes_spec) - 1 + 1.35)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)

    handles = _gamma_legend_handles(
        gamma_colors,
        has_baseline=bool((ordered[GAMMA_COLUMN] == 0.0).any()),
        baseline_filled=True,
    )
    figure.legend(
        handles=handles, loc="outside lower center", frameon=False,
        labelcolor=TEXT_SECONDARY, fontsize=9, ncol=len(handles),
    )
    hidden = len(ordered) - emphasised
    subtitle = f", {emphasised} best-ranked named" if hidden else ""
    axis.set_title(
        f"Pareto front: {len(ordered)} configuration(s) across all three objectives{subtitle}",
        color=TEXT_PRIMARY, fontsize=12,
    )
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return output_path


FIGURE_FILENAMES = {
    "leakage_vs_efficiency_by_architecture": "pareto_L_vs_efficiency_by_architecture.png",
    "leakage_vs_correlation_by_architecture": "pareto_L_vs_correlation_by_architecture.png",
    "leakage_vs_efficiency_by_correlation": "pareto_L_vs_efficiency_coloured_by_E.png",
    "front_parallel_coordinates": "pareto_front_parallel_coordinates.png",
}

_REQUIRED_COLUMNS = (
    CONFIGURATION_ID_COLUMN, GAMMA_COLUMN, BINS_COLUMN, ARCHITECTURE_COLUMN,
    VALID_COLUMN, FEASIBLE_COLUMN, FRONT_COLUMN,
    LEAKAGE_COLUMN, CORRELATION_COLUMN, EFFICIENCY_COLUMN,
)


def write_pareto_figures(
    candidates: pd.DataFrame, front: pd.DataFrame, *, output_dir: str | Path
) -> dict[str, Path]:
    """Write every Phase 4 figure and return where each one landed."""

    missing = sorted(set(_REQUIRED_COLUMNS) - set(candidates.columns))
    if missing:
        raise ParetoPlotError(f"The candidate table is missing columns: {missing}.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    written["leakage_vs_efficiency_by_architecture"] = faceted_figure(
        candidates, y_column=EFFICIENCY_COLUMN,
        output_path=destination / FIGURE_FILENAMES["leakage_vs_efficiency_by_architecture"],
    )
    written["leakage_vs_correlation_by_architecture"] = faceted_figure(
        candidates, y_column=CORRELATION_COLUMN,
        output_path=destination / FIGURE_FILENAMES["leakage_vs_correlation_by_architecture"],
    )
    written["leakage_vs_efficiency_by_correlation"] = third_objective_figure(
        candidates,
        output_path=destination / FIGURE_FILENAMES["leakage_vs_efficiency_by_correlation"],
    )
    if not front.empty:
        written["front_parallel_coordinates"] = front_parallel_coordinates(
            front,
            output_path=destination / FIGURE_FILENAMES["front_parallel_coordinates"],
        )
    return written


def plot_pareto_study(
    candidates_path: str | Path, front_path: str | Path, *, output_dir: str | Path
) -> dict[str, Path]:
    """Read the two Phase 3 tables and draw every figure from them."""

    return write_pareto_figures(
        pd.read_csv(candidates_path), pd.read_csv(front_path), output_dir=output_dir
    )


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw the Phase 4 figures from the Phase 3 selection tables."
    )
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Phase 3 pareto_candidates.csv.")
    parser.add_argument("--front", type=Path, required=True,
                        help="Phase 3 pareto_front.csv.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for the figures.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    written = plot_pareto_study(args.candidates, args.front, output_dir=args.output_dir)
    for name, path in sorted(written.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
