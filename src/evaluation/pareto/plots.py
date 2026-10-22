"""Phase 4: the figures of the validation Pareto study.

Phase 3 selects and writes tables; this module draws and writes nothing else.
Keeping them apart means a figure can be restyled without re-running selection,
and selection cannot be silently changed by a plotting edit.

Every figure reads the two Phase 3 tables and nothing else, so what is drawn is
exactly what was selected. Each configuration is a single training run, so
points carry no error bars.

The figures:

* One scatter per pair of objectives: L against E, L against 1 - efficiency and
  E against 1 - efficiency. Efficiency enters as the cost ``1 - eff``, so lower
  is better on both axes of every pair and the best corner is bottom left. The
  two pairs with ``1 - eff`` are drawn twice, with a linear and with a
  logarithmic ``1 - eff`` axis. Every configuration gets the same mark: no
  channel for architecture, gamma or front membership, only position.
* All three objectives at once: L (y) against E (x), coloured by the median
  signal efficiency.
* The Pareto front in parallel coordinates, coloured by gamma: gamma spans four
  decades (0.01 to 100), a magnitude rather than a handful of categories, so it
  takes the sequential blue ramp on a logarithmic scale, read off a colourbar.
  Gamma zero is deliberately NOT the lightest step of that ramp: it is the
  unregularised baseline every feasibility constraint is measured against, so
  it is drawn as a red square with its own legend entry.

Invalid and infeasible configurations are drawn hollow and grey in the scatters,
so a collapsed run is never mistaken for one that competed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .selection import (
    CONFIGURATION_ID_COLUMN,
    CORRELATION_COLUMN,
    EFFICIENCY_COLUMN,
    FEASIBLE_COLUMN,
    LEAKAGE_COLUMN,
    VALID_COLUMN,
)

GAMMA_COLUMN = "mi_gamma"
#: Derived column: the efficiency cost 1 - median efficiency.
INEFFICIENCY_COLUMN = "one_minus_median_efficiency"

#: Sequential blue, 100 -> 700.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
#: Gamma ramp: sequential blue from step 250, whose light end still clears the
#: surface at 2.06:1, so the weakest regularisation stays visible as a mark.
GAMMA_RAMP = SEQUENTIAL_BLUE[3:]
#: The one colour of every configuration in the pairwise scatters.
POINT_COLOR = SEQUENTIAL_BLUE[9]
#: The gamma-zero baseline: categorical red, a square, never part of the blue
#: ordinal ramp. 3.85:1 on the chart surface, so a thin open edge stays
#: legible, and deliberately not the reserved "critical" status token.
BASELINE_MARK = "#e34948"
BASELINE_MARKER = "s"
EXCLUDED_INK = "#bdbcb5"
GRID_INK = "#e8e7e3"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

AXIS_LABELS = {
    LEAKAGE_COLUMN: "Leakage $L$ (worst of four probes, held-out $R^2$)",
    EFFICIENCY_COLUMN: r"Signal efficiency $\epsilon_{\mathrm{median}}$",
    CORRELATION_COLUMN: "Residual correlation $E$",
    INEFFICIENCY_COLUMN: r"$1 - \epsilon_{\mathrm{median}}$",
}

#: (y, x) of the pairwise scatters, and whether a log-x version is drawn too.
PAIRS = (
    (LEAKAGE_COLUMN, CORRELATION_COLUMN, False),
    (LEAKAGE_COLUMN, INEFFICIENCY_COLUMN, True),
    (CORRELATION_COLUMN, INEFFICIENCY_COLUMN, True),
)


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


def _gamma_levels(frame: pd.DataFrame) -> list[float]:
    return sorted({float(value) for value in frame[GAMMA_COLUMN].dropna() if float(value) > 0.0})


def _gamma_norm(levels: Sequence[float]):
    """Logarithmic colour scale over the regularised gamma levels."""

    from matplotlib.colors import LogNorm

    low, high = min(levels), max(levels)
    if low == high:
        low, high = low / 10.0, high * 10.0
    return LogNorm(vmin=low, vmax=high)


def _gamma_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("gamma_sequential_blue", list(GAMMA_RAMP))


def _gamma_colors(frame: pd.DataFrame) -> dict[float, str]:
    """Map regularised gamma levels onto the log-scaled ramp, darkest = strongest."""

    from matplotlib.colors import to_hex

    levels = _gamma_levels(frame)
    if not levels:
        return {}
    norm, colormap = _gamma_norm(levels), _gamma_colormap()
    return {level: to_hex(colormap(norm(level))) for level in levels}


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
            Line2D([], [], marker=BASELINE_MARKER, linestyle="none", markersize=8,
                   markerfacecolor=BASELINE_MARK if baseline_filled else "none",
                   markeredgecolor=BASELINE_MARK, markeredgewidth=1.6,
                   label=r"$\gamma = 0$  (baseline)")
        )
    handles.extend(
        Line2D([], [], marker="s", linestyle="none", markersize=8,
               markerfacecolor=color, markeredgecolor="none",
               label=rf"$\gamma = {level:g}$")
        for level, color in sorted(gamma_colors.items())
    )
    return handles



def _plain(column: str) -> str:
    return {
        EFFICIENCY_COLUMN: r"$\epsilon_{\mathrm{median}}$",
        INEFFICIENCY_COLUMN: r"$1 - \epsilon_{\mathrm{median}}$",
        CORRELATION_COLUMN: "$E$",
        LEAKAGE_COLUMN: "$L$",
    }[column]


def _excluded_legend_handle():
    from matplotlib.lines import Line2D

    return [Line2D([], [], marker="o", linestyle="none", markersize=6,
                   markerfacecolor="none", markeredgecolor=EXCLUDED_INK,
                   markeredgewidth=0.9, label="Invalid or infeasible")]



def _with_inefficiency(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    frame[INEFFICIENCY_COLUMN] = 1.0 - frame[EFFICIENCY_COLUMN].astype(float)
    return frame


def _plain_ticks(axis) -> None:
    """Full numbers on a linear axis: no offset ("+9.947e-1") and no 1e-3 factor."""

    axis.set_major_formatter(_number_formatter())


def _number_formatter():
    """Plain decimals, the same number of them on every tick of an axis."""

    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    return formatter


def _log_axis(axis, values: np.ndarray) -> None:
    """Log scale whose ticks stay readable when the data span less than a decade.

    1 - efficiency lies within 0.5 % of 1, so the decade ticks of a log axis are
    all outside the data and matplotlib would label nothing. Within one decade
    the ticks are placed at round values instead; across decades the usual log
    ticks are kept.
    """

    from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator

    if axis is axis.axes.xaxis:
        axis.axes.set_xscale("log")
    else:
        axis.axes.set_yscale("log")
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return
    low, high = float(finite.min()), float(finite.max())
    if high / low >= 10.0:
        return
    ticks = [float(t) for t in MaxNLocator(nbins=5).tick_values(low, high) if low <= t <= high]
    axis.set_major_locator(FixedLocator(ticks or [low, high]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(_number_formatter())


def pair_figure(
    candidates: pd.DataFrame,
    *,
    y_column: str,
    x_column: str,
    output_path: Path,
    log_x: bool = False,
) -> Path:
    """One objective against another, every configuration in the same colour.

    ``x_column`` / ``y_column`` may be INEFFICIENCY_COLUMN (1 - median
    efficiency), which is derived here. ``log_x`` puts the x axis on a log scale.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = _with_inefficiency(candidates)
    eligible, excluded = _split(frame)

    figure, axis = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)
    any_excluded = _draw_excluded(axis, excluded, x_column, y_column)
    axis.scatter(
        eligible[x_column], eligible[y_column],
        s=70.0, c=POINT_COLOR, edgecolors=SURFACE, linewidths=0.8, zorder=2,
    )
    _style_axis(axis, x_column=x_column, y_column=y_column)
    _plain_ticks(axis.yaxis)
    if log_x:
        _log_axis(axis.xaxis, frame[x_column].to_numpy(dtype=float))
        axis.set_xlabel(AXIS_LABELS[x_column] + "  (log scale)", color=TEXT_PRIMARY)
    else:
        _plain_ticks(axis.xaxis)
    if any_excluded:
        axis.legend(handles=_excluded_legend_handle(), frameon=False,
                    labelcolor=TEXT_SECONDARY, fontsize=9)
    axis.set_title(f"{_plain(y_column)} against {_plain(x_column)}",
                   color=TEXT_PRIMARY, fontsize=12)
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return output_path


def three_objective_figure(candidates: pd.DataFrame, *, output_path: Path) -> Path:
    """All three objectives at once: L (y) against E (x), coloured by efficiency.

    Dominance is decided in three dimensions, so this is the scatter in which the
    front can actually be read: a point that looks beaten in L and E can still
    win on the efficiency carried by its colour. Darker means a higher median
    efficiency.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    eligible, excluded = _split(candidates)
    # The 100-150 steps of the sequential ramp recede toward the surface, which
    # suits a heatmap and not an 8px mark. Marks start at step 200 so the lowest
    # efficiencies stay visible.
    colormap = LinearSegmentedColormap.from_list(
        "sequential_blue_marks", list(SEQUENTIAL_BLUE[2:])
    )

    figure, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)
    any_excluded = _draw_excluded(axis, excluded, CORRELATION_COLUMN, LEAKAGE_COLUMN)
    points = axis.scatter(
        eligible[CORRELATION_COLUMN], eligible[LEAKAGE_COLUMN],
        c=eligible[EFFICIENCY_COLUMN], cmap=colormap,
        s=90.0, edgecolors=SURFACE, linewidths=0.8, zorder=2,
    )
    _style_axis(axis, x_column=CORRELATION_COLUMN, y_column=LEAKAGE_COLUMN)
    _plain_ticks(axis.xaxis)
    _plain_ticks(axis.yaxis)

    colorbar = figure.colorbar(points, ax=axis)
    colorbar.set_label(AXIS_LABELS[EFFICIENCY_COLUMN], color=TEXT_PRIMARY)
    colorbar.ax.yaxis.set_major_formatter(_number_formatter())
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    colorbar.outline.set_edgecolor(GRID_INK)

    if any_excluded:
        axis.legend(handles=_excluded_legend_handle(), frameon=False,
                    labelcolor=TEXT_SECONDARY, fontsize=9)
    axis.set_title(r"All three objectives: $L$ against $E$, coloured by $\epsilon_{\mathrm{median}}$",
                   color=TEXT_PRIMARY, fontsize=12)
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
            color=BASELINE_MARK if gamma == 0.0 else gamma_colors[gamma],
            linewidth=1.0, alpha=0.35, zorder=1, solid_capstyle="round",
        )

    for row in range(emphasised):
        configuration = ordered.iloc[row]
        gamma = float(configuration[GAMMA_COLUMN])
        axis.plot(
            positions, normalised[row],
            color=BASELINE_MARK if gamma == 0.0 else gamma_colors[gamma],
            linewidth=2.2, marker=BASELINE_MARKER if gamma == 0.0 else "o",
            markersize=8, markeredgecolor=SURFACE,
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

    levels = _gamma_levels(ordered)
    if levels:
        from matplotlib.cm import ScalarMappable

        colorbar = figure.colorbar(
            ScalarMappable(norm=_gamma_norm(levels), cmap=_gamma_colormap()),
            ax=axis, location="bottom", shrink=0.5, aspect=40, pad=0.02,
        )
        colorbar.set_label(r"MI weight $\gamma$ (log scale)", color=TEXT_PRIMARY)
        colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        colorbar.outline.set_edgecolor(GRID_INK)
    handles = _gamma_legend_handles(
        {},
        has_baseline=bool((ordered[GAMMA_COLUMN] == 0.0).any()),
        baseline_filled=True,
    )
    if handles:
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


def _pair_key(y_column: str, x_column: str, log_x: bool) -> str:
    names = {LEAKAGE_COLUMN: "leakage", CORRELATION_COLUMN: "correlation",
             INEFFICIENCY_COLUMN: "inefficiency"}
    return f"{names[y_column]}_vs_{names[x_column]}" + ("_log" if log_x else "")


def _pair_filename(y_column: str, x_column: str, log_x: bool) -> str:
    names = {LEAKAGE_COLUMN: "L", CORRELATION_COLUMN: "E",
             INEFFICIENCY_COLUMN: "1_minus_eff"}
    return f"pareto_{names[y_column]}_vs_{names[x_column]}" + ("_log" if log_x else "") + ".png"


FIGURE_FILENAMES = {
    **{
        _pair_key(y, x, log_x): _pair_filename(y, x, log_x)
        for y, x, with_log in PAIRS
        for log_x in ((False, True) if with_log else (False,))
    },
    "three_objectives": "pareto_L_vs_E_coloured_by_eff.png",
    "front_parallel_coordinates": "pareto_front_parallel_coordinates.png",
}

_REQUIRED_COLUMNS = (
    CONFIGURATION_ID_COLUMN, GAMMA_COLUMN, VALID_COLUMN, FEASIBLE_COLUMN,
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

    for y_column, x_column, with_log in PAIRS:
        for log_x in (False, True) if with_log else (False,):
            key = _pair_key(y_column, x_column, log_x)
            written[key] = pair_figure(
                candidates, y_column=y_column, x_column=x_column, log_x=log_x,
                output_path=destination / FIGURE_FILENAMES[key],
            )
    written["three_objectives"] = three_objective_figure(
        candidates, output_path=destination / FIGURE_FILENAMES["three_objectives"],
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
