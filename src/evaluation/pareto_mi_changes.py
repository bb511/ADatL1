"""Phase 4c of the Pareto study: how the MI weight γ and the binning move each metric.

Two one-dimensional sweeps through the Phase 3 table ``pareto_candidates.csv``:

* **γ sweep** at a fixed nominal bin count (default 50): x = ``mi_gamma``.
* **Bin sweep** at a fixed γ (default 0.1): x = the *effective* number of MI bins.
  The quantile binner merges coinciding edges, so e.g. 50 nominal bins give 48
  effective ones. The count is read per run from the bin-width diagnostic under the
  local checkpoints, ``<checkpoints>/<study>/<run>/plots/mi_diagnostics/data/
  epoch_*/mi_bin_widths_epoch*.csv`` (``src.analysis.decorrelation``), and runs are
  found through the study map.

For each of the 14 metrics of the phase 4b matrices one PNG per sweep is drawn, plus
an overview grid per sweep and the plotted table as CSV::

    <output>/gamma/NN_<metric>.png             <output>/00_overview_gamma.png
    <output>/effective_bins/NN_<metric>.png    <output>/00_overview_effective_bins.png
    <output>/sweep_gamma.csv                   <output>/sweep_effective_bins.csv

Marks: feasible runs are joined by a line, which breaks at rejected runs; rejected
runs (collapse rule) are open red markers; Pareto-front runs carry a black ring; the
γ = 0 baseline is a dashed grey reference (in the bin sweep: the γ = 0 run at each
bin count).

Usage::

    python3 scripts/plot_pareto_mi_changes.py \\
        --candidates       <STUDY_ROOT>/phase3/pareto_candidates.csv \\
        --study-map        <STUDY_ROOT>/study_map.yaml \\
        --checkpoints-root checkpoints \\
        --output-dir       <STUDY_ROOT>/phase4/mi-changes
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.decorrelation import load_effective_bin_count
from src.evaluation.pareto_matrices import (
    BINS,
    CRITICAL,
    DIAGNOSTICS,
    GAMMA,
    INK,
    INK2,
    METRICS,
    MUTED,
    SCALE,
    SCALE_UNIT,
    SURFACE,
    _collapse_duplicate_cells,
    _read_yaml,
    _style,
    fmt,
    load_candidates,
)

EFFECTIVE_BINS = "effective_bins"
SERIES = "#2a78d6"  # categorical slot 1 (dataviz reference palette), the one series
BASELINE = INK2
SWEEP_GAMMA, SWEEP_BINS = "gamma", "effective_bins"


class ParetoMiChangeError(ValueError):
    """The candidates table cannot be drawn as γ / bin sweeps."""


@dataclass
class Sweep:
    """One one-dimensional cut through the γ × bins grid."""

    name: str            # SWEEP_GAMMA or SWEEP_BINS
    x_column: str        # GAMMA or EFFECTIVE_BINS
    x_label: str
    fixed_text: str      # what is held fixed, for the subtitle
    table: pd.DataFrame  # one row per configuration, sorted by x
    baseline: pd.DataFrame  # γ = 0 rows to draw as reference (x_column, metrics)


# ------------------------------------------------------------------ effective bins
def _local_run_dir(run: Mapping[str, Any], checkpoints_root: Path) -> Optional[Path]:
    """Run directory of a study-map entry under the local checkpoints root."""
    run_dir = str(run.get("checkpoint_run_dir") or "")
    candidates = []
    if "/checkpoints/" in run_dir:
        candidates.append(Path(checkpoints_root) / run_dir.split("/checkpoints/", 1)[1])
    if run_dir:
        candidates.append(Path(run_dir))
    return next((path for path in candidates if path.is_dir()), None)


def effective_bins_by_configuration(study_map: Path, checkpoints_root: Path,
                                    configuration_ids: Sequence[str]) -> Dict[str, float]:
    """Effective MI bin count per configuration (mean over its seeds)."""
    runs = (_read_yaml(study_map) or {}).get("runs") or []
    wanted = set(configuration_ids)
    counts: Dict[str, List[int]] = {}
    missing: List[str] = []
    for run in runs:
        cid = run.get("configuration_id")
        if cid not in wanted:
            continue
        run_dir = _local_run_dir(run, checkpoints_root)
        if run_dir is None:
            missing.append(f"{cid}: no local run directory for {run.get('checkpoint_run_dir')}")
            continue
        try:
            counts.setdefault(cid, []).append(load_effective_bin_count(run_dir))
        except (FileNotFoundError, ValueError) as error:
            missing.append(f"{cid}: {error}")
    absent = sorted(wanted - set(counts))
    if absent:
        details = "\n  ".join(missing) or "not in the study map"
        raise ParetoMiChangeError(
            f"No effective bin count for {len(absent)} configuration(s): {absent}\n  {details}")
    return {cid: float(np.mean(values)) for cid, values in counts.items()}


# ------------------------------------------------------------------------ sweeps
def _close(series: pd.Series, value: float) -> pd.Series:
    return np.isclose(series.astype(float), float(value))


def build_sweeps(table: pd.DataFrame, *, bins_for_gamma: int = 50, gamma_for_bins: float = 0.1,
                 effective_bins: Optional[Mapping[str, float]] = None) -> List[Sweep]:
    """Cut the per-run table into the γ sweep and the effective-bin sweep."""
    table = table.copy()
    if effective_bins is not None:
        table[EFFECTIVE_BINS] = table["configuration_id"].map(effective_bins)
    table, _ = _collapse_duplicate_cells(table)

    gamma_rows = table[_close(table[BINS], bins_for_gamma)].sort_values(GAMMA)
    if len(gamma_rows) < 2:
        raise ParetoMiChangeError(f"Fewer than two runs with {bins_for_gamma} bins to sweep γ over.")
    eff_text = ""
    if EFFECTIVE_BINS in gamma_rows and gamma_rows[EFFECTIVE_BINS].notna().any():
        eff = gamma_rows[EFFECTIVE_BINS].dropna().unique()
        eff_text = f" ({', '.join(f'{v:g}' for v in eff)} effective)"
    sweeps = [Sweep(
        name=SWEEP_GAMMA, x_column=GAMMA, x_label="MI weight γ (mi_gamma)",
        fixed_text=f"{int(bins_for_gamma)} FET.Et bins{eff_text}",
        table=gamma_rows.reset_index(drop=True),
        baseline=gamma_rows[_close(gamma_rows[GAMMA], 0)].reset_index(drop=True),
    )]

    if effective_bins is not None:
        bin_rows = table[_close(table[GAMMA], gamma_for_bins)].sort_values(EFFECTIVE_BINS)
        if len(bin_rows) < 2:
            raise ParetoMiChangeError(f"Fewer than two runs with γ = {gamma_for_bins:g} to sweep bins over.")
        baseline = table[_close(table[GAMMA], 0) & table[BINS].isin(bin_rows[BINS])]
        sweeps.append(Sweep(
            name=SWEEP_BINS, x_column=EFFECTIVE_BINS, x_label="Effective number of FET.Et bins",
            fixed_text=f"γ = {gamma_for_bins:g}",
            table=bin_rows.reset_index(drop=True),
            baseline=baseline.sort_values(EFFECTIVE_BINS).reset_index(drop=True),
        ))
    return sweeps


def bin_mapping_text(sweep: Sweep) -> str:
    """'nominal → effective' pairs of a bin sweep, for the footer."""
    pairs = [f"{int(b)}→{e:g}" for b, e in zip(sweep.table[BINS], sweep.table[EFFECTIVE_BINS])]
    return "Nominal → effective bins: " + ", ".join(pairs)


# ----------------------------------------------------------------------- drawing
def _y_formatter(f: Optional[str]):
    from matplotlib.ticker import FuncFormatter

    mult = SCALE.get(f, 1.0)
    return FuncFormatter(lambda v, _: f"{v * mult:g}")


def _best_index(values: pd.Series, feasible: pd.Series, better: str) -> Optional[int]:
    pool = values.where(feasible)
    if pool.notna().sum() == 0:
        return None
    return int(pool.idxmin() if better == "lower" else pool.idxmax())


def draw_metric(ax, sweep: Sweep, column: str, better: str, f: Optional[str], *,
                compact: bool = False) -> None:
    """Draw one metric of one sweep into ``ax``."""
    rows = sweep.table
    x = rows[sweep.x_column].to_numpy(dtype=float)
    y = rows[column].to_numpy(dtype=float)
    feasible = rows["feasible"].to_numpy(dtype=bool)
    front = rows["is_pareto_front"].to_numpy(dtype=bool)
    marker = 5.5 if compact else 8.5

    # γ = 0 reference.
    base = sweep.baseline
    if len(base) and column in base and base[column].notna().any():
        if sweep.name == SWEEP_GAMMA or base[column].nunique() == 1:
            value = float(base[column].dropna().iloc[0])
            ax.axhline(value, color=BASELINE, lw=1.1, ls=(0, (4, 3)), zorder=1)
            if not compact:
                ax.annotate(f"γ = 0 baseline: {fmt(value, f)}", xy=(1, value),
                            xycoords=("axes fraction", "data"), xytext=(-4, 4),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=8, color=INK2)
        else:
            ax.plot(base[sweep.x_column], base[column], color=BASELINE, lw=1.1, ls=(0, (4, 3)),
                    marker="o", ms=3.5, zorder=1)

    # Feasible runs: one line, broken at rejected runs.
    ax.plot(x, np.where(feasible, y, np.nan), color=SERIES, lw=2, solid_capstyle="round",
            solid_joinstyle="round", zorder=2)
    ax.plot(x[feasible], y[feasible], ls="none", marker="o", ms=marker, mfc=SERIES, mec=SURFACE,
            mew=2, zorder=3)
    # Rejected runs: open red markers.
    ax.plot(x[~feasible], y[~feasible], ls="none", marker="o", ms=marker, mfc=SURFACE,
            mec=CRITICAL, mew=1.8, zorder=3)
    # Pareto front: black ring.
    ax.plot(x[front], y[front], ls="none", marker="o", ms=marker + (5 if compact else 7),
            mfc="none", mec=INK, mew=1.6 if compact else 2.0, zorder=4)

    ax.yaxis.set_major_formatter(_y_formatter(f))
    ax.grid(axis="y", color=MUTED, lw=0.8, ls="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, length=3, labelsize=7.5 if compact else 9)
    if sweep.name == SWEEP_GAMMA:
        ax.set_xlim(-0.03, max(1.0, float(np.nanmax(x))) + 0.03)
    else:
        span = float(np.nanmax(x) - np.nanmin(x))
        ax.set_xlim(float(np.nanmin(x)) - 0.04 * span, float(np.nanmax(x)) + 0.04 * span)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    if len(base) and column in base and base[column].notna().any():
        ymin, ymax = min(ymin, base[column].min()), max(ymax, base[column].max())
    pad = 0.12 * (ymax - ymin) if ymax > ymin else max(abs(ymax) * 0.05, 0.5)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Direct label on the best feasible run only, kept inside the axes.
    best = _best_index(rows[column], rows["feasible"].astype(bool), better)
    if best is not None and not compact:
        xb, yb = float(x[best]), float(y[best])
        x0, x1 = ax.get_xlim()
        frac = (xb - x0) / (x1 - x0)
        ha, dx = ("left", 6) if frac < 0.2 else ("right", -6) if frac > 0.8 else ("center", 0)
        where = f"γ {xb:g}" if sweep.name == SWEEP_GAMMA else f"{xb:g} eff. bins"
        ax.annotate(f"best feasible: {fmt(yb, f)} ({where})", xy=(xb, yb),
                    xytext=(dx, 13 if better == "higher" else -17), textcoords="offset points",
                    ha=ha, fontsize=8.5, color=INK, fontweight="bold")


def _legend_handles(compact: bool = False):
    from matplotlib.lines import Line2D

    ms = 6 if compact else 8
    return [
        Line2D([], [], color=SERIES, lw=2, marker="o", ms=ms, mfc=SERIES, mec=SURFACE, label="Feasible run"),
        Line2D([], [], ls="none", marker="o", ms=ms, mfc=SURFACE, mec=CRITICAL, mew=1.8,
               label="Rejected (collapse rule)"),
        Line2D([], [], ls="none", marker="o", ms=ms + 5, mfc="none", mec=INK, mew=1.8, label="Pareto front"),
        Line2D([], [], color=BASELINE, lw=1.1, ls=(0, (4, 3)), label="γ = 0 baseline"),
    ]


def _unit(f: Optional[str]) -> str:
    return f" [{SCALE_UNIT[f]}]" if f in SCALE_UNIT else ""


def _footer(sweep: Sweep, study_name: str, source_label: str) -> str:
    n_rej = int((~sweep.table["feasible"].astype(bool)).sum())
    lines = [f"{study_name} · {len(sweep.table)} runs in this sweep, {n_rej} rejected · "
             f"Source: {source_label}"]
    if sweep.name == SWEEP_BINS:
        lines.append(bin_mapping_text(sweep) + " (from mi_bin_widths in the run checkpoints)")
    return "\n".join(lines)


def draw_sweep(sweep: Sweep, output_dir: Path, *, study_name: str, source_label: str) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    sweep_dir = output_dir / sweep.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    footer = _footer(sweep, study_name, source_label)
    written: List[Path] = []
    drawn = []
    for k, (col, title, sub, better, f) in enumerate(METRICS, start=1):
        if col not in sweep.table or sweep.table[col].isna().all():
            continue
        drawn.append((col, title, better, f))
        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.25)
        draw_metric(ax, sweep, col, better, f)
        verdict = f"{better} is better"
        if col in DIAGNOSTICS:
            verdict = f"diagnostic, {verdict}"
        fig.text(0.11, 0.945, f"{title} vs {'γ' if sweep.name == SWEEP_GAMMA else 'effective bins'}",
                 fontsize=14, fontweight="bold", color=INK)
        fig.text(0.11, 0.9, f"{sub} · {verdict} · {sweep.fixed_text}", fontsize=9, color=INK2)
        ax.set_xlabel(sweep.x_label, color=INK2)
        ax.set_ylabel(f"{title}{_unit(f)}", color=INK2)
        ax.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
                  frameon=False, fontsize=8.5)
        fig.text(0.11, 0.025, footer, fontsize=7.2, color=INK2)
        path = sweep_dir / f"{k:02d}_{col}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        written.append(path)

    # Overview: every metric as a small multiple.
    ncols = 4
    nrows = int(np.ceil((len(drawn) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.3 * nrows + 1.4), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.985, top=1 - 1.05 / (3.3 * nrows + 1.4), bottom=0.07,
                        hspace=0.55, wspace=0.28)
    flat = axes.ravel()
    for ax, (col, title, better, f) in zip(flat, drawn):
        draw_metric(ax, sweep, col, better, f, compact=True)
        ax.set_title(f"{title}{_unit(f)}\n{better} is better", fontsize=9, color=INK, loc="left")
    for ax in flat[len(drawn):]:
        ax.axis("off")
    flat[len(drawn)].legend(handles=_legend_handles(compact=True), loc="center", frameon=False, fontsize=9)
    # x label on the lowest drawn panel of every column.
    for r in range(nrows):
        for c in range(ncols):
            below = axes[r + 1][c] if r + 1 < nrows else None
            if axes[r][c].axison and (below is None or not below.axison):
                axes[r][c].set_xlabel(sweep.x_label, fontsize=8.5, color=INK2)
    what = "γ" if sweep.name == SWEEP_GAMMA else "the effective number of bins"
    fig.text(0.05, 1 - 0.45 / (3.3 * nrows + 1.4), f"How {what} changes every metric · {sweep.fixed_text}",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.05, 0.012, footer.replace("\n", " · "), fontsize=8, color=INK2)
    path = output_dir / f"00_overview_{sweep.name}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.insert(0, path)

    columns = ["configuration_id", GAMMA, BINS] + ([EFFECTIVE_BINS] if EFFECTIVE_BINS in sweep.table else [])
    columns += ["feasible", "is_pareto_front"] + [c for c, *_ in drawn]
    kept = [c for c in columns if c in sweep.table]
    sweep.table[kept].to_csv(output_dir / f"sweep_{sweep.name}.csv", index=False)
    return written


def write_mi_changes(candidates: Path, output_dir: Path, *, study_map: Optional[Path] = None,
                     checkpoints_root: Optional[Path] = None, bins_for_gamma: int = 50,
                     gamma_for_bins: float = 0.1, study_name: Optional[str] = None) -> List[Path]:
    """Draw both sweeps. The bin sweep needs the study map and the checkpoints."""
    candidates = Path(candidates)
    table = load_candidates(candidates)
    if table.empty:
        raise ParetoMiChangeError(f"{candidates} has no rows")
    if "architecture_id" in table and table["architecture_id"].nunique() > 1:
        raise ParetoMiChangeError("Several architectures in one table; pass one architecture's rows.")
    effective = None
    if study_map is not None and checkpoints_root is not None:
        wanted = table.loc[_close(table[GAMMA], gamma_for_bins) | _close(table[GAMMA], 0)
                           | _close(table[BINS], bins_for_gamma), "configuration_id"]
        effective = effective_bins_by_configuration(Path(study_map), Path(checkpoints_root), list(wanted))
    sweeps = build_sweeps(table, bins_for_gamma=bins_for_gamma, gamma_for_bins=gamma_for_bins,
                          effective_bins=effective)
    name = study_name or candidates.resolve().parent.parent.name
    source = f"{candidates.parent.name}/{candidates.name}"
    output_dir = Path(output_dir)
    written: List[Path] = []
    for sweep in sweeps:
        written += draw_sweep(sweep, output_dir, study_name=name, source_label=source)
    return written


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot every Pareto metric against γ and against the "
                                                 "effective number of MI bins.")
    parser.add_argument("--candidates", type=Path, required=True, help="Phase 3 pareto_candidates.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the PNGs and CSVs.")
    parser.add_argument("--study-map", type=Path, help="study_map.yaml, to find each run's checkpoints.")
    parser.add_argument("--checkpoints-root", type=Path,
                        help="Local checkpoints/ dir; with --study-map enables the effective-bin sweep.")
    parser.add_argument("--bins-for-gamma", type=int, default=50, help="Nominal bins of the γ sweep.")
    parser.add_argument("--gamma-for-bins", type=float, default=0.1, help="γ of the bin sweep.")
    parser.add_argument("--study-name", help="Name in the footer (default: the study root's directory name).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    for path in write_mi_changes(args.candidates, args.output_dir, study_map=args.study_map,
                                 checkpoints_root=args.checkpoints_root,
                                 bins_for_gamma=args.bins_for_gamma,
                                 gamma_for_bins=args.gamma_for_bins, study_name=args.study_name):
        print(path)


if __name__ == "__main__":
    main()
