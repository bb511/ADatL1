"""Phase 4b of the Pareto study: gamma x bins matrices, one figure per metric.

Reads only the Phase 3 table ``pareto_candidates.csv`` (plus, optionally,
``pareto_selection.json`` and the study map), so what is drawn is exactly what
Phase 3 selected. Every cell is one configuration: rejected configurations are
white, hatched and framed red, the Pareto front is outlined in black with its value
in bold, and grid points that were never trained stay empty.

Colour scale: it runs from the lowest to the highest value of the feasible
cells; rejected cells take no colour (they would otherwise stretch the scale, e.g.
collapsed runs at leakage 0). The worst value of a metric is always dark blue and
the best value always pale blue, so the direction flips with ``better`` (lower- or
higher-is-better). The scale is a framed colour bar the full height of the matrix
right next to it, as in the correlation matrices (``src/plot/matrix.py``); its
tick scale always shows the minimum and maximum value.

Generalised from the one-off ``make_matrices.py`` written for
Pareto-Front-260928:

- the study name, run count, seeds and architectures come from the table;
- the collapse rule shown in the footer is read from the study's resolved
  config (``pareto_study.collapse_constraint.rule``) when the study map points
  at one, so a per-bin paired baseline (Pareto-Front-261002) is described as
  such;
- the selected configuration comes from ``pareto_selection.json``, not a
  hard-coded label;
- several architectures are drawn into one sub-directory each, because the
  matrix axes are gamma and bins only.

Usage (run by scripts/physics/runcollect.sh after phase 4)::

    python3 scripts/plot_pareto_matrices.py \\
        --candidates  <STUDY_ROOT>/phase3/pareto_candidates.csv \\
        --selection   <STUDY_ROOT>/phase3/pareto_selection.json \\
        --study-map   <STUDY_ROOT>/study_map.yaml \\
        --output-dir  <STUDY_ROOT>/phase4/matrices
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# Reference palette (dataviz skill): sequential blue 100 -> 700 (pale -> dark), text
# ink, surface, status critical. Rejected cells carry a hatch texture and a legend
# entry as well, so status is never colour alone. BLUE[0] marks the best value of a
# metric and BLUE[-1] the worst.
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
CRITICAL = "#c62828"
REJECTED_FILL = "#ffffff"  # rejected cells: white, red hatch and red frame
STATUS_COLOURS = [REJECTED_FILL, "#9ec5f4", "#1c5cab"]  # rejected, dominated, front

GAMMA, BINS = "mi_gamma", "mi_sensitive_num_bins"

# (column, title, unit/meaning, better, number format). ``better`` ("lower" or
# "higher") decides which end of the colour scale is pale (best) and which is dark
# (worst). Diagnostics (not Pareto objectives) carry a direction too: more code
# entropy / more used codes means less collapse, less redundancy is better.
METRICS = [
    ("leakage_worst", "Leakage L", "worst of four probes, held-out R²", "lower", "{:.3f}"),
    ("residual_correlation", "Residual correlation E", "max(mean |Pearson|, mean |Spearman|)", "lower", "{:.3f}"),
    ("mean_spearman_correlation", "Mean Spearman correlation", "FET.Et vs reconstruction", "lower", "{:.3f}"),
    ("median_efficiency", "Median signal efficiency", "at 0.25 kHz, ×10⁻³ in cells", "higher", "x1e3"),
    ("mean_efficiency", "Mean signal efficiency", "at 0.25 kHz, ×10⁻² in cells", "higher", "x1e2"),
    ("cvar25_efficiency", "CVaR25 signal efficiency", "mean of the worst 25 % of signals, ×10⁻⁴ in cells", "higher", "x1e4"),
    ("median_auroc", "Median AUROC", "over signal samples", "higher", "{:.4f}"),
    ("min_auroc", "Minimum AUROC", "worst signal sample", "higher", "{:.3f}"),
    ("median_partial_auroc", "Median partial AUROC", "×10⁻³ in cells", "higher", "x1e3"),
    ("joint_code_entropy_bits", "Joint code entropy H(L)", "bits, 0–8; collapse rule uses this", "higher", "{:.2f}"),
    ("summed_marginal_bit_entropy_bits", "Summed marginal bit entropy Σ h(θⱼ)", "bits, 0–8", "higher", "{:.2f}"),
    ("redundancy_bits", "Redundancy Σ h(θⱼ) − H(L)", "bits, derived", "lower", "{:.2f}"),
    ("effective_code_count", "Effective code count 2^H", "equally used codes", "higher", "{:.2f}"),
    ("observed_code_count", "Observed code count", "distinct codes on normal validation", "higher", "{:.0f}"),
]
DIAGNOSTICS = {
    "joint_code_entropy_bits", "summed_marginal_bit_entropy_bits", "redundancy_bits",
    "effective_code_count", "observed_code_count",
}
# Unit shown in the colour-bar label of the metrics printed ×10^k in the cells.
SCALE_UNIT = {"x1e3": "×10⁻³", "x1e2": "×10⁻²", "x1e4": "×10⁻⁴"}
SCALE = {"x1e3": 1e3, "x1e2": 1e2, "x1e4": 1e4}
STATUS_FILENAME = "00_selection_status.png"


class ParetoMatrixError(ValueError):
    """The candidates table cannot be drawn as gamma x bins matrices."""


@dataclass
class StudyContext:
    study_name: str
    rule_text: str
    selected_configuration_id: Optional[str]
    source_label: str


# --------------------------------------------------------------------------- data
def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})


def load_candidates(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [c for c in (GAMMA, BINS, "feasible", "is_pareto_front") if c not in table]
    if missing:
        raise ParetoMatrixError(f"{path} lacks required columns: {missing}")
    table["feasible"] = _as_bool(table["feasible"])
    table["is_pareto_front"] = _as_bool(table["is_pareto_front"])
    if {"summed_marginal_bit_entropy_bits", "joint_code_entropy_bits"} <= set(table):
        table["redundancy_bits"] = (
            table.summed_marginal_bit_entropy_bits - table.joint_code_entropy_bits
        )
        table["joint_code_entropy_bits"] = table.joint_code_entropy_bits.clip(lower=0)
    return table


def _collapse_duplicate_cells(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per (gamma, bins). Several seeds of one cell are averaged."""
    counts = table.groupby([GAMMA, BINS]).size()
    n_max = int(counts.max()) if len(counts) else 1
    if n_max == 1:
        return table, 1
    numeric = [c for c in table.select_dtypes("number").columns if c not in (GAMMA, BINS)]
    agg: Dict[str, Any] = {c: "mean" for c in numeric}
    agg.update({"feasible": "all", "is_pareto_front": "any"})
    if "pareto_rank" in table:
        agg["pareto_rank"] = "min"
    if "configuration_id" in table:
        agg["configuration_id"] = "first"
    return table.groupby([GAMMA, BINS], as_index=False).agg(agg), n_max


# ------------------------------------------------------------------ study context
def _read_yaml(path: Path) -> Any:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolved_config_candidates(run: Mapping[str, Any], checkpoints_root: Optional[Path]) -> List[Path]:
    paths = []
    if run.get("manifest_path"):
        paths.append(Path(run["manifest_path"]))
    # The study map records the paths of wherever stage 4 ran (EOS on lxplus).
    # On another machine, look for the same run under the local checkpoints dir.
    run_dir = str(run.get("checkpoint_run_dir") or "")
    if checkpoints_root is not None and "/checkpoints/" in run_dir:
        tail = run_dir.split("/checkpoints/", 1)[1]
        paths.append(Path(checkpoints_root) / tail / "resolved_config.yaml")
    return paths


def collapse_rule_text(study_map: Optional[Path], checkpoints_root: Optional[Path]) -> Optional[str]:
    """Describe pareto_study.collapse_constraint.rule of the study, if it can be found."""
    if study_map is None or not Path(study_map).is_file():
        return None
    runs = (_read_yaml(study_map) or {}).get("runs") or []
    for run in runs:
        for path in _resolved_config_candidates(run, checkpoints_root):
            if not path.is_file():
                continue
            rule = (((_read_yaml(path) or {}).get("pareto_study") or {})
                    .get("collapse_constraint") or {}).get("rule") or {}
            if not rule:
                continue
            absolute = rule.get("minimum_joint_code_entropy_bits")
            fraction = rule.get("minimum_fraction_of_paired_gamma_zero_joint_entropy")
            reference = {
                "same_architecture_and_bins": "the γ=0 run with the same bins",
                "same_architecture": "the γ=0 baseline",
            }.get(str(rule.get("paired_reference")), "the paired γ=0 run")
            parts = []
            if absolute is not None:
                parts.append(f"H(L) < {float(absolute):g} bit")
            if fraction is not None:
                parts.append(f"< {float(fraction):g} × H(L) of {reference}")
            return " or ".join(parts) if parts else None
    return None


def _baseline_text(table: pd.DataFrame, column: str, fmt_spec: str) -> str:
    base = table[table[GAMMA] == 0]
    if base.empty or column not in base or base[column].isna().all():
        return "baseline γ=0: n/a"
    values = base[column].dropna().to_numpy(dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if base[BINS].nunique() == 1:
        return f"baseline γ=0: {fmt(lo, fmt_spec)}"
    if fmt(lo, fmt_spec) == fmt(hi, fmt_spec):  # equal at the precision shown
        return f"baseline γ=0: {fmt(lo, fmt_spec)} (same at every bin count)"
    return f"baseline γ=0: {fmt(lo, fmt_spec)}–{fmt(hi, fmt_spec)} across bin counts"


# ---------------------------------------------------------------------- drawing
def fmt(v: float, f: Optional[str]) -> str:
    if f is None or v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if f in SCALE:
        return f"{v * SCALE[f]:.2f}"
    return f.format(v)


def _ink_for(rgb: Sequence[float]) -> str:
    r, g, b = rgb[:3]
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def _grid(table: pd.DataFrame, gammas: List[float], bins: List[int], column: str):
    values = np.full((len(gammas), len(bins)), np.nan)
    rejected = np.zeros_like(values, dtype=bool)
    front = np.zeros_like(values, dtype=bool)
    for row in table.itertuples(index=False):
        i, j = gammas.index(getattr(row, GAMMA)), bins.index(getattr(row, BINS))
        values[i, j] = getattr(row, column)
        rejected[i, j] = not row.feasible
        front[i, j] = bool(row.is_pareto_front)
    return values, rejected, front


def _draw_cells(ax, values, rejected, front, cmap, norm, f, gammas, bins):
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    ny, nx = values.shape
    for i in range(ny):
        for j in range(nx):
            v = values[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=SURFACE, edgecolor=MUTED, lw=0.6))
                continue
            # Rejected cells are outside the colour scale: white, hatched and framed red.
            c = REJECTED_FILL if rejected[i, j] else cmap(norm(v))
            ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor=c, edgecolor="none"))
            if rejected[i, j]:
                ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor="none",
                                       hatch="////", edgecolor=CRITICAL, lw=0))
                ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, facecolor="none",
                                       edgecolor=CRITICAL, lw=1.6))
            if front[i, j]:
                ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, facecolor="none",
                                       edgecolor=INK, lw=2.0))
            txt = fmt(v, f)
            if txt:
                ax.text(j + 0.5, i + 0.52, txt, ha="center", va="center", fontsize=7.2,
                        color=_ink_for(to_rgba(c)), fontweight="bold" if front[i, j] else "normal",
                        bbox=dict(boxstyle="round,pad=0.12", fc=c, ec="none", alpha=0.85)
                        if rejected[i, j] else None)
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_xticks(np.arange(nx) + 0.5, [str(b) for b in bins])
    ax.set_yticks(np.arange(ny) + 0.5, [f"{g:g}" for g in gammas])
    ax.set_xlabel("FET.Et bins (mi_sensitive_num_bins)")
    ax.set_ylabel("MI weight γ (mi_gamma)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def metric_colormap(better: str):
    """Pale blue at the best end of a metric, dark blue at the worst end."""
    from matplotlib.colors import LinearSegmentedColormap

    if better not in {"lower", "higher"}:
        raise ParetoMatrixError(f"better must be 'lower' or 'higher', got {better!r}")
    # The colormap runs from the low to the high value.
    colours = BLUE if better == "lower" else BLUE[::-1]
    return LinearSegmentedColormap.from_list(f"blue_best_{better}", colours)


def colorbar_ticks(lo: float, hi: float, nbins: int = 6) -> List[float]:
    """Ticks from ``lo`` to ``hi``: both ends plus round values that do not crowd them."""
    from matplotlib.ticker import MaxNLocator

    if hi - lo < 1e-12:
        return [lo]
    margin = 0.07 * (hi - lo)
    inner = [float(t) for t in MaxNLocator(nbins=nbins).tick_values(lo, hi)
             if lo + margin < t < hi - margin]
    return [lo, *inner, hi]


def tick_labels(ticks: Sequence[float], f: Optional[str]) -> List[str]:
    """Labels in the cells' number format, with more decimals if two would coincide."""
    mult = SCALE.get(f, 1.0)
    if f in SCALE:
        decimals = 2
    elif f and f.startswith("{:.") and f.endswith("f}"):
        decimals = int(f[3:-2])
    else:
        decimals = 3
    while True:
        labels = [f"{t * mult:.{decimals}f}" for t in ticks]
        if len(set(labels)) == len(labels) or decimals >= 6:
            return labels
        decimals += 1


def draw_colorbar(fig, ax, norm, cmap, *, label: str, f: Optional[str],
                  value_range: tuple[float, float]):
    """Colour bar in the style of the correlation matrices, from min to max.

    Like ``src/plot/matrix.py``: appended to the right of the matrix with the same
    height (5 % wide, 0.15 in gap), framed in black, no extension arrows, no minor
    ticks. The tick scale starts at the lowest and ends at the highest value drawn
    (``value_range``), both labelled in the number format of the cells.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.outline.set_visible(True)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(1.6)
    cb.ax.minorticks_off()
    ticks = colorbar_ticks(*value_range)
    cb.locator = FixedLocator(ticks)
    cb.formatter = FixedFormatter(tick_labels(ticks, f))
    cb.update_ticks()
    cb.ax.tick_params(which="major", direction="out", length=4, width=1.0,
                      colors=INK, labelcolor=INK, labelsize=9.5)
    cb.set_label(label, fontsize=10, color=INK, labelpad=8)
    return cb


def _figure():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 11.2))
    fig.subplots_adjust(left=0.1, right=0.86, top=0.9, bottom=0.1)
    return fig, ax


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def _footer(table: pd.DataFrame, ctx: StudyContext, architecture: Optional[str], n_seeds_per_cell: int,
            runs: Optional[pd.DataFrame] = None) -> str:
    n_feas, n_rej = int(table.feasible.sum()), int((~table.feasible).sum())
    runs = table if runs is None else runs  # per-run rows, before seeds are averaged
    seeds = sorted({int(s) for s in runs["autoencoder_seed"].dropna()}) if "autoencoder_seed" in runs else []
    archs = [architecture] if architecture else (
        sorted(table["architecture_id"].dropna().unique()) if "architecture_id" in table else [])
    who = ", ".join(filter(None, [
        f"seed{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))}" if seeds else "",
        ", ".join(archs),
    ]))
    mean_note = f" · cells average {n_seeds_per_cell} seeds" if n_seeds_per_cell > 1 else ""
    rule = f"Rejected if {ctx.rule_text} · " if ctx.rule_text else ""
    return (f"{ctx.study_name} · {len(table)} configurations{', ' + who if who else ''} · "
            f"{n_feas} feasible, {n_rej} rejected{mean_note}\n"
            f"{rule}Axes are categorical, not to scale · Source: {ctx.source_label}")


def _selected_label(table: pd.DataFrame, ctx: StudyContext) -> str:
    row = None
    if ctx.selected_configuration_id and "configuration_id" in table:
        hit = table[table["configuration_id"] == ctx.selected_configuration_id]
        if len(hit):
            row = hit.iloc[0]
    if row is None and "pareto_rank" in table and table["pareto_rank"].notna().any():
        row = table.loc[table["pareto_rank"].idxmin()]
    if row is None:
        return ""
    return f" (#1 = selected: γ {row[GAMMA]:g}, {int(row[BINS])} bins)"


def draw_matrices(table: pd.DataFrame, output_dir: Path, ctx: StudyContext,
                  architecture: Optional[str] = None) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.patches import Patch, Rectangle

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = table
    table, n_seeds = _collapse_duplicate_cells(table)
    gammas = sorted(table[GAMMA].unique())
    bins = sorted(table[BINS].unique())
    foot = _footer(table, ctx, architecture, n_seeds, runs=runs)
    legend = [
        Patch(facecolor=REJECTED_FILL, edgecolor=CRITICAL, hatch="////", lw=1.6,
              label="Rejected (collapse rule, not on the colour scale)"),
        Patch(facecolor=SURFACE, edgecolor=INK, lw=2.0, label="Pareto front (bold value)"),
        Patch(facecolor=SURFACE, edgecolor=MUTED, lw=0.6, label="Not trained"),
    ]
    written: List[Path] = []

    for k, (col, title, sub, better, f) in enumerate(METRICS, start=1):
        if col not in table or table[col].isna().all():
            continue
        values, rejected, front = _grid(table, gammas, bins, col)
        # The colour scale runs from the lowest to the highest feasible value; rejected
        # cells are drawn white. Without feasible cells, all values set the range.
        feasible = values[~rejected & ~np.isnan(values)]
        pool = feasible if feasible.size else values[~np.isnan(values)]
        lo, hi = float(pool.min()), float(pool.max())
        # A constant metric sits mid-scale on a +-0.5 range; its value is the only tick.
        norm = Normalize(lo, hi) if hi - lo >= 1e-12 else Normalize(lo - 0.5, hi + 0.5)
        cmap = metric_colormap(better)
        fig, ax = _figure()
        _draw_cells(ax, values, rejected, front, cmap, norm, f, gammas, bins)
        verdict = f"{better} is better"
        if col in DIAGNOSTICS:
            verdict = f"diagnostic, {verdict}"
        fig.text(0.1, 0.955, title, fontsize=14, fontweight="bold", color=INK)
        fig.text(0.1, 0.932, f"{sub} · {verdict} · {_baseline_text(table, col, f)}", fontsize=9, color=INK2)
        fig.text(0.1, 0.913, "Colour scale runs from the lowest to the highest feasible value; "
                 "pale blue = best, dark blue = worst; rejected cells are white.",
                 fontsize=8, color=INK2)
        unit = f" [{SCALE_UNIT[f]}]" if f in SCALE_UNIT else ""
        draw_colorbar(fig, ax, norm, cmap, label=f"{title}{unit}", f=f, value_range=(lo, hi))
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=3,
                  frameon=False, fontsize=8.5, handlelength=1.6, handleheight=1.2)
        fig.text(0.1, 0.022, foot, fontsize=7.2, color=INK2)
        path = output_dir / f"{k:02d}_{col}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        written.append(path)

    # Overview: status + Pareto rank.
    status = np.full((len(gammas), len(bins)), np.nan)
    rank = np.full_like(status, np.nan)
    for row in table.itertuples(index=False):
        i, j = gammas.index(getattr(row, GAMMA)), bins.index(getattr(row, BINS))
        status[i, j] = 0 if not row.feasible else (2 if row.is_pareto_front else 1)
        r = getattr(row, "pareto_rank", np.nan)
        rank[i, j] = r if row.is_pareto_front else np.nan
    st_cmap = ListedColormap(STATUS_COLOURS)
    st_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    fig, ax = _figure()
    _draw_cells(ax, np.full_like(status, np.nan), np.zeros_like(status, dtype=bool),
                np.zeros_like(status, dtype=bool), st_cmap, st_norm, None, gammas, bins)
    for i in range(len(gammas)):
        for j in range(len(bins)):
            s = status[i, j]
            if np.isnan(s):
                continue
            ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor=st_cmap(st_norm(s)), edgecolor="none"))
            if s == 0:
                ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor="none", hatch="////", edgecolor=CRITICAL, lw=0))
                ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, facecolor="none", edgecolor=CRITICAL, lw=1.6))
            if s == 2 and not np.isnan(rank[i, j]):
                ax.text(j + 0.5, i + 0.52, f"#{int(rank[i, j])}", ha="center", va="center",
                        fontsize=8, color="#ffffff", fontweight="bold")
    n_front, n_dom, n_rej = int((status == 2).sum()), int((status == 1).sum()), int((status == 0).sum())
    fig.text(0.1, 0.955, "Selection status", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.1, 0.932, f"#n = rank on the Pareto front by ideal-point distance{_selected_label(table, ctx)} · "
             f"{n_front} front, {n_dom} dominated, {n_rej} rejected", fontsize=9, color=INK2)
    handles = [Patch(facecolor=STATUS_COLOURS[2], label="Pareto front"),
               Patch(facecolor=STATUS_COLOURS[1], label="Feasible, dominated"),
               Patch(facecolor=STATUS_COLOURS[0], edgecolor=CRITICAL, hatch="////", lw=1.6, label="Rejected (collapse rule)"),
               Patch(facecolor=SURFACE, edgecolor=MUTED, lw=0.6, label="Not trained")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=4, frameon=False, fontsize=8.5)
    fig.text(0.1, 0.022, foot, fontsize=7.2, color=INK2)
    path = output_dir / STATUS_FILENAME
    fig.savefig(path, dpi=170)
    plt.close(fig)
    written.insert(0, path)
    return written


def write_pareto_matrices(candidates: Path, output_dir: Path, *, selection: Optional[Path] = None,
                          study_map: Optional[Path] = None, checkpoints_root: Optional[Path] = None,
                          study_name: Optional[str] = None) -> List[Path]:
    candidates = Path(candidates)
    table = load_candidates(candidates)
    if table.empty:
        raise ParetoMatrixError(f"{candidates} has no rows")
    selected = None
    if selection is not None and Path(selection).is_file():
        selected = json.loads(Path(selection).read_text()).get("selected_configuration_id")
    # <study_root>/phase3/pareto_candidates.csv -> study name = <study_root>.name
    name = study_name or candidates.resolve().parent.parent.name
    ctx = StudyContext(
        study_name=name,
        rule_text=collapse_rule_text(study_map, checkpoints_root) or "",
        selected_configuration_id=selected,
        source_label=f"{candidates.parent.name}/{candidates.name}",
    )
    output_dir = Path(output_dir)
    archs = sorted(table["architecture_id"].dropna().unique()) if "architecture_id" in table else []
    if len(archs) <= 1:
        return draw_matrices(table, output_dir, ctx, archs[0] if archs else None)
    written: List[Path] = []
    for arch in archs:
        written += draw_matrices(table[table["architecture_id"] == arch], output_dir / str(arch), ctx, arch)
    return written


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw gamma x bins matrices from the Phase 3 candidates table.")
    parser.add_argument("--candidates", type=Path, required=True, help="Phase 3 pareto_candidates.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the PNGs.")
    parser.add_argument("--selection", type=Path, help="Phase 3 pareto_selection.json (selected configuration).")
    parser.add_argument("--study-map", type=Path, help="study_map.yaml, used to read the collapse rule.")
    parser.add_argument("--checkpoints-root", type=Path,
                        help="Local checkpoints/ dir, to find resolved configs when the map holds paths of another machine.")
    parser.add_argument("--study-name", help="Name in the footer (default: the study root's directory name).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    for path in write_pareto_matrices(args.candidates, args.output_dir, selection=args.selection,
                                      study_map=args.study_map, checkpoints_root=args.checkpoints_root,
                                      study_name=args.study_name):
        print(path)


if __name__ == "__main__":
    main()
