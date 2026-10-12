"""Compare a run's test outputs with those of its γ = 0 / 50-bin run.

The counterpart of the correlation-matrix comparison
(``src/evaluation/pareto/correlation_gamma0.py``) for everything else the old
test evaluation produced. For each γ ≠ 0 run it writes

    <run>/plots/<split>/<ckpt>/comparison/          (default plots/test/loss_total/)
        reference.json                   which γ = 0 run, matched on what
        summary.csv / summary.json       every quantity: run, γ = 0, difference,
                                         change in %, direction, better
        summary.png                      the scalar quantities, run vs γ = 0
        efficiency/efficiency_per_signal.{csv,png}
        ascore_operational/mean_ascore_per_dataset.{csv,png}
        reco/<dataset>/<object>_<feature>.{csv,png}
        reco/reco_summary.csv

from the outputs of the evaluation callbacks (scripts/physics/runae_test.sh):

=====================  ===================================================  =========
quantity               source                                               better
=====================  ===================================================  =========
efficiency per signal  <ckpt>/eff/eff_summary.json                          higher
median/min/mean/CVaR25 <ckpt>/eff/eff_summary.json                          higher
mean anomaly score     ascore_operational_summary/summary.pkl (per dataset) n/a
threshold drift        thres_drift_summary/summary.pkl (per target rate)    lower
Wasserstein W1         wasserstein_summary/summary.pkl                      lower
reconstruction         <ckpt>/reco/<dataset>/data/*.csv (histograms)        see below
=====================  ===================================================  =========

``change_percent = 100 * (run - γ0) / γ0``; ``better`` follows the direction
(``None`` where there is none, e.g. the mean anomaly score, whose scale differs
between models). Threshold drift is ``|log((p̂ + ε) / (FPR + ε))|`` (0 = no drift)
and W1 is between the score distributions of zero bias and the simulated,
anomaly-free SingleNeutrino gun, so lower is better for both.

Reconstruction: the reco callback writes each histogram with the same,
input-defined bins for every model, so the run's and the γ = 0 run's
reconstructed distributions are drawn in one plot (input as reference) with
their difference per bin below. ``reco_summary.csv`` gives per feature the total
variation distance (TVD, half the summed |difference| of the normalised
histograms, 0 = identical) of each reconstruction to the input (lower is better)
and between the two reconstructions.

The reference is the γ = 0 / 50-bin run of the same experiment with the same seed,
architecture and epochs (``correlation_gamma0.find_baseline_runs``),
and it must have been evaluated on the same split. γ = 0 runs are skipped.
Quantities missing on either side are left out and listed in ``summary.json``.

Usage::

    python3 -m src.evaluation.pareto.run_vs_gamma0 --experiment-dir checkpoints/Pareto-Front-261002 \\
        [--run-name <run> ...] [--split test] [--ckpt loss_total]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.pareto.correlation_gamma0 import (
    NO_BASELINE,
    RunInfo,
    find_baseline_runs,
    read_run_info,
)

COMPARISON_DIR = "comparison"
RECO_DATA_DIR = "data"  # src/evaluation/callbacks/reco.py: DATA_DIR

# Reference palette (dataviz skill): categorical slots 1/2 for the two models,
# status good/critical for better/worse (always with a text label), inks, surface.
RUN_COLOR, GAMMA0_COLOR = "#2a78d6", "#eb6834"
GOOD, CRITICAL, NEUTRAL = "#0ca30c", "#d03b3b", "#8a8984"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
INPUT_FILL = "#d7d6d0"


# ------------------------------------------------------------------- quantities
@dataclass
class Quantity:
    group: str
    name: str
    run: float
    gamma0: float
    direction: Optional[str]  # "higher", "lower" or None

    @property
    def difference(self) -> float:
        return self.run - self.gamma0

    @property
    def change_percent(self) -> Optional[float]:
        if self.gamma0 == 0 or not math.isfinite(self.gamma0):
            return None
        return 100.0 * (self.run - self.gamma0) / self.gamma0

    @property
    def better(self) -> Optional[bool]:
        if self.direction is None or self.run == self.gamma0:
            return None
        return (self.run > self.gamma0) == (self.direction == "higher")

    def row(self) -> dict:
        return {"group": self.group, "quantity": self.name, "run": self.run,
                "gamma0": self.gamma0, "difference": self.difference,
                "change_percent": self.change_percent, "direction": self.direction,
                "better": self.better}


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pair(group: str, name: str, run: Any, gamma0: Any, direction: Optional[str]) -> Optional[Quantity]:
    run, gamma0 = _finite(run), _finite(gamma0)
    if run is None or gamma0 is None:
        return None
    return Quantity(group, name, run, gamma0, direction)


# ---------------------------------------------------------------------- loading
@dataclass
class Outputs:
    """The evaluation outputs of one run on one split/checkpoint."""

    efficiency: Optional[dict] = None             # eff_summary.json
    ascore: Optional[dict] = None                 # dataset -> mean score
    thres_drift: Optional[dict] = None            # target rate -> drift
    wasserstein: Optional[float] = None
    reco: dict = field(default_factory=dict)      # dataset -> feature -> DataFrame

    @property
    def empty(self) -> bool:
        return (self.efficiency is None and self.ascore is None and self.thres_drift is None
                and self.wasserstein is None and not self.reco)


def _read_json(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_pickle(path: Path) -> Optional[dict]:
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None
    return payload if isinstance(payload, dict) else None


def load_outputs(run_dir: Path, split: str = "test", ckpt: str = "loss_total") -> Outputs:
    split_dir = Path(run_dir) / "plots" / split
    ckpt_dir = split_dir / ckpt
    outputs = Outputs(efficiency=_read_json(ckpt_dir / "eff" / "eff_summary.json"))

    ascore = _read_pickle(split_dir / "ascore_operational_summary" / "summary.pkl")
    if ascore and isinstance(ascore.get(ckpt), dict):
        outputs.ascore = ascore[ckpt]
    drift = _read_pickle(split_dir / "thres_drift_summary" / "summary.pkl")
    if drift:
        rates = {rate: by_ckpt.get(ckpt) for rate, by_ckpt in drift.items() if isinstance(by_ckpt, dict)}
        outputs.thres_drift = {rate: value for rate, value in rates.items() if value is not None} or None
    wasserstein = _read_pickle(split_dir / "wasserstein_summary" / "summary.pkl")
    if wasserstein and ckpt in wasserstein:
        outputs.wasserstein = wasserstein[ckpt]

    for csv in sorted((ckpt_dir / "reco").glob(f"*/{RECO_DATA_DIR}/*.csv")):
        outputs.reco.setdefault(csv.parent.parent.name, {})[csv.stem] = pd.read_csv(csv)
    return outputs


# ------------------------------------------------------------------- comparing
EFFICIENCY_SUMMARIES = ("median_efficiency", "min_efficiency", "mean_efficiency", "cvar25_efficiency")


def scalar_quantities(run: Outputs, ref: Outputs) -> list[Quantity]:
    quantities = []
    if run.efficiency and ref.efficiency:
        for name in EFFICIENCY_SUMMARIES:
            quantities.append(_pair("efficiency", name, run.efficiency.get(name),
                                    ref.efficiency.get(name), "higher"))
    if run.thres_drift and ref.thres_drift:
        for rate in sorted(set(run.thres_drift) & set(ref.thres_drift), key=str):
            quantities.append(_pair("thres_drift", f"drift_{rate}", run.thres_drift[rate],
                                    ref.thres_drift[rate], "lower"))
    if run.wasserstein is not None and ref.wasserstein is not None:
        quantities.append(_pair("wasserstein", "W1(normal, SingleNeutrino_E-10-gun)",
                                run.wasserstein, ref.wasserstein, "lower"))
    return [q for q in quantities if q is not None]


def per_key_quantities(group: str, run: Optional[dict], ref: Optional[dict],
                       direction: Optional[str]) -> list[Quantity]:
    if not run or not ref:
        return []
    keys = [key for key in run if key in ref]
    return [q for q in (_pair(group, str(key), run[key], ref[key], direction) for key in keys)
            if q is not None]


def tvd(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Total variation distance of two histograms (normalised to unit sum)."""
    sa, sb = float(np.sum(a)), float(np.sum(b))
    if sa <= 0 or sb <= 0:
        return None
    return 0.5 * float(np.abs(a / sa - b / sb).sum())


def reco_comparison(run_hist: pd.DataFrame, ref_hist: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Bin-by-bin table of both reconstructions, or None if the bins differ."""
    same_bins = (len(run_hist) == len(ref_hist)
                 and np.allclose(run_hist["bin_low"], ref_hist["bin_low"], equal_nan=True)
                 and np.allclose(run_hist["bin_high"], ref_hist["bin_high"], equal_nan=True))
    if not same_bins:
        return None
    total = max(float(run_hist["reco"].sum()), 1.0)
    total_ref = max(float(ref_hist["reco"].sum()), 1.0)
    total_input = max(float(run_hist["input"].sum()), 1.0)
    return pd.DataFrame({
        "bin_low": run_hist["bin_low"], "bin_high": run_hist["bin_high"],
        "input": run_hist["input"], "reco_run": run_hist["reco"], "reco_gamma0": ref_hist["reco"],
        "input_fraction": run_hist["input"] / total_input,
        "reco_run_fraction": run_hist["reco"] / total,
        "reco_gamma0_fraction": ref_hist["reco"] / total_ref,
        "difference_fraction": run_hist["reco"] / total - ref_hist["reco"] / total_ref,
    })


# --------------------------------------------------------------------- plotting
def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "grid.color": MUTED, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })
    return plt


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    magnitude = abs(value)
    return f"{value:.3g}" if 1e-3 <= magnitude < 1e4 else f"{value:.2e}"


def _verdict(q: Quantity) -> str:
    return {True: "better", False: "worse", None: ""}[q.better]


def _change_color(q: Quantity) -> str:
    return {True: GOOD, False: CRITICAL, None: NEUTRAL}[q.better]


def _legend(fig, *, with_status: bool, y: float) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(color=GAMMA0_COLOR, label="γ = 0 (50 bins)"), Patch(color=RUN_COLOR, label="this run")]
    if with_status:
        handles += [Patch(color=GOOD, label="change: better"), Patch(color=CRITICAL, label="change: worse")]
    else:
        handles += [Patch(color=NEUTRAL, label="change (no better/worse)")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, y))


def plot_paired_bars(quantities: Sequence[Quantity], path: Path, *, title: str, value_label: str,
                     log_values: bool = False) -> None:
    """Run vs γ = 0 per key (left) and the change in % (right): one axis each."""
    plt = _style()
    n = len(quantities)
    height = 0.36 * n + 2.2
    fig = plt.figure(figsize=(11, height))
    grid = fig.add_gridspec(1, 2, width_ratios=[3, 2], top=1 - 1.05 / height,
                            bottom=0.7 / height, wspace=0.06)
    ax_values = fig.add_subplot(grid[0])
    ax_change = fig.add_subplot(grid[1], sharey=ax_values)

    y = np.arange(n)
    bar = 0.38
    gamma0 = [q.gamma0 for q in quantities]
    runs = [q.run for q in quantities]
    ax_values.barh(y - bar / 2, gamma0, height=bar - 0.04, color=GAMMA0_COLOR)
    ax_values.barh(y + bar / 2, runs, height=bar - 0.04, color=RUN_COLOR)
    if log_values:
        positive = [v for v in gamma0 + runs if v > 0]
        if positive and len(positive) == len(gamma0 + runs):
            ax_values.set_xscale("log")
        elif positive:  # zeros (e.g. an efficiency of 0) stay visible as empty bars
            ax_values.set_xscale("symlog", linthresh=min(positive))
    ax_values.set_yticks(y, [q.name for q in quantities])
    ax_values.invert_yaxis()
    ax_values.set_xlabel(value_label)
    ax_values.grid(axis="x")

    changes = [q.change_percent if q.change_percent is not None else 0.0 for q in quantities]
    ax_change.barh(y, changes, height=0.6, color=[_change_color(q) for q in quantities])
    ax_change.axvline(0, color=INK2, lw=0.8)
    ax_change.set_xlabel("change vs γ = 0 [%]")
    ax_change.grid(axis="x")
    ax_change.tick_params(axis="y", left=False, labelleft=False)
    span = max([abs(c) for c in changes] + [1.0])
    ax_change.set_xlim(-1.45 * span, 1.45 * span)
    for yi, q, c in zip(y, quantities, changes):
        label = "n/a" if q.change_percent is None else f"{c:+.1f} %"
        ax_change.annotate(label, xy=(c, yi), xytext=(4 if c >= 0 else -4, 0),
                           textcoords="offset points", ha="left" if c >= 0 else "right",
                           va="center", fontsize=8, color=INK2)
    fig.suptitle(title, y=1 - 0.08 / height, va="top", fontsize=12, fontweight="bold")
    _legend(fig, with_status=any(q.direction for q in quantities), y=1 - 0.62 / height)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scalars(quantities: Sequence[Quantity], path: Path, *, title: str) -> None:
    """One small panel per scalar quantity (scales differ): γ = 0 and run bars."""
    plt = _style()
    n = len(quantities)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.3 * rows + 0.8), squeeze=False)
    for ax, q in zip(axes.flat, quantities):
        ax.barh([0, 1], [q.gamma0, q.run], height=0.62, color=[GAMMA0_COLOR, RUN_COLOR])
        ax.set_yticks([0, 1], ["γ = 0", "run"])
        ax.invert_yaxis()
        largest = max(abs(q.gamma0), abs(q.run))
        if largest == 0:
            ax.set_xlim(0, 1)
            ax.set_xticks([])
        else:
            ax.set_xlim(min(0.0, 1.3 * min(q.gamma0, q.run)), 1.3 * max(q.gamma0, q.run, 0.0) or largest)
        for yi, value in ((0, q.gamma0), (1, q.run)):
            ax.annotate(_fmt(value), xy=(value, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
        change = "n/a" if q.change_percent is None else f"{q.change_percent:+.1f} %"
        verdict = _verdict(q)
        mark = {"better": "  ▲ better", "worse": "  ▼ worse"}.get(verdict, "")
        ax.set_title(f"{q.name}\nΔ = {_fmt(q.difference)} ({change}){mark}", fontsize=9, color=INK)
        ax.grid(axis="x")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _is_et(feature: str) -> bool:
    # As src/plot/overlaid_hist.check_feature_is_Et.
    return ("Et" in feature or "ETTEM" in feature) and "Eta" not in feature


def plot_reco(table: pd.DataFrame, path: Path, *, dataset: str, feature: str) -> None:
    """Input (reference fill) and both reconstructions; their difference below."""
    plt = _style()
    inner = table.iloc[1:-1]  # under/overflow rows are reported, not drawn
    edges = np.concatenate([inner["bin_low"].to_numpy(), inner["bin_high"].to_numpy()[-1:]])
    fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06})
    ax.stairs(inner["input_fraction"], edges, fill=True, color=INPUT_FILL, label="input")
    ax.stairs(inner["reco_gamma0_fraction"], edges, color=GAMMA0_COLOR, lw=2, label="reco, γ = 0 (50 bins)")
    ax.stairs(inner["reco_run_fraction"], edges, color=RUN_COLOR, lw=2, label="reco, this run")
    if _is_et(feature):
        ax.set_yscale("log")
    ax.set_ylabel("fraction of entries")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y")

    difference = inner["difference_fraction"].to_numpy()
    widths = np.diff(edges)
    ax_diff.bar(edges[:-1], difference, width=widths, align="edge", color=RUN_COLOR, alpha=0.85,
                edgecolor=SURFACE, linewidth=0.5)
    ax_diff.axhline(0, color=INK2, lw=0.8)
    ax_diff.set_ylabel("run − γ = 0")
    ax_diff.set_xlabel(feature)
    ax_diff.grid(axis="y")

    tvd_runs = tvd(table["reco_run"].to_numpy(), table["reco_gamma0"].to_numpy())
    flow = table.iloc[[0, -1]][["reco_run", "reco_gamma0"]].to_numpy()
    note = f"TVD(run, γ = 0) = {_fmt(tvd_runs)}"
    if flow.sum() > 0:
        note += (f"   outside the bins: run {int(flow[:, 0].sum())}, "
                 f"γ = 0 {int(flow[:, 1].sum())} entries")
    ax.set_title(f"{dataset}: {feature}\n{note}", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reco_overlay(run_hist: pd.DataFrame, ref_hist: pd.DataFrame, path: Path, *,
                      dataset: str, feature: str) -> None:
    """Fallback when the bins differ: both reconstructions, no difference panel."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hist, color, label in ((ref_hist, GAMMA0_COLOR, "reco, γ = 0 (50 bins)"),
                               (run_hist, RUN_COLOR, "reco, this run")):
        inner = hist.iloc[1:-1]
        edges = np.concatenate([inner["bin_low"].to_numpy(), inner["bin_high"].to_numpy()[-1:]])
        ax.stairs(inner["reco"] / max(float(hist["reco"].sum()), 1.0), edges, color=color, lw=2, label=label)
    if _is_et(feature):
        ax.set_yscale("log")
    ax.set_title(f"{dataset}: {feature}\nbins differ between the runs: no difference shown", fontsize=10)
    ax.set_xlabel(feature)
    ax.set_ylabel("fraction of entries")
    ax.legend(frameon=False, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- run
def _write_table(quantities: Sequence[Quantity], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([q.row() for q in quantities]).to_csv(path, index=False)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare_run(run: RunInfo, reference: RunInfo, *, split: str = "test", ckpt: str = "loss_total",
                note: str = "") -> dict:
    """Write ``<run>/plots/<split>/<ckpt>/comparison/`` and return its summary."""
    out = run.run_dir / "plots" / split / ckpt / COMPARISON_DIR
    out.mkdir(parents=True, exist_ok=True)
    run_out, ref_out = load_outputs(run.run_dir, split, ckpt), load_outputs(reference.run_dir, split, ckpt)
    missing = []

    scalars = scalar_quantities(run_out, ref_out)
    efficiency = per_key_quantities(
        "efficiency_per_signal",
        (run_out.efficiency or {}).get("signal_efficiencies"),
        (ref_out.efficiency or {}).get("signal_efficiencies"), "higher")
    ascore = per_key_quantities("mean_ascore", run_out.ascore, ref_out.ascore, None)
    for label, present in (("efficiency", efficiency), ("ascore_operational", ascore),
                           ("thres_drift", any(q.group == "thres_drift" for q in scalars)),
                           ("wasserstein", any(q.group == "wasserstein" for q in scalars))):
        if not present:
            missing.append(label)

    subtitle = f"{run.name}  vs  {reference.name}  ({split}, {ckpt})"
    if scalars:
        plot_scalars(scalars, out / "summary.png", title=f"Run vs γ = 0: summaries\n{subtitle}")
    if efficiency:
        _write_table(efficiency, out / "efficiency" / "efficiency_per_signal.csv")
        plot_paired_bars(efficiency, out / "efficiency" / "efficiency_per_signal.png",
                         title=f"Signal efficiency at the operating point\n{subtitle}",
                         value_label="efficiency (log scale)", log_values=True)
    if ascore:
        _write_table(ascore, out / "ascore_operational" / "mean_ascore_per_dataset.csv")
        plot_paired_bars(ascore, out / "ascore_operational" / "mean_ascore_per_dataset.png",
                         title=f"Mean anomaly score (ascore/operational) per dataset\n{subtitle}",
                         value_label="mean anomaly score")

    reco_rows = []
    for dataset in sorted(set(run_out.reco) & set(ref_out.reco)):
        for feature in sorted(set(run_out.reco[dataset]) & set(ref_out.reco[dataset])):
            run_hist, ref_hist = run_out.reco[dataset][feature], ref_out.reco[dataset][feature]
            folder = out / "reco" / dataset
            table = reco_comparison(run_hist, ref_hist)
            if table is None:
                plot_reco_overlay(run_hist, ref_hist, folder / f"{feature}.png", dataset=dataset, feature=feature)
                reco_rows.append({"dataset": dataset, "feature": feature, "same_bins": False})
                continue
            folder.mkdir(parents=True, exist_ok=True)
            table.to_csv(folder / f"{feature}.csv", index=False)
            plot_reco(table, folder / f"{feature}.png", dataset=dataset, feature=feature)
            reco_rows.append({
                "dataset": dataset, "feature": feature, "same_bins": True,
                "tvd_reco_run_vs_input": tvd(table["reco_run"].to_numpy(), table["input"].to_numpy()),
                "tvd_reco_gamma0_vs_input": tvd(table["reco_gamma0"].to_numpy(), table["input"].to_numpy()),
                "tvd_reco_run_vs_reco_gamma0": tvd(table["reco_run"].to_numpy(), table["reco_gamma0"].to_numpy()),
            })
    if reco_rows:
        pd.DataFrame(reco_rows).to_csv(out / "reco" / "reco_summary.csv", index=False)
    else:
        missing.append("reco (no histogram data: needs the reco callback that writes reco/<dataset>/data/)")

    quantities = scalars + efficiency + ascore
    _write_table(quantities, out / "summary.csv")
    reference_info = {
        "reference_run": reference.name,
        "matched_on": {"gamma": 0.0, "mi_sensitive_num_bins": reference.bins, "seed": reference.seed,
                       "encoder_nodes": list(reference.architecture or ()), "max_epochs": reference.epochs},
        "split": split, "checkpoint": ckpt, "note": note,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "reference.json").write_text(json.dumps(reference_info, indent=2) + "\n")
    summary = {
        "run": run.name, **reference_info,
        "change_percent_definition": "100 * (run - gamma0) / gamma0",
        "quantities": [{k: _json_safe(v) for k, v in q.row().items()} for q in quantities],
        "reco": [{k: _json_safe(v) for k, v in row.items()} for row in reco_rows],
        "missing": missing,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@dataclass
class Report:
    written: list = field(default_factory=list)   # (run name, reference name, missing)
    skipped: list = field(default_factory=list)   # (run name, reason)


def process_experiment(experiment_dir: Path, *, run_names: Optional[Iterable[str]] = None,
                       split: str = "test", ckpt: str = "loss_total") -> Report:
    experiment_dir = Path(experiment_dir)
    all_runs = [read_run_info(d) for d in sorted(experiment_dir.iterdir()) if (d / "plots").is_dir()]
    by_name = {run.name: run for run in all_runs}
    report = Report()
    names = list(run_names) if run_names is not None else [r.name for r in all_runs]
    for name in names:
        run = by_name.get(name)
        if run is None:
            report.skipped.append((name, f"no such run in {experiment_dir.name}"))
            continue
        if not run.identified:
            report.skipped.append((name, "run not identified (seed/architecture/epochs/γ missing)"))
            continue
        if run.gamma == 0.0:
            report.skipped.append((name, "γ = 0 run (the reference)"))
            continue
        if load_outputs(run.run_dir, split, ckpt).empty:
            report.skipped.append((name, f"no {split} outputs"))
            continue
        candidates = [ref for ref in find_baseline_runs(run, all_runs)
                      if not load_outputs(ref.run_dir, split, ckpt).empty]
        if not candidates:
            reason = (NO_BASELINE if not find_baseline_runs(run, all_runs)
                      else f"its γ = 0 / 50-bin run has no {split} outputs (run runae_test.sh for it)")
            report.skipped.append((name, reason))
            continue
        note = f"several γ = 0 / 50-bin runs; used {candidates[0].name}" if len(candidates) > 1 else ""
        summary = compare_run(run, candidates[0], split=split, ckpt=ckpt, note=note)
        report.written.append((name, candidates[0].name, summary["missing"]))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare runs' test outputs with their γ = 0 / 50-bin run.")
    parser.add_argument("--experiment-dir", type=Path, required=True, help="checkpoints/<experiment>")
    parser.add_argument("--run-name", action="append", dest="run_names",
                        help="Only this run. Repeatable; default: every run with outputs.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ckpt", default="loss_total")
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    report = process_experiment(args.experiment_dir, run_names=args.run_names,
                                split=args.split, ckpt=args.ckpt)
    print(f"{args.experiment_dir.name}: {len(report.written)} run(s) compared with γ = 0 "
          f"({args.split}, {args.ckpt})")
    for name, reference, missing in report.written:
        extra = f"; missing: {', '.join(missing)}" if missing else ""
        print(f"  {name} vs {reference}{extra}")
    for name, reason in report.skipped:
        print(f"  skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
