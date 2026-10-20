"""Compare a run's test outputs with those of its γ = 0 / 50-bin run.

The counterpart of the correlation-matrix comparison
(``src/evaluation/pareto/correlation_gamma0.py``) for everything else the old
test evaluation produced. For each γ ≠ 0 run it writes

    <run>/plots/<split>/<ckpt>/comparison/          (default plots/test/loss_total/)
        reference.json                   which γ = 0 run, matched on what
        summary.csv / summary.json       every quantity: run, γ = 0, difference,
                                         change in %, direction, better
        eff/<stem>.{jpg,csv}             efficiency per signal at the operating point
        ascore_operational_summary/ascore_operational.{jpg,csv}
                                         mean anomaly score per dataset
        eff_summary/<stem>.{jpg,csv}     25% CVaR, median and minimum efficiency
        thres_drift_summary/<stem>.{jpg,csv}   one per target rate
        wasserstein_summary/<stem>.{jpg,csv}
        reco/<dataset>/<object>_<feature>.{csv,png}
        reco/reco_summary.csv

Every bar chart mirrors one plot of ``plots/<split>/`` (same folder, same file
stem, same x label) and is drawn by ``src/plot/horizontal_bar.plot_comparison`` in
the style of the test plots: per key an orange bar for the γ = 0 run (legend
"γ = 0, bins = 50") above a blue bar for the run (legend "γ = <γ>, bins = <bins>",
read from the run name), values at the bar ends, and on the right a gray panel
"Change" with ``100 * (run - γ0) / γ0``. Whether a change is better or worse is
not coloured; ``summary.csv`` keeps the direction for reference. The mean
efficiency is only in ``summary.csv``, as it has no plot in ``plots/<split>/``.

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

MLflow: with ``--mlruns-root`` the comparison is also put into the run's MLflow
artifacts, under ``<split>/<ckpt>/comparison/`` next to the evaluator's plots:
per plot folder the images and a gallery page beside it (``<folder>.html``,
``<dataset>_reco.html`` for ``reco/<dataset>``, as in ``<split>/<ckpt>/``), plus
``summary.csv``, ``reference.json`` and ``reco/reco_summary.csv``. The files are
written straight into the file store, like the correlation galleries of
``correlation_gamma0``: the run's ``artifact_uri`` can be the sandbox of the job
that trained it (``/srv/outputs/...``), where MLflow's own logging would lose them.
``--mlflow-only`` rewrites just these artifacts from the existing comparison folders.

Usage::

    python3 -m src.evaluation.pareto.run_vs_gamma0 --experiment-dir checkpoints/Pareto-Front-261002 \\
        [--run-name <run> ...] [--split test] [--ckpt loss_total] \\
        [--mlruns-root logs/mlflow/mlruns [--mlflow-only]]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
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
    mlflow_runs,
    read_run_info,
)

COMPARISON_DIR = "comparison"
RECO_DATA_DIR = "data"  # src/evaluation/callbacks/reco.py: DATA_DIR

# Folders of plots/<split>/ the bar charts mirror (one comparison plot per plot there).
EFF_DIR = "eff"
ASCORE_SUMMARY_DIR = "ascore_operational_summary"
EFF_SUMMARY_DIR = "eff_summary"
DRIFT_SUMMARY_DIR = "thres_drift_summary"
WASSERSTEIN_SUMMARY_DIR = "wasserstein_summary"
#: Outputs of the previous layout, removed when a run is compared again.
LEGACY_OUTPUTS = ("summary.png", "efficiency", "ascore_operational")

# Colours of the reconstruction plots (the bar charts take theirs from
# src/plot/horizontal_bar.py, which uses the same orange and blue).
RUN_COLOR, GAMMA0_COLOR = "#2a78d6", "#eb6834"
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


def model_label(run: RunInfo) -> str:
    """Legend label ``γ = <γ>, bins = <bins>``, read from the run name.

    The name carries both (``..._Gamma_0.1_Bins_40_...``); a run named otherwise
    falls back to its resolved config.
    """
    gamma = re.search(r"Gamma_([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", run.name)
    bins = re.search(r"Bins_([0-9]+)", run.name)
    gamma_text = gamma.group(1) if gamma else ("n/a" if run.gamma is None else f"{run.gamma:g}")
    bins_text = bins.group(1) if bins else ("n/a" if run.bins is None else str(run.bins))
    return rf"$\gamma$ = {gamma_text}, bins = {bins_text}"


def _drift_display(rate: str) -> str:
    # As src/evaluation/callbacks/thres_drift.ThresholdDrift.plot_summary.
    return "operational" if rate == "operational" else str(rate).replace("trate", "").replace("kHz", "")


@dataclass
class BarPlot:
    """One comparison bar chart: where it goes and what it shows."""

    folder: str                # sub-folder of comparison/, as in plots/<split>/
    xlabel: str                # the x label of the mirrored plot
    title: str
    quantities: list           # Quantity per key; the key is Quantity.name
    percent: bool = False
    filename: Optional[str] = None  # default: the stem horizontal_bar derives from xlabel


def bar_plots(run_out: Outputs, ref_out: Outputs, ckpt: str) -> list[BarPlot]:
    """The comparison charts, one per plot of plots/<split>/ that both runs have."""
    plots = []
    efficiency = run_out.efficiency or {}
    point = (efficiency.get("operating_point") or {}).get("label") or "operational"
    score = f"anomaly score: {efficiency.get('anomaly_score') or 'ascore/full'}"

    per_signal = per_key_quantities(
        "efficiency_per_signal", efficiency.get("signal_efficiencies"),
        (ref_out.efficiency or {}).get("signal_efficiencies"), "higher")
    if per_signal:
        plots.append(BarPlot(EFF_DIR, f"efficiency at threshold: {point}\n{score}",
                             "Signal efficiency at the operating point", per_signal, percent=True))

    ascore = per_key_quantities("mean_ascore", run_out.ascore, ref_out.ascore, None)
    if ascore:
        plots.append(BarPlot(ASCORE_SUMMARY_DIR, "ascore/operational",
                             "Mean anomaly score (ascore/operational) per dataset", ascore,
                             filename="ascore_operational"))

    if run_out.efficiency and ref_out.efficiency:
        for name, label, title in (
            ("cvar25_efficiency", f"25% CVaR at threshold: {point}\n{score}",
             "25% CVaR of the signal efficiencies"),
            ("median_efficiency", f"med eff at: {point}\n{score}", "Median signal efficiency"),
            ("min_efficiency", f"min eff at: {point}\n{score}", "Minimum signal efficiency"),
        ):
            q = _pair("efficiency", ckpt, run_out.efficiency.get(name),
                      ref_out.efficiency.get(name), "higher")
            if q is not None:
                plots.append(BarPlot(EFF_SUMMARY_DIR, label, title, [q], percent=True))

    if run_out.thres_drift and ref_out.thres_drift:
        for rate in sorted(set(run_out.thres_drift) & set(ref_out.thres_drift), key=str):
            display = _drift_display(rate)
            q = _pair("thres_drift", ckpt, run_out.thres_drift[rate], ref_out.thres_drift[rate], "lower")
            if q is not None:
                plots.append(BarPlot(
                    DRIFT_SUMMARY_DIR,
                    f"split-transfer drift at threshold: {display}\nlog((p̂+ε)/(FPR+ε))",
                    f"Split-transfer threshold drift ({display})", [q]))

    if run_out.wasserstein is not None and ref_out.wasserstein is not None:
        q = _pair("wasserstein", ckpt, run_out.wasserstein, ref_out.wasserstein, "lower")
        if q is not None:
            plots.append(BarPlot(WASSERSTEIN_SUMMARY_DIR, "W1(normal,\nSingleNeutrino_E-10-gun)",
                                 "Wasserstein distance W1(normal, SingleNeutrino_E-10-gun)", [q]))
    return plots


def draw_bar_plot(plot: BarPlot, out: Path, *, run_label: str, reference_label: str,
                  subtitle: str) -> Path:
    """Draw ``plot`` into ``out/<folder>/`` and write its table next to it."""
    from src.plot import horizontal_bar

    path = horizontal_bar.plot_comparison(
        {q.name: q.gamma0 for q in plot.quantities},
        {q.name: q.run for q in plot.quantities},
        plot.xlabel,
        out / plot.folder,
        reference_label=reference_label,
        run_label=run_label,
        title=f"{plot.title}\n{subtitle}",
        percent=plot.percent,
        filename=plot.filename,
    )
    _write_table(plot.quantities, path.with_suffix(".csv"))
    return path


def _is_et(feature: str) -> bool:
    # As src/plot/overlaid_hist.check_feature_is_Et.
    return ("Et" in feature or "ETTEM" in feature) and "Eta" not in feature


def plot_reco(table: pd.DataFrame, path: Path, *, dataset: str, feature: str,
              run_label: str, reference_label: str) -> None:
    """Input (reference fill) and both reconstructions; their difference below.

    ``run_label`` / ``reference_label`` name the two models in the legend
    (:func:`model_label`, e.g. ``γ = 0.1, bins = 40`` and ``γ = 0, bins = 50``).
    """
    plt = _style()
    inner = table.iloc[1:-1]  # under/overflow rows are reported, not drawn
    edges = np.concatenate([inner["bin_low"].to_numpy(), inner["bin_high"].to_numpy()[-1:]])
    fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06})
    ax.stairs(inner["input_fraction"], edges, fill=True, color=INPUT_FILL, label="input")
    ax.stairs(inner["reco_gamma0_fraction"], edges, color=GAMMA0_COLOR, lw=2,
              label=f"reco, {reference_label}")
    ax.stairs(inner["reco_run_fraction"], edges, color=RUN_COLOR, lw=2, label=f"reco, {run_label}")
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
    ax_diff.set_ylabel("run - gamma0")
    ax_diff.set_xlabel(feature)
    ax_diff.grid(axis="y")

    # The TVD and the entries outside the bins are in reco_summary.csv / the CSV.
    ax.set_title(f"{dataset}: {feature}", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reco_overlay(run_hist: pd.DataFrame, ref_hist: pd.DataFrame, path: Path, *,
                      dataset: str, feature: str, run_label: str, reference_label: str) -> None:
    """Fallback when the bins differ: both reconstructions, no difference panel."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hist, color, label in ((ref_hist, GAMMA0_COLOR, f"reco, {reference_label}"),
                               (run_hist, RUN_COLOR, f"reco, {run_label}")):
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
    for legacy in LEGACY_OUTPUTS:  # the previous layout of this folder
        legacy_path = out / legacy
        if legacy_path.is_dir():
            shutil.rmtree(legacy_path)
        elif legacy_path.exists():
            legacy_path.unlink()
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
    run_label, reference_label = model_label(run), model_label(reference)
    for plot in bar_plots(run_out, ref_out, ckpt):
        draw_bar_plot(plot, out, run_label=run_label, reference_label=reference_label,
                      subtitle=subtitle)

    reco_rows = []
    for dataset in sorted(set(run_out.reco) & set(ref_out.reco)):
        for feature in sorted(set(run_out.reco[dataset]) & set(ref_out.reco[dataset])):
            run_hist, ref_hist = run_out.reco[dataset][feature], ref_out.reco[dataset][feature]
            folder = out / "reco" / dataset
            table = reco_comparison(run_hist, ref_hist)
            if table is None:
                plot_reco_overlay(run_hist, ref_hist, folder / f"{feature}.png", dataset=dataset,
                                  feature=feature, run_label=run_label, reference_label=reference_label)
                reco_rows.append({"dataset": dataset, "feature": feature, "same_bins": False})
                continue
            folder.mkdir(parents=True, exist_ok=True)
            table.to_csv(folder / f"{feature}.csv", index=False)
            plot_reco(table, folder / f"{feature}.png", dataset=dataset, feature=feature,
                      run_label=run_label, reference_label=reference_label)
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


#: Tables of the comparison folder that also go to MLflow (shown there as tables).
MLFLOW_TABLES = ("summary.csv", "reference.json", "reco/reco_summary.csv")
MLFLOW_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def mlflow_artifacts(out: Path) -> dict[str, tuple[str, Path]]:
    """What the comparison folder ``out`` becomes in MLflow.

    Maps each artifact path, relative to ``<split>/<ckpt>/comparison/``, to
    ``("copy", file)`` or ``("gallery", image folder)``. As the evaluator's
    ``log_plots_to_mlflow`` does it: per plot folder the images and a gallery page
    beside the folder, named ``<dataset>_reco.html`` for ``reco/<dataset>``.
    """
    files: dict[str, tuple[str, Path]] = {}
    images = [p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in MLFLOW_IMAGE_SUFFIXES]
    for folder in sorted({p.parent for p in images}):
        rel = folder.relative_to(out)
        for image in sorted(p for p in images if p.parent == folder):
            files[(rel / image.name).as_posix()] = ("copy", image)
        if len(rel.parts) == 2 and rel.parts[0] == "reco":
            gallery = f"{rel.parts[1]}_reco.html"
        else:
            gallery = f"{rel.as_posix() if rel.parts else 'plots'}.html"
        files[gallery] = ("gallery", folder)
    for table in MLFLOW_TABLES:
        if (out / table).is_file():
            files[table] = ("copy", out / table)
    return files


def write_mlflow_artifacts(run_dir: Path, mlflow_run_dirs: Iterable[Path], *, split: str = "test",
                           ckpt: str = "loss_total") -> int:
    """Put ``run_dir``'s comparison into ``<mlflow run>/artifacts/<split>/<ckpt>/comparison/``.

    The previous comparison artifacts are replaced, as ``compare_run`` replaces the
    folder itself. Returns the number of files written per MLflow run.
    """
    from src.plot.gallery import build_gallery_html

    files = mlflow_artifacts(run_dir / "plots" / split / ckpt / COMPARISON_DIR)
    galleries = {rel: build_gallery_html(source, ckpt)
                 for rel, (kind, source) in files.items() if kind == "gallery"}
    for mlflow_run in mlflow_run_dirs:
        target = Path(mlflow_run) / "artifacts" / split / ckpt / COMPARISON_DIR
        staging = target.with_name(f".{COMPARISON_DIR}.tmp")
        if staging.exists():
            shutil.rmtree(staging)
        for rel, (kind, source) in files.items():
            path = staging / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "gallery":
                path.write_text(galleries[rel], encoding="utf-8")
            else:
                shutil.copyfile(source, path)
        if target.exists():
            shutil.rmtree(target)
        if files:
            staging.replace(target)
    return len(files)


@dataclass
class Report:
    written: list = field(default_factory=list)   # (run name, reference name, missing)
    skipped: list = field(default_factory=list)   # (run name, reason)
    mlflow: dict = field(default_factory=dict)    # run name -> (MLflow runs, files per run)


def process_experiment(experiment_dir: Path, *, run_names: Optional[Iterable[str]] = None,
                       split: str = "test", ckpt: str = "loss_total",
                       mlruns_root: Optional[Path] = None, mlflow_only: bool = False) -> Report:
    """Compare every listed run with its γ = 0 run; with ``mlruns_root`` also fill MLflow.

    ``mlflow_only`` skips the comparison and only rewrites the MLflow artifacts from
    the comparison folders that already exist.
    """
    if mlflow_only and mlruns_root is None:
        raise ValueError("mlflow_only needs mlruns_root")
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
        if mlflow_only:
            out = run.run_dir / "plots" / split / ckpt / COMPARISON_DIR
            reference = _read_json(out / "reference.json")
            if reference is None:
                report.skipped.append((name, f"no {split} comparison yet (run without --mlflow-only)"))
                continue
            report.written.append((name, reference.get("reference_run", "?"), []))
            _fill_mlflow(report, run, experiment_dir, mlruns_root, split, ckpt)
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
        if mlruns_root is not None:
            _fill_mlflow(report, run, experiment_dir, mlruns_root, split, ckpt)
    return report


def _fill_mlflow(report: Report, run: RunInfo, experiment_dir: Path, mlruns_root: Path,
                 split: str, ckpt: str) -> None:
    runs = mlflow_runs(mlruns_root, experiment_dir.name, run.name)
    files = write_mlflow_artifacts(run.run_dir, runs, split=split, ckpt=ckpt) if runs else 0
    report.mlflow[run.name] = (len(runs), files)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare runs' test outputs with their γ = 0 / 50-bin run.")
    parser.add_argument("--experiment-dir", type=Path, required=True, help="checkpoints/<experiment>")
    parser.add_argument("--run-name", action="append", dest="run_names",
                        help="Only this run. Repeatable; default: every run with outputs.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ckpt", default="loss_total")
    parser.add_argument("--mlruns-root", type=Path, default=None,
                        help="MLflow file store; also put the comparison into each run's artifacts "
                             "under <split>/<ckpt>/comparison/.")
    parser.add_argument("--mlflow-only", action="store_true",
                        help="Only rewrite the MLflow artifacts from the existing comparison folders.")
    args = parser.parse_args(argv)
    if args.mlflow_only and args.mlruns_root is None:
        parser.error("--mlflow-only needs --mlruns-root")
    import matplotlib

    matplotlib.use("Agg")
    report = process_experiment(args.experiment_dir, run_names=args.run_names,
                                split=args.split, ckpt=args.ckpt,
                                mlruns_root=args.mlruns_root, mlflow_only=args.mlflow_only)
    verb = "copied to MLflow" if args.mlflow_only else "compared with γ = 0"
    print(f"{args.experiment_dir.name}: {len(report.written)} run(s) {verb} ({args.split}, {args.ckpt})")
    for name, reference, missing in report.written:
        extra = f"; missing: {', '.join(missing)}" if missing else ""
        if name in report.mlflow:
            n_runs, n_files = report.mlflow[name]
            extra += (f"; MLflow: {n_files} files in {n_runs} run(s)" if n_runs
                      else f"; MLflow: no run named {name} in {args.experiment_dir.name}")
        print(f"  {name} vs {reference}{extra}")
    for name, reason in report.skipped:
        print(f"  skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
