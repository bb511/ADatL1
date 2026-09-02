"""Observation-only plots generated from the same diagnostics stored in JSON."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import rc_context, rcParamsDefault
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


def _style_axis(axis, title: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=11, loc="left")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def _draw_history(axis, history: dict, key: str, label: str, **kwargs) -> bool:
    values = np.asarray(history.get(key, []), dtype=float)
    if not values.size or not np.isfinite(values).any():
        return False
    axis.plot(np.arange(1, len(values) + 1), values, label=label, **kwargs)
    return True


def _finish_history_axis(axis) -> None:
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=8, frameon=False)
    else:
        axis.text(0.5, 0.5, "History unavailable", ha="center", va="center", transform=axis.transAxes)


def _plot_mlp(probe: dict, title: str, scope: str) -> Figure:
    figure = Figure(figsize=(11, 7.2))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.84, wspace=0.30, hspace=0.42)
    figure.suptitle(f"{title} | fitting diagnostics\n{scope}", fontsize=14, y=0.97)

    _style_axis(axes[0, 0], "Candidate seeds — training loss", "Standardized objective")
    _style_axis(axes[0, 1], "Selected-seed fresh refit — training loss", "Standardized objective")
    _style_axis(axes[1, 0], "Candidates — internal early stopping", "Internal validation R²")
    _style_axis(axes[1, 1], "Refit — internal early stopping", "Internal validation R²")

    failed = []
    for candidate in probe["seed_selection"]["candidates"]:
        if candidate["status"] != "successful":
            failed.append(str(candidate["seed"]))
            continue
        selected = candidate["selected"]
        label = f"seed {candidate['seed']}" + (" (selected)" if selected else "")
        history = candidate.get("training_history", {})
        style = {"linewidth": 2.2 if selected else 1.3, "alpha": 1.0 if selected else 0.75}
        _draw_history(axes[0, 0], history, "loss", label, **style)
        _draw_history(axes[1, 0], history, "early_stopping_validation_r2", label, **style)

    history = probe.get("training_history", {})
    refit_label = f"refit seed {probe['selected_seed']}"
    _draw_history(axes[0, 1], history, "loss", refit_label, color="#0e7490", linewidth=2)
    _draw_history(axes[1, 1], history, "early_stopping_validation_r2", refit_label, color="#0e7490", linewidth=2)
    scores = np.asarray(history.get("early_stopping_validation_r2", []), dtype=float)
    if scores.size and np.isfinite(scores).any():
        best_epoch = int(np.nanargmax(scores)) + 1
        for axis in axes[:, 1]:
            axis.axvline(best_epoch, color="#b45309", linestyle="--", linewidth=1, label=f"best internal R²: epoch {best_epoch}")
    for axis in axes.flat:
        _finish_history_axis(axis)
    if failed:
        axes[0, 0].text(0.98, 0.98, "Failed seeds: " + ", ".join(failed), ha="right", va="top", transform=axes[0, 0].transAxes, fontsize=8)

    figure.text(
        0.5, 0.03,
        "Candidate and refit scalers/pools differ. Loss is dimensionless (includes L2), not GeV².\n"
        "Internal validation curves are not held-out scores. Refit retains the best internal-validation weights.",
        ha="center", va="bottom", fontsize=9, color="#475569",
    )
    return figure


def _plot_linear(probe: dict, title: str, scope: str, held_out_split: str) -> Figure:
    figure = Figure(figsize=(7, 4.8))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    figure.subplots_adjust(left=0.16, right=0.95, top=0.78, bottom=0.21)
    figure.suptitle(f"{title} | final loss\n{scope}", fontsize=13, y=0.96)
    summary = probe["loss_summary"]
    values = [summary["development_mse_gev2"], summary["held_out_mse_gev2"]]
    colors = ["#0e7490", "#b45309"]
    for index, value in enumerate(values):
        if value is None:
            axis.text(index, 0.05, "Unavailable", ha="center", transform=axis.get_xaxis_transform())
        else:
            axis.bar(index, value, width=0.5, color=colors[index])
            axis.annotate(f"{value:.4g}", (index, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    axis.set_xticks([0, 1], ["Development", f"Held-out ({held_out_split})"])
    axis.set_xlim(-0.6, 1.6)
    axis.set_ylabel("Mean squared error [GeV²]")
    axis.set_ylim(bottom=0, top=max([value for value in values if value is not None] + [1e-12]) * 1.25)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(0.5, 0.05, "Direct least-squares solve: no epochs and no loss curve.\nFinal model evaluated once on each pool; held-out data is never fitted.", ha="center", fontsize=9, color="#475569")
    return figure


def write_probe_loss_plots(payload: dict[str, Any], artifact_path: Path) -> None:
    """Write one PNG per JSON probe and attach paths relative to that JSON.

    Smoke and scientific plots have separate directories. A plotting failure is
    recorded explicitly but cannot invalidate or erase the numerical measurement.
    """
    evaluation = payload["evaluation"]
    scope = (
        "SMOKE TEST — NON-REPORTABLE"
        if evaluation["purpose"] == "smoke_test"
        else evaluation["mode"].replace("_", " ").upper()
    )
    held_out_split = evaluation["held_out_data"]["split"]
    entries = list(payload["probes"].items())
    controls = payload.get("diagnostics", {}).get("shuffled_targets", {})
    if controls.get("enabled", False):
        entries.extend((f"shuffled_mlp/{name}", controls[name]) for name in ("z_logits", "reconstruction"))

    plot_directory = artifact_path.parent / f"{artifact_path.stem}_loss_plots"
    for name, probe in entries:
        figure = None
        try:
            plot_directory.mkdir(parents=True, exist_ok=True)
            path = plot_directory / f"{name.replace('/', '_')}.png"
            title = name.replace("shuffled_mlp/", "Shuffled-target MLP · ").replace("mlp/", "MLP · ").replace("linear/", "Linear · ")
            # Other evaluation modules install a large-font mplhep style globally.
            # Keep these compact diagnostics independent without changing that style.
            defaults = {key: value for key, value in rcParamsDefault.items() if key != "backend"}
            with rc_context(defaults):
                if "training_history" in probe:
                    figure = _plot_mlp(probe, title, scope)
                else:
                    figure = _plot_linear(probe, title, scope, held_out_split)
                figure.savefig(path, dpi=150, facecolor="white")
            probe["loss_plot"] = {"status": "created", "path": path.relative_to(artifact_path.parent).as_posix()}
            log.info("Saved %s loss plot: %s", name, path)
        except Exception as error:
            probe["loss_plot"] = {"status": "failed", "path": None, "error": str(error)}
            log.warning("Could not save %s loss plot: %s", name, error, exc_info=True)
        finally:
            if figure is not None:
                figure.clear()
