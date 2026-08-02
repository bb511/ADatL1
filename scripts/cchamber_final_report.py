"""Create the curated Causal Chamber report figures and physical-shift tables.

This is a presentation and complementary-analysis layer over the immutable
paper-analysis bundle.  It does not select models, change thresholds, or run
new confirmatory tests.  Physical associations follow the outcome-blind
``physical_shift_estimand_v1.json`` contract: interventions are the unit,
reporting seeds are averaged first, and associations are stratified by
semantic family or intervention target.  A pooled cross-family causal
regression is intentionally not produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ("ae", "vae", "svdd", "realnvp")
PRESENTATION_MODELS = ("svdd", "ae", "vae", "realnvp")
STRATEGIES = (
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "cap_random",
    "cap_cdf",
    "drift",
    "wasserstein",
)
METRICS = ("auprc", "efficiency_operational")
MODEL_LABELS = {"ae": "AE", "vae": "VAE", "svdd": "SVDD", "realnvp": "RealNVP"}
STRATEGY_LABELS = {
    "cap_metadata_nearest": "CAP (metadata)",
    "cap_encoder_nearest": "CAP (encoder)",
    "cap_random": "CAP (random pairs)",
    "cap_cdf": "CAP (CDF ranks)",
    "drift": "Marginal drift",
    "wasserstein": "Wasserstein",
}
STRATEGY_COLORS = {
    "cap_metadata_nearest": "#0072B2",
    "cap_encoder_nearest": "#56B4E9",
    "cap_random": "#999999",
    "cap_cdf": "#CC79A7",
    "drift": "#E69F00",
    "wasserstein": "#009E73",
}
CLASS_MAP = {
    "process_or_actuator": "process",
    "measurement_chain": "measurement",
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    """Require finite numeric values in selected columns."""
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values.")


def _prepare_output(path: Path) -> Path:
    """Create an empty output directory without overwriting artifacts."""
    path = path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_inputs(
    paper_dir: Path,
    threshold_results: Path,
    physical_dir: Path,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Load and validate every frozen input used by the report layer."""
    paths = {
        "strategy_summary": paper_dir / "strategy_summary.csv",
        "contrasts": paper_dir / "prespecified_strategy_contrasts.csv",
        "rank": paper_dir / "candidate_rank_associations.csv",
        "intervention_contrasts": paper_dir / "intervention_cap_baseline_summary.csv",
        "paper_provenance": paper_dir / "analysis_provenance.json",
        "threshold_results": threshold_results,
        "physical_shift": physical_dir / "physical_shift_magnitude.csv",
        "physical_provenance": physical_dir / "physical_shift_provenance.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    tables = {name: _read_csv(path) for name, path in paths.items() if path.suffix == ".csv"}
    strategy = tables["strategy_summary"]
    expected_strategy = {
        (model, method, metric) for model in MODELS for method in STRATEGIES for metric in METRICS
    }
    observed_strategy = set(
        strategy.loc[:, ["model", "strategy", "metric"]].itertuples(index=False, name=None)
    )
    if observed_strategy != expected_strategy or len(strategy) != len(expected_strategy):
        raise ValueError("Strategy summary does not have exact 4x6x2 coverage.")
    _finite(strategy, ["mean", "ci_low", "ci_high"], "strategy summary")

    rank = tables["rank"]
    observed_rank = set(
        rank.loc[:, ["model", "strategy", "metric"]].itertuples(index=False, name=None)
    )
    if observed_rank != expected_strategy or len(rank) != len(expected_strategy):
        raise ValueError("Candidate-rank table does not have exact 4x6x2 coverage.")
    _finite(rank, ["spearman_rho", "spearman_holm_p"], "candidate-rank table")

    results = tables["threshold_results"]
    identity = ["model", "strategy", "seed", "intervention", "metric"]
    if len(results) != 27_840 or results.duplicated(identity).any():
        raise ValueError("Threshold results do not have exact unique 27,840-row coverage.")
    if set(results["model"]) != set(MODELS) or set(results["strategy"]) != set(STRATEGIES):
        raise ValueError("Threshold result model/strategy identities do not match the report.")
    if set(results["metric"]) != set(METRICS):
        raise ValueError("Threshold result metrics do not match the report.")
    if results["seed"].nunique() != 10 or results["intervention"].nunique() != 58:
        raise ValueError("Threshold results require 10 seeds and 58 interventions.")
    _finite(results, ["value"], "threshold results")

    physical = tables["physical_shift"].copy()
    if len(physical) != 58 or physical["intervention"].duplicated().any():
        raise ValueError("Physical-shift table must contain 58 unique interventions.")
    if set(physical["physical_class"]) != set(CLASS_MAP):
        raise ValueError("Unknown physical intervention class.")
    physical["system_group"] = physical["physical_class"].map(CLASS_MAP)
    _finite(physical, ["biased_energy_distance"], "physical-shift table")
    if (physical["biased_energy_distance"] < 0).any():
        raise ValueError("Physical shift magnitude cannot be negative.")
    if set(physical["intervention"]) != set(results["intervention"]):
        raise ValueError("Physical and outcome intervention identities differ.")
    tables["physical_shift"] = physical
    return paths, tables


def _spearman_rows(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    x: str,
    y: str,
    minimum: int,
) -> pd.DataFrame:
    """Compute descriptive Spearman coefficients within fixed strata."""
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        x_values = group[x].to_numpy(dtype=float)
        y_values = group[y].to_numpy(dtype=float)
        if len(group) < minimum or np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
            rho = math.nan
            reason = "insufficient_or_constant"
        else:
            rho = float(stats.spearmanr(x_values, y_values).statistic)
            reason = ""
        rows.append(
            {name: value for name, value in zip(group_columns, keys)}
            | {
                "n_interventions": len(group),
                "spearman_rho": rho,
                "undefined_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def physical_associations(
    results: pd.DataFrame,
    physical: pd.DataFrame,
    official_contrasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return seed-first intervention values and the three frozen association tables."""
    metadata = physical[
        [
            "intervention",
            "target",
            "strength",
            "semantic_family",
            "system_group",
            "biased_energy_distance",
        ]
    ]
    intervention = (
        results.groupby(["model", "strategy", "metric", "intervention"], sort=True)["value"]
        .mean()
        .rename("mean_performance")
        .reset_index()
        .merge(metadata, on="intervention", validate="many_to_one")
    )
    family = _spearman_rows(
        intervention,
        ["model", "strategy", "metric", "system_group", "semantic_family"],
        x="biased_energy_distance",
        y="mean_performance",
        minimum=3,
    )
    target = _spearman_rows(
        intervention,
        ["model", "strategy", "metric", "system_group", "semantic_family", "target"],
        x="biased_energy_distance",
        y="mean_performance",
        minimum=3,
    )
    target = target[target["n_interventions"] >= 3].reset_index(drop=True)

    identity = ["model", "metric", "intervention", "strategy"]
    pivot = intervention.pivot(
        index=[
            "model",
            "metric",
            "intervention",
            "target",
            "strength",
            "semantic_family",
            "system_group",
            "biased_energy_distance",
        ],
        columns="strategy",
        values="mean_performance",
    ).reset_index()
    gain_frames = []
    contrasts = (
        official_contrasts[["contrast_id", "strategy_left", "strategy_right"]]
        .drop_duplicates()
        .sort_values("contrast_id")
    )
    for row in contrasts.itertuples(index=False):
        gain = pivot[
            [
                "model",
                "metric",
                "intervention",
                "target",
                "strength",
                "semantic_family",
                "system_group",
                "biased_energy_distance",
            ]
        ].copy()
        gain["contrast_id"] = row.contrast_id
        gain["strategy_left"] = row.strategy_left
        gain["strategy_right"] = row.strategy_right
        gain["mean_gain"] = pivot[row.strategy_left] - pivot[row.strategy_right]
        gain_frames.append(gain)
    gains = pd.concat(gain_frames, ignore_index=True)

    official = official_contrasts[
        ["model", "metric", "intervention", "contrast_id", "mean_difference"]
    ]
    check = gains.merge(
        official,
        on=["model", "metric", "intervention", "contrast_id"],
        validate="one_to_one",
    )
    if len(check) != len(gains) or not np.allclose(
        check["mean_gain"], check["mean_difference"], rtol=0, atol=1e-12
    ):
        raise ValueError("Recomputed CAP gains disagree with the frozen paper analysis.")

    gain_family = _spearman_rows(
        gains,
        [
            "model",
            "metric",
            "contrast_id",
            "strategy_left",
            "strategy_right",
            "system_group",
            "semantic_family",
        ],
        x="biased_energy_distance",
        y="mean_gain",
        minimum=3,
    )
    if intervention.duplicated(identity).any():
        raise ValueError("Intervention performance identities are not unique.")
    return intervention, family, target, gain_family


def _plot_architecture_overview(
    strategy: pd.DataFrame,
    output: Path,
    *,
    models: tuple[str, ...] = PRESENTATION_MODELS,
    strategies: tuple[str, ...] = STRATEGIES,
) -> None:
    """Plot selected performance and seed-level intervals for every architecture."""
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True)
    x = np.arange(len(models), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(strategies))
    for axis, metric in zip(axes, METRICS):
        for offset, method in zip(offsets, strategies):
            selected = (
                strategy[(strategy["metric"] == metric) & (strategy["strategy"] == method)]
                .set_index("model")
                .reindex(models)
            )
            mean = selected["mean"].to_numpy(dtype=float)
            low = mean - selected["ci_low"].to_numpy(dtype=float)
            high = selected["ci_high"].to_numpy(dtype=float) - mean
            axis.errorbar(
                x + offset,
                mean,
                yerr=np.vstack([low, high]),
                fmt="o",
                capsize=2.5,
                markersize=5,
                color=STRATEGY_COLORS[method],
                label=STRATEGY_LABELS[method],
            )
        axis.set_xticks(x, [MODEL_LABELS[model] for model in models])
        axis.set_ylabel("Mean AUPRC" if metric == "auprc" else "Efficiency at frozen 1% FPR")
        axis.set_title(
            "Primary: intervention-weighted AUPRC"
            if metric == "auprc"
            else "Secondary operating-point endpoint"
        )
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(strategies), frameon=False)
    figure.suptitle("Selected-checkpoint performance by architecture and label-free criterion")
    figure.tight_layout(rect=(0, 0.12, 1, 0.95))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_rank_heatmap(
    rank: pd.DataFrame,
    output: Path,
    *,
    models: tuple[str, ...] = PRESENTATION_MODELS,
    strategies: tuple[str, ...] = STRATEGIES,
) -> None:
    """Plot candidate-rank validity."""
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    short_labels = tuple(STRATEGY_LABELS[strategy] for strategy in strategies)
    for axis, metric in zip(axes, METRICS):
        selected = rank[rank["metric"] == metric]
        rho = selected.pivot(index="model", columns="strategy", values="spearman_rho").reindex(
            index=models, columns=strategies
        )
        image = axis.imshow(rho.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        for row in range(len(models)):
            for column in range(len(strategies)):
                value = rho.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(value) > 0.55 else "black",
                )
        axis.set_xticks(
            np.arange(len(strategies)),
            short_labels,
            rotation=35,
            ha="right",
        )
        axis.set_yticks(
            np.arange(len(models)),
            [MODEL_LABELS[name] for name in models],
        )
        axis.set_title("AUPRC" if metric == "auprc" else "Efficiency at 1% FPR")
    color_axis = figure.add_axes((0.925, 0.22, 0.016, 0.58))
    figure.colorbar(image, cax=color_axis, label="Candidate-rank Spearman ρ")
    figure.suptitle("Does the label-free criterion rank sealed candidate performance?")
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.22, top=0.84, wspace=0.08)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _class_rho(frame: pd.DataFrame, y: str, group: str) -> float:
    """Return a descriptive within-system-group Spearman coefficient."""
    selected = frame[frame["system_group"] == group]
    if len(selected) < 3:
        return math.nan
    return float(stats.spearmanr(selected["biased_energy_distance"], selected[y]).statistic)


def _add_presentation_contrasts(pivot: pd.DataFrame) -> pd.DataFrame:
    """Add the three strategy contrasts shown in the physical synthesis."""
    required = {
        "cap_metadata_nearest",
        "cap_encoder_nearest",
        "cap_random",
        "drift",
        "wasserstein",
    }
    missing = sorted(required - set(pivot.columns))
    if missing:
        raise ValueError(f"Physical synthesis misses strategies {missing}.")
    output = pivot.copy()
    output["metadata_minus_random"] = output["cap_metadata_nearest"] - output["cap_random"]
    output["encoder_minus_drift"] = output["cap_encoder_nearest"] - output["drift"]
    output["encoder_minus_wasserstein"] = output["cap_encoder_nearest"] - output["wasserstein"]
    return output


def _plot_physical_synthesis(
    intervention: pd.DataFrame,
    output: Path,
) -> None:
    """Plot physical shift against detectability and metadata-CAP gain."""
    auprc = intervention[intervention["metric"] == "auprc"]
    index = [
        "model",
        "intervention",
        "target",
        "strength",
        "semantic_family",
        "system_group",
        "biased_energy_distance",
    ]
    pivot = auprc.pivot(index=index, columns="strategy", values="mean_performance").reset_index()
    pivot = _add_presentation_contrasts(pivot)
    figure, axes = plt.subplots(
        4,
        len(MODELS),
        figsize=(15.5, 13.2),
        sharex="col",
        sharey="row",
    )
    class_style = {
        "process": ("#D55E00", "o", "process / actuator"),
        "measurement": ("#0072B2", "^", "measurement chain"),
    }
    for column, model in enumerate(MODELS):
        selected = pivot[pivot["model"] == model]
        top = axes[0, column]
        metadata_random = axes[1, column]
        encoder_drift = axes[2, column]
        encoder_wasserstein = axes[3, column]
        for group, (color, marker, label) in class_style.items():
            group_rows = selected[selected["system_group"] == group]
            top.scatter(
                group_rows["biased_energy_distance"],
                group_rows["cap_metadata_nearest"],
                c=color,
                marker=marker,
                s=24,
                alpha=0.8,
                label=label,
            )
            top.scatter(
                group_rows["biased_energy_distance"],
                group_rows["cap_random"],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=24,
                alpha=0.55,
            )
            metadata_random.scatter(
                group_rows["biased_energy_distance"],
                group_rows["metadata_minus_random"],
                c=color,
                marker=marker,
                s=24,
                alpha=0.8,
            )
            encoder_drift.scatter(
                group_rows["biased_energy_distance"],
                group_rows["encoder_minus_drift"],
                c=color,
                marker=marker,
                s=24,
                alpha=0.8,
            )
            encoder_wasserstein.scatter(
                group_rows["biased_energy_distance"],
                group_rows["encoder_minus_wasserstein"],
                c=color,
                marker=marker,
                s=24,
                alpha=0.8,
            )
        process_rho = _class_rho(selected, "cap_metadata_nearest", "process")
        measurement_rho = _class_rho(selected, "cap_metadata_nearest", "measurement")
        top.text(
            0.03,
            0.96,
            f"CAP ρ: process {process_rho:+.2f}\nmeasurement {measurement_rho:+.2f}",
            transform=top.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        for contrast_axis in (metadata_random, encoder_drift, encoder_wasserstein):
            contrast_axis.axhline(0, color="black", linewidth=0.8)
        for axis in (top, metadata_random, encoder_drift, encoder_wasserstein):
            axis.set_xscale("log")
            axis.grid(alpha=0.18)
        top.set_ylim(0, 1.02)
        top.set_title(MODEL_LABELS[model])
        encoder_wasserstein.set_xlabel("Physical shift (energy distance, log scale)")
        if column == 0:
            top.set_ylabel("AUPRC")
            metadata_random.set_ylabel("Metadata CAP − random-pair CAP")
            encoder_drift.set_ylabel("Encoder CAP − marginal drift")
            encoder_wasserstein.set_ylabel("Encoder CAP − Wasserstein")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.text(
        0.5,
        0.045,
        "Filled: metadata CAP; open: random-pair CAP. "
        "All contrasts are AUPRC differences. Correlations are descriptive "
        "within system group; no pooled causal regression.",
        ha="center",
        fontsize=9,
    )
    figure.suptitle("Controlled physical shift, anomaly detectability, and CAP selection gain")
    figure.tight_layout(rect=(0, 0.07, 1, 0.97), h_pad=1.7)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run(
    paper_dir: Path,
    threshold_results: Path,
    physical_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Validate inputs and write the curated tables, figures, and provenance."""
    paper_dir = paper_dir.expanduser().resolve()
    threshold_results = threshold_results.expanduser().resolve()
    physical_dir = physical_dir.expanduser().resolve()
    output_dir = _prepare_output(output_dir)
    paths, tables = _validate_inputs(paper_dir, threshold_results, physical_dir)
    intervention, family, target, gain_family = physical_associations(
        tables["threshold_results"],
        tables["physical_shift"],
        tables["intervention_contrasts"],
    )
    outputs: list[Path] = []
    for name, table in (
        ("intervention_seed_first_performance.csv", intervention),
        ("shift_performance_by_semantic_family.csv", family),
        ("shift_performance_within_target.csv", target),
        ("shift_cap_gain_by_semantic_family.csv", gain_family),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)

    architecture = output_dir / "main_architecture_strategy_overview.png"
    _plot_architecture_overview(tables["strategy_summary"], architecture)
    outputs.append(architecture)
    rank = output_dir / "main_candidate_rank_validity.png"
    _plot_rank_heatmap(tables["rank"], rank)
    outputs.append(rank)
    physical = output_dir / "main_physical_shift_synthesis.png"
    _plot_physical_synthesis(intervention, physical)
    outputs.append(physical)

    provenance_path = output_dir / "curated_report_provenance.json"
    provenance = {
        "schema_version": 1,
        "analysis_classification": "prespecified complementary physical associations and presentation",
        "selection_or_threshold_changes": False,
        "pooled_cross_family_causal_regression": False,
        "input_files": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs.append(provenance_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-analysis-dir", type=Path, required=True)
    parser.add_argument("--threshold-results", type=Path, required=True)
    parser.add_argument("--physical-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report builder and print each created artifact."""
    args = parse_args(argv)
    for path in run(
        args.paper_analysis_dir,
        args.threshold_results,
        args.physical_results_dir,
        args.output_dir,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
