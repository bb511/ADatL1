"""Analyze the frozen residual-OAS Causal Chamber AE campaign.

Candidate selection uses only validation-normal branch monitors. Intervention outcomes are averaged
within reporting seed before any across-seed summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import cchamber_ae_residual_oas_campaign as campaign  # noqa: E402
import cchamber_candidate_rank_audit as rank  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

METRIC_LABELS = {"auprc": "AUPRC", "efficiency_operational": "Efficiency at 1% FPR"}
STRATEGY_LABELS = {
    "cap_metadata_nearest": "CAP metadata",
    "cap_encoder_nearest": "CAP encoder",
    "cap_cdf": "CAP CDF",
    "cap_random": "CAP random",
    "drift": "Marginal drift",
    "wasserstein": "Wasserstein",
}


def _holm(values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in their original order."""
    raw = np.asarray(values, dtype=float)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(raw) - position) * raw[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _interval(values: pd.Series) -> tuple[float, float, float, float]:
    """Return mean, SD, and a two-sided t interval over reporting seeds."""
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def _load(root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Authenticate frozen checkpoints and complete intervention outcomes."""
    root = root.resolve()
    design = json.loads((root / "design.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoints = pd.DataFrame(frozen["checkpoints"])
    expected_checkpoints = campaign.EXPECTED_TRAJECTORIES * len(rank.STRATEGIES)
    if len(checkpoints) != expected_checkpoints:
        raise ValueError("The checkpoint manifest does not contain exactly 288 branches.")
    for row in frozen["checkpoints"]:
        if campaign._sha256(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {row['checkpoint']}")
    outcomes_path = root / "results" / "evaluation_rows.csv"
    outcomes = pd.read_csv(outcomes_path, dtype={"candidate_id": str})
    outcomes["candidate_id"] = outcomes["candidate_id"].str.zfill(3)
    expected_rows = campaign.EXPECTED_TRAJECTORIES * len(rank.STRATEGIES) * 58 * 2
    keys = ["candidate_id", "reporting_seed", "strategy", "intervention", "metric"]
    if len(outcomes) != expected_rows or outcomes.duplicated(keys).any():
        raise ValueError("Residual-OAS outcomes do not have exact unique coverage.")
    if not np.isfinite(outcomes["value"].to_numpy(dtype=float)).all():
        raise ValueError("Residual-OAS outcomes contain non-finite values.")
    checkpoints["candidate_id"] = checkpoints["candidate_id"].astype(str).str.zfill(3)
    return design, checkpoints, outcomes


def analyze(root: Path, output_dir: Path, permutations: int = 10_000) -> list[Path]:
    """Write outcome-blind selections, performance, rank validity, and figures."""
    design, checkpoints, outcomes = _load(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_first = (
        outcomes.groupby(["candidate_id", "reporting_seed", "strategy", "metric"], sort=True)[
            "value"
        ]
        .mean()
        .reset_index()
    )
    proxy = checkpoints[
        [
            "trajectory_index",
            "candidate_id",
            "reporting_seed",
            "strategy",
            "monitor_value",
            "checkpoint",
            "checkpoint_sha256",
            "selected_epoch",
        ]
    ].copy()

    selections = []
    for strategy in rank.STRATEGIES:
        branch = proxy[proxy.strategy == strategy]
        candidate_proxy = branch.groupby("candidate_id").monitor_value.mean()
        direction = rank.DIRECTIONS[strategy]
        selected_candidate = (
            candidate_proxy.idxmax() if direction == "maximize" else candidate_proxy.idxmin()
        )
        selected_seed_rows = branch[branch.candidate_id == selected_candidate].copy()
        selected_seed_rows = selected_seed_rows.sort_values(
            ["monitor_value", "reporting_seed"],
            ascending=[direction == "minimize", True],
            kind="stable",
        )
        exact = selected_seed_rows.iloc[0]
        selections.append(
            {
                "strategy": strategy,
                "direction": direction,
                "selected_candidate": selected_candidate,
                "candidate_mean_proxy": float(candidate_proxy.loc[selected_candidate]),
                "initialization_seed": int(exact.reporting_seed),
                "initialization_proxy": float(exact.monitor_value),
                "initialization_checkpoint": exact.checkpoint,
                "initialization_checkpoint_sha256": exact.checkpoint_sha256,
                "initialization_epoch": int(exact.selected_epoch),
            }
        )
    selection = pd.DataFrame(selections)
    selected_seed = seed_first.merge(
        selection[["strategy", "selected_candidate"]],
        left_on=["strategy", "candidate_id"],
        right_on=["strategy", "selected_candidate"],
        validate="many_to_one",
    )
    summary_rows = []
    for (strategy, metric), group in selected_seed.groupby(["strategy", "metric"], sort=True):
        mean, sd, low, high = _interval(group.value)
        summary_rows.append(
            {
                "model": "ae",
                "score": "residual_oas",
                "strategy": strategy,
                "metric": metric,
                "mean": mean,
                "sd": sd,
                "ci_low": low,
                "ci_high": high,
                "n_reporting_seeds": group.reporting_seed.nunique(),
                "selected_candidate": group.selected_candidate.iloc[0],
            }
        )
    summary = pd.DataFrame(summary_rows)

    rng = np.random.default_rng(2_608_2026)
    association_rows = []
    for strategy in rank.STRATEGIES:
        direction = rank.DIRECTIONS[strategy]
        candidate_proxy = (
            proxy[proxy.strategy == strategy].groupby("candidate_id").monitor_value.mean()
        )
        utility = candidate_proxy if direction == "maximize" else -candidate_proxy
        for metric in campaign.rank.METRICS:
            candidate_outcome = (
                seed_first[(seed_first.strategy == strategy) & (seed_first.metric == metric)]
                .groupby("candidate_id")
                .value.mean()
                .reindex(utility.index)
            )
            observed = float(stats.spearmanr(utility, candidate_outcome).statistic)
            exceed = 0
            for _ in range(permutations):
                permuted = rng.permutation(candidate_outcome.to_numpy(dtype=float))
                exceed += int(float(stats.spearmanr(utility, permuted).statistic) >= observed)
            association_rows.append(
                {
                    "model": "ae",
                    "score": "residual_oas",
                    "strategy": strategy,
                    "metric": metric,
                    "direction": direction,
                    "spearman_rho": observed,
                    "one_sided_permutation_p": (exceed + 1) / (permutations + 1),
                    "n_candidates": len(utility),
                }
            )
    associations = pd.DataFrame(association_rows)
    for metric in campaign.rank.METRICS:
        indices = associations.index[associations.metric == metric].tolist()
        associations.loc[indices, "holm_p"] = _holm(
            associations.loc[indices, "one_sided_permutation_p"].tolist()
        )

    paths = []
    for name, frame in (
        ("selection_manifest.csv", selection),
        ("seed_first_candidate_performance.csv", seed_first),
        ("selected_strategy_summary.csv", summary),
        ("candidate_rank_associations.csv", associations),
    ):
        path = output_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    positions = np.arange(len(rank.STRATEGIES))
    for axis, metric in zip(axes, campaign.rank.METRICS):
        panel = summary[summary.metric == metric].set_index("strategy").loc[list(rank.STRATEGIES)]
        axis.errorbar(
            positions,
            panel["mean"],
            yerr=np.vstack((panel["mean"] - panel["ci_low"], panel["ci_high"] - panel["mean"])),
            fmt="o",
            color="#0072B2",
            capsize=4,
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(
            positions,
            [STRATEGY_LABELS[value] for value in rank.STRATEGIES],
            rotation=35,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Causal Chamber AE with train-normal residual OAS scoring")
    figure.tight_layout()
    performance_plot = output_dir / "ae_selected_checkpoint_performance.png"
    figure.savefig(performance_plot, dpi=220, bbox_inches="tight")
    plt.close(figure)
    paths.append(performance_plot)

    matrix = associations.pivot(index="strategy", columns="metric", values="spearman_rho").loc[
        list(rank.STRATEGIES), list(campaign.rank.METRICS)
    ]
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center")
    axis.set_yticks(range(len(matrix)), [STRATEGY_LABELS[value] for value in matrix.index])
    axis.set_xticks(range(len(matrix.columns)), [METRIC_LABELS[value] for value in matrix.columns])
    axis.set_title("Residual-OAS proxy rank validity (16 candidates)")
    figure.colorbar(image, ax=axis, label="Spearman rho")
    figure.tight_layout()
    rank_plot = output_dir / "ae_candidate_rank_validity.png"
    figure.savefig(rank_plot, dpi=220, bbox_inches="tight")
    plt.close(figure)
    paths.append(rank_plot)

    primary = selection[selection.strategy == design["primary_strategy"]].iloc[0].to_dict()
    provenance = {
        "schema_version": 1,
        "campaign_design": str((root / "design.json").resolve()),
        "campaign_design_sha256": campaign._sha256(root / "design.json"),
        "checkpoint_manifest_sha256": campaign._sha256(root / "checkpoint_manifest.json"),
        "evaluation_rows_sha256": campaign._sha256(root / "results" / "evaluation_rows.csv"),
        "primary_strategy": design["primary_strategy"],
        "primary_initialization": primary,
        "selection_used_intervention_outcomes": False,
        "n_permutations": permutations,
        "outputs": {path.name: campaign._sha256(path) for path in paths},
    }
    provenance_path = output_dir / "analysis_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths.append(provenance_path)
    return paths


def main() -> None:
    """Run the frozen AE analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    for path in analyze(args.root, args.output_dir, args.permutations):
        print(path)


if __name__ == "__main__":
    main()
