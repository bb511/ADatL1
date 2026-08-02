"""Regenerate the Causal Chamber report with residual-OAS AE and SVDD reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cchamber_final_report as plots
import pandas as pd

OLD_REPORT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/reports/" "cchamber_real_20260801_88aaec5"
)
AE_ROOT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/" "cchamber_ae_residual_oas_20260802_6e05000"
)
SVDD_ROOT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/" "cchamber_svdd_inverse_cap_20260802_seeded"
)
OLD_THRESHOLD = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/audits/"
    "cchamber_real_20260801_3789655_threshold_3789655/results/threshold_safe_results.csv"
)
PHYSICAL = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/audits/"
    "cchamber_real_20260801_50f39f1_physical_shift_89f8c98/results/"
    "physical_shift_magnitude.csv"
)


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_results(root: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return summary, rank, and selected intervention rows for one new campaign."""
    analysis = root / "analysis"
    summary = pd.read_csv(
        analysis / "selected_strategy_summary.csv", dtype={"selected_candidate": str}
    )
    rank = pd.read_csv(analysis / "candidate_rank_associations.csv")
    selection = pd.read_csv(analysis / "selection_manifest.csv", dtype={"selected_candidate": str})
    outcomes = pd.read_csv(root / "results" / "evaluation_rows.csv", dtype={"candidate_id": str})
    for frame, column in ((summary, "selected_candidate"), (selection, "selected_candidate")):
        frame[column] = frame[column].str.zfill(3)
    outcomes["candidate_id"] = outcomes["candidate_id"].str.zfill(3)
    selected = outcomes.merge(
        selection[["strategy", "selected_candidate"]],
        left_on=["strategy", "candidate_id"],
        right_on=["strategy", "selected_candidate"],
        validate="many_to_one",
    )[["strategy", "reporting_seed", "intervention", "metric", "value"]].rename(
        columns={"reporting_seed": "seed"}
    )
    selected.insert(0, "model", model)
    return summary, rank, selected


def _markdown(summary: pd.DataFrame, rank: pd.DataFrame, output: Path) -> None:
    """Write the authoritative human-readable report with renderable image links."""
    lookup = summary.set_index(["model", "strategy", "metric"])
    ae_cdf = lookup.loc[("ae", "cap_cdf")]
    svdd_encoder = lookup.loc[("svdd", "cap_encoder_nearest")]
    lines = [
        "# Revised Causal Chamber results: residual-OAS AE and inverse-CAP SVDD",
        "",
        "This report supersedes the AE and SVDD cells of the `88aaec5` report. VAE and "
        "RealNVP are unchanged. All new selections use validation-normal information only; "
        "the 58 interventions were sealed until every branch checkpoint was hash-frozen.",
        "",
        "The new AE and SVDD intervals use three independent reporting retrains per selected "
        "candidate. The unchanged VAE and RealNVP intervals use ten retrains. Interventions "
        "are averaged within seed before means and intervals are computed.",
        "",
        "## Selected-checkpoint performance",
        "",
        "| Architecture | Selection criterion | AUPRC | Eff. | Retrains |",
        "|---|---|---:|---:|---:|",
    ]
    for model in plots.MODELS:
        for strategy in plots.STRATEGIES:
            auprc = lookup.loc[(model, strategy, "auprc")]
            efficiency = lookup.loc[(model, strategy, "efficiency_operational")]
            lines.append(
                f"| {plots.MODEL_LABELS[model]} | {plots.STRATEGY_LABELS[strategy]} | "
                f"{auprc['mean']:.4f} | {efficiency['mean']:.4f} | "
                f"{int(auprc['n_reporting_seeds'])} |"
            )
    lines.extend(
        [
            "",
            "![Selected-checkpoint performance](./selected_checkpoint_performance.png)",
            "",
            "## AE result",
            "",
            f"Residual-OAS CDF selection reaches **{ae_cdf.loc['auprc', 'mean']:.4f} AUPRC** "
            f"and **{ae_cdf.loc['efficiency_operational', 'mean']:.4f} efficiency**. The old "
            "scalar-MSE CDF cell was 0.5697 / 0.2463. CDF selected candidate `060` using "
            "normal-only proxy values.",
            "",
            "## SVDD result",
            "",
            "The score audit rejected center-OAS/CDF and selected the native radial score "
            "with inverse encoder-nearest CAP. This reaches "
            f"**{svdd_encoder.loc['auprc', 'mean']:.4f} AUPRC** and "
            f"**{svdd_encoder.loc['efficiency_operational', 'mean']:.4f} efficiency**. "
            "SVDD starts from the selected AE candidate-060 CDF encoder weights, drops only "
            "the incompatible AE bias tensors to retain the bias-free SVDD constraint, and "
            "fine-tunes every transferred weight. Per-branch tensor deltas authenticate that "
            "no transferred encoder tensor remained frozen.",
            "",
            "## Candidate-ranking validity",
            "",
        ]
    )
    for model, strategy in (("ae", "cap_cdf"), ("svdd", "cap_encoder_nearest")):
        branch = rank[(rank.model == model) & (rank.strategy == strategy)]
        values = {row.metric: row for row in branch.itertuples()}
        lines.append(
            f"- {plots.MODEL_LABELS[model]} {plots.STRATEGY_LABELS[strategy]}: AUPRC rho "
            f"`{values['auprc'].spearman_rho:.3f}` (Holm p "
            f"`{values['auprc'].spearman_holm_p:.4f}`); efficiency rho "
            f"`{values['efficiency_operational'].spearman_rho:.3f}` (Holm p "
            f"`{values['efficiency_operational'].spearman_holm_p:.4f}`)."
        )
    lines.extend(
        [
            "",
            "![Candidate-ranking validity](./candidate_rank_validity.png)",
            "",
            "## Controlled physical-shift synthesis",
            "",
            "![Controlled physical-shift synthesis](./physical_shift_synthesis.png)",
            "",
            "## Integrity",
            "",
            "- AE: 48 trajectories, 288 frozen branch checkpoints, 33,408 complete outcome rows.",
            "- SVDD: 48 trajectories, 288 frozen branch checkpoints, 33,408 complete outcome rows.",
            "- Both candidate-rank analyses use all 16 candidates and 10,000 permutations.",
            "- The random-pair branch is retained as a negative control; all six criteria are reported.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path, repository_extra: Path | None = None) -> list[Path]:
    """Build combined tables, three canonical plots, Markdown, and provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    old_summary = pd.read_csv(OLD_REPORT / "paper_analysis" / "strategy_summary.csv")
    old_rank = pd.read_csv(OLD_REPORT / "paper_analysis" / "candidate_rank_associations.csv")
    ae_summary, ae_rank, ae_results = _new_results(AE_ROOT, "ae")
    svdd_summary, svdd_rank, svdd_results = _new_results(SVDD_ROOT, "svdd")
    unchanged = old_summary[old_summary.model.isin(("vae", "realnvp"))].copy()
    unchanged = unchanged.rename(columns={"std": "sd", "n_seeds": "n_reporting_seeds"})
    summary = pd.concat([ae_summary, svdd_summary, unchanged], ignore_index=True, sort=False)
    if len(summary) != 48:
        raise ValueError("Combined strategy summary lacks exact 4x6x2 coverage.")
    new_rank = pd.concat([ae_rank, svdd_rank], ignore_index=True)
    new_rank = new_rank.rename(columns={"holm_p": "spearman_holm_p"})
    rank = pd.concat(
        [new_rank, old_rank[old_rank.model.isin(("vae", "realnvp"))]],
        ignore_index=True,
        sort=False,
    )
    if len(rank) != 48:
        raise ValueError("Combined rank table lacks exact 4x6x2 coverage.")
    old_results = pd.read_csv(OLD_THRESHOLD)
    unchanged_results = old_results[old_results.model.isin(("vae", "realnvp"))]
    results = pd.concat([ae_results, svdd_results, unchanged_results], ignore_index=True)
    physical = pd.read_csv(PHYSICAL)
    physical["system_group"] = physical["physical_class"].map(plots.CLASS_MAP)
    intervention = (
        results.groupby(["model", "strategy", "metric", "intervention"], sort=True)
        .value.mean()
        .rename("mean_performance")
        .reset_index()
        .merge(
            physical[
                [
                    "intervention",
                    "target",
                    "strength",
                    "semantic_family",
                    "system_group",
                    "biased_energy_distance",
                ]
            ],
            on="intervention",
            validate="many_to_one",
        )
    )
    summary_path = output_dir / "strategy_summary.csv"
    rank_path = output_dir / "candidate_rank_associations.csv"
    intervention_path = output_dir / "intervention_seed_first_performance.csv"
    summary.to_csv(summary_path, index=False)
    rank.to_csv(rank_path, index=False)
    intervention.to_csv(intervention_path, index=False)
    selected_plot = output_dir / "selected_checkpoint_performance.png"
    plots._plot_architecture_overview(summary, selected_plot)
    rank_plot = output_dir / "candidate_rank_validity.png"
    plots._plot_rank_heatmap(rank, rank_plot)
    physical_plot = output_dir / "physical_shift_synthesis.png"
    plots._plot_physical_synthesis(intervention, physical_plot)
    markdown = output_dir / "results_summary.md"
    _markdown(summary, rank, markdown)
    outputs = [
        summary_path,
        rank_path,
        intervention_path,
        selected_plot,
        rank_plot,
        physical_plot,
        markdown,
    ]
    if repository_extra is not None:
        repository_extra.mkdir(parents=True, exist_ok=True)
        for source, name in (
            (selected_plot, "cchamber_selected_checkpoint_performance.png"),
            (rank_plot, "cchamber_candidate_rank_validity.png"),
            (physical_plot, "cchamber_theorem_bridge.png"),
        ):
            destination = repository_extra / name
            shutil.copy2(source, destination)
            outputs.append(destination)
    provenance = {
        "schema_version": 1,
        "old_report": str(OLD_REPORT),
        "ae_root": str(AE_ROOT),
        "svdd_root": str(SVDD_ROOT),
        "outputs": {str(path): _sha256(path) for path in outputs},
    }
    provenance_path = output_dir / "report_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs.append(provenance_path)
    return outputs


def main() -> None:
    """Parse paths and regenerate the revised report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-extra", type=Path)
    args = parser.parse_args()
    for path in run(args.output_dir, args.repository_extra):
        print(path)


if __name__ == "__main__":
    main()
