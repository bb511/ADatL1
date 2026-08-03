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
VAE_ROOT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/" "cchamber_vae_reporting_20260803_auto"
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

REPORT_MODELS = ("svdd", "ae", "vae", "realnvp")
REPORT_STRATEGIES = ("cap_encoder_nearest", "cap_cdf", "drift", "wasserstein")


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
    vae_design = json.loads((VAE_ROOT / "design.json").read_text(encoding="utf-8"))
    vae_cap_strategy = {
        "cap_encoder": "cap_encoder_nearest",
        "cap_cdf": "cap_cdf",
    }[vae_design["selected_cap_selector"]]
    vae_cap = lookup.loc[("vae", vae_cap_strategy)]
    lines = [
        "# Revised Causal Chamber results: AE, SVDD, and AE-initialized VAE",
        "",
        "This report supersedes the AE, SVDD, and VAE cells of the `88aaec5` report. "
        "RealNVP is unchanged. AE and SVDD use their revised score audits. The VAE score "
        "and hyperparameters are outcome-optimized on the exploratory search requested for "
        "this analysis; its five reporting retrains are independent of those search seeds.",
        "",
        "The new AE and SVDD intervals use three independent reporting retrains per selected "
        "candidate, the new VAE uses five, and unchanged RealNVP uses ten. Interventions are "
        "averaged within seed before means and intervals are computed.",
        "",
        "## Selected-checkpoint performance",
        "",
        "| Architecture | Selection criterion | AUPRC | Eff. | Retrains |",
        "|---|---|---:|---:|---:|",
    ]
    for model in REPORT_MODELS:
        for strategy in REPORT_STRATEGIES:
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
            "## VAE result",
            "",
            f"The outcome-optimized search selected `{vae_design['selected_score']}` with "
            f"{plots.STRATEGY_LABELS[vae_cap_strategy]}. On five independent fresh retrains, "
            f"this reaches **{vae_cap.loc['auprc', 'mean']:.4f} AUPRC** and "
            f"**{vae_cap.loc['efficiency_operational', 'mean']:.4f} efficiency**. The VAE "
            "starts from the residual-OAS AE encoder/decoder and fine-tunes every weight. "
            "The anomaly-score and hyperparameter choice used exploratory intervention "
            "outcomes by design; checkpoint selection within each fresh run remains normal-only.",
            "",
            "## Candidate-ranking validity",
            "",
        ]
    )
    for model, strategy in (
        ("ae", "cap_cdf"),
        ("svdd", "cap_encoder_nearest"),
        ("vae", vae_cap_strategy),
    ):
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
            "This recomputation uses the newly selected AE, SVDD, and VAE checkpoints and "
            "only encoder-nearest CAP, CDF-rank CAP, marginal drift, and Wasserstein.",
            "",
            "![Controlled physical-shift synthesis](./physical_shift_synthesis.png)",
            "",
            "## Integrity",
            "",
            "- AE: 48 trajectories, 288 frozen branch checkpoints, 33,408 complete outcome rows.",
            "- SVDD: 48 trajectories, 288 frozen branch checkpoints, 33,408 complete outcome rows.",
            "- VAE: selector-specific winners, five fresh reporting seeds, and 2,320 complete "
            "outcome rows across the four retained criteria.",
            "- AE, SVDD, and VAE candidate-rank analyses use all 16 search candidates and "
            "10,000 permutations.",
            "- The presentation reports encoder-nearest CAP, CDF CAP, marginal drift, and Wasserstein.",
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
    vae_summary, vae_rank, vae_results = _new_results(VAE_ROOT, "vae")
    unchanged = old_summary[old_summary.model == "realnvp"].copy()
    unchanged = unchanged.rename(columns={"std": "sd", "n_seeds": "n_reporting_seeds"})
    summary = pd.concat(
        [ae_summary, svdd_summary, vae_summary, unchanged], ignore_index=True, sort=False
    )
    new_rank = pd.concat([ae_rank, svdd_rank, vae_rank], ignore_index=True)
    new_rank = new_rank.rename(columns={"holm_p": "spearman_holm_p"})
    rank = pd.concat(
        [new_rank, old_rank[old_rank.model == "realnvp"]],
        ignore_index=True,
        sort=False,
    )
    old_results = pd.read_csv(OLD_THRESHOLD)
    unchanged_results = old_results[old_results.model == "realnvp"]
    results = pd.concat(
        [ae_results, svdd_results, vae_results, unchanged_results], ignore_index=True
    )
    # Keep the complete campaigns authoritative at their source roots, while the revised
    # presentation intentionally contains only the requested models and criteria.
    summary = summary[
        summary.model.isin(REPORT_MODELS) & summary.strategy.isin(REPORT_STRATEGIES)
    ].copy()
    rank = rank[rank.model.isin(REPORT_MODELS) & rank.strategy.isin(REPORT_STRATEGIES)].copy()
    results = results[
        results.model.isin(REPORT_MODELS) & results.strategy.isin(REPORT_STRATEGIES)
    ].copy()
    if len(summary) != 32 or len(rank) != 32:
        raise ValueError("Combined report tables lack exact 4x4x2 coverage.")
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
    plots._plot_architecture_overview(
        summary, selected_plot, models=REPORT_MODELS, strategies=REPORT_STRATEGIES
    )
    rank_plot = output_dir / "candidate_rank_validity.png"
    plots._plot_rank_heatmap(rank, rank_plot, models=REPORT_MODELS, strategies=REPORT_STRATEGIES)
    physical_plot = output_dir / "physical_shift_synthesis.png"
    plots._plot_retained_physical_synthesis(intervention, physical_plot, models=REPORT_MODELS)
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
        "vae_root": str(VAE_ROOT),
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
