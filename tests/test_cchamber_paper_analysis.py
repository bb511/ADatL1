from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scripts import cchamber_paper_analysis

MODELS = ["ae", "vae", "svdd", "realnvp"]
STRATEGIES = [
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "cap_cdf",
    "cap_random",
    "drift",
    "wasserstein",
]
SEEDS = [1001, 1002, 1003, 1004]
FULL_TARGETS = [
    ("process", "flow", "flow_0"),
    ("process", "flow", "flow_1"),
    ("process", "flow", "flow_2"),
    ("process", "temperature", "temperature_0"),
    ("process", "temperature", "temperature_1"),
    ("measurement", "color", "blue"),
    ("measurement", "color", "green"),
    ("measurement", "color", "red"),
    ("measurement", "pressure", "pressure_0"),
]
MID_STRONG_TARGETS = [
    ("process", "speed", "speed_0"),
    ("measurement", "color", "yellow"),
]
TARGETS = [
    *(target + (("weak", "mid", "strong"),) for target in FULL_TARGETS),
    *(target + (("mid", "strong"),) for target in MID_STRONG_TARGETS),
]
TAXONOMY_ROWS = [
    {
        "intervention": f"{target}_{strength}",
        "intervention_target": target,
        "strength": strength,
        "semantic_family": family,
        "system_group": system,
    }
    for system, family, target, strengths in TARGETS
    for strength in strengths
]
INTERVENTIONS = [row["intervention"] for row in TAXONOMY_ROWS]
METRICS = ["auprc", "efficiency_operational"]


def _write_json(path: Path, value: object) -> None:
    """Write a deterministic JSON test fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frozen_bundle(tmp_path: Path, *, with_pairing: bool = True) -> dict[str, Path]:
    """Build a fully synthetic frozen campaign-analysis bundle."""
    campaign_root = tmp_path / "campaign"
    paper = campaign_root / "paper"
    paper.mkdir(parents=True)
    campaign = {
        "schema_version": 1,
        "campaign_id": "unit-test-campaign",
        "models": MODELS,
        "strategies": STRATEGIES,
        "reporting_seeds": SEEDS,
        "interventions": INTERVENTIONS,
    }
    campaign_path = campaign_root / "campaign.json"
    _write_json(campaign_path, campaign)

    contrasts = [
        {
            "id": f"{left}_vs_{right}",
            "family": "cap_vs_baselines",
            "left": left,
            "right": right,
            "alternative": "greater",
        }
        for left in ("cap_metadata_nearest", "cap_encoder_nearest", "cap_cdf")
        for right in ("cap_random", "drift", "wasserstein")
    ]
    plan = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "models": MODELS,
        "strategies": STRATEGIES,
        "reporting_seeds": SEEDS,
        "interventions": INTERVENTIONS,
        "metrics": METRICS,
        "strength_order": ["weak", "mid", "strong"],
        "contrasts": contrasts,
    }
    plan_path = tmp_path / "analysis_plan.json"
    _write_json(plan_path, plan)

    taxonomy_path = tmp_path / "taxonomy.csv"
    pd.DataFrame(reversed(TAXONOMY_ROWS)).to_csv(taxonomy_path, index=False)

    strategy_effect = {
        "cap_metadata_nearest": 0.30,
        "cap_encoder_nearest": 0.295,
        "cap_cdf": 0.29,
        "cap_random": 0.05,
        "drift": 0.00,
        "wasserstein": -0.05,
    }
    strength_effect = {"weak": 0.00, "mid": 0.04, "strong": 0.08}
    taxonomy_by_intervention = {row["intervention"]: row for row in TAXONOMY_ROWS}
    target_effect = {
        target: index * 0.002 for index, (_, _, target) in enumerate(FULL_TARGETS)
    } | {target: 0.15 for _, _, target in MID_STRONG_TARGETS}
    result_rows = []
    for model_index, model in enumerate(MODELS):
        for strategy in STRATEGIES:
            for seed_index, seed in enumerate(SEEDS):
                for intervention in INTERVENTIONS:
                    taxonomy = taxonomy_by_intervention[intervention]
                    strength = taxonomy["strength"]
                    for metric in METRICS:
                        metric_base = 0.20 if metric == "auprc" else 0.30
                        result_rows.append(
                            {
                                "model": model,
                                "strategy": strategy,
                                "seed": seed,
                                "intervention": intervention,
                                "metric": metric,
                                "value": (
                                    metric_base
                                    + strategy_effect[strategy]
                                    + strength_effect[strength]
                                    + target_effect[taxonomy["intervention_target"]]
                                    + seed_index * 0.002
                                    + model_index * 0.001
                                ),
                            }
                        )
    results_path = paper / "results.csv"
    pd.DataFrame(result_rows).to_csv(results_path, index=False)

    optional = {}
    if with_pairing:
        selection = campaign_root / "selection"
        selection.mkdir()
        candidate_rows = []
        sensitivity_rows = []
        for seed in (101, 202, 303):
            for candidate in range(5):
                candidate_id = f"{candidate:03d}"
                for strategy in ("cap_random", "cap_encoder_nearest"):
                    candidate_rows.append(
                        {
                            "model": "ae",
                            "seed": seed,
                            "candidate_id": candidate_id,
                            "strategy": strategy,
                            "value": candidate
                            + (0.1 if strategy == "cap_encoder_nearest" else 0.0),
                        }
                    )
                for variant in ("cap_random_seed271829", "cap_encoder_seed456"):
                    sensitivity_rows.append(
                        {
                            "model": "ae",
                            "seed": seed,
                            "candidate_id": candidate_id,
                            "variant": variant,
                            "value": candidate + 0.01,
                        }
                    )
        candidate_path = selection / "candidate_metrics.csv"
        sensitivity_path = selection / "pairing_proxy_sensitivity.csv"
        pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
        pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
        optional = {
            "candidate_metrics": {
                "path": str(candidate_path.relative_to(campaign_root)),
                "sha256": cchamber_paper_analysis._sha256(candidate_path),
            },
            "pairing_proxy_sensitivity": {
                "path": str(sensitivity_path.relative_to(campaign_root)),
                "sha256": cchamber_paper_analysis._sha256(sensitivity_path),
            },
        }

    integrity = {
        "schema_version": 1,
        "files": {
            "campaign": {
                "path": "campaign.json",
                "sha256": cchamber_paper_analysis._sha256(campaign_path),
            },
            "results": {
                "path": "paper/results.csv",
                "sha256": cchamber_paper_analysis._sha256(results_path),
            },
            "analysis_plan": {
                "path": str(plan_path),
                "sha256": cchamber_paper_analysis._sha256(plan_path),
            },
            "taxonomy": {
                "path": str(taxonomy_path),
                "sha256": cchamber_paper_analysis._sha256(taxonomy_path),
            },
        },
        "optional_artifacts": optional,
    }
    integrity_path = tmp_path / "integrity.json"
    _write_json(integrity_path, integrity)
    return {
        "campaign_root": campaign_root,
        "plan": plan_path,
        "taxonomy": taxonomy_path,
        "integrity": integrity_path,
        "results": results_path,
        "output": tmp_path / "analysis-output",
    }


def test_analysis_is_seed_first_hash_pinned_and_marks_pending_components(tmp_path) -> None:
    """The complete analysis is seed-first, hash-pinned, and status explicit."""
    bundle = _frozen_bundle(tmp_path)

    outputs = cchamber_paper_analysis.analyze(
        bundle["campaign_root"],
        bundle["plan"],
        bundle["taxonomy"],
        bundle["integrity"],
        bundle["output"],
    )

    assert len(outputs) >= 18
    seed_summary = pd.read_csv(bundle["output"] / "seed_first_summary.csv")
    assert len(seed_summary) == len(MODELS) * len(STRATEGIES) * len(SEEDS) * len(METRICS)
    assert set(seed_summary.groupby(["model", "strategy", "metric"]).size()) == {len(SEEDS)}

    contrasts = pd.read_csv(bundle["output"] / "prespecified_strategy_contrasts.csv")
    assert len(contrasts) == len(MODELS) * len(METRICS) * 9
    assert set(contrasts["n_seeds"]) == {len(SEEDS)}
    assert (contrasts["mean_difference"] > 0).all()
    assert {"p_signflip_holm", "p_sign_holm"}.issubset(contrasts.columns)

    equivalence = pd.read_csv(bundle["output"] / "prespecified_metadata_encoder_equivalence.csv")
    assert equivalence["model"].tolist() == MODELS
    assert set(equivalence["holm_family_size"]) == {4}
    assert set(equivalence["equivalence_margin"]) == {0.02}
    assert set(equivalence["ci_multiplicity_adjustment"]) == {"none"}
    assert np.allclose(equivalence["mean_difference"], 0.005)
    assert equivalence["equivalent_tost_holm_0.05"].all()

    equal_family = pd.read_csv(bundle["output"] / "equal_family_seed_summary.csv")
    equal_target = pd.read_csv(bundle["output"] / "equal_target_seed_summary.csv")
    expected_seed_rows = len(MODELS) * len(STRATEGIES) * len(SEEDS) * len(METRICS)
    assert len(equal_family) == expected_seed_rows
    assert len(equal_target) == expected_seed_rows
    equal_target_value = equal_target[
        (equal_target["model"] == "ae")
        & (equal_target["strategy"] == "cap_metadata_nearest")
        & (equal_target["metric"] == "auprc")
        & (equal_target["seed"] == 1001)
    ]["value"].item()
    expected_equal_target = 0.5 + (9 * 0.048 + 2 * 0.21) / 11
    assert equal_target_value == pytest.approx(expected_equal_target)

    strength = pd.read_csv(bundle["output"] / "within_target_strength_contrasts.csv")
    assert set(zip(strength["higher_strength"], strength["lower_strength"])) == {
        ("mid", "weak"),
        ("strong", "weak"),
        ("strong", "mid"),
    }
    assert (strength["mean_difference"] > 0).all()

    eligibility = pd.read_csv(bundle["output"] / "strength_target_eligibility.csv")
    assert eligibility["panel"].value_counts().to_dict() == {
        "mid_strong_all": 11,
        "complete_weak_mid_strong": 9,
    }
    panel_seed = pd.read_csv(bundle["output"] / "strength_panel_equal_target_seed_summary.csv")
    assert set(panel_seed.loc[panel_seed["panel"] == "complete_weak_mid_strong", "n_targets"]) == {
        9
    }
    assert set(panel_seed.loc[panel_seed["panel"] == "mid_strong_all", "n_targets"]) == {11}
    complete_value = panel_seed[
        (panel_seed["model"] == "ae")
        & (panel_seed["strategy"] == "cap_metadata_nearest")
        & (panel_seed["metric"] == "auprc")
        & (panel_seed["seed"] == 1001)
        & (panel_seed["panel"] == "complete_weak_mid_strong")
        & (panel_seed["strength"] == "weak")
    ]["value"].item()
    complete_mid_value = panel_seed[
        (panel_seed["model"] == "ae")
        & (panel_seed["strategy"] == "cap_metadata_nearest")
        & (panel_seed["metric"] == "auprc")
        & (panel_seed["seed"] == 1001)
        & (panel_seed["panel"] == "complete_weak_mid_strong")
        & (panel_seed["strength"] == "mid")
    ]["value"].item()
    mid_strong_value = panel_seed[
        (panel_seed["model"] == "ae")
        & (panel_seed["strategy"] == "cap_metadata_nearest")
        & (panel_seed["metric"] == "auprc")
        & (panel_seed["seed"] == 1001)
        & (panel_seed["panel"] == "mid_strong_all")
        & (panel_seed["strength"] == "mid")
    ]["value"].item()
    assert complete_value == pytest.approx(0.508)
    assert complete_mid_value == pytest.approx(0.548)
    assert complete_mid_value - complete_value == pytest.approx(0.04)
    assert mid_strong_value == pytest.approx((9 * 0.548 + 2 * 0.69) / 11)

    systems = pd.read_csv(bundle["output"] / "process_measurement_summary.csv")
    assert set(systems["system_group"]) == {"process", "measurement"}
    robustness = pd.read_csv(bundle["output"] / "pairing_robustness_summary.csv")
    assert np.allclose(robustness["spearman_mean"], 1.0)
    assert np.allclose(robustness["winner_agreement_rate"], 1.0)

    status = json.loads((bundle["output"] / "component_status.json").read_text())
    assert status["components"]["background_acceptance_diagnostics"]["status"] == "pending"
    assert status["components"]["candidate_audit_results"]["status"] == "pending"
    assert status["components"]["pairing_robustness_analysis"]["status"] == "completed"

    catalog = pd.read_csv(bundle["output"] / "analysis_catalog.csv")
    exploratory = catalog[catalog["artifact"].str.startswith("exploratory_")]
    assert not exploratory.empty
    assert set(exploratory["classification"]) == {"exploratory_outcome_selected"}
    main_heatmap = "prespecified_intervention_cap_minus_baseline_heatmaps.png"
    assert catalog.loc[catalog["artifact"] == main_heatmap, "classification"].item() != (
        "exploratory_outcome_selected"
    )
    intervention_summary = pd.read_csv(bundle["output"] / "intervention_cap_baseline_summary.csv")
    physical_order = (
        intervention_summary[["taxonomy_order", "intervention"]]
        .drop_duplicates()
        .sort_values("taxonomy_order")["intervention"]
        .tolist()
    )
    expected_physical_order = [
        row["intervention"]
        for row in sorted(
            TAXONOMY_ROWS,
            key=lambda row: (
                {"process": 0, "measurement": 1}[row["system_group"]],
                row["semantic_family"],
                row["intervention_target"],
                {"weak": 0, "mid": 1, "strong": 2}[row["strength"]],
                row["intervention"],
            ),
        )
    ]
    assert physical_order == expected_physical_order
    assert (bundle["output"] / "confirmatory_contrast_forest.png").stat().st_size > 0
    assert (bundle["output"] / main_heatmap).stat().st_size > 0
    assert (bundle["output"] / "analysis_provenance.json").is_file()


def test_analysis_rejects_hash_mismatch_before_creating_output(tmp_path) -> None:
    """A changed outcome file is rejected before any analysis output is created."""
    bundle = _frozen_bundle(tmp_path)
    with bundle["results"].open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch for results"):
        cchamber_paper_analysis.analyze(
            bundle["campaign_root"],
            bundle["plan"],
            bundle["taxonomy"],
            bundle["integrity"],
            bundle["output"],
        )
    assert not bundle["output"].exists()


def test_analysis_integrates_background_and_candidate_rank_audits(tmp_path) -> None:
    """Optional operating-point and rank audits become validated analysis tables."""
    bundle = _frozen_bundle(tmp_path)
    diagnostics_path = tmp_path / "threshold-sidecar" / "seed_level_operating_point.csv"
    diagnostics_path.parent.mkdir()
    diagnostics_rows = []
    manifest_index = 0
    for model in MODELS:
        for strategy in STRATEGIES:
            for seed_index, seed in enumerate(SEEDS):
                triggered = 9 + seed_index
                achieved = triggered / 1_000
                diagnostics_rows.append(
                    {
                        "model": model,
                        "strategy": strategy,
                        "seed": seed,
                        "manifest_index": manifest_index,
                        "test_normal_count": 1_000,
                        "triggered_count": triggered,
                        "achieved_test_normal_acceptance": achieved,
                        "target_fpr": 0.01,
                        "achieved_minus_target_fpr": achieved - 0.01,
                        "wilson_95_ci_low": max(0.0, achieved - 0.005),
                        "wilson_95_ci_high": min(1.0, achieved + 0.005),
                    }
                )
                manifest_index += 1
    pd.DataFrame(diagnostics_rows).to_csv(diagnostics_path, index=False)

    rank_path = tmp_path / "candidate-audit" / "analysis" / "rank_associations.csv"
    rank_path.parent.mkdir(parents=True)
    rank_rows = []
    for metric in METRICS:
        for model in MODELS:
            for strategy in STRATEGIES:
                rank_rows.append(
                    {
                        "metric": metric,
                        "model": model,
                        "strategy": strategy,
                        "spearman_rho": 0.4,
                        "spearman_permutation_p": 0.02,
                        "spearman_holm_p": 0.4,
                        "kendall_tau_b": 0.3,
                        "top_k": 4,
                        "top_k_overlap": 2,
                        "top_k_enrichment": 0.05,
                        "top_k_oracle_regret": 0.01,
                        "proxy_best_regret": 0.02,
                        "bootstrap_spearman_ci_low": 0.1,
                        "bootstrap_spearman_ci_high": 0.7,
                        "n_permutations": 10_000,
                        "n_bootstrap_requested": 10_000,
                        "n_bootstrap_effective": 9_900,
                        "n_bootstrap_effective_paired": 9_800,
                        "holm_family_size": 24,
                    }
                )
    pd.DataFrame(rank_rows).to_csv(rank_path, index=False)
    rank_provenance = rank_path.with_name("rank_analysis_provenance.json")
    _write_json(
        rank_provenance,
        {
            "schema_version": 1,
            "n_permutations": 10_000,
            "n_bootstrap_requested": 10_000,
            "outputs": {rank_path.name: cchamber_paper_analysis._sha256(rank_path)},
        },
    )

    integrity = json.loads(bundle["integrity"].read_text())
    integrity["optional_artifacts"].update(
        {
            "background_acceptance_diagnostics": {
                "path": str(diagnostics_path),
                "sha256": cchamber_paper_analysis._sha256(diagnostics_path),
            },
            "candidate_audit_results": {
                "path": str(rank_path),
                "sha256": cchamber_paper_analysis._sha256(rank_path),
            },
            "candidate_audit_provenance": {
                "path": str(rank_provenance),
                "sha256": cchamber_paper_analysis._sha256(rank_provenance),
            },
        }
    )
    _write_json(bundle["integrity"], integrity)

    cchamber_paper_analysis.analyze(
        bundle["campaign_root"],
        bundle["plan"],
        bundle["taxonomy"],
        bundle["integrity"],
        bundle["output"],
    )
    background = pd.read_csv(bundle["output"] / "background_acceptance_summary.csv")
    assert len(background) == len(MODELS) * len(STRATEGIES)
    assert not background["event_pooling_for_inference"].any()
    rank = pd.read_csv(bundle["output"] / "candidate_rank_associations.csv")
    assert len(rank) == len(METRICS) * len(MODELS) * len(STRATEGIES)
    status = json.loads((bundle["output"] / "component_status.json").read_text())
    assert status["components"]["background_acceptance_analysis"]["status"] == "completed"
    assert status["components"]["candidate_rank_analysis"]["status"] == "completed"
    provenance = json.loads((bundle["output"] / "analysis_provenance.json").read_text())
    assert "candidate_audit_provenance" in provenance["inputs"]


def test_analysis_rejects_incomplete_result_coverage(tmp_path) -> None:
    """Missing factorial coverage is rejected even when its hash is updated."""
    bundle = _frozen_bundle(tmp_path)
    results = pd.read_csv(bundle["results"]).iloc[:-1]
    results.to_csv(bundle["results"], index=False)
    integrity = json.loads(bundle["integrity"].read_text())
    integrity["files"]["results"]["sha256"] = cchamber_paper_analysis._sha256(bundle["results"])
    _write_json(bundle["integrity"], integrity)

    with pytest.raises(ValueError, match="exact frozen coverage"):
        cchamber_paper_analysis.analyze(
            bundle["campaign_root"],
            bundle["plan"],
            bundle["taxonomy"],
            bundle["integrity"],
            bundle["output"],
        )


def test_threshold_safe_results_require_sidecar_integrity_chain(tmp_path) -> None:
    """Threshold provenance columns require pinned manifest and collection records."""
    bundle = _frozen_bundle(tmp_path)
    results = pd.read_csv(bundle["results"])
    results["manifest_index"] = 0
    results["checkpoint_sha256"] = "checkpoint"
    results["threshold_manifest_sha256"] = "threshold-manifest"
    results["threshold_artifact"] = "/frozen/thresholds/000.json"
    results["threshold_artifact_sha256"] = "threshold-artifact"
    results["threshold_bytes_sha256"] = "threshold-bytes"
    results.to_csv(bundle["results"], index=False)
    integrity = json.loads(bundle["integrity"].read_text())
    integrity["files"]["results"]["sha256"] = cchamber_paper_analysis._sha256(bundle["results"])
    _write_json(bundle["integrity"], integrity)

    with pytest.raises(ValueError, match="require integrity entries"):
        cchamber_paper_analysis.analyze(
            bundle["campaign_root"],
            bundle["plan"],
            bundle["taxonomy"],
            bundle["integrity"],
            bundle["output"],
        )


def test_analysis_refuses_to_write_inside_campaign_root(tmp_path) -> None:
    """Analysis never writes into the immutable campaign tree."""
    bundle = _frozen_bundle(tmp_path, with_pairing=False)

    with pytest.raises(ValueError, match="outside the immutable campaign root"):
        cchamber_paper_analysis.analyze(
            bundle["campaign_root"],
            bundle["plan"],
            bundle["taxonomy"],
            bundle["integrity"],
            bundle["campaign_root"] / "analysis",
        )


def test_exact_paired_tests_and_holm() -> None:
    """Exact paired tests and Holm adjustment match small known examples."""
    assert cchamber_paper_analysis._exact_sign_flip_greater([1.0, 1.0, 1.0]) == 0.125
    sign_p, positives, nonzero = cchamber_paper_analysis._exact_sign_test_greater(
        [1.0, 1.0, -1.0, 0.0]
    )
    assert positives == 2
    assert nonzero == 3
    assert sign_p == pytest.approx(0.5)
    assert cchamber_paper_analysis._holm_adjust([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.06, 0.06]
    )

    differences = {
        "ae": np.array([0.000, 0.005, 0.010, 0.015]),
        "vae": np.array([0.002, 0.004, 0.006, 0.008]),
        "svdd": np.array([0.010, 0.012, 0.014, 0.016]),
        "realnvp": np.array([-0.010, -0.012, -0.014, -0.016]),
    }
    rows = []
    for model, model_differences in differences.items():
        for seed, difference in zip(SEEDS, model_differences):
            rows.extend(
                [
                    {
                        "model": model,
                        "strategy": "cap_metadata_nearest",
                        "seed": seed,
                        "metric": "auprc",
                        "value": 0.5 + difference,
                    },
                    {
                        "model": model,
                        "strategy": "cap_encoder_nearest",
                        "seed": seed,
                        "metric": "auprc",
                        "value": 0.5,
                    },
                ]
            )
    equivalence = cchamber_paper_analysis._metadata_encoder_equivalence(
        pd.DataFrame(rows),
        {"models": MODELS, "reporting_seeds": SEEDS},
    )
    ae = equivalence[equivalence["model"] == "ae"].iloc[0]
    ae_differences = differences["ae"]
    standard_error = ae_differences.std(ddof=1) / np.sqrt(len(SEEDS))
    critical = stats.t.ppf(0.95, len(SEEDS) - 1)
    expected_p = max(
        stats.t.sf(
            (ae_differences.mean() + 0.02) / standard_error,
            len(SEEDS) - 1,
        ),
        stats.t.cdf(
            (ae_differences.mean() - 0.02) / standard_error,
            len(SEEDS) - 1,
        ),
    )
    assert ae["p_tost_unadjusted"] == pytest.approx(expected_p)
    assert ae["ci90_unadjusted_low"] == pytest.approx(
        ae_differences.mean() - critical * standard_error
    )
    assert ae["ci90_unadjusted_high"] == pytest.approx(
        ae_differences.mean() + critical * standard_error
    )
    assert equivalence["p_tost_holm_four_models"].tolist() == pytest.approx(
        cchamber_paper_analysis._holm_adjust(equivalence["p_tost_unadjusted"])
    )
