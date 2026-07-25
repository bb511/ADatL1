from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import cchamber_paper_analysis

MODELS = ["ae"]
STRATEGIES = [
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "cap_random",
    "drift",
    "wasserstein",
]
SEEDS = [1001, 1002, 1003, 1004]
INTERVENTIONS = [
    "uniform_red_weak",
    "uniform_red_mid",
    "uniform_red_strong",
    "uniform_t_ir_1_weak",
    "uniform_t_ir_1_mid",
    "uniform_t_ir_1_strong",
]
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
        for left in ("cap_metadata_nearest", "cap_encoder_nearest")
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

    taxonomy_rows = []
    for intervention in INTERVENTIONS:
        color = intervention.startswith("uniform_red")
        taxonomy_rows.append(
            {
                "intervention": intervention,
                "intervention_target": "red" if color else "t_ir_1",
                "strength": intervention.rsplit("_", 1)[-1],
                "semantic_family": "color" if color else "temperature",
                "system_group": "measurement" if color else "process",
            }
        )
    taxonomy_path = tmp_path / "taxonomy.csv"
    pd.DataFrame(taxonomy_rows).to_csv(taxonomy_path, index=False)

    strategy_effect = {
        "cap_metadata_nearest": 0.30,
        "cap_encoder_nearest": 0.25,
        "cap_random": 0.05,
        "drift": 0.00,
        "wasserstein": -0.05,
    }
    strength_effect = {"weak": 0.00, "mid": 0.04, "strong": 0.08}
    result_rows = []
    for model in MODELS:
        for strategy in STRATEGIES:
            for seed_index, seed in enumerate(SEEDS):
                for intervention in INTERVENTIONS:
                    strength = intervention.rsplit("_", 1)[-1]
                    for metric in METRICS:
                        metric_base = 0.25 if metric == "auprc" else 0.35
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
                                    + seed_index * 0.002
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
    assert len(contrasts) == len(MODELS) * len(METRICS) * 6
    assert set(contrasts["n_seeds"]) == {len(SEEDS)}
    assert (contrasts["mean_difference"] > 0).all()
    assert {"p_signflip_holm", "p_sign_holm"}.issubset(contrasts.columns)

    strength = pd.read_csv(bundle["output"] / "within_target_strength_contrasts.csv")
    assert set(zip(strength["higher_strength"], strength["lower_strength"])) == {
        ("mid", "weak"),
        ("strong", "weak"),
        ("strong", "mid"),
    }
    assert (strength["mean_difference"] > 0).all()

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
    assert (bundle["output"] / "confirmatory_contrast_forest.png").stat().st_size > 0
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
