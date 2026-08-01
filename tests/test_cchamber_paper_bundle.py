from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from scripts import cchamber_paper_bundle as bundle

MODELS = ["ae", "vae", "svdd", "realnvp"]
STRATEGIES = [
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "cap_cdf",
    "cap_random",
    "drift",
    "wasserstein",
]
REPORTING_SEEDS = list(range(1001, 1011))
DEVELOPMENT_SEEDS = [101, 202, 303, 404, 505]
INTERVENTIONS = [f"intervention_{index:02d}" for index in range(58)]
METRICS = ["auprc", "efficiency_operational"]


@pytest.fixture(autouse=True)
def _clean_analysis_checkout(monkeypatch) -> None:
    """Keep bundle unit tests independent of unrelated worktree changes."""
    monkeypatch.setattr(bundle, "_git_commit", lambda: "a" * 40)


VARIANTS = ["cap_random_seed271829", "cap_encoder_seed456"]


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, with_rank: bool = False) -> dict[str, Path]:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir(parents=True)
    campaign = {
        "schema_version": 1,
        "campaign_id": "paper-bundle-unit",
        "models": MODELS,
        "strategies": STRATEGIES,
        "reporting_seeds": REPORTING_SEEDS,
        "development_seeds": DEVELOPMENT_SEEDS,
        "interventions": INTERVENTIONS,
        "sensitivity_design": {"metric_definitions": {variant: {} for variant in VARIANTS}},
    }
    campaign_path = campaign_root / "campaign.json"
    _json(campaign_path, campaign)

    plan = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "models": MODELS,
        "strategies": STRATEGIES,
        "reporting_seeds": REPORTING_SEEDS,
        "interventions": INTERVENTIONS,
        "metrics": METRICS,
        "strength_order": ["weak", "mid", "strong"],
        "contrasts": [
            {
                "id": f"{left}_vs_{right}",
                "family": "cap_vs_baselines",
                "left": left,
                "right": right,
                "alternative": "greater",
            }
            for left in ("cap_metadata_nearest", "cap_encoder_nearest", "cap_cdf")
            for right in ("cap_random", "drift", "wasserstein")
        ],
    }
    plan_path = tmp_path / "analysis_plan.json"
    _json(plan_path, plan)
    taxonomy_path = tmp_path / "taxonomy.csv"
    pd.DataFrame(
        [
            {
                "intervention": intervention,
                "intervention_target": f"target_{index // 2:02d}",
                "strength": ("weak", "mid", "strong")[index % 3],
                "semantic_family": f"family_{index % 4}",
                "system_group": "process" if index < 29 else "measurement",
            }
            for index, intervention in enumerate(INTERVENTIONS)
        ]
    ).to_csv(taxonomy_path, index=False)

    inventory_path = tmp_path / "threshold" / "inventory.json"
    _json(inventory_path, {"schema_version": 1, "records": []})
    threshold_records = []
    index_by_identity = {}
    for index, (model, strategy, seed) in enumerate(product(MODELS, STRATEGIES, REPORTING_SEEDS)):
        artifact = tmp_path / "threshold" / "artifacts" / f"{index:03d}.json"
        checkpoint_sha = f"{index:064x}"
        threshold_bytes_sha = f"{index + 1000:064x}"
        _json(
            artifact,
            {
                "schema_version": 1,
                "manifest_index": index,
                "checkpoint_sha256": checkpoint_sha,
                "threshold_float32": {"bytes_sha256": threshold_bytes_sha},
            },
        )
        threshold_records.append(
            {
                "manifest_index": index,
                "threshold_artifact": str(artifact.resolve()),
                "threshold_artifact_sha256": bundle._sha256(artifact),
                "checkpoint_sha256": checkpoint_sha,
                "threshold_bytes_sha256": threshold_bytes_sha,
            }
        )
        index_by_identity[(model, strategy, seed)] = index
    threshold_manifest_path = tmp_path / "threshold" / "threshold_manifest.json"
    _json(
        threshold_manifest_path,
        {
            "schema_version": 1,
            "test_or_intervention_data_loaded_before_freeze": False,
            "inventory": str(inventory_path.resolve()),
            "inventory_sha256": bundle._sha256(inventory_path),
            "expected_records": 240,
            "records": threshold_records,
        },
    )
    threshold_manifest_sha = bundle._sha256(threshold_manifest_path)
    result_rows = []
    for model, strategy, seed, intervention, metric in product(
        MODELS,
        STRATEGIES,
        REPORTING_SEEDS,
        INTERVENTIONS,
        METRICS,
    ):
        index = index_by_identity[(model, strategy, seed)]
        threshold = threshold_records[index]
        result_rows.append(
            {
                "model": model,
                "strategy": strategy,
                "seed": seed,
                "intervention": intervention,
                "metric": metric,
                "value": 0.5,
                "manifest_index": index,
                "threshold_manifest_sha256": threshold_manifest_sha,
                **{
                    key: threshold[key]
                    for key in (
                        "checkpoint_sha256",
                        "threshold_artifact",
                        "threshold_artifact_sha256",
                        "threshold_bytes_sha256",
                    )
                },
            }
        )
    results_path = tmp_path / "threshold" / "results" / "threshold_safe_results.csv"
    results_path.parent.mkdir(parents=True)
    pd.DataFrame(result_rows).to_csv(results_path, index=False)

    diagnostics_path = tmp_path / "threshold" / "results" / "seed_level_operating_point.csv"
    diagnostic_rows = []
    for model, strategy, seed in product(MODELS, STRATEGIES, REPORTING_SEEDS):
        index = index_by_identity[(model, strategy, seed)]
        diagnostic_rows.append(
            {
                "model": model,
                "strategy": strategy,
                "seed": seed,
                "manifest_index": index,
                "test_normal_count": 1000,
                "triggered_count": 10,
                "achieved_test_normal_acceptance": 0.01,
                "target_fpr": 0.01,
                "achieved_minus_target_fpr": 0.0,
                "wilson_95_ci_low": 0.005,
                "wilson_95_ci_high": 0.018,
            }
        )
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_path, index=False)
    threshold_provenance_path = (
        tmp_path / "threshold" / "results" / "threshold_safe_provenance.json"
    )
    _json(
        threshold_provenance_path,
        {
            "schema_version": 1,
            "inventory": str(inventory_path.resolve()),
            "inventory_sha256": bundle._sha256(inventory_path),
            "threshold_manifest": str(threshold_manifest_path.resolve()),
            "threshold_manifest_sha256": threshold_manifest_sha,
            "results": str(results_path.resolve()),
            "results_sha256": bundle._sha256(results_path),
            "seed_level_summary": str(diagnostics_path.resolve()),
            "seed_level_summary_sha256": bundle._sha256(diagnostics_path),
            "expected_records": 240,
            "expected_result_rows": 27_840,
        },
    )

    candidate_path = campaign_root / "selection" / "candidate_metrics.csv"
    candidate_path.parent.mkdir()
    candidate_rows = [
        {
            "model": model,
            "strategy": strategy,
            "seed": seed,
            "candidate_id": "000",
            "value": 0.1,
        }
        for model, strategy, seed in product(MODELS, STRATEGIES, DEVELOPMENT_SEEDS)
    ]
    pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
    sensitivity_path = campaign_root / "selection" / "pairing_proxy_sensitivity.csv"
    sensitivity_rows = [
        {
            "model": model,
            "seed": seed,
            "candidate_id": "000",
            "variant": variant,
            "value": 0.1,
        }
        for model, seed, variant in product(MODELS, DEVELOPMENT_SEEDS, VARIANTS)
    ]
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    candidate_provenance_path = campaign_root / "selection" / "candidate_metrics_provenance.json"
    _json(
        candidate_provenance_path,
        {
            "schema_version": 1,
            "campaign": str(campaign_path.resolve()),
            "campaign_sha256": bundle._sha256(campaign_path),
            "candidate_metrics": str(candidate_path.resolve()),
            "candidate_metrics_sha256": bundle._sha256(candidate_path),
            "pairing_proxy_sensitivity": str(sensitivity_path.resolve()),
            "pairing_proxy_sensitivity_sha256": bundle._sha256(sensitivity_path),
            "n_rows": len(candidate_rows),
        },
    )

    fixture = {
        "campaign_root": campaign_root,
        "plan": plan_path,
        "taxonomy": taxonomy_path,
        "results": results_path,
        "threshold_manifest": threshold_manifest_path,
        "threshold_provenance": threshold_provenance_path,
        "candidate_metrics": candidate_path,
        "candidate_provenance": candidate_provenance_path,
        "sensitivity": sensitivity_path,
        "diagnostics": diagnostics_path,
        "bundle_dir": tmp_path / "paper-bundle",
        "analysis_output": tmp_path / "paper-analysis",
    }
    if with_rank:
        rank_root = tmp_path / "rank"
        outcomes = rank_root / "sealed_candidate_outcomes.csv"
        outcomes.parent.mkdir()
        outcomes.write_text("outcome\n", encoding="utf-8")
        outcome_provenance = rank_root / "sealed_candidate_outcomes_provenance.json"
        _json(
            outcome_provenance,
            {
                "combined": str(outcomes.resolve()),
                "combined_sha256": bundle._sha256(outcomes),
            },
        )
        associations = rank_root / "rank_associations.csv"
        pd.DataFrame(
            [
                {
                    "metric": metric,
                    "model": model,
                    "strategy": strategy,
                    "spearman_rho": 0.2,
                    "spearman_permutation_p": 0.1,
                    "spearman_holm_p": 0.5,
                    "holm_family_size": 24,
                }
                for metric, model, strategy in product(METRICS, MODELS, STRATEGIES)
            ]
        ).to_csv(associations, index=False)
        rank_provenance = rank_root / "rank_analysis_provenance.json"
        _json(
            rank_provenance,
            {
                "candidate_metrics": str(candidate_path.resolve()),
                "candidate_metrics_sha256": bundle._sha256(candidate_path),
                "outcomes": str(outcomes.resolve()),
                "outcomes_sha256": bundle._sha256(outcomes),
                "outcome_provenance": str(outcome_provenance.resolve()),
                "outcome_provenance_sha256": bundle._sha256(outcome_provenance),
                "outputs": {associations.name: bundle._sha256(associations)},
            },
        )
        fixture["rank_associations"] = associations
        fixture["rank_provenance"] = rank_provenance
    return fixture


def _build(fixture: dict[str, Path]) -> tuple[Path, Path]:
    return bundle.build_bundle(
        campaign_root=fixture["campaign_root"],
        analysis_plan=fixture["plan"],
        taxonomy=fixture["taxonomy"],
        threshold_results=fixture["results"],
        threshold_manifest=fixture["threshold_manifest"],
        threshold_provenance=fixture["threshold_provenance"],
        candidate_metrics=fixture["candidate_metrics"],
        candidate_metrics_provenance=fixture["candidate_provenance"],
        pairing_sensitivity=fixture["sensitivity"],
        background_diagnostics=fixture["diagnostics"],
        bundle_dir=fixture["bundle_dir"],
        analysis_output=fixture["analysis_output"],
        rank_associations=fixture.get("rank_associations"),
        rank_provenance=fixture.get("rank_provenance"),
    )


def test_builds_hash_pinned_manifest_and_cpu_debug_launcher(tmp_path) -> None:
    fixture = _fixture(tmp_path, with_rank=True)
    manifest_path, launcher_path = _build(fixture)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_contract"] == {
        "metrics": METRICS,
        "outcome_values_summarized_or_compared_by_builder": False,
        "rank_analysis": "available",
        "required_result_rows": 27_840,
        "required_selected_checkpoints": 240,
    }
    assert set(manifest["files"]) == {
        "campaign",
        "results",
        "analysis_plan",
        "taxonomy",
        "threshold_manifest",
        "threshold_safe_provenance",
    }
    assert {
        "candidate_metrics",
        "candidate_metrics_provenance",
        "pairing_proxy_sensitivity",
        "background_acceptance_diagnostics",
        "candidate_audit_results",
        "candidate_audit_provenance",
    } == set(manifest["optional_artifacts"])
    for section in ("files", "optional_artifacts"):
        for record in manifest[section].values():
            assert bundle._sha256(Path(record["path"])) == record["sha256"]

    launcher = launcher_path.read_text(encoding="utf-8")
    assert "#SBATCH --account=a0166" in launcher
    assert "#SBATCH --partition=debug" in launcher
    assert "--cpus-per-task=16" in launcher
    assert "--gres" not in launcher
    assert 'CUDA_VISIBLE_DEVICES=""' in launcher
    assert "scripts/cchamber_paper_analysis.py" in launcher
    assert str(manifest_path) in launcher
    assert bundle._sha256(manifest_path) in launcher
    assert str(fixture["analysis_output"]) in launcher

    # The builder is reproducible and accepts a byte-identical existing bundle.
    assert _build(fixture) == (manifest_path, launcher_path)


def test_rejects_broken_threshold_and_rank_provenance_before_emitting(tmp_path) -> None:
    fixture = _fixture(tmp_path, with_rank=True)
    provenance = json.loads(fixture["threshold_provenance"].read_text(encoding="utf-8"))
    provenance["results_sha256"] = "0" * 64
    _json(fixture["threshold_provenance"], provenance)
    with pytest.raises(ValueError, match="Threshold-safe provenance"):
        _build(fixture)
    assert not fixture["bundle_dir"].exists()

    fixture = _fixture(tmp_path / "rank-case", with_rank=True)
    rank = pd.read_csv(fixture["rank_associations"])
    rank = rank.iloc[:-1]
    rank.to_csv(fixture["rank_associations"], index=False)
    rank_provenance = json.loads(fixture["rank_provenance"].read_text(encoding="utf-8"))
    rank_provenance["outputs"][fixture["rank_associations"].name] = bundle._sha256(
        fixture["rank_associations"]
    )
    _json(fixture["rank_provenance"], rank_provenance)
    with pytest.raises(ValueError, match="Rank association|Candidate-rank association"):
        _build(fixture)
    assert not fixture["bundle_dir"].exists()
