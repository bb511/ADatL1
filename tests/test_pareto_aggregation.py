"""Synthetic acceptance tests for the lean Phase 2 Pareto collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.evaluation.leakage_probe.constants import LEAKAGE_PROBE_PROTOCOL_VERSION
from src.evaluation.leakage_probe.provenance import leakage_probe_configuration_id
from src.evaluation.pareto_aggregation import (
    ParetoCollectionError,
    collect_and_write_pareto_study,
    collect_pareto_study,
)


SEED = 10
STUDY_ID = "synthetic-fet-study"
PROTOCOL_VERSION = "fet-et-pareto-v2"


def _configuration_id(gamma: float, architecture_id: str = "h64_32") -> str:
    return (
        f"{STUDY_ID}__gamma-{gamma}__bins-50__arch-{architecture_id}"
    )


def _manifest(
    *,
    configuration_id: str,
    gamma: float,
    seed: int,
    architecture_id: str = "h64_32",
) -> dict[str, Any]:
    return {
        "seed": seed,
        "test": False,
        "algorithm": {
            "mi_gamma": gamma,
            "mi_sensitive_num_bins": 50,
            "encoder": {"nodes": [64, 32, 8]},
        },
        "evaluation": {
            "leakage_probes": {
                "enabled": True,
                "mode": "validation",
                "smoke_test": {"enabled": False},
            }
        },
        "pareto_study": {
            "enabled": True,
            "study_id": STUDY_ID,
            "protocol_version": PROTOCOL_VERSION,
            "configuration_id": configuration_id,
            "sensitive_variable": {"raw_target": "FET.Et"},
            "search_space": {
                "regularized": {"architectures": {"h64_32": [64, 32, 8]}},
                "gamma_zero_baseline": {"architectures": {"h64_32": [64, 32, 8]}},
            },
            "candidate": {
                "autoencoder_seed": seed,
                "mi_gamma": gamma,
                "mi_sensitive_num_bins": 50,
                "architecture_id": architecture_id,
                "encoder_nodes": [64, 32, 8],
            },
            "collapse_constraint": {
                "rule": {
                    "minimum_joint_code_entropy_bits": 1.0,
                    "minimum_fraction_of_paired_gamma_zero_joint_entropy": 0.5,
                }
            },
            "minimum_efficiency_constraint": {
                "max_relative_degradation": 0.05,
            },
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(
    tmp_path: Path,
    *,
    configuration_id: str,
    gamma: float,
    seed: int = SEED,
    leakage: float | None = 0.1,
    collapsed: bool = False,
    median_efficiency: float = 0.7,
    min_efficiency: float = 0.5,
) -> dict[str, str | int]:
    manifest = _manifest(
        configuration_id=configuration_id,
        gamma=gamma,
        seed=seed,
    )
    manifest_path = tmp_path / "hydra" / configuration_id / str(seed) / "pareto_manifest.resolved.yaml"
    _write_json(manifest_path, manifest)

    checkpoint_dir = tmp_path / "checkpoints" / configuration_id / str(seed)
    algorithm_hash = leakage_probe_configuration_id(manifest["algorithm"])
    probe_valid = leakage is not None
    _write_json(
        checkpoint_dir / "plots/val/loss_total/probes/leakage_probes.json",
        {
            "leakage_probe_protocol_version": LEAKAGE_PROBE_PROTOCOL_VERSION,
            "probe_valid": probe_valid,
            "rejection_reason": None if probe_valid else "synthetic_probe_failure",
            "leakage_worst": leakage,
            "run": {
                "autoencoder_seed": seed,
                "configuration_id": algorithm_hash,
            },
            "evaluation": {
                "mode": "validation",
                "purpose": "scientific",
                "reporting_eligible": True,
                "development_data": {
                    "source_splits": ["train"],
                    "n_events": 10,
                    "sample_seed": 12345,
                    "max_samples": None,
                    "event_manifest_hash": "train-manifest",
                    "data_cache_id": "cache-a",
                },
                "held_out_data": {
                    "source_splits": ["valid"],
                    "n_events": 8,
                    "sample_seed": 12345,
                    "max_samples": None,
                    "event_manifest_hash": "valid-manifest",
                    "data_cache_id": "cache-a",
                },
            },
        },
    )
    _write_json(
        checkpoint_dir / "plots/val/loss_total/eff/eff_summary.json",
        {
            "checkpoint": "loss_total.ckpt",
            "split": "val",
            "mean_efficiency": median_efficiency,
            "median_efficiency": median_efficiency,
            "min_efficiency": min_efficiency,
            "cvar25_efficiency": min_efficiency,
            "signal_efficiencies": {
                "signal_a": min_efficiency,
                "signal_b": median_efficiency,
            },
        },
    )
    _write_json(
        checkpoint_dir
        / "plots/val/loss_total/correlation_matrix/normal/mean_correlations.json",
        {
            "sensitive_variable": "FET.Et",
            "C": 0.2,
            "mean_pearson_correlation": 0.15,
            "mean_spearman_correlation": 0.2,
        },
    )
    _write_json(
        checkpoint_dir / "plots/val/loss_total/latent_collapse/collapse_summary.json",
        {
            "checkpoint": "loss_total.ckpt",
            "split": "val",
            "dataset": "normal",
            "representation": {"name": "latent_sample"},
            "metrics": {
                "joint_code_entropy_bits": 0.8 if collapsed else 2.0,
                "summed_marginal_bit_entropy_bits": 3.0,
                "effective_code_count": 4.0,
                "observed_code_count": 4,
            },
            "decision": {"absolute_entropy_pass": not collapsed},
        },
    )
    _write_json(
        checkpoint_dir / "plots/val/loss_total/auroc/auroc_summary.json",
        {
            "checkpoint": "loss_total.ckpt",
            "split": "val",
            "score_direction": "higher_score_is_more_anomalous",
            "per_signal": {
                "signal_a": {"auroc": 0.8, "partial_auroc": 0.6},
                "signal_b": {"auroc": 0.9, "partial_auroc": 0.7},
            },
            "summaries": {
                "median_auroc": 0.85,
                "min_auroc": 0.8,
                "median_partial_auroc": 0.65,
                "min_partial_auroc": 0.6,
            },
        },
    )
    return {
        "configuration_id": configuration_id,
        "autoencoder_seed": seed,
        "manifest_path": str(manifest_path),
        "checkpoint_run_dir": str(checkpoint_dir),
    }


def _study_map(runs: list[dict[str, str | int]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "runs": runs,
    }


def _configuration(collection: dict[str, Any], configuration_id: str) -> dict[str, Any]:
    return next(
        item
        for item in collection["configurations"]
        if item["configuration_id"] == configuration_id
    )


def test_collects_baseline_and_candidate_and_writes_tables(tmp_path: Path) -> None:
    baseline_id = _configuration_id(0.0)
    candidate_id = _configuration_id(0.1)
    runs = [
        _write_run(tmp_path, configuration_id=baseline_id, gamma=0.0, leakage=0.2),
        _write_run(
            tmp_path,
            configuration_id=candidate_id,
            gamma=0.1,
            leakage=0.1,
            min_efficiency=0.49,
        ),
    ]

    study_map_path = tmp_path / "study_map.yaml"
    _write_json(study_map_path, _study_map(runs))
    collection = collect_and_write_pareto_study(
        study_map_path,
        output_dir=tmp_path / "study_output",
    )
    candidate = _configuration(collection, candidate_id)

    assert candidate["configuration_valid"] is True
    assert candidate["feasible"] is True
    assert candidate["autoencoder_seed"] == SEED
    assert candidate["metrics"]["leakage_worst"] == pytest.approx(0.1)
    assert candidate["constraints"]["paired_gamma_zero_configuration_id"] == baseline_id

    paths = {name: Path(path) for name, path in collection["output_paths"].items()}
    assert paths["csv"].is_file()
    assert paths["parquet"].is_file()
    assert (
        tmp_path / "study_output" / candidate_id / "pareto_metrics.json"
    ).is_file()
    table = pd.read_parquet(paths["parquet"])
    assert set(table["configuration_id"]) == {baseline_id, candidate_id}
    row = table.loc[table["configuration_id"] == candidate_id]
    assert row["feasible"].item()
    assert row["leakage_worst"].item() == pytest.approx(0.1)
    assert not any(column.endswith(("_mean", "_ci95_low", "_n_seeds")) for column in table.columns)


def test_two_runs_of_one_configuration_are_refused(tmp_path: Path) -> None:
    configuration_id = _configuration_id(0.0)
    runs = [
        _write_run(tmp_path, configuration_id=configuration_id, gamma=0.0, seed=seed)
        for seed in (SEED, SEED + 1)
    ]

    with pytest.raises(ParetoCollectionError, match="exactly one run"):
        collect_pareto_study(_study_map(runs))


def test_invalid_probe_rejects_configuration(tmp_path: Path) -> None:
    configuration_id = _configuration_id(0.0)
    runs = [
        _write_run(tmp_path, configuration_id=configuration_id, gamma=0.0, leakage=None),
    ]

    configuration = _configuration(collect_pareto_study(_study_map(runs)), configuration_id)
    assert configuration["configuration_valid"] is False
    assert configuration["metrics"] is None
    assert any("invalid_probe" in reason for reason in configuration["rejection_reasons"])


def test_collapsed_candidate_is_visible_but_infeasible(tmp_path: Path) -> None:
    baseline_id = _configuration_id(0.0)
    candidate_id = _configuration_id(0.1)
    runs = [
        _write_run(tmp_path, configuration_id=baseline_id, gamma=0.0),
        _write_run(tmp_path, configuration_id=candidate_id, gamma=0.1, collapsed=True),
    ]

    configuration = _configuration(collect_pareto_study(_study_map(runs)), candidate_id)
    assert configuration["configuration_valid"] is True
    assert configuration["feasible"] is False
    assert configuration["constraints"]["candidate_absolute_entropy_pass"] is False


def test_incorrect_manifest_mapping_is_rejected(tmp_path: Path) -> None:
    actual_id = _configuration_id(0.0)
    mapped_id = "unexpected-configuration-id"
    run = _write_run(tmp_path, configuration_id=actual_id, gamma=0.0)
    run["configuration_id"] = mapped_id

    configuration = _configuration(collect_pareto_study(_study_map([run])), mapped_id)
    assert configuration["configuration_valid"] is False
    assert any("manifest_invalid" in reason for reason in configuration["rejection_reasons"])


def test_nonfinite_utility_metric_rejects_configuration(tmp_path: Path) -> None:
    configuration_id = _configuration_id(0.0)
    runs = [_write_run(tmp_path, configuration_id=configuration_id, gamma=0.0)]
    efficiency_path = Path(runs[0]["checkpoint_run_dir"]) / "plots/val/loss_total/eff/eff_summary.json"
    payload = json.loads(efficiency_path.read_text(encoding="utf-8"))
    payload["median_efficiency"] = float("nan")
    efficiency_path.write_text(json.dumps(payload), encoding="utf-8")

    configuration = _configuration(collect_pareto_study(_study_map(runs)), configuration_id)
    assert configuration["configuration_valid"] is False
    assert any("efficiency_artifact_invalid" in reason for reason in configuration["rejection_reasons"])


def test_architecture_outside_the_search_space_is_rejected(tmp_path: Path) -> None:
    configuration_id = _configuration_id(0.0, "h128_64")
    run = _write_run(tmp_path, configuration_id=configuration_id, gamma=0.0)
    manifest_path = Path(run["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pareto_study"]["candidate"]["architecture_id"] = "h128_64"
    manifest["pareto_study"]["candidate"]["encoder_nodes"] = [128, 64, 8]
    manifest["algorithm"]["encoder"]["nodes"] = [128, 64, 8]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    configuration = _configuration(collect_pareto_study(_study_map([run])), configuration_id)
    assert configuration["configuration_valid"] is False
    assert any("not part of the study" in reason for reason in configuration["rejection_reasons"])
