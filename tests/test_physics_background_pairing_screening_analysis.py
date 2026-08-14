"""Tests for signal-blind physics screening convergence diagnostics."""

from __future__ import annotations

import json

import pytest

from scripts import physics_background_pairing_report as report
from src.evaluation.physics_background_pairing_screening_analysis import (
    build_screening_convergence,
)


def _design(target: int = 3) -> dict:
    return {
        "code_commit": "a" * 40,
        "target_trials_per_study": target,
        "studies": [
            {
                "model": "ae",
                "objectives": [
                    {
                        "id": "native__cap__cdf",
                        "score": "native",
                        "metric": "cap",
                        "strategy": "cdf",
                        "value_indices": [0, 1],
                        "directions": ["maximize", "minimize"],
                    }
                ],
            }
        ],
    }


def _trials() -> list[dict]:
    return [
        {
            "model": "ae",
            "trial_number": 0,
            "selected_for_freeze": True,
            "nsga_generation": 0,
            "values_json": json.dumps([0.0, 10.0]),
        },
        {
            "model": "ae",
            "trial_number": 1,
            "selected_for_freeze": True,
            "nsga_generation": 0,
            "values_json": json.dumps([1.0, 20.0]),
        },
        {
            "model": "ae",
            "trial_number": 2,
            "selected_for_freeze": True,
            "nsga_generation": 1,
            "values_json": json.dumps([2.0, 5.0]),
        },
    ]


def test_generation_diagnostics_use_fixed_frozen_pool_scaling() -> None:
    analysis = build_screening_convergence(_design(), _trials())
    rows = analysis["search_convergence"]
    assert len(rows) == 2
    assert rows[0]["cumulative_nondominated_size"] == 2
    assert rows[0]["previous_front_jaccard"] == ""
    assert rows[1]["cumulative_nondominated_size"] == 1
    assert rows[1]["front_entered"] == 1
    assert rows[1]["front_exited"] == 2
    assert rows[1]["previous_front_jaccard"] == 0.0
    assert rows[1]["q_prime_endpoint_improvement"] == pytest.approx(0.5)
    assert rows[1]["q_double_prime_endpoint_improvement"] == pytest.approx(1.0 / 3.0)
    assert rows[0]["normalized_hypervolume"] == 0.0
    assert rows[1]["normalized_hypervolume"] == 1.0
    assert {(row["generation"], row["trial_number"]) for row in analysis["front_evolution"]} == {
        (0, 0),
        (0, 1),
        (1, 2),
    }
    assert analysis["method"]["downstream_outcomes_used"] is False
    assert analysis["method"]["hypervolume_reference"] == [0.0, 0.0]


def test_constant_axis_maps_to_observed_ideal() -> None:
    trials = _trials()[:2]
    for row in trials:
        row["values_json"] = json.dumps([5.0, json.loads(row["values_json"])[1]])
    analysis = build_screening_convergence(_design(target=2), trials)
    rows = analysis["front_evolution"]
    assert all(row["q_prime_normalized"] == 1.0 for row in rows)
    assert analysis["search_convergence"][0]["normalized_hypervolume"] == 1.0


def test_missing_frozen_trial_or_generation_fails_closed() -> None:
    with pytest.raises(ValueError, match="Expected 3 frozen trials"):
        build_screening_convergence(_design(), _trials()[:2])
    trials = _trials()
    trials[-1]["nsga_generation"] = ""
    with pytest.raises(ValueError, match="lacks an NSGA generation"):
        build_screening_convergence(_design(), trials)


def test_post_freeze_report_export_authenticates_final_front(tmp_path, monkeypatch) -> None:
    root = tmp_path / "campaign"
    output = tmp_path / "portable"
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "code_commit": "a" * 40,
        "selection_uses_downstream_anomalies": False,
        "front_memberships": [
            {
                "model": "ae",
                "logical_objective": "native__cap__cdf",
                "trial_number": 2,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    freeze_path = root / "state" / "freeze_validation.json"
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text(
        json.dumps(
            {
                "code_commit": "a" * 40,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": report._sha256(manifest_path),
                "logical_fronts": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "EXPECTED_FRONTS", 1)
    report._write_screening_audit(root, output, _design(), _trials())
    assert (output / "SCREENING_AUDIT.md").is_file()
    method = json.loads((output / "analysis" / "screening_convergence_method.json").read_text())
    assert method["downstream_outcomes_used"] is False
    assert method["final_front_memberships"] == 1
    assert (output / "analysis" / "search_convergence.csv").is_file()
    assert (output / "analysis" / "front_evolution.csv").is_file()
