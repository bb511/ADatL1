"""Tests for deterministic physics campaign exports."""

from __future__ import annotations

import importlib.util
import json
import pickle  # nosec B403 -- writes only synthetic trusted-local test fixtures
from pathlib import Path

import optuna
import pytest

CAMPAIGN_SCRIPT = Path(__file__).parents[1] / "scripts/physics_background_pairing_campaign.py"
REPORT_SCRIPT = Path(__file__).parents[1] / "scripts/physics_background_pairing_report.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


campaign = _load("physics_background_pairing_campaign_for_report", CAMPAIGN_SCRIPT)
report = _load("physics_background_pairing_report", REPORT_SCRIPT)
report.TARGET_TRIALS = 1


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _campaign(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "campaign"
    preflight = tmp_path / "preflight.json"
    _json(
        preflight,
        {
            "dataset_1": "ZB_run396102",
            "dataset_2": "ZB_run398183",
            "events_per_source": 163840,
            "tables": [
                {
                    "stage": stage,
                    "strategy": strategy,
                    "authenticated": True,
                    "pairs": 163840,
                }
                for stage in ("validate", "test")
                for strategy in (
                    "flat_physical",
                    "physics_summary",
                    "typed_sliced_wasserstein",
                    "jetclr",
                )
            ],
        },
    )
    studies = campaign._studies()
    design = {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "code_commit": "a" * 40,
        "preflight": str(preflight),
        "preflight_sha256": report._sha256(preflight),
        "target_trials_per_study": 1,
        "search_epochs": 50,
        "retrain_epochs": 200,
        "shared_trial_pool": True,
        "models": list(campaign.MODELS),
        "scores_by_model": {
            model: list(scores) for model, scores in campaign.SCORES_BY_MODEL.items()
        },
        "cap_strategies": list(campaign.CAP_STRATEGIES),
        "studies": studies,
    }
    _json(root / "design.json", design)
    for study in studies:
        database = root / "studies" / f"{study['model']}.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        directions = [
            direction for objective in study["objectives"] for direction in objective["directions"]
        ]
        loaded = optuna.create_study(
            study_name=study["study_name"],
            storage=f"sqlite:///{database}",
            directions=directions,
        )
        loaded.add_trial(optuna.trial.create_trial(values=[0.5] * len(directions)))
        artifact = root / "sweeps" / study["model"] / "stamp" / "0" / "optimized_metric.json"
        objective_order = [objective["id"] for objective in study["objectives"]]
        _json(
            artifact,
            {
                "schema_version": 2,
                "objective_order": objective_order,
                "optimized_metric": [0.5] * len(directions),
                "selections": {
                    objective: {
                        "optimized_ckpt_name": "last",
                        "optimized_metric": [0.5, 0.5],
                    }
                    for objective in objective_order
                },
                "run_provenance": {
                    "experiment_name": f"physics_bgpair_search_{study['model']}",
                    "run_name": f"search-{study['model']}",
                    "hydra_output_dir": str(artifact.parent),
                    "hydra_job_num": 0,
                    "checkpoint_run_dir": str(root / "runtime" / "checkpoints" / study["model"]),
                    "loggers": [{"experiment_id": "1", "run_id": f"run-{study['model']}"}],
                },
            },
        )
        _json(
            artifact.parents[1] / "trial_bindings.json",
            {
                "schema_version": 1,
                "model": study["model"],
                "study_name": study["study_name"],
                "prior_trial_numbers": [],
                "bindings": [
                    {
                        "model": study["model"],
                        "study_name": study["study_name"],
                        "trial_number": 0,
                        "hydra_job_num": 0,
                        "hydra_run_dir": str(artifact.parent),
                        "optimized_metric_artifact": str(artifact),
                        "optimized_metric_artifact_sha256": report._sha256(artifact),
                        "state": "COMPLETE",
                        "params": {},
                        "values": [0.5] * len(directions),
                    }
                ],
            },
        )
    return root, design


def test_snapshot_is_stable_and_selects_one_trial_per_study(tmp_path) -> None:
    """Repeated search snapshots are byte-stable for unchanged sources."""
    root, _ = _campaign(tmp_path)
    output = root / "portable"

    first = report.snapshot(root, output)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    second = report.snapshot(root, output)
    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}

    assert first == second
    assert before == after
    with (output / "study_status.csv").open(encoding="utf-8") as handle:
        rows = list(report.csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["selected_for_freeze"] for row in rows} == {"1"}
    with (output / "trials.csv").open(encoding="utf-8") as handle:
        trials = list(report.csv.DictReader(handle))
    assert len(trials) == 6
    assert all(row["selected_for_freeze"] == "True" for row in trials)


def test_snapshot_does_not_read_downstream_oracle(tmp_path) -> None:
    """Pre-freeze status remains usable even when a downstream file is invalid."""
    root, _ = _campaign(tmp_path)
    oracle = root / "results" / "oracle_winners.json"
    oracle.parent.mkdir(parents=True)
    oracle.write_text("not-json\n", encoding="utf-8")

    status = report.snapshot(root, root / "portable")

    assert status["ready_to_freeze"] is True


def test_snapshot_rejects_binding_that_differs_from_optuna(tmp_path) -> None:
    """Chunk identity cannot be inferred or accepted when its objective vector differs."""
    root, _ = _campaign(tmp_path)
    path = root / "sweeps" / "ae" / "stamp" / "trial_bindings.json"
    binding = json.loads(path.read_text(encoding="utf-8"))
    binding["bindings"][0]["values"][0] = 999.0
    _json(path, binding)

    with pytest.raises(RuntimeError, match="differs from Optuna trial"):
        report.snapshot(root, root / "portable")


def _frozen_results(root: Path, design: dict) -> None:
    memberships = []
    retrains = []
    winners = []
    anomalies = list(report.DOWNSTREAM_DATASETS)
    for retrain_index, study in enumerate(design["studies"]):
        objectives = [objective["id"] for objective in study["objectives"]]
        retrain_row = {
            "model": study["model"],
            "trial_number": 0,
            "params": {},
            "logical_objectives": sorted(objectives),
            "seed": 123,
            "epochs": 200,
            "retrain_index": retrain_index,
        }
        retrains.append(retrain_row)
        artifact = root / "retraining" / "hydra" / f"{retrain_index:04d}" / "optimized_metric.json"
        _json(
            artifact,
            {
                "schema_version": 2,
                "objective_order": objectives,
                "optimized_metric": [0.5] * (2 * len(objectives)),
                "selections": {
                    objective: {
                        "optimized_ckpt_name": "last",
                        "optimized_metric": [0.5, 0.5],
                    }
                    for objective in objectives
                },
            },
        )
        checkpoint_root = root / "runtime" / "checkpoints" / study["model"]
        for score in design["scores_by_model"][study["model"]]:
            callbacks = [
                (f"cap_{score}_{strategy}_summary", ({"last": 0.5}, {"last": 0.4}))
                for strategy in design["cap_strategies"]
            ]
            callbacks.extend(
                [
                    (f"wasserstein_{score}_summary", ({"last": 0.5},)),
                    (
                        f"drift_{score}_summary",
                        (
                            {"operational": {"last": 0.5}},
                            {"operational": {"last": 0.6}},
                        ),
                    ),
                ]
            )
            for callback, objects in callbacks:
                summary = checkpoint_root / "last" / "plots" / "test" / callback / "summary.pkl"
                summary.parent.mkdir(parents=True, exist_ok=True)
                with summary.open("wb") as handle:
                    for value in objects:
                        pickle.dump(value, handle)
        marker = {
            "schema_version": 2,
            "started_at": "2026-01-01T23:00:00+00:00",
            "completed_at": "2026-01-02T00:00:00+00:00",
            "elapsed_seconds": 3600.0,
            "slurm_job_id": "1234",
            "slurm_array_job_id": "1234",
            "slurm_array_task_id": str(retrain_index),
            "row": retrain_row,
            "command": [],
            "optimized_metric_artifact": str(artifact),
            "optimized_metric_artifact_sha256": report._sha256(artifact),
            "checkpoint_root": str(checkpoint_root),
            "value_files": [],
        }
        _json(root / "retraining" / "markers" / f"{retrain_index:04d}.json", marker)
        for objective in study["objectives"]:
            membership = {
                "membership_index": len(memberships),
                "model": study["model"],
                "logical_objective": objective["id"],
                "score": objective["score"],
                "metric": objective["metric"],
                "strategy": objective["strategy"],
                "trial_number": 0,
                "objective_values": [0.5, 0.5],
                "retrain_index": retrain_index,
            }
            memberships.append(membership)
            winners.append(
                {
                    **membership,
                    "params": {},
                    "optimized_ckpt_name": "last",
                    "retrained_objective_values": [0.5, 0.5],
                    "mean_downstream_efficiency": 0.5,
                    "downstream_efficiencies": {name: 0.5 for name in anomalies},
                }
            )
    manifest = {
        "schema_version": 2,
        "created_at": "2026-01-02T00:00:00+00:00",
        "code_commit": design["code_commit"],
        "selection_uses_downstream_anomalies": False,
        "front_memberships": memberships,
        "retrain_runs": retrains,
    }
    _json(root / "selection" / "pareto_retrain_manifest.json", manifest)
    _json(
        root / "results" / "oracle_winners.json",
        {
            "schema_version": 2,
            "created_at": "2026-01-03T00:00:00+00:00",
            "oracle_uses_downstream_anomalies": True,
            "winners": winners,
            "all_front_memberships": winners,
        },
    )


def test_finalize_exports_56_winners_and_reverse_drift(tmp_path) -> None:
    """Finalization enforces the full logical matrix and portable row counts."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    output = root / "portable"

    report.finalize(root, output)

    with (output / "winners.csv").open(encoding="utf-8") as handle:
        assert len(list(report.csv.DictReader(handle))) == 56
    with (output / "downstream_efficiencies.csv").open(encoding="utf-8") as handle:
        assert len(list(report.csv.DictReader(handle))) == 56 * 20
    with (output / "primary_metrics.csv").open(encoding="utf-8") as handle:
        metrics = list(report.csv.DictReader(handle))
    assert sum(row["metric_name"] == "reverse_drift_posthoc" for row in metrics) == 8
    assert sum(row["metric_name"] == "rankcorr_test" for row in metrics) == 40
    manifest_lines = (output / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    assert all(not line.endswith("  manifest.sha256") for line in manifest_lines)
    assert (output / "REPORT.md").is_file()


def test_finalize_rejects_changed_retrain_artifact(tmp_path) -> None:
    """A marker hash mismatch stops final reporting rather than hiding corruption."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    artifact = root / "retraining" / "hydra" / "0000" / "optimized_metric.json"
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Changed optimized artifact"):
        report.finalize(root, root / "portable")


def test_finalize_rejects_tampered_pareto_membership(tmp_path) -> None:
    """Finalization reconstructs fronts instead of trusting the frozen manifest."""
    root, design = _campaign(tmp_path)
    _frozen_results(root, design)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["front_memberships"][0]["objective_values"][0] = 999.0
    _json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="differ from reconstructed Optuna fronts"):
        report.finalize(root, root / "portable")


def test_pickle_reader_rejects_path_outside_authenticated_root(tmp_path) -> None:
    """Reporter pickle loading is confined to authenticated checkpoint trees."""
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    external = tmp_path / "external.pkl"
    with external.open("wb") as handle:
        pickle.dump({}, handle)

    with pytest.raises(RuntimeError, match="outside trusted roots"):
        report._read_pickle_objects(external, [trusted])
