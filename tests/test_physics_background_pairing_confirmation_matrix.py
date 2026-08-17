"""Contracts for the authenticated 28-config confirmation matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.evaluation import physics_background_pairing_confirmation_matrix as matrix

COMMIT = "a" * 40


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create complete CSV/database fixtures and replace only Optuna deserialization."""
    root = tmp_path / "discovery"
    (root / "results/endpoint_analysis").mkdir(parents=True)
    (root / "studies").mkdir()
    cells = matrix._expected_cells()
    with (root / matrix.FRONTS_RELATIVE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "score", "logical_objective", "trial_number"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "model": row["model"],
                "score": "native",
                "logical_objective": row["logical_objective"],
                "trial_number": row["trial_number"],
            }
            for row in cells
        )
    keys = sorted({(row["model"], row["trial_number"]) for row in cells})
    params = {
        key: {"algorithm.optimizer.lr": (index + 1) / 10000, "choice": f"v{index}"}
        for index, key in enumerate(keys)
    }
    with (root / matrix.TRIALS_RELATIVE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "trial_number", "state", "usable", "params_json"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "model": model,
                "trial_number": number,
                "state": "COMPLETE",
                "usable": "True",
                "params_json": json.dumps(params[(model, number)]),
            }
            for model, number in keys
        )
    for model in matrix.MODELS:
        (root / "studies" / f"{model}.db").write_bytes(f"db-{model}".encode())
    (root / "preflight.json").write_text("{}\n", encoding="utf-8")
    (root / "design.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_at": "2026-08-01T00:00:00+00:00",
                "code_commit": "0" * 40,
                "preflight": str((root / "preflight.json").resolve()),
                "preflight_sha256": matrix._sha256(root / "preflight.json"),
                "retrain_epochs": 200,
                "screening_sampler": None,
                "studies": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def database_trials(database: Path, study_name: str, numbers) -> dict:
        model = database.stem
        assert study_name == f"physics_{model}_background_all"
        n_values = 28 if model in {"ae", "vae"} else 14
        return {
            number: {"params": params[(model, number)], "values": [0.1] * n_values}
            for number in numbers
        }

    monkeypatch.setattr(matrix, "_database_trials", database_trials)
    return root


def test_build_authenticates_exact_28_42_84_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen matrix must retain every exact deduplication cardinality."""
    root = _fixture(tmp_path, monkeypatch)
    payload = matrix.build_confirmation_matrix(root, execution_commit=COMMIT)
    assert payload["cardinality"] == {
        "models": 6,
        "logical_cells": 42,
        "unique_configurations": 28,
        "seeds": 3,
        "runs": 84,
    }
    assert payload["training_seeds"] == list(matrix.TRAINING_SEEDS)
    assert len({row["run_id"] for row in payload["runs"]}) == 84
    assert [row["model"] for row in payload["runs"][:6]] == list(matrix.MODELS)
    assert {
        model: sum(row["model"] == model for row in payload["runs"][:18])
        for model in matrix.MODELS
    } == {model: 3 for model in matrix.MODELS}
    assert [
        sum(row["model"] == model for row in payload["configurations"]) for model in matrix.MODELS
    ] == [4, 5, 2, 7, 5, 5]
    assert payload["content_sha256"] == matrix._content_hash(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )


def test_matrix_rejects_endpoint_or_optuna_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Either endpoint or database disagreement must stop construction."""
    root = _fixture(tmp_path, monkeypatch)
    front = root / matrix.FRONTS_RELATIVE
    rows = matrix._csv_rows(front)
    rows[0]["trial_number"] = "999"
    with front.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="winner differs"):
        matrix.build_confirmation_matrix(root, execution_commit=COMMIT)

    root = _fixture(tmp_path / "second", monkeypatch)
    original = matrix._database_trials

    def mismatched(database: Path, study_name: str, numbers) -> dict:
        result = original(database, study_name, numbers)
        result[next(iter(result))]["params"] = {"changed": True}
        return result

    monkeypatch.setattr(matrix, "_database_trials", mismatched)
    with pytest.raises(ValueError, match="parameters differ"):
        matrix.build_confirmation_matrix(root, execution_commit=COMMIT)


def test_seed_contract_and_atomic_freeze_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeds are predeclared and an existing manifest is immutable."""
    root = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="three or five distinct"):
        matrix.build_confirmation_matrix(root, execution_commit=COMMIT, training_seeds=(123, 2, 3))
    extended = matrix.build_confirmation_matrix(
        root,
        execution_commit=COMMIT,
        training_seeds=matrix.TRAINING_SEEDS + matrix.RESERVE_TRAINING_SEEDS,
    )
    assert extended["cardinality"]["runs"] == 140
    with pytest.raises(ValueError, match="preregistered reserve"):
        matrix.build_confirmation_matrix(
            root, execution_commit=COMMIT, training_seeds=(1, 2, 3, 4, 5)
        )
    confirmation = tmp_path / "confirmation"
    output = matrix.freeze_confirmation_matrix(root, confirmation, execution_commit=COMMIT)
    original = output.read_bytes()
    assert matrix.freeze_confirmation_matrix(root, confirmation, execution_commit=COMMIT) == output
    assert output.read_bytes() == original
    with pytest.raises(FileExistsError, match="different frozen metadata"):
        matrix.freeze_confirmation_matrix(
            root,
            confirmation,
            execution_commit="b" * 40,
        )


def test_freeze_campaign_projects_exact_runner_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart-safe runner projection preserves 84 runs and 126 cell/seed uses."""
    discovery = _fixture(tmp_path, monkeypatch)
    root = tmp_path / "confirmation"
    paths = matrix.freeze_confirmation_campaign(
        discovery,
        root,
        execution_commit=COMMIT,
        execution_preflight=discovery / "preflight.json",
    )
    # Repeating before any result exists must be byte-identical and harmless.
    assert (
        matrix.freeze_confirmation_campaign(
            discovery,
            root,
            execution_commit=COMMIT,
            execution_preflight=discovery / "preflight.json",
        )
        == paths
    )
    frozen = json.loads(paths["matrix"].read_text(encoding="utf-8"))
    runner = json.loads(paths["runner_manifest"].read_text(encoding="utf-8"))
    design = json.loads(paths["design"].read_text(encoding="utf-8"))
    assert len(runner["retrain_runs"]) == 84
    assert len(runner["front_memberships"]) == 126
    assert runner["selection_uses_downstream_anomalies"] is True
    assert {row["seed"] for row in runner["retrain_runs"]} == set(matrix.TRAINING_SEEDS)
    assert len({row["run_id"] for row in runner["retrain_runs"]}) == 84
    assert all(row["params"] for row in runner["retrain_runs"])
    assert runner["confirmation_matrix_content_sha256"] == frozen["content_sha256"]
    assert design["code_commit"] == COMMIT
    assert design["campaign_kind"] == "physics_background_native_confirmation"
    authenticated = matrix.load_authenticated_confirmation_campaign(root)
    assert authenticated["matrix"] == frozen

    runner_path = paths["runner_manifest"]
    runner_path.write_text(runner_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="identities differ"):
        matrix.load_authenticated_confirmation_campaign(root)
    runner_path.write_text(
        json.dumps(runner, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    mutated = dict(runner)
    mutated["retrain_runs"] = mutated["retrain_runs"][:-1]
    runner_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash differs"):
        matrix.load_authenticated_confirmation_campaign(root)


def test_reserve_extension_overlay_supports_only_dsvae_mixed_seed_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One triggered model appends only its exact runs and memberships."""
    discovery = _fixture(tmp_path, monkeypatch)
    root = tmp_path / "confirmation"
    matrix.freeze_confirmation_campaign(
        discovery,
        root,
        execution_commit=COMMIT,
        execution_preflight=discovery / "preflight.json",
    )
    decisions = root / "results/analysis/reserve_seed_decisions.csv"
    decisions.parent.mkdir(parents=True)
    with decisions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "observed_seed_count", "decision"))
        writer.writeheader()
        writer.writerows(
            {
                "model": model,
                "observed_seed_count": 3,
                "decision": ("add_both_reserve_seeds" if model == "dsvae" else "no_extension"),
            }
            for model in matrix.MODELS
        )
    analysis = root / "results/analysis/confirmation_endpoint_analysis.json"
    payload = {
        "schema_version": 1,
        "artifacts": {decisions.name: matrix._sha256(decisions)},
    }
    payload["content_sha256"] = matrix._content_hash(payload)
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        matrix,
        "_reserve_execution_identity",
        lambda repository, base: {
            "base_code_commit": base,
            "execution_commit": "b" * 40,
            "execution_changed_paths": sorted(matrix.REQUIRED_RESERVE_EXECUTION_PATHS),
            "execution_diff_sha256": "c" * 64,
        },
    )
    extension_path = matrix.freeze_reserve_extension_manifest(root, decisions, analysis, tmp_path)
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    assert extension["cardinality"] == {
        "added_runs": 14,
        "added_memberships": 14,
        "effective_runs": 98,
        "effective_memberships": 140,
    }
    frozen = matrix.load_authenticated_confirmation_campaign(root)
    assert len(frozen["runner_manifest"]["retrain_runs"]) == 98
    assert len(frozen["runner_manifest"]["front_memberships"]) == 140
    assert frozen["training_seeds_by_model"]["dsvae"] == list(
        matrix.TRAINING_SEEDS + matrix.RESERVE_TRAINING_SEEDS
    )
    assert frozen["training_seeds_by_model"]["ae"] == list(matrix.TRAINING_SEEDS)
    assert (
        len(
            matrix.load_authenticated_confirmation_campaign(root, include_extension=False)[
                "runner_manifest"
            ]["retrain_runs"]
        )
        == 84
    )


def test_reserve_extension_rejects_partial_or_tampered_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overlay cannot be detached from exact three-seed decision evidence."""
    discovery = _fixture(tmp_path, monkeypatch)
    root = tmp_path / "confirmation"
    matrix.freeze_confirmation_campaign(
        discovery,
        root,
        execution_commit=COMMIT,
        execution_preflight=discovery / "preflight.json",
    )
    decisions = root / "decisions.csv"
    with decisions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "observed_seed_count", "decision"))
        writer.writeheader()
        writer.writerows(
            {"model": model, "observed_seed_count": 3, "decision": "no_extension"}
            for model in matrix.MODELS
        )
    analysis = root / "analysis.json"
    payload = {"artifacts": {decisions.name: matrix._sha256(decisions)}}
    payload["content_sha256"] = matrix._content_hash(payload)
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        matrix,
        "_reserve_execution_identity",
        lambda repository, base: {
            "base_code_commit": base,
            "execution_commit": "b" * 40,
            "execution_changed_paths": sorted(matrix.REQUIRED_RESERVE_EXECUTION_PATHS),
            "execution_diff_sha256": "c" * 64,
        },
    )
    matrix.freeze_reserve_extension_manifest(root, decisions, analysis, tmp_path)
    decisions.write_text(decisions.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source evidence changed"):
        matrix.load_authenticated_confirmation_campaign(root)


def test_database_reader_requires_finite_complete_exact_trial(tmp_path: Path) -> None:
    """The SQLite reader accepts only requested finite COMPLETE trials."""
    optuna = pytest.importorskip("optuna")
    database = tmp_path / "study.db"
    storage = f"sqlite:///{database}"
    study = optuna.create_study(study_name="study", storage=storage, directions=["maximize"])
    study.optimize(lambda trial: 0.5 + 0.0 * trial.suggest_int("x", 1, 1), n_trials=1)
    assert matrix._database_trials(database, "study", [0]) == {
        0: {"params": {"x": 1}, "values": [0.5]}
    }
    with pytest.raises(ValueError, match="lacks selected trials"):
        matrix._database_trials(database, "study", [1])
