from __future__ import annotations

import csv
import json
import subprocess  # nosec B404
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from scripts import cchamber_candidate_rank_audit as audit
from src.callbacks.audit import CheckpointBranchManifest, TrajectoryFingerprint
from src.utils.pairing.io import compose_config


def test_frozen_design_inputs_match_the_authorized_hashes() -> None:
    """The checked-in workflow must remain pinned to both authorized design files."""
    panel, contract = audit._validate_frozen_design()
    assert audit._sha256(audit.PANEL_PATH) == audit.PANEL_SHA256
    assert audit._sha256(audit.CONTRACT_PATH) == audit.CONTRACT_SHA256
    assert len(panel["candidate_ids"]) == 16
    assert contract["sealed_evaluation"]["expected_rows"] == 111_360


def test_all_candidate_rank_configs_compose_with_five_branches(monkeypatch) -> None:
    """Every detector should compose the same five checkpoint monitors."""
    monkeypatch.setenv("CCHAMBER_VALID_PAIR_TABLE", "/synthetic/valid.pt")
    monkeypatch.setenv("CCHAMBER_AUDIT_CHECKPOINT_MANIFEST", "/synthetic/branches.json")
    monkeypatch.setenv("CCHAMBER_AUDIT_TRAJECTORY_FINGERPRINT", "/synthetic/fingerprint.json")
    for model in audit.MODELS:
        cfg = compose_config(overrides=[f"experiment=cchamber/{model}_candidate_rank_audit"])
        monitors = {
            cfg.callbacks[f"audit_ckpt_{strategy}"].monitor for strategy in audit.STRATEGIES
        }
        assert monitors == set(audit.MONITORS.values())
        assert cfg.data.signal_experiments == []
        assert cfg.test is False
        assert cfg.evaluation is None


def test_candidate_parameters_refuse_missing_or_replaced_panel_candidates() -> None:
    """A failed frozen-panel candidate must remain missing rather than be replaced."""
    panel = {"models": ["ae"], "candidate_ids": ["000", "006"]}
    frame = pd.DataFrame(
        [
            {
                "model": "ae",
                "candidate_id": "000",
                "strategy": strategy,
                "seed": seed,
                "value": 1.0,
                "params_json": '{"algorithm.optimizer.lr":0.001}',
            }
            for strategy, seed in product(audit.STRATEGIES, (101, 202))
        ]
    )
    with pytest.raises(ValueError, match="must not be replaced"):
        audit._candidate_parameters(frame, panel)


def _synthetic_candidate_metrics() -> pd.DataFrame:
    """Build a complete 4-by-5 full pool with five development seeds."""
    rows = []
    for model, strategy, candidate, seed in product(
        audit.MODELS,
        audit.STRATEGIES,
        range(20),
        (101, 202, 303, 404, 505),
    ):
        value = float(candidate)
        if audit.DIRECTIONS[strategy] == "minimize":
            value = -value
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "candidate_id": f"{candidate:03d}",
                "seed": seed,
                "value": value,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_candidate_provenance(tmp_path):
    """Create a complete label-free candidate provenance chain."""
    campaign_root = tmp_path / "campaign"
    design_root = campaign_root / "design"
    design_root.mkdir(parents=True)
    pool_hashes = {}
    pool_rows = {}
    for model in audit.MODELS:
        pool = [
            {
                "candidate_id": candidate_id,
                "model": model,
                "params": {"algorithm.optimizer.lr": 0.001 + int(candidate_id) * 1e-5},
                "params_sha256": "unused-in-this-validator",
            }
            for candidate_id in ("000", "001")
        ]
        pool_path = design_root / f"{model}_candidates.json"
        pool_path.write_text(json.dumps(pool), encoding="utf-8")
        pool_hashes[model] = audit._sha256(pool_path)
        pool_rows[model] = pool
    campaign = {
        "campaign_id": "cchamber_real_20260725_63b941a",
        "git_commit": audit.CAMPAIGN_COMMIT,
        "models": list(audit.MODELS),
        "strategies": list(audit.STRATEGIES),
        "development_seeds": list(audit.DEVELOPMENT_SEEDS),
        "n_candidates_per_model": 2,
        "pool_sha256": pool_hashes,
    }
    campaign_path = campaign_root / "campaign.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    rows = []
    pair_hash = "a" * 64
    for model, strategy, seed, candidate_id in product(
        audit.MODELS,
        audit.STRATEGIES,
        audit.DEVELOPMENT_SEEDS,
        ("000", "001"),
    ):
        params = next(
            row["params"] for row in pool_rows[model] if row["candidate_id"] == candidate_id
        )
        rows.append(
            {
                "dataset": "cchamber",
                "model": model,
                "seed": seed,
                "candidate_id": candidate_id,
                "strategy": strategy,
                "value": float(int(candidate_id) + seed),
                "params_json": audit._canonical_json(params),
                "mlflow_run_id": "synthetic",
                "pool_sha256": pool_hashes[model],
                "pair_table_sha256": pair_hash,
                "git_commit": audit.CAMPAIGN_COMMIT,
            }
        )
    candidate_path = campaign_root / "selection" / "candidate_metrics.csv"
    audit._write_csv(candidate_path, rows)
    expected_rows = len(rows)
    provenance = {
        "campaign": str(campaign_path.resolve()),
        "campaign_sha256": audit._sha256(campaign_path),
        "candidate_metrics": str(candidate_path.resolve()),
        "candidate_metrics_sha256": audit._sha256(candidate_path),
        "n_rows": expected_rows,
        "expected_rows_after_global_exclusion": expected_rows,
        "surviving_candidates_per_model": {model: 2 for model in audit.MODELS},
        "globally_excluded_candidates": {model: [] for model in audit.MODELS},
    }
    provenance_path = candidate_path.with_name("candidate_metrics_provenance.json")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return campaign, campaign_path, candidate_path, provenance, provenance_path, pair_hash


def test_candidate_provenance_authenticates_shared_survivor_pools(tmp_path) -> None:
    """Design provenance must bind campaign, pool files, and every shared survivor."""
    (
        campaign,
        campaign_path,
        candidate_path,
        provenance,
        provenance_path,
        pair_hash,
    ) = _synthetic_candidate_provenance(tmp_path)
    frame = audit._validate_candidate_metrics_provenance(
        provenance_path,
        audit._sha256(provenance_path),
        candidate_path,
        audit._sha256(candidate_path),
        campaign_path,
        audit._sha256(campaign_path),
        campaign,
        pair_hash,
    )
    assert len(frame) == 200

    rows = list(csv.DictReader(candidate_path.open(encoding="utf-8", newline="")))
    rows = [
        row
        for row in rows
        if not (
            row["model"] == "ae"
            and row["strategy"] == audit.STRATEGIES[0]
            and int(row["seed"]) == audit.DEVELOPMENT_SEEDS[0]
            and row["candidate_id"] == "001"
        )
    ]
    audit._write_csv(candidate_path, rows)
    provenance["candidate_metrics_sha256"] = audit._sha256(candidate_path)
    provenance["n_rows"] = len(rows)
    provenance["expected_rows_after_global_exclusion"] = len(rows)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="shared candidate pool"):
        audit._validate_candidate_metrics_provenance(
            provenance_path,
            audit._sha256(provenance_path),
            candidate_path,
            audit._sha256(candidate_path),
            campaign_path,
            audit._sha256(campaign_path),
            campaign,
            pair_hash,
        )


def test_pairing_provenance_binds_seed_123_table_checkpoint_and_metadata(tmp_path) -> None:
    """The primary pair table must trace to the pinned seed-123 encoder checkpoint."""
    checkpoint = tmp_path / "encoder.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    checkpoint_sha = audit._sha256(checkpoint)
    table_path = tmp_path / "validate_pairs.pt"
    table = {
        "schema_version": 1,
        "dataset_1": "normal",
        "dataset_2": "reference_normal",
        "split": "validate",
        "encoder_ckpt": str(checkpoint.resolve()),
        "idx_1": torch.arange(1000),
        "idx_2": torch.arange(1000),
        "distance": torch.zeros(1000),
        "rank_1_to_2": torch.zeros(1000, dtype=torch.long),
        "rank_2_to_1": torch.zeros(1000, dtype=torch.long),
        "metadata": {
            "pairing_mode": "one_to_one_nearest",
            "normalized": True,
            "n_dataset_1": 1000,
            "n_dataset_2": 1000,
            "n_pairs": 1000,
            "encoder_checkpoint_sha256": checkpoint_sha,
            "source_1_sha256": "1" * 64,
            "source_2_sha256": "2" * 64,
            "data_seed": 314159,
        },
    }
    torch.save(table, table_path)
    table_sha = audit._sha256(table_path)
    campaign = {
        "campaign_id": "cchamber_real_20260725_63b941a",
        "pair_encoder_seeds": [123],
        "data_seed": 314159,
    }
    primary = {
        "campaign_id": campaign["campaign_id"],
        "encoder_seed": 123,
        "data_seed": 314159,
        "git_commit": audit.CAMPAIGN_COMMIT,
        "encoder_checkpoint": str(checkpoint.resolve()),
        "encoder_checkpoint_sha256": checkpoint_sha,
        "validation_table": str(table_path.resolve()),
        "validation_table_sha256": table_sha,
    }
    pairing = {
        "campaign_id": campaign["campaign_id"],
        "primary_encoder_seed": 123,
        "primary_validation_table": str(table_path.resolve()),
        "primary_validation_table_sha256": table_sha,
        "encoder_runs": [primary],
    }
    pairing_path = tmp_path / "pairing_manifest.json"
    pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
    _, validated = audit._validate_pairing_provenance(
        pairing_path,
        audit._sha256(pairing_path),
        table_path,
        table_sha,
        campaign,
    )
    assert validated["encoder_checkpoint_sha256"] == checkpoint_sha

    changed = deepcopy(table)
    changed["encoder_ckpt"] = str((tmp_path / "wrong.ckpt").resolve())
    torch.save(changed, table_path)
    changed_sha = audit._sha256(table_path)
    pairing["primary_validation_table_sha256"] = changed_sha
    pairing["encoder_runs"][0]["validation_table_sha256"] = changed_sha
    pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata/provenance"):
        audit._validate_pairing_provenance(
            pairing_path,
            audit._sha256(pairing_path),
            table_path,
            changed_sha,
            campaign,
        )


def _synthetic_outcomes(panel_ids: list[str]) -> pd.DataFrame:
    """Build monotonic candidate outcomes for both metrics and three seeds."""
    rows = []
    for model, strategy, candidate_id, seed, metric in product(
        audit.MODELS,
        audit.STRATEGIES,
        panel_ids,
        (1001, 1002, 1003),
        audit.METRICS,
    ):
        value = (int(candidate_id) + 1) / 25.0 + (seed - 1001) * 1e-4
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "candidate_id": candidate_id,
                "reporting_seed": seed,
                "metric": metric,
                "intervention": "synthetic_intervention",
                "value": value,
            }
        )
    return pd.DataFrame(rows)


def test_rank_analysis_matches_frozen_families_and_estimands() -> None:
    """Synthetic monotonic outcomes should yield exact positive rank association."""
    panel_ids = [f"{value:03d}" for value in range(16)]
    panel = {
        "candidate_ids": panel_ids,
        "reporting_seeds": [1001, 1002, 1003],
    }
    contract = {"estimands": {"top_k": 4}}
    associations, seed_rows, bootstrap = audit.rank_analysis(
        _synthetic_outcomes(panel_ids),
        _synthetic_candidate_metrics(),
        panel,
        contract,
        n_permutations=40,
        n_bootstrap=40,
        random_seed=7,
    )
    assert len(associations) == 40
    assert len(seed_rows) == 120
    assert len(bootstrap) == 2 * 4 * 40
    assert set(associations["holm_family_size"]) == {20}
    assert np.allclose(associations["spearman_rho"], 1.0)
    assert np.allclose(associations["kendall_tau_b"], 1.0)
    assert set(associations["top_k_overlap"]) == {4}
    assert np.allclose(associations["top_k_oracle_regret"], 0.0)
    assert np.allclose(associations["proxy_best_regret"], 0.0)
    assert associations.groupby("metric")["spearman_holm_p"].count().to_dict() == {
        "auprc": 20,
        "efficiency_operational": 20,
    }
    assert associations["n_bootstrap_requested"].eq(40).all()
    assert associations["n_bootstrap_effective"].between(1, 40).all()
    assert associations["n_bootstrap_effective_paired"].between(1, 40).all()
    assert bootstrap["n_bootstrap_requested"].eq(40).all()
    for _, group in bootstrap.groupby(["metric", "model"]):
        assert group["paired_draw_finite"].sum() == group["n_bootstrap_effective_paired"].iloc[0]


def test_search_utility_rejects_extra_identities_and_strategy_pool_drift() -> None:
    """Utility ranks require the exact frozen identities and one shared survivor pool."""
    frame = _synthetic_candidate_metrics()
    extra = frame.iloc[[0]].copy()
    extra["strategy"] = "unfrozen_proxy"
    with pytest.raises(ValueError, match="strategy identities"):
        audit.compute_search_utility(
            pd.concat([frame, extra], ignore_index=True),
            audit.DEVELOPMENT_SEEDS,
        )
    mask = ~(
        (frame["model"] == "ae")
        & (frame["strategy"] == audit.STRATEGIES[0])
        & (frame["candidate_id"] == "019")
    )
    with pytest.raises(ValueError, match="shared across strategies"):
        audit.compute_search_utility(frame[mask], audit.DEVELOPMENT_SEEDS)
    with pytest.raises(ValueError, match="exact five"):
        audit.compute_search_utility(frame, audit.DEVELOPMENT_SEEDS[:-1])


def test_sealed_callback_emits_exact_58_by_2_numeric_rows(tmp_path) -> None:
    """The plot-free callback should emit only the frozen sealed metrics."""
    interventions = [f"intervention_{index:02d}" for index in range(58)]
    output = tmp_path / "sealed.csv"
    callback = audit._SealedMetricsCallback(
        interventions,
        output,
        {"trajectory_index": 0, "strategy": "cap_random"},
    )
    trainer = SimpleNamespace(
        test_dataloaders={"normal": object(), **{name: object() for name in interventions}}
    )
    module = SimpleNamespace(thres_operational=torch.tensor(0.5))
    callback.on_test_start(trainer, module)
    callback.on_test_epoch_start(trainer, module)
    callback.on_test_batch_end(
        trainer,
        module,
        {"ascore/full": torch.tensor([0.1, 0.2, 0.3])},
        None,
        0,
        0,
    )
    for index, _ in enumerate(interventions, start=1):
        callback.on_test_batch_end(
            trainer,
            module,
            {"ascore/full": torch.tensor([0.4, 0.8, 0.9])},
            None,
            0,
            index,
        )
    callback.on_test_epoch_end(trainer, module)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 116
    assert {row["metric"] for row in rows} == set(audit.METRICS)
    assert len({row["intervention"] for row in rows}) == 58
    assert all(0.0 <= float(row["value"]) <= 1.0 for row in rows)


def test_checkpoint_manifest_records_all_five_hashes_and_epochs(tmp_path) -> None:
    """The callback should freeze exact branch files and selected values."""
    callbacks = []
    for epoch, strategy in enumerate(audit.STRATEGIES):
        checkpoint = tmp_path / strategy / f"{strategy}.ckpt"
        checkpoint.parent.mkdir()
        torch.save({"epoch": epoch, "state_dict": {}}, checkpoint)
        callback = ModelCheckpoint(
            dirpath=checkpoint.parent,
            monitor=audit.MONITORS[strategy],
            mode="max" if audit.DIRECTIONS[strategy] == "maximize" else "min",
        )
        callback.best_model_path = str(checkpoint)
        callback.best_model_score = torch.tensor(float(epoch + 1))
        callbacks.append(callback)
    output = tmp_path / "branches.json"
    manifest = CheckpointBranchManifest(output, audit.MONITORS)
    manifest.on_fit_end(SimpleNamespace(callbacks=callbacks), None)
    value = json.loads(output.read_text(encoding="utf-8"))
    assert len(value["branches"]) == 5
    assert {row["strategy"] for row in value["branches"]} == set(audit.STRATEGIES)
    assert all(
        audit._sha256(Path(row["checkpoint"])) == row["checkpoint_sha256"]
        for row in value["branches"]
    )


def test_checkpoint_validation_enforces_earliest_equal_tie(tmp_path) -> None:
    """Equal optima must retain the earliest epoch for every direction."""
    branches = []
    histories = {}
    for strategy in audit.STRATEGIES:
        checkpoint = tmp_path / f"{strategy}.ckpt"
        torch.save({"epoch": 1, "state_dict": {}}, checkpoint)
        values = [1.0, 3.0, 3.0] if audit.DIRECTIONS[strategy] == "maximize" else [3.0, 1.0, 1.0]
        histories[audit.MONITORS[strategy]] = [
            SimpleNamespace(value=value, timestamp=index, step=index)
            for index, value in enumerate(values)
        ]
        branches.append(
            {
                "strategy": strategy,
                "monitor": audit.MONITORS[strategy],
                "monitor_value": values[1],
                "selected_epoch": 1,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": audit._sha256(checkpoint),
            }
        )
    path = tmp_path / "branches.json"
    path.write_text(json.dumps({"branches": branches}), encoding="utf-8")
    client = SimpleNamespace(get_metric_history=lambda run_id, monitor: histories[monitor])
    assert (
        len(
            audit._validate_branch_manifest(
                path, expected_epochs=3, client=client, run_id="synthetic"
            )
        )
        == 5
    )
    branches[0]["selected_epoch"] = 2
    path.write_text(json.dumps({"branches": branches}), encoding="utf-8")
    with pytest.raises(ValueError, match="Earliest-equal"):
        audit._validate_branch_manifest(path, expected_epochs=3, client=client, run_id="synthetic")


def test_training_marker_branches_must_equal_validated_manifest(tmp_path) -> None:
    """Freeze input must use the authenticated branch rows, not copied marker claims."""
    branches = []
    for strategy in audit.STRATEGIES:
        checkpoint = tmp_path / f"{strategy}.ckpt"
        torch.save({"epoch": 0, "state_dict": {}}, checkpoint)
        branches.append(
            {
                "strategy": strategy,
                "monitor": audit.MONITORS[strategy],
                "monitor_value": 1.0,
                "selected_epoch": 0,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": audit._sha256(checkpoint),
            }
        )
    manifest = tmp_path / "branches.json"
    manifest.write_text(json.dumps({"branches": branches}), encoding="utf-8")
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text("{}", encoding="utf-8")
    trajectory = {
        "trajectory_index": 0,
        "model": "ae",
        "candidate_id": "000",
        "reporting_seed": 1001,
        "params_sha256": "params",
    }
    audit_design = {
        "campaign_id": "campaign",
        "audit_code_commit": "commit",
    }
    marker = {
        **trajectory,
        **audit_design,
        "branch_manifest": str(manifest),
        "branch_manifest_sha256": audit._sha256(manifest),
        "trajectory_fingerprint": str(fingerprint),
        "trajectory_fingerprint_sha256": audit._sha256(fingerprint),
        "branches": branches,
    }
    marker_path = tmp_path / "training.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert (
        audit._resume_training_marker(marker_path, audit_design, trajectory)["branches"]
        == branches
    )
    marker["branches"] = deepcopy(branches)
    marker["branches"][0]["checkpoint_sha256"] = "f" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        audit._resume_training_marker(marker_path, audit_design, trajectory)


def test_evaluation_rows_must_match_every_frozen_checkpoint_field(tmp_path) -> None:
    """Resume and collection validation must reject checkpoint identity substitution."""
    trajectory = {
        "trajectory_index": 0,
        "model": "ae",
        "candidate_id": "000",
        "reporting_seed": 1001,
    }
    frozen = {
        strategy: {
            "strategy": strategy,
            "checkpoint": str((tmp_path / f"{strategy}.ckpt").resolve()),
            "checkpoint_sha256": f"{index + 1:064x}",
            "monitor": audit.MONITORS[strategy],
            "selected_epoch": index,
            "monitor_value": float(index),
        }
        for index, strategy in enumerate(audit.STRATEGIES)
    }
    rows = []
    for strategy, metric in product(audit.STRATEGIES, audit.METRICS):
        rows.append(
            {
                **trajectory,
                **frozen[strategy],
                "intervention": "synthetic",
                "metric": metric,
                "value": 0.5,
                "validation_derived_threshold": 1.0,
            }
        )
    path = tmp_path / "evaluation.csv"
    audit._write_csv(path, rows)
    assert len(audit._validate_evaluation_rows(path, trajectory, ["synthetic"], frozen)) == 10
    rows[0]["checkpoint_sha256"] = "0" * 64
    audit._write_csv(path, rows)
    with pytest.raises(ValueError, match="frozen-checkpoint"):
        audit._validate_evaluation_rows(path, trajectory, ["synthetic"], frozen)


def test_trajectory_fingerprint_records_required_determinism_fields(tmp_path) -> None:
    """The canary artifact must cover initialization, order, state, metrics, and RNG."""
    output = tmp_path / "fingerprint.json"
    callback = TrajectoryFingerprint(output)
    module = torch.nn.Linear(2, 1)
    trainer = SimpleNamespace(
        callback_metrics={"val/metric": torch.tensor(1.25)},
        current_epoch=0,
    )
    callback.on_fit_start(trainer, module)
    callback.on_train_batch_start(trainer, module, {"x": torch.tensor([[1.0, 2.0]])}, 0)
    callback.on_validation_end(trainer, module)
    callback.on_fit_end(trainer, module)
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["initial_model_state_sha256"] == value["final_model_state_sha256"]
    assert len(value["train_batch_sha256"]) == 1
    assert value["epochs"][0]["metrics"]["val/metric"] == pytest.approx(1.25)
    assert {"initial_rng", "final_rng"}.issubset(value)


def test_canary_gate_rejects_fingerprint_divergence(tmp_path) -> None:
    """Production gate validation must fail if checkpointing changes a trajectory."""
    control = tmp_path / "control.json"
    checkpoint = tmp_path / "checkpoint.json"
    control.write_text('{"state":"same"}', encoding="utf-8")
    checkpoint.write_text('{"state":"same"}', encoding="utf-8")
    marker = {
        "audit_sha256": "audit-hash",
        "audit_code_commit": "audit-commit",
        "equivalent": True,
        "control_fingerprint": str(control),
        "control_fingerprint_sha256": audit._sha256(control),
        "checkpoint_fingerprint": str(checkpoint),
        "checkpoint_fingerprint_sha256": audit._sha256(checkpoint),
    }
    path = tmp_path / "canary" / "trajectory_fingerprint_canary.json"
    path.parent.mkdir()
    path.write_text(json.dumps(marker), encoding="utf-8")
    assert (
        audit._validate_canary(
            tmp_path,
            "audit-hash",
            {"audit_code_commit": "audit-commit"},
        )["equivalent"]
        is True
    )
    checkpoint.write_text('{"state":"changed"}', encoding="utf-8")
    marker["checkpoint_fingerprint_sha256"] = audit._sha256(checkpoint)
    path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        audit._validate_canary(
            tmp_path,
            "audit-hash",
            {"audit_code_commit": "audit-commit"},
        )


def test_generated_slurm_scripts_are_packed_and_resource_exact(tmp_path) -> None:
    """Generated GPU and CPU launchers should match bounded Clariden resources."""
    audit._write_slurm_scripts(tmp_path)
    scripts = tmp_path / "slurm"
    train = (scripts / "train_packed.sh").read_text(encoding="utf-8")
    evaluate = (scripts / "evaluate_packed.sh").read_text(encoding="utf-8")
    canary = (scripts / "debug_fingerprint_canary.sh").read_text(encoding="utf-8")
    timing = (scripts / "production_timing_canary.sh").read_text(encoding="utf-8")
    freeze = (scripts / "freeze_checkpoints.sh").read_text(encoding="utf-8")
    collect = (scripts / "collect.sh").read_text(encoding="utf-8")
    analyze = (scripts / "analyze.sh").read_text(encoding="utf-8")
    for script in (train, evaluate, canary, timing, freeze, collect, analyze):
        assert "#SBATCH --account=a0166" in script
    for script in (train, evaluate, canary, timing):
        assert "--gpus-per-node=1 --mem=110G" in script
    assert "#SBATCH --partition=normal" in train
    assert "#SBATCH --array=0-47%16" in train
    assert "#SBATCH --cpus-per-task=72" in train
    assert "#SBATCH --time=04:00:00" in train
    assert "--cpus-per-task=72 --gpus-per-node=1 --mem=110G" in train
    assert "#SBATCH --array=0-47%16" in evaluate
    assert "#SBATCH --cpus-per-task=72" in evaluate
    assert "#SBATCH --time=04:00:00" in evaluate
    assert "SLURM_ARRAY_TASK_ID * 4 + slot" in train
    assert "--checkpoint-manifest-sha256" in evaluate
    assert "#SBATCH --partition=debug" in canary
    assert "#SBATCH --cpus-per-task=72" in canary
    assert "run-train" in timing
    assert "--trajectory-index 0" in timing
    assert "#SBATCH --time=04:00:00" in timing
    assert "freeze-checkpoints" in freeze
    assert "#SBATCH --time=02:00:00" in freeze
    assert "candidate_rank_audit.py collect" in collect
    assert "#SBATCH --time=02:00:00" in collect
    assert "candidate_rank_audit.py analyze" in analyze
    assert "--n-permutations 10000 --n-bootstrap 10000" in analyze
    assert "#SBATCH --time=04:00:00" in analyze
    assert {path.name for path in scripts.glob("*.sh")} == {
        "debug_fingerprint_canary.sh",
        "production_timing_canary.sh",
        "train_packed.sh",
        "freeze_checkpoints.sh",
        "evaluate_packed.sh",
        "collect.sh",
        "analyze.sh",
        "submit_workflow.sh",
    }
    for path in scripts.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)  # nosec B603 B607


def test_generated_slurm_workflow_has_exact_afterok_chain(tmp_path) -> None:
    """No heavy candidate-rank stage may race or execute on the login node."""
    audit._write_slurm_scripts(tmp_path)
    workflow = (tmp_path / "slurm" / "submit_workflow.sh").read_text(encoding="utf-8")
    expected_dependencies = (
        'timing_job=$(sbatch --parsable --dependency="afterok:${canary_job}"',
        'training_job=$(sbatch --parsable --dependency="afterok:${timing_job}"',
        'freeze_job=$(sbatch --parsable --dependency="afterok:${training_job}"',
        'evaluation_job=$(sbatch --parsable --dependency="afterok:${freeze_job}"',
        'collect_job=$(sbatch --parsable --dependency="afterok:${evaluation_job}"',
        'analysis_job=$(sbatch --parsable --dependency="afterok:${collect_job}"',
    )
    positions = [workflow.index(fragment) for fragment in expected_dependencies]
    assert positions == sorted(positions)
    assert workflow.count("sbatch --parsable") == 7
    assert "candidate_rank_audit.py" not in workflow
