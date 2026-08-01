from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from scripts import jetclr_campaign

_STAGE7_SPECS = jetclr_campaign.stage7_specs()


def _pairing_metrics(collapse_pass: bool = True) -> dict:
    """Return a complete synthetic pairing artifact."""
    return {
        "selection_score": 0.5,
        "raw_selection_score": 0.5,
        "closure_recall_at_10": 0.8,
        "mnn_coverage": 0.7,
        "embedding_finite_fraction": 1.0,
        "embedding_active_fraction": 0.9,
        "embedding_effective_rank": 40.0,
        "embedding_participation_rank": 35.0,
        "embedding_top_pc_fraction": 0.1,
        "value_smd_before_mean": 0.4,
        "value_smd_after_mean": 0.3,
        "occupancy_smd_before_mean": 0.2,
        "occupancy_smd_after_mean": 0.1,
        "collapse_pass": collapse_pass,
        "collapse_failures": [] if collapse_pass else ["low_effective_rank"],
        "projector_embedding_finite_fraction": 1.0,
        "projector_embedding_active_fraction": 0.9,
        "projector_embedding_effective_rank": 20.0,
        "projector_embedding_participation_rank": 18.0,
        "projector_embedding_top_pc_fraction": 0.15,
        "projector_collapse_pass": collapse_pass,
        "projector_collapse_failures": [] if collapse_pass else ["low_effective_rank"],
    }


def _anomaly_metrics(value: float = 0.8) -> dict:
    """Return a complete synthetic anomaly-utility artifact."""
    return {
        "macro_median_auroc": value,
        "macro_mean_auroc": value - 0.01,
        "worst_quartile_mean_auroc": value - 0.1,
        "per_dataset": {"signal": {"auroc": value, "auprc": 0.4}},
    }


def _stage4_training_metrics(variance_weight: float, covariance_weight: float) -> dict:
    """Return an internally consistent Stage-4 training-loss decomposition."""
    ntxent = 2.0
    variance = 0.4 if variance_weight else 0.0
    covariance = 0.2 if covariance_weight else 0.0
    variance_weighted = variance_weight * variance
    covariance_weighted = covariance_weight * covariance
    return {
        "train/loss_mean": ntxent + variance_weighted + covariance_weighted,
        "train/loss_ntxent": ntxent,
        "train/loss_encoder_variance": variance,
        "train/loss_encoder_covariance": covariance,
        "train/loss_encoder_variance_weighted": variance_weighted,
        "train/loss_encoder_covariance_weighted": covariance_weighted,
    }


def _write_manifest(root: Path) -> dict:
    """Write a minimal authenticated campaign manifest for unit tests."""
    specs = jetclr_campaign.canary_specs()
    stage1 = jetclr_campaign.stage1_specs()
    stage2 = jetclr_campaign.stage2_specs()
    stage3 = jetclr_campaign.stage3_specs()
    stage4 = jetclr_campaign.stage4_specs()
    stage5 = jetclr_campaign.stage5_specs()
    stage6 = jetclr_campaign.stage6_specs()
    stage7 = _STAGE7_SPECS
    manifest = {
        "schema_version": 1,
        "campaign_id": "jetclr_test_deadbeef",
        "git": {"commit": "deadbeef", "branch": "test", "source": "/source"},
        "deployment": {"path": "/deployment", "commit": "deadbeef"},
        "config": {"tree_sha256": "config"},
        "data": {"root": "/data", "tree_sha256": "data"},
        "environment": {"uv": "/uv", "venv": "/venv"},
        "canary": {
            "trials": specs,
            "design_sha256": jetclr_campaign._value_sha256(specs),
        },
        "stage1": {
            "seed": jetclr_campaign.STAGE1_SEED,
            "train_batches": jetclr_campaign.STAGE1_TRAIN_BATCHES,
            "candidates": stage1,
            "design_sha256": jetclr_campaign._value_sha256(stage1),
        },
        "stage2": {
            "seed": jetclr_campaign.STAGE2_SEED,
            "full_epochs": 1,
            "source_campaign_id": jetclr_campaign.STAGE2_SOURCE_CAMPAIGN,
            "source_summary_sha256": jetclr_campaign.STAGE2_SOURCE_SUMMARY_SHA256,
            "source_summary_csv_sha256": jetclr_campaign.STAGE2_SOURCE_SUMMARY_CSV_SHA256,
            "candidates": stage2,
            "design_sha256": jetclr_campaign._value_sha256(stage2),
        },
        "stage3": {
            "seed": jetclr_campaign.STAGE3_SEED,
            "full_epochs": 1,
            "source_campaign_id": jetclr_campaign.STAGE3_SOURCE_CAMPAIGN,
            "source_candidate_id": jetclr_campaign.STAGE3_SOURCE_CANDIDATE_ID,
            "source_candidate_spec_sha256": jetclr_campaign.STAGE3_SOURCE_SPEC_SHA256,
            "frozen_primary_params": jetclr_campaign._stage3_primary_params(),
            "candidates": stage3,
            "design_sha256": jetclr_campaign._value_sha256(stage3),
        },
        "stage4": {
            "seed": jetclr_campaign.STAGE4_SEED,
            "full_epochs": 1,
            "source_campaign_id": jetclr_campaign.STAGE4_SOURCE_CAMPAIGN,
            "source_summary_sha256": jetclr_campaign.STAGE4_SOURCE_SUMMARY_SHA256,
            "source_summary_csv_sha256": jetclr_campaign.STAGE4_SOURCE_SUMMARY_CSV_SHA256,
            "candidates": stage4,
            "design_sha256": jetclr_campaign._value_sha256(stage4),
        },
        "stage5": {
            "seeds": list(jetclr_campaign.STAGE5_SEEDS),
            "full_epochs": 1,
            "source_campaign_id": jetclr_campaign.STAGE5_SOURCE_CAMPAIGN,
            "source_summary_sha256": jetclr_campaign.STAGE5_SOURCE_SUMMARY_SHA256,
            "source_summary_csv_sha256": jetclr_campaign.STAGE5_SOURCE_SUMMARY_CSV_SHA256,
            "candidates": stage5,
            "design_sha256": jetclr_campaign._value_sha256(stage5),
        },
        "stage6": {
            "seeds": list(jetclr_campaign.STAGE6_SEEDS),
            "full_epochs": jetclr_campaign.STAGE6_EPOCHS,
            "milestone_epochs": list(jetclr_campaign.STAGE6_MILESTONES),
            "source_campaign_id": jetclr_campaign.STAGE6_SOURCE_CAMPAIGN,
            "source_promotions": {"layers2": True, "official_projector": True},
            "candidates": stage6,
            "design_sha256": jetclr_campaign._value_sha256(stage6),
        },
        "stage7": {
            "train": False,
            "test": False,
            "milestone_epochs": list(jetclr_campaign.STAGE7_MILESTONES),
            "source_campaign_id": jetclr_campaign.STAGE7_SOURCE_CAMPAIGN,
            "candidates": stage7,
            "design_sha256": jetclr_campaign._value_sha256(stage7),
        },
    }
    manifest["manifest_payload_sha256"] = jetclr_campaign._value_sha256(manifest)
    jetclr_campaign._atomic_json(root / "campaign.json", manifest)
    return manifest


def _write_stage4_results(root: Path, manifest: dict) -> list[Path]:
    """Write twelve authenticated synthetic Stage-4 result bundles."""
    checkpoints = []
    for spec in manifest["stage4"]["candidates"]:
        candidate_id = spec["candidate_id"]
        trial_root = root / "stage4" / f"candidate_{candidate_id:03d}"
        trial_root.mkdir(parents=True)
        metrics = trial_root / "metrics.csv"
        variance_weight = spec["regularization_params"]["algorithm.encoder_variance_weight"]
        covariance_weight = spec["regularization_params"]["algorithm.encoder_covariance_weight"]
        training = _stage4_training_metrics(variance_weight, covariance_weight)
        metrics.write_text(
            ",".join(training) + "\n" + ",".join(str(value) for value in training.values()) + "\n",
            encoding="utf-8",
        )
        pairing_path = trial_root / "pairing_diagnostics.json"
        pairing = _pairing_metrics()
        pairing["embedding_effective_rank"] += candidate_id
        pairing["raw_selection_score"] += candidate_id / 100.0
        if candidate_id == 2:
            pairing["mnn_coverage"] = 0.0
        if candidate_id == 3:
            pairing["value_smd_after_mean"] = 0.5
        pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
        anomaly_path = trial_root / "embedding_anomaly.json"
        anomaly = _anomaly_metrics(0.7 + candidate_id / 100.0)
        anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
        checkpoint = trial_root / "last.ckpt"
        checkpoint.write_bytes(f"checkpoint {candidate_id}".encode())
        checkpoints.append(checkpoint)
        artifacts = {
            "training_csv": {"path": str(metrics), "sha256": jetclr_campaign._sha256(metrics)},
            "pairing_json": {
                "path": str(pairing_path),
                "sha256": jetclr_campaign._sha256(pairing_path),
            },
            "anomaly_json": {
                "path": str(anomaly_path),
                "sha256": jetclr_campaign._sha256(anomaly_path),
            },
            "last_checkpoint": {
                "path": str(checkpoint),
                "sha256": jetclr_campaign._sha256(checkpoint),
            },
        }
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "candidate_id": candidate_id,
            "spec_sha256": spec["spec_sha256"],
            "source_campaign_id": jetclr_campaign.STAGE4_SOURCE_CAMPAIGN,
            "source_candidate_id": spec["source_candidate_id"],
            "source_candidate_spec_sha256": spec["source_candidate_spec_sha256"],
            "artifacts": artifacts,
            "training_metrics": training,
            "pairing_metrics": pairing,
            "anomaly_metrics": anomaly,
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial_root / "result.json", result)
    return checkpoints


def _write_stage5_results(root: Path, manifest: dict) -> None:
    """Write eight authenticated fresh-seed confirmation bundles."""
    for spec in manifest["stage5"]["candidates"]:
        trial_root = root / "stage5" / f"candidate_{spec['candidate_id']:03d}"
        trial_root.mkdir(parents=True)
        weights = spec["regularization_params"]
        training = _stage4_training_metrics(
            weights["algorithm.encoder_variance_weight"],
            weights["algorithm.encoder_covariance_weight"],
        )
        metrics = trial_root / "metrics.csv"
        metrics.write_text("metrics", encoding="utf-8")
        pairing = _pairing_metrics()
        anomaly = _anomaly_metrics()
        if not spec["is_architecture_control"]:
            pairing["embedding_effective_rank"] = 46.0
            pairing["embedding_participation_rank"] = 40.25
            pairing["raw_selection_score"] = 0.49
            anomaly["macro_mean_auroc"] -= 0.005
            anomaly["worst_quartile_mean_auroc"] -= 0.005
        pairing_path = trial_root / "pairing_diagnostics.json"
        anomaly_path = trial_root / "embedding_anomaly.json"
        pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
        anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
        checkpoint = trial_root / "last.ckpt"
        checkpoint.write_bytes(f"fresh {spec['candidate_id']}".encode())
        artifacts = {
            name: {"path": str(path), "sha256": jetclr_campaign._sha256(path)}
            for name, path in {
                "training_csv": metrics,
                "pairing_json": pairing_path,
                "anomaly_json": anomaly_path,
                "last_checkpoint": checkpoint,
            }.items()
        }
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "candidate_id": spec["candidate_id"],
            "spec_sha256": spec["spec_sha256"],
            "source_campaign_id": jetclr_campaign.STAGE5_SOURCE_CAMPAIGN,
            "source_candidate_id": spec["source_candidate_id"],
            "source_candidate_spec_sha256": spec["source_candidate_spec_sha256"],
            "artifacts": artifacts,
            "training_metrics": training,
            "pairing_metrics": pairing,
            "anomaly_metrics": anomaly,
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial_root / "result.json", result)


def _write_stage6_results(root: Path, manifest: dict) -> None:
    """Write authenticated 16-epoch synthetic pilot bundles."""
    for spec in manifest["stage6"]["candidates"]:
        trial = root / "stage6" / f"candidate_{spec['candidate_id']:03d}"
        trial.mkdir(parents=True)
        weights = spec["regularization_params"]
        training = _stage4_training_metrics(
            weights["algorithm.encoder_variance_weight"],
            weights["algorithm.encoder_covariance_weight"],
        )
        metrics = trial / "metrics.csv"
        pairing_path = trial / "pairing_diagnostics.json"
        anomaly_path = trial / "embedding_anomaly.json"
        checkpoint_dir = trial / "checkpoints"
        checkpoint_dir.mkdir()
        metrics.write_text("metrics", encoding="utf-8")
        pairing = _pairing_metrics()
        anomaly = _anomaly_metrics()
        if not spec["is_architecture_control"]:
            pairing["embedding_effective_rank"] = 46.0
        pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
        anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
        inventory = []
        for index in range(16):
            checkpoint = checkpoint_dir / f"epoch-{index:02d}.ckpt"
            checkpoint.write_bytes(f"{spec['candidate_id']}/{index}".encode())
            inventory.append(
                {
                    "completed_epoch": index + 1,
                    "epoch_index": index,
                    "path": str(checkpoint),
                    "sha256": jetclr_campaign._sha256(checkpoint),
                    "is_milestone": index + 1 in jetclr_campaign.STAGE6_MILESTONES,
                }
            )
        last = checkpoint_dir / "last.ckpt"
        last.write_bytes(b"last")
        artifacts = {
            name: {"path": str(path), "sha256": jetclr_campaign._sha256(path)}
            for name, path in {
                "training_csv": metrics,
                "pairing_json": pairing_path,
                "anomaly_json": anomaly_path,
                "last_checkpoint": last,
            }.items()
        }
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "candidate_id": spec["candidate_id"],
            "spec_sha256": spec["spec_sha256"],
            "source_campaign_id": jetclr_campaign.STAGE6_SOURCE_CAMPAIGN,
            "source_candidate_id": spec["source_candidate_id"],
            "source_candidate_spec_sha256": spec["source_candidate_spec_sha256"],
            "artifacts": artifacts,
            "checkpoint_inventory": inventory,
            "training_metrics": training,
            "pairing_metrics": pairing,
            "anomaly_metrics": anomaly,
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial / "result.json", result)


def test_canary_specs_are_deterministic_and_cover_four_distinct_recipes() -> None:
    """The canary design must be stable and exercise distinct risk surfaces."""
    first = jetclr_campaign.canary_specs()
    second = jetclr_campaign.canary_specs()

    assert first == second
    assert [item["trial_id"] for item in first] == [0, 1, 2, 3]
    assert len({item["name"] for item in first}) == 4
    assert len({item["spec_sha256"] for item in first}) == 4
    assert any("algorithm.detector_smearing=null" in item["overrides"] for item in first)
    assert any("algorithm.model.d_model=256" in item["overrides"] for item in first)


def test_canary_disables_evaluation_callbacks_without_deleting_config_group() -> None:
    """The physics experiment's evaluation group must remain Hydra-composable."""
    source = Path(jetclr_campaign.__file__).read_text(encoding="utf-8")

    assert '"evaluation.callbacks=null"' in source
    assert '"evaluation=null"' not in source
    assert '"extras.enforce_tags=false"' in source
    assert '"callbacks.rich_progress_bar=null"' in source


def test_stage1_specs_are_fixed_sobol_design_with_anchor_ablations() -> None:
    """Stage 1 must reproducibly combine eight anchors and forty Sobol points."""
    first = jetclr_campaign.stage1_specs()
    second = jetclr_campaign.stage1_specs()

    assert first == second
    assert len(first) == 48
    assert [item["candidate_id"] for item in first] == list(range(48))
    assert {item["kind"] for item in first[:8]} == {"anchor"}
    assert {item["kind"] for item in first[8:]} == {"sobol"}
    assert {item["seed"] for item in first} == {123}
    assert {item["train_batches"] for item in first} == {256}
    assert len({item["spec_sha256"] for item in first}) == 48
    assert first[0]["name"] == "production"
    assert first[4]["name"] == "no_augmentation"


def test_stage1_launchers_pack_48_trials_and_chain_collector(tmp_path: Path) -> None:
    """Twelve array tasks should pack four trials and expose an afterok collector."""
    manifest = _write_manifest(tmp_path)
    launchers = jetclr_campaign._write_stage1_launchers(tmp_path, manifest)
    stage1 = launchers["stage1"].read_text(encoding="utf-8")
    collector = launchers["collector"].read_text(encoding="utf-8")
    submitter = launchers["submitter"].read_text(encoding="utf-8")

    assert "#SBATCH --partition=normal" in stage1
    assert "#SBATCH --array=0-11%4" in stage1
    assert "base=$((SLURM_ARRAY_TASK_ID * 4))" in stage1
    assert "run-stage1" in stage1
    assert "#SBATCH --gpus-per-node" not in collector
    assert "collect-stage1" in collector
    assert 'dependency="afterok:$stage1_job"' in submitter


def test_stage2_specs_promote_frozen_sources_and_targeted_refinements() -> None:
    """Stage 2 must preserve seven source candidates and add five fixed neighbors."""
    first = jetclr_campaign.stage2_specs()
    second = jetclr_campaign.stage2_specs()

    assert first == second
    assert len(first) == 12
    assert [item["source_candidate_id"] for item in first[:7]] == list(
        jetclr_campaign.STAGE2_PROMOTED_IDS
    )
    assert {item["kind"] for item in first[:7]} == {"stage1_promoted"}
    assert {item["kind"] for item in first[7:]} == {"targeted_refinement"}
    assert {item["seed"] for item in first} == {123}
    assert {item["full_epochs"] for item in first} == {1}
    assert len({item["spec_sha256"] for item in first}) == 12
    refinements = first[7:]
    assert {item["params"]["data.batch_size"] for item in refinements} == {2048, 4096, 8192}
    assert min(item["params"]["algorithm.optimizer.lr"] for item in refinements) == 5e-5
    assert max(item["params"]["algorithm.optimizer.lr"] for item in refinements) == 5e-4
    assert {item["params"]["algorithm.loss.temperature"] for item in refinements} == {
        0.05,
        0.1,
        0.2,
    }


def test_stage2_launchers_pack_twelve_trials_and_chain_collector(tmp_path: Path) -> None:
    """Three array tasks should pack four full-epoch trials and an afterok collector."""
    manifest = _write_manifest(tmp_path)
    launchers = jetclr_campaign._write_stage2_launchers(tmp_path, manifest)
    stage2 = launchers["stage2"].read_text(encoding="utf-8")
    collector = launchers["collector"].read_text(encoding="utf-8")
    submitter = launchers["submitter"].read_text(encoding="utf-8")

    assert "#SBATCH --array=0-2%3" in stage2
    assert "run-stage2" in stage2
    assert "base=$((SLURM_ARRAY_TASK_ID * 4))" in stage2
    assert "#SBATCH --gpus-per-node" not in collector
    assert "collect-stage2" in collector
    assert 'dependency="afterok:$stage2_job"' in submitter


def test_run_stage2_uses_one_full_epoch_without_train_batch_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 2 must omit a train-batch cap while retaining bounded validation metrics."""
    _write_manifest(tmp_path)
    monkeypatch.setattr(jetclr_campaign, "_assert_runtime", lambda _: tmp_path)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        output = tmp_path / "stage2" / "candidate_000" / "output"
        metrics = output / "csv" / "version_0" / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("train/loss_mean\n2.5\n", encoding="utf-8")
        pairing = output / "metrics" / "pairing_diagnostics" / "last"
        pairing.mkdir(parents=True)
        pairing_metrics = _pairing_metrics()
        pairing_metrics["value_smd_after_mean"] = None
        pairing_metrics["occupancy_smd_after_mean"] = None
        (pairing / "pairing_diagnostics.json").write_text(
            json.dumps(pairing_metrics), encoding="utf-8"
        )
        anomaly = output / "metrics" / "embedding_anomaly" / "last"
        anomaly.mkdir(parents=True)
        (anomaly / "embedding_anomaly.json").write_text(
            json.dumps(_anomaly_metrics()), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jetclr_campaign.subprocess, "run", fake_run)
    result_path = jetclr_campaign.run_stage2(tmp_path, 0)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    command = observed["command"]

    assert "trainer.min_epochs=1" in command
    assert "trainer.max_epochs=1" in command
    assert not any("limit_train_batches" in value for value in command)
    assert "+trainer.limit_val_batches=0" in command
    assert "data.max_normal_eval_batches=8" in command
    assert "evaluation.callbacks.pairing_diagnostics.max_events_per_dataset=8192" in command
    assert result["source_campaign_id"] == jetclr_campaign.STAGE2_SOURCE_CAMPAIGN
    assert result["source_candidate_id"] == jetclr_campaign.STAGE2_PROMOTED_IDS[0]


def test_stage3_specs_freeze_primary_and_cover_requested_architectures() -> None:
    """Stage 3 must vary only architecture over the exact Stage-2 primary recipe."""
    specs = jetclr_campaign.stage3_specs()

    assert specs == jetclr_campaign.stage3_specs()
    assert len(specs) == 12
    assert {item["seed"] for item in specs} == {123}
    assert all(
        item["frozen_primary_params"] == specs[0]["frozen_primary_params"] for item in specs
    )
    assert specs[0]["name"] == "current_exact"
    assert specs[1]["architecture_params"]["algorithm.model.pooling"] == "sum"
    assert specs[2]["architecture_params"]["algorithm.model.post_pool_norm"] is False
    assert specs[3]["architecture_params"]["algorithm.model.norm_first"] is False
    assert specs[6]["architecture_params"]["algorithm.model.d_model"] == 512
    assert specs[7]["architecture_params"]["algorithm.model.out_dim"] == 64
    assert specs[8]["architecture_params"]["algorithm.projector.nodes"] == [256]
    assert specs[9]["architecture_params"]["algorithm.projector.batchnorm"] is False
    assert specs[10]["architecture_params"]["algorithm.model.n_layers"] == 2
    assert specs[11]["architecture_params"]["algorithm.model.n_layers"] == 6
    assert all("algorithm.projector.in_dim" not in item["params"] for item in specs)
    assert {item["source_candidate_spec_sha256"] for item in specs} == {
        jetclr_campaign.STAGE3_SOURCE_SPEC_SHA256
    }


def test_stage3_launchers_pack_twelve_trials_and_chain_collector(tmp_path: Path) -> None:
    """Stage 3 should reuse the reviewed three-node four-GPU packing contract."""
    manifest = _write_manifest(tmp_path)
    launchers = jetclr_campaign._write_stage3_launchers(tmp_path, manifest)
    stage3 = launchers["stage3"].read_text(encoding="utf-8")
    collector = launchers["collector"].read_text(encoding="utf-8")
    submitter = launchers["submitter"].read_text(encoding="utf-8")

    assert "#SBATCH --array=0-2%3" in stage3
    assert "run-stage3" in stage3
    assert "collect-stage3" in collector
    assert 'dependency="afterok:$stage3_job"' in submitter


def test_run_stage3_full_epoch_authenticates_projector_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 3 must omit train limits and require encoder plus projector diagnostics."""
    _write_manifest(tmp_path)
    monkeypatch.setattr(jetclr_campaign, "_assert_runtime", lambda _: tmp_path)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        output = tmp_path / "stage3" / "candidate_000" / "output"
        metrics = output / "csv" / "version_0" / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("train/loss_mean\n2.5\n", encoding="utf-8")
        pairing = output / "metrics" / "pairing_diagnostics" / "last"
        pairing.mkdir(parents=True)
        (pairing / "pairing_diagnostics.json").write_text(
            json.dumps(_pairing_metrics()), encoding="utf-8"
        )
        anomaly = output / "metrics" / "embedding_anomaly" / "last"
        anomaly.mkdir(parents=True)
        (anomaly / "embedding_anomaly.json").write_text(
            json.dumps(_anomaly_metrics()), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jetclr_campaign.subprocess, "run", fake_run)
    result_path = jetclr_campaign.run_stage3(tmp_path, 0)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert not any("limit_train_batches" in value for value in observed["command"])
    assert "+algorithm.model.norm_first=true" in observed["command"]
    assert "+algorithm.model.post_pool_norm=true" in observed["command"]
    assert not any("algorithm.projector.in_dim=" in value for value in observed["command"])
    assert result["pairing_metrics"]["projector_collapse_pass"] is True


def test_stage4_specs_freeze_two_sources_and_cross_exact_regularization_grid() -> None:
    """Stage 4 must retain both promoted identities and include one control per architecture."""
    specs = jetclr_campaign.stage4_specs()

    assert specs == jetclr_campaign.stage4_specs()
    assert len(specs) == 12
    assert [item["candidate_id"] for item in specs] == list(range(12))
    assert {item["seed"] for item in specs} == {123}
    assert [item["source_candidate_id"] for item in specs] == [10] * 6 + [9] * 6
    assert [item["source_candidate_name"] for item in specs] == ["layers2"] * 6 + [
        "official_projector"
    ] * 6
    assert {item["source_candidate_spec_sha256"] for item in specs[:6]} == {
        "0ef568e754d5530bd517b55ecbd6d0b19aa9c84f38bff1dfcc5b80a827b73856"
    }
    assert {item["source_candidate_spec_sha256"] for item in specs[6:]} == {
        "c19d089c7261c9c2055fbc218fb44333b7e99bd9020937a3282cbe5f2de50d67"
    }
    expected_weights = [(0.0, 0.0), (0.1, 0.0), (0.5, 0.0), (1.0, 0.0), (0.5, 0.005), (0.5, 0.02)]
    for block in (specs[:6], specs[6:]):
        assert [
            (
                item["regularization_params"]["algorithm.encoder_variance_weight"],
                item["regularization_params"]["algorithm.encoder_covariance_weight"],
            )
            for item in block
        ] == expected_weights
        assert [item["is_architecture_control"] for item in block] == [
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        assert all(
            {
                key: value
                for key, value in item["params"].items()
                if key not in item["regularization_params"]
            }
            == item["frozen_stage3_params"]
            for item in block
        )


def test_stage4_exact_candidate_overrides_compose() -> None:
    """Every frozen Stage-4 command-line design must compose through Hydra."""
    config_dir = Path(jetclr_campaign.__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        for spec in jetclr_campaign.stage4_specs():
            config = compose(
                config_name="train",
                overrides=["experiment=physics/jetclr_pairing", "trainer=gpu", *spec["overrides"]],
            )
            assert config.seed == 123
            assert config.algorithm.model.n_layers in {2, 4}
            assert (
                config.algorithm.encoder_variance_weight
                == spec["regularization_params"]["algorithm.encoder_variance_weight"]
            )
            assert (
                config.algorithm.encoder_covariance_weight
                == spec["regularization_params"]["algorithm.encoder_covariance_weight"]
            )


def test_stage4_launchers_pack_twelve_trials_and_chain_collector(tmp_path: Path) -> None:
    """Stage 4 must pack four trials per node and collect only after all nodes succeed."""
    manifest = _write_manifest(tmp_path)
    launchers = jetclr_campaign._write_stage4_launchers(tmp_path, manifest)
    stage4 = launchers["stage4"].read_text(encoding="utf-8")
    collector = launchers["collector"].read_text(encoding="utf-8")
    submitter = launchers["submitter"].read_text(encoding="utf-8")

    assert "#SBATCH --array=0-2%3" in stage4
    assert "base=$((SLURM_ARRAY_TASK_ID * 4))" in stage4
    assert "run-stage4" in stage4
    assert "collect-stage4" in collector
    assert 'dependency="afterok:$stage4_job"' in submitter


def test_run_stage4_requires_loss_decomposition_and_authenticates_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 4 must persist all losses and the canonical final encoder checkpoint."""
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(jetclr_campaign, "_assert_runtime", lambda _: tmp_path)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        spec = manifest["stage4"]["candidates"][0]
        output = tmp_path / "stage4" / "candidate_000" / "output"
        metrics = output / "csv" / "version_0" / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        training = _stage4_training_metrics(0.0, 0.0)
        metrics.write_text(
            ",".join(training) + "\n" + ",".join(str(value) for value in training.values()) + "\n",
            encoding="utf-8",
        )
        pairing = output / "metrics" / "pairing_diagnostics" / "last"
        pairing.mkdir(parents=True)
        (pairing / "pairing_diagnostics.json").write_text(
            json.dumps(_pairing_metrics()), encoding="utf-8"
        )
        anomaly = output / "metrics" / "embedding_anomaly" / "last"
        anomaly.mkdir(parents=True)
        (anomaly / "embedding_anomaly.json").write_text(
            json.dumps(_anomaly_metrics()), encoding="utf-8"
        )
        checkpoint_dir = (
            tmp_path / "stage4" / "candidate_000" / "checkpoints" / "experiment" / "candidate_000"
        )
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "last.ckpt").write_bytes(b"canonical checkpoint")
        (checkpoint_dir / "last-v1.ckpt").write_bytes(b"duplicate checkpoint")
        assert spec["is_architecture_control"] is True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jetclr_campaign.subprocess, "run", fake_run)
    result_path = jetclr_campaign.run_stage4(tmp_path, 0)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert not any("limit_train_batches" in value for value in observed["command"])
    assert "test=false" in observed["command"]
    assert "algorithm.encoder_variance_weight=0" in observed["command"]
    assert result["training_metrics"]["train/loss_ntxent"] == 2.0
    assert Path(result["artifacts"]["last_checkpoint"]["path"]).name == "last.ckpt"
    assert result["artifacts"]["last_checkpoint"]["sha256"] == jetclr_campaign._sha256(
        Path(result["artifacts"]["last_checkpoint"]["path"])
    )


def test_run_stage1_composes_bounded_training_and_validates_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One trial must use the frozen budget and parse all three metric artifacts."""
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(jetclr_campaign, "_assert_runtime", lambda _: tmp_path)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        output = tmp_path / "stage1" / "candidate_000" / "output"
        metrics = output / "csv" / "version_0" / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("train/loss_mean\n2.5\n", encoding="utf-8")
        pairing = output / "metrics" / "pairing_diagnostics" / "last"
        pairing.mkdir(parents=True)
        (pairing / "pairing_diagnostics.json").write_text(
            json.dumps(_pairing_metrics()), encoding="utf-8"
        )
        anomaly = output / "metrics" / "embedding_anomaly" / "last"
        anomaly.mkdir(parents=True)
        (anomaly / "embedding_anomaly.json").write_text(
            json.dumps(_anomaly_metrics()), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jetclr_campaign.subprocess, "run", fake_run)
    result_path = jetclr_campaign.run_stage1(tmp_path, 0)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    command = observed["command"]

    assert "+trainer.limit_train_batches=256" in command
    assert "+trainer.limit_val_batches=0" in command
    assert "data.max_val_batches=4" in command
    assert "data.max_normal_eval_batches=8" in command
    assert "evaluation.callbacks.pairing_diagnostics.max_events_per_dataset=8192" in command
    assert "evaluation.callbacks.embedding_anomaly.reference_size=8192" in command
    assert "evaluation.callbacks.embedding_anomaly.max_query_events=8192" in command
    assert "callbacks.rich_progress_bar=null" in command
    assert "extras.enforce_tags=false" in command
    assert (tmp_path / "stage1" / "candidate_000" / "output").is_dir()
    assert result["pairing_metrics"]["collapse_pass"] is True
    assert result["anomaly_metrics"]["worst_quartile_mean_auroc"] == pytest.approx(0.7)


def test_campaign_manifest_detects_tampering(tmp_path: Path) -> None:
    """Changing any authenticated manifest field must fail closed."""
    manifest = _write_manifest(tmp_path)
    assert jetclr_campaign._load_campaign(tmp_path)["campaign_id"] == manifest["campaign_id"]

    path = tmp_path / "campaign.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign_id"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        jetclr_campaign._load_campaign(tmp_path)


def test_packed_launcher_assigns_one_gpu_to_each_trial(tmp_path: Path) -> None:
    """The launcher must use every GPU without recursively submitting work."""
    manifest = _write_manifest(tmp_path)
    launcher = jetclr_campaign._write_launcher(tmp_path, manifest)
    text = launcher.read_text(encoding="utf-8")

    assert "#SBATCH --gpus-per-node=4" in text
    assert "#SBATCH --ntasks=4" in text
    assert "for trial_id in 0 1 2 3" in text
    assert "srun --exclusive --ntasks=1" in text
    assert "--gpus-per-node=1" in text
    assert "sbatch " not in text


def test_collect_validates_and_summarizes_all_trials(tmp_path: Path) -> None:
    """Collection should authenticate and summarize four complete trials."""
    manifest = _write_manifest(tmp_path)
    for spec in manifest["canary"]["trials"]:
        trial_root = tmp_path / "canary" / f"{spec['trial_id']:02d}_{spec['name']}"
        metrics = trial_root / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        with metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train/loss_mean"])
            writer.writeheader()
            writer.writerow({"train/loss_mean": 1.0 + spec["trial_id"]})
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "spec_sha256": spec["spec_sha256"],
            "metrics_csv": str(metrics),
            "metrics_csv_sha256": jetclr_campaign._sha256(metrics),
            "metrics": {"train/loss_mean": 1.0 + spec["trial_id"]},
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial_root / "result.json", result)

    output = jetclr_campaign.collect(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["n_trials"] == 4
    assert jetclr_campaign._sha256(Path(summary["summary_csv"])) == summary["summary_csv_sha256"]


def test_collect_rejects_modified_metrics(tmp_path: Path) -> None:
    """Collection must reject metrics modified after trial completion."""
    manifest = _write_manifest(tmp_path)
    spec = manifest["canary"]["trials"][0]
    trial_root = tmp_path / "canary" / f"{spec['trial_id']:02d}_{spec['name']}"
    metrics = trial_root / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("train/loss_mean\n1.0\n", encoding="utf-8")
    result = {
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "spec_sha256": spec["spec_sha256"],
        "metrics_csv": str(metrics),
        "metrics_csv_sha256": "0" * 64,
        "metrics": {"train/loss_mean": 1.0},
    }
    result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
    jetclr_campaign._atomic_json(trial_root / "result.json", result)
    with pytest.raises(ValueError, match="Metrics artifact mismatch"):
        jetclr_campaign.collect(tmp_path)


@pytest.mark.parametrize("all_collapsed", [False, True])
def test_collect_stage1_authenticates_ranks_and_applies_collapse_gate(
    tmp_path: Path, all_collapsed: bool
) -> None:
    """Collection should rank eligible utility while retaining collapsed candidates."""
    manifest = _write_manifest(tmp_path)
    for spec in manifest["stage1"]["candidates"]:
        trial_root = tmp_path / "stage1" / f"candidate_{spec['candidate_id']:03d}"
        trial_root.mkdir(parents=True)
        metrics = trial_root / "metrics.csv"
        pairing_path = trial_root / "pairing_diagnostics.json"
        anomaly_path = trial_root / "embedding_anomaly.json"
        metrics.write_text("train/loss_mean\n2.0\n", encoding="utf-8")
        collapse = False if all_collapsed else spec["candidate_id"] != 1
        pairing = _pairing_metrics(collapse)
        anomaly = _anomaly_metrics(0.99 if spec["candidate_id"] == 1 else 0.75)
        pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
        anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
        artifacts = {
            "training_csv": {
                "path": str(metrics),
                "sha256": jetclr_campaign._sha256(metrics),
            },
            "pairing_json": {
                "path": str(pairing_path),
                "sha256": jetclr_campaign._sha256(pairing_path),
            },
            "anomaly_json": {
                "path": str(anomaly_path),
                "sha256": jetclr_campaign._sha256(anomaly_path),
            },
        }
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "candidate_id": spec["candidate_id"],
            "spec_sha256": spec["spec_sha256"],
            "artifacts": artifacts,
            "training_metrics": {"train/loss_mean": 2.0},
            "pairing_metrics": pairing,
            "anomaly_metrics": anomaly,
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial_root / "result.json", result)

    output = jetclr_campaign.collect_stage1(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    with Path(summary["summary_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["n_candidates"] == 48
    expected_eligible = 0 if all_collapsed else 47
    assert summary["n_collapse_pass"] == expected_eligible
    assert summary["n_eligible"] == expected_eligible
    if all_collapsed:
        assert summary["status"] == "complete_no_eligible_candidates"
        assert summary["best_candidate_id"] is None
    else:
        assert summary["status"] == "complete"
        assert summary["best_candidate_id"] != 1
    collapsed = next(row for row in rows if row["candidate_id"] == "1")
    assert collapsed["eligible"] == "False"
    assert float(collapsed["worst_quartile_mean_auroc"]) == pytest.approx(0.89)


def test_collect_stage2_preserves_all_failed_gates_and_reports_pareto(tmp_path: Path) -> None:
    """Stage-2 collection must finish with no eligible candidate and no scalar winner."""
    manifest = _write_manifest(tmp_path)
    for spec in manifest["stage2"]["candidates"]:
        trial_root = tmp_path / "stage2" / f"candidate_{spec['candidate_id']:03d}"
        trial_root.mkdir(parents=True)
        metrics = trial_root / "metrics.csv"
        pairing_path = trial_root / "pairing_diagnostics.json"
        anomaly_path = trial_root / "embedding_anomaly.json"
        metrics.write_text("train/loss_mean\n2.0\n", encoding="utf-8")
        pairing = _pairing_metrics(collapse_pass=False)
        if spec["candidate_id"] == 0:
            pairing["embedding_effective_rank"] = 50.0
            pairing["raw_selection_score"] = 0.4
        elif spec["candidate_id"] == 1:
            pairing["embedding_effective_rank"] = 40.0
            pairing["raw_selection_score"] = 0.6
            pairing["occupancy_smd_after_mean"] = 0.3
        else:
            pairing["embedding_effective_rank"] = 20.0
            pairing["raw_selection_score"] = 0.2
            if spec["candidate_id"] == 2:
                pairing["value_smd_after_mean"] = None
                pairing["occupancy_smd_after_mean"] = None
        anomaly = _anomaly_metrics(0.8 if spec["candidate_id"] == 1 else 0.7)
        pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
        anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
        artifacts = {
            "training_csv": {
                "path": str(metrics),
                "sha256": jetclr_campaign._sha256(metrics),
            },
            "pairing_json": {
                "path": str(pairing_path),
                "sha256": jetclr_campaign._sha256(pairing_path),
            },
            "anomaly_json": {
                "path": str(anomaly_path),
                "sha256": jetclr_campaign._sha256(anomaly_path),
            },
        }
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "candidate_id": spec["candidate_id"],
            "spec_sha256": spec["spec_sha256"],
            "source_campaign_id": spec["source_campaign_id"],
            "source_candidate_id": spec["source_candidate_id"],
            "artifacts": artifacts,
            "training_metrics": {"train/loss_mean": 2.0},
            "pairing_metrics": pairing,
            "anomaly_metrics": anomaly,
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial_root / "result.json", result)

    output = jetclr_campaign.collect_stage2(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    with Path(summary["summary_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["status"] == "complete_no_collapse_eligible_candidates"
    assert summary["n_collapse_eligible"] == 0
    assert set(summary["pareto_candidate_ids"]) == {0, 1}
    assert "best_candidate_id" not in summary
    assert "No scalar winner" in summary["selection_policy"]
    assert rows[0]["candidate_id"] == "0"
    assert rows[0]["pareto_nondominated"] == "True"
    assert rows[0]["balance_pass"] == "True"
    assert rows[1]["balance_pass"] == "False"
    assert rows[2]["balance_pass"] == "False"


def test_collect_stage3_reports_projector_gates_and_four_objective_pareto(
    tmp_path: Path,
) -> None:
    """Stage-3 collection must retain failed gates and avoid scalar selection."""
    manifest = _write_manifest(tmp_path)
    for spec in manifest["stage3"]["candidates"]:
        trial_root = tmp_path / "stage3" / f"candidate_{spec['candidate_id']:03d}"
        trial_root.mkdir(parents=True)
        metrics = trial_root / "metrics.csv"
        pairing_path = trial_root / "pairing_diagnostics.json"
        anomaly_path = trial_root / "embedding_anomaly.json"
        metrics.write_text("train/loss_mean\n2.0\n", encoding="utf-8")
        pairing = _pairing_metrics(collapse_pass=False)
        pairing["projector_collapse_pass"] = False
        pairing["projector_collapse_failures"] = ["low_effective_rank"]
        if spec["candidate_id"] == 0:
            pairing["embedding_effective_rank"] = 50.0
            pairing["raw_selection_score"] = 0.4
        elif spec["candidate_id"] == 1:
            pairing["embedding_effective_rank"] = 40.0
            pairing["raw_selection_score"] = 0.6
            pairing["value_smd_after_mean"] = 0.39
            pairing["occupancy_smd_after_mean"] = 0.19
        else:
            pairing["embedding_effective_rank"] = 20.0
            pairing["raw_selection_score"] = 0.2
        anomaly = _anomaly_metrics(0.8 if spec["candidate_id"] == 1 else 0.7)
        pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
        anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
        artifacts = {
            "training_csv": {
                "path": str(metrics),
                "sha256": jetclr_campaign._sha256(metrics),
            },
            "pairing_json": {
                "path": str(pairing_path),
                "sha256": jetclr_campaign._sha256(pairing_path),
            },
            "anomaly_json": {
                "path": str(anomaly_path),
                "sha256": jetclr_campaign._sha256(anomaly_path),
            },
        }
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "candidate_id": spec["candidate_id"],
            "spec_sha256": spec["spec_sha256"],
            "source_campaign_id": jetclr_campaign.STAGE3_SOURCE_CAMPAIGN,
            "source_candidate_id": jetclr_campaign.STAGE3_SOURCE_CANDIDATE_ID,
            "source_candidate_spec_sha256": jetclr_campaign.STAGE3_SOURCE_SPEC_SHA256,
            "artifacts": artifacts,
            "training_metrics": {"train/loss_mean": 2.0},
            "pairing_metrics": pairing,
            "anomaly_metrics": anomaly,
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial_root / "result.json", result)

    output = jetclr_campaign.collect_stage3(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    with Path(summary["summary_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["status"] == "complete_no_collapse_eligible_candidates"
    assert summary["n_collapse_eligible"] == 0
    assert set(summary["pareto_candidate_ids"]) == {0, 1}
    assert len(summary["pareto_objectives"]) == 4
    assert "best_candidate_id" not in summary
    assert rows[0]["encoder_collapse_pass"] == "False"
    assert rows[0]["projector_collapse_pass"] == "False"
    assert float(rows[0]["projector_effective_rank"]) == 20.0


def test_collect_stage4_reports_controls_hard_gates_deltas_and_pareto(tmp_path: Path) -> None:
    """Stage-4 collection must separate eligibility, balance, controls, and Pareto utility."""
    manifest = _write_manifest(tmp_path)
    _write_stage4_results(tmp_path, manifest)

    output = jetclr_campaign.collect_stage4(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    with Path(summary["summary_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["status"] == "complete"
    assert summary["n_scientifically_eligible"] == 11
    assert 2 not in summary["scientifically_eligible_candidate_ids"]
    assert 3 in summary["scientifically_eligible_candidate_ids"]
    assert summary["balance_is_scientific_eligibility_gate"] is False
    assert summary["architecture_control_candidate_ids"] == {"9": 6, "10": 0}
    assert len(summary["pareto_objectives"]) == 4
    assert "best_candidate_id" not in summary
    assert "No scalar winner" in summary["selection_policy"]
    assert rows[0]["is_architecture_control"] == "True"
    assert float(rows[1]["delta_vs_control_encoder_effective_rank"]) == 1.0
    assert rows[2]["mnn_nonzero"] == "False"
    assert rows[3]["balance_pass"] == "False"
    assert rows[3]["scientific_eligible"] == "True"
    assert Path(rows[0]["checkpoint_path"]).name == "last.ckpt"


def test_collect_stage4_rejects_tampered_handoff_checkpoint(tmp_path: Path) -> None:
    """The collector must fail closed when a final encoder checkpoint changes."""
    manifest = _write_manifest(tmp_path)
    checkpoints = _write_stage4_results(tmp_path, manifest)
    checkpoints[0].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Stage-4 artifact mismatch"):
        jetclr_campaign.collect_stage4(tmp_path)


def test_stage5_design_and_launcher_freeze_sources_seeds_and_packing(tmp_path: Path) -> None:
    """Stage 5 must cross four exact sources with two seeds on two packed nodes."""
    specs = jetclr_campaign.stage5_specs()
    assert len(specs) == 8
    assert [item["source_candidate_id"] for item in specs] == [0, 0, 4, 4, 6, 6, 10, 10]
    assert [item["seed"] for item in specs] == [321, 777] * 4
    assert {item["source_candidate_spec_sha256"] for item in specs} == {
        item[2] for item in jetclr_campaign.STAGE5_SOURCE_CANDIDATES
    }

    manifest = _write_manifest(tmp_path)
    launchers = jetclr_campaign._write_stage5_launchers(tmp_path, manifest)
    stage5 = launchers["stage5"].read_text(encoding="utf-8")
    submitter = launchers["submitter"].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1%2" in stage5
    assert "base=$((SLURM_ARRAY_TASK_ID * 4))" in stage5
    assert "run-stage5" in stage5
    assert 'dependency="afterok:$stage5_job"' in submitter


def test_collect_stage5_combines_authenticated_seed123_and_reports_promotions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 5 must retain all seed rows and compute matched per-architecture decisions."""
    source_root = tmp_path / "source"
    current_root = tmp_path / "current"
    source_manifest = _write_manifest(source_root)
    _write_stage4_results(source_root, source_manifest)
    current_manifest = _write_manifest(current_root)
    _write_stage5_results(current_root, current_manifest)
    source_summary = source_root / "stage4" / "summary.json"
    source_csv = source_root / "stage4" / "summary.csv"
    source_summary.write_text("{}\n", encoding="utf-8")
    source_csv.write_text("candidate_id\n0\n", encoding="utf-8")
    monkeypatch.setattr(jetclr_campaign, "STAGE5_SOURCE_ROOT", source_root)
    monkeypatch.setattr(jetclr_campaign, "STAGE5_SOURCE_SUMMARY", source_summary)
    monkeypatch.setattr(jetclr_campaign, "STAGE5_SOURCE_SUMMARY_CSV", source_csv)
    monkeypatch.setattr(
        jetclr_campaign, "STAGE5_SOURCE_SUMMARY_SHA256", jetclr_campaign._sha256(source_summary)
    )
    monkeypatch.setattr(
        jetclr_campaign,
        "STAGE5_SOURCE_SUMMARY_CSV_SHA256",
        jetclr_campaign._sha256(source_csv),
    )

    output = jetclr_campaign.collect_stage5(current_root)
    summary = json.loads(output.read_text(encoding="utf-8"))
    with Path(summary["summary_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["n_seed_rows"] == 12
    assert summary["seeds"] == [123, 321, 777]
    assert set(summary["confirmations"]) == {"layers2", "official_projector"}
    assert all(item["promotion"] for item in summary["confirmations"].values())
    assert "global_winner" not in summary
    assert "No scalar global winner" in summary["selection_policy"]
    assert len(rows) == 12
    assert {row["origin"] for row in rows} == {"stage4_seed123", "stage5_fresh"}
    assert all(row["result_payload_sha256"] for row in rows)
    assert all(row["checkpoint_sha256"] for row in rows)


def test_stage6_exact_design_configs_and_launcher(tmp_path: Path) -> None:
    """Stage 6 must compose twelve 16-epoch runs and pack them over three nodes."""
    specs = jetclr_campaign.stage6_specs()
    assert len(specs) == 12
    assert [item["source_candidate_id"] for item in specs] == [0] * 3 + [4] * 3 + [6] * 3 + [
        10
    ] * 3
    assert [item["seed"] for item in specs] == [123, 2027, 31415] * 4
    assert {item["full_epochs"] for item in specs} == {16}
    config_dir = Path(jetclr_campaign.__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        for spec in specs:
            config = compose(
                config_name="train",
                overrides=[
                    "experiment=physics/jetclr_pairing",
                    "trainer=gpu",
                    f"seed={spec['seed']}",
                    "trainer.min_epochs=16",
                    "trainer.max_epochs=16",
                    "callbacks.last_epoch_ckpt.filename='epoch-{epoch:02d}'",
                    "callbacks.last_epoch_ckpt.save_top_k=-1",
                    "callbacks.last_epoch_ckpt.every_n_epochs=1",
                    *spec["overrides"],
                ],
            )
            assert config.trainer.max_epochs == 16
            assert config.callbacks.last_epoch_ckpt.filename == "epoch-{epoch:02d}"

    manifest = _write_manifest(tmp_path)
    launchers = jetclr_campaign._write_stage6_launchers(tmp_path, manifest)
    stage6 = launchers["stage6"].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2%3" in stage6
    assert "#SBATCH --time=12:00:00" in stage6
    assert "run-stage6" in stage6


def test_collect_stage6_authenticates_inventory_source_promotions_and_epoch16_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 6 must retain checkpoint hashes and expose six matched final comparisons."""
    manifest = _write_manifest(tmp_path)
    _write_stage6_results(tmp_path, manifest)
    source_summary = tmp_path / "source_summary.json"
    source_summary_csv = tmp_path / "source_summary.csv"
    source_paired_csv = tmp_path / "source_paired.csv"
    source_summary.write_text(
        json.dumps(
            {
                "confirmations": {
                    "layers2": {"promotion": True},
                    "official_projector": {"promotion": True},
                }
            }
        ),
        encoding="utf-8",
    )
    source_summary_csv.write_text("candidate_id\n0\n", encoding="utf-8")
    source_paired_csv.write_text("seed\n123\n", encoding="utf-8")
    for path_name, digest_name, path in (
        ("STAGE6_SOURCE_SUMMARY", "STAGE6_SOURCE_SUMMARY_SHA256", source_summary),
        ("STAGE6_SOURCE_SUMMARY_CSV", "STAGE6_SOURCE_SUMMARY_CSV_SHA256", source_summary_csv),
        ("STAGE6_SOURCE_PAIRED_CSV", "STAGE6_SOURCE_PAIRED_CSV_SHA256", source_paired_csv),
    ):
        monkeypatch.setattr(jetclr_campaign, path_name, path)
        monkeypatch.setattr(jetclr_campaign, digest_name, jetclr_campaign._sha256(path))

    output = jetclr_campaign.collect_stage6(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["n_candidates"] == 12
    assert summary["epoch16_pair_count"] == 6
    assert summary["milestone_epochs"] == [1, 2, 4, 8, 16]
    assert summary["all_candidates_milestone_evaluation_ready"] is True
    assert summary["source_promotions"] == {"layers2": True, "official_projector": True}
    assert "winner" not in summary


def test_stage7_design_launcher_and_no_training_eval(tmp_path: Path, monkeypatch) -> None:
    """Stage 7 must pack 60 exact checkpoints and invoke validation without fitting."""
    specs = _STAGE7_SPECS
    assert len(specs) == 60
    assert [item["completed_epoch"] for item in specs[:5]] == [1, 2, 4, 8, 16]
    manifest = _write_manifest(tmp_path)
    launchers = jetclr_campaign._write_stage7_launchers(tmp_path, manifest)
    stage7 = launchers["stage7"].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-14%4" in stage7
    assert "#SBATCH --time=01:30:00" in stage7
    assert "run-stage7" in stage7
    monkeypatch.setattr(jetclr_campaign, "_assert_runtime", lambda _: tmp_path)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        output = tmp_path / "stage7" / "candidate_000" / "output"
        pairing = output / "metrics" / "pairing_diagnostics" / "last"
        anomaly = output / "metrics" / "embedding_anomaly" / "last"
        pairing.mkdir(parents=True)
        anomaly.mkdir(parents=True)
        (pairing / "pairing_diagnostics.json").write_text(
            json.dumps(_pairing_metrics()), encoding="utf-8"
        )
        (anomaly / "embedding_anomaly.json").write_text(
            json.dumps(_anomaly_metrics()), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jetclr_campaign.subprocess, "run", fake_run)
    result = jetclr_campaign.run_stage7(tmp_path, 0)
    command = observed["command"]
    assert "train=false" in command
    assert "test=false" in command
    assert "callbacks.clear_ckpts=null" in command
    assert "callbacks.last_epoch_ckpt=null" in command
    assert not any("max_epochs" in value for value in command)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["source_checkpoint_sha256"] == specs[0]["source_checkpoint_sha256"]


def test_collect_stage7_selects_global_one_se_epoch_and_reports_extension(tmp_path: Path) -> None:
    """The milestone collector must make an explicit global horizon calculation."""
    manifest = _write_manifest(tmp_path)
    for spec in manifest["stage7"]["candidates"]:
        trial = tmp_path / "stage7" / f"candidate_{spec['candidate_id']:03d}"
        trial.mkdir(parents=True)
        pairing_path = trial / "pairing.json"
        anomaly_path = trial / "anomaly.json"
        pairing = _pairing_metrics()
        epoch = spec["completed_epoch"]
        value = 0.70 + 0.005 * min(epoch, 8)
        if epoch == 16:
            value += 0.001
        if not spec["is_architecture_control"]:
            value += 0.01
        anomaly = _anomaly_metrics(value)
        pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
        anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
        result = {
            "spec_sha256": spec["spec_sha256"],
            "source_result_payload_sha256": spec["source_result_payload_sha256"],
            "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
            "artifacts": {
                "pairing_json": {
                    "path": str(pairing_path),
                    "sha256": jetclr_campaign._sha256(pairing_path),
                },
                "anomaly_json": {
                    "path": str(anomaly_path),
                    "sha256": jetclr_campaign._sha256(anomaly_path),
                },
            },
            "pairing_metrics": pairing,
            "anomaly_metrics": anomaly,
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial / "result.json", result)

    output = jetclr_campaign.collect_stage7(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    selection = summary["global_epoch_selection"]
    assert summary["n_evaluations"] == 60
    assert selection["selected_epoch"] in [8, 16]
    assert selection["one_se_threshold"] <= selection["best_median"]
    assert "smallest epoch" in selection["rule"]
    assert len(summary["epoch8_to16"]["paired_improvements"]) == 6
    assert "architecture winner" in summary["selection_policy"]
