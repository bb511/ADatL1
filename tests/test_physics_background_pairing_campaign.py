"""Tests for the six shared physics background-pairing studies."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

SCRIPT = Path(__file__).parents[1] / "scripts/physics_background_pairing_campaign.py"
SPEC = importlib.util.spec_from_file_location("physics_background_pairing_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(campaign)


def _write_exact_checkpoint(path: Path, epoch: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loops": {
                "fit_loop": {
                    "epoch_progress": {
                        "total": {
                            "ready": epoch + 1,
                            "started": epoch + 1,
                            "processed": epoch + 1,
                        }
                    }
                }
            },
            "callbacks": {
                campaign.EXACT_RETRAIN_RNG_STATE_KEY: {
                    "schema_version": 1,
                    "python": random.getstate(),
                    "numpy": {
                        "algorithm": "MT19937",
                        "keys": torch.zeros(624, dtype=torch.uint32),
                        "position": 0,
                        "has_gauss": 0,
                        "cached_gaussian": 0.0,
                    },
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": [torch.get_rng_state()],
                    "datamodule_shuffler": torch.Generator().manual_seed(123).get_state(),
                }
            },
        },
        path,
    )


def test_campaign_has_six_studies_and_56_logical_fronts() -> None:
    """AE/VAE add OAS views without duplicating trained trial pools."""
    studies = campaign._studies()
    assert len(studies) == 6
    assert {study["model"] for study in studies} == set(campaign.MODELS)
    assert sum(len(study["objectives"]) for study in studies) == 56
    for study in studies:
        expected = 14 if study["model"] in {"ae", "vae"} else 7
        assert len(study["objectives"]) == expected
        assert len({objective["id"] for objective in study["objectives"]}) == expected
        assert [
            index for objective in study["objectives"] for index in objective["value_indices"]
        ] == list(range(2 * expected))


def test_campaign_rejects_dirty_execution_checkout(monkeypatch) -> None:
    """A prior commit SHA cannot disguise uncommitted scientific changes."""
    monkeypatch.setattr(campaign.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M src/train.py\n"),
    )
    with pytest.raises(RuntimeError, match="clean commit-pinned"):
        campaign._require_clean_checkout()


def test_shared_sweep_command_preserves_paper_budget_and_logging(tmp_path) -> None:
    """One model command trains 50 epochs and retains the configured MLflow logger."""
    command = campaign._base_train_command(tmp_path, "ae")
    joined = " ".join(command)
    assert "-m" in command
    assert "experiment=physics/ae_background_all" in command
    assert "hparams_search=ae_shared_optuna" in command
    assert f"trainer.max_epochs={campaign.SEARCH_EPOCHS}" in command
    assert "test=false" in command
    assert "sqlite:///" in joined
    assert "?timeout=600" in joined
    assert any("storage='sqlite:///" in value for value in command)
    assert not any(value.startswith("logger=") for value in command)
    assert not any("selection_metric" in value for value in command)


def test_sweep_uses_one_packed_controller_without_nested_submitit(tmp_path, monkeypatch) -> None:
    """One Optuna sampler dispatches ordered jobs across the enclosing allocation."""
    design = {"studies": campaign._studies()}
    monkeypatch.setattr(campaign, "_load_design", lambda root: design)
    monkeypatch.setattr(
        campaign,
        "_trial_counts",
        lambda root, study: {"running": 0, "waiting": 0, "usable_complete": 0},
    )
    monkeypatch.setattr(campaign, "_study_trials", lambda root, study: [])
    monkeypatch.setattr(campaign, "_write_chunk_bindings", lambda *args: tmp_path / "bindings")
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command

    monkeypatch.setattr(campaign.subprocess, "run", run)
    campaign.sweep(tmp_path, "ae", trials=4, jobs=4)

    command = captured["command"]
    assert "hydra/launcher=packed_srun_clariden" in command
    assert "hydra.sweeper.n_trials=4" in command
    assert "hydra.sweeper.n_jobs=4" in command
    assert not any("submitit" in value for value in command)


def test_logical_objective_directions_match_the_metrics() -> None:
    """CAP is maximized; W1, drift, and native Q-double-prime are minimized."""
    objectives = campaign._logical_objectives("ae")
    for objective in objectives:
        expected = (
            ["maximize", "minimize"]
            if objective["metric"] == "cap"
            else [
                "minimize",
                "minimize",
            ]
        )
        assert objective["directions"] == expected


def test_shared_hpo_files_match_campaign_objective_order() -> None:
    """Optuna receives exactly the directions expected by offline front slicing."""
    root = Path(__file__).parents[1]
    for model in campaign.MODELS:
        config = yaml.safe_load(
            (root / "configs" / "hparams_search" / f"{model}_shared_optuna.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["hydra"]["sweeper"]["direction"] == [
            direction
            for objective in campaign._logical_objectives(model)
            for direction in objective["directions"]
        ]


def test_mixed_direction_pareto_front_uses_only_requested_pair() -> None:
    """Offline fronts must ignore all other values from the shared objective vector."""
    trials = [
        SimpleNamespace(number=0, values=(0.8, 2.0, 999.0), params={}),
        SimpleNamespace(number=1, values=(0.7, 3.0, -999.0), params={}),
        SimpleNamespace(number=2, values=(0.9, 4.0, 0.0), params={}),
        SimpleNamespace(number=3, values=(0.8, 1.0, 0.0), params={}),
    ]
    front = campaign._pareto_front(trials, [0, 1], ["maximize", "minimize"])
    assert [trial.number for trial in front] == [2, 3]


def test_only_full_finite_objective_vectors_count_toward_600() -> None:
    """Missing-metric fallback values must be replaced, not treated as usable trials."""
    assert campaign._usable_trial(SimpleNamespace(values=(0.5, 1.0)), expected_values=2)
    assert not campaign._usable_trial(
        SimpleNamespace(values=(0.5, float("inf"))), expected_values=2
    )
    assert not campaign._usable_trial(SimpleNamespace(values=(0.5,)), expected_values=2)


def test_checkpoint_identity_matches_evaluator_normalization() -> None:
    """Aggregation must match stable dataset-aware and ordinary checkpoint names."""
    assert campaign._checkpoint_identity("ascore_operational__ds=normal__stable") == "normal"
    assert campaign._checkpoint_identity("cap_native_cdf_ema") == "cap_native_cdf_ema"
    assert campaign._checkpoint_identity("last") == "last"


def test_retraining_uses_score_aware_downstream_suite() -> None:
    """Only AE/VAE need the additional OAS threshold and efficiency callback."""
    assert campaign._retrain_suite("ae") == "physics_background_native_oas"
    assert campaign._retrain_suite("vae") == "physics_background_native_oas"
    assert campaign._retrain_suite("dsae") == "physics_background_native"
    config = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "configs"
            / "retrain_suite"
            / "physics_background_native.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["callbacks"]["clear_ckpts"] is None


def test_retrain_attempts_advance_only_from_authenticated_boundaries(
    monkeypatch, tmp_path
) -> None:
    """Normal allocation continuation advances 40 cumulative epochs at a time."""
    root = tmp_path / "campaign"
    root.mkdir()
    identity = {
        "retrain_index": 0,
        "row": {"retrain_index": 0},
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "experiment_name": "experiment",
        "run_name": "run",
        "hydra_dir": str(root / "retraining" / "hydra" / "0000"),
        "code_commit": "a" * 40,
    }
    monkeypatch.setattr(campaign, "_mlflow_run_for_retrain", lambda *args: "one-run")
    monkeypatch.setattr(campaign, "_live_last_resume", lambda *args: None)

    state, first = campaign._prepare_retrain_attempt(root, identity)

    assert first["target_epoch_boundary"] == 40
    assert first["phase"] == "training"
    assert first["resume"] is None
    assert state["mlflow_run_id"] == "one-run"

    paths = campaign._retrain_paths(root, 0)
    checkpoint = paths["segments"] / "epoch_040.ckpt"
    config = paths["segments"] / "epoch_040_config.yaml"
    checkpoint.parent.mkdir(parents=True)
    _write_exact_checkpoint(checkpoint, epoch=39)
    config.write_text("trainer: {}\n", encoding="utf-8")
    state["segments"].append(
        {
            "epoch_boundary": 40,
            "continuation_checkpoint": str(checkpoint),
            "continuation_checkpoint_sha256": campaign._sha256(checkpoint),
            "resolved_config": str(config),
            "resolved_config_sha256": campaign._sha256(config),
        }
    )
    campaign._write_json(paths["state"], state)

    resumed_state, second = campaign._prepare_retrain_attempt(root, identity)

    assert resumed_state["mlflow_run_id"] == "one-run"
    assert second["target_epoch_boundary"] == 80
    assert second["phase"] == "training"
    assert second["resume"]["kind"] == "completed_segment"
    assert second["resume"]["sha256"] == campaign._sha256(checkpoint)


def test_retrain_boundary_can_resume_from_a_later_authenticated_batch(
    monkeypatch, tmp_path
) -> None:
    """Submission-map identity is per attempt, not part of the stable training identity."""
    root = tmp_path / "campaign"
    root.mkdir()
    identity = {
        "retrain_index": 7,
        "row": {"retrain_index": 7},
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "experiment_name": "experiment",
        "run_name": "run",
        "hydra_dir": str(root / "retraining" / "hydra" / "0007"),
        "code_commit": "a" * 40,
    }
    map_a = {"mode": "mapped", "local_task_id": 0, "retrain_index": 7, "task_map": "a"}
    map_b = {"mode": "mapped", "local_task_id": 1, "retrain_index": 7, "task_map": "b"}
    monkeypatch.setattr(campaign, "_mlflow_run_for_retrain", lambda *args: "one-run")
    monkeypatch.setattr(campaign, "_live_last_resume", lambda *args: None)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    state, _ = campaign._prepare_retrain_attempt(root, identity, map_a)
    paths = campaign._retrain_paths(root, 7)
    checkpoint = paths["segments"] / "epoch_040.ckpt"
    config = paths["segments"] / "epoch_040_config.yaml"
    checkpoint.parent.mkdir(parents=True)
    _write_exact_checkpoint(checkpoint, epoch=39)
    config.write_text("trainer: {}\n", encoding="utf-8")
    state["segments"].append(
        {
            "epoch_boundary": 40,
            "continuation_checkpoint": str(checkpoint),
            "continuation_checkpoint_sha256": campaign._sha256(checkpoint),
            "resolved_config": str(config),
            "resolved_config_sha256": campaign._sha256(config),
        }
    )
    campaign._write_json(paths["state"], state)

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    resumed, attempt = campaign._prepare_retrain_attempt(root, identity, map_b)

    assert resumed["identity"] == identity
    assert attempt["scheduler_identity"] == map_b
    assert attempt["resume"]["kind"] == "completed_segment"


def test_epoch_200_progress_schedules_fresh_evaluation_without_fit_resume(
    monkeypatch, tmp_path
) -> None:
    """Evaluation is a fresh task and never consumes fit/HPC resume state."""
    root = tmp_path / "campaign"
    root.mkdir()
    identity = {
        "retrain_index": 0,
        "row": {"retrain_index": 0},
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "experiment_name": "experiment",
        "run_name": "run",
        "hydra_dir": str(root / "retraining" / "hydra" / "0000"),
        "code_commit": "a" * 40,
    }
    monkeypatch.setattr(campaign, "_mlflow_run_for_retrain", lambda *args: "one-run")
    paths = campaign._retrain_paths(root, 0)
    segments = []
    for boundary in (40, 80, 120, 160, 200):
        checkpoint = paths["segments"] / f"epoch_{boundary:03d}.ckpt"
        config = paths["segments"] / f"epoch_{boundary:03d}_config.yaml"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _write_exact_checkpoint(checkpoint, epoch=boundary - 1)
        config.write_text("trainer: {}\n", encoding="utf-8")
        segments.append(
            {
                "epoch_boundary": boundary,
                "continuation_checkpoint": str(checkpoint),
                "continuation_checkpoint_sha256": campaign._sha256(checkpoint),
                "resolved_config": str(config),
                "resolved_config_sha256": campaign._sha256(config),
            }
        )
    paths["trainer"].mkdir(parents=True)
    (paths["trainer"] / "hpc_ckpt_1.ckpt").write_text("evaluation-hpc\n", encoding="utf-8")
    campaign._write_json(
        paths["state"],
        {
            "schema_version": 1,
            "identity": identity,
            "mlflow_run_id": "one-run",
            "attempts": [],
            "segments": segments,
        },
    )

    state, attempt = campaign._prepare_retrain_attempt(root, identity)

    assert attempt["phase"] == "evaluation"
    assert attempt["target_epoch_boundary"] == 200
    assert attempt["resume"] is None
    assert attempt["hpc_evidence_archived_before_fit"]
    assert not list(paths["trainer"].glob("hpc_ckpt_*.ckpt"))
    state["completed_at"] = "2026-08-05T00:00:00+00:00"
    campaign._write_json(paths["state"], state)

    _, finalization = campaign._prepare_retrain_attempt(root, identity)

    assert finalization["phase"] == "finalization"
    assert finalization["resume"] is None


def test_committed_training_segment_is_strictly_authenticated(tmp_path) -> None:
    """A terminal training task is valid progress only with exact state, config, and hashes."""
    root = tmp_path / "campaign"
    root.mkdir()
    design = {"schema_version": 2, "code_commit": "a" * 40}
    (root / "design.json").write_text(campaign.json.dumps(design), encoding="utf-8")
    row = {
        "retrain_index": 0,
        "model": "ae",
        "trial_number": 7,
        "seed": 123,
        "epochs": 200,
        "logical_objectives": ["native_wasserstein"],
        "params": {"algorithm.optimizer.lr": 0.001},
    }
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        campaign.json.dumps({"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": [row]}),
        encoding="utf-8",
    )
    identity = campaign._retrain_identity(root, design, manifest_path, 0, row)
    paths = campaign._retrain_paths(root, 0)
    checkpoint = paths["segments"] / "epoch_040.ckpt"
    config_path = paths["segments"] / "epoch_040_config.yaml"
    checkpoint.parent.mkdir(parents=True)
    _write_exact_checkpoint(checkpoint, epoch=39)
    config_path.write_text(
        """trainer:
  min_epochs: 40
  max_epochs: 40
train: true
test: false
skip_evaluation: true
seed: 123
algorithm:
  optimizer:
    lr: 0.001
""",
        encoding="utf-8",
    )
    completed_at = "2026-08-05T00:00:00+00:00"
    campaign._write_json(
        paths["state"],
        {
            "schema_version": 1,
            "identity": identity,
            "mlflow_run_id": "run-id",
            "attempts": [
                {
                    "attempt_number": 1,
                    "phase": "training",
                    "target_epoch_boundary": 40,
                    "completed_at": completed_at,
                    "scheduler_identity": None,
                    "slurm_array_job_id": "777",
                    "slurm_array_task_id": "4",
                }
            ],
            "segments": [
                {
                    "epoch_boundary": 40,
                    "attempt_number": 1,
                    "continuation_checkpoint": str(checkpoint),
                    "continuation_checkpoint_sha256": campaign._sha256(checkpoint),
                    "resolved_config": str(config_path),
                    "resolved_config_sha256": campaign._sha256(config_path),
                }
            ],
        },
    )

    assert (
        campaign._validate_retrain_progress(root, 0, row, scheduler_job_id="777_4")
        == paths["state"].resolve()
    )
    with pytest.raises(RuntimeError, match="not 778_4"):
        campaign._validate_retrain_progress(root, 0, row, scheduler_job_id="778_4")
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="artifact changed"):
        campaign._validate_retrain_progress(root, 0, row)


def test_retrain_retry_before_any_epoch_allows_exact_scratch_retry(monkeypatch, tmp_path) -> None:
    """An allocation that produced neither epoch history nor checkpoint may retry from zero."""
    root = tmp_path / "campaign"
    root.mkdir()
    identity = {
        "retrain_index": 0,
        "row": {"retrain_index": 0},
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "experiment_name": "experiment",
        "run_name": "run",
        "hydra_dir": str(root / "retraining" / "hydra" / "0000"),
        "code_commit": "a" * 40,
    }
    monkeypatch.setattr(campaign, "_mlflow_run_for_retrain", lambda *args: "one-run")
    monkeypatch.setattr(campaign, "_live_last_resume", lambda *args: None)
    campaign._prepare_retrain_attempt(root, identity)
    paths = campaign._retrain_paths(root, 0)
    paths["trainer"].mkdir(parents=True, exist_ok=True)
    hpc = paths["trainer"] / "hpc_ckpt_1.ckpt"
    hpc.write_text("mid-epoch scheduler evidence\n", encoding="utf-8")

    _, retry = campaign._prepare_retrain_attempt(root, identity)

    assert retry["phase"] == "training"
    assert retry["resume"] is None
    assert retry["target_epoch_boundary"] == 40
    assert len(retry["hpc_evidence_archived_before_fit"]) == 1
    evidence = retry["hpc_evidence_archived_before_fit"][0]
    assert Path(evidence["path"]).is_file()
    assert evidence["sha256"] == campaign._sha256(Path(evidence["path"]))
    assert not hpc.exists()


def test_live_completed_epoch_newer_than_boundary_has_priority_over_segment(
    monkeypatch, tmp_path
) -> None:
    """Exact live end-of-epoch state wins only when newer than committed progress."""
    root = tmp_path / "campaign"
    root.mkdir()
    identity = {
        "retrain_index": 0,
        "row": {"retrain_index": 0},
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "experiment_name": "experiment",
        "run_name": "run",
        "hydra_dir": str(root / "retraining" / "hydra" / "0000"),
        "code_commit": "a" * 40,
    }
    monkeypatch.setattr(campaign, "_mlflow_run_for_retrain", lambda *args: "one-run")
    paths = campaign._retrain_paths(root, 0)
    checkpoint = paths["segments"] / "epoch_040.ckpt"
    config = paths["segments"] / "epoch_040_config.yaml"
    checkpoint.parent.mkdir(parents=True)
    _write_exact_checkpoint(checkpoint, epoch=39)
    config.write_text("trainer: {}\n", encoding="utf-8")
    live = root / "runtime" / "checkpoints" / "experiment" / "run" / "last.ckpt"
    live.parent.mkdir(parents=True)
    live.write_text("epoch 57 checkpoint\n", encoding="utf-8")
    paths["trainer"].mkdir(parents=True)
    (paths["trainer"] / "hpc_ckpt_3.ckpt").write_text("mid-epoch checkpoint\n", encoding="utf-8")
    campaign._write_json(
        paths["state"],
        {
            "schema_version": 1,
            "identity": identity,
            "mlflow_run_id": "one-run",
            "attempts": [],
            "segments": [
                {
                    "epoch_boundary": 40,
                    "continuation_checkpoint": str(checkpoint),
                    "continuation_checkpoint_sha256": campaign._sha256(checkpoint),
                    "resolved_config": str(config),
                    "resolved_config_sha256": campaign._sha256(config),
                }
            ],
        },
    )
    monkeypatch.setattr(
        campaign,
        "_live_last_resume",
        lambda *args: {
            "kind": "live_last",
            "path": str(live),
            "sha256": campaign._sha256(live),
            "completed_epochs": 57,
        },
    )

    _, attempt = campaign._prepare_retrain_attempt(root, identity)

    assert attempt["resume"]["kind"] == "live_last"
    assert attempt["resume"]["completed_epochs"] == 57
    assert attempt["target_epoch_boundary"] == 80
    assert attempt["hpc_evidence_archived_before_fit"]
    assert not list(paths["trainer"].glob("hpc_ckpt_*.ckpt"))


def test_live_checkpoint_at_committed_boundary_uses_immutable_segment(
    monkeypatch, tmp_path
) -> None:
    """An equal live checkpoint cannot displace the authenticated boundary copy."""
    root = tmp_path / "campaign"
    root.mkdir()
    identity = {
        "retrain_index": 0,
        "row": {"retrain_index": 0},
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "experiment_name": "experiment",
        "run_name": "run",
        "hydra_dir": str(root / "retraining" / "hydra" / "0000"),
        "code_commit": "a" * 40,
    }
    monkeypatch.setattr(campaign, "_mlflow_run_for_retrain", lambda *args: "one-run")
    paths = campaign._retrain_paths(root, 0)
    checkpoint = paths["segments"] / "epoch_040.ckpt"
    config = paths["segments"] / "epoch_040_config.yaml"
    checkpoint.parent.mkdir(parents=True)
    _write_exact_checkpoint(checkpoint, epoch=39)
    config.write_text("trainer: {}\n", encoding="utf-8")
    campaign._write_json(
        paths["state"],
        {
            "schema_version": 1,
            "identity": identity,
            "mlflow_run_id": "one-run",
            "attempts": [],
            "segments": [
                {
                    "epoch_boundary": 40,
                    "continuation_checkpoint": str(checkpoint),
                    "continuation_checkpoint_sha256": campaign._sha256(checkpoint),
                    "resolved_config": str(config),
                    "resolved_config_sha256": campaign._sha256(config),
                }
            ],
        },
    )
    monkeypatch.setattr(
        campaign,
        "_live_last_resume",
        lambda *args: {
            "kind": "live_last",
            "path": "unused-live.ckpt",
            "sha256": "b" * 64,
            "completed_epochs": 40,
        },
    )

    _, attempt = campaign._prepare_retrain_attempt(root, identity)

    assert attempt["resume"]["kind"] == "completed_segment"
    assert attempt["resume"]["path"] == str(checkpoint.resolve())


def test_live_last_resume_requires_checkpoint_epoch_and_contiguous_mlflow(
    monkeypatch, tmp_path
) -> None:
    """Latest-epoch recovery is accepted only when checkpoint and MLflow agree exactly."""
    history = [SimpleNamespace(step=(index + 1) * 765, value=float(index)) for index in range(3)]

    class Client:
        def __init__(self, **kwargs):
            pass

        def get_metric_history(self, run_id, key):
            assert (run_id, key) == ("run-id", "epoch_idx")
            return history

    monkeypatch.setattr("mlflow.tracking.MlflowClient", Client)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    _write_exact_checkpoint(checkpoint_root / "last.ckpt", epoch=2)
    identity = {"mlflow_tracking_uri": "file:/unused", "mlflow_run_id": "run-id"}

    resume = campaign._live_last_resume(identity, checkpoint_root)

    assert resume["completed_epochs"] == 3
    assert resume["kind"] == "live_last"
    history[:] = [SimpleNamespace(step=765, value=float(index)) for index in range(3)]
    with pytest.raises(RuntimeError, match="differs from contiguous"):
        campaign._live_last_resume(identity, checkpoint_root)
    history[:] = [
        SimpleNamespace(step=(index + 1) * 765, value=float(index)) for index in range(3)
    ]
    _write_exact_checkpoint(checkpoint_root / "last.ckpt", epoch=3)
    with pytest.raises(RuntimeError, match="differs from contiguous"):
        campaign._live_last_resume(identity, checkpoint_root)

    _write_exact_checkpoint(checkpoint_root / "last.ckpt", epoch=2)
    payload = torch.load(checkpoint_root / "last.ckpt", weights_only=False)
    payload["loops"]["fit_loop"]["epoch_progress"]["total"]["processed"] = 2
    torch.save(payload, checkpoint_root / "last.ckpt")
    with pytest.raises(RuntimeError, match="completed-epoch boundary"):
        campaign._live_last_resume(identity, checkpoint_root)


def test_exact_partial_orphan_is_completed_and_adopted(monkeypatch, tmp_path) -> None:
    """A crash between two atomic copies is recoverable only from exact live sources."""
    root = tmp_path / "campaign"
    root.mkdir()
    identity = {
        "retrain_index": 0,
        "experiment_name": "experiment",
        "run_name": "run",
        "hydra_dir": str(root / "retraining" / "hydra" / "0000"),
    }
    paths = campaign._retrain_paths(root, 0)
    live = root / "runtime" / "checkpoints" / "experiment" / "run" / "last.ckpt"
    live.parent.mkdir(parents=True)
    live.write_text("checkpoint\n", encoding="utf-8")
    orphan = paths["segments"] / "epoch_040.ckpt"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("checkpoint\n", encoding="utf-8")
    resolved = Path(identity["hydra_dir"]) / ".hydra" / "config.yaml"
    resolved.parent.mkdir(parents=True)
    resolved.write_text("trainer: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        campaign,
        "_live_last_resume",
        lambda *args: {
            "kind": "live_last",
            "path": str(live),
            "sha256": campaign._sha256(live),
            "completed_epochs": 40,
        },
    )
    state = {
        "mlflow_run_id": "run-id",
        "attempts": [
            {
                "attempt_number": 1,
                "phase": "training",
                "target_epoch_boundary": 40,
            }
        ],
        "segments": [],
    }

    campaign._adopt_exact_orphan_segment(root, identity, state, paths)

    assert state["segments"][0]["epoch_boundary"] == 40
    assert Path(state["segments"][0]["resolved_config"]).read_text() == "trainer: {}\n"
    assert state["attempts"][0]["orphan_adopted_at"]


def test_retrain_slurm_signal_targets_lightning_job_step() -> None:
    """The emergency signal must reach Python; Slurm's B prefix would exclude it."""
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "clariden"
        / "physics_background_pairing_retrain.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --requeue" in script
    assert "#SBATCH --signal=USR1@900" in script
    assert "#SBATCH --open-mode=append" in script
    assert "--signal=B:" not in script
    assert '\nsrun "${UV_PROJECT_ENVIRONMENT}/bin/python"' in script
    assert "resolve-retrain-index" in script
    assert '--index "${PHYSICS_BGPAIR_RETRAIN_INDEX}"' in script
    assert "Normal 40-epoch" in script
    campaign_source = SCRIPT.read_text(encoding="utf-8")
    assert "_requeue_current_task" not in campaign_source
    assert "ckpt_path=hpc" not in campaign_source
    assert '"kind": "lightning_hpc"' not in campaign_source


def _mapped_retrain_fixture(monkeypatch, tmp_path: Path):
    """Create one authenticated sparse local-to-frozen retrain mapping."""
    root = tmp_path / "campaign"
    root.mkdir()
    design = {"schema_version": 2, "code_commit": "a" * 40}
    (root / "design.json").write_text(campaign.json.dumps(design), encoding="utf-8")
    monkeypatch.setattr(campaign, "_load_design", lambda campaign_root: design)
    rows = [{"retrain_index": index, "model": "ae", "trial_number": index} for index in range(4)]
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        campaign.json.dumps({"schema_version": 2, "code_commit": "a" * 40, "retrain_runs": rows}),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "code_commit": "a" * 40,
        "campaign_root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": campaign._sha256(manifest_path),
        "batch_size": 2,
        "entries": [
            {
                "local_task_id": local,
                "retrain_index": actual,
                "model": "ae",
                "row_sha256": campaign._json_sha256(rows[actual]),
            }
            for local, actual in enumerate((1, 3))
        ],
    }
    serialized = campaign.json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = campaign.hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    path = root / "state" / "retrain_task_maps" / f"{digest}.json"
    path.parent.mkdir(parents=True)
    path.write_text(serialized, encoding="utf-8")
    return root, path, digest


def test_retrain_task_map_resolves_sparse_actual_index(monkeypatch, tmp_path) -> None:
    """A bounded local array ID resolves to the authenticated frozen index."""
    root, path, digest = _mapped_retrain_fixture(monkeypatch, tmp_path)
    assert campaign.resolve_retrain_task_index(root, path, digest, 0) == 1
    assert campaign.resolve_retrain_task_index(root, path, digest, 1) == 3


def test_retrain_scheduler_identity_rejects_map_or_export_tampering(monkeypatch, tmp_path) -> None:
    """The campaign verifies path, digest, local task, and resolved actual index together."""
    root, path, digest = _mapped_retrain_fixture(monkeypatch, tmp_path)
    environment = {
        "SLURM_ARRAY_TASK_ID": "1",
        "PHYSICS_BGPAIR_RETRAIN_TASK_MAP": str(path),
        "PHYSICS_BGPAIR_RETRAIN_TASK_MAP_SHA256": digest,
        "PHYSICS_BGPAIR_RETRAIN_LOCAL_TASK_ID": "1",
        "PHYSICS_BGPAIR_RETRAIN_INDEX": "3",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    identity = campaign._retrain_scheduler_identity(root, 3, path, digest)
    assert identity["local_task_id"] == 1
    assert identity["retrain_index"] == 3

    monkeypatch.setenv("PHYSICS_BGPAIR_RETRAIN_INDEX", "2")
    with pytest.raises(RuntimeError, match="differ from their exported"):
        campaign._retrain_scheduler_identity(root, 3, path, digest)

    monkeypatch.setenv("PHYSICS_BGPAIR_RETRAIN_INDEX", "3")
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        campaign._retrain_scheduler_identity(root, 3, path, digest)


def test_retrain_hyperparameters_are_logged_only_on_first_segment() -> None:
    """The persistent MLflow run gets params once; changing cumulative bounds do not conflict."""
    assert campaign._retrain_hyperparameter_logging_override(1) == "log_hyperparameters=true"
    assert campaign._retrain_hyperparameter_logging_override(2) == "log_hyperparameters=false"
    assert campaign._retrain_hyperparameter_logging_override(5) == "log_hyperparameters=false"


def test_retrain_mlflow_run_preserves_required_immutable_tags(tmp_path) -> None:
    """Pre-created persistent runs carry tags Lightning skips when run_id is supplied."""
    from mlflow.tracking import MlflowClient

    tracking = tmp_path / "mlruns"
    identity = {
        "mlflow_tracking_uri": f"file:{tracking}",
        "experiment_name": "physics_bgpair_retrain_ae",
        "run_name": "ae__trial0007",
        "code_commit": "a" * 40,
        "retrain_index": 3,
    }
    run_id = campaign._mlflow_run_for_retrain(identity, None)
    assert campaign._mlflow_run_for_retrain(identity, run_id) == run_id
    run = MlflowClient(tracking_uri=f"file:{tracking}").get_run(run_id)
    assert run.data.tags["mlflow.runName"] == identity["run_name"]
    assert run.data.tags["mlflow.source.git.commit"] == identity["code_commit"]
    assert run.data.tags["physics_bgpair_code_commit"] == identity["code_commit"]
    assert run.data.tags["physics_bgpair_retrain_index"] == "3"


def test_retrain_mlflow_run_rejects_changed_identity_tag(tmp_path) -> None:
    """A reused MLflow run cannot silently lose report-critical source metadata."""
    from mlflow.tracking import MlflowClient

    tracking = tmp_path / "mlruns"
    identity = {
        "mlflow_tracking_uri": f"file:{tracking}",
        "experiment_name": "physics_bgpair_retrain_vae",
        "run_name": "vae__trial0002",
        "code_commit": "b" * 40,
        "retrain_index": 5,
    }
    run_id = campaign._mlflow_run_for_retrain(identity, None)
    MlflowClient(tracking_uri=f"file:{tracking}").set_tag(
        run_id, "mlflow.source.git.commit", "changed"
    )
    with pytest.raises(RuntimeError, match="tags differ"):
        campaign._mlflow_run_for_retrain(identity, run_id)


def test_oracle_ties_use_lowest_frozen_membership_index() -> None:
    """Equal downstream means have a documented deterministic winner."""
    candidates = [
        {"membership_index": 9, "mean_downstream_efficiency": 0.5},
        {"membership_index": 3, "mean_downstream_efficiency": 0.5},
        {"membership_index": 7, "mean_downstream_efficiency": 0.4},
    ]

    assert campaign._oracle_winner(candidates)["membership_index"] == 3


def test_existing_retrain_marker_is_hash_validated(monkeypatch, tmp_path) -> None:
    """Resume cannot silently accept stale result or restart-evidence artifacts."""
    root = tmp_path / "campaign"
    marker = root / "retraining" / "markers" / "0000.json"
    marker.parent.mkdir(parents=True)
    artifact = tmp_path / "optimized_metric.json"
    artifact.write_text("{}\n", encoding="utf-8")
    values = tmp_path / "values.csv"
    values.write_text("value\n1\n", encoding="utf-8")
    row = {
        "model": "ae",
        "trial_number": 1,
        "retrain_index": 0,
        "epochs": 200,
        "seed": 123,
        "params": {},
    }
    state_dir = root / "retraining" / "restart_state" / "0000"
    manifest = root / "selection" / "pareto_retrain_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    segments = []
    for boundary in (40, 80, 120, 160, 200):
        checkpoint = state_dir / "segments" / f"epoch_{boundary:03d}.ckpt"
        config = state_dir / "segments" / f"epoch_{boundary:03d}_config.yaml"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _write_exact_checkpoint(checkpoint, epoch=boundary - 1)
        config.write_text(f"boundary: {boundary}\n", encoding="utf-8")
        segments.append(
            {
                "epoch_boundary": boundary,
                "continuation_checkpoint": str(checkpoint),
                "continuation_checkpoint_sha256": campaign._sha256(checkpoint),
                "resolved_config": str(config),
                "resolved_config_sha256": campaign._sha256(config),
            }
        )
    completed_at = "2026-08-05T00:00:00+00:00"
    state = state_dir / "state.json"
    state.write_text(
        campaign.json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "row": row,
                    "manifest": str(manifest),
                    "manifest_sha256": campaign._sha256(manifest),
                },
                "mlflow_run_id": "run-id",
                "completed_at": completed_at,
                "segments": segments,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(campaign, "_audit_retrain_outputs", lambda *args: {})
    marker.write_text(
        campaign.json.dumps(
            {
                "schema_version": 2,
                "row": row,
                "completed_at": completed_at,
                "mlflow_run_id": "run-id",
                "restart_state_artifact": str(state),
                "restart_state_artifact_sha256": campaign._sha256(state),
                "optimized_metric_artifact": str(artifact),
                "optimized_metric_artifact_sha256": campaign._sha256(artifact),
                "value_files": [{"path": str(values), "sha256": campaign._sha256(values)}],
            }
        ),
        encoding="utf-8",
    )

    campaign._validate_retrain_marker(marker, row)
    values.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="value artifact changed"):
        campaign._validate_retrain_marker(marker, row)


def test_pilot_parser_exercises_the_complete_model_suite(monkeypatch) -> None:
    """A pilot validates all pairings and scores together, not one isolated cell."""
    monkeypatch.setattr(
        "sys.argv",
        ["physics_background_pairing_campaign.py", "pilot", "--model", "vae"],
    )
    args = campaign.parse_args()
    assert args.model == "vae"


def test_pilot_reduces_only_inner_cap_optimization(monkeypatch, tmp_path) -> None:
    """The unified pilot checks every path without changing data or pairing tables."""
    commands = []
    monkeypatch.setattr(campaign, "_load_design", lambda root: {"code_commit": "a" * 40})
    monkeypatch.setattr(
        campaign.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(
        campaign,
        "_audit_selected_threshold_checkpoints",
        lambda artifact, model: [{"checkpoint_family_id": "last"}],
    )
    artifact = tmp_path / "pilots" / "dsae" / "optimized_metric.json"
    artifact.parent.mkdir(parents=True)
    objectives = [objective["id"] for objective in campaign._logical_objectives("dsae")]
    artifact.write_text(
        campaign.json.dumps(
            {
                "schema_version": 2,
                "objective_order": objectives,
                "selections": {
                    name: {
                        "optimized_ckpt_name": "last",
                        "optimized_metric": [1.0, 1.0],
                    }
                    for name in objectives
                },
            }
        ),
        encoding="utf-8",
    )

    campaign.pilot(tmp_path, "dsae")

    assert "physics_selection.cap_metric_config.n_epochs=1" in commands[0]
    assert not any(value.startswith("data.max_val_batches=") for value in commands[0])


def test_packed_pilot_creates_storage_parent_before_optuna(monkeypatch, tmp_path) -> None:
    """SQLite must be able to create the temporary packed-pilot database."""
    from optuna.trial import TrialState

    design = {"code_commit": "a" * 40, "studies": campaign._studies()}
    trials = [
        SimpleNamespace(
            number=index,
            state=TrialState.COMPLETE,
            values=[1.0] * 28,
            params={"x": index},
        )
        for index in range(4)
    ]
    fake_optuna = SimpleNamespace(
        load_study=lambda **kwargs: SimpleNamespace(get_trials=lambda deepcopy=False: trials),
        trial=SimpleNamespace(TrialState=TrialState),
    )
    monkeypatch.setattr(campaign, "_load_design", lambda root: design)
    monkeypatch.setattr(campaign, "_optuna", lambda: fake_optuna)
    monkeypatch.setattr(
        campaign,
        "_audit_selected_threshold_checkpoints",
        lambda artifact, model: [{"checkpoint_family_id": "last"}],
    )

    def run(command, **kwargs):
        output = tmp_path / "packed_pilots" / "ae"
        assert output.is_dir()
        for index in range(4):
            run_dir = output / str(index)
            run_dir.mkdir()
            (run_dir / "optimized_metric.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(campaign.subprocess, "run", run)
    campaign.packed_pilot(tmp_path, "ae")
    assert (tmp_path / "packed_pilots" / "ae" / "success.json").is_file()


def test_pilot_audit_requires_finite_score_specific_checkpoint_thresholds(
    tmp_path,
) -> None:
    """The GPU gate authenticates thresholds restored from each selected checkpoint."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    torch.save(
        {
            "state_dict": {
                "thres_operational": torch.tensor(1.25),
                "thres_operational__residual_oas": torch.tensor(3.5),
            }
        },
        checkpoint_root / "last.ckpt",
    )
    # Multiple ModelCheckpoint callbacks legitimately create this unused sibling.
    torch.save({"state_dict": {}}, checkpoint_root / "last-v1.ckpt")
    artifact = tmp_path / "optimized_metric.json"
    artifact.write_text(
        campaign.json.dumps(
            {
                "run_provenance": {"checkpoint_run_dir": str(checkpoint_root)},
                "selections": {
                    "native__cap__cdf": {"optimized_ckpt_name": "last"},
                    "residual_oas__cap__cdf": {"optimized_ckpt_name": "last"},
                },
            }
        ),
        encoding="utf-8",
    )

    rows = campaign._audit_selected_threshold_checkpoints(artifact, "ae")

    assert rows[0]["thresholds"] == {
        "thres_operational": pytest.approx(1.25),
        "thres_operational__residual_oas": pytest.approx(3.5),
    }
    payload = torch.load(checkpoint_root / "last.ckpt", weights_only=True)
    payload["state_dict"].pop("thres_operational__residual_oas")
    torch.save(payload, checkpoint_root / "last.ckpt")
    with pytest.raises(RuntimeError, match="Threshold buffers differ"):
        campaign._audit_selected_threshold_checkpoints(artifact, "ae")


def test_chunk_bindings_preserve_global_trial_identity(monkeypatch, tmp_path) -> None:
    """Chunk-local Hydra indices map explicitly to persistent trial numbers."""
    study = campaign._studies()[0]
    sweep_dir = tmp_path / "sweeps" / "ae" / "stamp"
    for job in range(2):
        run_dir = sweep_dir / str(job)
        run_dir.mkdir(parents=True)
        (run_dir / "optimized_metric.json").write_text("{}\n", encoding="utf-8")
    trials = [
        SimpleNamespace(
            number=8,
            state=SimpleNamespace(name="COMPLETE"),
            params={"x": 1},
            values=(1.0,),
        ),
        SimpleNamespace(
            number=9,
            state=SimpleNamespace(name="FAIL"),
            params={"x": 2},
            values=None,
        ),
    ]
    monkeypatch.setattr(campaign, "_study_trials", lambda root, selected: trials)

    output = campaign._write_chunk_bindings(tmp_path, study, sweep_dir, {0, 1})
    bindings = campaign.json.loads(output.read_text(encoding="utf-8"))["bindings"]

    assert [(row["trial_number"], row["hydra_job_num"]) for row in bindings] == [
        (8, 0),
        (9, 1),
    ]
    assert all(row["optimized_metric_artifact_sha256"] for row in bindings)


def test_refreshing_chunk_bindings_reconciles_persistent_trial_state(
    monkeypatch, tmp_path
) -> None:
    """A recovery refresh overwrites stale state and objective values deterministically."""
    study = campaign._studies()[0]
    sweep_dir = tmp_path / "sweeps" / "ae" / "stamp"
    run_dir = sweep_dir / "0"
    run_dir.mkdir(parents=True)
    (run_dir / "optimized_metric.json").write_text("{}\n", encoding="utf-8")
    trial = SimpleNamespace(
        number=8,
        state=SimpleNamespace(name="RUNNING"),
        params={"x": 1},
        values=None,
    )
    monkeypatch.setattr(campaign, "_study_trials", lambda root, selected: [trial])
    output = campaign._write_chunk_bindings(tmp_path, study, sweep_dir, {0, 1})

    trial.state = SimpleNamespace(name="COMPLETE")
    trial.values = (1.0, 2.0)
    campaign._write_chunk_bindings(tmp_path, study, sweep_dir, {0, 1})
    binding = campaign.json.loads(output.read_text(encoding="utf-8"))["bindings"][0]

    assert binding["state"] == "COMPLETE"
    assert binding["values"] == [1.0, 2.0]
