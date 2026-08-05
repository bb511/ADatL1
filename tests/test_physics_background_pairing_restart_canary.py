from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import Callback, Trainer

from scripts import physics_background_pairing_restart_canary as canary


def test_canary_root_cannot_overlap_campaign_data(tmp_path: Path) -> None:
    valid = tmp_path / "canaries" / "retrain_restart_canary_abc123"
    assert canary._validate_root(valid) == valid.resolve()

    with pytest.raises(ValueError, match="campaign"):
        canary._validate_root(tmp_path / "campaigns" / "retrain_restart_canary_abc123")
    with pytest.raises(ValueError, match="basename"):
        canary._validate_root(tmp_path / "canaries" / "production")


def test_scheduler_log_requires_append_preserved_restart_evidence() -> None:
    text = (
        "CANARY_ATTEMPT_START restart=0\n"
        "CANARY_LIGHTNING_CHILD restart=0\n"
        "Handling auto-requeue signal: 10\n"
        "CANARY_ATTEMPT_START restart=1\n"
        "CANARY_LIGHTNING_CHILD restart=1"
    )
    evidence = canary._scheduler_log_evidence(text)
    assert evidence["CANARY_ATTEMPT_START restart=0"] == 1
    assert evidence["CANARY_ATTEMPT_START restart=1"] == 1

    with pytest.raises(RuntimeError, match="missing restart evidence"):
        canary._scheduler_log_evidence(text.replace("CANARY_ATTEMPT_START restart=0", ""))


def test_restart_canary_sbatch_exercises_production_signal_shape() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "clariden"
        / "physics_background_pairing_restart_canary.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=debug" in script
    assert "#SBATCH --requeue" in script
    assert "#SBATCH --signal=USR1@120" in script
    assert "#SBATCH --open-mode=append" in script
    assert "#SBATCH --signal=B:" not in script
    assert '\nsrun "${UV_PROJECT_ENVIRONMENT}/bin/python"' in script
    assert "physics_background_pairing_restart_canary.py" in script
    assert "physics_background_pairing_campaign.py" not in script
    source = Path(canary.__file__).read_text(encoding="utf-8")
    assert 'ckpt_path="hpc"' not in source
    assert "_discover_hpc" not in source
    assert source.count("num_sanity_val_steps=0") == 2


def test_mlflow_tags_include_explicit_source_commit(tmp_path: Path) -> None:
    root = canary._validate_root(tmp_path / "retrain_restart_canary_tags")
    commit = "a" * 40
    tags = canary._expected_tags(root, commit)
    assert tags["mlflow.runName"] == canary.RUN_NAME
    assert tags["mlflow.source.git.commit"] == commit
    assert tags["physics_bgpair_restart_canary"] == "true"
    assert len(tags["physics_bgpair_restart_canary_identity_sha256"]) == 64


def test_epoch_proof_logs_exact_contiguous_steps(tmp_path: Path) -> None:
    class Client:
        def __init__(self):
            self.rows = []

        def log_metric(self, run_id, key, value, step):
            self.rows.append((run_id, key, value, step))

    client = Client()
    proof = canary._RestartProof(tmp_path, {}, None, 0, client, "run")
    proof.on_validation_epoch_end(SimpleNamespace(current_epoch=0), None)
    proof.on_validation_epoch_end(SimpleNamespace(current_epoch=1), None)
    assert client.rows == [
        ("run", "epoch_idx", 0.0, 0),
        ("run", "epoch_idx", 1.0, 1),
    ]
    assert proof.validation_epochs == 2

    with pytest.raises(RuntimeError, match="not contiguous"):
        proof.on_validation_epoch_end(SimpleNamespace(current_epoch=3), None)


def test_real_production_callback_suite_round_trips_in_lightning_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    original_callbacks = canary._build_callbacks(tmp_path)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=list(original_callbacks.values()),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        canary._SyntheticModule(),
        datamodule=canary._CanaryDataModule(),
    )
    checkpoint = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(checkpoint)
    expected = canary._load_callback_states(checkpoint, original_callbacks)
    assert expected is not None

    resumed_callbacks = canary._build_callbacks(tmp_path)
    resumed = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        callbacks=list(resumed_callbacks.values()),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    resumed.fit(
        canary._SyntheticModule(),
        datamodule=canary._CanaryDataModule(),
        ckpt_path=checkpoint,
    )
    for name, callback in resumed_callbacks.items():
        assert canary._jsonable(callback.state_dict()) != {}, name


def test_exact_epoch_checkpoint_is_saved_before_train_end(monkeypatch, tmp_path: Path) -> None:
    """The forced preemption must not rely on Lightning's train-end fallback."""
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    checkpoint = canary._build_exact_epoch_checkpoint(tmp_path)
    assert checkpoint.save_last is True
    assert checkpoint.save_top_k == 1
    assert checkpoint._every_n_epochs == 1
    assert checkpoint._save_on_train_epoch_end is False

    class EpochBoundaryWitness(Callback):
        observed = False

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            if int(trainer.current_epoch) != 1:
                return
            path = tmp_path / "exact_epoch" / "last.ckpt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert payload["epoch"] == 0
            assert canary.ExactRetrainRNGState().state_key in payload["callbacks"]
            self.observed = True

    witness = EpochBoundaryWitness()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        callbacks=[canary.ExactRetrainRNGState(), witness, checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(canary._SyntheticModule(), datamodule=canary._CanaryDataModule())
    assert witness.observed


def test_hpc_is_archived_as_non_resume_evidence(tmp_path: Path) -> None:
    root = canary._validate_root(tmp_path / "retrain_restart_canary_archive")
    source = root / "lightning" / "hpc_ckpt_1.ckpt"
    source.parent.mkdir(parents=True)
    source.write_text("mid-epoch\n", encoding="utf-8")

    evidence = canary._archive_hpc(root, restart_count=1)

    assert evidence["resume_eligible"] is False
    assert Path(evidence["path"]).read_text(encoding="utf-8") == "mid-epoch\n"
    assert not source.exists()
    assert not list((root / "lightning").glob("hpc_ckpt_*.ckpt"))


def test_exact_epoch_resume_requires_contiguous_mlflow_history(tmp_path: Path) -> None:
    root = canary._validate_root(tmp_path / "retrain_restart_canary_exact")
    checkpoint = root / "exact_epoch" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 1}, checkpoint)

    class Client:
        def __init__(self):
            self.history = [
                SimpleNamespace(step=0, value=0.0),
                SimpleNamespace(step=1, value=1.0),
            ]

        def get_metric_history(self, run_id, key):
            assert (run_id, key) == ("run", "epoch_idx")
            return self.history

    client = Client()
    resume, evidence = canary._exact_epoch_resume(root, 1, client, "run")

    assert resume == Path(evidence["path"])
    assert evidence["completed_epochs"] == 2
    assert evidence["mlflow_epoch_steps"] == [0, 1]
    client.history[1] = SimpleNamespace(step=2, value=1.0)
    with pytest.raises(RuntimeError, match="differs from contiguous"):
        canary._exact_epoch_resume(root, 2, client, "run")
