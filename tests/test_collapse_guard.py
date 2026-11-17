"""Collapse guard on loss_total.ckpt (src/callbacks/checkpointing/collapse_guard.py)."""

import json
import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms import ADLightningModule
from src.algorithms.ae import AE
from src.callbacks.checkpointing.collapse_guard import (
    ENTROPY_METRIC,
    STATUS_FALLBACK,
    STATUS_MIN_EPOCH_NOT_REACHED,
    STATUS_OK,
    CollapseGuardedModelCheckpoint,
    count_codes,
    joint_code_entropy_bits,
)

# Four equally frequent 8-bit codes: H = 2 bits. All-zero codes: H = 0.
DIVERSE_CODES = torch.tensor(
    [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
    dtype=torch.uint8,
)
COLLAPSED_CODES = torch.zeros((4, 8), dtype=torch.uint8)


class _ScheduledModule(ADLightningModule):
    """val/loss_total and the latent codes follow a per-epoch schedule.

    schedule[epoch] = (val loss_reco, collapsed?). mi_gamma = 0 so
    val/loss_total == loss_reco exactly.
    """

    def __init__(self, schedule, emit_codes: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.mi_gamma = 0.0
        self.schedule = schedule
        self.emit_codes = emit_codes

    def model_step(self, batch):
        (x,) = batch
        if self.training:
            loss = (self.weight - x.float()).square().mean()
            return {"loss": loss, "loss_reco": loss.detach(), "loss_mi": torch.zeros(())}
        value, collapsed = self.schedule[self.current_epoch]
        out = {
            "loss": torch.tensor(value),
            "loss_reco": torch.tensor(value),
            "loss_mi": torch.tensor(0.0),
        }
        if self.emit_codes:
            out["latent_code"] = COLLAPSED_CODES if collapsed else DIVERSE_CODES
        return out

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _guard(tmp_path, **kwargs) -> CollapseGuardedModelCheckpoint:
    options = dict(
        dirpath=tmp_path,
        monitor="val/loss_total",
        filename="loss_total",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
        every_n_epochs=1,
        min_joint_code_entropy_bits=0.05,
    )
    options.update(kwargs)
    return CollapseGuardedModelCheckpoint(**options)


def _fit(module, guard, max_epochs, ckpt_path=None, **trainer_kwargs):
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        **trainer_kwargs,
        callbacks=[guard],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    loader = DataLoader(TensorDataset(torch.tensor([1.0, 3.0])), batch_size=1)
    trainer.fit(
        module,
        train_dataloaders=loader,
        val_dataloaders={"normal": loader},
        ckpt_path=ckpt_path,
    )
    return trainer


def _saved_epoch(path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


# ----------------------------------------------------------------------
# Entropy statistic
# ----------------------------------------------------------------------
def test_joint_code_entropy_bits() -> None:
    assert joint_code_entropy_bits({b"a": 10}) == 0.0
    assert joint_code_entropy_bits({b"a": 5, b"b": 5}) == pytest.approx(1.0)
    assert joint_code_entropy_bits({i: 3 for i in range(4)}) == pytest.approx(2.0)
    p = 0.995
    expected = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    assert joint_code_entropy_bits({b"a": 995, b"b": 5}) == pytest.approx(expected)
    with pytest.raises(ValueError):
        joint_code_entropy_bits({})


def test_count_codes_matches_latent_collapse_statistic() -> None:
    codes = torch.tensor([[1, 0, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
    counts = count_codes(codes)
    assert sorted(counts.values()) == [1, 1, 2]
    assert joint_code_entropy_bits(counts) == pytest.approx(1.5)


def test_count_codes_rejects_soft_training_samples() -> None:
    with pytest.raises(RuntimeError, match="hard 0/1"):
        count_codes(torch.tensor([[0.3, 1.0]]))


# ----------------------------------------------------------------------
# Checkpoint selection
# ----------------------------------------------------------------------
def test_collapsed_epoch_with_lower_loss_is_not_selected(tmp_path) -> None:
    # Epoch 2 has the lowest loss but a collapsed latent (the Run03 case).
    schedule = {0: (3.0, False), 1: (2.0, False), 2: (1.0, True), 3: (2.5, False)}
    guard = _guard(tmp_path)
    trainer = _fit(_ScheduledModule(schedule), guard, max_epochs=4)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 1
    assert guard.best_model_score == pytest.approx(2.0)
    assert not (tmp_path / "loss_total_collapsed.ckpt").exists()
    assert ENTROPY_METRIC in trainer.callback_metrics

    report = json.loads((tmp_path / "loss_total_guard.json").read_text())
    assert report["status"] == STATUS_OK
    assert report["selected"]["epoch"] == 1
    assert report["selected"]["fallback"] is False
    assert report["selected"]["joint_code_entropy_bits"] == pytest.approx(2.0)
    assert report["n_validation_epochs"] == 4
    assert report["n_collapsed_epochs"] == 1
    assert report["first_collapsed_epoch"] == 2
    assert [r["collapsed"] for r in report["history"]] == [False, False, True, False]
    assert report["history"][2]["joint_code_entropy_bits"] == 0.0
    assert report["guard"]["min_joint_code_entropy_bits"] == 0.05


def test_without_collapse_the_guard_behaves_like_model_checkpoint(tmp_path) -> None:
    schedule = {0: (3.0, False), 1: (1.0, False), 2: (2.0, False)}
    guard = _guard(tmp_path)
    _fit(_ScheduledModule(schedule), guard, max_epochs=3)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 1
    assert json.loads((tmp_path / "loss_total_guard.json").read_text())["n_collapsed_epochs"] == 0


def test_all_epochs_collapsed_falls_back_to_best_collapsed_epoch(tmp_path) -> None:
    schedule = {0: (3.0, True), 1: (1.0, True), 2: (2.0, True)}
    guard = _guard(tmp_path)
    _fit(_ScheduledModule(schedule), guard, max_epochs=3)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 1
    assert not (tmp_path / "loss_total_collapsed.ckpt").exists()
    assert guard.best_model_path == str(tmp_path / "loss_total.ckpt")
    report = json.loads((tmp_path / "loss_total_guard.json").read_text())
    assert report["status"] == STATUS_FALLBACK
    assert report["selected"]["fallback"] is True
    assert report["selected"]["epoch"] == 1
    assert report["n_collapsed_epochs"] == 3


def test_fallback_is_discarded_once_an_epoch_qualifies(tmp_path) -> None:
    # Collapsed at the start, then the latent recovers with a worse loss.
    schedule = {0: (1.0, True), 1: (5.0, False), 2: (4.0, False)}
    guard = _guard(tmp_path)
    _fit(_ScheduledModule(schedule), guard, max_epochs=3)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 2
    assert not (tmp_path / "loss_total_collapsed.ckpt").exists()
    assert json.loads((tmp_path / "loss_total_guard.json").read_text())["status"] == STATUS_OK


def test_threshold_zero_disables_the_guard(tmp_path) -> None:
    schedule = {0: (3.0, False), 1: (1.0, True)}
    guard = _guard(tmp_path, min_joint_code_entropy_bits=0.0)
    _fit(_ScheduledModule(schedule), guard, max_epochs=2)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 1


def test_missing_latent_codes_fail_loudly(tmp_path) -> None:
    schedule = {0: (1.0, False)}
    with pytest.raises(RuntimeError, match="latent_code"):
        _fit(_ScheduledModule(schedule, emit_codes=False), _guard(tmp_path), max_epochs=1)


def test_resume_keeps_guard_state(tmp_path) -> None:
    schedule = {0: (3.0, False), 1: (1.0, True), 2: (2.0, False), 3: (2.5, False)}
    first = _guard(tmp_path)
    _fit(_ScheduledModule(schedule), first, max_epochs=2)
    resume_from = tmp_path / "loss_total.ckpt"
    assert _saved_epoch(resume_from) == 0

    second = _guard(tmp_path)
    _fit(_ScheduledModule(schedule), second, max_epochs=4, ckpt_path=resume_from)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 2
    epochs = [record["epoch"] for record in second.history]
    # Restored from the epoch-0 checkpoint: epoch 0 from the state, 1-3 re-run.
    assert epochs == [0, 1, 2, 3]


# ----------------------------------------------------------------------
# Minimum training length
# ----------------------------------------------------------------------
def _report(tmp_path) -> dict:
    return json.loads((tmp_path / "loss_total_guard.json").read_text())


def test_epochs_before_the_minimum_are_never_saved(tmp_path) -> None:
    # Epoch 0 is by far the best (the Pareto-Front-261015 case); with 3 trained
    # epochs required, epochs 0 and 1 may not compete.
    schedule = {0: (0.1, False), 1: (0.2, False), 2: (3.0, False), 3: (2.0, False), 4: (2.5, False)}
    guard = _guard(tmp_path, min_trained_epochs=3)
    _fit(_ScheduledModule(schedule), guard, max_epochs=5)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 3
    assert guard.best_model_score == pytest.approx(2.0)
    report = _report(tmp_path)
    assert report["status"] == STATUS_OK
    assert report["selected"]["epoch"] == 3
    assert report["guard"]["min_trained_epochs"] == 3
    assert report["guard"]["min_trained_epochs_effective"] == 3
    assert report["guard"]["first_eligible_epoch"] == 2
    assert report["n_epochs_before_min_epoch"] == 2
    assert [r["before_min_epoch"] for r in report["history"]] == [True, True, False, False, False]
    # "saved" = written at that epoch: 2 is the first eligible one, 3 replaces it.
    assert [r["saved"] for r in report["history"]] == [False, False, True, True, False]
    # Early epochs are still recorded with their monitored value.
    assert report["history"][0]["monitor_value"] == pytest.approx(0.1)


@pytest.mark.parametrize("minimum", [0, 1])
def test_minimum_zero_or_one_allows_epoch_zero(tmp_path, minimum) -> None:
    schedule = {0: (1.0, False), 1: (2.0, False)}
    guard = _guard(tmp_path, min_trained_epochs=minimum)
    _fit(_ScheduledModule(schedule), guard, max_epochs=2)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 0
    assert _report(tmp_path)["guard"]["first_eligible_epoch"] == 0


def test_collapsed_fallback_respects_the_minimum(tmp_path) -> None:
    # Every epoch collapsed; the early ones have the lowest loss.
    schedule = {0: (0.1, True), 1: (0.2, True), 2: (3.0, True), 3: (2.0, True)}
    guard = _guard(tmp_path, min_trained_epochs=3)
    _fit(_ScheduledModule(schedule), guard, max_epochs=4)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 3
    assert not (tmp_path / "loss_total_collapsed.ckpt").exists()
    report = _report(tmp_path)
    assert report["status"] == STATUS_FALLBACK
    assert report["selected"]["fallback"] is True
    assert report["selected"]["epoch"] == 3


def test_early_healthy_epoch_does_not_prevent_the_collapsed_fallback(tmp_path) -> None:
    # Only the (too early) epoch 0 is healthy; from the minimum on every epoch
    # collapsed, so the best eligible collapsed epoch is the fallback.
    schedule = {0: (1.0, False), 1: (3.0, True), 2: (2.0, True)}
    guard = _guard(tmp_path, min_trained_epochs=2)
    _fit(_ScheduledModule(schedule), guard, max_epochs=3)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 2
    report = _report(tmp_path)
    assert report["status"] == STATUS_FALLBACK
    assert report["selected"]["epoch"] == 2


def test_short_run_clamps_the_minimum_to_max_epochs(tmp_path) -> None:
    # The 3-epoch local smoke run with min_trained_epochs=15: only the last
    # epoch is eligible.
    schedule = {0: (0.1, False), 1: (0.2, False), 2: (5.0, False)}
    guard = _guard(tmp_path, min_trained_epochs=15)
    _fit(_ScheduledModule(schedule), guard, max_epochs=3)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 2
    report = _report(tmp_path)
    assert report["status"] == STATUS_OK
    assert report["guard"]["min_trained_epochs"] == 15
    assert report["guard"]["min_trained_epochs_effective"] == 3
    assert report["guard"]["first_eligible_epoch"] == 2


def test_training_stopped_before_the_minimum_writes_no_checkpoint(tmp_path) -> None:
    # Two batches per epoch: max_steps=4 stops after epoch 1, long before the
    # first eligible epoch 4 (max_epochs=10 does not clamp).
    schedule = {epoch: (1.0, False) for epoch in range(10)}
    guard = _guard(tmp_path, min_trained_epochs=5)
    _fit(_ScheduledModule(schedule), guard, max_epochs=10, max_steps=4)

    assert not (tmp_path / "loss_total.ckpt").exists()
    assert not guard.best_model_path
    report = _report(tmp_path)
    assert report["status"] == STATUS_MIN_EPOCH_NOT_REACHED
    assert report["selected"] is None
    assert all(r["before_min_epoch"] for r in report["history"])


def test_resume_discards_a_selection_made_under_a_clamped_minimum(tmp_path) -> None:
    # First a 2-epoch run (minimum clamped to 2, epoch 1 saved), then resumed to
    # 5 epochs with the full minimum of 3: the restored epoch-1 selection is
    # before the first eligible epoch 2 and must not keep winning.
    schedule = {0: (9.0, False), 1: (0.1, False), 2: (3.0, False), 3: (2.0, False), 4: (2.5, False)}
    first = _guard(tmp_path, min_trained_epochs=3)
    _fit(_ScheduledModule(schedule), first, max_epochs=2)
    resume_from = tmp_path / "resume.ckpt"
    (tmp_path / "loss_total.ckpt").rename(resume_from)
    assert _saved_epoch(resume_from) == 1

    second = _guard(tmp_path, min_trained_epochs=3)
    _fit(_ScheduledModule(schedule), second, max_epochs=5, ckpt_path=resume_from)

    assert _saved_epoch(tmp_path / "loss_total.ckpt") == 3
    assert second.best_model_score == pytest.approx(2.0)
    report = _report(tmp_path)
    assert report["selected"]["epoch"] == 3
    assert report["guard"]["min_trained_epochs_effective"] == 3
    assert [r["epoch"] for r in report["history"]] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "bad",
    [
        {"save_on_train_epoch_end": True},
        {"every_n_train_steps": 10},
        {"filename": "{epoch}-loss"},
        {"monitor": None},
        {"min_joint_code_entropy_bits": -1.0},
        {"save_top_k": 2},
        {"min_trained_epochs": -1},
        {"min_trained_epochs": 1.5},
        {"min_trained_epochs": True},
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, bad) -> None:
    with pytest.raises(ValueError):
        _guard(tmp_path, **bad)


# ----------------------------------------------------------------------
# AE side: hard codes in validation, nothing new in training
# ----------------------------------------------------------------------
def _ae() -> AE:
    model = AE(
        encoder=nn.Identity(),
        decoder=nn.Identity(),
        input_noise_std=0.0,
        mi_sensitive_num_bins=2,
        mi_num_permutations=0,
    )
    model._compute_sensitive_bins = lambda x, mask: torch.zeros(x.shape[0], dtype=torch.long)
    model.compute_operational_ascore = lambda ascore: torch.tensor(0.0)
    return model


def test_ae_returns_hard_codes_outside_training_only() -> None:
    model = _ae()
    x = torch.tensor([[1.0, -1.0], [0.25, 0.5], [-2.0, -0.1]])
    batch = (x, torch.zeros(3))

    model.eval()
    rng_before = torch.get_rng_state()
    out = model.model_step(batch)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert out["latent_code"].dtype == torch.uint8
    torch.testing.assert_close(out["latent_code"], (x >= 0).to(torch.uint8))
    torch.testing.assert_close(
        out["latent_code"].float(), model.forward_with_representations(x)["latent_sample"]
    )

    model.train()
    assert "latent_code" not in model.model_step(batch)


# ----------------------------------------------------------------------
# Config wiring
# ----------------------------------------------------------------------
@pytest.mark.parametrize("experiment", ["physics/ae", "physics/pareto_fet_train"])
def test_physics_experiments_use_the_guarded_loss_total_checkpoint(
    monkeypatch, tmp_path, experiment
) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment={experiment}",
                "run_name=guard-test",
                f"paths.checkpoints_dir={tmp_path}",
            ],
        )

    guard = instantiate(cfg.callbacks.loss_total_ckpt)
    assert isinstance(guard, CollapseGuardedModelCheckpoint)
    assert guard.monitor == "val/loss_total"
    assert guard.filename == "loss_total"
    assert guard.mode == "min"
    assert guard.min_joint_code_entropy_bits == pytest.approx(0.05)
    assert guard.min_trained_epochs == 15
    assert guard.dataset == "normal"
    assert guard.checkpoint_path.name == "loss_total.ckpt"
    assert guard.checkpoint_path.parent.name == "guard-test"
