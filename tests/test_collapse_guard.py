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


def _fit(module, guard, max_epochs, ckpt_path=None):
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
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


@pytest.mark.parametrize(
    "bad",
    [
        {"save_on_train_epoch_end": True},
        {"every_n_train_steps": 10},
        {"filename": "{epoch}-loss"},
        {"monitor": None},
        {"min_joint_code_entropy_bits": -1.0},
        {"save_top_k": 2},
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
    assert guard.dataset == "normal"
    assert guard.checkpoint_path.name == "loss_total.ckpt"
    assert guard.checkpoint_path.parent.name == "guard-test"
