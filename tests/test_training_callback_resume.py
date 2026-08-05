from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.cap import CAPCallback
from src.callbacks.checkpointing.criterion import Stable
from src.callbacks.efficiency import AnomalyEfficiencyCallback
from src.callbacks.model_checkpoint import SingleDatasetModelCheckpoint
from src.callbacks.thres_drift import ThresholdDriftCallback
from src.callbacks.wasserstein import WassersteinCallback


class _RateModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hparams = SimpleNamespace(target_rate=10.0, base_rate=40_000.0)

    @property
    def device(self):
        return next(self.parameters(), torch.empty(0)).device


class _TinyValidationModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        return self.layer(x).square().mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/normal/anomaly_score", 1.0, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize(
    ("source", "target", "attribute", "value"),
    [
        (
            CAPCallback("score", "background0", "background1", "cdf", {}),
            CAPCallback("score", "background0", "background1", "cdf", {}),
            "cap_ema",
            1.25,
        ),
        (
            WassersteinCallback("score", "background0", "background1"),
            WassersteinCallback("score", "background0", "background1"),
            "w1dist_ema",
            2.5,
        ),
        (
            ThresholdDriftCallback("score", dataset_1="background0", dataset_2="background1"),
            ThresholdDriftCallback("score", dataset_1="background0", dataset_2="background1"),
            "drift_ema",
            {"10_0": 0.75},
        ),
    ],
)
def test_model_selection_ema_survives_callback_round_trip(source, target, attribute, value):
    setattr(source, attribute, value)

    target.load_state_dict(source.state_dict())

    restored = getattr(target, attribute)
    if isinstance(value, dict):
        assert dict(restored) == value
    else:
        assert restored == value


def test_efficiency_registers_thresholds_before_model_restore_and_preserves_ema():
    source_module = _RateModule()
    source = AnomalyEfficiencyCallback(
        "anomaly_score_oas",
        target_rates=[5.0],
        threshold_namespace="oas",
        metric_suffix="oas",
    )
    source.setup(None, source_module, "fit")
    source_module.thres_operational__oas.fill_(3.0)
    source_module.thres_5_0kHz__oas.fill_(4.0)
    source.cvar25_ema = {10.0: 0.4}
    source.cvar10_ema = {10.0: 0.2}

    restored_module = _RateModule()
    restored = AnomalyEfficiencyCallback(
        "anomaly_score_oas",
        target_rates=[5.0],
        threshold_namespace="oas",
        metric_suffix="oas",
    )
    restored.setup(None, restored_module, "fit")
    restored_module.load_state_dict(source_module.state_dict(), strict=True)
    restored.load_state_dict(source.state_dict())
    restored.on_fit_start(SimpleNamespace(), restored_module)

    assert restored_module.thres_operational__oas.item() == pytest.approx(3.0)
    assert restored_module.thres_5_0kHz__oas.item() == pytest.approx(4.0)
    assert restored.cvar25_ema == {10.0: 0.4}
    assert restored.cvar10_ema == {10.0: 0.2}


def test_callback_state_keys_distinguish_parallel_metric_instances():
    cap_cdf = CAPCallback("score", "background0", "background1", "cdf", {})
    cap_mapping = CAPCallback(
        "score",
        "background0",
        "background1",
        "mapping",
        {},
        pairing_index_path="pairing.pt",
    )
    drift_forward = ThresholdDriftCallback(
        "score", dataset_1="background0", dataset_2="background1"
    )
    drift_reverse = ThresholdDriftCallback(
        "score", dataset_1="background1", dataset_2="background0"
    )
    native_efficiency = AnomalyEfficiencyCallback("score")
    oas_efficiency = AnomalyEfficiencyCallback(
        "score_oas", threshold_namespace="oas", metric_suffix="oas"
    )

    keys = {
        cap_cdf.state_key,
        cap_mapping.state_key,
        drift_forward.state_key,
        drift_reverse.state_key,
        native_efficiency.state_key,
        oas_efficiency.state_key,
    }
    assert len(keys) == 6


def test_dataset_checkpoint_resume_preserves_stable_window_and_directory(tmp_path):
    callback = SingleDatasetModelCheckpoint(
        monitor="anomaly_score",
        criterion=Stable(top_k=1, threshold=0.05, patience=3),
        ds=["normal"],
        dirpath=str(tmp_path),
    )
    trainer = SimpleNamespace(
        val_dataloaders={"normal": object()}, sanity_checking=False, ckpt_path=None
    )
    callback.on_fit_start(trainer, None)
    callback.on_validation_start(trainer, None)
    callback.ds_criterion["normal"].check(1.00)
    callback.ds_criterion["normal"].check(1.01)
    retained = callback.dirpath / "retained.ckpt"
    retained.write_text("checkpoint", encoding="utf-8")
    callback.checkpoints["normal"].append({"value": 1.0, "epoch": 7, "fpath": retained})

    restored = SingleDatasetModelCheckpoint(
        monitor="anomaly_score",
        criterion=Stable(top_k=1, threshold=0.05, patience=3),
        ds=["normal"],
        dirpath=str(tmp_path),
    )
    restored.load_state_dict(callback.state_dict())
    restored.on_fit_start(SimpleNamespace(ckpt_path="last.ckpt"), None)

    assert retained.read_text(encoding="utf-8") == "checkpoint"
    assert list(restored.ds_criterion["normal"]._window) == [1.0, 1.01]
    assert restored.checkpoints["normal"][0]["epoch"] == 7


def test_dataset_checkpoint_fresh_fit_cleans_stale_directory(tmp_path):
    callback = SingleDatasetModelCheckpoint(
        monitor="anomaly_score",
        criterion=Stable(top_k=1, threshold=0.05, patience=3),
        ds=["normal"],
        dirpath=str(tmp_path),
    )
    stale = callback.dirpath / "stale.ckpt"
    stale.write_text("stale", encoding="utf-8")

    callback.on_fit_start(SimpleNamespace(ckpt_path=None), None)

    assert callback.dirpath.is_dir()
    assert not stale.exists()


@pytest.mark.parametrize("retained_path", ["missing.ckpt", "../escaped.ckpt"])
def test_dataset_checkpoint_resume_rejects_invalid_bookkeeping(tmp_path, retained_path):
    callback = SingleDatasetModelCheckpoint(
        monitor="anomaly_score",
        criterion=Stable(top_k=1, threshold=0.05, patience=3),
        ds=["normal"],
        dirpath=str(tmp_path),
    )
    callback.checkpoints["normal"].append(
        {"value": 1.0, "epoch": 7, "fpath": callback.dirpath / retained_path}
    )
    restored = SingleDatasetModelCheckpoint(
        monitor="anomaly_score",
        criterion=Stable(top_k=1, threshold=0.05, patience=3),
        ds=["normal"],
        dirpath=str(tmp_path),
    )
    restored.load_state_dict(callback.state_dict())

    with pytest.raises(RuntimeError, match="does not exist|escapes"):
        restored.on_fit_start(SimpleNamespace(ckpt_path="last.ckpt"), None)


def test_lightning_resume_restores_stable_window_before_fit_start(tmp_path):
    train_loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    val_loaders = {"normal": DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)}
    callback_kwargs = {
        "monitor": "anomaly_score",
        "criterion": Stable(top_k=1, threshold=0.05, patience=2),
        "ds": ["normal"],
        "dirpath": str(tmp_path / "retained"),
    }
    original = SingleDatasetModelCheckpoint(**callback_kwargs)
    trainer = Trainer(
        max_epochs=1,
        callbacks=[original],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(
        _TinyValidationModule(),
        train_dataloaders=train_loader,
        val_dataloaders=val_loaders,
    )
    resume_path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(resume_path)
    sentinel = original.dirpath / "preemption-sentinel"
    sentinel.write_text("retained", encoding="utf-8")

    resumed = SingleDatasetModelCheckpoint(**callback_kwargs)
    resumed_trainer = Trainer(
        max_epochs=2,
        callbacks=[resumed],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
    )
    resumed_trainer.fit(
        _TinyValidationModule(),
        train_dataloaders=train_loader,
        val_dataloaders=val_loaders,
        ckpt_path=resume_path,
    )

    assert sentinel.read_text(encoding="utf-8") == "retained"
    assert len(resumed.ds_criterion["normal"]._window) == 2
    assert resumed.checkpoints["normal"]
