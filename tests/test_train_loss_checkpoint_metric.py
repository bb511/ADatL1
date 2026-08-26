from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms import ADLightningModule


class _TinyLossModule(ADLightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        x, target = batch
        loss = torch.nn.functional.mse_loss(self.layer(x), target)
        return {"loss": loss}

    def outlog(self, outdict: dict) -> dict:
        return {"loss": outdict["loss"]}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_training_loss_is_bridged_until_validation_checkpointing() -> None:
    module = ADLightningModule()
    module._log_sum = {0: {"loss": 3.0, "loss_reco": 2.0}}
    module._log_nsteps = {0: 2}
    module.log_dict = Mock()

    module._log_on_epoch_end("train")

    assert module._checkpoint_train_loss_total == pytest.approx(1.5)
    module.log_dict.assert_called_once()
    assert module.log_dict.call_args.args[0]["train/loss"] == pytest.approx(1.5)

    module.log = Mock()
    module._trainer = SimpleNamespace(current_epoch=0)
    module.on_validation_epoch_end()

    checkpoint_log = module.log.call_args_list[0]
    assert checkpoint_log.args[0] == "checkpoint/train_loss_total"
    assert checkpoint_log.args[1] == pytest.approx(1.5)
    assert checkpoint_log.kwargs == {
        "on_step": False,
        "on_epoch": True,
        "logger": False,
        "prog_bar": False,
        "sync_dist": False,
    }


def test_model_checkpoint_can_monitor_bridged_train_loss_after_validation(
    tmp_path,
) -> None:
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="checkpoint/train_loss_total",
        filename="loss_total",
        save_top_k=1,
        mode="min",
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    dataset = TensorDataset(
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
    )
    loader = DataLoader(dataset, batch_size=2)

    trainer.fit(
        _TinyLossModule(),
        train_dataloaders=loader,
        val_dataloaders={"normal": loader, "other": loader},
    )

    assert (tmp_path / "loss_total.ckpt").is_file()
    assert checkpoint.best_model_score is not None
