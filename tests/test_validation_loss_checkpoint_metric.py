from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms import ADLightningModule
from src.train import _is_checkpoint_only_smoke_run
from src.train import _prepare_data_for_checkpoint_only_evaluation


class _TinyValidationLossModule(ADLightningModule):
    def __init__(self, mi_gamma: float = 0.25) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.mi_gamma = mi_gamma
        self.test_step_calls = 0

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        reco_value, mi_value = batch
        if self.training:
            loss_reco = (self.weight - reco_value.float()).square().mean()
            loss_mi = torch.zeros_like(loss_reco)
        else:
            # Epoch zero must remain the best checkpoint. The batch-dependent
            # values also prove that the metric is aggregated across the loader.
            loss_reco = reco_value.float().mean() + 10.0 * self.current_epoch
            loss_mi = mi_value.float().mean()

        loss_total = loss_reco + self.mi_gamma * loss_mi
        return {
            "loss": loss_total,
            "loss_reco": loss_reco.detach(),
            "loss_mi": loss_mi.detach(),
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.test_step_calls += 1
        return super().test_step(batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_checkpoint_only_evaluation_prepares_datamodule() -> None:
    datamodule = Mock()

    _prepare_data_for_checkpoint_only_evaluation(
        datamodule,
        train_enabled=False,
    )

    datamodule.prepare_data.assert_called_once_with()


def test_trained_run_does_not_prepare_datamodule_twice() -> None:
    datamodule = Mock()

    _prepare_data_for_checkpoint_only_evaluation(
        datamodule,
        train_enabled=True,
    )

    datamodule.prepare_data.assert_not_called()


def test_checkpoint_only_smoke_skips_uncapped_standard_evaluation() -> None:
    assert _is_checkpoint_only_smoke_run(
        train_enabled=False,
        test_enabled=False,
        probe_enabled=True,
        smoke_test_enabled=True,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"train_enabled": True},
        {"test_enabled": True},
        {"probe_enabled": False},
        {"smoke_test_enabled": False},
    ],
)
def test_standard_evaluation_is_retained_outside_checkpoint_only_smoke(
    options,
) -> None:
    settings = {
        "train_enabled": False,
        "test_enabled": False,
        "probe_enabled": True,
        "smoke_test_enabled": True,
    }
    settings.update(options)

    assert not _is_checkpoint_only_smoke_run(**settings)


def test_validation_total_loss_is_aggregated_from_the_normal_loader() -> None:
    module = ADLightningModule()
    module.mi_gamma = 0.25
    module._log_sum = {
        0: {
            "loss_reco": 4.0,
            "loss_mi": 6.0,
        }
    }
    module._log_nsteps = {0: 2}
    module.log_dict = Mock()
    module._trainer = SimpleNamespace(
        val_dataloaders={"normal": object(), "auxiliary": object()},
    )

    module._log_on_epoch_end("val", dataloader_idx=0)

    checkpoint_logs_call = module.log_dict.call_args_list[0]
    assert checkpoint_logs_call.args[0] == {
        "val/loss_reco": pytest.approx(2.0),
        "val/loss_mi": pytest.approx(3.0),
        "val/loss_total": pytest.approx(2.0 + 0.25 * 3.0),
    }
    assert checkpoint_logs_call.kwargs == {
        "on_step": False,
        "on_epoch": True,
        "logger": True,
        "prog_bar": False,
        "sync_dist": True,
        "add_dataloader_idx": False,
    }
    epoch_logs = module.log_dict.call_args_list[1].args[0]
    assert epoch_logs["val/normal/loss_reco"] == pytest.approx(2.0)
    assert epoch_logs["val/normal/loss_mi"] == pytest.approx(3.0)


def test_auxiliary_validation_loader_does_not_publish_checkpoint_metric() -> None:
    module = ADLightningModule()
    module.mi_gamma = 0.25
    module._log_sum = {1: {"loss_reco": 1.0, "loss_mi": 1.0}}
    module._log_nsteps = {1: 1}
    module.log_dict = Mock()
    module._trainer = SimpleNamespace(
        val_dataloaders={"normal": object(), "auxiliary": object()},
    )

    module._log_on_epoch_end("val", dataloader_idx=1)

    module.log_dict.assert_called_once()
    assert "val/loss_total" not in module.log_dict.call_args.args[0]


def test_checkpoint_selects_lowest_complete_validation_total_loss(tmp_path) -> None:
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="val/loss_total",
        filename="loss_total",
        save_top_k=1,
        mode="min",
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    dataset = TensorDataset(
        torch.tensor([1.0, 3.0]),
        torch.tensor([2.0, 4.0]),
    )
    loader = DataLoader(dataset, batch_size=1)
    module = _TinyValidationLossModule(mi_gamma=0.25)

    trainer.fit(
        module,
        train_dataloaders=loader,
        val_dataloaders={"normal": loader, "auxiliary": loader},
    )

    checkpoint_path = tmp_path / "loss_total.ckpt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_epoch_zero_loss = 2.0 + 0.25 * 3.0

    assert checkpoint_path.is_file()
    assert checkpoint.best_model_score == pytest.approx(expected_epoch_zero_loss)
    assert saved["epoch"] == 0
    assert module.test_step_calls == 0
