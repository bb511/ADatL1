from typing import Optional
import inspect

import torch
from torch import nn, optim

from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.memory import garbage_collection_cuda


class L1ADLightningModule(LightningModule):
    """Base class for AD@L1 LightningModules."""

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[DictConfig] = None,
        masking: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.loss = loss
        self.model = model
        self.masking = masking if masking is not None else nn.Identity()
        self.save_hyperparameters(ignore=["model", "loss"])

        self._log_sum = {}
        self._log_nsteps = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override with the forward pass."""
        return self.model(self.masking(x))

    def model_step(self, batch: tuple[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Template model step during training. Override for each model."""
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z=z, y=y)

        del z, x, y

        return {"loss": loss}

    def _log_dict(self, outdict: dict, dataloader_idx: int):
        """Compile the dictionary with loss/metric values to log during training."""
        outdict = self.outlog(outdict)
        if dataloader_idx not in self._log_sum:
            self._log_sum[dataloader_idx] = {}
            self._log_nsteps[dataloader_idx] = 0

        for mname, mvalue in outdict.items():
            if isinstance(mvalue, (int, float)):
                self._log_sum[dataloader_idx][mname] = \
                    self._log_sum[dataloader_idx].get(mname, 0.0) + float(mvalue)
            elif torch.is_tensor(mvalue) and mvalue.ndim == 0:
                self._log_sum[dataloader_idx][mname] = \
                    self._log_sum[dataloader_idx].get(mname, 0.0) + float(mvalue.detach())

        self._log_nsteps[dataloader_idx] += 1

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        outdict = self.model_step(batch)
        self._log_dict(outdict, dataloader_idx=0)
        return outdict

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ):
        outdict = self.model_step(batch)
        self._log_dict(outdict, dataloader_idx=dataloader_idx)
        return outdict

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        outdict = self.model_step(batch)
        return outdict

    def _log_on_epoch_end(self, stage: str, dataloader_idx: int = 0):
        nsteps = max(self._log_nsteps[dataloader_idx], 1)
        logs = self._log_sum[dataloader_idx]
        if stage == "train":
            logs = {f"train/{k}": v / nsteps for k, v in logs.items()}
        else:
            datasets = list(getattr(self.trainer, f"{stage}_dataloaders").keys())
            dataset_name = datasets[dataloader_idx]
            logs = {
                f"{stage}/{dataset_name}/{k}": v / nsteps
                for k, v in logs.items()
            }

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=False,  # set True only for a few key metrics if needed
            add_dataloader_idx=False
        )

        self._log_sum[dataloader_idx] = {}
        self._log_nsteps[dataloader_idx] = 0

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log metrics and clean up memory.

        For some reason on_train_epoch_end executes after on_validation_epoch_end
        so I have to do this hack to do things right at the end of the training.
        """
        is_last = (batch_idx + 1) == self.trainer.num_training_batches
        if is_last:
            self._log_on_epoch_end('train')
            garbage_collection_cuda()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Log quantities at the end of the last validation epoch."""
        is_last = (batch_idx + 1) == self.trainer.num_val_batches[dataloader_idx]
        if is_last:
            self._log_on_epoch_end('val', dataloader_idx=dataloader_idx)

    def on_train_epoch_end(self):
        """Clean up memory after finishing one training epoch and validation."""
        garbage_collection_cuda()

    def on_validation_epoch_end(self):
        """Log the epochs so the mlflow plotting is not buggy, clean memory."""
        self.log("epoch_idx", float(self.current_epoch), on_epoch=True, on_step=False)

    def on_test_epoch_end(self):
        """Clean up memory."""
        garbage_collection_cuda()

    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return outdict

    def configure_optimizers(self) -> dict:
        optimizer = LightningOptimizer(self.hparams.optimizer(params=self.parameters()))

        if self.hparams.scheduler:
            scheduler_dict = self._set_up_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": optimizer}

    def _set_up_scheduler(self, optimizer: optim.Optimizer) -> dict:
        scheduler_fn = self.hparams.scheduler.scheduler
        param_names = inspect.signature(scheduler_fn).parameters
        kwargs = {}
        if "total_steps" in param_names:
            kwargs["total_steps"] = int(self.trainer.estimated_stepping_batches)
        elif "T_max" in param_names:
            kwargs["T_max"] = int(self.trainer.estimated_stepping_batches)

        scheduler = scheduler_fn(optimizer=optimizer, **kwargs)
        scheduler_dict = OmegaConf.to_container(self.hparams.scheduler, resolve=True)
        scheduler_dict.update({"scheduler": scheduler})

        return scheduler_dict
