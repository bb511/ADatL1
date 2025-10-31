from typing import Optional, Tuple, Union, Dict

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
        masking: Optional[nn.Module] = None
    ):
        super().__init__()
        self.loss = loss
        self.model = model
        self.masking = masking if masking is not None else nn.Identity()
        self.save_hyperparameters(ignore=["model", "loss"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override with the forward pass."""
        return self.model(self.masking(x))

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Template model step during training. Override for each model."""
        x, y = batch
        z = self.forward(x)
        loss = self.loss(z=z, y=y)

        del z, x, y

        return {"loss": loss}

    def _log_dict(self, outdict: dict, stage: str, dataloader_idx: int):
        """Compile the dictionary with loss/metric values to log during training."""
        outdict = self.outlog(outdict)

        if stage == "train":
            return {f"{stage}/{k}": v for k, v in outdict.items()}

        dset_key = list(getattr(self.trainer, f"{stage}_dataloaders").keys())[
            dataloader_idx
        ]
        return {f"{stage}/{dset_key}/{k}": v for k, v in outdict.items()}

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        outdict = self.model_step(batch)

        # Decide what to log:
        self.log_dict(
            self._log_dict(outdict, "train", dataloader_idx=0),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return outdict

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = 0
    ):
        outdict = self.model_step(batch)
        self.log_dict(
            self._log_dict(outdict, "val", dataloader_idx=dataloader_idx),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        return outdict

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = 0
    ):
        outdict = self.model_step(batch)

        # Decide what to log:
        self.log_dict(
            self._log_dict(outdict, "test", dataloader_idx=dataloader_idx),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,  # !!
            add_dataloader_idx=False,
        )
        return outdict

    def on_train_epoch_end(self):
        """Clean up memory."""
        garbage_collection_cuda()

    def on_validation_epoch_end(self):
        """Log the epochs so the mlflow plotting is not buggy."""
        self.log("epoch_idx", float(self.current_epoch), on_epoch=True, on_step=False)
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
        scheduler = self.hparams.scheduler.scheduler(optimizer=optimizer)
        scheduler_dict = OmegaConf.to_container(self.hparams.scheduler, resolve=True)
        scheduler_dict.update({"scheduler": scheduler})

        return scheduler_dict
