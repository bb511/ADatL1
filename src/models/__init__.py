from typing import Optional

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
    ):
        super().__init__()
        self.loss = loss
        self.model = model
        self.save_hyperparameters(ignore=["model", "loss"])

    def _extract_batch(self, batch: tuple) -> torch.Tensor:
        """Override to adjust to the specific dataset."""
        return batch
    
    def model_step(self, batch: tuple, batch_idx: int):
        """Override with the model forward pass."""
        x = self._extract_batch(batch)
        garbage_collection_cuda()
        z = self.model(x)
        return {
            "loss": self.loss(x, z),
        }
    
    def _outlog(self, out: dict, stage: str):
        """Override with the values you want to log."""
        return {f"{stage}/loss": out["loss"]}
        
    def training_step(self, batch: tuple, batch_idx: int):
        out = self.model_step(batch, batch_idx)
        self.log_dict(self._outlog(out, "train"), prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return out

    def validation_step(self, batch: tuple, batch_idx: int):
        out = self.model_step(batch, batch_idx)
        self.log_dict(self._outlog(out, "val"), prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return out

    def test_step(self, batch: tuple, batch_idx: int):
        out = self.model_step(batch, batch_idx)
        self.log_dict(self._outlog(out, "test"), prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=False) # SINGLE DEVICE
        return out

    def configure_optimizers(self):
        optimizer = LightningOptimizer(self.hparams.optimizer(params=self.parameters()))

        if self.hparams.scheduler:
            scheduler = self.hparams.scheduler.scheduler(optimizer=optimizer)

            scheduler_dict = OmegaConf.to_container(self.hparams.scheduler, resolve=True) # convert to normal dict
            scheduler_dict.update({
                    "scheduler": scheduler,
            })

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_dict
            }
        
        return {"optimizer": optimizer}