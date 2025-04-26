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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override with the forward pass."""
        return self.model(x)
    
    def model_step(self, x: torch.Tensor):
        """Override with the model forward pass."""
        z = self.forward(x)
        loss = self.loss(x, z)
        del z, x; garbage_collection_cuda()
        return {
            "loss": loss
        }
    
    def filter_log_dict(self, outdict: dict, stage: str):
        """Override with the values you want to log."""
        return {f"{stage}/{k}": v for k, v in outdict.items()}
        
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        outdict = self.model_step(
            batch.flatten(start_dim=1).to(dtype=torch.float32)
        )

        # Decide what to log:
        self.log_dict(
            self.filter_log_dict(outdict, "train"),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True
        )
        return outdict

    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = 0):
        outdict = self.model_step(
            batch.flatten(start_dim=1).to(dtype=torch.float32)
        )

        # Decide what to log:
        self.log_dict(
            self.filter_log_dict(outdict, "val"),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=True
        )
        return outdict

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = 0):
        outdict = self.model_step(
            batch.flatten(start_dim=1).to(dtype=torch.float32)
        )

        # Decide what to log:
        self.log_dict(
            self.filter_log_dict(outdict, "test"),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False, # !!
            add_dataloader_idx=True
        )
        return outdict

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