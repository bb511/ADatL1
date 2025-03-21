from typing import Optional, Tuple

import torch
from torch import nn, optim

from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models import L1ADLightningModule

class QVAE(L1ADLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.loss = loss
        self.encoder, self.decoder = encoder, decoder
        self.save_hyperparameters(ignore=["encoder", "decoder", "loss"])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction
    
    def model_step(self, batch: torch.Tensor) -> torch.Tensor:
        batch_flat = torch.flatten(batch, start_dim=1)
        z_mean, z_log_var, z, reconstruction = self.forward(batch_flat)
        total_loss, reco_loss, kl_loss = self.loss(
            reconstruction=reconstruction,
            z_mean=z_mean,
            z_log_var=z_log_var,
            target=batch_flat
        )

        return {
            "loss": total_loss,
            "loss_reco": reco_loss,
            "loss_kl": kl_loss
        }
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        output = self.model_step(batch)

        loss = output.get("loss")
        self.log_dict({f"train/{key}": value for key, value in output.items()})
        return loss
        
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        output = self.model_step(batch)

        loss = output.get("loss")
        self.log_dict({f"val/{key}": value for key, value in output.items()})
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        output = self.model_step(batch)

        loss = output.get("loss")
        self.log_dict({f"test/{key}": value for key, value in output.items()})
        return loss
        
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
