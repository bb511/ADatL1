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
        import ipdb; ipdb.set_trace()
        total_loss, reco_loss, kl_loss = self.loss(*self(batch), target=batch)
        return {
            "loss": total_loss,
            "loss_reco": reco_loss,
            "loss_kl": kl_loss
        }