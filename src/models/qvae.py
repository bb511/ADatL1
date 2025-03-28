from typing import Optional, Tuple

import torch
from torch import nn, optim
import torch.nn.functional as F

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
        **kwargs
    ):
        super().__init__(model=None, **kwargs)
        self.encoder, self.decoder = encoder, decoder
        self.save_hyperparameters(ignore=["model", "encoder", "decoder", "loss"])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction
    
    def _extract_batch(self, batch: tuple) -> torch.Tensor:
        # TODO: Remove after debugging:
        batch = batch[0]
        return torch.flatten(batch, start_dim=1).to(dtype=torch.float32)
    
    def model_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = self._extract_batch(batch)

        z_mean, z_log_var, z, reconstruction = self.forward(x)
        total_loss, reco_loss, kl_loss = self.loss(
            reconstruction=reconstruction,
            z_mean=z_mean,
            z_log_var=z_log_var,
            target=x
        )

        # Compute anomaly score
        anomaly_score = self.get_anomaly_score(x)
        if self.logger != None:
            self.logger.experiment.add_histogram('val/anomaly_score_distribution', anomaly_score, self.current_epoch)

        return {
            "loss": total_loss,
            "loss_reco": reco_loss,
            "loss_kl": kl_loss,
        }
    
    def get_anomaly_score(self, x: torch.Tensor):
        """Calculate anomaly score based on reconstruction error"""
        z_mean, _, _ = self.encoder(x)
        reconstruction = self.decoder(z_mean)
        return F.mse_loss(reconstruction, x, reduction='none').sum(dim=1)
    


class FeatureQVAE(QVAE):
    """A feature extractor is passed first through the data."""

    def __init__(
        self,
        features: nn.Module,
        model = None,
        **kwargs
    ):
        super().__init__(model=None, **kwargs)
        self.features = features
        self.save_hyperparameters(ignore=["model", "encoder", "decoder", "loss", "features"])

    def _extract_batch(self, batch: tuple) -> torch.Tensor:
        batch = super()._extract_batch(batch)
        with torch.no_grad():
            batch = self.features(batch)

        import ipdb; ipdb.set_trace()
    