from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models import L1ADLightningModule

class QVAE(L1ADLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        features: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "encoder", "decoder", "loss"])

        self.encoder, self.decoder = encoder, decoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction
    
    def model_step(self, x: torch.Tensor) -> torch.Tensor:
        z_mean, z_log_var, z, reconstruction = self.forward(x)
        total_loss, reco_loss, kl_loss = self.loss(
            reconstruction=reconstruction,
            z_mean=z_mean,
            z_log_var=z_log_var,
            target=x
        )
        del z_mean, z_log_var, z, reconstruction; garbage_collection_cuda()

        # Compute anomaly score
        # anomaly_score = self.get_anomaly_score(x)
        # if self.logger != None:
        #     self.logger.experiment.add_histogram('val/anomaly_score_distribution', anomaly_score, self.current_epoch)

        return {
            "loss": total_loss,
            "loss_reco": reco_loss,
            "loss_kl": kl_loss,
        }
    
    # def get_anomaly_score(self, x: torch.Tensor):
    #     """Calculate anomaly score based on reconstruction error"""
    #     z_mean, _, _ = self.encoder(x)
    #     reconstruction = self.decoder(z_mean)
    #     return F.mse_loss(reconstruction, x, reduction='none').sum(dim=1)