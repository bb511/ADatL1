from typing import Tuple, Union

import torch
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models.vae import VAE


class MIVAE(VAE):
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use Bernoulli-sampled latents for decoding."""
        z_mean, z_log_var, z, z_sample = self.encoder(x)
        reconstruction = self.decoder(z_sample)
        return z_mean, z_log_var, z, reconstruction
    
    def model_step(self, batch: tuple) -> dict:
        """Add z and labels to loss calculation."""
        x, s = batch
        z_mean, z_log_var, z, reconstruction = self.forward(x)
        total_loss, reco_loss, kl_loss, mi_loss = self.loss(
            target=x,
            reconstruction=reconstruction,
            z_mean=z_mean,
            z_log_var=z_log_var,
            z=z,
            s=s
        )
        
        del z_log_var, z, s, reconstruction
        # garbage_collection_cuda()
        
        return {
            "loss": total_loss.mean() if hasattr(total_loss, 'mean') else total_loss,
            "loss/reco/mean": reco_loss.mean(),
            "loss/kl/mean": kl_loss.mean(),
            "loss/mi/mean": mi_loss.mean(),
            "loss/total/full": total_loss,
            "loss/reco/full": reco_loss,
            "loss/kl/full": kl_loss,
            "loss/mi/full": mi_loss,
            "z_mean_squared": z_mean.pow(2)
        }
    
    def _extract_batch(self, batch: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Extract data and labels from batch."""
        data, labels = batch
        labels = torch.tensor(
            [0 if label == 'zerobias' else 1 for label in labels],
            device=data.device,
            dtype=torch.long
        )
        return data.flatten(start_dim=1).to(dtype=torch.float32), labels
    
    def training_step(self, batch, batch_idx):
        return super().training_step(self._extract_batch(batch), batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return super().validation_step(self._extract_batch(batch), batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return super().test_step(self._extract_batch(batch), batch_idx, dataloader_idx)