from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models import L1ADLightningModule


class VAE(L1ADLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        features: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )

        self.encoder, self.decoder = encoder, decoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    def model_step(self, x: torch.Tensor) -> torch.Tensor:
        z_mean, z_log_var, z, reconstruction = self.forward(x)
        total_loss, reco_loss, kl_loss = self.loss(
            reconstruction=reconstruction, z_mean=z_mean, z_log_var=z_log_var, target=x
        )
        del z_mean, z_log_var, z, reconstruction
        garbage_collection_cuda()

        return {
            # Used for backpropagation:
            "loss": total_loss.mean(),
            # Used for logging:
            "loss/reco/mean": reco_loss.mean(),
            "loss/kl/mean": kl_loss.mean(),
            # Used for callbacks:
            "loss/total/full": total_loss,
            "loss/reco/full": reco_loss,
            "loss/kl/full": kl_loss,
        }

    def _filter_log_dict(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_kl": outdict.get("loss/kl/mean"),
        }
