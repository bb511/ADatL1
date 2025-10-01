from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models import L1ADLightningModule


class AE(L1ADLightningModule):
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
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def model_step(self, x: torch.Tensor) -> torch.Tensor:
        z, reconstruction = self.forward(x)
        total_loss, reco_loss, nad_loss = self.loss(
            reconstruction=reconstruction, latent=z, target=x
        )

        return {
            # Used for backpropagation:
            "loss": total_loss.mean(),
            # Used for logging:
            "loss/reco/mean": reco_loss.mean(),
            "loss/nad/mean": kl_loss.mean(),
        }

    def _filter_log_dict(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_nad": outdict.get("loss/nad/mean"),
        }
