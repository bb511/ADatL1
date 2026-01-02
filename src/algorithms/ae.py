# Vanilla auto-encoder model implementations
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.algorithms import L1ADLightningModule


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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def model_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _, _ = batch
        x = torch.flatten(x, start_dim=1)
        z, reconstruction = self.forward(x)
        reco_loss = self.loss(reco=reconstruction, target=x)

        del x, z

        return {
            # Used for backpropagation:
            "loss": reco_loss.mean(),
            # Used for logging:
            "loss/reco/mean": reco_loss.mean(),
        }

    def outlog(self, outdict: dict) -> dict:
        """The values of the loss that are logged."""
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
        }
