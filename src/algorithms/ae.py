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
        mse: nn.Module,
        mask: bool = True,
        features: Optional[nn.Module] = None,
        input_noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )
        self.encoder, self.decoder = encoder, decoder
        self.mse = mse
        self.input_noise_std = input_noise_std
        self.mask = mask
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def model_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, m, _, _ = batch
        x = torch.flatten(x, start_dim=1)
        m = torch.flatten(m, start_dim=1).float()

        x_noisy = x
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(x) * self.input_noise_std
            noise = noise * m
            x_noisy = x + noise

        z, reconstruction = self.forward(x_noisy)
        reco_loss = self.loss(reco=reconstruction, target=x, mask=m)
        mse = self.mse(x, reconstruction, m)
        del x, z

        with torch.no_grad():
            rmse_q95 = torch.quantile(torch.sqrt(mse), 0.95).item()

        return {
            # Used for backpropagation:
            "loss": reco_loss.mean(),
            # Used for logging:
            "loss/reco/mean": reco_loss.mean(),
            "loss/mse/sqrt_q95": rmse_q95,
            # Used for callbacks:
            "loss/reco/full": reco_loss.detach(),
            "loss/mse/full": mse.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """The values of the loss that are logged."""
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_rmse_q95": outdict.get("loss/mse/sqrt_q95"),
        }
