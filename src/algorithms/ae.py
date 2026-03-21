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
        operational_rate: float = 0.25,
        bc_rate: float = 28608.8064,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.encoder, self.decoder = encoder, decoder
        self.mse = mse
        self.input_noise_std = input_noise_std
        self.mask = mask
        self.operational_quantile = 1 - operational_rate / bc_rate

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
            n = mse.numel()

            # Number of extreme tail samples
            k = max(10, int((1.0 - self.operational_quantile) * n))
            topk_vals = torch.topk(mse, k).values
            mean_top_vals = topk_vals.mean().item()

            mse_q99 = torch.quantile(mse, 0.99).item()

        return {
            # Used for backpropagation:
            "loss": reco_loss.mean(),
            # Used for logging:
            "loss/reco/mean": reco_loss.mean(),
            "loss/mse/mean_top_vals": mean_top_vals,
            "loss/mse/q99": mse_q99,
            # Used for callbacks:
            "loss/total/full": reco_loss.detach(),
            "loss/mse/full": mse.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """The values of the loss that are logged."""
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_mse_mean_top_vals": outdict.get("loss/mse/mean_top_vals"),
        }
