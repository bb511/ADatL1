from typing import Optional, Tuple, Dict

import torch
from torch import nn

from src.algorithms import L1ADLightningModule


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

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, s = batch
        z_mean, z_log_var, z, reconstruction = self.forward(x)
        loss = self.loss(
            reconstruction=reconstruction,
            target=x,
            z=z,
            z_mean=z_mean,
            z_log_var=z_log_var,
            s=s
        )
        del reconstruction, x, z, s

        outdict = {
            "loss": loss,

            "loss/total": loss,
            "z_mean.abs().mean()": z_mean.abs().mean(),
            "z_log_var.mean()": z_log_var.mean(),
        }
        if hasattr(self.loss, "losses"):
            outdict.update({
                f"loss/{module.name}": value
                for module, value in zip(
                    self.loss.losses.values(),
                    self.loss.values.values()
                )
            })
        return outdict

    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss/total": outdict.get("loss"),
            **outdict
        }
        
