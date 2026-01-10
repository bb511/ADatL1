# Variational auto-encoder model implementation.
from typing import Optional, Tuple, Dict

import torch
from torch import nn
import keras

from src.algorithms import L1ADLightningModule
from src.algorithms.components.utils import LinearWarmup
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class VAE(L1ADLightningModule):
    """Variational autoencoder architecture.

    :param encoder: The encoder nn module.
    :param decoder: The decoder nn module.
    :param kl_warmup_frac: Float specifying the fraction of the total epochs to warmp
        up the KL scaling factor over.
    :param features: An nn module to apply to the data and give a space for the vae
        to act on, e.g., vicreg.
    :param mask: Boolean determining whether one should mask input features, i.e., mask
        where the padded values are in the mse loss.
    """
    def __init__(
        self,
        encoder: nn.Module | keras.Model,
        decoder: nn.Module | keras.Model,
        kl_warmup_frac: float = 0.0,
        features: Optional[nn.Module] = None,
        mask: bool = True,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss"]
        )
        self.mask = mask
        self.encoder, self.decoder = encoder, decoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()
        self.kl_warmup_frac = kl_warmup_frac

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.features(x)
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    def on_fit_start(self):
        total_steps = int(self.trainer.estimated_stepping_batches)
        self._setup_kl_annealing(self.kl_warmup_frac, total_steps)

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, m, _ = batch
        x = torch.flatten(x, start_dim=1)
        m = torch.flatten(m, start_dim=1).float()
        z_mean, z_log_var, z, reconstruction = self.forward(x)

        kl_current_scale = self.kl_scale(int(self.global_step))
        reco_loss, kl_raw, kl_scaled, total_loss = self.loss(
            reconstruction=reconstruction,
            z_mean=z_mean,
            z_log_var=z_log_var,
            target=self.features(x),
            mask=m if self.mask else None,
            kl_scale=kl_current_scale
        )
        del x, z, z_mean, z_log_var

        # Add additional losses if using HGQ.
        add_loss = 0.0
        if hasattr(self.encoder, "losses") and len(self.encoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.encoder.losses]).sum()
        if hasattr(self.decoder, "losses") and len(self.decoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.decoder.losses]).sum()

        total_loss = total_loss + add_loss

        # Diagnosis metrics.
        with torch.no_grad():
            kl_quantiles = torch.tensor([0.5, 0.95, 0.99, 0.999], device=kl_raw.device)
            kl_q50, kl_q95, kl_q99, kl_q999 = torch.quantile(kl_raw, kl_quantiles).tolist()
            rmse_quantile = torch.tensor(0.95, device=reco_loss.device)
            rmse_q95 = torch.quantile(torch.sqrt(reco_loss), rmse_quantile).item()

        return {
            # Used for backpropagation:
            "loss": total_loss.mean(),

            # Used for logging:
            "loss/reco/mean": reco_loss.detach().mean(),
            "loss/kl_scaled/mean": kl_scaled.detach().mean(),
            "loss/kl_raw/mean": kl_raw.detach().mean(),
            "loss/kl_raw/median": kl_q50,
            "loss/kl_raw/q95": kl_q95,
            "loss/kl_raw/q99": kl_q99,
            "loss/kl_raw/q999": kl_q999,
            "kl_scale": kl_current_scale,
            "loss/reco/sqrt_q95": rmse_q95,

            # Used for callbacks:
            "loss/total/full": total_loss.detach(),
            "loss/reco/full": reco_loss.detach(),
            "loss/kl_raw/full": kl_raw.detach(),
            "reconstructed_data": reconstruction.detach()
        }

    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_kl_raw": outdict.get("loss/kl_raw/mean"),
            "loss_kl_scaled": outdict.get("loss/kl_scaled/mean"),
            "loss_kl_median": outdict.get("loss/kl_raw/median"),
            "loss_kl_raw_q95": outdict.get("loss/kl_raw/q95"),
            "loss_kl_raw_q99": outdict.get("loss/kl_raw/q99"),
            "loss_kl_raw_q999": outdict.get("loss/kl_raw/q999"),
            "loss_rmse_q95": outdict.get("loss/reco/sqrt_q95"),
            "kl_scale": outdict.get("kl_scale"),
        }

    def _setup_kl_annealing(self, kl_warmup_frac: float, total_steps: int):
        """Sets up the kl annealing if the given loss is compatible with it."""
        if hasattr(self.loss, "kl_scale_final"):
            fin_scale = float(self.loss.kl_scale_final)
        elif kl_warmup_frac > 0:
            log.warn(
                f"Given kl_warmup_frac > 0 but given loss {type(self.loss).__name__} "
                "does not have attribute 'kl_scale_final'. "
                "Annealing of the kl_scale is disabled."
            )
            fin_scale = 1.0
        else:
            log.warn(f"kl_warmup_frac={kl_warmup_frac} <= 0. Annealing disabled.")

        self.kl_scale = LinearWarmup(
            final_value=fin_scale, warmup_frac=kl_warmup_frac, total_steps=total_steps
        )
