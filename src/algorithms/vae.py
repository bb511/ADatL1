# Variational auto-encoder model implementation.
from typing import Optional, Tuple, Dict

import torch
from torch import nn

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
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        kl_warmup_frac: float = 0.0,
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
            target=x,
            mask=m,
            kl_scale=kl_current_scale
        )
        del x, z, z_mean, z_log_var

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


class RVAE(VAE):
    """Regularized VAE. Overriding loss and logging."""

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, y = batch
        x = torch.flatten(x, start_dim=1)
        z_mean, z_log_var, z, reconstruction = self.forward(x)
        loss = self.loss(
            reconstruction=reconstruction,
            target=x,
            z=z,
            z_mean=z_mean,
            z_log_var=z_log_var,
            y=y,
        )
        del reconstruction, x, z, y

        outdict = {
            "loss": loss.mean(),
            "loss/total/full": loss.detach(),
            "z_mean.abs()": z_mean.abs(),
            "z_log_var": z_log_var,
        }
        if hasattr(self.loss, "losses"):
            individual_losses = {
                f"loss/{module.name}": value
                for module, value in zip(
                    self.loss.losses.values(), self.loss.values.values()
                )
            }
            outdict.update(
                {
                    "loss/total": sum(list(individual_losses.values())),
                    **{k: v.mean() for k, v in individual_losses.items()},
                }
            )
        return outdict

    def outlog(self, outdict: dict) -> dict:
        return {
            k: v for k, v in outdict.items() if "train/" in k and k != "loss/total/full"
        }
