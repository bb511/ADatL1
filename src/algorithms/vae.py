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
        x, _ = batch
        x = torch.flatten(x, start_dim=1)
        z_mean, z_log_var, z, reconstruction = self.forward(x)

        kl_current_scale = self.kl_scale(int(self.global_step))
        reco_loss, kl_loss, total_loss = self.loss(
            reconstruction=reconstruction,
            z_mean=z_mean,
            z_log_var=z_log_var,
            target=x,
            kl_scale=kl_current_scale
        )
        del reconstruction, x, z

        return {
            # Used for backpropagation:
            "loss": total_loss.mean(),
            # Used for logging:
            "loss/reco/mean": reco_loss.mean(),
            "loss/kl/mean": kl_loss.mean(),
            "kl_scale": kl_current_scale,
            # Used for callbacks:
            "loss/total/full": total_loss.detach(),
            "loss/reco/full": reco_loss.detach(),
            "loss/kl/full": kl_loss.detach(),
            "z_mean_squared": z_mean.pow(2),
        }

    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss": outdict.get("loss"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_kl": outdict.get("loss/kl/mean"),
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
