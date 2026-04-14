# Variational auto-encoder model implementation.
from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.vae import ClassicVAELoss
from src.algorithms.schedulers.linear import LinearWarmup
from src.algorithms.utils.weight_loader import load_weights
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.algorithms.components.encoder import VariationalEncoder
from src.algorithms.components.decoder import Decoder
from src.data.utils import unpack_batch
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class VAE(ADLightningModule):
    """Variational autoencoder module.

    :param encoder: The encoder nn module.
    :param decoder: The decoder nn module.
    :param kl_warmup_frac: Fraction of total steps used to warm up the KL scale.
    :param features: Optional nn module applied to the flattened input.
    :param ckpt: Optional checkpoint path to resume weights from.
    :param target_rate: Target background rate or FPR.
    :param base_rate: Base rate used to convert target_rate into an FPR.
        If None, target_rate is interpreted directly as an FPR.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        kl_warmup_frac: float = 0.0,
        kl_scale: float = 1.0,
        features: Optional[nn.Module] = None,
        ckpt: str = "",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder"]
        )

        self.ckpt_path = ckpt
        self.encoder = encoder
        self.decoder = decoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()
        self.kl_warmup_frac = kl_warmup_frac

        self.loss = ClassicVAELoss(kl_scale=kl_scale, reduction="none")

    def on_fit_start(self):
        inject_object_feature_map(self)
        self.features.to(self.device)

        total_steps = int(self.trainer.estimated_stepping_batches)
        self._setup_kl_annealing(self.kl_warmup_frac, total_steps)

        if self.ckpt_path:
            self._load_checkpoint()

    def on_test_start(self):
        inject_object_feature_map(self)

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.features(x)
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        b = unpack_batch(batch)

        x = torch.flatten(b.x, start_dim=1)
        m = b.mask
        if m is not None:
            m = torch.flatten(m, start_dim=1).float()

        z_mean, z_log_var, z, reconstruction = self.forward(x)

        kl_current_scale = self.kl_scale(int(self.global_step))
        reco_loss, kl_raw, kl_scaled, loss = self.loss(
            reconstruction=reconstruction,
            z_mean=z_mean,
            z_log_var=z_log_var,
            target=x,
            mask=m,
            kl_scale=kl_current_scale,
        )

        # The operational anomaly score is hard-configured to the raw KL.
        ascore = kl_raw
        if ascore.ndim != 1:
            raise ValueError(
                f"Expected per-event ascores, got {tuple(ascore.shape)}."
            )

        loss = self._add_hgq_loss(loss)

        with torch.no_grad():
            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))

            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(
                    ascore, 1.0 - self.target_fpr
                ).item()

            q50, q99 = torch.quantile(
                ascore, torch.tensor([0.5, 0.99], device=ascore.device)
            ).tolist()

            z_mean_squared = torch.square(z_mean).sum(dim=1)
            z_mean_squared_mean = z_mean_squared.mean().item()

        del x, z, z_log_var

        loss_mean = loss.mean()
        reco_loss_mean = reco_loss.mean()
        kl_scaled_mean = kl_scaled.mean()
        kl_raw_mean = kl_raw.mean()

        return {
            # Used for backpropagation:
            "loss": loss_mean,
            # Used for logging:
            "loss/mean": loss_mean,
            "loss/reco/mean": reco_loss_mean,
            "loss/kl_scaled/mean": kl_scaled_mean,
            "loss/kl_raw/mean": kl_raw_mean,
            "ascore/operational": operational_ascore,
            "ascore/q50": q50,
            "ascore/q99": q99,
            "z_mean_squared": z_mean_squared_mean,
            "kl_scale": kl_current_scale,
            # Used for callbacks:
            "loss/full": loss.detach(),
            "loss/reco/full": reco_loss.detach(),
            "loss/kl_raw/full": kl_raw.detach(),
            "ascore/full": ascore.detach(),
            "z_mean_squared/full": z_mean_squared.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "loss_reco": outdict.get("loss/reco/mean"),
            "loss_kl_scaled": outdict.get("loss/kl_scaled/mean"),
            "loss_kl_raw": outdict.get("loss/kl_raw/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
            "ascore_q50": outdict.get("ascore/q50"),
            "ascore_q99": outdict.get("ascore/q99"),
            "z_mean_squared": outdict.get("z_mean_squared"),
            "kl_scale": outdict.get("kl_scale"),
        }

    def _setup_kl_annealing(self, kl_warmup_frac: float, total_steps: int):
        """Set up KL annealing if supported by the loss."""
        if hasattr(self.loss, "kl_scale_final"):
            fin_scale = float(self.loss.kl_scale_final)
        elif kl_warmup_frac > 0:
            log.warn(
                f"Given kl_warmup_frac > 0 but loss {type(self.loss).__name__} "
                "does not have attribute 'kl_scale_final'. "
                "Annealing of the KL scale is disabled."
            )
            fin_scale = 1.0
        else:
            fin_scale = 1.0

        self.kl_scale = LinearWarmup(
            final_value=fin_scale,
            warmup_frac=kl_warmup_frac,
            total_steps=total_steps,
        )

    def _load_checkpoint(self):
        """Load checkpoint weights to continue the training from, if provided."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]

        is_lightning_encoder = isinstance(self.encoder, VariationalEncoder)
        is_lightning_decoder = isinstance(self.decoder, Decoder)
        if is_lightning_encoder and is_lightning_decoder:
            self.load_state_dict(state_dict, strict=True)
            return

        enc_mlp = self.encoder.get_layer("enc_mlp")
        dec_mlp = self.decoder.get_layer("dec_mlp")
        load_weights(enc_mlp, state_dict, "encoder", False)
        load_weights(dec_mlp, state_dict, "decoder", False)

    def _add_hgq_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Add additional HGQ losses if they exist."""
        add_loss = 0.0

        if hasattr(self.encoder, "losses") and len(self.encoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.encoder.losses]).sum()

        if hasattr(self.decoder, "losses") and len(self.decoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.decoder.losses]).sum()

        return loss + add_loss

