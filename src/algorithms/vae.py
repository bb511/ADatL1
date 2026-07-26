# Variational auto-encoder model implementation.
from contextlib import nullcontext
from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.components.decoder import Decoder
from src.algorithms.components.encoder import VariationalEncoder
from src.algorithms.losses.vae import ClassicVAELoss
from src.algorithms.schedulers.linear import LinearWarmup
from src.algorithms.utils.object_feature_map_loader import inject_object_feature_map
from src.algorithms.utils.weight_loader import load_weights
from src.data.utils import unpack_batch
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class VAE(ADLightningModule):
    """Variational autoencoder module.

    :param encoder: The encoder nn module.
    :param decoder: The decoder nn module.
    :param kl_warmup_frac: Fraction of total steps used to warm up the KL scale.
    :param kl_scale: Float pertaining to the scaling factor before the KL divergence.
    :param features: Optional nn module applied to the flattened input.
    :param ckpt: Optional checkpoint path to resume weights from.
    :param target_rate: Target background rate or FPR.
    :param base_rate: Base rate used to convert target_rate into an FPR. If None, target_rate is
        interpreted directly as an FPR.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        kl_warmup_frac: float = 0.0,
        kl_scale: float = 1.0,
        features: Optional[nn.Module] = None,
        loss: nn.Module | None = None,
        mask: bool = True,
        masking: nn.Module | None = None,
        ckpt: str = "",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, masking=masking, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "encoder", "decoder", "loss", "masking"]
        )

        self.ckpt_path = ckpt
        self.encoder = encoder
        self.decoder = decoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()
        self.use_mask = bool(mask)
        self.kl_warmup_frac = kl_warmup_frac

        self.loss = (
            loss if loss is not None else ClassicVAELoss(kl_scale=kl_scale, reduction="none")
        )
        # Validation/test-only sidecars do not call ``on_fit_start``.  Initialize
        # inference at the final KL scale; fitting replaces this with the
        # configured step-aware warmup below.
        self._setup_kl_annealing(0.0, total_steps=1)
        self._maybe_build_keras_modules()

    def on_fit_start(self):
        """Initialize feature injection and step-aware KL annealing for fitting."""
        inject_object_feature_map(self)
        self.features.to(self.device)

        total_steps = int(self.trainer.estimated_stepping_batches)
        self._setup_kl_annealing(self.kl_warmup_frac, total_steps)

        if self.ckpt_path:
            self._load_checkpoint()

    def on_test_start(self):
        """Inject the configured object-feature mapping before test inference."""
        inject_object_feature_map(self)

    @property
    def target_fpr(self) -> float:
        """Return the configured target false-positive probability."""
        return self.compute_target_fpr()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Encode, sample, and reconstruct one input batch."""
        x = self.features(x)
        with self._keras_device_scope(x.device):
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute VAE losses, anomaly scores, and diagnostic quantities."""
        b = unpack_batch(batch)

        x = torch.flatten(b.x, start_dim=1)
        m = b.mask if self.use_mask else None
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
            raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

        loss = self._add_hgq_loss(loss)

        with torch.no_grad():
            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))

            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(ascore, 1.0 - self.target_fpr).item()

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
        """Select scalar training and validation quantities for logging."""
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
        add_loss = loss.new_tensor(0.0)

        if hasattr(self.encoder, "losses") and len(self.encoder.losses) > 0:
            add_loss = (
                add_loss
                + torch.stack(
                    [self._to_loss_device(value, loss) for value in self.encoder.losses]
                ).sum()
            )

        if hasattr(self.decoder, "losses") and len(self.decoder.losses) > 0:
            add_loss = (
                add_loss
                + torch.stack(
                    [self._to_loss_device(value, loss) for value in self.decoder.losses]
                ).sum()
            )

        return loss + add_loss

    @staticmethod
    def _to_loss_device(value, reference: torch.Tensor) -> torch.Tensor:
        """Move or construct one auxiliary loss on the reference tensor device."""
        if torch.is_tensor(value):
            return value.to(device=reference.device, dtype=reference.dtype)
        return reference.new_tensor(value)

    def _maybe_build_keras_modules(self) -> None:
        """Build Keras/HGQ modules early so Lightning sees their torch parameters."""
        if not hasattr(self.encoder, "trainable_variables"):
            return
        if len(getattr(self.encoder, "trainable_variables", [])) > 0:
            return

        nodes = getattr(self.encoder, "nodes", None)
        if not nodes:
            return

        try:
            from keras.src.backend.torch.core import device_scope

            device_ctx = device_scope("cpu")
        except Exception:
            device_ctx = nullcontext()

        with torch.no_grad(), device_ctx:
            x = torch.zeros(1, int(nodes[0]), dtype=torch.float32)
            enc_out = self.encoder(x)
            z = enc_out[-1] if isinstance(enc_out, (tuple, list)) else enc_out
            self.decoder(z)

    def _keras_device_scope(self, device: torch.device):
        """Return a Keras torch-backend device scope when available."""
        if not hasattr(self.encoder, "trainable_variables"):
            return nullcontext()
        try:
            from keras.src.backend.torch.core import device_scope

            return device_scope(str(device))
        except Exception:
            return nullcontext()


class RVAE(VAE):
    """Regularized VAE backed by a configurable ``MultiLoss``.

    This restores the experimental ``rvae`` configuration from ``dev/victor`` while
    preserving the current VAE batch contract and logging surface.
    """

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)

        m = b.mask if self.use_mask else None
        if m is not None:
            m = torch.flatten(m, start_dim=1).float()

        z_mean, z_log_var, z, reconstruction = self.forward(x)
        loss_out = self.loss(
            target=x,
            reconstruction=reconstruction,
            reco=reconstruction,
            mask=m,
            z_mean=z_mean,
            z_log_var=z_log_var,
            z=z,
            y=b.y,
        )

        if not isinstance(loss_out, dict):
            loss_full = loss_out
            component_logs = {}
        else:
            loss_full = loss_out["loss/total"]
            component_logs = loss_out

        if loss_full.ndim == 0:
            loss_full = loss_full.expand(x.shape[0])

        ascore = component_logs.get("loss/kl_raw", loss_full).detach()
        if ascore.ndim == 0:
            ascore = ascore.expand(x.shape[0])
        if ascore.ndim != 1:
            ascore = ascore.view(ascore.shape[0], -1).mean(dim=1)

        loss_full = self._add_hgq_loss(loss_full)

        with torch.no_grad():
            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))
            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(ascore, 1.0 - self.target_fpr).item()

            q50, q99 = torch.quantile(
                ascore, torch.tensor([0.5, 0.99], device=ascore.device)
            ).tolist()
            z_mean_squared = torch.square(z_mean).sum(dim=1)

        loss_mean = loss_full.mean()
        out = {
            "loss": loss_mean,
            "loss/mean": loss_mean,
            "ascore/operational": operational_ascore,
            "ascore/q50": q50,
            "ascore/q99": q99,
            "z_mean_squared": z_mean_squared.mean().item(),
            "loss/full": loss_full.detach(),
            "ascore/full": ascore.detach(),
            "z_mean_squared/full": z_mean_squared.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

        for key, value in component_logs.items():
            if key == "loss/total":
                continue
            if torch.is_tensor(value):
                out[f"{key}/mean"] = value.mean()
                out[f"{key}/full"] = value.detach()

        return out
