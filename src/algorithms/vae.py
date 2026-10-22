# Variational auto-encoder model implementation.
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


def _unfitted_residual_state(dim: int) -> dict[str, torch.Tensor]:
    return {
        "residual_oas_location": torch.zeros(dim),
        "residual_oas_precision": torch.eye(dim),
        "residual_oas_ready": torch.tensor(False),
    }


def _whitened_energy(
    residual: torch.Tensor,
    location: torch.Tensor,
    precision: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mahalanobis energy of the centred residual per observed feature.

    The residual and mask have shape (batch, features) and the result (batch,).
    Unobserved features are zeroed before the quadratic form.
    """
    mask = torch.ones_like(residual) if mask is None else mask.to(residual.dtype)
    centred = (residual - location) * mask
    n_observed = mask.sum(dim=1).clamp_min(1.0)
    return torch.einsum("bi,ij,bj->b", centred, precision, centred) / n_observed


class VAE(ADLightningModule):
    """Variational autoencoder module.

    :param encoder: The encoder nn module.
    :param decoder: The decoder nn module.
    :param kl_warmup_frac: Fraction of total steps used to warm up the KL scale.
    :param kl_scale: Float pertaining to the scaling factor before the KL divergence.
    :param features: Optional nn module applied to the flattened input.
    :param ckpt: Optional checkpoint path to resume weights from.
    :param anomaly_score: Score emitted as 'ascore/full' in validation and test:
        'kl', the raw KL divergence, or 'residual_oas', the whitened residual of the
        posterior-mean reconstruction, which needs the state set by
        src.callbacks.residual_state.ResidualStateCallback.
    :param target_rate: Target background rate or FPR.
    :param base_rate: Base rate used to convert target_rate into an FPR.
        If None, target_rate is interpreted directly as an FPR.
    :param eval_mean: Whether the decoder reads the posterior mean instead of a sample
        in eval mode, which makes the reconstruction and the loss deterministic there.
        The sample is drawn either way, so the random stream is the same.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        kl_warmup_frac: float = 0.0,
        kl_scale: float = 1.0,
        features: Optional[nn.Module] = None,
        ckpt: str = "",
        anomaly_score: str = "kl",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        eval_mean: bool = False,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "encoder", "decoder"])

        self.ckpt_path = ckpt
        self.encoder = encoder
        self.decoder = decoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()
        self.kl_warmup_frac = kl_warmup_frac
        self.eval_mean = eval_mean

        self.loss = ClassicVAELoss(kl_scale=kl_scale, reduction="none")

        if anomaly_score not in ("kl", "residual_oas"):
            raise ValueError(f"Unknown VAE anomaly_score {anomaly_score!r}.")
        self.anomaly_score = anomaly_score
        # Buffers, so that every checkpoint carries the state fitted at its epoch.
        for name, value in _unfitted_residual_state(decoder.out_dim).items():
            self.register_buffer(name, value)

    def on_fit_start(self):
        inject_object_feature_map(self)
        self.features.to(self.device)

        total_steps = int(self.trainer.estimated_stepping_batches)
        self._setup_kl_annealing(self.kl_warmup_frac, total_steps)

        if self.ckpt_path:
            self._load_checkpoint()

    def on_test_start(self):
        inject_object_feature_map(self)
        if not hasattr(self, "kl_scale"):
            # Eval-only runs (train=false) skip on_fit_start; at step 0 the
            # warmup value does not depend on total_steps.
            self._setup_kl_annealing(self.kl_warmup_frac, 1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Checkpoints written before the residual state existed load as unfitted.
        if "residual_oas_ready" not in state_dict:
            dim = self.residual_oas_location.numel()
            state_dict = {**state_dict, **_unfitted_residual_state(dim)}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.features(x)
        z_mean, z_log_var, z = self.encoder(x)
        decoded = z_mean if self.eval_mean and not self.training else z
        reconstruction = self.decoder(decoded)
        return z_mean, z_log_var, z, reconstruction

    def residuals(self, x: torch.Tensor) -> torch.Tensor:
        """Residual x - x̂ of a flattened input, x̂ decoded from the posterior mean."""
        z_mean, _, _ = self.encoder(self.features(x))
        return x - self.decoder(z_mean)

    def set_residual_state(self, location: torch.Tensor, precision: torch.Tensor):
        self.residual_oas_location.copy_(location)
        self.residual_oas_precision.copy_(precision)
        self.residual_oas_ready.fill_(True)

    def residual_oas_score(
        self, residual: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not bool(self.residual_oas_ready):
            raise RuntimeError(
                "The residual_oas score needs a fitted residual state: add the "
                "residual_state callback, or load a checkpoint written with it."
            )
        location, precision = self.residual_oas_location, self.residual_oas_precision
        return _whitened_energy(residual, location, precision, mask)

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

        scores = self._anomaly_scores(x, z_mean, kl_raw, m)
        ascore = scores["kl"] if self.training else scores[self.anomaly_score]
        if ascore.ndim != 1:
            raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

        loss = self._add_hgq_loss(loss)

        with torch.no_grad():
            operational_ascore = self.compute_operational_ascore(ascore)

            q50, q99 = self.compute_score_quantiles(ascore)

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
            **{f"ascore/{name}": score.detach() for name, score in scores.items()},
            "z_mean_squared/full": z_mean_squared.detach(),
            "reconstructed_data": reconstruction.detach(),
        }

    def _anomaly_scores(
        self,
        x: torch.Tensor,
        z_mean: torch.Tensor,
        kl_raw: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Per-event scores by name, the residual one outside training only.

        The residual decodes the posterior mean, so it carries no sampling noise. The
        residual state is refitted at the start of every validation epoch, so during
        training it is stale, or unfitted in the first epoch.
        """
        scores = {"kl": kl_raw}
        fitted = bool(self.residual_oas_ready)
        if not self.training and (fitted or self.anomaly_score == "residual_oas"):
            residual = x - self.decoder(z_mean)
            scores["residual_oas"] = self.residual_oas_score(residual, mask)
        return scores

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
            log.warning(
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
            add_loss = add_loss + torch.stack(list(self.encoder.losses)).sum()

        if hasattr(self.decoder, "losses") and len(self.decoder.losses) > 0:
            add_loss = add_loss + torch.stack(list(self.decoder.losses)).sum()

        return loss + add_loss
