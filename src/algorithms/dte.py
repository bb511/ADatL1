"""Categorical Diffusion Time Estimation for anomaly detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms import ADLightningModule
from src.data.utils import unpack_batch

# (failed, message) pairs over the saved hyperparameters, checked in this order.
_HPARAM_CHECKS = (
    (lambda hp: hp.n_steps < 2, "n_steps must be at least 2."),
    (lambda hp: hp.n_bins < 2, "n_bins must be at least 2."),
    (lambda hp: hp.n_bins > hp.n_steps, "n_bins cannot exceed n_steps."),
    (lambda hp: not 0.0 <= hp.beta_start < 1.0, "beta_start must lie in [0, 1)."),
    (lambda hp: not 0.0 < hp.beta_end < 1.0, "beta_end must lie in (0, 1)."),
    (lambda hp: hp.beta_start > hp.beta_end, "beta_start cannot exceed beta_end."),
)


def _multiplicity_correlation(
    ascore: torch.Tensor, mask: torch.Tensor | None
) -> float | None:
    """Correlate the score with how many features an event actually populates.

    A large magnitude means the score tracks occupancy rather than anomalousness, which
    is the failure mode masked corruption can produce. Returns None when the data
    carries no mask, so the key is simply absent from the logs.
    """
    if mask is None:
        return None

    n_active = torch.flatten(mask, start_dim=1).sum(dim=1).to(ascore.dtype)
    centred_score = ascore - ascore.mean()
    centred_count = n_active - n_active.mean()
    denominator = centred_score.norm() * centred_count.norm()
    if denominator <= 0:
        return 0.0

    return float(torch.dot(centred_score, centred_count) / denominator)


class DTE(ADLightningModule):
    """Categorical Diffusion Time Estimation (DTE-C)."""

    def __init__(
        self,
        predictor: nn.Module,
        n_steps: int = 300,
        n_bins: int = 7,
        beta_start: float = 0.0,
        beta_end: float = 0.01,
        target_rate: float = 0.25,
        base_rate: float | None = None,
        features: nn.Module | None = None,
        encoder: None = None,
        **kwargs,
    ) -> None:
        if encoder is not None:
            raise ValueError(
                "DTE does not accept an encoder; configure algorithm.predictor instead."
            )
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "predictor", "features", "loss"])
        self.predictor = predictor
        self._check_hparams()
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        betas = torch.linspace(float(beta_start), float(beta_end), int(n_steps))
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        noise_scales = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("noise_scales", noise_scales)
        # Normalised bin ladder: the score is the expected position along it, so it
        # lands in [0, 1] whatever n_bins is. Without this the score would scale with
        # n_bins, and the searched objective that minimises it would simply prefer the
        # smallest n_bins on offer.
        self.register_buffer(
            "bin_values",
            torch.arange(int(n_bins), dtype=torch.float32) / max(int(n_bins) - 1, 1),
        )

    def on_fit_start(self):
        """Move the optional frozen feature extractor to the training device."""
        self.features.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.predictor(x)
        self._validate_logits(logits, x.shape[0])
        return logits

    def time_to_bin(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map integer diffusion times to approximately equal-width ordered bins."""
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be a one-dimensional tensor.")
        if timesteps.numel() and (
            timesteps.min().item() < 0
            or timesteps.max().item() >= int(self.hparams.n_steps)
        ):
            raise ValueError("timesteps must lie in [0, n_steps).")
        bins = torch.div(
            timesteps.long() * int(self.hparams.n_bins),
            int(self.hparams.n_steps),
            rounding_mode="floor",
        )
        return bins.clamp_max(int(self.hparams.n_bins) - 1)

    def corrupt(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the paper's additive diffusion corruption.

        Padded coordinates are preserved exactly. The mask is an explicit data contract; zero-
        valued active features are never mistaken for padding.

        Masking the noise costs something: a padded slot is constant, so noising it would let
        the predictor read the diffusion time straight off it. Not noising it means an event
        with few active features carries less evidence about the time, which pulls its
        posterior toward the prior and so its score upward. ``ascore/mult_corr`` measures how
        far that goes.
        """
        self._validate_corruption_inputs(x, timesteps, mask, noise)
        if noise is None:
            noise = torch.randn_like(x)
        if mask is not None:
            noise = noise * mask.to(device=x.device, dtype=x.dtype)

        scale = self.noise_scales[timesteps.long()].to(dtype=x.dtype).unsqueeze(1)
        return x + scale * noise

    def anomaly_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the expected diffusion-time bin, normalised onto ``[0, 1]``."""
        self._validate_logits(logits, logits.shape[0])
        probabilities = torch.softmax(logits, dim=1)
        return probabilities @ self.bin_values.to(dtype=probabilities.dtype)

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        mask = None
        if b.mask is not None:
            mask = torch.flatten(b.mask, start_dim=1).bool()
            if mask.shape != x.shape:
                raise ValueError(
                    f"Flattened DTE mask {tuple(mask.shape)} does not match "
                    f"input {tuple(x.shape)}."
                )

        timesteps = torch.randint(
            0,
            int(self.hparams.n_steps),
            (x.shape[0],),
            device=x.device,
        )
        noisy_x = self.corrupt(x, timesteps, mask=mask)
        noisy_logits = self.forward(noisy_x)
        targets = self.time_to_bin(timesteps)
        loss_full = F.cross_entropy(noisy_logits, targets, reduction="none")

        # The score is read off a clean pass, so it must not carry the corruption drawn
        # for the loss, nor the dropout that is active while training.
        was_training = self.predictor.training
        self.predictor.eval()
        with torch.no_grad():
            ascore = self.anomaly_score(self.forward(x))
            if ascore.ndim != 1:
                raise ValueError(
                    f"Expected per-event ascores, got {tuple(ascore.shape)}."
                )

            operational_ascore = self.compute_operational_ascore(ascore)
            q50, q99 = self.compute_score_quantiles(ascore)
            mult_corr = _multiplicity_correlation(ascore, mask)
        if was_training:
            self.predictor.train()

        loss = loss_full.mean()
        outdict = {
            # Used for backpropagation:
            "loss": loss,
            # Used for logging:
            "loss/mean": loss.detach(),
            "ascore/operational": operational_ascore,
            "ascore/q50": q50,
            "ascore/q99": q99,
            # Used for callbacks:
            "loss/full": loss_full.detach(),
            "ascore/full": ascore,
        }
        if mult_corr is not None:
            outdict["ascore/mult_corr"] = mult_corr

        return outdict

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
            "ascore_q50": outdict.get("ascore/q50"),
            "ascore_q99": outdict.get("ascore/q99"),
            "ascore_mult_corr": outdict.get("ascore/mult_corr"),
        }

    def _check_hparams(self) -> None:
        hp = self.hparams
        for failed, message in _HPARAM_CHECKS:
            if failed(hp):
                raise ValueError(message)
        predictor_out = getattr(self.predictor, "out_dim", None)
        if predictor_out is not None and int(predictor_out) != int(hp.n_bins):
            raise ValueError(
                f"predictor out_dim={predictor_out} does not match n_bins={hp.n_bins}."
            )

    def _validate_corruption_inputs(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor | None,
        noise: torch.Tensor | None,
    ) -> None:
        shape, steps_shape = tuple(x.shape), tuple(timesteps.shape)
        if x.ndim != 2:
            raise ValueError(f"DTE corruption expects [batch, features], got {shape}.")
        if steps_shape != (x.shape[0],):
            raise ValueError(f"Expected one timestep per event, got {steps_shape}.")
        self.time_to_bin(timesteps)
        for name, arg in (("noise", noise), ("mask", mask)):
            if arg is not None and arg.shape != x.shape:
                raise ValueError(f"{name} shape {tuple(arg.shape)} does not match x.")

    def _validate_logits(self, logits: torch.Tensor, batch_size: int) -> None:
        expected = (int(batch_size), int(self.hparams.n_bins))
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"DTE predictor must return logits with shape {expected}, "
                f"got {tuple(logits.shape)}."
            )
