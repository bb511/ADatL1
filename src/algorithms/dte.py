"""Categorical Diffusion Time Estimation for anomaly detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms import ADLightningModule
from src.data.utils import unpack_batch


class DTE(ADLightningModule):
    """Categorical Diffusion Time Estimation (DTE-C).

    Training adds a sampled amount of Gaussian noise to nominal observations and
    predicts its ordered time bin. Evaluation is clean and deterministic: the
    anomaly score is the expected predicted bin, normalised onto ``[0, 1]`` so that
    it means the same thing for every ``n_bins``.

    The implementation follows the additive corruption used by the DTE training
    algorithm, ``x_t = x_0 + sqrt(1 - alpha_bar_t) * epsilon``. It deliberately
    does not apply the signal-scaling term from conventional forward diffusion.

    Three further deviations from the reference implementation (vicliv/DTE) are
    deliberate and should not be "fixed" back:

    - ``beta_start`` defaults to 0 rather than 1e-4, so ``noise_scales[0]`` is
      exactly 0. The uncorrupted event is then something the predictor sees during
      training, which is what makes scoring a clean event well posed.
    - The predictor emits raw logits, and the softmax is applied exactly once: by
      ``F.cross_entropy`` when computing the loss, by :meth:`anomaly_score` when
      scoring. The reference softmaxes inside the model and then hands the result
      to ``nn.CrossEntropyLoss``, applying it twice and squashing the gradients.
    - Noise is masked on padded coordinates, so a padded slot is never a free
      readout of the noise level. See :meth:`corrupt`.
    """

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
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2.")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if n_bins > n_steps:
            raise ValueError("n_bins cannot exceed n_steps.")
        if not 0.0 <= beta_start < 1.0:
            raise ValueError("beta_start must lie in [0, 1).")
        if not 0.0 < beta_end < 1.0:
            raise ValueError("beta_end must lie in (0, 1).")
        if beta_start > beta_end:
            raise ValueError("beta_start cannot exceed beta_end.")

        predictor_out = getattr(predictor, "out_dim", None)
        if predictor_out is not None and int(predictor_out) != int(n_bins):
            raise ValueError(f"predictor out_dim={predictor_out} does not match n_bins={n_bins}.")

        self.save_hyperparameters(ignore=["model", "predictor", "features", "loss"])
        self.predictor = predictor
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

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

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
            timesteps.min().item() < 0 or timesteps.max().item() >= int(self.hparams.n_steps)
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
        if x.ndim != 2:
            raise ValueError(f"DTE corruption expects [batch, features], got {tuple(x.shape)}.")
        if timesteps.shape != (x.shape[0],):
            raise ValueError(f"Expected one timestep per event, got {tuple(timesteps.shape)}.")
        self.time_to_bin(timesteps)

        if noise is None:
            noise = torch.randn_like(x)
        elif noise.shape != x.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match x.")

        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match x.")
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
                raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))
            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(ascore, 1.0 - self.target_fpr).item()
            q50, q99 = torch.quantile(
                ascore,
                torch.tensor([0.5, 0.99], device=ascore.device),
            ).tolist()
            mult_corr = self._multiplicity_correlation(ascore, mask)
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

    @staticmethod
    def _multiplicity_correlation(
        ascore: torch.Tensor, mask: torch.Tensor | None
    ) -> float | None:
        """Correlate the score with how many features an event actually populates.

        A large magnitude means the score tracks occupancy rather than anomalousness,
        which is the failure mode masked corruption can produce. Returns None when the
        data carries no mask, so the key is simply absent from the logs.
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

    def _validate_logits(self, logits: torch.Tensor, batch_size: int) -> None:
        expected = (int(batch_size), int(self.hparams.n_bins))
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"DTE predictor must return logits with shape {expected}, "
                f"got {tuple(logits.shape)}."
            )
