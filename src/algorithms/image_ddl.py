"""The diffusion denoising loss (DDL) on images, scored by its own loss.

A noise predictor ε_θ(x_t, t) is trained on the simplified loss of Ho et al. (NeurIPS
2020, arXiv:2006.11239, eq. 14). Each image is scored by that loss averaged over a fixed
set of timesteps under one fixed noise draw per timestep and pixel, as TabADM scores
tabular rows with a sum over every timestep (Zamberg et al., arXiv:2307.12336, Alg. 2).
DDL in ddl.py is the same model on flattened inputs: keep the two files in step.
"""

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.components.ddl import alpha_bars, fixed_noise, score_timesteps
from src.data.utils import unpack_batch


class ImageDDL(ADLightningModule):
    """Diffusion denoising loss on images.

    In training, each image takes a timestep drawn uniformly from 0, ..., n_steps - 1
    and fresh noise. In eval mode, the per-image loss is averaged over the
    ``n_score_steps`` timesteps of ``score_timesteps`` under ``fixed_noise``, so that
    an image's score depends on its pixels and the weights alone; this score is the
    anomaly score, the per-image loss and the fit.

    Args:
        denoiser: ε_θ, called with x_t of shape (batch, channels, height, width) and
            integer timesteps of shape (batch,).
        input_shape: (channels, height, width), the shape of the fixed noise.
        n_steps: T, the number of diffusion steps.
        beta_end: The end of the noise schedule, as ``alpha_bars`` scales it.
        n_score_steps: The number of timesteps the score averages over.
        noise_seed: The seed of the fixed noise.
        target_rate: The FPR when ``base_rate`` is None, else a rate that ``base_rate``
            divides.
        base_rate: See ``target_rate``.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        input_shape: list[int] | tuple[int, ...] = (3, 32, 32),
        n_steps: int = 100,
        beta_end: float = 0.02,
        n_score_steps: int = 10,
        noise_seed: int = 0,
        target_rate: float = 0.01,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "denoiser"])
        self.denoiser = denoiser
        self._register_schedule(tuple(input_shape))

    def _register_schedule(self, input_shape: tuple[int, ...]) -> None:
        # Not persistent: a replay rebuilds them from the run's configuration.
        hp = self.hparams
        alpha_bar = alpha_bars(hp.n_steps, hp.beta_end)
        steps = score_timesteps(hp.n_steps, hp.n_score_steps)
        noise = fixed_noise(hp.n_score_steps, input_shape, hp.noise_seed)
        buffers = {
            "sqrt_alpha_bar": alpha_bar.sqrt().float(),
            "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt().float(),
            "score_steps": steps,
            "score_noise": noise,
        }
        for name, value in buffers.items():
            self.register_buffer(name, value, persistent=False)

    def denoising_loss(
        self, x: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Per-image |ε - ε_θ(x_t, t)|² averaged over the pixels, shape (batch,), for
        x and noise of shape (batch, channels, height, width)."""
        signal = self.sqrt_alpha_bar[timesteps].view(-1, 1, 1, 1)
        spread = self.sqrt_one_minus_alpha_bar[timesteps].view(-1, 1, 1, 1)
        x_t = signal * x + spread * noise
        residual = (noise - self.denoiser(x_t, timesteps)) ** 2
        return residual.flatten(start_dim=1).mean(dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """The deterministic per-image score, shape (batch,).

        One denoiser call per timestep keeps the memory of a batch; the timesteps are
        summed in a fixed order.
        """
        total = torch.zeros(len(x), device=x.device)
        for step, noise in zip(self.score_steps.tolist(), self.score_noise):
            timesteps = torch.full((len(x),), step, device=x.device)
            total = total + self.denoising_loss(x, timesteps, noise.expand_as(x))
        return total / len(self.score_noise)

    def _random_loss(self, x: torch.Tensor) -> torch.Tensor:
        n_steps = self.hparams.n_steps
        timesteps = torch.randint(0, n_steps, (len(x),), device=x.device)
        return self.denoising_loss(x, timesteps, torch.randn_like(x))

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        """The loss at a random timestep in training, the score in eval mode."""
        x = unpack_batch(batch).x
        loss_full = self._random_loss(x) if self.training else self.score(x)
        ascore = loss_full.detach()
        q50, q99 = self.compute_score_quantiles(ascore)
        return {
            "loss": loss_full.mean(),
            "loss/mean": ascore.mean(),
            "ascore/operational": self.compute_operational_ascore(ascore),
            "ascore/q50": q50,
            "ascore/q99": q99,
            "loss/full": ascore,
            "ascore/full": ascore,
            "fit/full": ascore,
        }

    def outlog(self, outdict: dict) -> dict:
        keys = ("loss", "loss/mean", "ascore/operational", "ascore/q50", "ascore/q99")
        return {key.replace("/", "_"): outdict.get(key) for key in keys}
