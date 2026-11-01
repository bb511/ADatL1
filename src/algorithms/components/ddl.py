"""Parts of the diffusion denoising loss (DDL): the noise schedule, the timesteps and
the fixed noise of the score, the timestep embedding and the noise predictors.

Ho, Jain and Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020,
arXiv:2006.11239; Zamberg et al., "TabADM: Unsupervised Tabular Anomaly Detection with
Diffusion Models", arXiv:2307.12336.
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Ho et al. (Sec. 4) raise beta linearly from 1e-4 to 0.02 over T = 1000 steps.
REFERENCE_STEPS = 1000
BETA_START = 1.0e-4
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


def alpha_bars(n_steps: int, beta_end: float) -> torch.Tensor:
    """The products ᾱ_t of 1 - β_s over s ≤ t, for t = 0, ..., n_steps - 1, in float64.

    β rises linearly from BETA_START to beta_end, both scaled by REFERENCE_STEPS /
    n_steps, so that the sum of β does not depend on n_steps and the noise reached at
    the last step, exp(-Σβ) to first order in β, mostly on beta_end.

    Raises:
        ValueError: If a β reaches 1.
    """
    scale = REFERENCE_STEPS / n_steps
    betas = torch.linspace(
        BETA_START * scale, beta_end * scale, n_steps, dtype=torch.float64
    )
    if betas.max() >= 1.0:
        raise ValueError(f"beta_end {beta_end} gives a beta of 1 at {n_steps} steps.")
    return torch.cumprod(1.0 - betas, dim=0)


def score_timesteps(n_steps: int, n_score_steps: int) -> torch.Tensor:
    """The zero-based timesteps of the score, the midpoints of n_score_steps equal bins
    of 0, ..., n_steps - 1; every timestep when the two counts are equal.

    Raises:
        ValueError: If n_score_steps is not in [1, n_steps].
    """
    if not 1 <= n_score_steps <= n_steps:
        raise ValueError(f"n_score_steps must lie in [1, {n_steps}].")
    bins = torch.arange(n_score_steps, dtype=torch.float64) + 0.5
    return torch.floor(bins * n_steps / n_score_steps).long()


def fixed_noise(n_score_steps: int, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """One standard normal draw per score timestep and input position, shape
    (n_score_steps, *shape), shared by every event (TabADM, Alg. 2).

    numpy's legacy RandomState keeps its stream fixed across releases (NEP 19), which
    torch's CPU generator does not, so an offline replay under another torch version
    draws the noise of training.
    """
    draws = np.random.RandomState(seed).standard_normal((n_score_steps, *shape))
    return torch.from_numpy(draws).float()


def timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps (batch,), shape (batch, width).

    The Transformer position embedding, which Ho et al. use (Sec. 4).
    """
    half = width // 2
    exponents = torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    angles = timesteps.float().unsqueeze(1) * torch.exp(-math.log(1e4) * exponents)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def _time_mlp(time_dim: int, width: int, activation: str) -> nn.Module:
    if time_dim % 2:
        raise ValueError(f"time_dim must be even, got {time_dim}.")
    return nn.Sequential(nn.Linear(time_dim, width), ACTIVATIONS[activation]())


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str, dropout: float):
        super().__init__()
        self.fc1, self.time = nn.Linear(width, width), nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act, self.dropout = ACTIVATIONS[activation](), nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        inner = self.fc1(self.act(h)) + self.time(embedding)
        return h + self.fc2(self.dropout(self.act(inner)))


class DenoiserMLP(nn.Module):
    """The noise predictor ε_θ(x_t, t) on flattened inputs.

    A linear map to ``width``, ``depth`` residual blocks, each adding a projection of
    the timestep embedding, and a linear map back to ``in_dim``, after an activation
    and with no activation on the output.
    """

    def __init__(
        self,
        in_dim: int,
        width: int = 128,
        depth: int = 2,
        time_dim: int = 32,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        self.in_dim, self.time_dim = in_dim, time_dim
        self.time = _time_mlp(time_dim, width, activation)
        self.inp, self.out = nn.Linear(in_dim, width), nn.Linear(width, in_dim)
        blocks = [_ResidualBlock(width, activation, dropout) for _ in range(depth)]
        self.blocks, self.act = nn.ModuleList(blocks), ACTIVATIONS[activation]()

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict the noise of x_t, shape (batch, in_dim), at integer timesteps."""
        embedding = self.time(timestep_embedding(timesteps, self.time_dim))
        h = self.inp(x_t)
        for block in self.blocks:
            h = block(h, embedding)
        return self.out(self.act(h))


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, time_width: int, act: str, p: float):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.time = nn.Linear(time_width, c_out)
        self.act, self.dropout = ACTIVATIONS[act](), nn.Dropout(p)

    def forward(self, h: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        inner = self.conv1(self.act(h)) + self.time(embedding)[:, :, None, None]
        return self.skip(h) + self.conv2(self.dropout(self.act(inner)))


class ImageDenoiser(nn.Module):
    """The noise predictor ε_θ(x_t, t) on images, a small U-Net.

    One residual block per entry of ``nodes`` (its channel width), each adding a
    projection of the timestep embedding; the resolution halves between levels by
    average pooling and is restored by nearest-neighbour upsampling, whose output is
    concatenated with the block of its level on the way down. The image side must be
    divisible by 2 ** (len(nodes) - 1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        nodes: list[int] | tuple[int, ...] = (16, 32),
        time_dim: int = 32,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        nodes, self.time_dim = list(nodes), time_dim
        width = max(nodes)
        self.time = _time_mlp(time_dim, width, activation)
        self.stem = nn.Conv2d(in_channels, nodes[0], 3, padding=1)
        block = partial(_ConvBlock, time_width=width, act=activation, p=dropout)
        self.down = nn.ModuleList(map(block, [nodes[0], *nodes[:-1]], nodes))
        ups = [block(nodes[i + 1] + nodes[i], nodes[i]) for i in range(len(nodes) - 1)]
        self.up = nn.ModuleList(ups[::-1])
        self.out = nn.Conv2d(nodes[0], in_channels, 3, padding=1)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict the noise of x_t, shape (batch, channels, height, width)."""
        embedding = self.time(timestep_embedding(timesteps, self.time_dim))
        h, skips = self.stem(x_t), []
        for level, block in enumerate(self.down):
            h = block(F.avg_pool2d(h, 2) if level else h, embedding)
            skips.append(h)
        for block, skip in zip(self.up, reversed(skips[:-1])):
            h = F.interpolate(h, scale_factor=2.0, mode="nearest")
            h = block(torch.cat([h, skip], dim=1), embedding)
        return self.out(self.act(h))
