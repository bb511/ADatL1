"""Normalized Autoencoder (NAE).

S. Yoon, Y.-K. Noh and F. C. Park, "Autoencoding Under Normalization Constraints",
ICML 2021, arXiv:2105.05735.
"""

import math
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.components.reconstruction import MSEReconstructionLoss
from src.data.utils import unpack_batch

# (failed, message) pairs over the saved hyperparameters, checked in this order.
_HPARAM_CHECKS = (
    (lambda hp: hp.pretrain_epochs < 0, "pretrain_epochs must be non-negative."),
    (lambda hp: hp.temperature <= 0.0, "temperature must be positive."),
    (lambda hp: hp.neg_energy_weight < 0.0, "neg_energy_weight must be non-negative."),
    (lambda hp: min(hp.latent_steps, hp.input_steps) < 0, "steps must be >= 0."),
    (
        lambda hp: min(hp.latent_step_size, hp.input_step_size) <= 0.0,
        "step sizes must be positive.",
    ),
    (lambda hp: hp.buffer_size < 1, "buffer_size must be positive."),
    (lambda hp: not 0.0 <= hp.replay_ratio <= 1.0, "replay_ratio must lie in [0, 1]."),
    (
        lambda hp: hp.energy_lr is not None and hp.energy_lr <= 0.0,
        "energy_lr must be positive or None.",
    ),
)


def _flat_inputs(batch) -> tuple[torch.Tensor, torch.Tensor | None]:
    b = unpack_batch(batch)
    x = torch.flatten(b.x, start_dim=1)
    mask = None if b.mask is None else torch.flatten(b.mask, start_dim=1).to(x.dtype)
    return x, mask


def _project(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, dim=1)


def _clip_rows(grad: torch.Tensor, max_norm: float | None) -> torch.Tensor:
    """Scale every row of ``grad`` down to norm ``max_norm`` at most."""
    if max_norm is None:
        return grad
    return grad * (max_norm / grad.norm(dim=1, keepdim=True).clamp_min(max_norm))


def _langevin(
    x: torch.Tensor,
    energy: Callable[[torch.Tensor], torch.Tensor],
    steps: int,
    step_size: float,
    temperature: float,
    project: Callable[[torch.Tensor], torch.Tensor],
    grad_clip: float | None = None,
    free: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run an unadjusted Langevin chain on energy/temperature and return its end point.

    One step is x <- project(x - step_size * g / temperature + sqrt(2 step_size) eps),
    where g is the gradient of the energy, its rows clipped to norm ``grad_clip``. Where
    ``free`` (of the shape of x) is 0, the entry does not move.
    """
    noise_std = math.sqrt(2.0 * step_size)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        grad = _clip_rows(torch.autograd.grad(energy(x).sum(), x)[0], grad_clip)
        step = noise_std * torch.randn_like(x) - step_size / temperature * grad
        x = project(x.detach() + (step if free is None else step * free))
    return x.detach()


def _potential(
    x: torch.Tensor,
    energy: Callable[[torch.Tensor], torch.Tensor],
    temperature: float,
    grad_clip: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return energy/temperature per row and its gradient, the rows of grad E clipped to
    norm ``grad_clip`` before the division."""
    x = x.detach().requires_grad_(True)
    e = energy(x)
    grad = _clip_rows(torch.autograd.grad(e.sum(), x)[0], grad_clip)
    return e.detach() / temperature, grad / temperature


def _log_gaussian_proposal(
    to: torch.Tensor,
    mean: torch.Tensor,
    step_size: float,
    free: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log density of N(mean, 2 step_size I) at ``to`` over the free entries, up to a
    constant."""
    diff = to - mean if free is None else (to - mean) * free
    return -diff.square().sum(dim=1) / (4.0 * step_size)


def _log_radial_integral(a: torch.Tensor, k: int) -> torch.Tensor:
    """Return log int_0^inf t^k exp(-t^2/2 + a t) dt for every entry of ``a``, k >= 1.

    The integrand is log-concave, with its peak at t* = (a + sqrt(a^2 + 4k))/2 and a
    curvature of 1 + k/t*^2 there, so that 513 nodes from 12 widths below the peak (cut
    at t = 0) to 20 above it resolve it for any a. Against a trapezoid on 4e5 nodes,
    with a in [-300, 300], the log is off by 3e-4 at k = 1, by 4e-6 at k = 2 and by less
    than 1e-7 at k = 3, 7 and 15.
    """
    root = (a.square() + 4 * k).sqrt()
    # The second form avoids the cancellation in a + root for a < 0.
    peak = torch.where(a > 0, (a + root) / 2, 2 * k / (root - a))
    width = (1.0 + k / peak.square()).rsqrt()
    s = torch.linspace(-12.0, 20.0, 513, dtype=a.dtype, device=a.device)
    t = peak[:, None] + width[:, None] * s
    log_f = k * t.clamp_min(1e-300).log() - t.square() / 2 + a[:, None] * t
    log_f = log_f.masked_fill(t <= 0, -math.inf)
    return torch.logsumexp(log_f, dim=1) + (width * (s[1] - s[0])).log()


def _log_projected_normal(
    u: torch.Tensor, mean: torch.Tensor, step_size: float
) -> torch.Tensor:
    """Log density of normalize(mean + sqrt(2 step_size) eps) at the unit rows ``u``,
    relative to the surface measure of the sphere.

    In polar coordinates the radius integrates out: with sigma^2 = 2 step_size,
    a = u.mean/sigma and d the dimension, the density is
    (2 pi)^(-d/2) exp(-|mean|^2/(2 sigma^2)) int_0^inf t^(d-1) exp(-t^2/2 + a t) dt.
    """
    sigma_sq = 2.0 * step_size
    dim = u.shape[1]
    # float64: |mean|^2/(2 sigma^2) and the log of the integral (about a^2/2) reach 1e6
    # for the smallest step sizes and cancel.
    mean = mean.double()
    a = (u.double() * mean).sum(dim=1) / math.sqrt(sigma_sq)
    exponent = mean.square().sum(dim=1) / (2.0 * sigma_sq)
    log_norm = -0.5 * dim * math.log(2.0 * math.pi) - exponent
    return (log_norm + _log_radial_integral(a, dim - 1)).to(u.dtype)


def _select(
    rows: torch.Tensor, new: tuple[torch.Tensor, ...], old: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """Take each row of every tensor from ``new`` where ``rows`` is true, else from
    ``old``."""
    return tuple(
        torch.where(rows.view(-1, *[1] * (n.ndim - 1)), n, o) for n, o in zip(new, old)
    )


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _mala(
    x: torch.Tensor,
    energy: Callable[[torch.Tensor], torch.Tensor],
    steps: int,
    step_size: float,
    temperature: float,
    log_proposal: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    project: Callable[[torch.Tensor], torch.Tensor],
    grad_clip: float | None = None,
    free: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a Metropolis-adjusted Langevin chain on energy/temperature; return its end
    point and the fraction of accepted proposals.

    The proposal of the step from x is x' = project(m(x) + sqrt(2 step_size) eps), with
    m(x) = x - step_size grad E(x)/T, accepted with probability
    min(1, exp(U(x) - U(x')) q(x | m(x'))/q(x' | m(x))), where U = E/T and
    ``log_proposal(to, mean)`` is log q up to a constant. Every row is its own chain.
    An infinite energy (outside the support) rejects. Where ``free`` (of the shape of
    x) is 0, the entry does not move.
    """
    # One transition is kept in one block: the proposal and its reverse density must
    # match term by term for the chain to be exact.
    noise_std = math.sqrt(2.0 * step_size)
    u, grad = _potential(x, energy, temperature, grad_clip)
    accepted = x.new_zeros(())
    for _ in range(steps):
        mean = x - step_size * grad
        step = noise_std * torch.randn_like(x) - step_size * grad
        proposal = project(x + (step if free is None else step * free))
        u_new, grad_new = _potential(proposal, energy, temperature, grad_clip)
        forward = log_proposal(proposal, mean)
        backward = log_proposal(x, proposal - step_size * grad_new)
        log_ratio = u - u_new + backward - forward
        accept = torch.rand_like(u).log() < log_ratio
        x, u, grad = _select(accept, (proposal, u_new, grad_new), (x, u, grad))
        accepted = accepted + accept.float().mean()
    return x.detach(), accepted / max(steps, 1)


@torch.no_grad()
def _nn_two_sample_accuracy(a: torch.Tensor, b: torch.Tensor) -> float:
    """Leave-one-out accuracy of the 1-nearest-neighbour rule telling the rows of ``a``
    from those of ``b``: near 0.5 for two samples of one distribution, 1 when apart."""
    points = torch.cat([a, b])
    labels = torch.cat([a.new_zeros(len(a)), b.new_ones(len(b))])
    distances = torch.cdist(points, points).fill_diagonal_(float("inf"))
    return (labels[distances.argmin(dim=1)] == labels).float().mean().item()


class NAE(ADLightningModule):
    """Normalized Autoencoder; the module docstring describes the model and sampler.

    Args:
        encoder: Network whose output, projected onto the unit sphere, is the code.
        decoder: Network from the code back to the input; ``in_dim`` is the code size.
        pretrain_epochs: Number of epochs of plain autoencoder training before the
            energy phase.
        temperature: T of p(x) ∝ exp(-E(x)/T), which enters the sampler only.
        neg_energy_weight: gamma, the weight of the mean squared negative energy.
        latent_steps: Steps of the latent chain.
        latent_step_size: lambda of the latent chain.
        input_steps: Steps of the input chain.
        input_step_size: lambda of the input chain.
        input_grad_clip: Largest norm of each row of grad E in the input chain, taken
            before the division by T; None does not clip.
        input_bound: The decoded starts are clamped to [-input_bound, input_bound],
            and so is every unadjusted input step, while an adjusted one outside the
            box is rejected; None leaves them unbounded.
        metropolis: Accept every Langevin step of both chains with the
            Metropolis-Hastings probability (MALA); False takes every step (ULA).
        buffer_size: Number of latent end points kept for replay.
        replay_ratio: Probability that a latent chain starts from the buffer.
        energy_lr: Learning rate of every epoch from ``pretrain_epochs`` on; None keeps
            the optimiser's.
        target_rate: Target rate, or the target FPR when ``base_rate`` is None.
        base_rate: Base rate that turns ``target_rate`` into an FPR.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        pretrain_epochs: int,
        temperature: float,
        neg_energy_weight: float,
        latent_steps: int,
        latent_step_size: float,
        input_steps: int,
        input_step_size: float,
        input_grad_clip: float | None = None,
        input_bound: float | None = None,
        metropolis: bool = False,
        buffer_size: int = 10000,
        replay_ratio: float = 0.95,
        energy_lr: float | None = None,
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "encoder", "decoder"])
        self._check_hparams()
        self.encoder, self.decoder = encoder, decoder
        self.mse = MSEReconstructionLoss(reduction="none")
        # Plain attributes, so that the state dict holds the weights alone.
        self._buffer: torch.Tensor | None = None
        self._buffer_next = 0
        self._buffer_filled = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return _project(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decoder(z)

    def energy(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Per-dimension mean squared reconstruction error over the present entries.

        ``x`` and ``mask`` have shape (batch, features), the result (batch,).
        """
        return self.mse(x, self.forward(x)[1], mask)

    def model_step(self, batch) -> dict[str, torch.Tensor]:
        x, mask = _flat_inputs(batch)
        positive = self.energy(x, mask)
        if not self._energy_phase():
            return self._outputs(positive, positive.mean())

        negative_x, decoded, acceptance = self.sample_negatives(x, mask)
        negative = self.energy(negative_x, mask)
        gamma = self.hparams.neg_energy_weight
        loss = positive.mean() - negative.mean() + gamma * negative.square().mean()
        logs = self._negative_logs(x, negative_x, negative, decoded, mask)
        return {**self._outputs(positive, loss), **logs, **acceptance}

    def sample_negatives(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Draw one negative per row of ``x``.

        Returns:
            The negatives and the decoded latent end points that started their input
            chains, both detached, and the fractions of accepted proposals of the two
            chains, empty without ``metropolis``.
        """
        z, latent_rate = self._latent_chain(x, mask)
        self._push(z)
        decoded = self._clamp(self._decode_into(z, x, mask).detach())
        negative, input_rate = self._input_chain(decoded, mask)
        if latent_rate is None:
            return negative, decoded, {}
        rates = {"sample/accept_latent": latent_rate, "sample/accept_input": input_rate}
        return negative, decoded, rates

    def _latent_chain(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hp = self.hparams
        starts = self._latent_starts(len(x), x.device)
        energy = partial(self._latent_energy, x=x, mask=mask)
        args = (hp.latent_steps, hp.latent_step_size, hp.temperature)
        if not hp.metropolis:
            return _langevin(starts, energy, *args, _project), None
        log_q = partial(_log_projected_normal, step_size=hp.latent_step_size)
        return _mala(starts, energy, *args, log_q, _project)

    def _input_chain(
        self, decoded: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hp = self.hparams
        args = (hp.input_steps, hp.input_step_size, hp.temperature)
        clip = hp.input_grad_clip
        if not hp.metropolis:
            energy = partial(self.energy, mask=mask)
            return _langevin(decoded, energy, *args, self._clamp, clip, mask), None
        energy = partial(self._boxed_energy, mask=mask)
        log_q = partial(_log_gaussian_proposal, step_size=hp.input_step_size, free=mask)
        return _mala(decoded, energy, *args, log_q, _identity, clip, mask)

    def on_train_epoch_start(self) -> None:
        if self.hparams.energy_lr is None or not self._energy_phase():
            return
        for group in self.trainer.optimizers[0].param_groups:
            group["lr"] = self.hparams.energy_lr

    def _energy_phase(self) -> bool:
        return self.training and self.current_epoch >= self.hparams.pretrain_epochs

    def _decode_into(
        self, z: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Decoded codes, with the entries of ``x`` wherever ``mask`` is 0."""
        decoded = self.decoder(z)
        return decoded if mask is None else torch.where(mask.bool(), decoded, x)

    def _latent_energy(
        self, z: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        return self.energy(self._decode_into(z, x, mask), mask)

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        bound = self.hparams.input_bound
        return x if bound is None else x.clamp(-bound, bound)

    def _boxed_energy(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The energy, infinite on the rows with an entry outside the input bounds."""
        e = self.energy(x, mask)
        bound = self.hparams.input_bound
        if bound is None:
            return e
        return e.masked_fill((x.abs() > bound).any(dim=1), math.inf)

    def _latent_starts(self, n: int, device: torch.device) -> torch.Tensor:
        # A normalised Gaussian draw is uniform on the sphere.
        fresh = _project(torch.randn(n, self.decoder.in_dim, device=device))
        if self._buffer_filled == 0:
            return fresh
        rows = torch.randint(self._buffer_filled, (n,), device=device)
        replay = torch.rand(n, 1, device=device) < self.hparams.replay_ratio
        return torch.where(replay, self._buffer[rows], fresh)

    def _push(self, z: torch.Tensor) -> None:
        """Write the latent end points into the ring buffer, oldest first out."""
        if self._buffer is None:
            self._buffer = z.new_zeros(self.hparams.buffer_size, z.shape[1])
        size = len(self._buffer)
        rows = (self._buffer_next + torch.arange(len(z), device=z.device)) % size
        self._buffer[rows] = z
        self._buffer_next = (self._buffer_next + len(z)) % size
        self._buffer_filled = min(self._buffer_filled + len(z), size)

    def _outputs(self, positive: torch.Tensor, loss: torch.Tensor) -> dict:
        with torch.no_grad():
            operational_ascore = self.compute_operational_ascore(positive)
        return {
            # Used for backpropagation:
            "loss": loss,
            # Used for logging:
            "energy/positive": positive.mean().detach(),
            "ascore/operational": operational_ascore,
            # Used for callbacks:
            "loss/full": positive.detach(),
            "ascore/full": positive.detach(),
            "fit/full": positive.detach(),
        }

    @torch.no_grad()
    def _negative_logs(
        self,
        x: torch.Tensor,
        negative_x: torch.Tensor,
        negative: torch.Tensor,
        decoded: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict:
        return {
            "energy/negative": negative.mean(),
            "energy/decoded": self.energy(decoded, mask).mean(),
            "sample/nn_accuracy": _nn_two_sample_accuracy(x, negative_x),
        }

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "energy_positive": outdict.get("energy/positive"),
            "energy_negative": outdict.get("energy/negative"),
            "energy_decoded": outdict.get("energy/decoded"),
            "sample_nn_accuracy": outdict.get("sample/nn_accuracy"),
            "accept_latent": outdict.get("sample/accept_latent"),
            "accept_input": outdict.get("sample/accept_input"),
            "ascore_operational": outdict.get("ascore/operational"),
        }

    def _check_hparams(self) -> None:
        for failed, message in _HPARAM_CHECKS:
            if failed(self.hparams):
                raise ValueError(message)
