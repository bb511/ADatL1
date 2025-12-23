from typing import Any, Callable, Dict, Optional, Tuple
import math
import random

import torch
from torch import nn
import torch.autograd as autograd


class LangevinSampler(nn.Module):
    """
    Langevin Monte Carlo sampler module.

    :param n_steps: Number of Langevin updates to perform.
    :param step_size: Step size of the drift term.  If ``None``, it is inferred from
        ``noise_scale`` via ``step_size = noise_scale**2 / 2``.
    :param noise_scale: Standard deviation of the noise term.  If ``None``, it is
        inferred from ``step_size`` via ``noise_scale = sqrt(2 * step_size)``.
    :param temperature: Temperature parameter controlling acceptance probabilities.
    :param clip: Optional boundaries for clipping or rejection.
    :param clip_grad: Gradient clipping threshold.
    :param spherical: Whether to project samples to the unit hypersphere.
    :param mh: Whether to perform Metropolis-Hastings correction.
    """

    """
    BUG WARNING:
    Make sure the input tensor passed to :func:`forward()` has
    ``requires_grad=True`` if you need gradients with respect to the
    starting point.  Otherwise, the sampling will still function but
    gradients will not propagate through the sampling process.
    """

    def __init__(
        self,
        n_steps: int = 50,
        step_size: Optional[float] = 10.0,
        noise_scale: Optional[float] = 0.05,
        temperature: float = 1.0,
        clip: Optional[Tuple[float, float]] = (0.0, 1.0),
        clip_grad: Optional[float] = None,
        spherical: Optional[bool] = False,
        mh: bool = False,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.step_size = step_size
        self.noise_scale = (
            math.sqrt(2 * self.step_size) if noise_scale is None else noise_scale
        )
        self.temperature = temperature
        self.clip = tuple(clip)
        self.clip_grad = clip_grad
        self.reject_boundary = False if self.clip is None else False
        self.spherical = spherical
        self.mh = mh

        # To store some logs
        self.sampler_dict: Dict[str, Any] = {}

    def step(
        self,
        x: torch.Tensor,
        energy: torch.Tensor,
        grad_energy: torch.Tensor,
        model: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Perform a single Langevin (MALA) update.

        :param x: Current state of the chain.  Shape ``(batch, *dims)``.
        :param energy: Energy of the current state.  Shape ``(batch,)``.
        :param grad_energy: Gradient of ``energy`` with respect to ``x``.
        :param model: Function mapping a tensor ``x`` to its energy.

        :return: A tuple containing the proposed next state, its energy, its gradient, and a dictionary of diagnostic information.
        """
        # Gaussian noise for the diffusion term
        noise = torch.randn_like(x) * self.noise_scale
        # Drift term proportional to the negative gradient of the energy
        drift = -self.step_size * grad_energy
        # Langevin proposal
        dynamics = drift / self.temperature + noise
        x_new = x + dynamics

        # Apply boundary handling
        reject = torch.zeros(len(x_new), dtype=torch.bool)
        if self.clip is not None:
            if self.reject_boundary:
                # Reject any sample that falls outside the bounds
                accept = (
                    ((x_new >= self.clip[0]) & (x_new <= self.clip[1]))
                    .view(len(x), -1)
                    .all(dim=1)
                )
                reject = ~accept
                # Revert rejected samples to original state
                x_new[reject] = x[reject]
            else:
                # Simply clamp values to the boundaries
                x_new = torch.clamp(x_new, self.clip[0], self.clip[1])

        # Project onto hypersphere if requested
        if self.spherical:
            x_new = x_new / x_new.norm(dim=1, p=2, keepdim=True)

        # Compute energy and gradient at proposal
        x_new.requires_grad_(True)
        energy_new = model(x_new)
        grad_energy_new = autograd.grad(energy_new.sum(), x_new, create_graph=True)[0]
        if self.clip_grad is not None:
            grad_energy_new = torch.clamp(
                grad_energy_new, -self.clip_grad, self.clip_grad
            )

        # Metropolis–Hastings acceptance (optional)
        results_dict: Dict[str, Any] = {}
        if self.mh:
            # See: https://arxiv.org/abs/2102.07796 for details on MALA acceptance
            from_new = (grad_energy + grad_energy_new) * self.step_size - noise
            from_new = from_new.view(len(x), -1).norm(p=2, dim=1, keepdim=True) ** 2
            to_new = noise.view(len(x), -1).norm(dim=1, keepdim=True, p=2) ** 2
            transition = -(from_new - to_new) / (4 * self.step_size)
            prob = -energy_new + energy
            accept_prob = torch.exp((transition + prob) / self.temperature)[:, 0]
            accept = torch.rand_like(accept_prob) < accept_prob

            # Replace rejected samples with original ones
            x_new[~accept] = x[~accept]
            energy_new[~accept] = energy[~accept]
            grad_energy_new[~accept] = grad_energy[~accept]
            results_dict["mh_accept"] = accept

        # Diagnostics
        results_dict.update(
            {
                "dynamics": dynamics.detach().cpu(),
                "drift": drift.detach().cpu(),
                "diffusion": noise.detach().cpu(),
            }
        )
        return x_new, energy_new, grad_energy_new, results_dict

    def forward(
        self, x: torch.Tensor, model: Callable[[torch.Tensor], torch.Tensor]
    ) -> Dict[str, Any]:
        """Run a Langevin Monte Carlo chain.

        :param x: Current state of the chain.  Shape ``(batch, *dims)``.
        :param model: Function mapping a tensor ``x`` to its energy.

        :return: A dictionary containing the final sample and diagnostic information.
        """

        # Ensure gradient tracking
        x = x.clone().detach().requires_grad_(True)
        energy_x = model(x)
        grad_energy_x = autograd.grad(energy_x.sum(), x, create_graph=True)[0]
        if self.clip_grad is not None:
            grad_energy_x = torch.clamp(grad_energy_x, -self.clip_grad, self.clip_grad)

        # self.sampler_dict: Dict[str, Any] = {}
        # self.sampler_dict["samples"] = [x.detach().cpu()]
        for _ in range(self.n_steps):
            x, energy_x, grad_energy_x, results_dict = self.step(
                x,
                energy_x,
                grad_energy_x,
                model,
            )
            # sampler_dict["samples"].append(x.detach().cpu())

            # Record diagnostics over the chain
            # for key, value in results_dict.items():
            #     evo_key = f"{key}_evolution"
            #     if evo_key not in sampler_dict:
            #         self.sampler_dict[evo_key] = [value]
            #     else:
            #         self.sampler_dict[evo_key].append(value)

        # self.sampler_dict["sample"] = x
        return x


class SampleBuffer:
    """
    Replay buffer used for persistent MCMC chains.

    When training with persistent contrastive divergence (PCD) or on-
    manifold initialization (OMI), the WNAE algorithm benefits from
    maintaining a buffer of previously generated samples. During
    subsequent sampling calls, some of these buffered samples are
    randomly selected to seed new chains.  This stabilizes training
    and speeds up convergence.

    :param max_samples: Maximum number of samples held in the buffer.
    :param replay_ratio: Fraction of new chains to initialize from the replay buffer.
    """

    def __init__(
        self, max_samples: Optional[int] = 10_000, replay_ratio: Optional[float] = 0.95
    ) -> None:
        self.max_samples = max_samples
        self.buffer: list[torch.Tensor] = []
        self.replay_ratio = replay_ratio

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, samples: torch.Tensor) -> None:
        """Add new samples to the buffer."""
        samples = samples.detach().to("cpu")
        for sample in samples:
            self.buffer.append(sample)
            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples: int) -> torch.Tensor:
        """Retrieve ``n_samples`` from the buffer with replacement."""

        if n_samples > len(self.buffer):
            print(f"Sampling {n_samples} from buffer with size {len(self.buffer)}!")

        samples = random.choices(self.buffer, k=n_samples)
        return torch.stack(list(samples), dim=0)
