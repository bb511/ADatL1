import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    """Affine coupling layer for dense latent vectors.

    This is a standard RealNVP-style coupling layer operating on a dense vector.
    A binary mask splits the input dimensions into:
      - masked dimensions: copied through unchanged
      - unmasked dimensions: affinely transformed using parameters predicted
        from the masked dimensions.

    The optional ``present`` of forward() and inverse() is a data mask shaped like the
    input, 1 where a coordinate holds data and 0 where it is padding, unrelated to the
    alternating ``mask``. Padded coordinates are copied through unchanged, with log-det
    0, and the conditioner reads them as zeros, so that their values reach neither the
    other coordinates nor the log-det.

    :param input_dim: Int with number of input dimensions to the layer.
    :param hidden_dim: Int with the number of nodes in the hidden layer.
    :param mask: Torch tensor for input mask.
    :param n_hidden_layers: Int with the number of MLP layers in a coupling layer.
    :param activation: String with the activation used in the MLP.
    :param scale_clamp: Float with the clamp factor, for numerical stability.
    :param identity_init: Bool, whether to zero the weight and bias of the last MLP
        layer, so that the untrained layer is the identity map with log-det 0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        n_hidden_layers: int = 2,
        activation: str = "relu",
        scale_clamp: float = 5.0,
        identity_init: bool = False,
    ):
        super().__init__()

        self.register_buffer("mask", mask.float())

        # Clamp factor controlling the maximum magnitude of log-scale outputs.
        # This helps prevent numerical instability from very large exponentials.
        self.scale_clamp = scale_clamp

        out_dim = 2 * input_dim

        layers = []
        mlp_in = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(mlp_in, hidden_dim))
            layers.append(self._get_activation(activation))
            mlp_in = hidden_dim
        layers.append(nn.Linear(mlp_in, out_dim))
        self.net = nn.Sequential(*layers)

        # Zeroed after the default init rather than instead of it, so that the RNG
        # draws, and every other parameter, match those of a flow without the flag.
        if identity_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def _get_activation(self, activation: str) -> nn.Module:
        """Map activation name to torch module."""
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {activation}")

    def _shift_log_scale(
        self, x: torch.Tensor, present: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shift and log-scale, zero on every coordinate copied through untouched."""
        x_masked = x * self.mask
        if present is not None:
            x_masked = x_masked * present
        shift, log_scale = torch.chunk(self.net(x_masked), 2, dim=1)

        active = 1.0 - self.mask
        if present is not None:
            active = active * present
        # Bound the log-scale for numerical stability.
        log_scale = torch.tanh(log_scale) * self.scale_clamp
        return shift * active, log_scale * active

    def forward(
        self, x: torch.Tensor, present: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transform x -> z.

        The masked part of x is passed unchanged. The unmasked part is transformed using
        an affine function whose parameters are predicted from the masked part.
        """
        shift, log_scale = self._shift_log_scale(x, present)
        z = x * self.mask + (1.0 - self.mask) * (x * torch.exp(log_scale) + shift)

        # Jacobian is triangular, so log|det J| is just the sum of log-scales
        # over transformed dimensions.
        return z, log_scale.sum(dim=1)

    def inverse(
        self, z: torch.Tensor, present: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transform z -> x.

        Uses the same masked conditioning mechanism as forward(), but analytically
        inverts the affine transformation on the unmasked coordinates.
        """
        shift, log_scale = self._shift_log_scale(z, present)
        x = z * self.mask + (1.0 - self.mask) * (z - shift) * torch.exp(-log_scale)
        return x, (-log_scale).sum(dim=1)


class RealNVP(nn.Module):
    """RealNVP operating on a dense vector.

    This flow models the distribution of the flattened events or images given to
    src.algorithms.realnvp.RealNVP. It learns an invertible mapping between the input
    space and a simple Gaussian base distribution.

    :param input_dim: Int with the number of input dimensions.
    :param n_flows: Int with the number of flows that should be used.
    :param hidden_dim: Int with the number of nodes in the MLP.
    :param n_hidden_layers: Int with the number of hidden layers in the MLP.
    :param activation: String with the activation to use in the MLP.
    :param noise_scale: Float with the standard deviation of the Gaussian noise added to
        the inputs in training mode.
    :param scale_clamp: Float with the clamp factor, for numerical stability.
    :param identity_init: Bool, whether every coupling layer starts as the identity map,
        so that the untrained flow is the base Gaussian.
    """

    def __init__(
        self,
        input_dim: int,
        n_flows: int = 6,
        hidden_dim: int = 32,
        n_hidden_layers: int = 2,
        activation: str = "relu",
        noise_scale: float = 0.0,
        scale_clamp: float = 5.0,
        identity_init: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.noise_scale = noise_scale

        # Stack multiple affine coupling layers with alternating masks.
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            # Alternate which coordinates are masked / transformed in each layer.
            mask = self._make_alternating_mask(input_dim, flip=bool(i % 2))
            self.flows.append(
                CouplingLayer(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    mask=mask,
                    n_hidden_layers=n_hidden_layers,
                    activation=activation,
                    scale_clamp=scale_clamp,
                    identity_init=identity_init,
                )
            )

        self.register_buffer("base_mean", torch.zeros(input_dim))
        self.register_buffer("base_log_std", torch.zeros(input_dim))

    def _make_alternating_mask(self, dim: int, flip: bool = False) -> torch.Tensor:
        """Create a standard alternating binary mask."""
        mask = torch.zeros(dim)
        mask[::2] = 1.0
        if flip:
            mask = 1.0 - mask
        return mask

    def encode(
        self, x: torch.Tensor, present: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform data x into base-space z."""
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x

        for flow in self.flows:
            z, log_det = flow(z, present)
            log_det_sum += log_det

        return z, log_det_sum

    def decode(
        self, z: torch.Tensor, present: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform from base-space z back to data-space x."""
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        x = z

        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x, present)
            log_det_sum += log_det

        return x, log_det_sum

    def _base_log_prob(
        self, z: torch.Tensor, present: torch.Tensor | None
    ) -> torch.Tensor:
        """Log-probability under a diagonal Gaussian base distribution."""
        terms = (
            ((z - self.base_mean) ** 2) / torch.exp(2 * self.base_log_std)
            + 2 * self.base_log_std
            + math.log(2 * math.pi)
        )
        if present is not None:
            terms = terms * present
        return -0.5 * terms.sum(dim=1)

    def log_prob(
        self, x: torch.Tensor, present: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute log probability under the flow model.

        With a data mask ``present`` (see CouplingLayer), the padded coordinates add
        nothing to the base log-density or the log-det, so that the result is the
        log-density of the present coordinates alone.
        """
        if self.training and self.noise_scale > 0.0:
            x = x + torch.randn_like(x) * self.noise_scale

        z, log_det = self.encode(x, present)

        # Change-of-variables formula.
        return self._base_log_prob(z, present) + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the learned flow distribution."""
        z = torch.randn(n_samples, self.input_dim, device=self.base_mean.device)

        x, _ = self.decode(z)
        return x
