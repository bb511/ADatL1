from typing import Optional, Tuple

import math
import torch
import torch.nn as nn


class LatentCouplingLayer(nn.Module):
    """Affine coupling layer for dense latent vectors.

    This is a standard RealNVP-style coupling layer operating on a dense vector.
    A binary mask splits the input dimensions into:
      - masked dimensions: copied through unchanged
      - unmasked dimensions: affinely transformed using parameters predicted
        from the masked dimensions.

    :param input_dim: Int with number of input dimensions to the layer.
    :param hidden_dim: Int with the number of nodes in the hidden layer.
    :param mask: Torch tensor for input mask.
    :param n_hidden_layers: Int with the number of MLP layers in a coupling layer.
    :param activation: String with the activation used in the MLP.
    :param scale_clamp: Float with the clamp factor, for numerical stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        n_hidden_layers: int = 2,
        activation: str = "relu",
        scale_clamp: float = 5.0,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transform x -> z.

        The masked part of x is passed unchanged.
        The unmasked part is transformed using an affine function whose parameters
        are predicted from the masked part.
        """
        x_masked = x * self.mask

        shift_logscale = self.net(x_masked)
        shift, log_scale = torch.chunk(shift_logscale, 2, dim=1)

        inv_mask = 1.0 - self.mask

        # Zero out parameters on masked dimensions so they are copied through untouched.
        shift = shift * inv_mask

        # Bound the log-scale for numerical stability, then apply only to transformed dims.
        log_scale = torch.tanh(log_scale) * self.scale_clamp
        log_scale = log_scale * inv_mask

        # RealNVP affine coupling transform:
        # masked dims: copied through
        # unmasked dims: x * exp(log_scale) + shift
        z = x_masked + inv_mask * (x * torch.exp(log_scale) + shift)

        # Jacobian is triangular, so log|det J| is just the sum of log-scales
        # over transformed dimensions.
        log_det = log_scale.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transform z -> x.

        Uses the same masked conditioning mechanism as forward(), but analytically
        inverts the affine transformation on the unmasked coordinates.
        """
        z_masked = z * self.mask

        shift_logscale = self.net(z_masked)
        shift, log_scale = torch.chunk(shift_logscale, 2, dim=1)

        inv_mask = 1.0 - self.mask
        shift = shift * inv_mask
        log_scale = torch.tanh(log_scale) * self.scale_clamp
        log_scale = log_scale * inv_mask

        x = z_masked + inv_mask * (z - shift) * torch.exp(-log_scale)

        log_det = (-log_scale).sum(dim=1)
        return x, log_det


class LatentRealNVP(nn.Module):
    """RealNVP operating on a dense latent vector.

    This flow models the distribution of latent representations produced by the
    DS-VAE encoder. It learns an invertible mapping between the latent space and
    a simple Gaussian base distribution.

    :param input_dim: Int with the number of input dimensions.
    :param n_flows: Int with the number of flows that should be used.
    :param hidden_dim: Int with the number of nodes in the MLP.
    :param n_hidden_layers: Int with the number of hidden layers in the MLP.
    :param activation: String with the activation to use in the MLP.
    :param scale_clamp: Float with the clamp factor, for numerical stability.
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
                LatentCouplingLayer(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    mask=mask,
                    n_hidden_layers=n_hidden_layers,
                    activation=activation,
                    scale_clamp=scale_clamp,
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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform data x into base-space z."""
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x

        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det

        return z, log_det_sum

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform from base-space z back to data-space x."""
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        x = z

        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_sum += log_det

        return x, log_det_sum

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability under the flow model."""
        if self.training and self.noise_scale > 0.0:
            x = x + torch.randn_like(x) * self.noise_scale

        z, log_det = self.encode(x)

        # Log-probability under a diagonal Gaussian base distribution.
        log_pz = -0.5 * (
            ((z - self.base_mean) ** 2) / torch.exp(2 * self.base_log_std)
            + 2 * self.base_log_std
            + math.log(2 * math.pi)
        ).sum(dim=1)

        # Change-of-variables formula.
        return log_pz + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the learned flow distribution."""
        z = torch.randn(n_samples, self.input_dim, device=self.base_mean.device)

        x, _ = self.decode(z)
        return x
