"""Per-event data terms of the Deep SVDD objectives.

Ruff et al., "Deep One-Class Classification", ICML 2018, PMLR 80:4393-4402, eqs. 3 and
4. The one-class term is the squared distance itself. The weight-decay term of both
objectives reads the network, so the module adds it.
"""

import torch


def soft_boundary_loss(
    distances: torch.Tensor, radius: torch.Tensor, nu: float
) -> torch.Tensor:
    """Return R² + max(0, d - R²) / ν per event; the batch mean is eq. 3's term.

    ``distances`` holds the squared distances d to the centre, shape (batch,). The
    radius enters as a constant, since the quantile rule sets it, not the gradient.
    """
    squared_radius = radius.detach() ** 2
    return squared_radius + torch.relu(distances - squared_radius) / nu
