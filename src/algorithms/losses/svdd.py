import torch
from torch import nn


class SVDDLoss(nn.Module):
    """Stateless Deep SVDD data term.

    Center/radius lifecycle management belongs to the Lightning module. Network regularization is
    computed from encoder parameters, never latent vectors.
    """

    def __init__(self, objective: str = "one_class", nu: float = 0.1):
        super().__init__()
        if objective not in {"one_class", "soft_boundary"}:
            raise ValueError(
                "objective must be 'one_class' or 'soft_boundary', " f"got {objective!r}."
            )
        if not (0.0 < nu <= 1.0):
            raise ValueError(f"nu must satisfy 0 < nu <= 1, got {nu}.")
        self.objective = objective
        self.nu = float(nu)

    def forward(
        self,
        distances: torch.Tensor,
        radius: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the per-event one-class or soft-boundary data term."""
        if distances.ndim != 1:
            raise ValueError(f"distances must have shape [B], got shape {tuple(distances.shape)}")
        if self.objective == "one_class":
            return distances
        if radius is None or radius.ndim != 0:
            raise ValueError("soft_boundary requires a scalar radius.")

        radius_squared = radius.square()
        return radius_squared + torch.clamp(distances - radius_squared, min=0.0) / self.nu
