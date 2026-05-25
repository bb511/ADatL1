import torch
import torch.nn.functional as F

from src.algorithms.losses.components import L1ADLoss


class NTXentLoss(L1ADLoss):
    """Normalized temperature-scaled cross entropy loss for two augmented views."""

    def __init__(self, temperature: float = 0.1):
        super().__init__(scale=None, reduction=None)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(
                f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}."
            )
        if z1.ndim != 2:
            raise ValueError(f"Expected rank-2 projections, got shape {z1.shape}.")

        batch_size = z1.shape[0]
        if batch_size < 2:
            raise ValueError("NTXentLoss needs at least two samples per batch.")

        z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
        logits = (z @ z.T) / self.temperature
        logits = logits.masked_fill(
            torch.eye(2 * batch_size, device=z.device, dtype=torch.bool),
            float("-inf"),
        )

        labels = torch.arange(2 * batch_size, device=z.device)
        labels = (labels + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, labels)

    def reduce(self):
        pass
