import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

from src.algorithms.losses.components import L1ADLoss


class NTXentLoss(L1ADLoss):
    """Normalized temperature-scaled cross entropy loss for two augmented views."""

    def __init__(
        self,
        temperature: float = 0.1,
        gather_distributed: bool = True,
    ):
        super().__init__(scale=None, reduction=None)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)
        self.gather_distributed = bool(gather_distributed)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}.")
        if z1.ndim != 2:
            raise ValueError(f"Expected rank-2 projections, got shape {z1.shape}.")

        batch_size = z1.shape[0]
        if batch_size < 2:
            raise ValueError("NTXentLoss needs at least two samples per batch.")

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        global_z1, rank, _ = self._gather(z1)
        global_z2, _, _ = self._gather(z2)
        global_batch = global_z1.shape[0]

        candidates = torch.cat([global_z1, global_z2], dim=0)
        anchors = torch.cat([z1, z2], dim=0)
        logits = (anchors @ candidates.T) / self.temperature

        local_indices = rank * batch_size + torch.arange(batch_size, device=z1.device)
        self_indices = torch.cat([local_indices, global_batch + local_indices])
        anchor_indices = torch.arange(2 * batch_size, device=z1.device)
        logits[anchor_indices, self_indices] = float("-inf")
        labels = torch.cat([global_batch + local_indices, local_indices])
        return F.cross_entropy(logits, labels)

    def _gather(self, z: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if not self.gather_distributed or not dist.is_available() or not dist.is_initialized():
            return z, 0, 1

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sizes = [torch.zeros((), dtype=torch.long, device=z.device) for _ in range(world_size)]
        local_size = torch.tensor(z.shape[0], dtype=torch.long, device=z.device)
        dist.all_gather(sizes, local_size)
        if any(int(size.item()) != z.shape[0] for size in sizes):
            raise RuntimeError(
                "Distributed NT-Xent requires equal per-rank batch sizes. "
                "Use drop_last=True or a fixed-size iterable batcher."
            )

        return torch.cat(all_gather(z), dim=0), rank, world_size

    def reduce(self):
        pass
