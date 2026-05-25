from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.data.utils import unpack_batch


@dataclass
class PairingResult:
    idx_1: torch.Tensor
    idx_2: torch.Tensor
    distance: torch.Tensor
    rank_1_to_2: torch.Tensor
    rank_2_to_1: torch.Tensor

    @property
    def coverage_1(self) -> float:
        if self.idx_1.numel() == 0:
            return 0.0
        return float(self.idx_1.numel())


@torch.no_grad()
def collect_representations(
    model,
    dataloader,
    device: torch.device | str = "cpu",
    max_events: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reps = []
    xs = []
    ys = []
    model.eval()
    for batch in dataloader:
        batch = _batch_to_device(batch, device)
        b = unpack_batch(batch)
        rep = model.encode_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        reps.append(rep.detach().cpu())
        xs.append(x.detach().cpu())
        ys.append(b.y.detach().cpu())
        if max_events is not None and sum(t.shape[0] for t in reps) >= max_events:
            break

    rep = torch.cat(reps, dim=0)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    if max_events is not None:
        rep = rep[:max_events]
        x = x[:max_events]
        y = y[:max_events]
    return rep, x, y


@torch.no_grad()
def collect_closure_representations(
    model,
    dataloader,
    device: torch.device | str = "cpu",
    max_events: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    reps1 = []
    reps2 = []
    model.eval()
    for batch in dataloader:
        batch = _batch_to_device(batch, device)
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        m = model._flat_mask(b.mask) if hasattr(model, "_flat_mask") else None
        x1, x2 = model.augment_pair(x)
        reps1.append(model.encode_flat(x1, m).detach().cpu())
        reps2.append(model.encode_flat(x2, m).detach().cpu())
        if max_events is not None and sum(t.shape[0] for t in reps1) >= max_events:
            break

    z1 = torch.cat(reps1, dim=0)
    z2 = torch.cat(reps2, dim=0)
    if max_events is not None:
        z1 = z1[:max_events]
        z2 = z2[:max_events]
    return z1, z2


def closure_metrics(
    z1: torch.Tensor,
    z2: torch.Tensor,
    ks: tuple[int, ...] = (1, 10),
) -> dict[str, float]:
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    sim = z1 @ z2.T
    labels = torch.arange(z1.shape[0])
    metrics = {}
    for k in ks:
        k_eff = min(int(k), z2.shape[0])
        topk = torch.topk(sim, k=k_eff, dim=1).indices
        metrics[f"closure_recall_at_{k}"] = (
            topk == labels[:, None]
        ).any(dim=1).float().mean().item()

    order = torch.argsort(sim, dim=1, descending=True)
    true_pos = (order == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    metrics["closure_median_rank"] = true_pos.float().median().item()
    metrics["closure_mean_pos_distance"] = (
        1.0 - torch.diagonal(sim, 0)
    ).mean().item()
    return metrics


def mutual_nearest_pairs(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int = 20,
    caliper: float | None = None,
) -> PairingResult:
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    k12 = min(int(k), z2.shape[0])
    k21 = min(int(k), z1.shape[0])

    sim12 = z1 @ z2.T
    sim21 = sim12.T
    val12, idx12 = torch.topk(sim12, k=k12, dim=1)
    _, idx21 = torch.topk(sim21, k=k21, dim=1)

    reverse_ranks: dict[tuple[int, int], int] = {}
    for j in range(idx21.shape[0]):
        for rank, i in enumerate(idx21[j].tolist()):
            reverse_ranks[(i, j)] = rank + 1

    candidates = []
    for i in range(idx12.shape[0]):
        for rank12, j in enumerate(idx12[i].tolist()):
            rank21 = reverse_ranks.get((i, j))
            if rank21 is None:
                continue
            distance = float(1.0 - val12[i, rank12].item())
            if caliper is not None and distance > caliper:
                continue
            candidates.append((distance, i, j, rank12 + 1, rank21))

    candidates.sort(key=lambda item: item[0])
    used1 = set()
    used2 = set()
    rows = []
    for distance, i, j, rank12, rank21 in candidates:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        rows.append((i, j, distance, rank12, rank21))

    if not rows:
        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=torch.float32)
        return PairingResult(empty_long, empty_long, empty_float, empty_long, empty_long)

    idx_1, idx_2, distance, rank_1_to_2, rank_2_to_1 = zip(*rows)
    return PairingResult(
        idx_1=torch.tensor(idx_1, dtype=torch.long),
        idx_2=torch.tensor(idx_2, dtype=torch.long),
        distance=torch.tensor(distance, dtype=torch.float32),
        rank_1_to_2=torch.tensor(rank_1_to_2, dtype=torch.long),
        rank_2_to_1=torch.tensor(rank_2_to_1, dtype=torch.long),
    )


def standardized_mean_differences(
    x1: torch.Tensor,
    x2: torch.Tensor,
    idx1: torch.Tensor | None = None,
    idx2: torch.Tensor | None = None,
) -> torch.Tensor:
    if idx1 is not None:
        x1 = x1[idx1]
    if idx2 is not None:
        x2 = x2[idx2]
    if x1.numel() == 0 or x2.numel() == 0:
        return torch.empty(0)

    x1 = x1.float()
    x2 = x2.float()
    pooled = 0.5 * (x1.var(dim=0, unbiased=False) + x2.var(dim=0, unbiased=False))
    return (x1.mean(dim=0) - x2.mean(dim=0)).abs() / torch.sqrt(pooled + 1e-8)


def pair_table_dict(
    pairs: PairingResult,
    *,
    dataset_1: str,
    dataset_2: str,
    split: str,
    encoder_ckpt: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "idx_1": pairs.idx_1,
        "idx_2": pairs.idx_2,
        "distance": pairs.distance,
        "rank_1_to_2": pairs.rank_1_to_2,
        "rank_2_to_1": pairs.rank_2_to_1,
        "dataset_1": dataset_1,
        "dataset_2": dataset_2,
        "split": split,
        "encoder_ckpt": encoder_ckpt,
        "metadata": metadata or {},
    }


def _batch_to_device(batch, device):
    if isinstance(batch, (tuple, list)):
        return tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
    if isinstance(batch, dict):
        return {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
    return batch
