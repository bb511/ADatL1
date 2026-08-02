from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.data.utils import unpack_batch
from src.utils.pairing.table import PAIR_TABLE_SCHEMA_VERSION


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
        rep = encode_batch(model, batch)
        x = torch.flatten(b.x, start_dim=1)
        reps.append(rep.detach().cpu())
        xs.append(x.detach().cpu())
        ys.append(b.y.detach().cpu())
        if max_events is not None and sum(t.shape[0] for t in reps) >= max_events:
            break

    if not reps:
        raise ValueError("Cannot collect representations from an empty dataloader.")
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
        try:
            x1, x2, m1, m2 = model.augment_pair(x, m, return_masks=True)
        except TypeError:
            x1, x2 = model.augment_pair(x)
            m1 = m2 = m
        reps1.append(model.encode_flat(x1, m1).detach().cpu())
        reps2.append(model.encode_flat(x2, m2).detach().cpu())
        if max_events is not None and sum(t.shape[0] for t in reps1) >= max_events:
            break

    if not reps1:
        raise ValueError("Cannot collect closure representations from an empty dataloader.")
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
    chunk_size: int = 1024,
) -> dict[str, float]:
    _validate_embedding_pair(z1, z2, require_equal_rows=True)
    if not ks or any(int(k) <= 0 for k in ks):
        raise ValueError("Closure recall values k must be positive.")
    if int(chunk_size) <= 0:
        raise ValueError("Closure chunk_size must be positive.")
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    max_k = min(max(int(k) for k in ks), z2.shape[0])
    hits = {int(k): 0 for k in ks}
    true_ranks = []
    positive_distances = []

    # Computing the full N x N matrix is prohibitive for real L1 validation sets.
    # Chunking over queries keeps the exact retrieval metrics while bounding memory by
    # chunk_size x N. The positive rank is one plus the number of strictly closer
    # candidates; exact ties are extraordinarily unlikely for continuous embeddings.
    for start in range(0, z1.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), z1.shape[0])
        sim = z1[start:stop] @ z2.T
        labels = torch.arange(start, stop, device=sim.device)
        positive = sim[torch.arange(stop - start, device=sim.device), labels]
        topk = torch.topk(sim, k=max_k, dim=1).indices
        for k in hits:
            k_eff = min(k, max_k)
            hits[k] += int((topk[:, :k_eff] == labels[:, None]).any(dim=1).sum())
        true_ranks.append(1 + (sim > positive[:, None]).sum(dim=1).cpu())
        positive_distances.append((1.0 - positive).cpu())

    metrics = {f"closure_recall_at_{k}": hits[int(k)] / z1.shape[0] for k in ks}
    ranks = torch.cat(true_ranks)
    metrics["closure_median_rank"] = ranks.float().median().item()
    metrics["closure_mean_pos_distance"] = torch.cat(positive_distances).mean().item()
    return metrics


def mutual_nearest_pairs(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int = 20,
    caliper: float | None = None,
) -> PairingResult:
    _validate_embedding_pair(z1, z2)
    if int(k) <= 0:
        raise ValueError("k must be positive for mutual-nearest pairing.")
    if caliper is not None and (not torch.isfinite(torch.tensor(caliper)) or caliper < 0):
        raise ValueError("caliper must be finite and non-negative.")
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    k12 = min(int(k), z2.shape[0])
    k21 = min(int(k), z1.shape[0])

    val12, idx12 = topk_inner_product(z1, z2, k=k12)
    _, idx21 = topk_inner_product(z2, z1, k=k21)

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


def one_to_one_nearest_pairs(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int | None = None,
    caliper: float | None = None,
    normalize: bool = False,
) -> PairingResult:
    """Greedy one-to-one nearest-neighbor pairing.

    Uses FAISS when available and falls back to torch. Unlike
    ``mutual_nearest_pairs``, this does not require mutual neighbors and is useful
    when the experiment needs broad coverage, e.g. metadata-nearest controls.
    """

    _validate_embedding_pair(z1, z2)
    if k is not None and int(k) <= 0:
        raise ValueError("k must be positive or None for one-to-one pairing.")
    if caliper is not None and (not torch.isfinite(torch.tensor(caliper)) or caliper < 0):
        raise ValueError("caliper must be finite and non-negative.")
    z1 = z1.float()
    z2 = z2.float()
    if normalize:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

    grow_k = k is None
    k_eff = min(64 if grow_k else int(k), z2.shape[0])

    while True:
        rows = _greedy_l2_rows(z1, z2, k=k_eff, caliper=caliper)
        if not grow_k or len(rows) >= min(z1.shape[0], z2.shape[0]) or k_eff == z2.shape[0]:
            break
        k_eff = min(k_eff * 2, z2.shape[0])

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


def _greedy_l2_rows(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    k: int,
    caliper: float | None,
) -> list[tuple[int, int, float, int, int]]:
    distances, indices = topk_l2(z1, z2, k=k)
    candidates = []
    for i in range(indices.shape[0]):
        for rank, j in enumerate(indices[i].tolist()):
            distance = float(distances[i, rank].item())
            if caliper is not None and distance > caliper:
                continue
            candidates.append((distance, i, j, rank + 1))

    candidates.sort(key=lambda item: item[0])
    used1 = set()
    used2 = set()
    rows = []
    for distance, i, j, rank12 in candidates:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        rows.append((i, j, distance, rank12, 0))
    return rows


def topk_inner_product(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    faiss = _try_import_faiss()
    x_cpu = x.detach().cpu().contiguous().float()
    y_cpu = y.detach().cpu().contiguous().float()
    k = min(int(k), y_cpu.shape[0])

    if faiss is not None:
        index = faiss.IndexFlatIP(y_cpu.shape[1])
        index.add(y_cpu.numpy())
        values, indices = index.search(x_cpu.numpy(), k)
        return torch.from_numpy(values), torch.from_numpy(indices).long()

    sim = x_cpu @ y_cpu.T
    return torch.topk(sim, k=k, dim=1)


def topk_l2(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    faiss = _try_import_faiss()
    x_cpu = x.detach().cpu().contiguous().float()
    y_cpu = y.detach().cpu().contiguous().float()
    k = min(int(k), y_cpu.shape[0])

    if faiss is not None:
        index = faiss.IndexFlatL2(y_cpu.shape[1])
        index.add(y_cpu.numpy())
        distances, indices = index.search(x_cpu.numpy(), k)
        return torch.from_numpy(distances), torch.from_numpy(indices).long()

    distances = torch.cdist(x_cpu, y_cpu).pow(2)
    return torch.topk(distances, k=k, dim=1, largest=False)


def encode_batch(model, batch) -> torch.Tensor:
    if hasattr(model, "encode_batch"):
        return model.encode_batch(batch)

    b = unpack_batch(batch)
    x = torch.flatten(b.x, start_dim=1)
    if hasattr(model, "encode_flat"):
        mask = None
        if b.mask is not None:
            mask = torch.flatten(b.mask, start_dim=1).float()
        return model.encode_flat(x, mask)
    if hasattr(model, "encoder"):
        features = getattr(model, "features", torch.nn.Identity())
        return model.encoder(features(x))
    if hasattr(model, "model"):
        try:
            return model.model(x)
        except TypeError:
            mask = None if b.mask is None else torch.flatten(b.mask, start_dim=1).float()
            return model.model(x, mask)
    raise TypeError(f"Model {type(model).__name__} does not expose an encoder.")


def _try_import_faiss():
    try:
        import faiss

        return faiss
    except Exception:
        return None


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
        "schema_version": PAIR_TABLE_SCHEMA_VERSION,
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


def _validate_embedding_pair(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    require_equal_rows: bool = False,
) -> None:
    if not torch.is_tensor(z1) or not torch.is_tensor(z2):
        raise TypeError("Pairing inputs must be torch tensors.")
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("Pairing inputs must be two-dimensional embedding matrices.")
    if z1.shape[0] == 0 or z2.shape[0] == 0:
        raise ValueError("Pairing inputs must contain at least one embedding.")
    if z1.shape[1] != z2.shape[1]:
        raise ValueError("Pairing inputs must have the same embedding dimension.")
    if require_equal_rows and z1.shape[0] != z2.shape[0]:
        raise ValueError("Closure views must contain the same number of embeddings.")
    if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
        raise ValueError("Pairing inputs must contain only finite embeddings.")


def _batch_to_device(batch, device):
    if isinstance(batch, (tuple, list)):
        return tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    return batch
