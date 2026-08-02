from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from src.utils.pairing.artifacts import FullPairingTensors

SearchBackend = Literal["auto", "faiss", "faiss_hnsw", "torch"]


@dataclass(frozen=True)
class CandidateNeighbors:
    """Top-k reference candidates for every target event."""

    squared_distance: torch.Tensor
    reference_index: torch.Tensor

    @property
    def n_target(self) -> int:
        """Return the number of target rows."""
        return int(self.reference_index.shape[0])

    @property
    def k(self) -> int:
        """Return the number of candidates per target."""
        return int(self.reference_index.shape[1])

    def validate(self, *, n_reference: int | None = None) -> None:
        """Validate candidate shapes, types, values, uniqueness, and bounds."""
        distance = self.squared_distance
        index = self.reference_index
        if distance.ndim != 2 or index.ndim != 2 or distance.shape != index.shape:
            raise ValueError("Candidate distances and indices must be equal-size matrices.")
        if distance.shape[0] == 0 or distance.shape[1] == 0:
            raise ValueError("Candidate matrices must be non-empty.")
        if not torch.is_floating_point(distance) or not torch.isfinite(distance).all():
            raise ValueError("Candidate distances must be finite floating-point values.")
        if torch.any(distance < 0):
            raise ValueError("Candidate squared distances must be non-negative.")
        if index.dtype != torch.long:
            raise ValueError("Candidate reference indices must be a LongTensor.")
        if torch.any(index < 0):
            raise ValueError("Candidate reference indices must be non-negative.")
        if n_reference is not None and torch.any(index >= int(n_reference)):
            raise ValueError("Candidate reference index is out of bounds.")
        if index.shape[1] > 1:
            repeated = torch.sort(index, dim=1).values
            if torch.any(repeated[:, 1:] == repeated[:, :-1]):
                raise ValueError("Each target's candidate reference indices must be unique.")


@torch.no_grad()
def exact_topk_l2(
    target: torch.Tensor,
    reference: torch.Tensor,
    *,
    k: int,
    backend: SearchBackend = "auto",
    query_batch_size: int = 1024,
    reference_batch_size: int = 32768,
) -> CandidateNeighbors:
    """Retrieve exact squared-L2 neighbors without a dense global distance matrix.

    FAISS uses its exhaustive ``IndexFlatL2`` index. The torch fallback searches
    reference chunks exhaustively and merges candidates with lexicographic
    ``(distance, reference_index)`` ordering. Both paths return CPU tensors.
    """
    target, reference = _validated_matrices(target, reference)
    if isinstance(k, bool) or int(k) <= 0:
        raise ValueError("k must be a positive integer.")
    if backend not in {"auto", "faiss", "faiss_hnsw", "torch"}:
        raise ValueError("backend must be one of: auto, faiss, faiss_hnsw, torch.")
    if int(query_batch_size) <= 0 or int(reference_batch_size) <= 0:
        raise ValueError("Search batch sizes must be positive.")
    k_eff = min(int(k), reference.shape[0])

    use_faiss = backend in {"faiss", "faiss_hnsw"} or (
        backend == "auto" and target.device.type == "cpu"
    )
    faiss = _try_import_faiss() if use_faiss else None
    if backend == "faiss" and faiss is None:
        raise ImportError("FAISS was requested but is not installed.")
    if faiss is not None:
        return _faiss_topk_l2(
            target,
            reference,
            k=k_eff,
            query_batch_size=int(query_batch_size),
            faiss=faiss,
            approximate=backend == "faiss_hnsw",
        )
    return _torch_topk_l2(
        target,
        reference,
        k=k_eff,
        query_batch_size=int(query_batch_size),
        reference_batch_size=int(reference_batch_size),
    )


def greedy_sparse_assignment(
    candidates: CandidateNeighbors,
    *,
    n_reference: int,
    caliper: float | None = None,
) -> FullPairingTensors:
    """Greedily assign candidate edges in deterministic global order.

    Edges are processed by ``(squared_distance, target_index,
    reference_index)``. The first edge whose endpoints are both unused is
    assigned. A caliper, when supplied, is in ordinary L2 distance units and
    only controls ``caliper_valid``; it never removes or changes an assignment.
    """
    candidates.validate(n_reference=n_reference)
    if isinstance(n_reference, bool) or int(n_reference) <= 0:
        raise ValueError("n_reference must be a positive integer.")
    if caliper is not None and (not torch.isfinite(torch.tensor(caliper)) or float(caliper) < 0):
        raise ValueError("caliper must be finite and non-negative.")

    n_target, k = candidates.reference_index.shape
    target_index = torch.arange(n_target).repeat_interleave(k)
    reference_index = candidates.reference_index.cpu().reshape(-1)
    squared_distance = candidates.squared_distance.float().cpu().reshape(-1)
    candidate_rank = torch.arange(1, k + 1).repeat(n_target)
    order = _lexicographic_edge_order(
        squared_distance,
        target_index,
        reference_index,
    )
    target_to_reference = torch.full((n_target,), -1, dtype=torch.long)
    reference_to_target = torch.full((int(n_reference),), -1, dtype=torch.long)
    distance = torch.full((n_target,), torch.inf, dtype=torch.float32)
    valid = torch.zeros(n_target, dtype=torch.bool)
    caliper_valid = torch.zeros(n_target, dtype=torch.bool)
    rank = torch.zeros(n_target, dtype=torch.long)

    # NumPy scalar access is substantially cheaper than repeated Tensor.item()
    # for the multi-million-edge candidate graphs used in production.
    ordered_target = target_index[order].numpy()
    ordered_reference = reference_index[order].numpy()
    ordered_distance = squared_distance[order].numpy()
    ordered_rank = candidate_rank[order].numpy()
    target_map = target_to_reference.numpy()
    inverse_map = reference_to_target.numpy()
    output_distance = distance.numpy()
    output_valid = valid.numpy()
    output_caliper_valid = caliper_valid.numpy()
    output_rank = rank.numpy()
    maximum_pairs = min(n_target, int(n_reference))
    accepted = 0
    for target_i, reference_i, distance_i, rank_i in zip(
        ordered_target,
        ordered_reference,
        ordered_distance,
        ordered_rank,
    ):
        if target_map[target_i] >= 0 or inverse_map[reference_i] >= 0:
            continue
        target_map[target_i] = reference_i
        inverse_map[reference_i] = target_i
        output_distance[target_i] = float(distance_i) ** 0.5
        output_valid[target_i] = True
        output_caliper_valid[target_i] = (
            caliper is None or float(distance_i) <= float(caliper) ** 2
        )
        output_rank[target_i] = rank_i
        accepted += 1
        if accepted == maximum_pairs:
            break

    result = FullPairingTensors(
        target_to_reference=target_to_reference,
        reference_to_target=reference_to_target,
        distance=distance,
        valid=valid,
        caliper_valid=caliper_valid,
        candidate_rank=rank,
    )
    result.validate()
    return result


@torch.no_grad()
def deterministic_one_to_one_match(
    target: torch.Tensor,
    reference: torch.Tensor,
    *,
    initial_k: int = 64,
    max_k: int | None = None,
    caliper: float | None = None,
    backend: SearchBackend = "auto",
    query_batch_size: int = 1024,
    reference_batch_size: int = 32768,
) -> tuple[FullPairingTensors, CandidateNeighbors]:
    """Retrieve and assign, growing k until full coverage or the search limit.

    The final candidate graph is returned so candidate-rank and sensitivity diagnostics can be
    reproduced without repeating the search.
    """
    target, reference = _validated_matrices(target, reference)
    if isinstance(initial_k, bool) or int(initial_k) <= 0:
        raise ValueError("initial_k must be a positive integer.")
    if max_k is not None and (isinstance(max_k, bool) or int(max_k) <= 0):
        raise ValueError("max_k must be a positive integer or None.")
    search_limit = reference.shape[0] if max_k is None else min(int(max_k), reference.shape[0])

    k = min(int(initial_k), search_limit)
    while True:
        candidates = exact_topk_l2(
            target,
            reference,
            k=k,
            backend=backend,
            query_batch_size=query_batch_size,
            reference_batch_size=reference_batch_size,
        )
        pairing = greedy_sparse_assignment(
            candidates,
            n_reference=reference.shape[0],
            caliper=caliper,
        )
        if pairing.n_pairs == min(target.shape[0], reference.shape[0]) or k == search_limit:
            return pairing, candidates
        k = min(2 * k, search_limit)


def _faiss_topk_l2(
    target, reference, *, k, query_batch_size, faiss, approximate=False
):
    """Retrieve deterministic CPU neighbors with flat L2 or an HNSW graph."""
    target = target.cpu()
    reference = reference.cpu()
    # HNSW insertion order changes with parallel scheduling. A single thread makes
    # independently generated pair tables bitwise reproducible.
    faiss.omp_set_num_threads(1)
    if approximate:
        index = faiss.IndexHNSWFlat(reference.shape[1], 16)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = max(128, 2 * int(k))
    else:
        index = faiss.IndexFlatL2(reference.shape[1])
    index.add(reference.numpy())
    all_distances = []
    all_indices = []
    for start in range(0, target.shape[0], query_batch_size):
        values, indices = index.search(target[start : start + query_batch_size].numpy(), k)
        distances = torch.from_numpy(values).float().clamp_min_(0)
        references = torch.from_numpy(indices).long()
        distances, references = _sort_rows_lexicographically(distances, references)
        all_distances.append(distances)
        all_indices.append(references)
    result = CandidateNeighbors(torch.cat(all_distances), torch.cat(all_indices))
    result.validate(n_reference=reference.shape[0])
    return result


def _torch_topk_l2(
    target: torch.Tensor,
    reference: torch.Tensor,
    *,
    k: int,
    query_batch_size: int,
    reference_batch_size: int,
) -> CandidateNeighbors:
    """Retrieve exact neighbors by merging exhaustive torch search chunks."""
    output_distances = []
    output_indices = []
    for query_start in range(0, target.shape[0], query_batch_size):
        query = target[query_start : query_start + query_batch_size]
        best_distance = torch.empty((query.shape[0], 0), dtype=torch.float32, device=query.device)
        best_index = torch.empty((query.shape[0], 0), dtype=torch.long, device=query.device)
        for ref_start in range(0, reference.shape[0], reference_batch_size):
            ref = reference[ref_start : ref_start + reference_batch_size]
            block_distance = (
                query.square().sum(dim=1, keepdim=True)
                + ref.square().sum(dim=1).unsqueeze(0)
                - 2.0 * (query @ ref.T)
            ).clamp_min_(0)
            block_k = min(k, ref.shape[0])
            block_index = torch.arange(
                ref_start,
                ref_start + ref.shape[0],
                device=query.device,
            ).expand(query.shape[0], -1)
            block_distance, block_index = _lexicographic_topk(
                block_distance,
                block_index,
                k=block_k,
                n_reference=reference.shape[0],
            )
            merged_distance = torch.cat((best_distance, block_distance), dim=1)
            merged_index = torch.cat((best_index, block_index), dim=1)
            if merged_distance.shape[1] > k:
                merged_distance, merged_index = _lexicographic_topk(
                    merged_distance,
                    merged_index,
                    k=k,
                    n_reference=reference.shape[0],
                )
            else:
                merged_distance, merged_index = _lexicographic_topk(
                    merged_distance,
                    merged_index,
                    k=merged_distance.shape[1],
                    n_reference=reference.shape[0],
                )
            best_distance = merged_distance
            best_index = merged_index
        output_distances.append(best_distance.cpu())
        output_indices.append(best_index.cpu())
    result = CandidateNeighbors(torch.cat(output_distances), torch.cat(output_indices))
    result.validate(n_reference=reference.shape[0])
    return result


def _sort_rows_lexicographically(
    distance: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort candidate rows by distance and then reference index."""
    # Stable least-significant-key sort followed by the primary-key sort.
    by_index = torch.argsort(index, dim=1, stable=True)
    distance = torch.gather(distance, 1, by_index)
    index = torch.gather(index, 1, by_index)
    by_distance = torch.argsort(distance, dim=1, stable=True)
    return torch.gather(distance, 1, by_distance), torch.gather(index, 1, by_distance)


def _lexicographic_topk(
    distance: torch.Tensor,
    index: torch.Tensor,
    *,
    k: int,
    n_reference: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k by exact float32 distance bits, then reference index.

    Non-negative IEEE-754 float bit patterns have the same ordering as their numeric values.
    Packing those bits and the reference index into an int64 gives torch.topk an exact
    lexicographic key without sorting every distance in a large reference block.
    """
    stride = int(n_reference) + 1
    distance_bits = distance.contiguous().view(torch.int32).long()
    key = distance_bits * stride + index
    _, keep = torch.topk(key, k=int(k), dim=1, largest=False, sorted=True)
    return torch.gather(distance, 1, keep), torch.gather(index, 1, keep)


def _lexicographic_edge_order(
    distance: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return edge order by distance, target index, and reference index."""
    order = torch.arange(distance.numel())
    order = order[torch.argsort(reference[order], stable=True)]
    order = order[torch.argsort(target[order], stable=True)]
    return order[torch.argsort(distance[order], stable=True)]


def _validated_matrices(
    target: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and co-locate target and reference descriptor matrices."""
    if not torch.is_tensor(target) or not torch.is_tensor(reference):
        raise TypeError("Target and reference descriptors must be torch tensors.")
    if target.ndim != 2 or reference.ndim != 2:
        raise ValueError("Target and reference descriptors must be two-dimensional.")
    if target.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("Target and reference descriptors must be non-empty.")
    if target.shape[1] != reference.shape[1]:
        raise ValueError("Target and reference descriptors must have equal dimensions.")
    target = target.detach().contiguous().float()
    reference = reference.detach().to(target.device).contiguous().float()
    if not torch.isfinite(target).all() or not torch.isfinite(reference).all():
        raise ValueError("Target and reference descriptors must contain finite values.")
    return target, reference


def _try_import_faiss():
    """Return the optional FAISS module when it is available."""
    try:
        import faiss

        return faiss
    except ImportError:
        return None
