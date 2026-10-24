from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

# Row blocks of the distance matrices hold at most this many float64 elements (16 MB).
_BLOCK_ELEMENTS = 2**21


def _get_small_large(tensor1: Tensor, tensor2: Tensor) -> Tuple[Tensor, Tensor]:
    """Helper function to return tensors in (small, large) order."""
    if len(tensor1) <= len(tensor2):
        return tensor1, tensor2, False
    else:
        return tensor2, tensor1, True


def cdf(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Empirical CDF pairing: matches samples based on their percentile ranks.

    Rationale: Preserves distributional relationships by pairing samples at similar
    percentile ranks in their respective distributions.
    """
    small_tensor, large_tensor, swapped = _get_small_large(tensor1, tensor2)

    # Get sorted indices (ranks) for both tensors
    small_sorted_indices = torch.argsort(small_tensor)
    large_sorted_indices = torch.argsort(large_tensor)

    # For each position in small tensor, find corresponding percentile in large tensor
    small_size = len(small_tensor)
    large_size = len(large_tensor)

    # Map each rank in small tensor to corresponding rank in large tensor
    large_ranks = torch.round(
        (torch.arange(small_size, dtype=torch.float) / (small_size - 1))
        * (large_size - 1)
    ).long()

    # Handle edge case for single element
    if small_size == 1:
        large_ranks = torch.tensor([large_size // 2])  # Use median

    # Get the actual indices
    small_indices = small_sorted_indices
    large_indices = large_sorted_indices[large_ranks]

    if not swapped:
        return small_indices, large_indices
    return large_indices, small_indices


class Pairs(NamedTuple):
    """Row indices of the matched events of the two sides, [n_pairs] each."""

    index_1: Tensor
    index_2: Tensor


def _permutations(n: int, size: int, generator: torch.Generator) -> Tensor:
    # One randperm per row: three to five times faster than the argsort of uniforms.
    return torch.stack([torch.randperm(size, generator=generator) for _ in range(n)])


def _permuted_partners(
    pairs: list[Pairs], n: int, generator: torch.Generator
) -> list[list[Pairs]]:
    """Return n draws of the pairs, each subsample's partner column permuted anew."""
    perms = [_permutations(n, len(p.index_2), generator) for p in pairs]
    return [
        [Pairs(p.index_1, p.index_2[perm[draw]]) for p, perm in zip(pairs, perms)]
        for draw in range(n)
    ]


def _relabelled_partners(
    pairs: list[Pairs], n_2: int, n: int, generator: torch.Generator
) -> list[list[Pairs]]:
    """Return n draws of the pairs, each relabelling X2's n_2 rows by one permutation.

    The draw's permutation relabels the partners of every subsample, so that a row that
    several subsamples share, or that several X1 events take as partner, stays one row.
    """
    perms = _permutations(n, n_2, generator)
    partners = [perms[:, p.index_2] for p in pairs]
    return [
        [Pairs(p.index_1, rows[draw]) for p, rows in zip(pairs, partners)]
        for draw in range(n)
    ]


def _distances(x: Tensor, y: Tensor) -> Tensor:
    # Exact differences: the matrix-product form loses digits to cancellation.
    return torch.cdist(
        x.to(torch.float64),
        y.to(torch.float64),
        compute_mode="donot_use_mm_for_euclid_dist",
    )


def _nearest(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Return the distance to and the index of the nearest row of y, per row of x."""
    rows = max(1, _BLOCK_ELEMENTS // max(1, len(y)))
    blocks = [_distances(block, y).min(dim=1) for block in x.split(rows)]
    return torch.cat([b.values for b in blocks]), torch.cat([b.indices for b in blocks])


def _block_size(n_1: int, n_2: int, blocks: int) -> int:
    if n_2 % blocks or n_2 // blocks < n_1:
        raise ValueError(
            f"blocks = {blocks} must cut N2 = {n_2} into equal blocks of at least "
            f"N1 = {n_1} rows."
        )
    return n_2 // blocks


def _identity_fraction(
    index_1: Tensor, index_2: Tensor, n_1: int, n_2: int, blocks: int | None
) -> list[float] | None:
    """Return per block of X2 the fraction of the pairs whose partner is X1's own row.

    X2's rows are read as `blocks` equal consecutive blocks, row i of each the
    counterpart of X1's row i. index_1 and index_2 are [..., n_pairs], and every
    fraction is over n_pairs: the pairs of one ot subsample, which lie in one block, or
    all of nn's pairs, which spread over every block. None without blocks.
    """
    if blocks is None:
        return None
    size = _block_size(n_1, n_2, blocks)
    own = index_2 % size == index_1
    counts = torch.bincount(index_2[own] // size, minlength=blocks)
    return (counts.to(torch.float64) / index_1.shape[-1]).tolist()


def _block_diagnostics(identity_fraction: list[float] | None) -> dict:
    if identity_fraction is None:
        return {}
    return {"blocks": len(identity_fraction), "identity_fraction": identity_fraction}


# A coupling's `null` names the permutation null of bank.cap_bank: "product", every
# pair of the two paired columns; "relabel", X2's rows permuted under the pairs;
# "draws", the pairs of the coupling's null draws.
@dataclass
class CDFCoupling:
    """Rank pairing of the two score samples, the coupling of the NeurIPS CAP."""

    name: ClassVar[str] = "cdf"
    null: ClassVar[str] = "product"

    def pairs(self, scores_1: Tensor, scores_2: Tensor) -> list[Pairs]:
        return [Pairs(*cdf(scores_1, scores_2))]

    def null_draws(
        self, scores_1: Tensor, scores_2: Tensor, n: int, generator: torch.Generator
    ) -> list[list[Pairs]]:
        return _permuted_partners(self.pairs(scores_1, scores_2), n, generator)

    def diagnostics(self) -> dict:
        """Return nothing: what cdf discards depends on the sizes of the scores."""
        return {}


@dataclass
class OTCoupling:
    """Exact optimal-transport assignments of X1 to subsamples of X2.

    Row k of index_1 and index_2 [n_subsamples, n_kept] holds the pairs kept from
    subsample k; mean_distance is the mean Euclidean distance of the kept pairs. With
    X2 cut into blocks, identity_fraction holds per block the fraction of the kept
    pairs whose partner is X1's own row of the block (``ot_coupling``); None otherwise.
    """

    index_1: Tensor
    index_2: Tensor
    n_1: int
    mean_distance: float
    identity_fraction: list[float] | None = None
    name: ClassVar[str] = "ot"
    null: ClassVar[str] = "relabel"

    def pairs(self, scores_1: Tensor, scores_2: Tensor) -> list[Pairs]:
        return [Pairs(i_1, i_2) for i_1, i_2 in zip(self.index_1, self.index_2)]

    def null_draws(
        self, scores_1: Tensor, scores_2: Tensor, n: int, generator: torch.Generator
    ) -> list[list[Pairs]]:
        pairs = self.pairs(scores_1, scores_2)
        return _relabelled_partners(pairs, len(scores_2), n, generator)

    def diagnostics(self) -> dict:
        discarded = 1.0 - self.index_1.shape[1] / self.n_1
        values = {"discarded": discarded, "mean_distance": self.mean_distance}
        return values | _block_diagnostics(self.identity_fraction)


def _ot_subsample(
    x_1: Tensor, x_2: Tensor, subsample: Tensor, n_kept: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the n_kept cheapest pairs of the assignment of x_1 to x_2[subsample].

    The cost is the squared Euclidean distance; the pairs are returned in X1's row
    order, with their distances.
    """
    cost = _distances(x_1, x_2[subsample]).square()
    rows, cols = map(torch.from_numpy, linear_sum_assignment(cost.numpy()))
    pair_cost = cost[rows, cols]
    keep = torch.argsort(pair_cost, stable=True)[:n_kept].sort().values
    return rows[keep], subsample[cols[keep]], pair_cost[keep].sqrt()


def _subsamples(
    n_1: int, n_2: int, n_subsamples: int, seed: int, blocks: int | None
) -> list[Tensor]:
    """Return X2's rows of every subsample: n_subsamples random draws of N1 rows from
    one generator seeded once with `seed`, or, with blocks, X2's blocks."""
    if blocks is not None:
        return list(torch.arange(n_2).split(_block_size(n_1, n_2, blocks)))
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(n_2, generator=generator)[:n_1] for _ in range(n_subsamples)]


def ot_coupling(
    x_1: Tensor,
    x_2: Tensor,
    n_subsamples: int = 5,
    seed: int = 0,
    discard: float = 0.05,
    blocks: int | None = None,
) -> OTCoupling:
    """Assign X1 [N1, d] exactly to subsamples of X2 [N2, d].

    The subsamples are n_subsamples random draws of N1 rows or, with `blocks`, X2's
    rows cut into that many equal consecutive blocks, which n_subsamples and seed then
    leave unused: a pool whose members are concatenated in order, each a copy of X1
    under one change, gives one assignment per member. A block longer than N1 takes
    a rectangular assignment, in which every X1 row has a partner. In each subsample
    the round(discard * N1) pairs of largest cost are dropped. With blocks, the
    coupling records per block the fraction of the kept pairs whose partner is X1's
    own row of the block, row i of X1 with row i of the block.

    Raises:
        ValueError: If X2 has fewer rows than X1, or blocks do not cut X2 into equal
            blocks of at least N1 rows.
    """
    if len(x_2) < len(x_1):
        raise ValueError(f"ot needs N2 >= N1; got N1 = {len(x_1)}, N2 = {len(x_2)}.")
    n_kept = len(x_1) - round(discard * len(x_1))
    subsamples = _subsamples(len(x_1), len(x_2), n_subsamples, seed, blocks)
    index_1, index_2, distance = zip(
        *(_ot_subsample(x_1, x_2, s, n_kept) for s in subsamples)
    )
    index_1, index_2 = torch.stack(index_1), torch.stack(index_2)
    fraction = _identity_fraction(index_1, index_2, len(x_1), len(x_2), blocks)
    mean_distance = torch.stack(distance).mean().item()
    return OTCoupling(index_1, index_2, len(x_1), mean_distance, fraction)


@dataclass
class NNCoupling:
    """Nearest X2 neighbour of every X1 event within the caliper; partners may repeat.

    caliper_distance is the largest Euclidean distance at which a pair is kept, and
    mean_distance the mean distance of the kept pairs. With X2 cut into blocks,
    identity_fraction holds per block the fraction of the kept pairs whose partner is
    X1's own row of that block (``nn_coupling``); None otherwise.
    """

    index_1: Tensor
    index_2: Tensor
    n_1: int
    caliper_distance: float
    mean_distance: float
    identity_fraction: list[float] | None = None
    name: ClassVar[str] = "nn"
    null: ClassVar[str] = "relabel"

    def pairs(self, scores_1: Tensor, scores_2: Tensor) -> list[Pairs]:
        return [Pairs(self.index_1, self.index_2)]

    def null_draws(
        self, scores_1: Tensor, scores_2: Tensor, n: int, generator: torch.Generator
    ) -> list[list[Pairs]]:
        pairs = self.pairs(scores_1, scores_2)
        return _relabelled_partners(pairs, len(scores_2), n, generator)

    def diagnostics(self) -> dict:
        values = {
            "discarded": 1.0 - len(self.index_1) / self.n_1,
            "mean_distance": self.mean_distance,
            "caliper": self.caliper_distance,
        }
        return values | _block_diagnostics(self.identity_fraction)


def nn_coupling(
    x_1: Tensor,
    x_2: Tensor,
    x_ref: Tensor,
    caliper: float = 0.95,
    blocks: int | None = None,
) -> NNCoupling:
    """Pair every X1 row with its nearest X2 row, if no farther than the caliper.

    The caliper distance is the `caliper` quantile of the nearest-neighbour distances
    from X1 to the reference sample x_ref. With `blocks`, X2's rows are read as that
    many equal consecutive blocks, and the coupling records per block the fraction of
    the kept pairs whose partner is X1's own row of the block; the pairs are the same.

    Raises:
        ValueError: If blocks do not cut X2 into equal blocks of at least N1 rows.
    """
    distance, partner = _nearest(x_1, x_2)
    limit = torch.quantile(_nearest(x_1, x_ref)[0], caliper).item()
    kept = torch.nonzero(distance <= limit).flatten()
    mean_distance = distance[kept].mean().item()
    fraction = _identity_fraction(kept, partner[kept], len(x_1), len(x_2), blocks)
    return NNCoupling(kept, partner[kept], len(x_1), limit, mean_distance, fraction)


def _stratified_order(labels: Tensor, scores: Tensor) -> Tensor:
    """Return, per row of labels [n_draws, N], the indices by stratum, then by score."""
    by_score = torch.argsort(scores, stable=True)
    return by_score[torch.argsort(labels[:, by_score], dim=1, stable=True)]


def _rank_matches(n_1: int, n_2: int) -> tuple[Tensor, Tensor]:
    """Return the rank positions that cdf pairs between n_1 and n_2 events."""
    n_small, n_large = min(n_1, n_2), max(n_1, n_2)
    small = torch.arange(n_small)
    # cdf's own float32 arithmetic and single-event rule, so that the positions are
    # those cdf picks.
    large = torch.round(
        (torch.arange(n_small, dtype=torch.float) / (n_small - 1)) * (n_large - 1)
    ).long()
    if n_small == 1:
        large = torch.tensor([n_large // 2])
    return (small, large) if n_1 <= n_2 else (large, small)


def _stratum_positions(counts_1: Tensor, counts_2: Tensor) -> tuple[Tensor, Tensor]:
    """Return the positions, in each side's stratified order, of the pairs made."""
    offsets_1 = (torch.cumsum(counts_1, dim=0) - counts_1).tolist()
    offsets_2 = (torch.cumsum(counts_2, dim=0) - counts_2).tolist()
    matches = [_rank_matches(*c) for c in zip(counts_1.tolist(), counts_2.tolist())]
    positions_1 = [m[0] + offset for m, offset in zip(matches, offsets_1)]
    positions_2 = [m[1] + offset for m, offset in zip(matches, offsets_2)]
    return torch.cat(positions_1), torch.cat(positions_2)


@dataclass
class StratCoupling:
    """Rank pairing of the scores within strata of a score-blind proxy.

    strata_1 [N1] and strata_2 [N2] hold every event's stratum: the n_strata-quantile of
    its Mahalanobis distance under its own side's mean and covariance.
    """

    strata_1: Tensor
    strata_2: Tensor
    n_strata: int
    name: ClassVar[str] = "strat"
    null: ClassVar[str] = "draws"

    def pairs(self, scores_1: Tensor, scores_2: Tensor) -> list[Pairs]:
        index_1, index_2 = self._paired(self.strata_2[None], scores_1, scores_2)
        return [Pairs(index_1, index_2[0])]

    def null_draws(
        self, scores_1: Tensor, scores_2: Tensor, n: int, generator: torch.Generator
    ) -> list[list[Pairs]]:
        """Return n draws of the pairs with X2's stratum labels permuted, sizes kept."""
        labels = self.strata_2[_permutations(n, len(self.strata_2), generator)]
        index_1, index_2 = self._paired(labels, scores_1, scores_2)
        return [[Pairs(index_1, row)] for row in index_2]

    def diagnostics(self) -> dict:
        counts_1, counts_2 = self._counts()
        paired = torch.minimum(counts_1, counts_2).sum().item()
        return {
            "discarded": 1.0 - paired / len(self.strata_1),
            "counts_1": counts_1.tolist(),
            "counts_2": counts_2.tolist(),
        }

    def _counts(self) -> tuple[Tensor, Tensor]:
        return (
            torch.bincount(self.strata_1, minlength=self.n_strata),
            torch.bincount(self.strata_2, minlength=self.n_strata),
        )

    def _paired(
        self, labels_2: Tensor, scores_1: Tensor, scores_2: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return X1's indices [n_pairs] and X2's [n_draws, n_pairs] of every pair.

        Row r of X2's indices pairs under the stratum labels of row r of labels_2
        [n_draws, N2]. The pair positions depend on the stratum counts alone, which
        every row shares, so that one gather pairs every draw.
        """
        positions_1, positions_2 = _stratum_positions(*self._counts())
        order_1 = _stratified_order(self.strata_1[None], scores_1)[0]
        order_2 = _stratified_order(labels_2, scores_2)
        return order_1[positions_1], order_2[:, positions_2]


def _strata(x: Tensor, n_strata: int) -> Tensor:
    """Return every row's stratum, equal counts up to one per stratum.

    The proxy is the squared Mahalanobis distance under the sample's own mean and
    covariance, whose pseudo-inverse keeps a constant readout from failing it.
    """
    x = x.to(torch.float64)
    centred = x - x.mean(dim=0)
    precision = torch.linalg.pinv(centred.T @ centred / (len(x) - 1), hermitian=True)
    distance = ((centred @ precision) * centred).sum(dim=1)
    rank = torch.empty(len(x), dtype=torch.long)
    rank[torch.argsort(distance, stable=True)] = torch.arange(len(x))
    return rank * n_strata // len(x)


def strat_coupling(x_1: Tensor, x_2: Tensor, n_strata: int = 10) -> StratCoupling:
    return StratCoupling(_strata(x_1, n_strata), _strata(x_2, n_strata), n_strata)


def cdf_coupling() -> CDFCoupling:
    return CDFCoupling()


# The builders; a coupling is rebuilt from its saved fields as
# COUPLING_TYPES[name](**dataclasses.asdict(coupling)).
COUPLINGS = {
    "cdf": cdf_coupling,
    "ot": ot_coupling,
    "nn": nn_coupling,
    "strat": strat_coupling,
}
COUPLING_TYPES = {
    cls.name: cls for cls in (CDFCoupling, OTCoupling, NNCoupling, StratCoupling)
}
