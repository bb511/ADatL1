"""The CAP bank: every registered CAP statistic of one model's scores on two sets.

A coupling of ``binary/pairing.py`` pairs the events of the two sets. Each side's scores
become evidence z, positive towards the anomalous end, through a map fitted on the
whole side (``std``, ``rank``, ``sq``) or, under ``disp``, as z = 2 (r(s) - r(s0)) / N,
the shift of every event's tied rank within its side from its rank under the run's
initial state; an event's energy gap is D = -z. The couplings that read the scores
(``cdf``, ``strat``) pair each map's evidence, so that ``disp`` is paired on its own
order; under the other maps, which increase with the score, those pairs carry the same
values as the scores' pairs. With the kernel k of ``posterior_agreement_curve``, which
puts a flat posterior at -log 2, the statistics put it at zero:

    A(beta)    = mean_pairs k + log 2, the agreement
    F(beta)    = mean_pairs (log|q1| + log|q2|) + log 2, the confidence, in [0, log 2)
    C(beta)    = A - F <= 0, the consistency, mean_pairs log cos(q1, q2)
    A_pi(beta) = A under the null, and C_pi = A_pi - F with the observed pairs' F
    A-sup = max of A over the beta grid; A-fix = A(beta_fix); D-fix = (A - A_pi)(beta_fix)
    A-lim = mean(z1 z2) and D-lim = corr(z1, z2) over the pairs: as beta -> 0,
            A -> (beta**2 / 4) mean(z1 z2) and A - A_pi -> (beta**2 / 4) cov(z1, z2)

The null permutes what the coupling did not read. For ``cdf`` it is the product
coupling of the two paired columns as they stand, so that A and A_pi share both
marginals: A_pi is the mean of k over every pair of the two columns, from ``null_bins``
quantile bins of each or exact, and sigma_pi the standard deviation of A over
permutations of the partner column. For ``ot`` and ``nn`` every draw relabels X2's
rows by one permutation, which the partners of every subsample follow, so that a row
shared by subsamples or repeated as a partner stays one row. The centre of a partner
column is the mean over it of g(z), the mean of k between z and the subsample's kept
X1 column, from ``null_bins`` quantile bins of that column or exact. A_pi is the centre
of the observed partners, and sigma_pi the standard deviation over the draws of A less
the centre of the drawn partners. For ``strat`` the draws permute X2's stratum labels,
and A_pi and sigma_pi are the mean and the standard deviation of A over the draws.
Under ``ot`` every quantity is the mean over the subsamples: A-sup is the supremum of
the mean curve, and sigma_pi the standard deviation over draws of the subsample mean.
Everything is computed in float64 on the CPU.
"""

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from .binary import get_energy_fn, get_normalizer_fn
from .binary.normalization import (
    disp_evidence,
    rank_evidence,
    sq_evidence,
    std_evidence,
)
from .binary.pairing import COUPLING_TYPES, COUPLINGS, Pairs
from .kernel import (
    indexed_agreement,
    log_norm,
    null_agreement_curve,
    permuted_agreement,
    posterior_agreement_curve,
    sample_agreement,
)

MAPS = {
    "std": std_evidence,
    "rank": rank_evidence,
    "sq": sq_evidence,
    "disp": disp_evidence,
}
# The maps that read the scores of the run's initial state beside the candidate's.
INITIAL_MAPS = ("disp",)
STATISTICS = ("A-sup", "A-fix", "A-lim", "D-fix", "D-lim")
# The pairings that match X1's rows to X2's own rows, and so read X2's blocks.
BLOCK_PAIRINGS = ("ot", "nn")
_LOG_2 = math.log(2.0)
_LEGACY_KEYS = ("legacy_A-sup", "legacy_beta_star", "legacy_at_bound")


def _check_config(config, initial_1, initial_2) -> None:
    unknown_maps = set(config["maps"]) - set(MAPS)
    unknown_statistics = set(config["statistics"]) - set(STATISTICS)
    if unknown_maps or unknown_statistics:
        unknown = sorted(unknown_maps | unknown_statistics)
        raise ValueError(f"Unknown evidence maps or statistics: {unknown}.")
    needing = [name for name in config["maps"] if name in INITIAL_MAPS]
    if needing and (initial_1 is None or initial_2 is None):
        raise ValueError(f"The maps {needing} need the initial scores of both sets.")


def _evidence(name: str, params, scores: Tensor, initial: Tensor | None) -> Tensor:
    if name in INITIAL_MAPS:
        return MAPS[name](scores, initial.detach().cpu(), **params)
    return MAPS[name](scores, **params)


def _betas(config) -> Tensor:
    # The grid of ApproximationCapacity, which the legacy corner reproduces.
    low, high = math.log10(config["beta_min"]), math.log10(config["beta_max"])
    return torch.logspace(low, high, config["n_beta"], dtype=torch.float64)


def _supremum(curve: Tensor, betas: Tensor) -> tuple[float, float, float]:
    """Return the maximum of curve, its beta, and 1.0 if that beta is beta_max.

    A non-finite maximum, from non-finite evidence, gives NaN for all three.
    """
    best = int(torch.argmax(curve))
    value = curve[best].item()
    if not math.isfinite(value):
        return math.nan, math.nan, math.nan
    return value, betas[best].item(), float(best == len(betas) - 1)


def _pearson(a: Tensor, b: Tensor) -> Tensor:
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / ((a * a).sum() * (b * b).sum()).sqrt()


def _binned(gap: Tensor, n_bins: int | None) -> tuple[Tensor, Tensor | None]:
    """Return the means and masses of n_bins contiguous quantile bins of gap.

    The sorted sample is cut into bins whose sizes differ by at most one. With n_bins
    None, or no more events than bins, the sample itself is returned, unweighted.
    """
    if n_bins is None or gap.numel() <= n_bins:
        return gap, None
    bins = torch.arange(gap.numel()) * n_bins // gap.numel()
    counts = torch.bincount(bins, minlength=n_bins).to(torch.float64)
    sums = torch.zeros(n_bins, dtype=torch.float64)
    sums.index_add_(0, bins, gap.sort().values)
    return sums / counts, counts / gap.numel()


def _paired_terms(a: Tensor, b: Tensor, beta: float, config) -> dict[str, Tensor]:
    """Return A and F at beta, A-lim, D-lim and, with A-sup, the A curve of z pairs."""
    at_beta = torch.tensor([beta], dtype=torch.float64)
    terms = {
        "A": posterior_agreement_curve(-a, -b, at_beta)[0] + _LOG_2,
        "F": (log_norm(-a, beta) + log_norm(-b, beta)).mean() + _LOG_2,
        "A-lim": (a * b).mean(),
        "D-lim": _pearson(a, b),
    }
    if "A-sup" in config["statistics"]:
        terms["curve"] = posterior_agreement_curve(-a, -b, _betas(config)) + _LOG_2
    return terms


def _product_null(
    a: Tensor, b: Tensor, beta: float, config, generator: torch.Generator
) -> dict[str, Tensor]:
    """Return A_pi over every pair of the columns a and b, and A of permuted draws."""
    (bins_a, mass_a), (bins_b, mass_b) = (
        _binned(-v, config["null_bins"]) for v in (a, b)
    )
    at_beta = torch.tensor([beta], dtype=torch.float64)
    a_pi = null_agreement_curve(bins_a, bins_b, at_beta, mass_a, mass_b)[0]
    draws = permuted_agreement(-a, -b, beta, config["n_permutations"], generator)
    return {"A_pi": a_pi + _LOG_2, "draws": draws + _LOG_2}


def _drawn_null(
    z_1: Tensor, z_2: Tensor, null: Pairs, beta: float
) -> dict[str, Tensor]:
    draws = indexed_agreement(-z_1[null.index_1], -z_2, null.index_2, beta) + _LOG_2
    return {"A_pi": draws.mean(), "draws": draws}


def _relabelled_null(
    z_1: Tensor, z_2: Tensor, pairs: Pairs, null: Pairs, beta: float, config
) -> dict[str, Tensor]:
    """Return A_pi and, per draw, A less the centre of its partners, for one subsample.

    null holds X2's relabelled rows of the pairs [n_draws, n_pairs]. The centre of a
    partner column is the mean of g [N2] over it, g the mean kernel of every X2 event
    against the kept X1 column, binned.
    """
    bins, mass = _binned(-z_1[pairs.index_1], config["null_bins"])
    g = sample_agreement(bins, -z_2, beta, mass)
    agreement = indexed_agreement(-z_1[null.index_1], -z_2, null.index_2, beta)
    draws = agreement - g[null.index_2].mean(dim=1)
    return {"A_pi": g[pairs.index_2].mean() + _LOG_2, "draws": draws}


def _statistics(mean: dict[str, Tensor], beta: float, config) -> dict[str, float]:
    """Return the registered statistics and the diagnostics of the subsample means."""
    a, a_pi, f = mean["A"], mean["A_pi"], mean["F"]
    values = {
        "A-fix": a,
        "A-lim": mean["A-lim"],
        "D-fix": a - a_pi,
        "D-lim": mean["D-lim"],
    }
    values = {k: float(v) for k, v in values.items() if k in config["statistics"]}
    if "curve" in mean:
        sup = _supremum(mean["curve"], _betas(config))
        values |= dict(zip(("A-sup", "beta_star", "at_bound"), sup))
    diagnostics = {"sigma_pi": mean["draws"].std(), "A_pi": a_pi, "C": a - f, "F": f}
    diagnostics["C_pi"] = a_pi - f
    return values | {k: float(v) for k, v in diagnostics.items()} | {"beta_fix": beta}


def _columns(z_1: Tensor, z_2: Tensor, pairs: list[Pairs]) -> list[tuple]:
    return [(z_1[p.index_1], z_2[p.index_2]) for p in pairs]


def _null_draws(coupling, z_1, z_2: Tensor, generator, config) -> list[Pairs]:
    """Return per subsample X1's indices and X2's of every null draw [n_draws, n_pairs].

    The draws of a subsample are taken to share X1's indices, as those of ot, nn and
    strat do.
    """
    draws = coupling.null_draws(z_1, z_2, config["n_permutations"], generator)
    return [
        Pairs(first.index_1, torch.stack([draw[k].index_2 for draw in draws]))
        for k, first in enumerate(draws[0])
    ]


def _relabelled_draws(coupling, scores_2: Tensor, config) -> list[Pairs] | None:
    """Return the null draws of ot and nn, which read no score and so serve every map;
    None for the other couplings."""
    if coupling.null != "relabel":
        return None
    generator = torch.Generator().manual_seed(config["permutation_seed"])
    return _null_draws(coupling, None, scores_2, generator, config)


def _nulls(coupling, z_1, z_2, pairs, relabelled, beta: float, config) -> list[dict]:
    """Return per subsample A_pi and the draws whose spread is sigma_pi."""
    if relabelled is not None:
        null = zip(pairs, relabelled)
        return [_relabelled_null(z_1, z_2, p, n, beta, config) for p, n in null]
    # Seeded anew for every map, so that every map sees the same permutations; one
    # generator serves every subsample of a product null, permuted independently.
    generator = torch.Generator().manual_seed(config["permutation_seed"])
    if coupling.null == "product":
        columns = _columns(z_1, z_2, pairs)
        return [_product_null(a, b, beta, config, generator) for a, b in columns]
    null = _null_draws(coupling, z_1, z_2, generator, config)
    return [_drawn_null(z_1, z_2, n, beta) for n in null]


def _map_statistics(z_1, z_2, coupling, relabelled, beta: float, config) -> dict:
    """Return the statistics of one evidence map, averaged over the subsamples.

    The pairs and the null draws are read from the evidence, so that cdf and strat pair
    every map on its own order; relabelled holds the draws of ot and nn.
    """
    pairs = coupling.pairs(z_1, z_2)
    columns = _columns(z_1, z_2, pairs)
    terms = [_paired_terms(a, b, beta, config) for a, b in columns]
    nulls = _nulls(coupling, z_1, z_2, pairs, relabelled, beta, config)
    terms = [t | n for t, n in zip(terms, nulls)]
    mean = {k: torch.stack([t[k] for t in terms]).mean(dim=0) for k in terms[0]}
    return _statistics(mean, float(beta), config)


def _legacy_gaps(paired_1: Tensor, paired_2: Tensor, legacy) -> tuple[Tensor, Tensor]:
    """Return the energy gaps that ApproximationCapacity.update computes.

    The scores are normalised in their own dtype, jointly unless the parameters say
    otherwise, and the energy reads the mean and the standard deviation of the pooled
    normalised scores, as in ApproximationCapacity._energy_gaps.
    """
    params = legacy["normalization_params"]
    normalize = get_normalizer_fn(legacy["normalization_type"], params)
    prob_1, prob_2 = (p.to(torch.float64) for p in normalize(paired_1, paired_2))
    pooled = torch.cat([prob_1, prob_2], dim=0)
    moments = {"mean": pooled.mean().item(), "std": pooled.std().item()}
    energy_params = dict(legacy["energy_params"] or {}) | moments
    energy = get_energy_fn(legacy["energy_type"], energy_params)
    ones, zeros = torch.ones_like(prob_1), torch.zeros_like(prob_1)
    gap_1 = energy(prob_1, ones) - energy(prob_1, zeros)
    return gap_1, energy(prob_2, ones) - energy(prob_2, zeros)


def _legacy_corner(scores_1: Tensor, scores_2: Tensor, pairs: Pairs, config) -> dict:
    """Return the NeurIPS CAP plus log 2, with its beta_star and at_bound."""
    paired_1, paired_2 = scores_1[pairs.index_1], scores_2[pairs.index_2]
    gap_1, gap_2 = _legacy_gaps(paired_1, paired_2, config["legacy"])
    curve = posterior_agreement_curve(gap_1, gap_2, _betas(config)) + _LOG_2
    return dict(zip(_LEGACY_KEYS, _supremum(curve, _betas(config))))


def _discarded(pairs: list[Pairs], n_1: int) -> float:
    return sum(1.0 - len(p.index_1) / n_1 for p in pairs) / len(pairs)


def _map_values(scores, initial, coupling, relabelled, config) -> dict[str, float]:
    """Return the keys of every registered map, from the scores and the initial state's
    scores of the two sides."""
    values = {}
    for name, params in config["maps"].items():
        z_1, z_2 = (_evidence(name, params, s, i) for s, i in zip(scores, initial))
        beta = config["beta_fix"][name]
        stats = _map_statistics(z_1, z_2, coupling, relabelled, beta, config)
        values |= {f"{name}_{key}": value for key, value in stats.items()}
    return values


def cap_bank(
    scores_1: Tensor,
    scores_2: Tensor,
    coupling,
    config,
    initial_1: Tensor | None = None,
    initial_2: Tensor | None = None,
) -> dict[str, float]:
    """Return every registered statistic of the two score samples under one coupling.

    The keys are ``<map>_<statistic>`` for every registered map and statistic;
    ``<map>_beta_star`` and ``<map>_at_bound`` (1.0 if the supremum sits at beta_max)
    with A-sup; ``<map>_sigma_pi``, ``<map>_A_pi``, ``<map>_C``, ``<map>_F`` and
    ``<map>_C_pi`` at the declared ``<map>_beta_fix``; ``discarded``, the fraction of
    X1's events left without a partner; and, for ``cdf`` with ``legacy`` set,
    ``legacy_A-sup`` (the NeurIPS CAP plus log 2), ``legacy_beta_star`` and
    ``legacy_at_bound``.

    Args:
        scores_1: Scores of X1 in the row order of the coupling's inputs [N1], in
            their original dtype, which the legacy corner reads
        scores_2: Scores of X2 [N2]
        coupling: A coupling of ``binary.pairing`` built on the inputs of X1 and X2
        config: The bank's configuration, as in ``configs/cap/bank.yaml``
        initial_1: Scores of X1 under the run's initial state [N1], which disp reads
        initial_2: Scores of X2 under the run's initial state [N2]

    Raises:
        ValueError: If an evidence map or a statistic is unknown, or if disp is
            registered without the initial scores of both sets.
    """
    _check_config(config, initial_1, initial_2)
    scores = scores_1.detach().cpu(), scores_2.detach().cpu()
    # Every map's pairs discard the same events; the legacy corner pairs the scores.
    pairs = coupling.pairs(*scores)
    relabelled = _relabelled_draws(coupling, scores[1], config)
    values = {"discarded": _discarded(pairs, len(scores[0]))}
    values |= _map_values(scores, (initial_1, initial_2), coupling, relabelled, config)
    if config.get("legacy") and coupling.name == "cdf":
        values |= _legacy_corner(*scores, pairs[0], config)
    return values


def _input_hash(
    name: str, params: dict, inputs: tuple[Tensor, ...], representation: dict | None
) -> str:
    """Return 16 hex digits of the SHA-256 of the pairing, its parameters, the
    representation unless it is None, and the inputs."""
    key = [name, params] if representation is None else [name, params, representation]
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode())
    for x in inputs:
        x = x.detach().cpu().contiguous()
        digest.update(f"{tuple(x.shape)} {x.dtype}".encode())
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()[:16]


def _atomic_write(path: Path, write: Callable) -> None:
    # os.replace is atomic, so that a concurrent worker never reads a partial file.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        write(handle)
    os.replace(handle.name, path)


def _stored_coupling(name: str, params: dict, inputs: tuple, path: Path, label):
    """Build a coupling, write it to path and its record beside it as JSON."""
    start = time.perf_counter()
    coupling = COUPLINGS[name](*inputs, **params)
    seconds = time.perf_counter() - start
    sizes = dict(zip(("n_1", "n_2", "n_ref"), (len(x) for x in inputs)))
    record = {"pairing": name, "parameters": params, "label": label, **sizes}
    record |= {"diagnostics": coupling.diagnostics(), "build_seconds": seconds}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2).encode()
    _atomic_write(path.with_suffix(".json"), lambda handle: handle.write(text))
    _atomic_write(path, lambda handle: torch.save(asdict(coupling), handle))
    return coupling


def _coupling(
    name: str, params: dict, x_1, x_2, x_ref, cache_dir, label, representation
):
    if name == "cdf":
        return COUPLINGS[name](**params)
    inputs = (x_1, x_2, x_ref) if name == "nn" else (x_1, x_2)
    if cache_dir is None:
        return COUPLINGS[name](*inputs, **params)
    digest = _input_hash(name, params, inputs, representation)
    path = Path(cache_dir) / f"{name}_{digest}.pt"
    if path.exists():
        return COUPLING_TYPES[name](**torch.load(path, weights_only=True))
    return _stored_coupling(name, params, inputs, path, label)


def _check_pairings(pairings, x_ref: Tensor | None) -> None:
    unknown = sorted(set(pairings) - set(COUPLINGS))
    if unknown:
        raise ValueError(f"Unknown pairings: {unknown}.")
    if "nn" in pairings and x_ref is None:
        raise ValueError("The nn coupling needs the caliper's reference x_ref.")


def _parameters(name: str, params, blocks: int | None) -> dict:
    params = dict(params or {})
    if blocks is not None and name in BLOCK_PAIRINGS:
        params["blocks"] = blocks
    return params


def build_couplings(
    x_1: Tensor,
    x_2: Tensor,
    x_ref: Tensor | None,
    pairings: dict,
    cache_dir: Path | None = None,
    label: dict | None = None,
    blocks: int | None = None,
    representation: dict | None = None,
) -> dict:
    """Return the registered couplings of X1 [N1, d] and X2 [N2, d], by pairing name.

    With a cache_dir, every coupling but cdf is read from
    ``<cache_dir>/<pairing>_<hash16>.pt`` if that exists, and otherwise built and
    written there, with ``<pairing>_<hash16>.json`` beside it holding the parameters,
    the diagnostics, N1, N2 (and N_ref for nn), the build time in seconds and `label`,
    which must be JSON-serialisable (the names of the sets, say). The hash is that of
    the input bytes, x_ref's for nn included, and of the pairing and its parameters.

    With `blocks`, X2's rows are read as that many equal consecutive blocks, row i of
    each the counterpart of X1's row i (the members of a pool of changed copies of
    X1): ot takes every block as one subsample, and ot and nn record per block the
    fraction of their pairs whose partner is X1's own row of the block. ``blocks``
    enters the parameters of the pairings of ``BLOCK_PAIRINGS``, and so their hash.
    `representation`, the name and parameters of the map that produced the inputs
    from the sets' own, enters every hash and the label, under 'representation'; None,
    for the identity, adds nothing to either.

    Raises:
        ValueError: If a pairing is unknown, nn is registered without x_ref, or blocks
            do not cut X2 into equal blocks of at least N1 rows.
    """
    _check_pairings(pairings, x_ref)
    if representation is not None:
        label = {**(label or {}), "representation": representation}
    params = {name: _parameters(name, p, blocks) for name, p in pairings.items()}
    return {
        name: _coupling(name, p, x_1, x_2, x_ref, cache_dir, label, representation)
        for name, p in params.items()
    }
