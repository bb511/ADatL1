"""The CAP bank: every registered CAP statistic of one model's scores on two sets.

A coupling of ``binary/pairing.py`` pairs the events of the two sets. Each side's scores
become evidence z, positive towards the anomalous end, through a map fitted on the
whole side (``std``, ``rank``, ``sq``), and an event's energy gap is D = -z. With the
kernel k of ``posterior_agreement_curve``, which puts a flat posterior at -log 2, the
statistics put it at zero:

    A(beta)    = mean_pairs k + log 2, the agreement
    F(beta)    = mean_pairs (log|q1| + log|q2|) + log 2, the confidence, in [0, log 2)
    C(beta)    = A - F <= 0, the consistency, mean_pairs log cos(q1, q2)
    A_pi(beta) = A under the null, and C_pi = A_pi - F with the observed pairs' F
    A-sup = max of A over the beta grid; A-fix = A(beta_fix); D-fix = (A - A_pi)(beta_fix)
    A-lim = mean(z1 z2) and D-lim = corr(z1, z2) over the pairs: as beta -> 0,
            A -> (beta**2 / 4) mean(z1 z2) and A - A_pi -> (beta**2 / 4) cov(z1, z2)

The null permutes what the coupling did not read. For ``cdf``, ``ot`` and ``nn`` it is
the product coupling of the two paired columns as they stand, repeats of ``nn``
included, so that A and A_pi share both marginals: A_pi is the mean of k over every
pair of the two columns, from ``null_bins`` quantile bins of each or exact, and
sigma_pi the standard deviation of A over permutations of the partner column. For
``strat`` the draws permute X2's stratum labels, and A_pi and sigma_pi are the mean and
the standard deviation over the draws. Under ``ot`` every quantity is the mean over the
subsamples: A-sup is the supremum of the mean curve, and sigma_pi the standard
deviation over draws of the subsample mean. Everything is computed in float64 on the
CPU.
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
from .binary.normalization import rank_evidence, sq_evidence, std_evidence
from .binary.pairing import COUPLING_TYPES, COUPLINGS, Pairs
from .kernel import (
    indexed_agreement,
    log_norm,
    null_agreement_curve,
    permuted_agreement,
    posterior_agreement_curve,
)

MAPS = {"std": std_evidence, "rank": rank_evidence, "sq": sq_evidence}
STATISTICS = ("A-sup", "A-fix", "A-lim", "D-fix", "D-lim")
_LOG_2 = math.log(2.0)
_LEGACY_KEYS = ("legacy_A-sup", "legacy_beta_star", "legacy_at_bound")


def _check_config(config) -> None:
    unknown_maps = set(config["maps"]) - set(MAPS)
    unknown_statistics = set(config["statistics"]) - set(STATISTICS)
    if unknown_maps or unknown_statistics:
        unknown = sorted(unknown_maps | unknown_statistics)
        raise ValueError(f"Unknown evidence maps or statistics: {unknown}.")


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


def _map_statistics(
    z_1: Tensor,
    z_2: Tensor,
    pairs: list[Pairs],
    null: list[Pairs] | None,
    beta: float,
    config,
) -> dict[str, float]:
    """Return the statistics of one evidence map, averaged over the subsamples."""
    # One generator for every subsample, so that each is permuted independently, and
    # seeded anew for every map, so that every map sees the same permutations.
    generator = torch.Generator().manual_seed(config["permutation_seed"])
    columns = [(z_1[p.index_1], z_2[p.index_2]) for p in pairs]
    terms = [_paired_terms(a, b, beta, config) for a, b in columns]
    if null is None:
        nulls = [_product_null(a, b, beta, config, generator) for a, b in columns]
    else:
        nulls = [_drawn_null(z_1, z_2, n, beta) for n in null]
    terms = [t | n for t, n in zip(terms, nulls)]
    mean = {k: torch.stack([t[k] for t in terms]).mean(dim=0) for k in terms[0]}
    return _statistics(mean, float(beta), config)


def _null_draws(coupling, scores_1: Tensor, scores_2: Tensor, config) -> list | None:
    """Return per subsample X1's indices and X2's of every null draw [n_draws, n_pairs].

    None for a product null, which is computed from the paired columns. The draws of a
    subsample are taken to share X1's indices, as those of strat do.
    """
    if coupling.exact_null:
        return None
    generator = torch.Generator().manual_seed(config["permutation_seed"])
    draws = coupling.null_draws(scores_1, scores_2, config["n_permutations"], generator)
    return [
        Pairs(first.index_1, torch.stack([draw[k].index_2 for draw in draws]))
        for k, first in enumerate(draws[0])
    ]


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


def _map_values(name: str, params, scores_1, scores_2, pairs, null, config) -> dict:
    z_1, z_2 = MAPS[name](scores_1, **params), MAPS[name](scores_2, **params)
    beta = config["beta_fix"][name]
    stats = _map_statistics(z_1, z_2, pairs, null, beta, config)
    return {f"{name}_{key}": value for key, value in stats.items()}


def cap_bank(scores_1: Tensor, scores_2: Tensor, coupling, config) -> dict[str, float]:
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
            their original dtype, which the cdf pairs and the legacy corner read
        scores_2: Scores of X2 [N2]
        coupling: A coupling of ``binary.pairing`` built on the inputs of X1 and X2
        config: The bank's configuration, as in ``configs/cap/bank.yaml``

    Raises:
        ValueError: If an evidence map or a statistic is unknown.
    """
    _check_config(config)
    scores_1, scores_2 = scores_1.detach().cpu(), scores_2.detach().cpu()
    # The maps are increasing, so that the pairs and the null draws read from the
    # scores serve every map.
    pairs = coupling.pairs(scores_1, scores_2)
    null = _null_draws(coupling, scores_1, scores_2, config)
    values = {"discarded": _discarded(pairs, len(scores_1))}
    for name, params in config["maps"].items():
        values |= _map_values(name, params, scores_1, scores_2, pairs, null, config)
    if config.get("legacy") and coupling.name == "cdf":
        values |= _legacy_corner(scores_1, scores_2, pairs[0], config)
    return values


def _input_hash(name: str, params: dict, inputs: tuple[Tensor, ...]) -> str:
    """Return 16 hex digits of the SHA-256 of the pairing, its parameters and inputs."""
    digest = hashlib.sha256(json.dumps([name, params], sort_keys=True).encode())
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


def _coupling(name: str, params: dict, x_1, x_2, x_ref, cache_dir, label):
    if name == "cdf":
        return COUPLINGS[name](**params)
    inputs = (x_1, x_2, x_ref) if name == "nn" else (x_1, x_2)
    if cache_dir is None:
        return COUPLINGS[name](*inputs, **params)
    path = Path(cache_dir) / f"{name}_{_input_hash(name, params, inputs)}.pt"
    if path.exists():
        return COUPLING_TYPES[name](**torch.load(path, weights_only=True))
    return _stored_coupling(name, params, inputs, path, label)


def build_couplings(
    x_1: Tensor,
    x_2: Tensor,
    x_ref: Tensor | None,
    pairings: dict,
    cache_dir: Path | None = None,
    label: dict | None = None,
) -> dict:
    """Return the registered couplings of X1 [N1, d] and X2 [N2, d], by pairing name.

    With a cache_dir, every coupling but cdf is read from
    ``<cache_dir>/<pairing>_<hash16>.pt`` if that exists, and otherwise built and
    written there, with ``<pairing>_<hash16>.json`` beside it holding the parameters,
    the diagnostics, N1, N2 (and N_ref for nn), the build time in seconds and `label`,
    which must be JSON-serialisable (the names of the sets, say). The hash is that of
    the input bytes, x_ref's for nn included, and of the pairing and its parameters.

    Raises:
        ValueError: If a pairing is unknown, or nn is registered without x_ref.
    """
    unknown = sorted(set(pairings) - set(COUPLINGS))
    if unknown:
        raise ValueError(f"Unknown pairings: {unknown}.")
    if "nn" in pairings and x_ref is None:
        raise ValueError("The nn coupling needs the caliper's reference x_ref.")
    return {
        name: _coupling(name, dict(params or {}), x_1, x_2, x_ref, cache_dir, label)
        for name, params in pairings.items()
    }
