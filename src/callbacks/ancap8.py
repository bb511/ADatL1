"""ANCAP v8, the cross-source capacity of a detector on a domain's pushed points, as
functions of scores and as a training callback.

The design is frozen in ``scratchpad/adl1t/theory/theory_xxxi/step4/V8.md`` (sections 2
to 4, as amended by ``theory_xxxii/redo/FREEZE_2.md``), which keeps the statistic of
``theory_xxiv/step4/V7_r3.md`` (section 2) with the readings chosen in
``theory_xxvii/step4/V7_r3_CHOICES.md``. The parts per kind and the readings of the
condition are those of ``theory_xxxiii/step5/FAB_R4.md`` and
``theory_xxxi/step4/cond/FAB_COND.md``. This module ports the frozen first code of step
5 (``theory_xxxiii/step5/a_p/code/``: ``ancap7r3.py``, ``parts.py`` and ``cond.py``)
and reproduces its integers exactly and its floats within 1e-9. The selection of a
lattice, the factor h and the bound are in ``src/evaluation/components/ancap8.py``.

A candidate is one detector's scores on three sets: S_1, the typical sample of source 1
(zero bias at the LHC), of n_1 rows; S_2, a cell's second source (the gun at the LHC),
of n_2 rows; and P, the domain's pushed points, n_P of them, each of weight rho = 1 /
n_P. With alpha = 1 %, for a sample H of h scores and a score t::

    c_H(t)  = #{s in H : s >= t}                        a tie counts against t
    m_H(t)  = log(alpha (h + 1) / (c_H(t) + 1/2))       the evidence
    flagged under H  <=>  100 c_H(t) + 50 < h + 1       m > 0, in integers
    k(h)    = #{r = 1 ... h : 100 r + 50 < h + 1}       a tie-free sample's own flags

A point is flagged when it lies strictly above the sample's ((h + 50) // 100)-th largest
score. On the full samples, all integers: L and L2, the points flagged under S_1 and
S_2; C2, those flagged under both; ND = L + L2 - 2 C2; N21, the rows of S_2 flagged
under S_1; N12, the rows of S_1 flagged under S_2; N11 and N22, each sample's rows
flagged under itself; k1 = k(n_1), k2 = k(n_2).

The pairs. Pair k = 0 ... 127 draws from ``numpy.random.default_rng([20261111, k])`` a
permutation of range(n_1) and then one of range(n_2). H_1 is S_1 at the first n_1 // 2
entries of the first, H_2 is S_2 at the first n_2 // 2 of the second; for I0, H_2 is S_1
at the remaining entries of the first, so that I and I0 share H_1. The pairs depend on
(n_1, n_2) alone and are cached. Then::

    p_d(q)    = sigmoid(beta m_{H_d}(t(q)) + logit alpha)
    xi_d(q)   = (p_d(q) - alpha) / sqrt(alpha (1 - alpha))
    A_k(beta) = sum over q of rho log(1 + xi_1(q) xi_2(q))       on pair k
    I         = max over the grid of the mean of A_k over the pairs
    beta_star = the smallest grid beta whose mean lies within 1e-9 of I
    spread    = sd, ddof 1, over the pairs of A_k(beta_star), divided by sqrt(128)
    Cbar2     = sum over the pairs of B_k, divided by 128 n_P in one division of
                integers; B_k the points flagged under both halves of pair k
    spread_c  = sd, ddof 1, over the pairs of B_k / n_P, divided by sqrt(128)

The grid is 0 and 0.01 x 10^(j / 40), j = 0 ... 240; A(0) = 0 is set exactly. I0,
beta_star0 and spread0 are I, beta_star and the spread on the pairs of I0, for every
candidate. The kernel is evaluated in the softplus form, with c = logit alpha::

    log(1 + xi_1 xi_2) = sp(beta (m_1 + m_2) + c) - sp(beta m_1 + c) - sp(beta m_2 + c)
                         - log(1 - alpha)

which splits the mean curve into a joint term, gathered on the distinct values of
(2 k_1 + 1)(2 k_2 + 1), and two marginal terms, gathered on the distinct counts, before
the betas are applied. The three terms grow like beta abs(m) and cancel to O(1), so that
the float64 rounding of A grows with beta: about 1e-10 nats at beta = 1e4.

The parts per kind (FAB_R4.md, readings 2 to 5), where each point carries a kind 1 ...
K: n_k, the points of kind k; L_k and C2_k, its points flagged under S_1 and under both
full samples; Cbar2_k, the sum over the pairs of its points flagged under both halves
divided by 128 n_k; I_k, the mean over the pairs of the sum over its points of rho
log(1 + xi_1 xi_2) at the candidate's own beta_star; share_k = I_k / I, NaN where
round(I, 9) is 0. The I_k sum to I within 1e-9, the L_k to L and the C2_k to C2; a
breach raises. K is the caller's number of kinds, so that a kind without a point keeps
its keys (n_k 0, Cbar2_k NaN).

The condition (V8.md, section 2, on the full samples), whose factor h and bound are
computed at the lattice step for the reference detector alone: the cut of S_i is its
((n_i + 50) // 100)-th largest score, cut1 and cut2; the looser source a has the lower
cut, source 1 at equal cuts (``looser``); rows_at_depth = (n_a + 1) alpha e^-M - 1/2
with M = 2 nats, and the condition is read (``read``) when it is at least the least
count 25. Over the thresholds, the distinct scores t of S_a strictly above its cut with
c_a(t) >= 25 and c_a(t) + 1/2 >= (n_a + 1) alpha e^-M (n_thresholds of them)::

    ratio_max = max over t of (2 c_b + 1)(n_a + 1) / ((2 c_a + 1)(n_b + 1))   exact
    r_hat     = max(1, ratio_max)

taken at the lowest score that reaches the maximum, where c_a_at_max, c_b_at_max and
depth_of_max = log(alpha (n_a + 1) / (c_a + 1/2)) are read; r_at_cut is the ratio at the
cut of S_a. Of the points flagged under S_a (flagged_a), flagged_both are flagged under
S_b too, and share_between = (flagged_a - flagged_both) / flagged_a. Of the rows of S_a
flagged under S_a (rows_a_flagged), rows_a_flagged_both are flagged under S_b too, and
consequence_holds = flagged_both rows_a_flagged >= rows_a_flagged_both flagged_a. The
last two are NaN where flagged_a is 0. ``marker`` says how the condition was read::

    0  read
    1  not read: sample size. Either rows_at_depth < 25, and the ratio fields are NaN;
       or a sample has fewer than 50 rows and so no cut, and then read is 0, the cut
       of a sample that has one is kept and every other field is NaN
    2  not read: no threshold. read is 1 and n_thresholds 0, and the other ratio
       fields are NaN

The fields of marker 1 for a sample without a cut, and of marker 2, are the user's
reading of V8.md, section 2 (12 Nov 2026), planted in
``scratchpad/adl1t/test_bed/v8_repo/agents/markers/MARKERS.txt``. The condition's k_i,
(n_i + 50) // 100, and its L, the statistic's L where S_1 has a cut, are left to the
lattice step.

A candidate is usable when its three arrays are present, non-empty, of the declared
sizes where sizes are declared, and finite. An unusable candidate gives usable 0 and NaN
elsewhere, except n1, n2 and n_p, the lengths of its present arrays; an array of another
length than its declared one is read as absent and its length is not written
(``theory_xxxiii/step5/DECLARATION_5A.md``, readings 3 and 10). Scores are cast to
float64 first. Every value is a float: booleans 1.0 or 0.0, NaN where undefined.

``candidate`` returns the values of one cell, ``ancap8`` those of every cell keyed
``<key>_<cell>``; the callbacks log the latter.
"""

import functools
import math
from fractions import Fraction

import numpy as np
import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader

from src.callbacks.components.points import load_points
from src.callbacks.components.utils import placeholder_loader
from src.callbacks.score_dump import score_in_eval_mode
from src.callbacks.threshold_posterior import LOG_KWARGS

# The rate alpha, also the prior of a label (V8.md, section 4).
ALPHA = 0.01
LOGIT = float(np.log(ALPHA / (1 - ALPHA)))
# K = 128 pairs and the seed of their draws (V8.md, section 4; V7_r3.md, section 2).
N_PAIRS = 128
SEED = 20261111
# The grid of beta, 0 and 0.01 x 10^(j / 40) for j = 0 ... 240 (V8.md, section 4).
BETAS = np.concatenate([[0.0], 0.01 * 10.0 ** (np.arange(241) / 40)])
# Points of the mean curve this close are tied, and every rank and comparison reads
# values rounded to this many decimals (V7_r3.md, section 2).
TIE = 1e-9
DIGITS = 9
# The condition's cap on the depth, in nats, and its least count (V8.md, section 4).
M_CAP = 2.0
LEAST_COUNT = 25
# The condition's markers (V8.md, section 2).
READ, NOT_READ_SIZE, NOT_READ_THRESHOLD = 0, 1, 2
# The parts must sum to the capacity within this, in nats (FAB_R4.md, reading 5).
IDENTITY_TOL = 1e-9
# exp(-746) is 0.0 in float64, so a softplus argument below this adds exactly nothing.
UNDERFLOW = -746.0
# Rows of a (values, betas) block of the softplus sums: about 8 MB at 242 betas.
CHUNK = 4096
SIZES = ("n1", "n2", "n_p")
COUNTS = ("L", "L2", "C2", "ND", "N21", "N12", "N11", "N22")
CAPACITIES = ("I", "beta_star", "spread", "I0", "beta_star0", "spread0")
STATISTIC = ("usable", *SIZES, "k1", "k2", *COUNTS, *CAPACITIES, "Cbar2", "spread_c")
RATIO_KEYS = ("n_thresholds", "ratio_max", "r_hat", "r_at_cut")
RATIO_KEYS += ("c_a_at_max", "c_b_at_max", "depth_of_max")
FLAG_KEYS = ("flagged_a", "flagged_both", "share_between", "rows_a_flagged")
FLAG_KEYS += ("rows_a_flagged_both", "consequence_holds")
CONDITION = ("cut1", "cut2", "looser", "rows_at_depth", "read", *RATIO_KEYS)
CONDITION += (*FLAG_KEYS, "marker")
PARTS = ("n_k", "L_k", "C2_k", "Cbar2_k", "I_k", "share_k")


def keys(n_kinds: int = 0) -> tuple[str, ...]:
    """Return the keys of ``candidate`` in order, with the parts of n_kinds kinds."""
    parts = tuple(f"{p}{k}" for k in range(1, n_kinds + 1) for p in PARTS)
    return STATISTIC + CONDITION + parts


def evidence(k: np.ndarray, h: int) -> np.ndarray:
    """Return m = log(alpha (h + 1) / (k + 1/2)), k of h scores at or above a point."""
    return np.log(ALPHA * (h + 1) / (k + 0.5))


def flagged(k: np.ndarray, h: int) -> np.ndarray:
    """Return m > 0 by the integer comparison 100 k + 50 < h + 1 (1 / alpha = 100)."""
    return 100 * np.asarray(k, dtype=np.int64) + 50 < h + 1


def k_of(h: int) -> int:
    """Return k(h), the rows of a tie-free sample of h flagged under itself."""
    return max(0, (h - 50) // 100)


def softplus(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def kernel(beta: float, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Return log(1 + xi_1 xi_2) in the softplus form, which does not saturate."""
    joint = softplus(beta * (m1 + m2) + LOGIT)
    marginal = softplus(beta * m1 + LOGIT) + softplus(beta * m2 + LOGIT)
    return joint - marginal - np.log1p(-ALPHA)


@functools.lru_cache(maxsize=4)
def _permutations(n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
    """Return each pair's permutation of range(n1), (N_PAIRS, n1), and the first n2 // 2
    entries of its permutation of range(n2), (N_PAIRS, n2 // 2).

    Stored as int32, which halves the cache: 38 MB at the LHC's 50,000 rows a side.
    """
    first, second = [], []
    for k in range(N_PAIRS):
        rng = np.random.default_rng([SEED, k])
        first.append(rng.permutation(n1))
        second.append(rng.permutation(n2)[: n2 // 2])
    drawn = (np.stack(first).astype(np.int32), np.stack(second).astype(np.int32))
    for indices in drawn:
        indices.flags.writeable = False
    return drawn


def _ordering(s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each score's rank in s and, per point, the first rank at or above it."""
    order = np.argsort(s, kind="stable")
    rank = np.empty(len(s), dtype=np.int64)
    rank[order] = np.arange(len(s))
    return rank, np.searchsorted(s[order], t, side="left")


def _count_at_or_above(
    rank: np.ndarray, members: np.ndarray, first: np.ndarray
) -> np.ndarray:
    """Return #{i in members : s_i >= t(q)} per point, from ``_ordering``'s output."""
    hit = np.zeros(len(rank) + 1, dtype=np.int64)
    hit[rank[members]] = 1
    return np.cumsum(hit[::-1])[::-1][first]


def pair_counts(
    s1: np.ndarray, s2: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the counts at or above each point of t in H_1, H_2 and the rest of S_1,
    (N_PAIRS, len(t)) each.

    The pairs of I are (H_1, H_2), those of I0 (H_1, the rest of S_1).
    """
    first, second = _permutations(len(s1), len(s2))
    (r1, f1), (r2, f2) = _ordering(s1, t), _ordering(s2, t)
    half = len(s1) // 2
    k1 = np.stack([_count_at_or_above(r1, perm[:half], f1) for perm in first])
    k0 = np.stack([_count_at_or_above(r1, perm[half:], f1) for perm in first])
    k2 = np.stack([_count_at_or_above(r2, head, f2) for head in second])
    return k1, k2, k0


def _softplus_sum(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return sum_u w_u sp(beta x_u + c) for every beta of the grid, (n_beta,).

    Where a block's largest x is negative, the betas beyond (UNDERFLOW - c) / max(x) add
    exactly zero and are skipped; x from ``np.unique`` is sorted, so that most blocks of
    strongly negative evidence skip most of the grid.
    """
    out = np.zeros(len(BETAS))
    for i in range(0, len(x), CHUNK):
        xs, ws = x[i : i + CHUNK], w[i : i + CHUNK]
        top, stop = xs.max(), len(BETAS)
        if top < 0:
            stop = np.searchsorted(BETAS, (UNDERFLOW - LOGIT) / top, side="right")
        out[:stop] += ws @ softplus(np.multiply.outer(xs, BETAS[:stop]) + LOGIT)
    return out


def _marginal(k: np.ndarray, h: int) -> np.ndarray:
    """Return the sum over the entries of k of sp(beta m + c), halves of h rows."""
    wk = np.bincount(k.ravel(), minlength=h + 1)
    kept = np.flatnonzero(wk)
    return _softplus_sum(evidence(kept, h), wk[kept].astype(np.float64))


def _joint(k1: np.ndarray, k2: np.ndarray, h1: int, h2: int) -> tuple:
    """Return the distinct values of m1 + m2 over the entries and their counts."""
    product, n = np.unique((2 * k1 + 1) * (2 * k2 + 1), return_counts=True)
    # m1 + m2 = log(4 alpha^2 (h1 + 1)(h2 + 1) / ((2 k1 + 1)(2 k2 + 1)))
    return np.log(4 * ALPHA**2 * (h1 + 1) * (h2 + 1) / product), n


def mean_curve(k1: np.ndarray, k2: np.ndarray, h1: int, h2: int) -> np.ndarray:
    """Return the mean over the pairs of A(beta) on the grid, (n_beta,).

    The entries are counted as integers, so that the joint and the two marginal sums
    have exactly equal totals: summed float weights give them different last bits, which
    beta abs(m) turns into 6e-7 nats at beta = 1e4.
    """
    m12, n = _joint(k1, k2, h1, h2)
    part = _softplus_sum(m12, n.astype(np.float64))
    part -= np.log1p(-ALPHA) * k1.size
    part = part - _marginal(k1, h1) - _marginal(k2, h2)
    curve = 1.0 / k1.shape[1] / len(k1) * part
    curve[BETAS == 0] = 0.0
    return curve


def capacity(
    k1: np.ndarray, k2: np.ndarray, h1: int, h2: int
) -> tuple[float, float, float, np.ndarray]:
    """Return I, beta_star, the spread and the kernel at beta_star, (N_PAIRS, n_P)."""
    curve = mean_curve(k1, k2, h1, h2)
    top = curve.max()
    beta_star = BETAS[np.argmax(curve >= top - TIE)]
    kern = kernel(beta_star, evidence(k1, h1), evidence(k2, h2))
    per_pair = kern @ np.full(k1.shape[1], 1.0 / k1.shape[1])
    spread = per_pair.std(ddof=1) / np.sqrt(len(k1))
    return float(top), float(beta_star), float(spread), kern


def pair_count(both: np.ndarray) -> tuple[float, float]:
    """Return Cbar2 and its spread from the flags under both halves, (N_PAIRS, n_P)."""
    b = both.sum(axis=1)
    mean = int(b.sum()) / (len(both) * both.shape[1])
    share = b / both.shape[1]
    return mean, float(share.std(ddof=1) / np.sqrt(len(both)))


def _flags(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return whether each entry of t is flagged under the full sample s."""
    return flagged(len(s) - np.searchsorted(np.sort(s), t, side="left"), len(s))


def _counts(
    s1: np.ndarray, s2: np.ndarray, f1: np.ndarray, f2: np.ndarray
) -> dict[str, int]:
    """Return the integer counts on the full samples; f1, f2 flag P under S_1, S_2."""
    level, level2, both = int(f1.sum()), int(f2.sum()), int((f1 & f2).sum())
    out = {"L": level, "L2": level2, "C2": both, "ND": level + level2 - 2 * both}
    out |= {"N21": int(_flags(s1, s2).sum()), "N12": int(_flags(s2, s1).sum())}
    return out | {"N11": int(_flags(s1, s1).sum()), "N22": int(_flags(s2, s2).sum())}


def _statistic(s1: np.ndarray, s2: np.ndarray, p: np.ndarray) -> tuple[dict, tuple]:
    """Return the statistic's values and what the parts read: the flags of P under the
    full samples, the flags under both halves and the kernel at beta_star."""
    k1, k2, k0 = pair_counts(s1, s2, p)
    h1, h2, h0 = len(s1) // 2, len(s2) // 2, len(s1) - len(s1) // 2
    *two, kern = capacity(k1, k2, h1, h2)
    *same, _ = capacity(k1, k0, h1, h0)
    both, flags = flagged(k1, h1) & flagged(k2, h2), (_flags(s1, p), _flags(s2, p))
    out = dict(zip(SIZES, (len(s1), len(s2), len(p))))
    out |= {"usable": 1, "k1": k_of(len(s1)), "k2": k_of(len(s2))}
    out |= _counts(s1, s2, *flags) | dict(zip(CAPACITIES, two + same))
    return out | dict(zip(("Cbar2", "spread_c"), pair_count(both))), (flags, both, kern)


def _kind_part(kind: int, at: np.ndarray, measured: tuple, total: float) -> dict:
    """Return the parts of one kind, keyed '<part><kind>'; at selects its points and
    measured holds the flags of P, the flags under both halves and rho times the kernel
    at beta_star, (N_PAIRS, n_P)."""
    (f1, f2), both, terms = measured
    n_k = int(at.sum())
    part = {"n_k": n_k, "L_k": int(f1[at].sum()), "C2_k": int((f1 & f2)[at].sum())}
    part["Cbar2_k"] = int(both[:, at].sum()) / (len(both) * n_k) if n_k else math.nan
    part["I_k"] = float(terms[:, at].sum(axis=1).mean())
    zero = round(total, DIGITS) == 0
    part["share_k"] = math.nan if zero else part["I_k"] / total
    return {f"{name}{kind}": value for name, value in part.items()}


def _check_identity(parts: dict, values: dict, n_kinds: int) -> None:
    """Raise where the parts miss I by more than IDENTITY_TOL, or L or C2 by a count."""
    kinds = range(1, n_kinds + 1)
    gap = abs(sum(parts[f"I_k{k}"] for k in kinds) - values["I"])
    levels = sum(parts[f"L_k{k}"] for k in kinds)
    joint = sum(parts[f"C2_k{k}"] for k in kinds)
    if gap > IDENTITY_TOL or levels != values["L"] or joint != values["C2"]:
        raise ArithmeticError(f"The parts miss I by {gap:.3e}, or L or C2 by a count.")


def _parts(
    kinds: np.ndarray, measured: tuple, values: dict, n_kinds: int
) -> dict[str, float]:
    """Return the parts of kinds 1 to n_kinds, raising where reading 5 fails."""
    flags, both, kern = measured
    measured = (flags, both, kern / kern.shape[1])
    parts = {}
    for kind in range(1, n_kinds + 1):
        parts |= _kind_part(kind, kinds == kind, measured, values["I"])
    _check_identity(parts, values, n_kinds)
    return parts


def _cut(ascending: np.ndarray) -> float:
    """Return the ((n + 50) // 100)-th largest score, ties counted with multiplicity."""
    return float(ascending[len(ascending) - (len(ascending) + 50) // 100])


def _at_or_above(ascending: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Return c(t), the scores of an ascending sample at or above t."""
    return len(ascending) - np.searchsorted(ascending, t, side="left")


def _ratio(c_a: int, c_b: int, n_a: int, n_b: int) -> Fraction:
    """Return u_b / u_a exactly, in Python ints.

    (2 c_b + 1)(n_a + 1) times its partner overflows int64, and the maximum's ties must
    be exact.
    """
    return Fraction((2 * c_b + 1) * (n_a + 1), (2 * c_a + 1) * (n_b + 1))


def _threshold_counts(
    a: np.ndarray, b: np.ndarray, cut_a: float, product: float
) -> tuple[list[int], list[int]]:
    """Return c_a and c_b at each threshold, the scores ascending."""
    above = np.unique(a[a > cut_a])
    c_above = _at_or_above(a, above)
    keep = (c_above >= LEAST_COUNT) & (c_above + 0.5 >= product)
    return c_above[keep].tolist(), _at_or_above(b, above[keep]).tolist()


def _maximum_fields(
    a: np.ndarray, b: np.ndarray, cut_a: float, c_a: list[int], c_b: list[int]
) -> dict:
    """Return the fields read at the largest ratio of the thresholds and at the cut."""
    n_a, n_b = len(a), len(b)
    ratios = [_ratio(x, y, n_a, n_b) for x, y in zip(c_a, c_b)]
    # The thresholds ascend, so that index() finds the lowest score at the maximum.
    i, best = ratios.index(max(ratios)), max(ratios)
    at_cut = _ratio(int(_at_or_above(a, cut_a)), int(_at_or_above(b, cut_a)), n_a, n_b)
    out = {"ratio_max": float(best), "r_hat": float(max(Fraction(1), best))}
    out |= {"r_at_cut": float(at_cut), "c_a_at_max": c_a[i], "c_b_at_max": c_b[i]}
    out["depth_of_max"] = math.log(ALPHA * (n_a + 1) / (c_a[i] + 0.5))
    return out


def _ratio_fields(a: np.ndarray, b: np.ndarray, cut_a: float, product: float) -> dict:
    """Return the ratio fields and the marker of a condition read by its size; a and b
    are S_a and S_b ascending, product is (n_a + 1) alpha e^-M."""
    c_a, c_b = _threshold_counts(a, b, cut_a, product)
    out = dict.fromkeys(RATIO_KEYS, math.nan) | {"n_thresholds": len(c_a)}
    if not c_a:
        return out | {"marker": NOT_READ_THRESHOLD}
    return out | _maximum_fields(a, b, cut_a, c_a, c_b) | {"marker": READ}


def _flag_fields(p: np.ndarray, a: np.ndarray, cut_a: float, cut_b: float) -> dict:
    """Return the counts beside the condition: P and the rows of S_a under both cuts."""
    above_a, rows_above_a = p > cut_a, a > cut_a
    flagged_a, flagged_both = int(above_a.sum()), int((above_a & (p > cut_b)).sum())
    rows, rows_both = int(rows_above_a.sum()), int((rows_above_a & (a > cut_b)).sum())
    out = {"flagged_a": flagged_a, "flagged_both": flagged_both}
    out |= {"rows_a_flagged": rows, "rows_a_flagged_both": rows_both}
    if flagged_a == 0:
        return out | {"share_between": math.nan, "consequence_holds": math.nan}
    out["share_between"] = (flagged_a - flagged_both) / flagged_a
    return out | {"consequence_holds": flagged_both * rows >= rows_both * flagged_a}


def _no_cut(ascending: tuple) -> dict[str, float]:
    """Return the condition where a sample of fewer than 50 rows has no cut."""
    out = dict.fromkeys(CONDITION, math.nan) | {"read": 0.0, "marker": NOT_READ_SIZE}
    for i, s in enumerate(ascending, 1):
        if (len(s) + 50) // 100 > 0:
            out[f"cut{i}"] = _cut(s)
    return {key: float(value) for key, value in out.items()}


def _read_fields(ascending: tuple, cuts: tuple, a: int) -> dict:
    """Return the depth test and the ratio fields; a indexes the looser source."""
    product = (len(ascending[a]) + 1) * ALPHA * math.exp(-M_CAP)
    out = {"rows_at_depth": product - 0.5, "read": product - 0.5 >= LEAST_COUNT}
    if not out["read"]:
        return out | dict.fromkeys(RATIO_KEYS, math.nan) | {"marker": NOT_READ_SIZE}
    return out | _ratio_fields(ascending[a], ascending[1 - a], cuts[a], product)


def condition(s1: np.ndarray, s2: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Return the condition's keys of one candidate, without h and the bound."""
    ascending = (np.sort(s1), np.sort(s2))
    if min(len(s1), len(s2)) < 50:
        return _no_cut(ascending)
    cuts = (_cut(ascending[0]), _cut(ascending[1]))
    a = 0 if cuts[0] <= cuts[1] else 1
    out = {"cut1": cuts[0], "cut2": cuts[1], "looser": a + 1}
    out |= _read_fields(ascending, cuts, a)
    out |= _flag_fields(p, ascending[a], cuts[a], cuts[1 - a])
    return {key: float(out[key]) for key in CONDITION}


def _lengths(arrays: list, sizes: dict | None) -> list[int | None]:
    """Return each array's length, None where it is absent or of another declared
    length."""
    declared = [None] * 3 if sizes is None else [sizes.get(k) for k in SIZES]
    return [
        None if x is None or (n is not None and len(x) != n) else len(x)
        for x, n in zip(arrays, declared)
    ]


def _as_arrays(
    scores_1: np.ndarray | torch.Tensor | None,
    scores_2: np.ndarray | torch.Tensor | None,
    points: np.ndarray | torch.Tensor | None,
) -> list[np.ndarray | None]:
    """Return the three sets as float64 vectors, None where absent."""
    return [
        None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)
        for x in (scores_1, scores_2, points)
    ]


def _usable(arrays: list, lengths: list) -> bool:
    """Return whether every set is present, of its declared size, non-empty and
    finite."""
    return all(n for n in lengths) and all(np.isfinite(x).all() for x in arrays)


def _kind_count(kinds: np.ndarray | None, n_kinds: int | None) -> int:
    """Return the number of kinds: 0 without kinds, else n_kinds or the largest kind."""
    if kinds is None:
        return 0
    return int(kinds.max(initial=0)) if n_kinds is None else n_kinds


def _values(arrays: list, kinds: np.ndarray | None, n_kinds: int) -> dict:
    """Return the statistic, the condition and, with kinds, the parts of a usable
    candidate."""
    values, measured = _statistic(*arrays)
    values |= condition(*arrays)
    if kinds is None:
        return values
    if len(kinds) != len(arrays[2]):
        raise ValueError(f"{len(kinds)} kinds for {len(arrays[2])} points.")
    return values | _parts(kinds, measured, values, n_kinds)


def candidate(
    scores_1: np.ndarray | torch.Tensor | None,
    scores_2: np.ndarray | torch.Tensor | None,
    points: np.ndarray | torch.Tensor | None,
    kinds: np.ndarray | None = None,
    sizes: dict | None = None,
    n_kinds: int | None = None,
) -> dict[str, float]:
    """Return every value of one candidate in one cell, keyed by ``keys``.

    Args:
        scores_1: Scores of S_1 [n_1], as anything ``np.asarray`` accepts, or None.
        scores_2: Scores of S_2 [n_2], or None.
        points: Scores of P [n_P], or None.
        kinds: The kind of each point, integers 1 to K [n_P], or None for no parts.
        sizes: Declared sizes, any of ``n1``, ``n2`` and ``n_p``.
        n_kinds: The number of kinds K, whose keys are given even to a kind without a
            point; None takes the largest of kinds.
    """
    arrays = _as_arrays(scores_1, scores_2, points)
    kinds = None if kinds is None else np.asarray(kinds, dtype=np.int64).reshape(-1)
    n_kinds = _kind_count(kinds, n_kinds)
    names, lengths = keys(n_kinds), _lengths(arrays, sizes)
    if not _usable(arrays, lengths):
        present = {k: float(n) for k, n in zip(SIZES, lengths) if n is not None}
        return dict.fromkeys(names, math.nan) | {"usable": 0.0} | present
    values = _values(arrays, kinds, n_kinds)
    return {key: float(values[key]) for key in names}


def ancap8(
    scores_1: np.ndarray | torch.Tensor,
    views: dict[str, np.ndarray | torch.Tensor],
    points: list[np.ndarray | torch.Tensor],
    kinds: bool = False,
    sizes: dict | None = None,
) -> dict[str, float]:
    """Return every value of ``candidate`` for every cell, keyed '<key>_<cell>'.

    Args:
        scores_1: Scores of S_1 [n_1], as anything ``np.asarray`` accepts.
        views: The scores of S_2 of each cell, by cell.
        points: The scores of each set of points, joined in list order into P.
        kinds: Whether the k-th set holds the points of kind k, counted from 1; every
            set keeps its kind's keys, an empty one included.
        sizes: Declared sizes, as ``candidate`` takes them.
    """
    sets = [np.asarray(x, dtype=np.float64).reshape(-1) for x in points]
    joined = np.concatenate(sets) if sets else np.empty(0)
    labels = np.repeat(np.arange(1, len(sets) + 1), [len(x) for x in sets])
    labels, n_kinds = (labels, len(sets)) if kinds else (None, None)
    values = {}
    for cell, scores_2 in views.items():
        stats = candidate(scores_1, scores_2, joined, labels, sizes, n_kinds)
        values |= {f"{key}_{cell}": value for key, value in stats.items()}
    return values


def _points_loader(spec: dict, batch_size: int) -> DataLoader:
    """Return a loader of the points of one entry {path, md5_x, md5_mask}.

    ``load_points`` refuses a file whose x is not float32, whose mask is not bool, or
    whose md5s are not the declared ones.
    """
    x, mask = load_points(str(spec["path"]), spec["md5_x"], spec.get("md5_mask"))
    return placeholder_loader(x, batch_size, mask)


def _as_logged(values: dict[str, float], name: str) -> dict[str, torch.Tensor]:
    """Return values keyed 'val/<name>/<key>' as float64 tensors, which self.log keeps;
    a Python float would be logged in float32."""
    return {
        f"val/{name}/{key}": torch.tensor(value, dtype=torch.float64)
        for key, value in values.items()
    }


class ANCAP8Callback(Callback):
    """Log ANCAP v8's values of every cell at every validation epoch.

    The scores of dataset_1 and of each cell's view are collected batch by batch at
    every validation epoch after the sanity pass. Each points file is read once at fit
    start through ``load_points``, which refuses it unless its md5s are the declared
    ones, scored at every epoch's end in eval mode under forked RNG streams, as
    ThresholdPosteriorCallback scores its reference, and the files are joined in the
    order of ``points`` into P. Every key of ``ancap8`` is logged as
    ``val/<name>/<key>_<cell>`` in float64; nothing is averaged or selected.

    Args:
        output_name: Key in the outputs dict of the per-event anomaly scores.
        dataset_1: Name of the validation dataloader of S_1.
        views: Name of the validation dataloader of S_2, by cell.
        points: ``{name: {path, md5_x, md5_mask}}``, the entries of the evaluation's
            ``extra_sets.points``: an ``.npz`` file with float32 inputs ``x`` in the
            model's input space and, on data with a mask, a bool ``mask`` of the shape
            of ``x``, with the md5 of each array's bytes (``md5_mask`` None or absent
            for a file without a mask).
        kinds: Whether the k-th entry of points holds the points of kind k, counted
            from 1.
        sizes: Declared sizes, any of ``n1``, ``n2`` and ``n_p``.
        batch_size: Batch size of the passes over the points: the LHC dump's auxiliary
            batch size, at which the evaluation scores them.
        name: The callback's name, a segment of every logged key.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        views: dict[str, str],
        points: dict[str, dict],
        kinds: bool = False,
        sizes: dict[str, int] | None = None,
        batch_size: int = 1024,
        name: str = "ancap8",
    ):
        super().__init__()
        self.output_name, self.name = output_name, name
        self.dataset_1, self.views = dataset_1, dict(views)
        self.points, self.kinds = dict(points), kinds
        self.sizes, self.batch_size = sizes, batch_size
        self.datasets = [dataset_1, *self.views.values()]

    def on_fit_start(self, trainer, pl_module):
        """Read and check the points files and build their loaders."""
        self.loaders = {
            name: _points_loader(spec, self.batch_size)
            for name, spec in self.points.items()
        }

    def on_validation_epoch_start(self, trainer, pl_module):
        """Check the dataloaders and reset the scores."""
        dset_names = list(trainer.val_dataloaders.keys())
        missing = [name for name in self.datasets if name not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires validation dataloaders {missing}. "
                f"Available validation dataloaders: {dset_names}"
            )
        # The empty tensor stands for a set whose loader serves no batch.
        self.scores = {name: [torch.empty(0)] for name in self.datasets}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect the scores of S_1 and of the views."""
        if trainer.sanity_checking:
            return
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name in self.scores:
            score = outputs[self.output_name].detach().reshape(-1)
            self.scores[dset_name].append(score.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Score the points, compute every cell's values and log them."""
        if trainer.sanity_checking:
            return
        scores = {name: torch.cat(self.scores[name]) for name in self.datasets}
        views = {cell: scores[name] for cell, name in self.views.items()}
        points = self._points_scores(trainer, pl_module)
        values = ancap8(scores[self.dataset_1], views, points, self.kinds, self.sizes)
        # One value of batch size 1: the epoch's mean returns it to the last bit.
        logged = _as_logged(values, self.name)
        pl_module.log_dict(logged, batch_size=1, **LOG_KWARGS)

    def _points_scores(self, trainer, pl_module) -> list[torch.Tensor]:
        """Return the scores of each points file, in the order of ``points``."""
        wanted = [self.output_name]
        scored = score_in_eval_mode(trainer, pl_module, self.loaders, wanted).scores()
        # An empty points file serves no batch and so leaves no entry.
        return [
            scored[name][self.output_name] if name in scored else torch.empty(0)
            for name in self.loaders
        ]
