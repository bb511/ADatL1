"""Label-free model-selection criteria for anomaly detectors, as functions of scores.

Every function takes anomaly scores, higher meaning more anomalous, as anything that
``np.asarray`` accepts (CPU torch tensors included), and computes in float64. Each
implements one published criterion, checked against the authors' public code where it
exists; the docstrings give the source and fix every convention the source leaves open.

    npd                normalised pseudo discrepancy (AutoUAD), higher is better
    rtm, eag           relative top-median and expected anomaly gap (AutoUAD), higher
    excess_mass        excess-mass area (Goix), higher
    mass_volume        mass-volume area (Goix), lower
    loss_entropy       loss entropy (EntropyStop), lower
    entropy_stop       the offline stopping rule of EntropyStop
    udr                seed agreement, UDR as adapted by Ma et al., higher
    model_centrality   model centrality with Spearman's rho (Ma et al.), higher
    sds_pseudo_sets    pseudo-targets and pseudo-outliers of self-adaptive data shifting
    sds_error          the SDS error of a thresholded score, lower

Sources. W. Dai and J. Fan, "AutoUAD", ICLR 2025, OpenReview ErQPdaD5wJ. N. Goix,
arXiv:1607.01152, with github.com/ngoix/EMMV_benchmarks, em.py and em_bench.py (commit
490d423). Y. Huang et al., "EntropyStop", KDD 2024, doi:10.1145/3637528.3671943, with
github.com/goldenNormal/EntropyStop-KDD2024, EntropyEarlyStop.py (commit b1e0774).
M. Q. Ma et al., SIGKDD Explor. 25(1), 2023, doi:10.1145/3606274.3606277, with
github.com/yzhao062/UOMS, UDR.py and MC.py (commit a27ebbb). S. Wang et al., Pattern
Recognit. 74, 198-211 (2018), doi:10.1016/j.patcog.2017.09.012, whose text is not open;
SDS follows sds.m and main.m of github.com/homarques/SDS-hyperparameter-selection
(commit 8bdbe4a), the implementation behind Marques et al., DMKD 37 (2023),
doi:10.1007/s10618-023-00931-x.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.metrics import auc

# AutoUAD, App. G, p. 23: "For all eps, we set it as 1 x 10^-9".
EPS = 1e-9
# AutoUAD: tau = 5 for RTM (Sec. 3.1, p. 5); rho = 0.2 for EAG (Assumption 2, App. A).
RTM_TOP_PERCENT = 5
EAG_TOP_PERCENT = 20
# Goix: the EM level t_max and the alpha grid of em_bench.py.
EM_T_MAX = 0.9
MV_ALPHAS = np.arange(0.9, 0.999, 0.0001)
# sds.m's default edge threshold; the Python port
# bzantium/OCSVM-hyperparameter-selection defaults to 0.01 with a strict inequality.
SDS_THRESHOLD = 0.1


def _scores(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _top_count(percent: int, n: int) -> int:
    # Integer ceil(percent n / 100), free of the rounding of percent / 100.
    return -(-percent * n // 100)


def npd(normal, noise, eps: float = EPS) -> float:
    """Normalised pseudo discrepancy, AutoUAD Def. 6 and Eq. (7), p. 6.

    ``normal`` scores held-out normal data (X_val) and ``noise`` draws from N(mu_trn,
    diag sigma_trn^2), the moments of the training inputs. The variances are population
    variances (ddof 0), as in the only public implementation (github.com/
    OliverHennhoefer/label-free); the paper does not state the estimator. The squared
    gap ignores the sign of the mean difference.
    """
    s_val, s_gen = _scores(normal), _scores(noise)
    gap = s_gen.mean() - s_val.mean()
    return float(gap**2 / (2 * (s_gen.var() + s_val.var()) + eps))


def rtm(normal, eps: float = EPS) -> float:
    """Relative top-median, AutoUAD Def. 3 and Eq. (4), p. 5, with tau = 5.

    Eq. (4) averages the scores at or above s_(tau N / 100) of the ascending order,
    which read literally is the top 95 %; the text ("the top tau %", p. 5) and the name
    RTM_5% mean the tau % largest scores, which are taken here. tau N / 100 is rounded
    up, and scores tied with the threshold enter, as the ">=" of Eq. (4) has it. The
    ratio is not translation-invariant: with a median at or below zero (a negative
    log-likelihood can be negative) its sign carries no meaning.
    """
    s = _scores(normal)
    threshold = np.sort(s)[-_top_count(RTM_TOP_PERCENT, s.size)]
    median = np.median(s)
    return float((s[s >= threshold].mean() - median) / (median + eps))


def _anomaly_gap(high: np.ndarray, low: np.ndarray, eps: float) -> float:
    """The anomaly gap of AutoUAD, equation (5), with population variances."""
    w1, w0 = high.size / (high.size + low.size), low.size / (high.size + low.size)
    gap = w0 * w1 * (low.mean() - high.mean()) ** 2
    return gap / (w0 * low.var() + w1 * high.var() + eps)


def eag(normal, eps: float = EPS) -> float:
    """Expected anomaly gap, AutoUAD Defs. 4-5 and Eqs. (5)-(6), pp. 5-6.

    The discrete form of Sec. 3.2 averages the anomaly gap over the thresholds at or
    above the 80th percentile. As in Algorithms 1 and 2 (App. A, pp. 14-15), the scores
    are sorted in descending order and the k-th threshold splits them by position into
    the k largest and the rest, for k = 1, ..., ceil(0.2 N) (at most N - 1). Each gap
    is Eq. (5), w0 w1 (mu0 - mu1)^2 / (w0 var0 + w1 var1 + eps), with the population
    variances of the empirical distribution that Eq. (5) implies. Algorithm 1 adds eps
    as eps / N in the units of Eq. (5), and Algorithm 2 uses sample variances; both
    differ from Eq. (5) only for small k or near-constant groups.
    """
    s = np.sort(_scores(normal))[::-1]
    count = min(_top_count(EAG_TOP_PERCENT, s.size), s.size - 1)
    gaps = [_anomaly_gap(s[:k], s[k:], eps) for k in range(1, count + 1)]
    return float(np.mean(gaps))


def _em(t: np.ndarray, t_max: float, volume: float, s_unif, s_x) -> tuple[float, bool]:
    """``em`` of EMMV_benchmarks/em.py on normality scores: (area, t_max reached).

    Kept line for line with the code, so that the two can be compared; the counts
    #{s > u} come from sorted copies, which gives the integers of ``(s > u).sum()``.
    """
    x_sorted, unif_sorted = np.sort(s_x), np.sort(s_unif)
    em_t = np.zeros(t.shape[0])
    em_t[0] = 1.0
    for u in np.unique(s_x):
        above_x = s_x.size - np.searchsorted(x_sorted, u, side="right")
        above_unif = s_unif.size - np.searchsorted(unif_sorted, u, side="right")
        mass = 1.0 / s_x.size * above_x
        em_t = np.maximum(em_t, mass - t * above_unif / s_unif.size * volume)
    amax = np.argmax(em_t <= t_max) + 1
    # The code's amax = -1 when t_max is never reached: the area runs over t[:-1].
    reached = bool(amax != 1)
    amax = amax if reached else -1
    return float(auc(t[:amax], em_t[:amax])), reached


def excess_mass(
    a_normal, a_uniform, volume: float, t_max: float = EM_T_MAX
) -> tuple[float, bool]:
    """Excess-mass area, Goix (2016) Eqs. (3) and (5), as em.py computes it.

    The normality score is s = -a. ``a_uniform`` scores points drawn uniformly over an
    axis-aligned box of volume ``volume``, which estimates Leb(s > u) as volume
    #{s(U) > u} / n_uniform. The conventions are the code's: the strict ">" in both
    counts (the paper has ">="), EM(0) = 1, the grid t = arange(0, 100 / volume,
    0.01 / volume) of em_bench.py, and the area truncated at the first t with
    EM(t) <= t_max. Return the area and whether t_max was reached.
    """
    t = np.arange(0, 100 / volume, 0.01 / volume)
    return _em(t, t_max, volume, -_scores(a_uniform), -_scores(a_normal))


def _mv(axis_alpha: np.ndarray, volume: float, s_unif, s_x) -> float:
    """``mv`` of EMMV_benchmarks/em.py on normality scores, kept line for line."""
    order, unif_sorted = s_x.argsort(), np.sort(s_unif)
    mass, cpt, u = 0, 0, s_x[order[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_x[order[-cpt]]
            mass = 1.0 / s_x.size * cpt
        at_or_above = s_unif.size - np.searchsorted(unif_sorted, u, side="left")
        mv[i] = float(at_or_above) / s_unif.size * volume
    return float(auc(axis_alpha, mv))


def mass_volume(a_normal, a_uniform, volume: float, alphas=MV_ALPHAS) -> float:
    """Mass-volume area, Goix (2016) Eqs. (4) and (6), as em.py computes it.

    MV(alpha) is the estimated volume of the smallest upper level set of s = -a that
    holds a fraction alpha of ``a_normal``, with Leb(s >= u) estimated as volume
    #{s(U) >= u} / n_uniform (the code's ">=", as in the paper); the curve is
    integrated over ``alphas``, by default em_bench.py's arange(0.9, 0.999, 0.0001).
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    return _mv(alphas, volume, -_scores(a_uniform), -_scores(a_normal))


def loss_entropy(losses) -> float:
    """Loss entropy H_L of EntropyStop, Eqs. (9)-(10), p. 4, in nats.

    The per-sample losses are normalised to a distribution u and H_L = -sum u log u,
    without the 1e-7 that the code's ``cal_entropy`` adds inside the log, which lowers
    its value by about 1e-7 per sample. The criterion assumes positive losses (p. 3):
    with a loss that is not positive it is undefined, and NaN is returned with a
    warning.
    """
    losses = _scores(losses)
    if not (losses > 0).all():
        warnings.warn("loss_entropy needs positive losses; returning NaN.")
        return math.nan
    u = losses / losses.sum()
    return float(-(u * np.log(u)).sum())


def entropy_stop(curve, k: int = 100, r_down: float = 0.01) -> tuple[int, int]:
    """Offline EntropyStop: ``getStopOffline`` of EntropyEarlyStop.py, ported.

    A value becomes the accepted minimum when it lies below the current one by more
    than ``r_down`` times the total variation of the curve since that minimum was set
    (Sec. 4.3 and Algorithm 1, pp. 5-6). Return the index of the last accepted minimum
    and the index at which the patience reached ``k``, or the last index when it never
    did. The defaults are the code's; the paper sets k = 100 and R_down = 0.1 (App.
    B.2, p. 13). Kept line for line with ``EntropyEarlyStop.step``.
    """
    curve = [float(value) for value in curve]
    minimum, best, patience, variation = None, 0, 0, 0.0
    for i, value in enumerate(curve):
        if minimum is None:
            minimum = value
            continue
        variation += abs(value - curve[i - 1])
        if value < minimum and (minimum - value) / variation > r_down:
            minimum, best, patience, variation = value, i, 0, 0.0
        else:
            patience += 1
        if patience == k:
            break
    return best, i


def _ranks(scores) -> np.ndarray:
    # UOMS replaces non-finite scores with np.nan_to_num before ranking.
    values = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    ranks = rankdata(np.nan_to_num(values), axis=1)
    return ranks - ranks.mean(axis=1, keepdims=True)


def _rank_correlations(scores, others) -> np.ndarray:
    """Spearman's rho of ``scores``, (n,), with every row of ``others``, (m, n).

    Pearson's correlation of average ranks, as scipy's spearmanr; an undefined
    correlation (a constant score vector) counts as 0, as UOMS's np.nan_to_num sets it.
    """
    r, rows = _ranks(scores)[0], _ranks(others)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = rows @ r / np.sqrt((rows**2).sum(axis=1) * (r @ r))
    return np.nan_to_num(rho)


def udr(scores, others) -> float:
    """Seed agreement: the median Spearman correlation with the other seeds.

    UDR of Duan et al. (ICLR 2020, arXiv:1905.12614) as Ma et al. adapt it to outlier
    detection (UOMS, UDR.py): the scores of one run and those of the runs with the same
    hyperparameters and other seeds, ``others`` (m, n), on the same points, compared by
    Spearman's rho and aggregated by the median.
    """
    return float(np.median(_rank_correlations(scores, others)))


def model_centrality(scores, pool) -> float:
    """Model centrality: the mean Spearman correlation with every other model.

    MC of Ma et al. (UOMS, MC.py, the Spearman variant; Lin et al., ICML 2020,
    arXiv:1906.06034): ``pool`` (m, n) holds the scores of every other model of the
    pool on the same points.
    """
    return float(np.mean(_rank_correlations(scores, pool)))


def _neighbours(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours of every row, itself excluded, (n, k)."""
    distances, indices = cKDTree(x).query(x, k=k + 1)
    if (distances[:, 1:] == 0).any():
        raise ValueError("SDS needs distinct rows: sds.m divides by zero distances.")
    return indices[:, 1:]


def _edge_patterns(v: np.ndarray, threshold: float) -> np.ndarray:
    """Edge flags of sds.m, from the unit offsets v (n, k, d) of every row."""
    theta = np.einsum("ikd,id->ik", v, v.sum(axis=1))
    return 1 / v.shape[1] * (theta >= 0).sum(axis=1) >= 1 - threshold


def _pseudo_targets(x: np.ndarray, unit: np.ndarray, offsets: np.ndarray):
    # lambda_ij = (-n_i) . (x_j - x_i); a point with no positive lambda yields no
    # pseudo-target, as in sds.m.
    lam = np.einsum("id,ikd->ik", unit, offsets)
    shift = np.where(lam > 0, lam, np.inf).min(axis=1)
    keep = np.isfinite(shift)
    return x[keep] - shift[keep, None] * unit[keep]


def sds_pseudo_sets(
    x, k: int | None = None, threshold: float = SDS_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-targets and pseudo-outliers of self-adaptive data shifting, as sds.m.

    ``x`` (n, d) holds distinct rows, the data a detector is trained on. Per row, with
    k = ceil(5 log10 n) neighbours: v_ij = (x_i - x_j) / norm(x_i - x_j) and
    n_i = sum_j v_ij, and the row is an edge pattern when the fraction of j with
    v_ij . n_i >= 0 is at least 1 - threshold. Pseudo-outliers shift every edge pattern
    along n_i / norm(n_i) by the mean, over the edge patterns, of their mean neighbour
    distance. Pseudo-targets shift every row along -n_i / norm(n_i) by the smallest
    positive projection of its neighbour offsets on that direction. The generation is
    deterministic. Return (targets, outliers), (n_t, d) and (n_o, d).
    """
    x = np.asarray(x, dtype=np.float64)
    k = math.ceil(5 * math.log10(len(x))) if k is None else k
    offsets = x[:, None, :] - x[_neighbours(x, k)]
    distances = np.linalg.norm(offsets, axis=2)
    v = offsets / distances[:, :, None]
    normal = v.sum(axis=1)
    unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    edge = _edge_patterns(v, threshold)
    outliers = x[edge] + unit[edge] * distances[edge].mean(axis=1).mean()
    return _pseudo_targets(x, unit, offsets), outliers


def _rejected(scores, threshold: float, tie_weight: float) -> float:
    s = _scores(scores)
    return float(((s > threshold).sum() + tie_weight * (s == threshold).sum()) / s.size)


def sds_error(
    a_targets, a_outliers, threshold: float, tie_weight: float = 1.0
) -> float:
    """SDS error of a detector cut at ``threshold``: pseudo-targets rejected plus
    pseudo-outliers accepted, each as a fraction of its set.

    A score above the threshold is rejected, and one equal to it with weight
    ``tie_weight``, as ``AnomalyRate`` counts exceedances. The two fractions are
    summed as in main.m (``err_t_sds(1) + err_o_sds(2)``); the Python port averages
    them, which orders models identically.
    """
    accepted = 1.0 - _rejected(a_outliers, threshold, tie_weight)
    return _rejected(a_targets, threshold, tie_weight) + accepted
