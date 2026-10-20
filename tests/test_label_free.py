import math

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.callbacks.metrics import label_free as lfc

# The hand example of lit/criteria/REPORT.md, Sec. 2(g): normality scores s = -a.
A_NORMAL = -np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
A_UNIFORM = -np.array([5.5, 3.5, 1.5, 0.5, 0.2, 0.1])
VOLUME = 6.0


def cal_entropy(score):
    # EntropyEarlyStop.py of github.com/goldenNormal/EntropyStop-KDD2024, verbatim.
    score = score.reshape(-1)
    score = score / np.sum(score)
    entropy = np.sum(-np.log(score + 10e-8) * score)
    return entropy


def test_npd_hand_example_ignores_the_sign():
    normal, noise = [1.0, 2.0, 3.0, 2.0], [3.0, 5.0, 7.0, 5.0]
    assert lfc.npd(normal, noise, eps=0.0) == pytest.approx(1.8)
    assert lfc.npd(noise, normal, eps=0.0) == pytest.approx(1.8)


def test_rtm_takes_the_top_tau_percent_with_ties():
    scores = [10.0, 0.0, 1.0, 10.0, 1.0, 2.0, 3.0, 3.0, 4.0, 10.0]
    # ceil(0.05 * 10) = 1 score sets the threshold 10, which three scores reach.
    assert lfc.rtm(scores, eps=0.0) == pytest.approx((10.0 - 3.0) / 3.0)


def test_eag_hand_example():
    scores = [3.0, 0.0, 12.0, 1.0, 2.0, 0.0, 10.0, 1.0, 3.0, 2.0]
    # k = 1 and 2: {12} against the rest, then {12, 10} against the rest.
    gap_1 = 0.1 * (86 / 9) ** 2 / (668 / 81)
    gap_2 = 0.16 * 9.5**2 / 1.2
    assert lfc.eag(scores, eps=0.0) == pytest.approx((gap_1 + gap_2) / 2)


def eag_loop(normal, eps=lfc.EPS):
    # The O(N^2) form: every split recomputed with np.mean and the two-pass np.var.
    s = np.sort(np.asarray(normal, dtype=np.float64))[::-1]
    count = min(lfc._top_count(lfc.EAG_TOP_PERCENT, s.size), s.size - 1)
    gaps = []
    for k in range(1, count + 1):
        high, low = s[:k], s[k:]
        w1, w0 = k / s.size, (s.size - k) / s.size
        gap = w0 * w1 * (low.mean() - high.mean()) ** 2
        gaps.append(gap / (w0 * low.var() + w1 * high.var() + eps))
    return float(np.mean(gaps))


@pytest.mark.parametrize("seed", range(5))
def test_eag_matches_the_loop(seed):
    rng = np.random.default_rng(seed)
    # Integer scores tie across the split points; the offset heavy tail checks the
    # running variances where the scores are large against their spread.
    tied = rng.integers(0, 12, size=997).astype(float)
    heavy = 1e3 + rng.lognormal(0.0, 2.0, size=503)
    for scores in (tied, heavy, np.concatenate([tied, heavy])):
        assert lfc.eag(scores) == pytest.approx(eag_loop(scores), rel=1e-9)


def test_em_port_reproduces_the_hand_example():
    t = np.array([0.0, 1 / 6, 1 / 3, 1 / 2])
    for t_max, area in ((0.9, 1 / 9), (0.2, 5 / 36)):
        result = lfc._em(t, t_max, VOLUME, -A_UNIFORM, -A_NORMAL)
        assert result == (pytest.approx(area), True)


def test_excess_mass_uses_the_code_grid():
    # On the grid t_i = i / 600, EM(t) = 5/6 - 3t first falls to 0.5 at t_67.
    t = 67 / 600
    area, reached = lfc.excess_mass(A_NORMAL, A_UNIFORM, VOLUME, t_max=0.5)
    assert area == pytest.approx(5 / 6 * t - 1.5 * t**2 + 1 / 7200)
    assert reached
    assert not lfc.excess_mass(A_NORMAL, A_UNIFORM, VOLUME, t_max=-1.0)[1]


def test_mass_volume_hand_example():
    alphas = [0.5, 0.75, 0.9]
    assert lfc.mass_volume(A_NORMAL, A_UNIFORM, VOLUME, alphas) == pytest.approx(0.75)
    grid = np.arange(0.9, 0.999, 0.001)
    assert lfc.mass_volume(A_NORMAL, A_UNIFORM, VOLUME, grid) == pytest.approx(0.294)
    assert lfc.mass_volume(A_NORMAL, A_UNIFORM, VOLUME) == pytest.approx(0.2967)


def test_goix_needs_a_finite_positive_volume():
    assert lfc.box_volume([0.0, 1.0], [2.0, 4.0]) == 6.0
    assert lfc.box_volume(np.zeros(200), np.full(200, 1e3)) == math.inf
    for volume in (math.inf, 0.0):
        with pytest.warns(UserWarning, match="volume"):
            area, reached = lfc.excess_mass(A_NORMAL, A_UNIFORM, volume)
        assert math.isnan(area) and not reached
        with pytest.warns(UserWarning, match="volume"):
            assert math.isnan(lfc.mass_volume(A_NORMAL, A_UNIFORM, volume))


def test_loss_entropy_matches_the_paper_and_the_code():
    cases = (([1, 1, 2], 1.5 * math.log(2)), ([2, 2, 2, 2], math.log(4)))
    for losses, entropy in cases:
        assert lfc.loss_entropy(losses) == pytest.approx(entropy, rel=1e-12)
        code = cal_entropy(np.array(losses, dtype=float))
        assert lfc.loss_entropy(losses) == pytest.approx(code, abs=1e-6)


def test_loss_entropy_is_undefined_for_non_positive_losses():
    with pytest.warns(UserWarning, match="positive"):
        assert math.isnan(lfc.loss_entropy([1.0, 0.0, 2.0]))


def test_entropy_stop_reproduces_get_stop_offline():
    curve = [5.0, 4.0, 4.5, 3.0, 3.2, 3.1, 3.3, 3.4]
    assert lfc.entropy_stop(curve, k=3, r_down=0.1) == (3, 6)
    curve = [5.0, 4.0, 4.9, 3.95, 3.9]
    assert lfc.entropy_stop(curve, k=3, r_down=0.1) == (1, 4)
    assert lfc.entropy_stop(curve, k=3, r_down=0.01) == (4, 4)


def test_udr_is_the_median_over_seeds():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    # rho = 1 and 1 - 6 * 2 / (5 * 24) = 0.9; a constant run counts as 0, as in UOMS.
    others = [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 3.0, 4.0, 5.0]]
    assert lfc.udr(scores, others) == pytest.approx(0.95)
    assert lfc.udr(scores, [[3.0] * 5, others[0]]) == pytest.approx(0.5)


def test_model_centrality_is_the_mean_over_the_pool():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    pool = [[5.0, 4.0, 3.0, 2.0, 1.0], [2.0, 1.0, 3.0, 4.0, 5.0], [1, 3, 2, 4, 5]]
    assert lfc.model_centrality(scores, pool) == pytest.approx((-1 + 0.9 + 0.9) / 3)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 40), rng.integers(0, 5, 40)
    assert lfc.model_centrality(a, [b]) == pytest.approx(spearmanr(a, b)[0])


def circle(n=36, centre=(2.0, -1.0)):
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)]) + np.array(centre)


def test_sds_shifts_along_the_local_normal():
    # k = ceil(5 log10 36) = 8: the neighbours at 10, 20, 30 and 40 degrees each side,
    # so every point is an edge pattern with a radial normal.
    x = circle()
    targets, outliers = lfc.sds_pseudo_sets(x)
    shift = np.mean([2 * np.sin(np.pi * m / 36) for m in range(1, 5)])
    centre = x.mean(axis=0)
    radii = [np.linalg.norm(v - centre, axis=1) for v in (targets, outliers)]
    assert targets.shape == outliers.shape == (36, 2)
    np.testing.assert_allclose(radii[0], np.cos(np.pi / 18))
    np.testing.assert_allclose(radii[1], 1.0 + shift)


def test_sds_error_counts_ties_with_the_weight():
    x = circle()
    targets, outliers = lfc.sds_pseudo_sets(x)
    distance = [np.linalg.norm(v - x.mean(axis=0), axis=1) for v in (targets, outliers)]
    assert lfc.sds_error(*distance, threshold=1.0) == 0.0
    # Rejected: 0.25 / 2 of the targets; accepted: 1 - 1.25 / 2 of the outliers.
    error = lfc.sds_error([1.0, 0.5], [1.0, 2.0], threshold=1.0, tie_weight=0.25)
    assert error == pytest.approx(0.125 + 0.375)


def test_sds_rejects_repeated_rows():
    x = circle()
    with pytest.raises(ValueError, match="distinct rows"):
        lfc.sds_pseudo_sets(np.vstack([x, x[:1]]))
