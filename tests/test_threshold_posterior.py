import math
from decimal import Context, Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.callbacks import threshold_posterior as tp

ROOT = Path(__file__).resolve().parents[1]
CONFIG = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/cap/posterior.yaml"))
# Fewer pairs and draws than registered, for speed; the code paths are the same.
SMALL = CONFIG | {"n_pairs": 20, "n_draws": 30}
SIZES = (37, 55, 29)
LOG_T = math.log(401)
MAPS = {
    "exp": torch.exp,
    "cubic": lambda x: x**3 + x,
    "affine": lambda x: 3.0 * x - 2.0,
}
# Every way a set can make a column NaN; None empties the set.
BAD_VALUES = [math.inf, -math.inf, math.nan, None]


def _scores(n, seed, shift=0.0):
    # Rounded to one decimal, so that every set carries ties, within and across sets.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g, dtype=torch.float64) + shift
    return torch.round(10.0 * x) / 10.0


def _case(seed=0):
    n_1, n_2, n_p = SIZES
    return _scores(n_1, seed), _scores(n_2, seed + 1, 0.3), _scores(n_p, seed + 2, 1.5)


def _spoiled(scores, value):
    if value is None:
        return scores[:0]
    scores = scores.clone()
    scores[3] = value
    return scores


def _resampled(scores, generator):
    return scores[torch.randint(len(scores), (len(scores),), generator=generator)]


def _literal_agreement(cost_1, cost_2, beta):
    # Decimal, whose range holds exp(-2e6): no minimum is taken out, unlike the code.
    with localcontext(Context(prec=60, Emin=-(10**12), Emax=10**12)):
        beta = Decimal(beta)
        w_1 = [(-beta * Decimal(r)).exp() for r in cost_1]
        w_2 = [(-beta * Decimal(r)).exp() for r in cost_2]
        overlap = sum(a * b for a, b in zip(w_1, w_2)) / (sum(w_1) * sum(w_2))
        return float((len(cost_1) * overlap).ln())


def _literal_costs(sets, config, k, pooled):
    # The specification's cells and draws, written apart from the code under test.
    s_1, s_2, reference = (s.numpy() for s in sets)
    rng = np.random.default_rng([config["seed"], k])
    grid = tp.threshold_grid(s_1, s_2, config["levels"]).numpy()
    cells = len(grid) + 1
    h_1 = np.bincount(np.searchsorted(grid, s_1, "left"), minlength=cells)
    h_2 = np.bincount(np.searchsorted(grid, s_2, "left"), minlength=cells)
    g = np.bincount(np.searchsorted(grid, reference, "right"), minlength=cells)
    sources = (h_1 + h_2, h_1 + h_2) if pooled else (h_1, h_2)
    n_p, drawn = len(reference), []
    for source, n in zip(sources, (len(s_1), len(s_2))):
        h = rng.multinomial(n, source / source.sum(), size=config["n_pairs"])
        p = rng.multinomial(n_p, g / n_p, size=config["n_pairs"])
        drawn.append(1 - h.cumsum(1)[:, :-1] / n + p.cumsum(1)[:, :-1] / n_p)
    return drawn


def _per_pair_agreement(cost_1, cost_2, betas):
    # Each pair's log(|T| sum_T p_1 p_2), [n_pairs, n_beta], by plain log-softmaxes.
    curves = []
    for r_1, r_2 in zip(torch.as_tensor(cost_1), torch.as_tensor(cost_2)):
        log_p_1 = torch.log_softmax(-betas[:, None] * r_1, dim=-1)
        log_p_2 = torch.log_softmax(-betas[:, None] * r_2, dim=-1)
        curves.append(LOG_T + torch.logsumexp(log_p_1 + log_p_2, dim=-1))
    return torch.stack(curves)


def _brute_depth(scores, reference):
    values = sorted({*scores, *reference})
    between = [(a + b) / 2 for a, b in zip(values, values[1:])]
    taus = [values[0] - 1.0, *values, *between, values[-1] + 1.0]
    risks = [
        sum(s > t for s in scores) / len(scores)
        + sum(p < t for p in reference) / len(reference)
        for t in taus
    ]
    return 1.0 - min(risks)


def test_grid_is_the_lower_order_statistic_of_the_pooled_scores():
    s_1, s_2, _ = _case()
    grid = tp.threshold_grid(s_1, s_2, 400)
    pooled = torch.cat([s_1, s_2]).sort().values
    n = len(pooled)
    assert grid.shape == (401,) and grid.dtype == torch.float64
    expected = torch.stack([pooled[math.floor(j * (n - 1) / 400)] for j in range(401)])
    assert torch.equal(grid, expected)
    # n - 1 = 91 is prime to 400, so that no level falls on a rounding edge.
    levels = torch.arange(401, dtype=torch.float64) / 400
    quantiles = torch.quantile(pooled, levels, interpolation="lower")
    assert torch.equal(grid, quantiles)
    assert grid.unique().numel() < 401


def test_costs_count_both_inequalities_strictly():
    scores = torch.tensor([1.0, 2.0, 2.0, 3.0])
    reference = torch.tensor([2.0, 3.0, 3.0])
    grid = torch.tensor([2.0, 3.0, 2.5], dtype=torch.float64)
    expected = torch.tensor([1 / 4, 1 / 3, 1 / 4 + 1 / 3], dtype=torch.float64)
    cost = tp.costs(scores, reference, grid)
    torch.testing.assert_close(cost, expected, rtol=0, atol=0)


@pytest.mark.parametrize("seed", range(4))
def test_cell_counts_reproduce_costs_exactly_on_any_resample(seed):
    s_1, s_2, reference = _case(seed)
    grid = tp.threshold_grid(s_1, s_2, 400)
    # The grid holds duplicates and the reference ties with it, by the rounding.
    assert grid.unique().numel() < 401
    assert set(reference.tolist()) & set(grid.tolist())
    g = torch.Generator().manual_seed(seed)
    for _ in range(50):
        sample, points = _resampled(s_1, g), _resampled(reference, g)
        typical = tp._cells(sample, grid, right=False)[None]
        probes = tp._cells(points, grid, right=True)[None]
        cost = tp._cell_costs(typical, probes)[0]
        assert torch.equal(cost, tp.costs(sample, points, grid))


def test_a_multinomial_redraw_has_the_moments_of_a_redraw_of_the_rows():
    s_1, s_2, reference = _case(seed=2)
    grid, n_draws = tp.threshold_grid(s_1, s_2, 400), 4000
    cells = tp._cells(s_1, grid, right=False)
    probes = tp._cells(reference, grid, right=True)
    rng = np.random.default_rng(1)
    drawn = tp._redrawn_costs(rng, [cells], [len(s_1)], probes, n_draws)[0]
    g = torch.Generator().manual_seed(1)
    rows = torch.stack(
        [
            tp.costs(_resampled(s_1, g), _resampled(reference, g), grid)
            for _ in range(n_draws)
        ]
    )
    for j in (40, 120, 200, 280, 360):
        var_cells, var_rows = drawn[:, j].var().item(), rows[:, j].var().item()
        spread = math.sqrt((var_cells + var_rows) / n_draws)
        difference = abs(drawn[:, j].mean().item() - rows[:, j].mean().item())
        assert difference < 5 * spread, j
        # The relative error of a variance is about sqrt(2 / n) per sample.
        assert abs(var_cells / var_rows - 1) < 5 * math.sqrt(4 / n_draws), j


@pytest.mark.parametrize("beta", [0.01, 7.5, 1000.0, 1e6])
def test_agreement_is_that_of_a_plain_loop_over_the_thresholds(beta):
    s_1, s_2, reference = _case(seed=3)
    grid = tp.threshold_grid(s_1, s_2, 400)
    cost_1, cost_2 = (tp.costs(s, reference, grid) for s in (s_1, s_2))
    value = tp.agreement(cost_1, cost_2, torch.tensor([beta], dtype=torch.float64))
    expected = _literal_agreement(cost_1.tolist(), cost_2.tolist(), beta)
    assert value.item() == pytest.approx(expected, rel=0, abs=1e-8)


def test_the_underflow_cut_keeps_every_bit_of_the_partition():
    s_1, s_2, reference = _case(seed=3)
    grid = tp.threshold_grid(s_1, s_2, 400)
    cost = tp._from_zero(tp.costs(s_1, reference, grid))
    betas = tp.beta_grid(CONFIG)
    plain = (cost * -betas[:, None]).exp().sum(dim=-1).log()
    assert ((cost * -betas[:, None]) < tp.UNDERFLOW).any()
    assert torch.equal(tp._log_partition(cost, betas), plain)
    assert torch.tensor(tp.UNDERFLOW, dtype=torch.float64).exp().item() == 0.0


def test_agreement_stays_finite_at_the_largest_beta_with_disjoint_minimisers():
    cost_1, cost_2 = torch.ones(401).double(), torch.ones(401).double()
    cost_1[10], cost_2[300] = 0.0, 0.25
    beta = torch.tensor([1e6], dtype=torch.float64)
    value = tp.agreement(cost_1, cost_2, beta).item()
    expected = _literal_agreement(cost_1.tolist(), cost_2.tolist(), 1e6)
    assert math.isfinite(value) and value < -1e5
    assert value == pytest.approx(expected, rel=1e-12)


def test_agreement_is_exactly_zero_at_flat_posteriors():
    betas = tp.beta_grid(CONFIG)
    flat = torch.full((401,), 0.7, dtype=torch.float64)
    assert torch.equal(tp.agreement(flat, flat, betas), torch.zeros_like(betas))
    cost = torch.linspace(0.0, 2.0, 401, dtype=torch.float64)
    no_beta = torch.zeros(1, dtype=torch.float64)
    assert torch.equal(tp.agreement(cost, cost.flip(0), no_beta), no_beta)


def test_agreement_is_at_most_log_t():
    betas = tp.beta_grid(CONFIG)
    peaked = torch.ones(401, dtype=torch.float64)
    peaked[7] = 0.0
    assert tp.agreement(peaked, peaked, betas)[-1].item() == LOG_T
    s_1, s_2, reference = _case(seed=6)
    grid = tp.threshold_grid(s_1, s_2, 400)
    cost_1, cost_2 = (tp.costs(s, reference, grid) for s in (s_1, s_2))
    assert tp.agreement(cost_1, cost_2, betas).max().item() <= LOG_T + 1e-12


def test_beta_grid_has_forty_values_per_decade_from_end_to_end():
    betas = tp.beta_grid(CONFIG)
    assert betas.shape == (321,) and betas.dtype == torch.float64
    assert betas[0].item() == pytest.approx(0.01, rel=1e-12)
    assert betas[-1].item() == pytest.approx(1e6, rel=1e-12)
    steps = betas.log10().diff()
    torch.testing.assert_close(steps, torch.full_like(steps, 1 / 40))


def test_the_first_maximum_takes_the_smallest_beta_at_a_tie():
    betas = tp.beta_grid(CONFIG)
    flat = tp._first_maximum(torch.zeros_like(betas), betas)
    assert flat == (0.0, betas[0].item())
    curve = torch.zeros_like(betas)
    curve[[100, 200]] = 1.5
    assert tp._first_maximum(curve, betas) == (1.5, betas[100].item())


def test_flat_posteriors_give_no_capacity_at_the_smallest_beta():
    constant = torch.ones(40, dtype=torch.float64)
    reference = torch.full((30,), 2.0, dtype=torch.float64)
    stats = tp.capacity(constant, constant, reference, SMALL)
    assert stats["capacity"] == 0.0 and stats["confidence"] == 0.0
    assert stats["beta_star"] == tp.beta_grid(SMALL)[0].item()
    assert stats["saturated"] == 0.0


def test_the_curve_is_the_mean_of_the_logs_not_the_log_of_the_mean():
    # One pair whose posteriors share their peak and one whose peaks are disjoint: the
    # agreements are about 401 and 1e-6, so the two means sit nine nats apart.
    first, second = torch.ones(401).double(), torch.ones(401).double()
    first[10], second[300] = 0.0, 0.0
    cost_1, cost_2 = torch.stack([first, first]), torch.stack([first, second])
    betas = torch.tensor([20.0], dtype=torch.float64)
    per_pair = _per_pair_agreement(cost_1, cost_2, betas)
    curve = tp._mean_curve(cost_1, cost_2, betas)
    log_of_mean = per_pair.exp().mean(dim=0).log()
    torch.testing.assert_close(curve, per_pair.mean(dim=0), rtol=0, atol=1e-12)
    assert (log_of_mean - curve).abs().item() > 5.0


def test_capacity_and_confidence_are_literal_maxima_of_the_mean_over_pairs():
    # Well-shifted views give pairs that disagree by a varying amount, where the mean of
    # the logs and the log of the mean part.
    sets = [_scores(400, 1), _scores(400, 2, 0.5), _scores(300, 3, 2.5)]
    config = CONFIG | {"n_pairs": 40, "n_beta": 81}
    betas = tp.beta_grid(config)
    stats = tp.capacity(*sets, config)
    per_pair = {
        pooled: _per_pair_agreement(*_literal_costs(sets, config, k, pooled), betas)
        for k, pooled in ((0, False), (1, True))
    }
    own, confidence = (per_pair[pooled].mean(dim=0) for pooled in (False, True))
    assert stats["capacity"] == pytest.approx(own.max().item(), rel=0, abs=1e-8)
    assert stats["confidence"] == pytest.approx(confidence.max().item(), abs=1e-8)
    # Each maximum is its own: the confidence is not read at I's beta_star.
    assert stats["beta_star"] == betas[own.argmax()].item()
    assert own.argmax() != confidence.argmax()
    log_of_mean = per_pair[False].exp().mean(dim=0).log().max().item()
    assert log_of_mean - stats["capacity"] > 0.05


def test_chunks_leave_the_result_unchanged(monkeypatch):
    sets = _case(seed=1)
    whole = tp.capacity(*sets, SMALL)
    monkeypatch.setattr(tp, "PAIR_CHUNK", 3)
    assert tp.capacity(*sets, SMALL) == whole


def test_the_streams_follow_the_seed_rule():
    for k in (tp.PAIRS_STREAM, tp.POOLED_STREAM, tp.NULL_STREAM):
        drawn = tp._stream(SMALL, k).integers(2**31, size=8)
        expected = np.random.default_rng([SMALL["seed"], k]).integers(2**31, size=8)
        assert np.array_equal(drawn, expected)
    assert (tp.PAIRS_STREAM, tp.POOLED_STREAM, tp.NULL_STREAM) == (0, 1, 2)
    first, second = (tp._stream(SMALL, k).integers(2**31, size=8) for k in (0, 1))
    assert not np.array_equal(first, second)


def test_the_same_call_gives_the_same_result():
    s_1, s_2, reference = _case(seed=4)
    views = {"a": s_2}
    assert tp.ancap(s_1, views, reference, SMALL) == tp.ancap(
        s_1, views, reference, SMALL
    )


def test_a_grade_depends_neither_on_the_others_nor_on_their_order():
    s_1, s_2, reference = _case(seed=4)
    other = _scores(41, 9, 0.8)
    both = tp.ancap(s_1, {"a": s_2, "b": other}, reference, SMALL)
    swapped = tp.ancap(s_1, {"b": other, "a": s_2}, reference, SMALL)
    alone = tp.ancap(s_1, {"a": s_2}, reference, SMALL)
    assert both == swapped
    assert alone == {key: v for key, v in both.items() if not key.endswith("_b")}


def test_ancap_holds_the_named_columns_of_capacity_and_direction():
    s_1, s_2, reference = _case(seed=4)
    values = tp.ancap(s_1, {"weak": s_2}, reference, SMALL)
    stats = tp.capacity(s_1, s_2, reference, SMALL)
    assert values == {
        "ancap_weak": stats["capacity"],
        "ancap_beta_weak": stats["beta_star"],
        "ancap_conf_weak": stats["confidence"],
        "ancap_cons_weak": stats["consistency"],
        "ancap_sat_weak": stats["saturated"],
        "ancap_dir": tp.direction(s_1, reference, SMALL),
    }
    assert list(stats) == list(tp.KEYS)


def test_the_number_of_draws_moves_the_direction_check_alone():
    s_1, s_2, reference = _case(seed=4)
    few = tp.ancap(s_1, {"weak": s_2}, reference, SMALL)
    more = SMALL | {"n_draws": 2 * SMALL["n_draws"]}
    many = tp.ancap(s_1, {"weak": s_2}, reference, more)
    assert few.pop("ancap_dir") != many.pop("ancap_dir")
    assert len(few) == 5 and few == many


def test_both_views_of_the_confidence_are_drawn_from_the_pool(monkeypatch):
    calls, redrawn = [], tp._redrawn_costs

    def spy(rng, sources, sizes, reference, n_pairs):
        calls.append((sources, sizes, reference))
        return redrawn(rng, sources, sizes, reference, n_pairs)

    monkeypatch.setattr(tp, "_redrawn_costs", spy)
    s_1, s_2, reference = _case()
    tp.capacity(s_1, s_2, reference, SMALL)
    grid = tp.threshold_grid(s_1, s_2, 400)
    h_1, h_2 = (tp._cells(s, grid, right=False) for s in (s_1, s_2))
    probes = tp._cells(reference, grid, right=True)
    (own, own_sizes, own_probes), (pool, pool_sizes, pool_probes) = calls
    assert np.array_equal(own[0], h_1) and np.array_equal(own[1], h_2)
    assert all(np.array_equal(cells, h_1 + h_2) for cells in pool)
    assert own_sizes == pool_sizes == [len(s_1), len(s_2)]
    # The observed pair's thresholds serve both: the reference's cells are the same.
    assert np.array_equal(own_probes, probes)
    assert np.array_equal(pool_probes, probes)


def test_identical_views_give_a_consistency_near_zero():
    scores, reference = _scores(500, 0), _scores(400, 1, 2.0)
    stats = tp.capacity(scores, scores, reference, CONFIG | {"n_beta": 81})
    assert abs(stats["consistency"]) < 0.1
    # Streams 0 and 1 give the two their own pairs, though the sources coincide.
    assert stats["consistency"] != 0.0


def test_well_shifted_views_give_a_negative_consistency():
    views = _scores(500, 0), _scores(500, 1, 1.0)
    stats = tp.capacity(*views, _scores(400, 2, 2.5), CONFIG | {"n_beta": 81})
    assert stats["consistency"] < -0.5
    assert stats["capacity"] <= LOG_T and stats["confidence"] <= LOG_T


@pytest.mark.parametrize("seed", range(3))
def test_consistency_is_exactly_capacity_less_confidence(seed):
    stats = tp.capacity(*_case(seed), SMALL)
    assert stats["consistency"] == stats["capacity"] - stats["confidence"]


@pytest.mark.parametrize("seed", range(4))
def test_depth_is_that_of_a_loop_over_every_tau(seed):
    s_1, _, reference = _case(seed)
    expected = _brute_depth(s_1.tolist(), reference.tolist())
    assert tp.depth(s_1, reference) == pytest.approx(expected, rel=0, abs=1e-12)


def test_a_tie_between_the_sets_counts_as_no_error():
    scores = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    reference = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64)
    # At tau = 2 no score lies above it and no reference point below it.
    assert tp.depth(scores, reference) == 1.0


def test_depth_reads_every_tau_and_not_the_grid_of_thresholds():
    # Far more scores than thresholds, and a reference at one point between two of them:
    # the cost's minimum sits there, and 678 of the 5,000 scores lie above it.
    scores = torch.arange(5000, dtype=torch.float64)
    reference = torch.full((100,), 4321.5, dtype=torch.float64)
    grid = tp.threshold_grid(scores, scores, 400)
    on_the_grid = 1.0 - tp.costs(scores, reference, grid).min().item()
    assert tp.depth(scores, reference) == pytest.approx(1 - 678 / 5000, abs=1e-12)
    assert tp.depth(scores, reference) > on_the_grid + 1e-3


def test_depth_at_both_ends():
    scores = torch.zeros(10, dtype=torch.float64)
    assert tp.depth(scores, torch.ones(10, dtype=torch.float64)) == 1.0
    assert tp.depth(scores, -torch.ones(10, dtype=torch.float64)) == 0.0


def test_the_content_null_keeps_the_counts_and_follows_the_seed_rule():
    s_1, _, reference = _case(seed=2)
    draws = tp._null_depths(s_1, reference, SMALL)
    rng = np.random.default_rng([SMALL["seed"], 2])
    pool, n = torch.cat([s_1, reference]), len(s_1)
    expected = []
    for _ in range(SMALL["n_draws"]):
        perm = rng.permutation(len(pool))
        view, rest = pool[perm[:n]], pool[perm[n:]]
        assert (len(view), len(rest)) == (n, len(reference))
        assert torch.equal(torch.cat([view, rest]).sort().values, pool.sort().values)
        expected.append(_brute_depth(view.tolist(), rest.tolist()))
        # The running count of a draw agrees with the sorted depth to the last bit.
        assert draws[len(expected) - 1].item() == tp.depth(view, rest)
    assert draws.shape == (SMALL["n_draws"],) and draws.dtype == torch.float64
    expected = torch.tensor(expected, dtype=torch.float64)
    torch.testing.assert_close(draws, expected, rtol=0, atol=1e-12)
    assert torch.equal(tp._null_depths(s_1, reference, SMALL), draws)
    other = tp._null_depths(s_1, reference, SMALL | {"seed": 1})
    assert not torch.equal(other, draws)


def test_direction_is_the_depth_in_units_of_the_null():
    s_1, _, reference = _case(seed=2)
    draws = tp._null_depths(s_1, reference, SMALL).tolist()
    mean = sum(draws) / len(draws)
    sigma = math.sqrt(sum((d - mean) ** 2 for d in draws) / (len(draws) - 1))
    t = (tp.depth(s_1, reference) - mean) / sigma
    assert tp.direction(s_1, reference, SMALL) == pytest.approx(t, rel=1e-10)


def test_a_reference_above_the_scores_gives_a_large_t():
    scores = _scores(300, 0)
    assert tp.direction(scores, _scores(300, 1, 8.0), SMALL) > 10.0


def test_a_reference_below_the_scores_gives_no_large_t():
    scores = _scores(300, 0)
    assert tp.direction(scores, _scores(300, 1, -8.0), SMALL) < 0.0


def test_constant_scores_give_a_nan_direction():
    scores, reference = torch.ones(50), torch.ones(30)
    assert tp.depth(scores, reference) == 1.0
    assert math.isnan(tp.direction(scores, reference, SMALL))


@pytest.mark.parametrize("offset, expected", [(0.0, 1.0), (1.0, 1.0), (-1e-9, 0.0)])
def test_the_saturation_mark_reads_the_smallest_size(monkeypatch, offset, expected):
    sets = _case()
    beta_star = min(SIZES) + offset
    monkeypatch.setattr(tp, "_capacities", lambda *args: (1.0, beta_star, 0.5))
    assert tp.capacity(*sets, SMALL)["saturated"] == expected


def test_a_reference_above_every_score_saturates():
    scores = _scores(300, 0)
    stats = tp.capacity(scores, _scores(300, 1, 0.25), _scores(300, 2, 8.0), SMALL)
    assert stats["beta_star"] >= 300 and stats["saturated"] == 1.0


@pytest.mark.parametrize("name", MAPS)
def test_every_column_is_exactly_invariant_under_one_increasing_map(name):
    s_1, s_2, reference = _case(seed=5)
    values = tp.ancap(s_1, {"a": s_2}, reference, SMALL)
    f = MAPS[name]
    mapped = tp.ancap(f(s_1), {"a": f(s_2)}, f(reference), SMALL)
    assert all(math.isfinite(value) for value in values.values())
    assert mapped == values


def test_the_columns_change_under_different_maps_on_the_two_views():
    s_1, s_2, reference = _case(seed=5)
    values = tp.capacity(s_1, s_2, reference, SMALL)
    mapped = tp.capacity(s_1, MAPS["affine"](s_2), reference, SMALL)
    for key in ("capacity", "confidence", "consistency"):
        assert abs(mapped[key] - values[key]) > 1e-6, key


@pytest.mark.parametrize("value", BAD_VALUES)
def test_a_bad_view_2_touches_its_own_grade_alone(value):
    s_1, s_2, reference = _case(seed=4)
    other = _scores(41, 9, 0.8)
    clean = tp.ancap(s_1, {"a": s_2, "b": other}, reference, SMALL)
    bad = tp.ancap(s_1, {"a": s_2, "b": _spoiled(other, value)}, reference, SMALL)
    assert set(bad) == set(clean)
    for key, column in bad.items():
        if key.endswith("_b"):
            assert math.isnan(column), key
        else:
            assert column == clean[key], key


@pytest.mark.parametrize("value", BAD_VALUES)
@pytest.mark.parametrize("which", ["view_1", "reference"])
def test_a_bad_set_read_by_every_column_makes_each_nan(which, value):
    s_1, s_2, reference = _case(seed=4)
    if which == "view_1":
        s_1 = _spoiled(s_1, value)
    else:
        reference = _spoiled(reference, value)
    values = tp.ancap(s_1, {"a": s_2}, reference, SMALL)
    assert len(values) == 6
    assert all(math.isnan(column) for column in values.values())
