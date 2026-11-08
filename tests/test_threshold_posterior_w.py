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
# Fewer pairs than registered, for speed; the code paths are the same.
SMALL = CONFIG | {"n_pairs": 12}
SIZES = (240, 410, 150)
ALPHA = CONFIG["alpha"]


def _scores(n, seed, shift=0.0, scale=1.0):
    # Rounded to two decimals, so that every set carries ties.
    g = torch.Generator().manual_seed(seed)
    x = scale * torch.randn(n, generator=g, dtype=torch.float64) + shift
    return torch.round(100.0 * x) / 100.0


def _case(seed=0, shift_2=0.2, shift_p=2.0):
    n_1, n_2, n_p = SIZES
    return (
        _scores(n_1, seed),
        _scores(n_2, seed + 1, shift_2),
        _scores(n_p, seed + 2, shift_p),
    )


def _literal_a(m_1, m_2, beta, alpha):
    with localcontext(Context(prec=60, Emin=-(10**12), Emax=10**12)):
        b, a = Decimal(beta), Decimal(alpha)
        c = (a / (1 - a)).ln()
        p_1 = 1 / (1 + (-(b * Decimal(m_1) + c)).exp())
        p_2 = 1 / (1 + (-(b * Decimal(m_2) + c)).exp())
        return float((p_1 * p_2 / a + (1 - p_1) * (1 - p_2) / (1 - a)).ln())


def _same(x, y):
    return all(
        (math.isnan(x[k]) and math.isnan(y[k])) or x[k] == y[k] for k in x
    ) and set(x) == set(y)


@pytest.mark.parametrize("n", [1, 2, 7, 240, 1000, 5400])
@pytest.mark.parametrize("q", [0.25, 0.75, 1.0 - ALPHA])
def test_the_observed_quantile_is_torch_quantile_to_the_bit(n, q):
    scores = _scores(n, n).sort().values
    assert torch.equal(tp._observed_quantile(scores, n, q), torch.quantile(scores, q))


@pytest.mark.parametrize("n", [100, 1000, 5400, 60000])
def test_the_cut_of_a_tail_is_that_of_the_whole_set(n):
    scores = _scores(n, n)
    tail = tp.tail_of(scores)
    assert len(tail.top) == math.ceil(n / 20) == tp.kept_size(n)
    cut = tp._observed_quantile(tail.top, n, 1.0 - ALPHA)
    assert torch.equal(cut, torch.quantile(scores, 1.0 - ALPHA))


def test_a_quantile_below_the_tail_is_nan():
    tail = tp.tail_of(_scores(400, 3))
    assert torch.isnan(tp._observed_quantile(tail.top, 400, 0.5))


@pytest.mark.parametrize("q", [0.25, 0.75, 1.0 - ALPHA])
def test_a_redrawn_quantile_is_torch_quantile_of_the_redrawn_rows(q):
    n, pairs = 240, 9
    ordered = _scores(n, 4).sort().values
    counts = tp._redraw_cells(np.random.default_rng(8), n, 0, n, pairs)
    rows = np.random.default_rng(8).integers(0, n, size=(pairs, n))
    expected = torch.stack(
        [torch.quantile(ordered[torch.from_numpy(r)], q) for r in rows]
    )
    found = tp._redrawn_quantile(counts, tp._cell_values(ordered), q)
    assert torch.equal(found, expected)


@pytest.mark.parametrize("size", [410, 240])
def test_a_tail_redraws_the_cut_of_the_whole_set_to_the_bit(size):
    view = _scores(410, 5)
    whole = tp.Tail(view.sort().values, 410)
    cuts = [
        tp._tail_cut(np.random.default_rng(3), t, size, SMALL)
        for t in (tp.tail_of(view), whole)
    ]
    assert torch.equal(cuts[0], cuts[1])
    assert torch.isfinite(cuts[0]).all()


def test_the_pool_redraws_the_cut_of_the_whole_pool_to_the_bit():
    view_1, view_2, _ = _case()
    pool = tp._pool(view_1, tp.tail_of(view_2))
    whole = tp.Tail(
        torch.cat([view_1, view_2]).sort().values, len(view_1) + len(view_2)
    )
    for size in (len(view_1), len(view_2)):
        cuts = [
            tp._tail_cut(np.random.default_rng(6), t, size, SMALL)
            for t in (pool, whole)
        ]
        assert torch.equal(cuts[0], cuts[1])


def test_a_cut_that_reaches_the_lump_is_nan():
    tail = tp.Tail(torch.tensor([5.0], dtype=torch.float64), 20)
    cuts = tp._tail_cut(np.random.default_rng(0), tail, 20, SMALL | {"n_pairs": 200})
    assert torch.isnan(cuts).any()


@pytest.mark.parametrize("beta", [0.01, 1.0, 37.0, 1e3, 1e6])
def test_the_agreement_sums_are_the_literal_agreement(beta):
    m = torch.tensor(
        [[-3.0, -0.2, 0.0, 1e-4, 0.3, 2.5, 40.0, -40.0]], dtype=torch.float64
    )
    m_2 = m.flip(-1) * 0.7
    betas = torch.tensor([beta], dtype=torch.float64)
    for i in range(m.shape[1]):
        one = tp._agreement_sums(
            m[:, i : i + 1], m_2[:, i : i + 1], torch.ones(1, 1), betas, ALPHA
        )
        expected = _literal_a(m[0, i].item(), m_2[0, i].item(), beta, ALPHA)
        assert one.item() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_the_prior_gives_no_agreement():
    m = torch.randn(3, 50, dtype=torch.float64)
    sums = tp._agreement_sums(m, -m, torch.ones(3, 50), torch.zeros(1), ALPHA)
    assert sums.abs().max() < 1e-14


def test_points_flagged_by_both_views_far_above_give_log_one_over_alpha():
    m = torch.full((1, 4), 1e4, dtype=torch.float64)
    sums = tp._agreement_sums(m, m, torch.ones(1, 4), torch.tensor([1e6]), ALPHA)
    assert sums.item() / 4 == pytest.approx(math.log(1 / ALPHA), abs=1e-12)


def test_ancap_w_holds_the_named_columns():
    view_1, view_2, reference = _case()
    values = tp.ancap_w(view_1, {"a": view_2, "b": view_2 + 0.1}, reference, SMALL)
    names = {f"{p}_{g}" for g in "ab" for p in tp.W_COLUMNS.values()} | {*tp.W_LEVELS}
    assert set(values) == names
    assert all(math.isfinite(v) for v in values.values())


def test_a_whole_view_and_its_shuffled_tail_give_the_same_bits():
    view_1, view_2, reference = _case()
    top = tp.tail_of(view_2).top
    shuffled = tp.Tail(top[torch.randperm(len(top))], len(view_2))
    whole = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    assert _same(whole, tp.ancap_w(view_1, {"g": shuffled}, reference, SMALL))


def test_a_grade_depends_neither_on_the_others_nor_on_their_order():
    view_1, view_2, reference = _case()
    views = {"a": view_2 - 0.3, "b": view_2}
    both = tp.ancap_w(view_1, views, reference, SMALL)
    alone = tp.ancap_w(view_1, {"b": view_2}, reference, SMALL)
    assert all(both[k] == alone[k] for k in alone)
    flipped = tp.ancap_w(view_1, dict(reversed(views.items())), reference, SMALL)
    assert _same(both, flipped)


def test_the_same_call_gives_the_same_result_and_the_seed_moves_it():
    view_1, view_2, reference = _case()
    first = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    assert _same(first, tp.ancap_w(view_1, {"g": view_2}, reference, SMALL))
    other = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL | {"seed": 1})
    assert other["ancapw_g"] != first["ancapw_g"]
    assert other["ancapw_f1"] == first["ancapw_f1"]


def test_the_streams_follow_the_seed_rule():
    view_1, view_2, reference = _case()
    tail, n_1, n_p, pairs = tp.tail_of(view_2), len(view_1), len(reference), 12
    shared = tp._shared(SMALL, 0, view_1.sort().values, n_p)
    rng = np.random.default_rng([SMALL["seed"], 0])
    view_draws = rng.integers(0, n_1, size=(pairs, n_1))
    reference_draws = rng.integers(0, n_p, size=(pairs, n_p))
    fresh = tp._tail_cut(rng, tail, len(view_2), SMALL)
    for _ in range(2):
        assert torch.equal(
            tp._tail_cut(tp._restored(shared), tail, len(view_2), SMALL), fresh
        )
    counts = np.stack([np.bincount(r, minlength=n_p) for r in reference_draws])
    assert torch.equal(shared.weights, torch.from_numpy(counts).double())
    ordered = view_1.sort().values
    taus = [
        torch.quantile(ordered[torch.from_numpy(r)], 1.0 - ALPHA) for r in view_draws
    ]
    assert torch.equal(shared.tau, torch.stack(taus))


def test_the_confidence_draws_both_cuts_from_the_pool_in_order(monkeypatch):
    view_1, view_2, reference = _case()
    seen = []
    cut = tp._tail_cut

    def spy(rng, tail, size, config):
        seen.append((tail.n, size))
        return cut(rng, tail, size, config)

    monkeypatch.setattr(tp, "_tail_cut", spy)
    tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    n_1, n_2 = len(view_1), len(view_2)
    assert seen == [(n_2, n_2), (n_1 + n_2, n_1), (n_1 + n_2, n_2)]


def test_chunks_leave_the_result_unchanged(monkeypatch):
    view_1, view_2, reference = _case()
    first = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    monkeypatch.setattr(tp, "W_PAIR_CHUNK", 5)
    second = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    assert all(
        first[k] == pytest.approx(second[k], rel=1e-13, abs=1e-13) for k in first
    )
    assert first["ancapw_beta_g"] == second["ancapw_beta_g"]


def test_an_increasing_affine_map_leaves_every_value():
    view_1, view_2, reference = _case()
    first = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    mapped = [3.0 * s - 2.0 for s in (view_1, view_2, reference)]
    second = tp.ancap_w(mapped[0], {"g": mapped[1]}, mapped[2], SMALL)
    assert all(first[k] == pytest.approx(second[k], rel=1e-9, abs=1e-9) for k in first)


def test_a_reference_far_above_reaches_log_one_over_alpha():
    view_1, view_2, reference = _case(shift_p=100.0)
    values = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    assert values["ancapw_g"] == pytest.approx(math.log(1 / ALPHA), abs=1e-6)
    assert values["ancapw_f1"] == values["ancapw_f2_g"] == 1.0


def test_a_reference_far_below_reaches_the_unflagged_limit():
    view_1, view_2, reference = _case(shift_p=-100.0)
    values = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    assert values["ancapw_g"] == pytest.approx(-math.log1p(-ALPHA), abs=1e-9)
    assert values["ancapw_f1"] == values["ancapw_f2_g"] == 0.0


def test_the_levels_are_the_shares_above_the_observed_cuts():
    view_1, view_2, reference = _case()
    values = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    tau_1, tau_2 = (torch.quantile(s, 1.0 - ALPHA) for s in (view_1, view_2))
    assert values["ancapw_f1"] == (reference > tau_1).double().mean().item()
    both = (reference > max(tau_1, tau_2)).double().mean().item()
    assert values["ancapw_f2_g"] == both


def test_capacity_lies_between_zero_and_log_one_over_alpha_and_consistency_is_the_difference():
    for seed in range(4):
        view_1, view_2, reference = _case(seed)
        values = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
        assert -1e-12 <= values["ancapw_g"] <= math.log(1 / ALPHA)
        assert values["ancapw_cons_g"] == values["ancapw_g"] - values["ancapw_conf_g"]
        assert values["ancapw_end_g"] in (0.0, 1.0)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, None])
def test_a_bad_view_2_makes_nan_of_its_grade_alone(bad):
    view_1, view_2, reference = _case()
    spoiled = view_2[:0] if bad is None else torch.cat([view_2, torch.tensor([bad])])
    values = tp.ancap_w(view_1, {"bad": spoiled, "good": view_2}, reference, SMALL)
    assert all(math.isnan(values[f"{p}_bad"]) for p in tp.W_COLUMNS.values())
    assert all(math.isfinite(values[f"{p}_good"]) for p in tp.W_COLUMNS.values())


@pytest.mark.parametrize("which", [0, 2])
@pytest.mark.parametrize("bad", [math.inf, math.nan, None])
def test_a_bad_view_1_or_reference_makes_nan_of_every_column(which, bad):
    sets = list(_case())
    sets[which] = (
        sets[which][:0]
        if bad is None
        else torch.cat([sets[which], torch.tensor([bad])])
    )
    values = tp.ancap_w(sets[0], {"g": sets[1]}, sets[2], SMALL)
    assert all(math.isnan(v) for v in values.values())


def test_a_scale_of_zero_makes_nan_of_every_column():
    view_1, view_2, reference = _case()
    flat = torch.cat([torch.zeros(230, dtype=torch.float64), view_1[:10]])
    values = tp.ancap_w(flat, {"g": view_2}, reference, SMALL)
    assert all(math.isnan(v) for v in values.values())


def test_a_record_that_keeps_the_wrong_number_of_values_raises():
    view_1, view_2, reference = _case()
    short = tp.Tail(tp.tail_of(view_2).top[1:], len(view_2))
    with pytest.raises(ValueError):
        tp.ancap_w(view_1, {"g": short}, reference, SMALL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_gpu_agrees_with_the_cpu():
    view_1, view_2, reference = _case()
    cpu = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL)
    gpu = tp.ancap_w(view_1, {"g": view_2}, reference, SMALL, device="cuda")
    assert all(cpu[k] == pytest.approx(gpu[k], rel=1e-12, abs=1e-12) for k in cpu)
    assert cpu["ancapw_beta_g"] == gpu["ancapw_beta_g"]
