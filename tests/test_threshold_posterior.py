import importlib.util
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from scipy.optimize import brentq

from src.callbacks import threshold_posterior as tp
from src.callbacks.components.cap import bank

ROOT = Path(__file__).resolve().parents[1]
CONFIG = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/cap/posterior.yaml"))
BANK = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/cap/bank.yaml"))
SCRIPT = ROOT / "scripts/analysis/posterior_beta_fix.py"
# Fewer draws than registered, for speed; the nulls' code paths are the same.
SMALL = CONFIG | {"n_draws": 30}
BETA = 7.5
SIZES = (37, 55, 29)
MAPS = {
    "exp": torch.exp,
    "cubic": lambda x: x**3 + x,
    "affine": lambda x: 3.0 * x - 2.0,
}


def _scores(n, seed, shift=0.0):
    # Rounded to one decimal, so that every set carries ties, within and across sets.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g, dtype=torch.float64) + shift
    return torch.round(10.0 * x) / 10.0


def _case(seed=0):
    n_1, n_2, n_p = SIZES
    return _scores(n_1, seed), _scores(n_2, seed + 1, 0.3), _scores(n_p, seed + 2, 1.5)


def _ranked(sets):
    return tp._ranked(torch.cat(sets))


def _stacked(chunks):
    # The counts of the two typical sets and the two probes of every draw, [n, N + 1].
    arrays = [[], [], [], []]
    for typical, probes in chunks:
        size = len(typical[0])
        for k, counts in enumerate([*typical, *probes]):
            arrays[k].append(counts.expand(size, -1))
    return [torch.cat(array) for array in arrays]


def _reference_cost(scores, probe, grid):
    scores, probe = scores.tolist(), probe.tolist()
    return [
        sum(s > t for s in scores) / len(scores)
        + sum(p < t for p in probe) / len(probe)
        for t in grid.tolist()
    ]


def _reference_kernel(cost_1, cost_2, beta):
    posteriors = []
    for cost in (cost_1, cost_2):
        weights = [math.exp(-beta * r) for r in cost]
        posteriors.append([w / sum(weights) for w in weights])
    p_1, p_2 = posteriors
    overlap = sum(a * b for a, b in zip(p_1, p_2))
    norms = math.sqrt(sum(a * a for a in p_1)) * math.sqrt(sum(b * b for b in p_2))
    size = len(cost_1)
    return math.log(size * overlap), math.log(overlap / norms), math.log(size * norms)


def _reference_transfer(s_1, s_2, probe, beta, n_draws, seed):
    generator = torch.Generator().manual_seed(seed)
    pool, n_1 = torch.cat([s_1, s_2]), len(s_1)
    draws = []
    for _ in range(n_draws):
        perm = torch.randperm(len(pool), generator=generator)
        view_1, view_2 = pool[perm[:n_1]], pool[perm[n_1:]]
        grid = tp.threshold_grid(view_1, view_2, 400)
        cost = [tp.costs(view, probe, grid) for view in (view_1, view_2)]
        draws.append({**tp.kernel(*cost, beta), "grid": grid})
    return draws


def _reference_content(s_1, s_2, probe, beta, n_draws, seed):
    generator = torch.Generator().manual_seed(seed)
    pools = [torch.cat([s_1, probe]), torch.cat([s_2, probe])]
    sizes = [len(s_1), len(s_2)]
    draws = []
    for _ in range(n_draws):
        perms = [torch.randperm(len(pool), generator=generator) for pool in pools]
        typical = [pool[p[:n]] for pool, p, n in zip(pools, perms, sizes)]
        probes = [pool[p[n:]] for pool, p, n in zip(pools, perms, sizes)]
        grid = tp.threshold_grid(*typical, 400)
        cost = [tp.costs(t, q, grid) for t, q in zip(typical, probes)]
        draws.append({**tp.kernel(*cost, beta), "grid": grid})
    return draws


REFERENCES = {"transfer": _reference_transfer, "content": _reference_content}
NULLS = {"transfer": tp._transfer_counts, "content": tp._content_counts}


def _two_level_entropy(beta, m, size):
    # m entries of the cost at 0 and the rest at 1: H = log Z + beta <R>.
    z = m + (size - m) * math.exp(-beta)
    return math.log(z) + beta * (size - m) * math.exp(-beta) / z


def _two_level_cost(m, size=401):
    return torch.cat([torch.zeros(m), torch.ones(size - m)]).double()


def _load_script():
    spec = importlib.util.spec_from_file_location("posterior_beta_fix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(directory, epochs):
    directory.mkdir()
    torch.save({}, directory / "meta.pt")
    for epoch, (normal, x2, probes) in enumerate(epochs):
        record = {
            "val": {"normal": {"ascore/full": normal}, "x2": {"ascore/full": x2}},
            "aux": {name: {"ascore/full": s} for name, s in probes.items()},
        }
        torch.save(record, directory / f"epoch_{epoch:04d}.pt")


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
    probe = torch.tensor([2.0, 3.0, 3.0])
    grid = torch.tensor([2.0, 3.0, 2.5], dtype=torch.float64)
    expected = torch.tensor([1 / 4, 1 / 3, 1 / 4 + 1 / 3], dtype=torch.float64)
    torch.testing.assert_close(tp.costs(scores, probe, grid), expected, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [0.05, 1.0, 7.5, 60.0])
def test_agreement_is_that_of_a_double_loop_over_the_thresholds(beta):
    s_1, s_2, probe = _case(seed=3)
    grid = tp.threshold_grid(s_1, s_2, 400)
    cost_1, cost_2 = (_reference_cost(s, probe, grid) for s in (s_1, s_2))
    a, c, f = _reference_kernel(cost_1, cost_2, beta)

    values = tp.threshold_posterior(s_1, s_2, probe, beta, SMALL)
    assert values["A-fix"] == pytest.approx(a, rel=0, abs=1e-12)
    assert values["C"] == pytest.approx(c, rel=0, abs=1e-12)
    assert values["F"] == pytest.approx(f, rel=0, abs=1e-12)
    assert values["A-fix"] == pytest.approx(values["C"] + values["F"], abs=1e-12)
    assert values["C"] <= 1e-15


def test_agreement_takes_one_beta_or_a_grid_and_gives_a_sup():
    sets = _case(seed=3)
    betas = torch.tensor([0.05, 1.0, 7.5, 60.0], dtype=torch.float64)
    curves = tp.agreement(*sets, betas, 400)
    assert {q: v.shape for q, v in curves.items()} == dict.fromkeys("ACF", (4,))
    grid = tp.threshold_grid(sets[0], sets[1], 400)
    cost_1, cost_2 = (_reference_cost(s, sets[2], grid) for s in sets[:2])
    for k, beta in enumerate(betas.tolist()):
        expected = _reference_kernel(cost_1, cost_2, beta)
        for q, value in zip("ACF", expected):
            assert curves[q][k].item() == pytest.approx(value, rel=0, abs=1e-12)
        single = tp.agreement(*sets, beta, 400)["A"]
        assert single.shape == (1,)
        assert single.item() == pytest.approx(curves["A"][k].item(), rel=0, abs=1e-14)

    values = tp.threshold_posterior(*sets, BETA, SMALL)
    curve = tp.agreement(*sets, tp.beta_grid(CONFIG), 400)["A"]
    assert values["A-sup"] == pytest.approx(curve.max().item(), rel=0, abs=1e-14)
    assert values["beta_star"] == tp.beta_grid(CONFIG)[curve.argmax()].item()


def test_flat_posteriors_give_zero():
    cost = torch.linspace(0.0, 2.0, 401, dtype=torch.float64)
    flat = torch.full_like(cost, 0.7)
    for values in (tp.kernel(flat, flat, 30.0), tp.kernel(cost, cost.flip(0), 0.0)):
        for value in values.values():
            assert value.item() == pytest.approx(0.0, abs=1e-12)


def test_a_sup_reads_the_bank_grid_and_flags_either_end():
    betas = tp.beta_grid(CONFIG)
    assert torch.equal(betas, bank._betas(BANK))
    log_beta = betas.log()
    rising = tp._supremum(log_beta, betas)
    falling = tp._supremum(-log_beta, betas)
    peaked = tp._supremum(-((log_beta - 1.0) ** 2), betas)
    assert rising["beta_star"] == betas[-1].item() and rising["at_bound"] == 1.0
    assert falling["beta_star"] == betas[0].item() and falling["at_bound"] == 1.0
    assert peaked["at_bound"] == 0.0
    assert peaked["beta_star"] == pytest.approx(math.e, rel=0.05)


@pytest.mark.parametrize("name", MAPS)
def test_every_output_is_invariant_under_one_increasing_map(name):
    sets = _case(seed=5)
    values = tp.threshold_posterior(*sets, BETA, SMALL)
    mapped = tp.threshold_posterior(*(MAPS[name](s) for s in sets), BETA, SMALL)
    assert list(values) == list(tp.KEYS)
    for key in tp.KEYS:
        assert math.isfinite(values[key]), key
        assert mapped[key] == pytest.approx(values[key], rel=0, abs=1e-10), key


def test_outputs_change_under_different_maps_on_the_two_views():
    s_1, s_2, probe = _case(seed=5)
    values = tp.threshold_posterior(s_1, s_2, probe, BETA, SMALL)
    mapped = tp.threshold_posterior(s_1, MAPS["affine"](s_2), probe, BETA, SMALL)
    for key in ("A-fix", "C", "F", "A_pi", "A_content"):
        assert abs(mapped[key] - values[key]) > 1e-6, key


@pytest.mark.parametrize("which", NULLS)
def test_nulls_are_seeded_and_keep_the_counts(which):
    sets, null = _case(), NULLS[which]
    ranked, (n_1, n_2, n_p) = _ranked(sets), SIZES
    first = _stacked(null(ranked, n_1, n_2, 25, 0))
    again = _stacked(null(ranked, n_1, n_2, 25, 0))
    other = _stacked(null(ranked, n_1, n_2, 25, 1))
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert not all(torch.equal(a, b) for a, b in zip(first, other))
    for counts, n in zip(first, (n_1, n_2, n_p, n_p)):
        assert counts.shape == (25, sum(SIZES) + 1)
        assert torch.equal(counts[:, -1], torch.full((25,), n))
    # Every draw relabels anew.
    assert len(first[0].unique(dim=0)) == 25


def test_transfer_null_keeps_the_grid_and_the_probe_term():
    sets = _case()
    ranked, (n_1, n_2, _) = _ranked(sets), SIZES
    position, size = ranked[1], len(ranked[1])
    typical_1, typical_2, probe_1, probe_2 = _stacked(
        tp._transfer_counts(ranked, n_1, n_2, 25, 0)
    )
    pooled = tp._counts(position[None, : n_1 + n_2], size)
    probe = tp._counts(position[None, n_1 + n_2 :], size)
    assert torch.equal(typical_1 + typical_2, pooled.expand(25, -1))
    assert torch.equal(probe_1, probe.expand(25, -1)) and torch.equal(probe_2, probe_1)
    grid = tp.threshold_grid(sets[0], sets[1], 400)
    for chunk in tp._transfer_counts(ranked, n_1, n_2, 25, 0):
        thresholds = tp._drawn_costs(ranked, *chunk, 400)[2]
        assert torch.equal(thresholds, grid.expand(len(thresholds), -1))


def test_content_null_keeps_each_pool_and_rebuilds_the_grid():
    sets = _case()
    ranked, (n_1, n_2, _) = _ranked(sets), SIZES
    position, size = ranked[1], len(ranked[1])
    typical_1, typical_2, probe_1, probe_2 = _stacked(
        tp._content_counts(ranked, n_1, n_2, 25, 0)
    )
    probe = position[n_1 + n_2 :]
    pool_1 = tp._counts(torch.cat([position[:n_1], probe])[None], size)
    pool_2 = tp._counts(torch.cat([position[n_1 : n_1 + n_2], probe])[None], size)
    assert torch.equal(typical_1 + probe_1, pool_1.expand(25, -1))
    assert torch.equal(typical_2 + probe_2, pool_2.expand(25, -1))
    chunks = tp._content_counts(ranked, n_1, n_2, 25, 0)
    thresholds = torch.cat([tp._drawn_costs(ranked, *c, 400)[2] for c in chunks])
    reference = _reference_content(*sets, BETA, 25, 0)
    assert torch.equal(thresholds, torch.stack([draw["grid"] for draw in reference]))
    grid = tp.threshold_grid(sets[0], sets[1], 400)
    assert any(not torch.equal(row, grid) for row in thresholds)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("which", NULLS)
def test_vectorised_nulls_equal_a_draw_by_draw_reference(which, chunked, monkeypatch):
    if chunked:
        # Chunks of three draws, so that the generator runs on across chunks.
        monkeypatch.setattr(tp, "CHUNK_ELEMENTS", 3 * sum(SIZES))
        assert tp._chunk_sizes(40, sum(SIZES)) == [3] * 13 + [1]
    sets = _case(seed=2)
    config = CONFIG | {"n_draws": 40, "seed": 11}
    draws = tp.null_draws(*sets, BETA, config, seed=3)[which]
    expected = REFERENCES[which](*sets, BETA, 40, 3)
    for q in ("A", "C", "F"):
        column = torch.stack([draw[q] for draw in expected])
        torch.testing.assert_close(draws[q], column, rtol=0, atol=1e-12)
    # Without a seed, the configuration's.
    default = tp.null_draws(*sets, BETA, config)[which]
    expected = REFERENCES[which](*sets, BETA, 40, 11)
    torch.testing.assert_close(default["A"], torch.stack([d["A"] for d in expected]))


def test_null_summaries_are_the_moments_of_the_reference_draws():
    sets, n_draws = _case(seed=4), SMALL["n_draws"]
    values = tp.threshold_posterior(*sets, BETA, SMALL)
    for which, names in (("transfer", tp.TRANSFER_KEYS), ("content", tp.CONTENT_KEYS)):
        draws = REFERENCES[which](*sets, BETA, n_draws, 0)
        for q, (mean_key, sigma_key) in names.items():
            column = [draw[q].item() for draw in draws]
            mean = sum(column) / n_draws
            # ddof 1, as the CAP bank's sigma_pi.
            sigma = math.sqrt(sum((v - mean) ** 2 for v in column) / (n_draws - 1))
            assert values[mean_key] == pytest.approx(mean, rel=0, abs=1e-12)
            assert values[sigma_key] == pytest.approx(sigma, rel=1e-9, abs=1e-14)
    assert values["D-fix"] == values["A-fix"] - values["A_pi"]
    assert values["beta_fix"] == BETA


@pytest.mark.parametrize("which, value", [(0, math.inf), (1, math.nan), (2, -math.inf)])
def test_non_finite_scores_give_nan_for_every_output(which, value):
    sets = [s.clone() for s in _case()]
    sets[which][3] = value
    values = tp.threshold_posterior(*sets, BETA, SMALL)
    assert list(values) == list(tp.KEYS) and values["beta_fix"] == BETA
    assert all(math.isnan(values[key]) for key in tp.KEYS if key != "beta_fix")


@pytest.mark.parametrize("m", [1, 5, 20])
def test_half_entropy_beta_of_a_two_level_cost(m):
    cost, size = _two_level_cost(m), 401
    for beta in (0.1, 1.0, 10.0, 100.0):
        expected = _two_level_entropy(beta, m, size)
        assert tp.entropy(cost, beta) == pytest.approx(expected, rel=1e-12, abs=1e-13)
    target = 0.5 * math.log(size)
    root = brentq(
        lambda b: _two_level_entropy(b, m, size) - target, 1e-6, 1e6, xtol=1e-14
    )
    beta = tp.half_entropy_beta(cost)
    assert beta == pytest.approx(root, rel=1e-9)
    assert tp.entropy(cost, beta) == pytest.approx(target, abs=1e-9)


def test_entropy_falls_with_beta():
    s_1, s_2, probe = _case(seed=7)
    cost = tp.costs(s_1, probe, tp.threshold_grid(s_1, s_2, 400))
    betas = torch.logspace(-3, 4, 300, dtype=torch.float64).tolist()
    entropies = [tp.entropy(cost, beta) for beta in betas]
    assert entropies[0] == pytest.approx(math.log(401), abs=1e-3)
    assert all(b <= a + 1e-12 for a, b in zip(entropies, entropies[1:]))
    beta = tp.half_entropy_beta(cost)
    assert math.isfinite(beta)
    assert tp.entropy(cost, beta) == pytest.approx(0.5 * math.log(401), abs=1e-9)


@pytest.mark.parametrize("m", [21, 200, 401])
def test_half_entropy_beta_is_infinite_with_enough_minimisers(m):
    # (1/2) log 401 lies between log 20 and log 21.
    assert tp.half_entropy_beta(_two_level_cost(m)) == math.inf


def test_beta_fix_script_pools_the_last_epochs_and_lets_inf_into_the_median(
    tmp_path,
):
    script = _load_script()
    config = {"levels": 400, "probes": ["autouad", "uniform"]}
    # uniform lies above every typical score, so that R_1 is zero on every threshold
    # of x2 and p_1 keeps more than half the entropy; a NaN leaves run_c's autouad
    # value out.
    constant = torch.zeros(10)
    expected = []
    for seed, run in enumerate(("run_a", "run_b", "run_c")):
        g = torch.Generator().manual_seed(seed)
        normal, x2 = torch.randn(200, generator=g), torch.randn(300, generator=g) + 100
        autouad = torch.randn(150, generator=g) + 3.0
        if run == "run_c":
            autouad[0] = math.nan
        probes = {"autouad": autouad, "uniform": torch.full((80,), 1e3)}
        old = (constant, constant, {"autouad": constant, "uniform": constant})
        _write_run(tmp_path / run, [old, (normal, x2, probes)])
        grid = tp.threshold_grid(normal, x2, 400)
        expected.append(tp.half_entropy_beta(tp.costs(normal, autouad, grid)))
    (tmp_path / "no_meta").mkdir()

    runs = script._runs([tmp_path])
    assert [run.name for run in runs] == ["run_a", "run_b", "run_c"]
    values = script.betas(runs, ["x2"], config)
    assert all(math.isfinite(beta) for beta in expected[:2])
    assert values["autouad"][:2] == expected[:2] and math.isnan(values["autouad"][2])
    assert values["uniform"] == [math.inf] * 3
    median = (expected[0] + expected[1]) / 2
    lines = script.report(values).splitlines()
    assert lines[0] == (
        "autouad: 2 values (1 non-finite left out, 0 +inf), "
        f"median beta = {median:.6g}"
    )
    assert lines[1] == (
        "uniform: 3 values (0 non-finite left out, 3 +inf), median beta = inf"
    )
    assert lines[2] == f"beta_fix: {{ autouad: {median:.6g}, uniform: .inf }}"


def test_beta_fix_median_takes_inf_as_the_largest_value():
    script = _load_script()
    assert script._summary("p", [1.0, math.inf, 2.0, math.nan])[1] == 2.0
    assert script._summary("p", [1.0, math.inf, math.inf, 2.0])[1] == math.inf
    assert script._yaml(math.inf) == ".inf" and script._yaml(2.5) == "2.5"
