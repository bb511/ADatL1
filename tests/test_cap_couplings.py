import json

import pytest
import torch

from src.callbacks.components.cap import bank
from src.callbacks.components.cap.binary import pairing
from src.evaluation.callbacks.components.cap import bank as evaluation_bank
from src.evaluation.callbacks.components.cap.binary import pairing as evaluation_pairing

PAIRINGS = [pairing, evaluation_pairing]
BANKS = [bank, evaluation_bank]


def _randn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _correlated(n, seed):
    mixing = torch.tensor(
        [[1.0, 0.5, 0.0], [0.0, 1.0, -0.3], [0.2, 0.0, 2.0]], dtype=torch.float64
    )
    return _randn(n, 3, seed=seed) @ mixing


@pytest.mark.parametrize("module", PAIRINGS)
def test_ot_recovers_a_monotone_map_of_one_coordinate(module):
    x_1 = _randn(300, 4)
    image = x_1.clone()
    # Nondecreasing and flat below zero.
    image[:, 2] = 3.0 * torch.relu(x_1[:, 2])
    shuffle = torch.randperm(300, generator=torch.Generator().manual_seed(1))
    coupling = module.ot_coupling(x_1, image[shuffle], n_subsamples=1, discard=0.0)

    (pairs,) = coupling.pairs(None, None)
    assert torch.equal(pairs.index_1, torch.arange(300))
    assert torch.equal(shuffle[pairs.index_2], pairs.index_1)


@pytest.mark.parametrize("module", PAIRINGS)
def test_ot_drops_the_costliest_pairs_of_every_subsample(module):
    x_1, x_2 = _randn(200, 3), _randn(700, 3, seed=1) + 0.5
    coupling = module.ot_coupling(x_1, x_2, n_subsamples=3, seed=2, discard=0.05)
    full = module.ot_coupling(x_1, x_2, n_subsamples=3, seed=2, discard=0.0)

    assert coupling.diagnostics()["discarded"] == pytest.approx(0.05)
    for kept, every in zip(coupling.pairs(None, None), full.pairs(None, None)):
        assert len(kept.index_2.unique()) == len(kept.index_2) == 190
        distance = (x_1[every.index_1] - x_2[every.index_2]).norm(dim=1)
        is_kept = torch.isin(every.index_1, kept.index_1)
        assert torch.equal(every.index_2[is_kept], kept.index_2)
        assert distance[is_kept].max() <= distance[~is_kept].min()
    subsamples = [set(p.index_2.tolist()) for p in full.pairs(None, None)]
    assert subsamples[0] != subsamples[1]


@pytest.mark.parametrize("module", PAIRINGS)
def test_ot_needs_the_larger_second_sample(module):
    with pytest.raises(ValueError):
        module.ot_coupling(_randn(20, 2), _randn(10, 2))


@pytest.mark.parametrize("module", PAIRINGS)
def test_nn_caliper_discards_the_far_events_and_partners_repeat(module):
    x_1 = torch.tensor([[0.0], [0.02], [10.0], [20.0], [30.0]], dtype=torch.float64)
    # Nearest-reference distances 0.2, 0.18, 0.2, 0.2, 0.6: the 0.9 quantile is 0.44.
    x_ref = torch.tensor([[0.2], [10.2], [20.2], [30.6]], dtype=torch.float64)
    x_2 = torch.tensor([[0.1], [10.25], [29.5], [100.0]], dtype=torch.float64)
    coupling = module.nn_coupling(x_1, x_2, x_ref, caliper=0.9)

    (pairs,) = coupling.pairs(None, None)
    assert pairs.index_1.tolist() == [0, 1, 2]
    assert pairs.index_2.tolist() == [0, 0, 1]
    diagnostics = coupling.diagnostics()
    assert diagnostics["discarded"] == pytest.approx(0.4)
    assert diagnostics["caliper"] == pytest.approx(0.44)
    assert diagnostics["mean_distance"] == pytest.approx((0.1 + 0.08 + 0.25) / 3)


@pytest.mark.parametrize("module", PAIRINGS)
def test_ot_and_nn_nulls_relabel_the_rows_of_x2(module):
    x_1, x_2, x_ref = _randn(100, 2), _randn(120, 2, seed=1), _randn(100, 2, seed=2)
    ot = module.ot_coupling(x_1, x_2, n_subsamples=3)
    nn = module.nn_coupling(x_1, x_2[:30], x_ref)
    for coupling, n_2 in ((ot, 120), (nn, 30)):
        pairs = coupling.pairs(None, None)
        partners = torch.cat([p.index_2 for p in pairs]).tolist()
        assert len(set(partners)) < len(partners)
        generator = torch.Generator().manual_seed(0)
        for draw in coupling.null_draws(None, torch.zeros(n_2), 20, generator):
            relabel = {}
            for observed, drawn in zip(pairs, draw, strict=True):
                assert torch.equal(drawn.index_1, observed.index_1)
                for j, k in zip(observed.index_2.tolist(), drawn.index_2.tolist()):
                    assert relabel.setdefault(j, k) == k
            assert len(set(relabel.values())) == len(relabel)
            assert any(j != k for j, k in relabel.items())


def _mahalanobis(x):
    centred = x - x.mean(dim=0)
    covariance = centred.T @ centred / (len(x) - 1)
    return (centred * torch.linalg.solve(covariance, centred.T).T).sum(dim=1)


@pytest.mark.parametrize("module", PAIRINGS)
def test_strata_are_equal_affine_invariant_mahalanobis_quantiles(module):
    x_1, x_2 = _correlated(1003, seed=0), _correlated(5407, seed=1) * 2.0
    coupling = module.strat_coupling(x_1, x_2, n_strata=10)

    for x, strata in ((x_1, coupling.strata_1), (x_2, coupling.strata_2)):
        counts = torch.bincount(strata, minlength=10)
        assert counts.sum() == len(x) and counts.max() - counts.min() <= 1
        assert (strata[_mahalanobis(x).argsort()].diff() >= 0).all()
    shift = torch.tensor([3.0, -1.0, 0.5], dtype=torch.float64)
    affine = module.strat_coupling(x_1 @ x_1[:3].T + shift, x_2 - shift, n_strata=10)
    assert torch.equal(affine.strata_1, coupling.strata_1)
    assert torch.equal(affine.strata_2, coupling.strata_2)


def _strat_case(module, seed=0):
    x_1, x_2 = _correlated(500, seed=seed), _correlated(1700, seed=seed + 1)
    scores_1 = x_1[:, 0] + _randn(500, seed=seed + 2)
    scores_2 = x_2[:, 0] + _randn(1700, seed=seed + 3)
    return module.strat_coupling(x_1, x_2, n_strata=5), scores_1, scores_2


def _within(coupling, pairs, k):
    in_k = coupling.strata_1[pairs.index_1] == k
    return pairs.index_1[in_k], pairs.index_2[in_k]


@pytest.mark.parametrize("module", PAIRINGS)
def test_strat_pairs_are_the_cdf_pairs_of_every_stratum(module):
    coupling, scores_1, scores_2 = _strat_case(module)
    (pairs,) = coupling.pairs(scores_1, scores_2)

    assert len(pairs.index_1) == 500
    for k in range(5):
        rows_1 = torch.nonzero(coupling.strata_1 == k).flatten()
        rows_2 = torch.nonzero(coupling.strata_2 == k).flatten()
        index_1, index_2 = module.cdf(scores_1[rows_1], scores_2[rows_2])
        expected = set(zip(rows_1[index_1].tolist(), rows_2[index_2].tolist()))
        index_1, index_2 = _within(coupling, pairs, k)
        assert set(zip(index_1.tolist(), index_2.tolist())) == expected
        order = scores_1[index_1].argsort()
        assert (scores_2[index_2][order].diff() > 0).all()


@pytest.mark.parametrize("module", PAIRINGS)
def test_strat_null_permutes_the_strata_of_x2_keeping_their_sizes(module):
    coupling, scores_1, scores_2 = _strat_case(module)
    (observed,) = coupling.pairs(scores_1, scores_2)
    generator = torch.Generator().manual_seed(0)
    draws = coupling.null_draws(scores_1, scores_2, 20, generator)

    assert len(draws) == 20
    for (draw,) in draws:
        assert torch.equal(draw.index_1, observed.index_1)
        assert len(draw.index_2.unique()) == len(draw.index_2)
        for k in range(5):
            index_1, index_2 = _within(coupling, draw, k)
            assert len(index_1) == (coupling.strata_1 == k).sum()
            order = scores_1[index_1].argsort()
            assert (scores_2[index_2][order].diff() > 0).all()
        assert not torch.equal(draw.index_2, observed.index_2)


def _fail(*args, **kwargs):
    raise AssertionError("built again although the cache holds it")


def _inputs():
    return _randn(120, 3), _randn(400, 3, seed=1) + 0.2, _randn(100, 3, seed=2)


PAIRING_CONFIG = {
    "cdf": {},
    "ot": {"n_subsamples": 2, "seed": 0, "discard": 0.05},
    "nn": {"caliper": 0.9},
    "strat": {"n_strata": 4},
}


@pytest.mark.parametrize("module", BANKS)
def test_coupling_cache_round_trip(module, tmp_path, monkeypatch):
    x_1, x_2, x_ref = _inputs()
    label = {"dataset_1": "normal", "dataset_2": "sensor_weak"}
    built = module.build_couplings(x_1, x_2, x_ref, PAIRING_CONFIG, tmp_path, label)

    names = sorted(path.name.split("_")[0] for path in tmp_path.iterdir())
    assert names == ["nn", "nn", "ot", "ot", "strat", "strat"]
    record = json.loads(next(tmp_path.glob("nn_*.json")).read_text())
    assert (record["n_1"], record["n_2"], record["n_ref"]) == (120, 400, 100)
    assert record["parameters"] == {"caliper": 0.9} and record["label"] == label
    assert record["diagnostics"] == built["nn"].diagnostics()

    for name in ("ot", "nn", "strat"):
        monkeypatch.setitem(module.COUPLINGS, name, _fail)
    loaded = module.build_couplings(x_1, x_2, x_ref, PAIRING_CONFIG, tmp_path, label)
    scores_1, scores_2 = _randn(120, seed=3), _randn(400, seed=4)
    for name in PAIRING_CONFIG:
        old = built[name].pairs(scores_1, scores_2)
        new = loaded[name].pairs(scores_1, scores_2)
        assert all(torch.equal(a, b) for p, q in zip(old, new) for a, b in zip(p, q))
    with pytest.raises(AssertionError):
        module.build_couplings(x_1, x_2 + 1.0, x_ref, PAIRING_CONFIG, tmp_path, label)


@pytest.mark.parametrize("module", BANKS)
def test_unknown_pairing_and_missing_reference_raise(module):
    x_1, x_2, x_ref = _inputs()
    with pytest.raises(ValueError):
        module.build_couplings(x_1, x_2, x_ref, {"knn": {}})
    with pytest.raises(ValueError):
        module.build_couplings(x_1, x_2, None, {"nn": {}})
