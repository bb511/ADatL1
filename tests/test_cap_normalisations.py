import pytest
import torch

from src.callbacks.metrics.cap.binary import get_normalizer_fn, normalization
from src.evaluation.callbacks.metrics.cap.binary import (
    get_normalizer_fn as evaluation_get_normalizer_fn,
)
from src.evaluation.callbacks.metrics.cap.binary import (
    normalization as evaluation_normalization,
)

MODULES = [normalization, evaluation_normalization]
FACTORIES = [get_normalizer_fn, evaluation_get_normalizer_fn]


def _gaussian(n=100_000, seed=0):
    return torch.randn(
        n, generator=torch.Generator().manual_seed(seed), dtype=torch.double
    )


def _scale(normalise, scores):
    # (x - centre) / logit(p) recovers the scale at the element nearest median + 1.
    centre = scores.median()
    i = torch.argmin((scores - centre - 1.0).abs())
    return ((scores[i] - centre) / torch.logit(normalise(scores)[i])).item()


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("name", ["rank", "rank_mid"])
def test_ranks_of_a_constant_are_one_half(module, name):
    normalised = getattr(module, name)(torch.full((7,), 3.0))
    assert torch.equal(normalised, torch.full((7,), 0.5))


@pytest.mark.parametrize("module", MODULES)
def test_tied_scores_share_their_mean_rank(module):
    scores = torch.tensor([3.0, 1.0, 3.0, 2.0])
    expected = torch.tensor([2.5, 0.0, 2.5, 1.0])

    torch.testing.assert_close(module.rank(scores), expected / 3)
    torch.testing.assert_close(module.rank_mid(scores), (expected + 0.5) / 4)


@pytest.mark.parametrize("module", MODULES)
def test_untied_ranks_are_the_positions(module):
    scores = torch.randn(50, generator=torch.Generator().manual_seed(1))
    positions = torch.empty(50)
    positions[torch.argsort(scores)] = torch.arange(50, dtype=torch.float)

    assert torch.equal(module.rank(scores), positions / 49)
    assert torch.equal(module.rank_mid(scores), (positions + 0.5) / 50)


@pytest.mark.parametrize("module", MODULES)
def test_log_sigmoid_is_finite_at_and_below_minus_one(module):
    scores = torch.tensor([-3.0, -1.0, -0.5, 0.0, 2.0, 10.0], dtype=torch.float32)
    assert torch.isfinite(module.log_sigmoid(scores)).all()


@pytest.mark.parametrize("module", MODULES)
def test_sigmoid_robust_matches_sigmoid_on_gaussian_scores(module):
    scores = _gaussian()
    robust = _scale(module.sigmoid_robust, scores)

    assert robust == pytest.approx(_scale(module.sigmoid, scores), rel=0.05)
    assert robust == pytest.approx(0.25, rel=0.05)


@pytest.mark.parametrize("module", MODULES)
def test_sigmoid_robust_ignores_one_extreme_score(module):
    scores = _gaussian()
    extreme = scores.clone()
    extreme[0] = 1e6

    robust = _scale(module.sigmoid_robust, extreme)
    assert robust == pytest.approx(_scale(module.sigmoid_robust, scores), rel=0.01)
    assert _scale(module.sigmoid, extreme) > 100 * _scale(module.sigmoid, scores)


@pytest.mark.parametrize("factory", FACTORIES)
def test_normalizer_fn_keeps_the_callers_mode(factory):
    params = {"mode": "split"}
    scores_1 = torch.linspace(0.0, 1.0, 11)
    scores_2 = torch.linspace(5.0, 7.0, 11)
    for _ in range(2):
        _, normalised_2 = factory("sigmoid", params)(scores_1, scores_2)

    assert params == {"mode": "split"}
    assert torch.equal(normalised_2, normalization.sigmoid(scores_2))


@pytest.mark.parametrize("factory", FACTORIES)
def test_sigmoid_robust_is_registered(factory):
    scores_1, scores_2 = _gaussian(n=100), _gaussian(n=100, seed=1)
    normalised_1, normalised_2 = factory("sigmoid_robust")(scores_1, scores_2)
    joint = normalization.sigmoid_robust(torch.cat([scores_1, scores_2]))

    assert torch.equal(torch.cat([normalised_1, normalised_2]), joint)
