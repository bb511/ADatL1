import pytest
import torch

from src.callbacks.components.cap.binary import normalization
from src.evaluation.callbacks.components.cap.binary import (
    normalization as evaluation_normalization,
)

MODULES = [normalization, evaluation_normalization]


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
def test_tied_scores_share_their_mean_rank(module):
    scores = torch.tensor([3.0, 1.0, 3.0, 2.0])
    expected = torch.tensor([2.5, 0.0, 2.5, 1.0])

    torch.testing.assert_close(module.rank(scores), expected / 3)
    torch.testing.assert_close(module.rank_mid(scores), (expected + 0.5) / 4)


@pytest.mark.parametrize("module", MODULES)
def test_log_sigmoid_is_finite_at_and_below_minus_one(module):
    scores = torch.tensor([-3.0, -1.0, -0.5, 0.0, 2.0, 10.0], dtype=torch.float32)
    assert torch.isfinite(module.log_sigmoid(scores)).all()


@pytest.mark.parametrize("module", MODULES)
def test_sigmoid_robust_ignores_one_extreme_score(module):
    scores = _gaussian()
    extreme = scores.clone()
    extreme[0] = 1e6

    robust = _scale(module.sigmoid_robust, extreme)
    assert robust == pytest.approx(_scale(module.sigmoid_robust, scores), rel=0.01)
    assert _scale(module.sigmoid, extreme) > 100 * _scale(module.sigmoid, scores)
