import pytest
import torch
from scipy.stats import wasserstein_distance

from src.callbacks.wasserstein import WassersteinCallback
from src.evaluation.callbacks.wasserstein import Wasserstein

CALLBACKS = [WassersteinCallback, Wasserstein]
# apply_log1p: scipy's W1 on the unequal pair (the old grid gave 1.7373 and 0.30348).
REGRESSION_UNEQUAL = {False: 1.5797307861173773, True: 0.30226396325401966}


@pytest.fixture(scope="module")
def samples() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.exp(torch.randn(2_000))
    y = torch.exp(0.5 + 1.2 * torch.randn(2_000))
    z = torch.exp(0.5 + 1.2 * torch.randn(50_000))
    return {"x": x, "y": y, "z": z}


def _w1(callback, x: torch.Tensor, y: torch.Tensor, apply_log1p: bool) -> float:
    cb = callback(
        output_name="s", dataset_1="a", dataset_2="b", apply_log1p=apply_log1p
    )
    cb.dataset_1_scores, cb.dataset_2_scores = [x], [y]
    return cb._compute_wasserstein()


@pytest.mark.parametrize("callback", CALLBACKS)
@pytest.mark.parametrize("apply_log1p", [False, True])
def test_unequal_sizes_match_scipy(samples, callback, apply_log1p):
    x, z = samples["x"], samples["z"]
    f = torch.log1p if apply_log1p else (lambda v: v)
    expected = wasserstein_distance(f(x).double().numpy(), f(z).double().numpy())
    new = _w1(callback, x, z, apply_log1p)
    assert new == pytest.approx(expected, rel=1e-6)
    assert new == pytest.approx(REGRESSION_UNEQUAL[apply_log1p], rel=1e-6)
    assert _w1(callback, z, x, apply_log1p) == pytest.approx(new, rel=1e-12)


def test_negative_scores_no_longer_collapse(samples):
    x, y = -samples["x"], -samples["y"]
    for callback in CALLBACKS:
        assert _w1(callback, x, y, apply_log1p=True) > 0.1
