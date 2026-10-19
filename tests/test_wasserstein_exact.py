import pytest
import torch
from scipy.stats import wasserstein_distance

from src.callbacks import wasserstein as train_module
from src.callbacks.wasserstein import WassersteinCallback
from src.evaluation.callbacks import wasserstein as eval_module
from src.evaluation.callbacks.wasserstein import Wasserstein

CALLBACKS = [WassersteinCallback, Wasserstein]
# apply_log1p: W1 of the code before October 2026 on the equal-size pair.
REGRESSION_EQUAL = {False: 1.398684024810791, True: 0.275943785905838}
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


def _old_equal_size_w1(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = torch.sort(x).values, torch.sort(y).values
    return torch.mean(torch.abs(x - y)).item()


def _transformed(module, monkeypatch, callback, x: torch.Tensor) -> torch.Tensor:
    captured = {}
    monkeypatch.setattr(
        module, "_wasserstein_1d", lambda a, b: captured.update(a=a) or 0.0
    )
    _w1(callback, x, x, apply_log1p=True)
    return captured["a"]


@pytest.mark.parametrize("callback", CALLBACKS)
@pytest.mark.parametrize("apply_log1p", [False, True])
def test_equal_sizes_match_the_old_formula(samples, callback, apply_log1p):
    x, y = samples["x"], samples["y"]
    if apply_log1p:
        old = _old_equal_size_w1(torch.log1p(x), torch.log1p(y))
    else:
        old = _old_equal_size_w1(x, y)
    new = _w1(callback, x, y, apply_log1p)
    assert new == pytest.approx(old, rel=1e-6)
    assert new == pytest.approx(REGRESSION_EQUAL[apply_log1p], rel=1e-6)


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


@pytest.mark.parametrize(
    "module, callback",
    [(train_module, WassersteinCallback), (eval_module, Wasserstein)],
)
def test_transform_is_log1p_for_non_negative_scores(module, callback, monkeypatch):
    torch.manual_seed(2)
    x = torch.cat([torch.zeros(10), 10.0 * torch.rand(1_000)])
    assert torch.equal(_transformed(module, monkeypatch, callback, x), torch.log1p(x))


@pytest.mark.parametrize(
    "module, callback",
    [(train_module, WassersteinCallback), (eval_module, Wasserstein)],
)
def test_transform_preserves_order_of_negative_scores(module, callback, monkeypatch):
    torch.manual_seed(3)
    x = -150.0 * torch.rand(1_000) + 20.0
    t = _transformed(module, monkeypatch, callback, x)
    assert (torch.diff(t[torch.argsort(x)]) >= 0).all()
    assert torch.equal(_transformed(module, monkeypatch, callback, -x), -t)
    assert (t[x < 0] < 0).all()


def test_negative_scores_no_longer_collapse(samples):
    x, y = -samples["x"], -samples["y"]
    for callback in CALLBACKS:
        assert _w1(callback, x, y, apply_log1p=True) > 0.1


@pytest.mark.parametrize("callback", CALLBACKS)
def test_nan_propagates(samples, callback):
    x = samples["x"].clone()
    x[7] = float("nan")
    for y in (samples["y"], samples["z"]):
        assert torch.isnan(torch.tensor(_w1(callback, x, y, apply_log1p=True)))


def test_empty_set_behaviour(samples):
    empty = torch.empty(0)
    with pytest.raises(RuntimeError):
        _w1(WassersteinCallback, empty, samples["y"], apply_log1p=True)
    assert torch.isnan(
        torch.tensor(_w1(Wasserstein, empty, samples["y"], apply_log1p=True))
    )
