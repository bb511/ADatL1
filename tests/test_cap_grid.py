import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.callbacks.cap import CAPCallback
from src.callbacks.metrics.cap import kernel
from src.callbacks.metrics.cap.binary import get_pairing_fn
from src.callbacks.metrics.cap.metric import ApproximationCapacity, PosteriorConsistency
from src.evaluation.callbacks.metrics.cap import kernel as evaluation_kernel
from src.evaluation.callbacks.metrics.cap.metric import (
    ApproximationCapacity as EvaluationCapacity,
)
from src.evaluation.callbacks.metrics.cap.metric import (
    PosteriorConsistency as EvaluationConsistency,
)

CAP_CLASSES = [ApproximationCapacity, EvaluationCapacity]
CONSISTENCY_CLASSES = [PosteriorConsistency, EvaluationConsistency]
KERNELS = [kernel, evaluation_kernel]

# PosteriorConsistency of _regression_pairs() at 7ee010d, before the beta grid
# replaced the Adam loop (olgpu, CPU, torch 2.7.1).
CONSISTENCY_AT_7EE010D = -0.00288588460534811
PAIR = {"dataset_1": "normal", "dataset_2": "reference_normal"}


def _config(**overrides) -> dict:
    config = {
        "normalization_type": "sigmoid",
        "normalization_params": None,
        "energy_type": "adaptive",
        "energy_params": {"scale": 0.5},
        "regularization_type": "none",
        "regularization_params": None,
        "binary": True,
    }
    return config | overrides


def _paired(scores_1, scores_2):
    idx_1, idx_2 = get_pairing_fn("cdf")(scores_1, scores_2)
    return scores_1[idx_1], scores_2[idx_2]


def _regression_pairs():
    torch.manual_seed(0)
    scores_1 = torch.randn(2000).exp()
    scores_2 = (torch.randn(2000) + 0.1).exp()
    return _paired(scores_1, scores_2)


def _random_pairs(n=4000, seed=7):
    g = torch.Generator().manual_seed(seed)
    scores_1 = torch.randn(n, generator=g).exp()
    scores_2 = (0.1 + 1.1 * torch.randn(n, generator=g)).exp()
    return _paired(scores_1, scores_2)


def _updated(metric_cls, scores_1, scores_2, **overrides):
    metric = metric_cls(**_config(**overrides))
    metric.update(scores_1, scores_2)
    return metric


@pytest.mark.parametrize("metric_cls", CAP_CLASSES)
def test_cap_is_the_maximum_of_the_stored_curve(metric_cls):
    metric = _updated(metric_cls, *_random_pairs(), beta_min=1e-3)

    assert metric.curve.dtype == torch.float64
    assert metric.curve.shape == metric.betas.shape == (200,)
    assert metric.compute() == metric.curve.max().item()
    assert metric.beta_star == metric.betas[metric.curve.argmax()].item()
    assert not metric.at_bound
    assert abs(metric.curve[0].item() + math.log(2.0)) < 1e-6


@pytest.mark.parametrize("metric_cls", CONSISTENCY_CLASSES)
def test_consistency_value_is_unchanged(metric_cls):
    metric = _updated(metric_cls, *_regression_pairs())

    assert metric.compute() == pytest.approx(CONSISTENCY_AT_7EE010D, abs=1e-9)
    assert metric.curve is None


@pytest.mark.parametrize("module", KERNELS)
def test_kernel_is_exact_and_finite_where_the_tanh_form_is_not(module):
    g = torch.Generator().manual_seed(5)
    gap1 = torch.randn(5000, generator=g, dtype=torch.float64)
    gap2 = gap1 + 0.3 * torch.randn(5000, generator=g, dtype=torch.float64)
    betas = torch.logspace(-2, 3, 200, dtype=torch.float64)
    curve = module.posterior_agreement_curve(gap1, gap2, betas)

    # Eq. cap_kernel as written, with E(p, 0) = 0 and E(p, 1) = D; its exponentials
    # stay finite up to beta = 50 for these gaps.
    low = betas <= 50.0
    b = betas[low, None]
    num = torch.log(1.0 + torch.exp(-b * (gap1 + gap2)))
    den = torch.log(1.0 + torch.exp(-b * gap1)) + torch.log(1.0 + torch.exp(-b * gap2))
    torch.testing.assert_close(curve[low], (num - den).mean(dim=1), rtol=0, atol=1e-12)

    m1 = torch.tanh(0.5 * betas[:, None] * gap1)
    m2 = torch.tanh(0.5 * betas[:, None] * gap2)
    tanh_form = torch.log1p(m1 * m2).mean(dim=1) - math.log(2.0)
    finite = torch.isfinite(tanh_form)
    assert torch.isfinite(curve).all()
    assert not finite[-1]
    # Short of saturation the tanh form is already inexact: 1 + m1 m2 is resolved
    # only to 1e-16 absolute.
    torch.testing.assert_close(curve[finite], tanh_form[finite], rtol=1e-4, atol=0)


class _Module:
    def __init__(self):
        self.device = torch.device("cpu")
        self.logged = {}

    def log_dict(self, metrics: dict, **kwargs) -> None:
        self.logged = dict(metrics)


def _fit(callback, root: Path, epochs: list[tuple]) -> list[dict]:
    trainer = SimpleNamespace(
        val_dataloaders=dict.fromkeys(PAIR.values()),
        default_root_dir=str(root),
        current_epoch=0,
        logger=None,
        sanity_checking=False,
    )
    module = _Module()
    logged = []
    callback.on_fit_start(trainer, module)
    for epoch, scores in enumerate(epochs):
        trainer.current_epoch = epoch
        callback.on_validation_epoch_start(trainer, module)
        for idx, score in enumerate(scores):
            outputs = {"ascore/full": score}
            callback.on_validation_batch_end(trainer, module, outputs, None, 0, idx)
        callback.on_validation_epoch_end(trainer, module)
        logged.append(module.logged)
    callback.on_fit_end(trainer, module)
    return logged


def _epochs(n: int) -> list[tuple]:
    return [_random_pairs(n=500, seed=seed) for seed in range(n)]


def _training_cap(**overrides) -> CAPCallback:
    return CAPCallback(
        output_name="ascore/full",
        pairing_type="cdf",
        cap_metric_config=_config(**overrides),
        **PAIR,
    )


def test_training_callback_stores_one_curve_per_epoch(tmp_path):
    callback = _training_cap(n_beta=50)
    logged = _fit(callback, tmp_path, _epochs(3))

    stored = np.load(tmp_path / "cap_curves_normal_vs_reference_normal.npz")
    assert stored["betas"].shape == (50,)
    assert stored["epochs"].tolist() == [0, 1, 2]
    assert stored["curves"].shape == (3, 50)
    np.testing.assert_array_equal(stored["curves"][-1], callback.capmetric.curve)
    assert logged[-1] == {
        "val/summary/cap_ema_normal_vs_reference_normal": callback.cap_ema,
        "val/summary/cap_beta_star_normal_vs_reference_normal": (
            callback.capmetric.beta_star
        ),
        "val/summary/cap_at_bound_normal_vs_reference_normal": 0.0,
    }
