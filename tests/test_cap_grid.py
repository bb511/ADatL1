import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pytorch_lightning.loggers import MLFlowLogger

from src.callbacks.cap import CAPCallback
from src.callbacks.consistency import PosteriorConsistencyCallback
from src.callbacks.metrics.cap import kernel
from src.callbacks.metrics.cap.binary import get_pairing_fn
from src.callbacks.metrics.cap.metric import ApproximationCapacity, PosteriorConsistency
from src.evaluation.callbacks import cap as evaluation_cap_module
from src.evaluation.callbacks.cap import CAP as EvaluationCAP
from src.evaluation.callbacks.consistency import (
    PosteriorConsistency as EvaluationConsistencyCallback,
)
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


@pytest.mark.parametrize("metric_cls", CAP_CLASSES)
def test_a_finer_grid_moves_the_maximum_little(metric_cls):
    pairs = _random_pairs()
    coarse = _updated(metric_cls, *pairs).compute()
    fine = _updated(metric_cls, *pairs, n_beta=4000).compute()
    # 200 points over five decades are 0.058 apart in log(beta); a peak about one unit
    # of log(beta) wide is then missed by up to about 5e-4 of its height (2.6e-4
    # measured on these scores and their neighbouring seeds).
    assert fine == pytest.approx(coarse, rel=1e-3)


@pytest.mark.parametrize("metric_cls", CAP_CLASSES)
def test_identical_sets_maximise_at_the_bound(metric_cls):
    scores = torch.randn(2000, generator=torch.Generator().manual_seed(3)).exp()
    metric = _updated(metric_cls, *_paired(scores, scores.clone()))

    assert metric.at_bound
    assert metric.beta_star == pytest.approx(1000.0)
    assert -1e-3 < metric.compute() < 0.0


@pytest.mark.parametrize("metric_cls", CAP_CLASSES)
def test_cap_is_independent_of_the_global_rng(metric_cls):
    pairs = _random_pairs()
    torch.manual_seed(0)
    first = _updated(metric_cls, *pairs).compute()

    torch.manual_seed(12345)
    torch.rand(17)
    state = torch.get_rng_state()
    second = _updated(metric_cls, *pairs).compute()

    assert first == second
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("metric_cls", CAP_CLASSES + CONSISTENCY_CLASSES)
def test_update_leaves_the_callers_energy_params_alone(metric_cls):
    energy_params = {"scale": 0.5}
    metric = metric_cls(**_config(energy_params=energy_params))
    metric.update(*_random_pairs())

    assert energy_params == {"scale": 0.5}


@pytest.mark.parametrize("metric_cls", CONSISTENCY_CLASSES)
def test_consistency_value_is_unchanged(metric_cls):
    metric = _updated(metric_cls, *_regression_pairs())

    assert metric.compute() == pytest.approx(CONSISTENCY_AT_7EE010D, abs=1e-9)
    assert metric.curve is None


@pytest.mark.parametrize("metric_cls", CAP_CLASSES)
def test_a_non_finite_score_propagates(metric_cls):
    scores_1, scores_2 = _random_pairs()
    scores_1 = scores_1.clone()
    scores_1[5] = math.nan
    metric = _updated(metric_cls, scores_1, scores_2)

    assert math.isnan(metric.compute())
    assert math.isnan(metric.beta_star)


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
    def __init__(self, ckpt_path: Path | None = None):
        self.device = torch.device("cpu")
        self._ckpt_path = str(ckpt_path)
        self.logged = {}

    def log_dict(self, metrics: dict, **kwargs) -> None:
        self.logged = dict(metrics)


def _fit(callback, root: Path, epochs: list[tuple], logger=None) -> list[dict]:
    trainer = SimpleNamespace(
        val_dataloaders=dict.fromkeys(PAIR.values()),
        default_root_dir=str(root),
        current_epoch=0,
        logger=logger,
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


@pytest.mark.filterwarnings("ignore:The filesystem tracking backend:FutureWarning")
def test_training_callback_logs_the_curves_to_mlflow(tmp_path):
    logger = MLFlowLogger(
        experiment_name="cap_grid", tracking_uri=(tmp_path / "mlruns").as_uri()
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _fit(_training_cap(n_beta=50), run_dir, _epochs(2), logger=logger)

    artifacts = logger.experiment.list_artifacts(logger.run_id)
    assert [a.path for a in artifacts] == ["cap_curves_normal_vs_reference_normal.npz"]


def test_training_ema_skips_a_non_finite_cap(tmp_path):
    epochs = _epochs(3)
    poisoned = epochs[1][0].clone()
    poisoned[0] = math.nan
    epochs[1] = (poisoned, epochs[1][1])
    key = "val/summary/cap_ema_normal_vs_reference_normal"
    emas = [logged[key] for logged in _fit(_training_cap(), tmp_path, epochs)]

    assert math.isfinite(emas[0])
    assert emas[1] == emas[0]
    assert math.isfinite(emas[2]) and emas[2] != emas[0]


def test_consistency_callback_writes_no_curves(tmp_path):
    callback = PosteriorConsistencyCallback(
        output_name="ascore/full",
        pairing_type="cdf",
        cap_metric_config=_config(),
        **PAIR,
    )
    logged = _fit(callback, tmp_path, _epochs(2))

    assert list(tmp_path.iterdir()) == []
    assert list(logged[-1]) == [
        "val/summary/consistency_ema_normal_vs_reference_normal"
    ]


def _evaluate(callback, ckpt_path: Path, monkeypatch) -> dict:
    logged = {}
    monkeypatch.setattr(
        evaluation_cap_module.utils.mlflow,
        "log_metrics_to_mlflow",
        lambda trainer, metrics, **kwargs: logged.update(metrics),
    )
    trainer = SimpleNamespace(
        test_dataloaders=dict.fromkeys(PAIR.values()), split="val"
    )
    module = _Module(ckpt_path)
    callback.on_test_start(trainer, module)
    callback.on_test_epoch_start(trainer, module)
    for idx, score in enumerate(_random_pairs(n=500)):
        callback.on_test_batch_end(
            trainer, module, {"ascore/full": score}, None, 0, idx
        )
    callback.on_test_epoch_end(trainer, module)
    return logged


def test_evaluation_callback_stores_the_checkpoint_curve(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "ds=normal__epoch=3.ckpt"
    callback = EvaluationCAP(
        output_name="ascore/full",
        pairing_type="cdf",
        cap_metric_config=_config(n_beta=50),
        **PAIR,
    )
    logged = _evaluate(callback, ckpt_path, monkeypatch)

    stored = np.load(
        tmp_path / "plots" / "val" / ckpt_path.stem / "cap" / "cap_curve.npz"
    )
    assert stored["betas"].shape == stored["curve"].shape == (50,)
    assert logged == {
        "cap": stored["curve"].max(),
        "cap_beta_star": callback.capmetric.beta_star,
        "cap_at_bound": 0.0,
    }


def test_evaluation_consistency_callback_writes_no_curve(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "ds=normal__epoch=3.ckpt"
    callback = EvaluationConsistencyCallback(
        output_name="ascore/full",
        pairing_type="cdf",
        cap_metric_config=_config(),
        **PAIR,
    )
    logged = _evaluate(callback, ckpt_path, monkeypatch)

    assert list(tmp_path.iterdir()) == []
    assert list(logged) == ["consistency"]
