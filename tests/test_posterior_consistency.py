"""Tests for the consistency-only (beta-free) variant of CAP.

The load-bearing test is `test_consistency_is_the_beta_zero_limit_of_log_cosine`:
the whole justification for reporting `-mean (D1 - D2)**2` is that it is the
leading-order term of CAP's posterior log-cosine, so that beta enters only as a
common prefactor and drops out of any ranking of models. Everything else here
guards the plumbing.
"""

import inspect
import math

import pytest
import torch

from src.callbacks.cap import CAPCallback
from src.callbacks.consistency import PosteriorConsistencyCallback
from src.callbacks.metrics.cap.metric import ApproximationCapacity, PosteriorConsistency
from src.evaluation.callbacks.cap import CAP as EvaluationCAP
from src.evaluation.callbacks.consistency import (
    PosteriorConsistency as EvaluationPosteriorConsistency,
)
from src.evaluation.callbacks.metrics.cap.metric import (
    PosteriorConsistency as EvaluationConsistencyMetric,
)


def _metric_config(**overrides) -> dict:
    """The production front-end (joint sigmoid + adaptive energy), no beta loop."""
    cfg = {
        "beta0": 1.0,
        "normalization_type": "sigmoid",
        "normalization_params": None,
        "energy_type": "adaptive",
        "energy_params": {"scale": 0.5},
        "regularization_type": "none",
        "regularization_params": None,
        "binary": True,
        "lr": 0.01,
        "n_epochs": 0,
        "batch_size": 64,
        "normalize_gradients": True,
    }
    cfg.update(overrides)
    return cfg


def _scores(n=256, seed=1234):
    g = torch.Generator().manual_seed(seed)
    scores_1 = torch.rand(n, generator=g, dtype=torch.double)
    scores_2 = scores_1 + 0.3 * torch.randn(n, generator=g, dtype=torch.double)
    return scores_1, scores_2


def _mean_log_cosine(metric, scores_1, scores_2, beta):
    """Exact mean log-cosine of the two Gibbs posteriors at inverse temperature beta.

    Reproduces CAP's front-end (joint normalisation, injected mean/std, energy
    function) and then forms the posteriors explicitly, so this is an
    independent reference for the quantity the metric approximates.
    """
    prob1, prob2 = metric.normalizer_fn(scores_1, scores_2)
    combined = torch.cat([prob1, prob2], dim=0)
    metric.energy_params.update(
        {"mean": combined.mean().item(), "std": combined.std().item()}
    )
    energy_fn = metric._get_energy_fn()

    ones = torch.ones_like(prob1)
    zeros = torch.zeros_like(prob1)
    log_w1 = torch.stack((-beta * energy_fn(prob1, zeros), -beta * energy_fn(prob1, ones)))
    log_w2 = torch.stack((-beta * energy_fn(prob2, zeros), -beta * energy_fn(prob2, ones)))
    log_p1 = log_w1 - torch.logsumexp(log_w1, dim=0, keepdim=True)
    log_p2 = log_w2 - torch.logsumexp(log_w2, dim=0, keepdim=True)

    log_overlap = torch.logsumexp(log_p1 + log_p2, dim=0)
    log_norm_1 = 0.5 * torch.logsumexp(2.0 * log_p1, dim=0)
    log_norm_2 = 0.5 * torch.logsumexp(2.0 * log_p2, dim=0)
    return torch.mean(log_overlap - log_norm_1 - log_norm_2).item()


# --------------------------------------------------------------------------- #
# The mathematical claim
# --------------------------------------------------------------------------- #


def test_consistency_is_the_beta_zero_limit_of_log_cosine():
    """log cos = -(beta**2 / 8) * sum (D1 - D2)**2 + O(beta**4)."""
    scores_1, scores_2 = _scores()
    metric = PosteriorConsistency(**_metric_config())
    metric.update(scores_1, scores_2)
    value = metric.compute()

    assert value < 0.0

    errors = {}
    for beta in (1e-2, 1e-3):
        approximation = 8.0 * _mean_log_cosine(metric, scores_1, scores_2, beta) / beta**2
        errors[beta] = abs(approximation - value) / abs(value)

    assert errors[1e-3] < 1e-4
    # The remainder is O(beta**2) relative, so shrinking beta tenfold must buy
    # roughly two orders of magnitude of accuracy.
    assert errors[1e-3] < errors[1e-2] / 50.0


def test_log_cosine_alone_is_degenerate_at_zero_temperature():
    """Why the limit is taken at all: log cos vanishes for every model at beta=0.

    It decays quadratically, so it cannot separate models on its own -- which is
    exactly why beta must not be chosen by maximising it.
    """
    scores_1, scores_2 = _scores()
    metric = PosteriorConsistency(**_metric_config())
    metric.update(scores_1, scores_2)
    scale = abs(metric.compute())

    for beta in (1e-5, 1e-6):
        log_cosine = _mean_log_cosine(metric, scores_1, scores_2, beta)
        assert abs(log_cosine) < scale * beta**2


# --------------------------------------------------------------------------- #
# Consistency metric semantics
# --------------------------------------------------------------------------- #


def test_consistency_is_zero_for_identical_scores():
    scores = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.double)
    metric = PosteriorConsistency(**_metric_config())
    metric.update(scores, scores)
    assert metric.compute() == pytest.approx(0.0, abs=1e-9)


def test_consistency_is_negative_for_differing_scores():
    scores_1 = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.double)
    scores_2 = 1.0 - scores_1
    metric = PosteriorConsistency(**_metric_config())
    metric.update(scores_1, scores_2)
    value = metric.compute()
    assert math.isfinite(value)
    assert value < 0.0


def test_consistency_ignores_the_beta_optimisation_hyperparameters():
    """beta0/lr/n_epochs/batch_size/normalize_gradients must have no effect."""
    scores_1, scores_2 = _scores()

    baseline = PosteriorConsistency(**_metric_config())
    baseline.update(scores_1, scores_2)

    other = PosteriorConsistency(
        **_metric_config(
            beta0=7.5, lr=0.5, n_epochs=13, batch_size=8, normalize_gradients=False
        )
    )
    other.update(scores_1, scores_2)

    assert other.compute() == pytest.approx(baseline.compute(), rel=1e-9)


def test_consistency_is_deterministic():
    scores_1, scores_2 = _scores()
    values = []
    for _ in range(3):
        metric = PosteriorConsistency(**_metric_config())
        metric.update(scores_1, scores_2)
        values.append(metric.compute())
    assert values[0] == values[1] == values[2]


# --------------------------------------------------------------------------- #
# CAP itself must be untouched
# --------------------------------------------------------------------------- #


def test_cap_still_computes_the_kernel_partition_ratio():
    """Pins CAP's own value against an independent reimplementation.

    With `n_epochs=0` no beta optimisation runs, so CAP is evaluated at `beta0`
    and reduces to the mean of log(Z_12 / (Z_1 * Z_2)) over paired samples.
    """
    scores_1, scores_2 = _scores(n=128)
    beta = 1.5
    metric = ApproximationCapacity(
        **_metric_config(
            beta0=beta, normalization_type="none", energy_type="baseline", energy_params=None
        )
    )
    metric.update(scores_1, scores_2)

    # E(p, 0) = p and E(p, 1) = 1 - p for the baseline energy.
    e1_0, e1_1 = scores_1, 1.0 - scores_1
    e2_0, e2_1 = scores_2, 1.0 - scores_2
    num = torch.log(
        torch.exp(-beta * (e1_0 + e2_0)) + torch.exp(-beta * (e1_1 + e2_1))
    )
    den = torch.log(
        (torch.exp(-beta * e1_0) + torch.exp(-beta * e1_1))
        * (torch.exp(-beta * e2_0) + torch.exp(-beta * e2_1))
    )
    expected = torch.mean(num - den).item()

    assert metric.compute() == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------- #
# Callback plumbing
# --------------------------------------------------------------------------- #


def test_training_callback_is_a_drop_in_cap_replacement():
    assert inspect.signature(PosteriorConsistencyCallback) == inspect.signature(CAPCallback)
    assert PosteriorConsistencyCallback.metric_cls is PosteriorConsistency
    assert PosteriorConsistencyCallback.metric_key == "consistency"
    assert CAPCallback.metric_cls is ApproximationCapacity
    assert CAPCallback.metric_key == "cap"


def test_evaluation_callback_registers_under_its_own_name():
    callback = EvaluationPosteriorConsistency(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="none",
        cap_metric_config=_metric_config(),
    )
    assert callback.name == "consistency"
    assert callback.metric_cls is EvaluationConsistencyMetric
    assert callback.metric_label == "Consistency"
    assert EvaluationCAP.metric_label == "CAP"


def test_compute_cap_returns_a_single_value():
    """The Spearman rank correlation is gone, so this is now one float."""
    callback = PosteriorConsistencyCallback(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="none",
        cap_metric_config=_metric_config(),
    )
    callback.device = "cpu"
    callback.capmetric = callback.metric_cls(**callback.cap_metric_config, device="cpu")
    callback.dataset_1_scores = [torch.tensor([0.1, 0.3, 0.8])]
    callback.dataset_2_scores = [torch.tensor([0.1, 0.3, 0.8])]

    value = callback._compute_cap()

    assert isinstance(value, float)
    assert value == pytest.approx(0.0, abs=1e-7)


def test_rank_correlation_is_no_longer_computed():
    """It was identically 1 under cdf pairing, which sorts both paired sides."""
    for cls in (CAPCallback, EvaluationCAP, PosteriorConsistencyCallback):
        assert not hasattr(cls, "_spearman_corr")
    assert not hasattr(EvaluationCAP("a", "b", "c", "none", _metric_config()), "rankcorr_summary")
