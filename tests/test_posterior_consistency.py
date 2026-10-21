"""Tests for the consistency-only (beta-free) variant of CAP.

The load-bearing test is `test_consistency_is_the_beta_zero_limit_of_log_cosine`:
the whole justification for reporting `-mean (D1 - D2)**2` is that it is the
leading-order term of CAP's posterior log-cosine, so that beta enters only as a
common prefactor and drops out of any ranking of models. The other test pins CAP's
own value, which the variant must leave untouched.
"""

import pytest
import torch

from src.callbacks.components.cap.metric import (
    ApproximationCapacity,
    PosteriorConsistency,
)


def _metric_config(**overrides) -> dict:
    """The production front-end (joint sigmoid + adaptive energy), no beta loop."""
    cfg = {
        "normalization_type": "sigmoid",
        "normalization_params": None,
        "energy_type": "adaptive",
        "energy_params": {"scale": 0.5},
        "regularization_type": "none",
        "regularization_params": None,
        "binary": True,
        "beta_min": 0.01,
        "beta_max": 1000.0,
        "n_beta": 200,
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

    Reproduces CAP's front-end (joint normalisation, injected mean/std, energy function)
    and then forms the posteriors explicitly, so this is an independent reference for
    the quantity the metric approximates.
    """
    prob1, prob2 = metric.normalizer_fn(scores_1, scores_2)
    combined = torch.cat([prob1, prob2], dim=0)
    metric.energy_params.update(
        {"mean": combined.mean().item(), "std": combined.std().item()}
    )
    energy_fn = metric._get_energy_fn()

    ones = torch.ones_like(prob1)
    zeros = torch.zeros_like(prob1)
    log_w1 = torch.stack(
        (-beta * energy_fn(prob1, zeros), -beta * energy_fn(prob1, ones))
    )
    log_w2 = torch.stack(
        (-beta * energy_fn(prob2, zeros), -beta * energy_fn(prob2, ones))
    )
    log_p1 = log_w1 - torch.logsumexp(log_w1, dim=0, keepdim=True)
    log_p2 = log_w2 - torch.logsumexp(log_w2, dim=0, keepdim=True)

    log_overlap = torch.logsumexp(log_p1 + log_p2, dim=0)
    log_norm_1 = 0.5 * torch.logsumexp(2.0 * log_p1, dim=0)
    log_norm_2 = 0.5 * torch.logsumexp(2.0 * log_p2, dim=0)
    return torch.mean(log_overlap - log_norm_1 - log_norm_2).item()


def test_consistency_is_the_beta_zero_limit_of_log_cosine():
    # log cos = -(beta**2 / 8) * sum (D1 - D2)**2 + O(beta**4).
    scores_1, scores_2 = _scores()
    metric = PosteriorConsistency(**_metric_config())
    metric.update(scores_1, scores_2)
    value = metric.compute()

    assert value < 0.0

    errors = {}
    for beta in (1e-2, 1e-3):
        approximation = (
            8.0 * _mean_log_cosine(metric, scores_1, scores_2, beta) / beta**2
        )
        errors[beta] = abs(approximation - value) / abs(value)

    assert errors[1e-3] < 1e-4
    # The remainder is O(beta**2) relative, so shrinking beta tenfold must buy
    # roughly two orders of magnitude of accuracy.
    assert errors[1e-3] < errors[1e-2] / 50.0


def test_cap_still_computes_the_kernel_partition_ratio():
    # On a one-point beta grid CAP reduces to the mean of log(Z_12 / (Z_1 * Z_2)).
    scores_1, scores_2 = _scores(n=128)
    beta = 1.5
    metric = ApproximationCapacity(
        **_metric_config(
            beta_min=beta,
            beta_max=beta,
            n_beta=1,
            normalization_type="none",
            energy_type="baseline",
            energy_params=None,
        )
    )
    metric.update(scores_1, scores_2)

    # E(p, 0) = p and E(p, 1) = 1 - p for the baseline energy.
    e1_0, e1_1 = scores_1, 1.0 - scores_1
    e2_0, e2_1 = scores_2, 1.0 - scores_2
    num = torch.log(torch.exp(-beta * (e1_0 + e2_0)) + torch.exp(-beta * (e1_1 + e2_1)))
    den = torch.log(
        (torch.exp(-beta * e1_0) + torch.exp(-beta * e1_1))
        * (torch.exp(-beta * e2_0) + torch.exp(-beta * e2_1))
    )
    expected = torch.mean(num - den).item()

    assert metric.compute() == pytest.approx(expected, rel=1e-5)
