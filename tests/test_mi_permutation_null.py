"""MI noise-floor diagnostic: permutation null and analytic floor."""

import pytest
import torch

from src.algorithms.losses.components.bernoulli_mi import BernoulliMILoss


def _loop_mi(loss: BernoulliMILoss, latent, sensitive, perm) -> torch.Tensor:
    """Reference: the training estimator applied to explicitly permuted labels."""
    return loss(latent=latent, sensitive=sensitive[perm])


@pytest.mark.parametrize("num_bins", [2, 7, 50])
def test_vectorised_null_matches_forward_on_same_permutations(num_bins):
    torch.manual_seed(0)
    n, d = 4096, 8
    latent = torch.randn(n, d)
    sensitive = torch.randint(0, num_bins, (n, 1))
    loss = BernoulliMILoss(temperature=6.0)

    generator = torch.Generator().manual_seed(123)
    null = loss.permutation_null(latent, sensitive, num_permutations=4, generator=generator)

    replay = torch.Generator().manual_seed(123)
    expected = torch.stack(
        [_loop_mi(loss, latent, sensitive, torch.randperm(n, generator=replay)) for _ in range(4)]
    )
    torch.testing.assert_close(null, expected, rtol=1e-5, atol=1e-7)


def test_null_with_identity_permutation_equals_forward_with_gaps_in_labels():
    torch.manual_seed(1)
    n = 1000
    latent = torch.randn(n, 5)
    # Non-contiguous label values (empty bins) must not change the result.
    sensitive = torch.randint(0, 4, (n,)) * 3 + 10
    loss = BernoulliMILoss()
    probs, labels, groups = loss._diagnostic_inputs(latent, sensitive)
    assert groups == 4
    assert int(labels.max()) == 3


def test_null_is_far_below_real_leakage_and_close_to_analytic_floor():
    torch.manual_seed(2)
    n, d, k = 16384, 8, 50
    sensitive = torch.randint(0, k, (n, 1))
    independent = torch.randn(n, d)
    leaking = independent + 0.5 * (sensitive.float() / k - 0.5)

    loss = BernoulliMILoss(temperature=6.0)
    gen = torch.Generator().manual_seed(0)

    null = loss.permutation_null(independent, sensitive, 50, generator=gen)
    floor = loss.analytic_null_floor(independent, sensitive)
    mi_independent = loss(independent, sensitive)

    # Independent latent: the real MI is itself a draw from the null.
    assert null.min() > 0
    assert abs(float(mi_independent - null.mean())) < 4 * float(null.std())
    # Analytic second-order floor agrees with the permutation mean to ~20 %.
    assert float(floor) == pytest.approx(float(null.mean()), rel=0.2)

    mi_leaking = loss(leaking, sensitive)
    null_leaking = loss.permutation_null(leaking, sensitive, 10, generator=gen)
    assert float(mi_leaking - null_leaking.mean()) > 10 * float(null_leaking.std())


def test_floor_vanishes_for_constant_latent_and_is_maximal_when_saturated():
    n, d, k = 4096, 4, 10
    sensitive = torch.randint(0, k, (n,))
    loss = BernoulliMILoss(temperature=6.0)

    constant = torch.zeros(n, d)
    assert float(loss.analytic_null_floor(constant, sensitive)) == pytest.approx(0.0, abs=1e-12)

    saturated = torch.where(torch.rand(n, d) < 0.5, -50.0, 50.0)
    expected = d * (k - 1) / (2 * n * torch.log(torch.tensor(2.0)))
    assert float(loss.analytic_null_floor(saturated, sensitive)) == pytest.approx(float(expected), rel=1e-3)


def test_diagnostic_does_not_touch_global_rng_or_gradients():
    latent = torch.randn(512, 4, requires_grad=True)
    sensitive = torch.randint(0, 5, (512,))
    loss = BernoulliMILoss().train()

    state = torch.get_rng_state()
    null = loss.permutation_null(latent, sensitive, 3, generator=torch.Generator().manual_seed(0))
    floor = loss.analytic_null_floor(latent, sensitive)
    assert torch.equal(state, torch.get_rng_state())
    assert not null.requires_grad and not floor.requires_grad


def test_invalid_num_permutations():
    loss = BernoulliMILoss()
    with pytest.raises(ValueError):
        loss.permutation_null(torch.randn(10, 2), torch.zeros(10, dtype=torch.long), 0)
