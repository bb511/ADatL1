import math

import pytest
import torch

from src.callbacks.metrics.cap.binary.energy import baseline
from src.callbacks.metrics.cap.kernel import ApproximationCapacityKernel
from src.callbacks.metrics.cap.metric import ApproximationCapacity as TrainingCAP
from src.evaluation.callbacks.metrics.cap.metric import (
    ApproximationCapacity as EvaluationCAP,
)


def _cap(
    x: torch.Tensor,
    y: torch.Tensor,
    normalization_type: str = "minmax",
) -> tuple[float, TrainingCAP]:
    metric = TrainingCAP(
        normalization_type=normalization_type,
        energy_type="baseline",
        n_epochs=12,
        batch_size=64,
        lr=0.08,
        normalize_gradients=True,
    )
    metric.update(x, y)
    return metric.compute(), metric


def test_training_and_evaluation_use_the_same_cap_implementation() -> None:
    assert TrainingCAP is EvaluationCAP


@pytest.mark.parametrize("metric_cls", [TrainingCAP, EvaluationCAP])
def test_cap_metric_updates_when_instantiated_in_inference_mode(metric_cls) -> None:
    with torch.inference_mode():
        metric = metric_cls(
            normalization_type="sigmoid",
            energy_type="adaptive",
            energy_params={"scale": 0.5},
            n_epochs=1,
            batch_size=4,
            normalize_gradients=True,
        )

    x = torch.linspace(-1.0, 1.0, steps=8)
    y = torch.linspace(-0.5, 1.5, steps=8)

    with torch.inference_mode(False), torch.enable_grad():
        metric.update(x.clone().requires_grad_(True), y.clone().requires_grad_(True))

    assert isinstance(metric.compute(), float)


def test_kernel_is_finite_for_extreme_energies() -> None:
    kernel = ApproximationCapacityKernel(beta0=100.0, energy_fn=baseline)
    scores = torch.tensor([-1e20, 1e20])

    cap = kernel.compute_mutual_information(scores, scores)

    assert torch.isfinite(cap)


def test_kernel_preserves_vectorized_candidate_dimension() -> None:
    def vectorized_baseline(probability: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = label.view(-1, *([1] * (probability.ndim - 1)))
        return label * (1.0 - probability) + (1.0 - label) * probability

    scores_1 = torch.tensor([[0.1, 0.8, 0.4], [0.3, 0.6, 0.5], [0.9, 0.2, 0.7]])
    scores_2 = torch.tensor([[0.2, 0.7, 0.5], [0.4, 0.5, 0.4], [0.8, 0.1, 0.9]])
    kernel = ApproximationCapacityKernel(
        beta0=2.0,
        energy_fn=vectorized_baseline,
    )

    vectorized = kernel.compute_mutual_information(scores_1, scores_2)
    separate = torch.stack(
        [
            kernel.compute_mutual_information(scores_1[:, i], scores_2[:, i])
            for i in range(scores_1.shape[1])
        ]
    )

    assert vectorized.shape == (scores_1.shape[1],)
    torch.testing.assert_close(vectorized, separate)


def test_cap_distinguishes_correlated_random_and_constant_scores() -> None:
    generator = torch.Generator().manual_seed(3)
    scores = torch.randn(256, generator=generator)
    correlated = scores + 0.15 * torch.randn(256, generator=generator)
    random = torch.randn(256, generator=generator)
    constant = torch.ones(256)

    correlated_cap, _ = _cap(scores, correlated)
    random_cap, _ = _cap(scores, random)
    constant_cap, _ = _cap(constant, constant)

    assert all(math.isfinite(value) for value in (correlated_cap, random_cap, constant_cap))
    assert correlated_cap > random_cap + 0.1
    assert constant_cap == pytest.approx(-math.log(2.0), abs=1e-6)


def test_cap_is_deterministic_and_minmax_affine_invariant() -> None:
    generator = torch.Generator().manual_seed(17)
    scores_1 = torch.randn(192, generator=generator)
    scores_2 = scores_1 + 0.2 * torch.randn(192, generator=generator)

    first, first_metric = _cap(scores_1, scores_2)
    repeated, repeated_metric = _cap(scores_1, scores_2)
    transformed, _ = _cap(7.0 * scores_1 + 3.0, 7.0 * scores_2 + 3.0)

    assert repeated == pytest.approx(first, abs=1e-7)
    assert transformed == pytest.approx(first, abs=1e-6)
    assert repeated_metric.epoch_logs == first_metric.epoch_logs


def test_rank_normalization_is_strictly_monotone_invariant() -> None:
    generator = torch.Generator().manual_seed(23)
    scores_1 = torch.randn(192, generator=generator)
    scores_2 = scores_1 + 0.2 * torch.randn(192, generator=generator)

    original, _ = _cap(scores_1, scores_2, normalization_type="rank")
    transformed, _ = _cap(
        torch.exp(scores_1),
        torch.exp(scores_2),
        normalization_type="rank",
    )

    assert transformed == pytest.approx(original, abs=1e-7)


def test_beta_is_projected_after_every_optimizer_step() -> None:
    metric = TrainingCAP(
        beta0=0.01,
        normalization_type="none",
        energy_type="baseline",
        n_epochs=3,
        batch_size=64,
        lr=10.0,
        normalize_gradients=True,
    )

    metric.update(torch.linspace(0.0, 1.0, 64), torch.linspace(1.0, 0.0, 64))

    assert metric.epoch_logs
    assert all(log["beta"] >= 0.0 for log in metric.epoch_logs)
    assert metric.epoch_logs[0]["beta"] == 0.0
    assert math.isfinite(metric.compute())


@pytest.mark.parametrize(
    ("scores_1", "scores_2", "message"),
    [
        (torch.tensor([]), torch.tensor([]), "at least one"),
        (torch.ones(2), torch.ones(3), "same number"),
        (torch.tensor([0.0, float("nan")]), torch.ones(2), "finite"),
    ],
)
def test_cap_rejects_invalid_score_pairs(
    scores_1: torch.Tensor,
    scores_2: torch.Tensor,
    message: str,
) -> None:
    metric = TrainingCAP(n_epochs=0)

    with pytest.raises(ValueError, match=message):
        metric.update(scores_1, scores_2)
