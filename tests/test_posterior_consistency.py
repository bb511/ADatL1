import inspect
import math

import pytest
import torch

from src.callbacks.cap import CAPCallback
from src.callbacks.consistency import PosteriorConsistencyCallback
from src.callbacks.metrics.cap.binary.energy import baseline
from src.callbacks.metrics.cap.kernel import ApproximationCapacityKernel
from src.callbacks.metrics.cap.metric import ApproximationCapacity, PosteriorConsistency
from src.evaluation.callbacks.cap import CAP as EvaluationCAP
from src.evaluation.callbacks.consistency import (
    PosteriorConsistency as EvaluationPosteriorConsistency,
)
from src.evaluation.callbacks.metrics.cap.metric import (
    PosteriorConsistency as EvaluationConsistencyMetric,
)


def _metric_config(n_epochs: int = 0) -> dict:
    return {
        "beta0": 2.0,
        "normalization_type": "none",
        "energy_type": "baseline",
        "n_epochs": n_epochs,
        "batch_size": 64,
        "lr": 0.05,
        "normalize_gradients": True,
    }


def test_kernel_components_sum_to_cap() -> None:
    kernel = ApproximationCapacityKernel(beta0=2.0, energy_fn=baseline)
    scores_1 = torch.tensor([0.1, 0.3, 0.8, 0.9])
    scores_2 = torch.tensor([0.2, 0.4, 0.7, 0.95])

    cap = kernel.compute_mutual_information(scores_1, scores_2)
    consistency, concentration = kernel.compute_log_components(scores_1, scores_2)

    torch.testing.assert_close(cap, consistency + concentration)
    assert consistency <= 0.0
    assert concentration <= 0.0


def test_consistency_is_zero_for_identical_posteriors() -> None:
    metric = PosteriorConsistency(**_metric_config())
    scores = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

    metric.update(scores, scores)

    assert metric.compute() == pytest.approx(0.0, abs=1e-7)
    assert metric.selected_beta == pytest.approx(2.0)


def test_consistency_penalizes_opposing_posteriors() -> None:
    metric = PosteriorConsistency(**_metric_config())
    scores_1 = torch.tensor([0.0, 0.1, 0.2, 0.3])
    scores_2 = 1.0 - scores_1

    metric.update(scores_1, scores_2)

    assert math.isfinite(metric.compute())
    assert metric.compute() < -0.1


def test_consistency_uses_the_temperature_selected_by_cap() -> None:
    generator = torch.Generator().manual_seed(1729)
    scores_1 = torch.rand(128, generator=generator)
    scores_2 = scores_1 + 0.1 * torch.randn(128, generator=generator)
    config = _metric_config(n_epochs=4)

    cap = ApproximationCapacity(**config)
    consistency = PosteriorConsistency(**config)
    cap.update(scores_1, scores_2)
    consistency.update(scores_1, scores_2)

    assert consistency.selected_beta == pytest.approx(cap.selected_beta)
    assert consistency.compute() >= cap.compute()
    assert consistency.compute() <= 1e-6


def test_training_callback_is_a_drop_in_cap_replacement() -> None:
    assert inspect.signature(PosteriorConsistencyCallback) == inspect.signature(CAPCallback)

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

    value, rankcorr = callback._compute_cap()

    assert callback.metric_name == "cap"
    assert value == pytest.approx(0.0, abs=1e-7)
    assert rankcorr == pytest.approx(1.0)


def test_evaluation_callback_is_a_drop_in_cap_replacement() -> None:
    assert inspect.signature(EvaluationPosteriorConsistency) == inspect.signature(EvaluationCAP)
    assert EvaluationConsistencyMetric is PosteriorConsistency

    callback = EvaluationPosteriorConsistency(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="none",
        cap_metric_config=_metric_config(),
    )

    assert callback.name == "cap"
    assert callback.metric_cls is PosteriorConsistency
