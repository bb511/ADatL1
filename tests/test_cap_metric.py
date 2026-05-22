import pytest
import torch

from src.callbacks.metrics.cap.metric import ApproximationCapacity as TrainingCAP
from src.evaluation.callbacks.metrics.cap.metric import (
    ApproximationCapacity as EvaluationCAP,
)


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
