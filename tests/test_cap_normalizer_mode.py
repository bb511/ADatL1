import pytest
import torch
from omegaconf import OmegaConf

from src.callbacks.components.cap.binary.normalization import minmax
from src.callbacks.components.cap.metric import ApproximationCapacity
from src.evaluation.callbacks.components.cap.metric import (
    ApproximationCapacity as EvaluationApproximationCapacity,
)


@pytest.mark.parametrize(
    "metric_cls",
    [ApproximationCapacity, EvaluationApproximationCapacity],
    ids=["train", "eval"],
)
def test_metrics_built_from_one_config_all_split(metric_cls):
    # The CAP callbacks rebuild their metric from one shared config; popping mode from
    # it left every metric after the first in joint mode.
    config = OmegaConf.create(
        {"normalization_type": "minmax", "normalization_params": {"mode": "split"}}
    )
    g = torch.Generator().manual_seed(0)
    # Disjoint ranges, on which joint and split normalization differ.
    scores_1, scores_2 = torch.rand(64, generator=g), 3.0 + torch.rand(64, generator=g)
    for _ in range(2):
        normalizer_fn = metric_cls(**config).normalizer_fn
        normalized_1, normalized_2 = normalizer_fn(scores_1, scores_2)
        assert torch.equal(normalized_1, minmax(scores_1))
        assert torch.equal(normalized_2, minmax(scores_2))
    assert config.normalization_params == {"mode": "split"}
