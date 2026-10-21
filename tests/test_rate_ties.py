import logging

import pytest
import torch

from src.callbacks.components import rate as train_rate
from src.callbacks.efficiency import AnomalyEfficiencyCallback as TrainEfficiency
from src.evaluation.callbacks.efficiency import (
    AnomalyEfficiencyCallback as EvalEfficiency,
)

RATES = [train_rate.AnomalyRate]
# fpr: threshold, validation count and test efficiency of the code before October
# 2026 on the `scores` fixture; fpr N = 36.5 at 3.65e-4, as at the physics point.
REGRESSION = {
    1e-3: (3.0973734855651855, 101, 0.016909999772906303),
    1e-2: (2.3225443363189697, 1000, 0.09239999949932098),
    3.65e-4: (3.4101455211639404, 37, 0.007420000154525042),
}


@pytest.fixture(scope="module")
def scores() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    val = torch.randn(100_000)
    test = torch.randn(100_000) + 1.0
    return val, test


def _old_count(scores: torch.Tensor, threshold: torch.Tensor) -> int:
    return int((scores.float().view(-1) >= threshold).sum())


@pytest.mark.parametrize("anomaly_rate", RATES)
@pytest.mark.parametrize("fpr", list(REGRESSION))
def test_untied_test_efficiency_unchanged(scores, anomaly_rate, fpr):
    val, test = scores
    reference = anomaly_rate(fpr, None)
    reference.set_threshold(val)
    old_eff = _old_count(test, reference.threshold) / test.numel()

    rate = anomaly_rate(fpr, None)
    rate.apply_threshold(reference.threshold, reference.tie_weight)
    for batch in test.split(16_384):
        rate.update(batch)
    assert rate.ntriggered.item() == _old_count(test, reference.threshold)
    assert rate.efficiency().item() == pytest.approx(old_eff, rel=1e-6)
    assert rate.efficiency().item() == pytest.approx(REGRESSION[fpr][2], rel=1e-6)

    reference.update(test)
    assert reference.efficiency().item() == rate.efficiency().item()


@pytest.mark.parametrize("anomaly_rate", RATES)
def test_constant_score_has_efficiency_fpr(anomaly_rate):
    fpr = 0.01
    rate = anomaly_rate(fpr, None)
    rate.set_threshold(torch.full((1_000,), 0.5))
    assert rate.tie_weight.item() == pytest.approx(fpr, rel=1e-12)

    rate.update(torch.full((5_000,), 0.5))
    assert rate.efficiency().item() == pytest.approx(fpr, rel=1e-6)


def test_tie_buffer_reaches_the_evaluation_callback(scores, caplog):
    val, _ = scores
    fpr = 3.65e-4
    rate = train_rate.AnomalyRate(fpr, None)
    rate.set_threshold(val)

    training = TrainEfficiency(output_name="s")
    training.module_target_rate = fpr
    module = torch.nn.Module()
    training._set_thres_on_module(module, fpr, rate)
    training._set_thres_on_module(module, 2 * fpr, rate)
    assert module.tie_operational.item() == pytest.approx(0.5, abs=1e-9)
    assert module.tie_operational.dtype == torch.float64
    assert getattr(module, f"tie_{2 * fpr}kHz".replace(".", "_")).item() == 0.5

    evaluation = EvalEfficiency(output_name="s", ds=[])
    evaluation.operational_rate = fpr
    evaluation.target_rates_resolved = [fpr]
    thresholds, tie_weights = evaluation._get_thres(module)
    assert thresholds[fpr].item() == rate.threshold.item()
    assert tie_weights[fpr].item() == pytest.approx(0.5, abs=1e-9)

    del module.tie_operational
    caplog.set_level(logging.INFO, logger="src.evaluation.callbacks.efficiency")
    _, tie_weights = evaluation._get_thres(module)
    assert tie_weights[fpr] == 1.0
    assert "tie_operational" in caplog.text
