import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.callbacks.components import selection
from src.callbacks.components.cap.binary import get_pairing_fn
from src.callbacks.components.cap.metric import ApproximationCapacity, consistency_limit
from src.callbacks.consistency_limit import ConsistencyLimitCallback
from src.evaluation.callbacks.components.cap.metric import (
    consistency_limit as evaluation_consistency_limit,
)
from src.evaluation.callbacks.consistency_limit import ConsistencyLimit

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PROVISIONAL = OmegaConf.to_container(
    OmegaConf.load(CONFIGS / "cap" / "provisional.yaml")
)
# Unnormalised scores under the baseline energy E(p, y) = y (1 - p) + (1 - y) p.
BARE = {"normalization_type": "none", "energy_type": "baseline"}
LIMITS = [consistency_limit, evaluation_consistency_limit]
KEY = "ascore/full"
SETS = ["normal", "other", "reference_normal"]
PAIR = {"output_name": KEY, "dataset_1": "normal", "dataset_2": "reference_normal"}
PAIR |= {"pairing_type": "cdf"}


def _scores(n=1000, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    scores_1 = torch.rand(n, generator=g, dtype=dtype)
    return scores_1, scores_1 + 0.2 * torch.randn(n, generator=g, dtype=dtype)


def _sets(seed):
    # Two sets of different sizes, so that the pairing drops rows.
    g = torch.Generator().manual_seed(seed)
    return torch.rand(300, generator=g), torch.rand(200 + 40 * seed, generator=g) ** 2


def _limit_of(scores_1, scores_2):
    index_1, index_2 = get_pairing_fn("cdf")(scores_1, scores_2)
    paired_1, paired_2 = scores_1[index_1], scores_2[index_2]
    n = min(len(paired_1), len(paired_2))
    return consistency_limit(paired_1[:n], paired_2[:n], **PROVISIONAL)


def _mean_log_cosine(p_1, p_2, beta):
    # The two-state posteriors over y of the baseline energies E(p, 0) = p and
    # E(p, 1) = 1 - p, in log space; shapes (2, n).
    log_q = [
        torch.log_softmax(-beta * torch.stack([p, 1 - p]), dim=0) for p in (p_1, p_2)
    ]
    overlap = torch.logsumexp(log_q[0] + log_q[1], dim=0)
    norm_1, norm_2 = (0.5 * torch.logsumexp(2 * q, dim=0) for q in log_q)
    return (overlap - norm_1 - norm_2).mean().item()


@pytest.mark.parametrize("limit", LIMITS)
def test_limit_is_the_leading_term_of_the_log_cosine(limit):
    p_1, p_2 = _scores()
    value = limit(p_1, p_2, **BARE)
    errors = [
        abs(8 * _mean_log_cosine(p_1, p_2, beta) / beta**2 - value) / abs(value)
        for beta in (1e-2, 1e-3)
    ]
    assert errors[1] < 1e-4
    # The remainder is O(beta**2) relative to the leading term.
    assert errors[1] < errors[0] / 50


@pytest.mark.parametrize("limit", LIMITS)
def test_baseline_limit_is_four_mean_squared_differences(limit):
    # The baseline gap is D = 1 - 2p, so D1 - D2 = 2 (p2 - p1).
    p_1, p_2 = _scores()
    expected = -4 * ((p_1 - p_2) ** 2).mean().item()
    assert limit(p_1, p_2, **BARE) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("limit", LIMITS)
def test_limit_is_zero_on_equal_samples_and_negative_otherwise(limit):
    scores_1, scores_2 = _scores(dtype=torch.float32)
    equal = limit(scores_1, scores_1.clone(), **PROVISIONAL)
    # Not exactly zero: the two halves of the jointly normalised scores can take
    # different vectorised paths through the energy and differ in their last bits.
    assert equal == pytest.approx(0, abs=1e-12)
    assert limit(scores_1, scores_2, **PROVISIONAL) < -1e-6


def test_limit_is_computed_in_the_dtype_of_the_scores():
    scores = _scores(dtype=torch.float32)
    value = consistency_limit(*scores, **PROVISIONAL)
    in_float64 = consistency_limit(*(s.double() for s in scores), **PROVISIONAL)
    assert float(np.float32(value)) == value
    assert value == pytest.approx(in_float64, rel=1e-5)


def test_configuration_is_read_and_left_unchanged():
    config = OmegaConf.create(PROVISIONAL)
    scores = _scores(dtype=torch.float32)
    value = consistency_limit(*scores, **config)
    grid = {"beta_min": 1.0, "beta_max": 2.0, "n_beta": 3}
    assert consistency_limit(*scores, **(PROVISIONAL | grid)) == value
    assert OmegaConf.to_container(config) == PROVISIONAL
    assert PROVISIONAL["energy_params"] == {"scale": 0.5}


def test_cap_keeps_the_pooled_moments_of_its_update():
    metric = ApproximationCapacity(**PROVISIONAL)
    scores = _scores(dtype=torch.float32)
    metric.update(*scores)
    pooled = torch.cat(metric.normalizer_fn(*scores)).double()
    moments = {"mean": pooled.mean().item(), "std": pooled.std().item()}
    assert metric.energy_params == {"scale": 0.5} | moments


class _Module:
    def __init__(self):
        self.device, self.logged = torch.device("cpu"), []

    def log_dict(self, metrics: dict, **kwargs) -> None:
        self.logged.append(metrics)


def _feed(batch_end, trainer, module, scores_1, scores_2):
    # The middle loader is a set that neither callback reads.
    for idx, scores in enumerate((scores_1, torch.zeros(10), scores_2)):
        for batch in scores.split(64):
            batch_end(trainer, module, {KEY: batch}, None, 0, idx)


def _validate(callback, trainer, module, scores):
    callback.on_validation_epoch_start(trainer, module)
    _feed(callback.on_validation_batch_end, trainer, module, *scores)
    callback.on_validation_epoch_end(trainer, module)


def test_training_callback_logs_the_moving_average_of_the_limit():
    callback = ConsistencyLimitCallback(**PAIR, cap_metric_config=PROVISIONAL)
    trainer = SimpleNamespace(val_dataloaders=dict.fromkeys(SETS), sanity_checking=True)
    module = _Module()
    callback.on_fit_start(trainer, module)
    _validate(callback, trainer, module, _sets(9))
    assert module.logged == []

    trainer.sanity_checking = False
    epochs = [_sets(seed) for seed in range(4)]
    epochs[2][1][7] = math.nan
    for scores in epochs:
        _validate(callback, trainer, module, scores)
    values = [_limit_of(*scores) for scores in epochs]
    assert math.isnan(values[2])
    key = "val/summary/consistency_ema_normal_vs_reference_normal"
    expected = selection.ema(values, skip_nonfinite=True)
    assert module.logged == [{key: average} for average in expected]


def test_training_callback_needs_both_sets():
    callback = ConsistencyLimitCallback(**PAIR, cap_metric_config=PROVISIONAL)
    trainer = SimpleNamespace(val_dataloaders={"normal": None})
    callback.on_fit_start(trainer, _Module())
    with pytest.raises(ValueError, match="reference_normal"):
        callback.on_validation_epoch_start(trainer, _Module())


def test_evaluation_callback_keeps_the_limit_of_every_checkpoint(tmp_path):
    callback = ConsistencyLimit(**PAIR, cap_metric_config=PROVISIONAL)
    trainer = SimpleNamespace(
        test_dataloaders=dict.fromkeys(SETS),
        split="val",
        metric_records={},
        logger=None,
    )
    values = {}
    for seed, ckpt in enumerate(["epoch_0000", "epoch_0001", "epoch_0002"]):
        module = SimpleNamespace(_ckpt_path=str(tmp_path / f"{ckpt}.ckpt"))
        callback.on_test_epoch_start(trainer, module)
        _feed(callback.on_test_batch_end, trainer, module, *_sets(seed))
        callback.on_test_epoch_end(trainer, module)
        values[ckpt] = _limit_of(*_sets(seed))
        assert trainer.metric_records == {"consistency/consistency": values[ckpt]}

    best = max(values, key=values.get)
    assert callback.get_optimized_metric() == (best, values[best])
    callback.plot_summary(trainer, tmp_path)
    summary = tmp_path / "plots" / "val" / "consistency_summary" / "summary.pkl"
    with open(summary, "rb") as file:
        assert pickle.load(file) == values
    callback.clear_crit_summary()
    with pytest.raises(ValueError):
        callback.get_optimized_metric()
