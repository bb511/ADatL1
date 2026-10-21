import math

import numpy as np
import pytest
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from src.callbacks.cap import CAPCallback
from src.callbacks.components import selection
from src.callbacks.thres_drift import ThresholdDriftCallback
from tests.test_score_dump import fit


class _Script(Callback):
    def __init__(self, values: dict[str, list[float]]):
        self.values = values

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        pl_module.log_dict({k: v[epoch] for k, v in self.values.items()})


def test_best_epoch_matches_model_checkpoint(tmp_path):
    # In float32, 0.7 + 1e-9 ties 0.7: the earlier epoch must be kept.
    maxima = [math.nan, 0.5, 0.7, 0.7, math.nan, 0.7 + 1e-9, 0.6, 0.0]
    minima = [1.0, math.nan, 0.4, 0.4 + 1e-9, 0.3, 0.3, 1.0, 1.0]
    values = {"val/summary/a_ema": maxima, "val/summary/b_ema": minima}
    modes = {"val/summary/a_ema": "max", "val/summary/b_ema": "min"}
    checkpoints = {
        key: ModelCheckpoint(
            dirpath=tmp_path / key.rsplit("/", 1)[1],
            monitor=key,
            mode=mode,
            save_top_k=1,
            save_on_train_epoch_end=False,
        )
        for key, mode in modes.items()
    }
    callbacks = [_Script(values), *checkpoints.values()]
    fit(tmp_path, callbacks, epochs=8, enable_checkpointing=True)

    for key, expected in (("val/summary/a_ema", 2), ("val/summary/b_ema", 4)):
        logged = [selection.as_logged(v) for v in values[key]]
        assert selection.best_epoch(logged, modes[key]) == expected
        saved = torch.load(checkpoints[key].best_model_path, weights_only=False)
        assert saved["epoch"] == expected


def test_ema_is_the_callbacks_average():
    callback = ThresholdDriftCallback(output_name="ascore/full")
    callback.on_fit_start(None, None)
    values = [0.3, 0.1, math.nan, 0.2]
    expected = []
    for value in values[:2]:
        callback._compute_drift_ema("operational", value)
        expected.append(callback.drift_ema["operational"])
    assert selection.ema(values[:2]) == expected
    assert all(math.isnan(v) for v in selection.ema(values)[2:])


def test_ema_skipping_non_finite_values_is_the_cap_average():
    callback = CAPCallback(
        output_name="ascore/full",
        dataset_1="normal",
        dataset_2="reference_normal",
        pairing_type="cdf",
        cap_metric_config={},
    )
    callback.on_fit_start(None, None)
    values = [math.nan, 0.3, math.inf, 0.1]
    expected = []
    for value in values:
        callback._compute_cap_ema(value)
        expected.append(callback.cap_ema)
    smoothed = selection.ema(values, skip_nonfinite=True)
    assert math.isnan(smoothed[0]) and math.isnan(expected[0])
    assert smoothed[1:] == expected[1:]


@pytest.mark.parametrize("seed", range(20))
def test_criterion_alone_is_the_checkpoint_rule(seed):
    rng = np.random.default_rng(seed)
    # Coarse values, so that ties occur, and NaNs, first value included.
    q_main = rng.integers(0, 4, size=12).astype(float)
    q_main[rng.random(12) < 0.2] = math.nan
    q_sec = rng.normal(size=12)
    for mode in ("max", "min"):
        assert selection.pick(q_main, q_sec, selection.W_STEPS, mode) == (
            selection.best_epoch(list(q_main), mode)
        )


def test_q_sec_alone_breaks_ties_by_the_better_q_main():
    q_main = [0.1, 0.5, 0.3, 0.5]
    q_sec = [2.0, 1.0, 1.0, 1.0]
    # Epochs 1, 2 and 3 tie on Q''; 1 and 3 tie on Q' too, so the earlier wins.
    assert selection.pick(q_main, q_sec, 0, "max") == 1
    assert selection.pick(q_main, q_sec, 0, "min") == 2


def test_weighted_rank_sum():
    # Ranks of Q' 1..8, of Q'' 3, 1, 2, 4..8: at w = 0.3 the sums are 2.4, 1.3, 2.3, ...
    q_main = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    q_sec = [3.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert selection.pick(q_main, q_sec, 3, "max") == 1
    # Ranks (1, 2, 3) against (3, 2, 1): at w = 0.5 all three tie, and the better Q'
    # wins.
    assert selection.pick([3.0, 2.0, 1.0], [3.0, 2.0, 1.0], 5, "max") == 0


def test_epoch_pick_reads_the_logged_ema():
    q_main = [0.0, 1.0, 0.0, 0.0, 0.0]
    q_sec = [0.0] * 5
    # The EMA of q_main peaks at epoch 1, as the raw value does.
    assert selection.epoch_pick(q_main, q_sec, selection.W_STEPS, "max") == 1
    assert selection.smoothed(q_main)[1] == selection.as_logged(0.1)
