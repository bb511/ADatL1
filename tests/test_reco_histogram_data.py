"""ReconstructionPlots writes its histograms as data with run-independent bins."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.evaluation.callbacks.reco import (  # noqa: E402
    ReconstructionPlots,
    counts_with_flow,
    histogram_frame,
)

KEY = ("jets", "Et")


def _feed(batches_in, batches_out, n_batches=10):
    callback = ReconstructionPlots(warmup_batches=0.2, output_name="reco", ckpts={"loss_total": True})
    # on_test_epoch_start needs a real module; set up the same state directly.
    callback._buffers, callback._edges, callback._hist_input = {}, {}, {}
    callback._hist_output, callback._batch_counts = {}, {}
    callback._data_edges, callback._data_input, callback._data_output = {}, {}, {}
    trainer = SimpleNamespace(test_dataloaders={"normal": list(range(n_batches))})
    for x_in, x_out in zip(batches_in, batches_out):
        callback._update_hist_pair(trainer, "normal", KEY, torch.tensor(x_in), torch.tensor(x_out))
    return callback


def test_counts_with_flow_keeps_every_entry():
    counts = counts_with_flow(np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 7.0]), np.array([0.0, 1.0, 2.0]))
    assert counts.tolist() == [1, 2, 2, 1]  # underflow, [0, 1), [1, 2], overflow
    frame = histogram_frame(np.array([0.0, 1.0, 2.0]), counts, counts)
    assert frame["bin_low"].tolist()[0] == -np.inf and frame["bin_high"].tolist()[-1] == np.inf


def test_two_models_on_the_same_input_get_the_same_bins(tmp_path):
    rng = np.random.default_rng(0)
    inputs = [rng.gamma(2.0, 10.0, 500).astype(np.float32) for _ in range(10)]
    good = _feed(inputs, [x + rng.normal(0, 1, x.size).astype(np.float32) for x in inputs])
    bad = _feed(inputs, [3 * x for x in inputs])

    assert np.array_equal(good._data_edges["normal"][KEY], bad._data_edges["normal"][KEY])
    # Every entry is counted, the warmup batches included, out-of-range ones as flow.
    for callback in (good, bad):
        assert callback._data_input["normal"][KEY].sum() == 5000
        assert callback._data_output["normal"][KEY].sum() == 5000
    assert np.array_equal(good._data_input["normal"][KEY], bad._data_input["normal"][KEY])
    assert bad._data_output["normal"][KEY][-1] > 0  # the 3x reconstruction overflows

    good._write_histogram_data(tmp_path, "normal", KEY)
    written = pd.read_csv(tmp_path / "data" / "jets_Et.csv")
    assert list(written.columns) == ["bin_low", "bin_high", "input", "reco"]
    assert written["input"].sum() == 5000


def test_plot_binning_is_unchanged():
    """The plots keep Doane edges of input and reconstruction together."""
    rng = np.random.default_rng(1)
    inputs = [rng.normal(0, 1, 300).astype(np.float32) for _ in range(10)]
    outputs = [2 * x for x in inputs]
    callback = _feed(inputs, outputs)
    warm = np.concatenate([np.concatenate([inputs[0], outputs[0]]),
                           np.concatenate([inputs[1], outputs[1]])])
    assert np.allclose(callback._edges["normal"][KEY], np.histogram_bin_edges(warm, bins="doane"))
