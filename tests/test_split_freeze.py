"""The frozen split must stay the split the paper trained on."""

import numpy as np
import pytest

from src.data.components.mlready import L1DataMLReady

# Measured on the processed 2025E+G data. If these move, the published dataset and every
# reported number disagree, so they are asserted rather than derived.
ZEROBIAS_EVENTS = 20_887_179
ZEROBIAS_SPLIT = (12_532_307, 4_177_435, 4_177_437)


def test_zerobias_split_sizes_are_the_published_ones():
    n = ZEROBIAS_EVENTS
    ntrain = int(0.6 * n)
    nvalid = int(0.2 * n) + ntrain

    assert (ntrain, nvalid - ntrain, n - nvalid) == ZEROBIAS_SPLIT


def _mlready(tmp_path, seed=42):
    ready = L1DataMLReady(
        processed_datapath=str(tmp_path),
        split={"train": 0.6, "valid": 0.2, "test": 0.2},
        split_aux=0.6,
        cache_root_dir=str(tmp_path),
        name="test",
        seed=seed,
    )
    ready.splits_dir = tmp_path / "splits"
    ready.rng = np.random.default_rng(seed)

    return ready


def test_frozen_split_wins_over_a_fresh_draw(tmp_path):
    # Once frozen, a different draw is ignored: that is the point of freezing.
    ready = _mlready(tmp_path)
    original = tuple(np.array_split(np.arange(100), 3))
    ready._frozen_split("zerobias", original)

    shuffled = tuple(np.array_split(np.random.default_rng(1).permutation(100), 3))
    loaded = _mlready(tmp_path)._frozen_split("zerobias", shuffled)

    assert all(np.array_equal(a, b) for a, b in zip(loaded, original))


def test_frozen_split_refuses_a_stale_file(tmp_path):
    # A frozen split covering the wrong number of events must not be used silently.
    ready = _mlready(tmp_path)
    ready._frozen_split("zerobias", tuple(np.array_split(np.arange(100), 3)))

    with pytest.raises(ValueError, match="covers"):
        _mlready(tmp_path)._frozen_split(
            "zerobias", tuple(np.array_split(np.arange(50), 3))
        )
