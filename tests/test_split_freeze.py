"""The frozen split must stay the split the paper trained on."""

import numpy as np
import pytest

from src.data.components.mlready import L1DataMLReady

# Measured on the processed 2025E+G data. If these move, the published dataset and every
# reported number disagree, so they are asserted rather than derived.
ZEROBIAS_EVENTS = 20_887_179
ZEROBIAS_SPLIT = (12_532_307, 4_177_435, 4_177_437)


def test_zerobias_split_sizes_are_the_published_ones():
    """The 60/20/20 rule reproduces the counts the deposition ships."""
    n = ZEROBIAS_EVENTS
    ntrain = int(0.6 * n)
    nvalid = int(0.2 * n) + ntrain

    assert (ntrain, nvalid - ntrain, n - nvalid) == ZEROBIAS_SPLIT


def test_seeded_permutation_is_stable():
    """NumPy's PCG64 stream for seed 42 still gives the same first draws.

    The whole split rests on this. A NumPy change here would silently repartition the
    data, so it is worth a test that fails loudly rather than a comment.
    """
    assert np.random.default_rng(42).permutation(10).tolist() == [5, 6, 0, 7, 3, 2, 4, 9, 1, 8]


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


def test_frozen_split_round_trips(tmp_path):
    """Freezing then reloading returns the identical arrays."""
    ready = _mlready(tmp_path)
    drawn = tuple(np.array_split(np.random.default_rng(0).permutation(100), 3))
    frozen = ready._frozen_split("zerobias", drawn)
    reloaded = _mlready(tmp_path)._frozen_split("zerobias", drawn)

    assert all(np.array_equal(a, b) for a, b in zip(frozen, reloaded))


def test_frozen_split_wins_over_a_fresh_draw(tmp_path):
    """Once frozen, a different draw is ignored -- that is the point of freezing."""
    ready = _mlready(tmp_path)
    original = tuple(np.array_split(np.arange(100), 3))
    ready._frozen_split("zerobias", original)

    shuffled = tuple(np.array_split(np.random.default_rng(1).permutation(100), 3))
    loaded = _mlready(tmp_path)._frozen_split("zerobias", shuffled)

    assert all(np.array_equal(a, b) for a, b in zip(loaded, original))


def test_frozen_split_refuses_a_stale_file(tmp_path):
    """A frozen split covering the wrong number of events must not be used silently."""
    ready = _mlready(tmp_path)
    ready._frozen_split("zerobias", tuple(np.array_split(np.arange(100), 3)))

    with pytest.raises(ValueError, match="covers"):
        _mlready(tmp_path)._frozen_split("zerobias", tuple(np.array_split(np.arange(50), 3)))


def test_aux_split_freezes_two_way(tmp_path):
    """Auxiliary datasets have only valid and test, and round trip the same way."""
    ready = _mlready(tmp_path)
    drawn = (np.arange(60), np.arange(60, 100))
    frozen = ready._frozen_split("aux__sample", drawn)
    reloaded = _mlready(tmp_path)._frozen_split("aux__sample", drawn)

    assert len(frozen) == 2
    assert all(np.array_equal(a, b) for a, b in zip(frozen, reloaded))
