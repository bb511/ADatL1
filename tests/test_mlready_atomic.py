import numpy as np
import pytest

from src.data.components import mlready
from src.data.components.mlready import L1DataMLReady


@pytest.fixture
def ready(tmp_path):
    ready = L1DataMLReady(
        processed_datapath=str(tmp_path),
        split={"train": 0.6, "valid": 0.2, "test": 0.2},
        split_aux=0.6,
        cache_root_dir=str(tmp_path),
        name="test",
    )
    ready.splits_dir = tmp_path / "splits"

    return ready


def _drawn():
    return tuple(np.array_split(np.random.default_rng(0).permutation(100), 3))


def test_atomic_write_matches_plain_savez(ready, tmp_path):
    drawn = _drawn()
    ready._frozen_split("zerobias", drawn)
    reference_file = tmp_path / "reference.npz"
    np.savez(reference_file, itrain=drawn[0], ivalid=drawn[1], itest=drawn[2])

    with np.load(ready.splits_dir / "zerobias.npz") as written:
        with np.load(reference_file) as reference:
            assert written.files == reference.files
            for key in reference.files:
                assert written[key].dtype == reference[key].dtype
                assert np.array_equal(written[key], reference[key])


def test_atomic_write_leaves_only_the_split(ready):
    ready._frozen_split("aux__sample", (np.arange(60), np.arange(60, 100)))
    split_file = ready.splits_dir / "aux__sample.npz"

    assert list(ready.splits_dir.iterdir()) == [split_file]
    assert split_file.stat().st_mode & 0o777 == 0o644


def test_failed_write_leaves_no_split(ready, monkeypatch):
    def fail_midway(file, **arrays):
        file.write(b"PK\x03\x04 half an archive")
        raise OSError("disk full")

    monkeypatch.setattr(mlready.np, "savez", fail_midway)
    with pytest.raises(OSError, match="disk full"):
        ready._frozen_split("zerobias", _drawn())

    assert list(ready.splits_dir.iterdir()) == []


def test_failed_write_does_not_block_a_retry(ready, monkeypatch):
    def fail(file, **arrays):
        raise OSError("disk full")

    monkeypatch.setattr(mlready.np, "savez", fail)
    with pytest.raises(OSError):
        ready._frozen_split("zerobias", _drawn())
    monkeypatch.undo()

    frozen = ready._frozen_split("zerobias", _drawn())
    reloaded = ready._frozen_split("zerobias", _drawn())
    assert all(np.array_equal(a, b) for a, b in zip(frozen, reloaded))
