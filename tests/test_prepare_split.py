"""src/prepare_split.py: the checks run on a freshly prepared split."""

from pathlib import Path

import numpy as np
import pytest

from src.data.components import split_info as split_record
from src.prepare_split import _reference_splits_dir, check_splits


def _freeze(splits_dir: Path, seed: int, n: int = 1000, aux: int = 50) -> None:
    splits_dir.mkdir(parents=True)
    permutation = np.random.default_rng(seed).permutation(n)
    train, valid, test = np.split(permutation, [int(0.6 * n), int(0.8 * n)])
    np.savez(splits_dir / "zerobias.npz", itrain=train, ivalid=valid, itest=test)
    aux_perm = np.random.default_rng(seed + 1).permutation(aux)
    np.savez(splits_dir / "aux__SignalA.npz", ivalid=aux_perm[:30], itest=aux_perm[30:])


def test_checks_count_every_split_and_measure_the_overlap(tmp_path):
    _freeze(tmp_path / "ref", seed=42)
    _freeze(tmp_path / "new", seed=666)

    record = check_splits(tmp_path / "new", tmp_path / "ref")

    assert record["zerobias"]["events"] == 1000
    assert (record["zerobias"]["train"], record["zerobias"]["valid"], record["zerobias"]["test"]) == (600, 200, 200)
    assert record["aux"] == {"SignalA": {"valid": 30, "test": 20}}
    overlap = record["overlap_with_reference"]
    # Independent random 60 % draws share about 60 % of their events.
    assert 0.5 < overlap["train_in_reference_train"] < 0.7
    assert 0.5 < overlap["test_in_reference_train"] < 0.7
    assert len(record["zerobias"]["sha256"]["train"]) == 64


def test_a_new_split_identical_to_the_reference_is_refused(tmp_path):
    _freeze(tmp_path / "ref", seed=42)
    _freeze(tmp_path / "new", seed=42)
    with pytest.raises(split_record.SplitMismatchError, match="identical"):
        check_splits(tmp_path / "new", tmp_path / "ref")


def test_a_broken_partition_is_refused(tmp_path):
    splits = tmp_path / "new"
    splits.mkdir()
    np.savez(splits / "zerobias.npz", itrain=np.arange(6), ivalid=np.arange(5, 8), itest=np.arange(8, 10))
    with pytest.raises(split_record.SplitMismatchError):
        check_splits(splits, None)


def test_reference_is_the_cache_without_the_suffix(tmp_path):
    assert _reference_splits_dir(tmp_path, "proc_default_split666", "_split666") == \
        tmp_path / "proc_default" / "splits"
    assert _reference_splits_dir(tmp_path, "proc_default", "") is None
