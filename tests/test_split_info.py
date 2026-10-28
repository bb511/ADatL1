"""The split record: seed and fractions of a frozen split, and the prepared-cache file."""

from pathlib import Path

import numpy as np
import pytest

from src.data.components import split_info as split_record

SETTINGS = split_record.split_settings(666, {"train": 0.6, "valid": 0.2, "test": 0.2}, 0.6)


def test_settings_compare_seed_exactly_and_fractions_numerically():
    assert split_record.same_settings(SETTINGS, {**SETTINGS, "train": 0.6 + 1e-15})
    assert not split_record.same_settings(SETTINGS, {**SETTINGS, "seed": 1})
    assert not split_record.same_settings(SETTINGS, {**SETTINGS, "valid": 0.25})


def test_partition_check_accepts_a_partition_and_reports_sizes():
    parts = np.array_split(np.random.default_rng(0).permutation(10), 3)
    assert split_record.check_partition(parts, 10) == [4, 3, 3]


@pytest.mark.parametrize("parts", [
    (np.array([0, 1, 2]), np.array([2, 3])),      # duplicate
    (np.array([0, 1]), np.array([3, 4])),         # gap
    (np.array([0, 1]), np.array([2, 7])),         # out of range
])
def test_partition_check_refuses_overlap_gaps_and_strays(parts):
    with pytest.raises(split_record.SplitMismatchError):
        split_record.check_partition(parts)


def test_partition_check_refuses_the_wrong_total():
    with pytest.raises(split_record.SplitMismatchError, match="expected 5"):
        split_record.check_partition((np.arange(3),), 5)


def test_overlap_fraction_and_digest():
    assert split_record.overlap_fraction(np.array([1, 2, 3, 4]), np.array([2, 4, 9])) == 0.5
    assert split_record.index_digest(np.arange(5)) == split_record.index_digest(np.arange(5, dtype=np.int32))
    assert split_record.index_digest(np.arange(5)) != split_record.index_digest(np.arange(1, 6))


def _record(status="complete", settings=SETTINGS):
    return {"status": status, "settings": settings, "cache_name": "c_split666"}


def test_required_record_must_exist(tmp_path: Path):
    with pytest.raises(split_record.SplitNotPreparedError, match="DATA_SPLIT_SEED=666"):
        split_record.require_split_info(tmp_path, SETTINGS, "c_split666")


def test_required_record_must_be_complete(tmp_path: Path):
    split_record.write_split_info(tmp_path, _record(status="in_progress"))
    with pytest.raises(split_record.SplitNotPreparedError, match="in_progress"):
        split_record.require_split_info(tmp_path, SETTINGS, "c_split666")


def test_required_record_must_match_the_requested_split(tmp_path: Path):
    split_record.write_split_info(tmp_path, _record())
    with pytest.raises(split_record.SplitMismatchError, match="seed 1"):
        split_record.require_split_info(tmp_path, {**SETTINGS, "seed": 1}, "c_split666")


def test_complete_matching_record_is_returned(tmp_path: Path):
    split_record.write_split_info(tmp_path, _record())
    assert split_record.require_split_info(tmp_path, SETTINGS, "c_split666")["status"] == "complete"
    assert not (tmp_path / "split_info.yaml.tmp").exists()
