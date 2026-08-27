import numpy as np
import pytest

from src.evaluation.leakage_probe import (
    PROBE_INNER_SPLIT_SEED,
    PROBE_INNER_VALIDATION_FRACTION,
    ProbePartitionError,
    make_probe_inner_partition,
)


def test_inner_partition_has_frozen_membership() -> None:
    partition = make_probe_inner_partition(10)

    np.testing.assert_array_equal(
        partition.fit_indices,
        np.array(
            [1, 2, 3, 4, 5, 6, 8, 9],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        partition.validation_indices,
        np.array(
            [0, 7],
            dtype=np.int64,
        ),
    )

    assert partition.seed == PROBE_INNER_SPLIT_SEED
    assert (
        partition.validation_fraction
        == PROBE_INNER_VALIDATION_FRACTION
    )
    assert len(partition.manifest_hash) == 64


def test_inner_partition_is_disjoint_and_exhaustive() -> None:
    n_events = 101
    partition = make_probe_inner_partition(n_events)

    fit = set(partition.fit_indices.tolist())
    validation = set(
        partition.validation_indices.tolist()
    )

    assert fit.isdisjoint(validation)
    assert fit | validation == set(range(n_events))

    assert len(partition.validation_indices) == 21
    assert len(partition.fit_indices) == 80


def test_inner_partition_is_reproducible() -> None:
    first = make_probe_inner_partition(100)
    second = make_probe_inner_partition(100)

    np.testing.assert_array_equal(
        first.fit_indices,
        second.fit_indices,
    )
    np.testing.assert_array_equal(
        first.validation_indices,
        second.validation_indices,
    )
    assert first.manifest_hash == second.manifest_hash


def test_different_seed_changes_partition_membership() -> None:
    first = make_probe_inner_partition(
        100,
        seed=12345,
    )
    second = make_probe_inner_partition(
        100,
        seed=54321,
    )

    assert not np.array_equal(
        first.validation_indices,
        second.validation_indices,
    )
    assert first.manifest_hash != second.manifest_hash


def test_partition_indices_can_be_reused_for_all_arrays() -> None:
    n_events = 20
    partition = make_probe_inner_partition(n_events)

    latent = np.arange(n_events * 2).reshape(
        n_events,
        2,
    )
    reconstruction = np.arange(n_events * 4).reshape(
        n_events,
        4,
    )
    target = np.arange(n_events)

    assert (
        latent[partition.fit_indices].shape[0]
        == reconstruction[partition.fit_indices].shape[0]
        == target[partition.fit_indices].shape[0]
    )

    assert (
        latent[partition.validation_indices].shape[0]
        == reconstruction[partition.validation_indices].shape[0]
        == target[partition.validation_indices].shape[0]
    )


def test_partition_indices_are_read_only() -> None:
    partition = make_probe_inner_partition(20)

    with pytest.raises(ValueError):
        partition.fit_indices[0] = 0

    with pytest.raises(ValueError):
        partition.validation_indices[0] = 0


@pytest.mark.parametrize(
    "n_events",
    [0, -1, 1.5, True],
)
def test_invalid_event_count_is_rejected(
    n_events,
) -> None:
    with pytest.raises(ProbePartitionError) as error:
        make_probe_inner_partition(n_events)

    assert error.value.reason == "invalid_event_count"


def test_partition_too_small_for_r2_is_rejected() -> None:
    with pytest.raises(ProbePartitionError) as error:
        make_probe_inner_partition(5)

    assert error.value.reason == "inner_partition_too_small"


@pytest.mark.parametrize(
    "validation_fraction",
    [
        -0.1,
        0.0,
        1.0,
        1.1,
        np.nan,
        np.inf,
    ],
)
def test_invalid_validation_fraction_is_rejected(
    validation_fraction: float,
) -> None:
    with pytest.raises(ProbePartitionError) as error:
        make_probe_inner_partition(
            20,
            validation_fraction=validation_fraction,
        )

    assert (
        error.value.reason
        == "invalid_inner_validation_fraction"
    )


@pytest.mark.parametrize(
    "seed",
    [1.5, "12345", True],
)
def test_invalid_seed_is_rejected(seed) -> None:
    with pytest.raises(ProbePartitionError) as error:
        make_probe_inner_partition(
            20,
            seed=seed,
        )

    assert error.value.reason == "invalid_inner_split_seed"