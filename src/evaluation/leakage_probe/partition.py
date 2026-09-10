"""Deterministic inner partitioning for MLP seed selection."""

import hashlib

import numpy as np

from .constants import (
    PROBE_INNER_SPLIT_SEED,
    PROBE_INNER_VALIDATION_FRACTION,
)
from .errors import ProbePartitionError
from .types import ProbeInnerPartition

def make_probe_inner_partition(
    n_events: int,
    *,
    seed: int = PROBE_INNER_SPLIT_SEED,
    validation_fraction: float = PROBE_INNER_VALIDATION_FRACTION,
) -> ProbeInnerPartition:
    """Create the deterministic 80/20 split used for probe-seed selection."""

    if (
        isinstance(n_events, bool)
        or not isinstance(n_events, (int, np.integer))
        or int(n_events) <= 0
    ):
        raise ProbePartitionError(
            "invalid_event_count",
            f"n_events must be a positive integer, got {n_events!r}.",
        )

    n_events = int(n_events)

    if (
        not np.isfinite(validation_fraction)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ProbePartitionError(
            "invalid_inner_validation_fraction",
            "validation_fraction must be finite and strictly between "
            f"zero and one, got {validation_fraction!r}.",
        )

    if isinstance(seed, bool) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ProbePartitionError(
            "invalid_inner_split_seed",
            f"seed must be an integer, got {seed!r}.",
        )

    seed = int(seed)

    # Ceil matches the conventional interpretation of a 20% held-out
    # partition while ensuring that the requested fraction is not reduced.
    n_validation = int(
        np.ceil(n_events * validation_fraction)
    )
    n_fit = n_events - n_validation

    # R² requires at least two observations. We enforce that minimum in
    # both partitions before attempting to train any probe.
    if n_fit < 2 or n_validation < 2:
        raise ProbePartitionError(
            "inner_partition_too_small",
            "Both probe_fit and probe_inner_validation require at "
            f"least two events, got {n_fit} and {n_validation}.",
        )

    # RandomState fixes the MT19937 permutation algorithm. The seed and
    # algorithm together make partition membership reproducible.
    random_state = np.random.RandomState(seed)
    permutation = random_state.permutation(n_events)

    validation_indices = np.sort(
        permutation[:n_validation]
    ).astype(np.int64, copy=False)

    fit_indices = np.sort(
        permutation[n_validation:]
    ).astype(np.int64, copy=False)

    manifest = hashlib.sha256()
    manifest.update(b"probe_fit\\0")
    manifest.update(
        np.asarray(
            fit_indices,
            dtype="<i8",
        ).tobytes()
    )
    manifest.update(b"probe_inner_validation\\0")
    manifest.update(
        np.asarray(
            validation_indices,
            dtype="<i8",
        ).tobytes()
    )

    # The dataclass is frozen, but NumPy arrays are mutable separately.
    # Marking them read-only prevents accidental partition changes.
    fit_indices.setflags(write=False)
    validation_indices.setflags(write=False)

    return ProbeInnerPartition(
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        seed=seed,
        validation_fraction=float(validation_fraction),
        manifest_hash=manifest.hexdigest(),
    )

