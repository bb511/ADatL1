"""Negative controls and secondary leakage diagnostics."""

from dataclasses import replace
import hashlib

import numpy as np

from .constants import PROBE_TARGET_SHUFFLE_SEED
from .errors import ProbeFitError
from .mlp import evaluate_primary_mlp_probes
from .types import (
    ProbeRepresentationSet,
    ShuffledTargetMLPResult,
    ShuffledTrainingTarget,
)

def make_shuffled_training_target(
    target: np.ndarray,
) -> ShuffledTrainingTarget:
    """Create the frozen shuffled-training-target control."""

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if (
        target.ndim == 2
        and target.shape[1] == 1
    ):
        target = target.reshape(-1)
    elif target.ndim != 1:
        raise ProbeFitError(
            "invalid_shuffled_control_target_shape",
            "The shuffled-control target must have shape "
            f"[events], got {target.shape}.",
        )

    if target.shape[0] < 2:
        raise ProbeFitError(
            "shuffled_control_target_too_small",
            "The shuffled-control target requires at least "
            "two events.",
        )

    if not np.isfinite(target).all():
        raise ProbeFitError(
            "non_finite_shuffled_control_target",
            "The shuffled-control target contains NaN or "
            "infinity.",
        )

    if np.unique(target).size < 2:
        raise ProbeFitError(
            "constant_shuffled_control_target",
            "The shuffled-control target is constant.",
        )

    random_state = np.random.RandomState(
        PROBE_TARGET_SHUFFLE_SEED
    )
    permutation_indices = random_state.permutation(
        target.shape[0]
    ).astype(
        np.int64,
        copy=False,
    )

    shuffled_values = target[
        permutation_indices
    ].copy()

    if np.array_equal(
        shuffled_values,
        target,
    ):
        raise ProbeFitError(
            "shuffled_target_unchanged",
            "The deterministic permutation did not change "
            "the training-target alignment.",
        )

    manifest = hashlib.sha256()
    manifest.update(
        b"shuffled_training_target\\0"
    )
    manifest.update(
        np.asarray(
            [PROBE_TARGET_SHUFFLE_SEED],
            dtype="<i8",
        ).tobytes()
    )
    manifest.update(
        np.asarray(
            permutation_indices,
            dtype="<i8",
        ).tobytes()
    )

    permutation_indices.setflags(
        write=False
    )
    shuffled_values.setflags(
        write=False
    )

    return ShuffledTrainingTarget(
        values=shuffled_values,
        permutation_indices=permutation_indices,
        seed=PROBE_TARGET_SHUFFLE_SEED,
        manifest_hash=manifest.hexdigest(),
    )


def evaluate_shuffled_target_mlp_controls(
    train_representations: ProbeRepresentationSet,
    validation_representations: ProbeRepresentationSet,
) -> ShuffledTargetMLPResult:
    """Repeat both MLP procedures with one shuffled train target."""

    if train_representations.split != "train":
        raise ProbeFitError(
            "invalid_shuffled_control_training_split",
            "Shuffled-target controls require the AE train "
            f"split, got "
            f"{train_representations.split!r}.",
        )

    if validation_representations.split != "valid":
        raise ProbeFitError(
            "invalid_shuffled_control_outer_split",
            "Shuffled-target controls require the held-out "
            f"AE valid split, got "
            f"{validation_representations.split!r}.",
        )

    if (
        train_representations.n_events
        != train_representations
        .sensitive_target.shape[0]
    ):
        raise ProbeFitError(
            "shuffled_control_train_event_count_mismatch",
            "Recorded training event count does not match "
            "the training target.",
        )

    shuffled_target = make_shuffled_training_target(
        train_representations.sensitive_target
    )

    shuffled_train_representations = replace(
        train_representations,
        sensitive_target=shuffled_target.values,
    )

    # This invokes the complete frozen procedure for both primary
    # representations: inner split, three candidates, seed selection,
    # full-training refit, and unchanged outer validation.
    control_result = evaluate_primary_mlp_probes(
        shuffled_train_representations,
        validation_representations,
    )

    return ShuffledTargetMLPResult(
        latent_logits=control_result.latent_logits,
        reconstructed_data=(
            control_result.reconstructed_data
        ),
        inner_partition=(
            control_result.inner_partition
        ),
        shuffle_seed=shuffled_target.seed,
        permutation_manifest_hash=(
            shuffled_target.manifest_hash
        ),
    )

