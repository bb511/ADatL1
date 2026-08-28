from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import src.evaluation.leakage_probe as leakage_probe
from src.evaluation.leakage_probe import (
    PROBE_TARGET_SHUFFLE_SEED,
    ProbeFitError,
    ProbeRepresentationSet,
    ShuffledTargetMLPResult,
    evaluate_shuffled_target_mlp_controls,
    make_shuffled_training_target,
)


def make_representation_set(
    split: str,
    n_events: int = 20,
) -> ProbeRepresentationSet:
    target = np.linspace(
        40.0,
        180.0,
        n_events,
        dtype=np.float64,
    )

    latent_logits = np.column_stack(
        [
            np.linspace(-1.0, 1.0, n_events),
            np.linspace(1.0, -1.0, n_events),
        ]
    )

    reconstructed_data = np.column_stack(
        [
            np.arange(n_events, dtype=np.float64),
            np.sin(
                np.arange(
                    n_events,
                    dtype=np.float64,
                )
            ),
            np.ones(n_events, dtype=np.float64),
        ]
    )

    return ProbeRepresentationSet(
        split=split,
        latent_logits=latent_logits,
        latent_sample=(
            latent_logits >= 0.0
        ).astype(np.float64),
        reconstructed_data=reconstructed_data,
        sensitive_target=target,
        n_events=n_events,
        sample_seed=12345,
        max_samples=None,
        manifest_hash=f"{split}-manifest",
    )


def test_training_target_shuffle_is_deterministic() -> None:
    target = np.linspace(
        20.0,
        200.0,
        20,
        dtype=np.float64,
    )
    original_target = target.copy()

    first = make_shuffled_training_target(target)
    second = make_shuffled_training_target(target)

    expected_permutation = np.random.RandomState(
        PROBE_TARGET_SHUFFLE_SEED
    ).permutation(target.shape[0])

    assert first.seed == PROBE_TARGET_SHUFFLE_SEED

    np.testing.assert_array_equal(
        first.permutation_indices,
        expected_permutation,
    )
    np.testing.assert_array_equal(
        first.values,
        target[expected_permutation],
    )

    np.testing.assert_array_equal(
        second.permutation_indices,
        first.permutation_indices,
    )
    np.testing.assert_array_equal(
        second.values,
        first.values,
    )

    assert (
        second.manifest_hash
        == first.manifest_hash
    )

    # The original target must not be modified.
    np.testing.assert_array_equal(
        target,
        original_target,
    )

    # The returned protocol arrays must not be mutable.
    assert first.values.flags.writeable is False
    assert (
        first.permutation_indices.flags.writeable
        is False
    )


def test_shuffled_target_is_not_original_alignment() -> None:
    target = np.arange(
        30,
        dtype=np.float64,
    )

    shuffled = make_shuffled_training_target(
        target
    )

    assert not np.array_equal(
        shuffled.values,
        target,
    )
    assert sorted(
        shuffled.values.tolist()
    ) == sorted(
        target.tolist()
    )


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        (
            np.ones(10, dtype=np.float64),
            "constant_shuffled_control_target",
        ),
        (
            np.asarray(
                [1.0, 2.0, np.nan, 4.0],
                dtype=np.float64,
            ),
            "non_finite_shuffled_control_target",
        ),
        (
            np.ones(
                (4, 2),
                dtype=np.float64,
            ),
            "invalid_shuffled_control_target_shape",
        ),
    ],
)
def test_invalid_shuffle_target_is_rejected(
    target: np.ndarray,
    reason: str,
) -> None:
    with pytest.raises(ProbeFitError) as error:
        make_shuffled_training_target(
            target
        )

    assert error.value.reason == reason


def test_control_permutates_only_training_target(
    monkeypatch,
) -> None:
    train = make_representation_set(
        "train",
        20,
    )
    validation = make_representation_set(
        "valid",
        10,
    )

    original_train_target = (
        train.sensitive_target.copy()
    )
    original_validation_target = (
        validation.sensitive_target.copy()
    )

    latent_result = object()
    reconstruction_result = object()
    inner_partition = object()

    primary_result = SimpleNamespace(
        latent_logits=latent_result,
        reconstructed_data=reconstruction_result,
        inner_partition=inner_partition,
    )

    primary_evaluator = Mock(
        return_value=primary_result
    )

    monkeypatch.setattr(
        leakage_probe,
        "evaluate_primary_mlp_probes",
        primary_evaluator,
    )

    result = evaluate_shuffled_target_mlp_controls(
        train,
        validation,
    )

    assert isinstance(
        result,
        ShuffledTargetMLPResult,
    )
    assert result.latent_logits is latent_result
    assert (
        result.reconstructed_data
        is reconstruction_result
    )
    assert result.inner_partition is inner_partition
    assert (
        result.shuffle_seed
        == PROBE_TARGET_SHUFFLE_SEED
    )

    primary_evaluator.assert_called_once()
    shuffled_train, received_validation = (
        primary_evaluator.call_args.args
    )

    assert shuffled_train is not train
    assert received_validation is validation

    # Representations and split metadata are reused unchanged.
    assert (
        shuffled_train.latent_logits
        is train.latent_logits
    )
    assert (
        shuffled_train.latent_sample
        is train.latent_sample
    )
    assert (
        shuffled_train.reconstructed_data
        is train.reconstructed_data
    )
    assert shuffled_train.split == "train"
    assert (
        shuffled_train.manifest_hash
        == train.manifest_hash
    )

    expected_permutation = np.random.RandomState(
        PROBE_TARGET_SHUFFLE_SEED
    ).permutation(train.n_events)

    np.testing.assert_array_equal(
        shuffled_train.sensitive_target,
        original_train_target[
            expected_permutation
        ],
    )

    # Neither original target may be modified.
    np.testing.assert_array_equal(
        train.sensitive_target,
        original_train_target,
    )
    np.testing.assert_array_equal(
        validation.sensitive_target,
        original_validation_target,
    )

    # Most importantly, outer validation stays unshuffled.
    assert (
        received_validation.sensitive_target
        is validation.sensitive_target
    )


def test_both_controls_reuse_one_shuffled_target(
    monkeypatch,
) -> None:
    train = make_representation_set(
        "train",
        20,
    )
    validation = make_representation_set(
        "valid",
        10,
    )

    captured_target = None

    def fake_primary_evaluation(
        shuffled_train,
        validation_representations,
    ):
        nonlocal captured_target
        captured_target = (
            shuffled_train.sensitive_target
        )

        return SimpleNamespace(
            latent_logits=Mock(),
            reconstructed_data=Mock(),
            inner_partition=Mock(),
        )

    monkeypatch.setattr(
        leakage_probe,
        "evaluate_primary_mlp_probes",
        fake_primary_evaluation,
    )

    evaluate_shuffled_target_mlp_controls(
        train,
        validation,
    )

    assert captured_target is not None

    # One ProbeRepresentationSet is supplied to the paired evaluator,
    # so both representation controls use this exact target object.
    assert captured_target.flags.writeable is False


@pytest.mark.parametrize(
    ("train_split", "validation_split", "reason"),
    [
        (
            "valid",
            "valid",
            "invalid_shuffled_control_training_split",
        ),
        (
            "train",
            "test",
            "invalid_shuffled_control_outer_split",
        ),
    ],
)
def test_shuffled_control_enforces_split_protocol(
    train_split: str,
    validation_split: str,
    reason: str,
) -> None:
    train = make_representation_set(
        train_split,
        20,
    )
    validation = make_representation_set(
        validation_split,
        10,
    )

    with pytest.raises(ProbeFitError) as error:
        evaluate_shuffled_target_mlp_controls(
            train,
            validation,
        )

    assert error.value.reason == reason