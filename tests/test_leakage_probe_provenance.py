from dataclasses import replace

import numpy as np
import pytest

from src.evaluation.leakage_probe import (
    ProbeExtractionError,
    ProbeRepresentationSet,
    concatenate_probe_representation_sets,
    leakage_probe_configuration_id,
    make_probe_evaluation_context,
)


def make_representations(
    split: str,
    n_events: int,
    *,
    cache_id: str = "cache-id",
) -> ProbeRepresentationSet:
    latent = np.arange(
        n_events * 2,
        dtype=np.float64,
    ).reshape(n_events, 2)
    reconstruction = np.arange(
        n_events * 3,
        dtype=np.float64,
    ).reshape(n_events, 3)
    return ProbeRepresentationSet(
        split=split,
        latent_logits=latent,
        latent_sample=(latent > 0).astype(np.float64),
        reconstructed_data=reconstruction,
        sensitive_target=np.linspace(1.0, 2.0, n_events),
        n_events=n_events,
        sample_seed=12345,
        max_samples=None,
        manifest_hash=f"{split}-manifest",
        data_cache_id=cache_id,
        data_cache_path="/data/cache",
        source_splits=(split,),
    )


def test_configuration_id_is_order_independent_and_sensitive() -> None:
    first = leakage_probe_configuration_id(
        {
            "mi_gamma": 0.1,
            "encoder": {"nodes": [64, 32, 8]},
        }
    )
    reordered = leakage_probe_configuration_id(
        {
            "encoder": {"nodes": [64, 32, 8]},
            "mi_gamma": 0.1,
        }
    )
    changed = leakage_probe_configuration_id(
        {
            "mi_gamma": 0.2,
            "encoder": {"nodes": [64, 32, 8]},
        }
    )

    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_train_and_valid_are_combined_for_final_development() -> None:
    train = make_representations("train", 4)
    valid = make_representations("valid", 3)

    combined = concatenate_probe_representation_sets(
        (train, valid),
        split="train+valid",
    )

    assert combined.split == "train+valid"
    assert combined.source_splits == ("train", "valid")
    assert combined.n_events == 7
    assert combined.data_cache_id == "cache-id"
    np.testing.assert_array_equal(
        combined.latent_logits,
        np.concatenate(
            [train.latent_logits, valid.latent_logits],
            axis=0,
        ),
    )
    assert len(combined.manifest_hash) == 64

    context = make_probe_evaluation_context(
        combined,
        make_representations("test", 2),
        mode="final_test",
    )
    assert context.development_data.source_splits == (
        "train",
        "valid",
    )
    assert context.held_out_data.source_splits == ("test",)


def test_combining_different_data_caches_is_rejected() -> None:
    train = make_representations("train", 4)
    valid = replace(
        make_representations("valid", 3),
        data_cache_id="different-cache",
    )

    with pytest.raises(ProbeExtractionError) as error:
        concatenate_probe_representation_sets(
            (train, valid),
            split="train+valid",
        )

    assert error.value.reason == "probe_data_cache_mismatch"


def test_validation_mode_rejects_test_as_held_out_data() -> None:
    with pytest.raises(ProbeExtractionError) as error:
        make_probe_evaluation_context(
            make_representations("train", 4),
            make_representations("test", 2),
            mode="validation",
        )

    assert error.value.reason == "invalid_probe_held_out_split"
