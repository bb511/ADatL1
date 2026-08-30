"""Data and run provenance for leakage-probe comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from .constants import LEAKAGE_PROBE_EVALUATION_MODES
from .errors import ProbeExtractionError
from .types import (
    LeakageProbeRunMetadata,
    ProbeEvaluationContext,
    ProbeRepresentationSet,
    ProbeSplitProvenance,
)


def leakage_probe_configuration_id(
    algorithm_config: Any,
) -> str:
    """Hash the resolved algorithm config, excluding the AE run seed."""

    if OmegaConf.is_config(algorithm_config):
        payload = OmegaConf.to_container(
            algorithm_config,
            resolve=True,
        )
    elif isinstance(algorithm_config, Mapping):
        payload = dict(algorithm_config)
    else:
        raise TypeError(
            "algorithm_config must be an OmegaConf config or mapping."
        )

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def make_leakage_probe_run_metadata(
    *,
    autoencoder_seed: int | None,
    algorithm_config: Any | None,
) -> LeakageProbeRunMetadata:
    """Construct run identity for persistence and aggregation."""

    configuration_id = (
        leakage_probe_configuration_id(algorithm_config)
        if algorithm_config is not None
        else None
    )
    return LeakageProbeRunMetadata(
        autoencoder_seed=(
            int(autoencoder_seed)
            if autoencoder_seed is not None
            else None
        ),
        configuration_id=configuration_id,
    )


def probe_split_provenance(
    representations: ProbeRepresentationSet,
) -> ProbeSplitProvenance:
    """Discard arrays while retaining comparable split identity."""

    return ProbeSplitProvenance(
        split=representations.split,
        source_splits=representations.source_splits,
        n_events=int(representations.n_events),
        sample_seed=int(representations.sample_seed),
        max_samples=representations.max_samples,
        event_manifest_hash=representations.manifest_hash,
        data_cache_id=representations.data_cache_id,
        data_cache_path=representations.data_cache_path,
    )


def make_probe_evaluation_context(
    development: ProbeRepresentationSet,
    held_out: ProbeRepresentationSet,
    *,
    mode: str,
) -> ProbeEvaluationContext:
    """Validate and record one validation or final-test split design."""

    if mode not in LEAKAGE_PROBE_EVALUATION_MODES:
        raise ProbeExtractionError(
            "invalid_probe_evaluation_mode",
            "Unknown leakage-probe evaluation mode "
            f"{mode!r}; expected one of "
            f"{list(LEAKAGE_PROBE_EVALUATION_MODES)}.",
        )

    expected = {
        "validation": (("train",), ("valid",)),
        "final_test": (("train", "valid"), ("test",)),
    }
    expected_development, expected_held_out = expected[mode]

    if development.source_splits != expected_development:
        raise ProbeExtractionError(
            "invalid_probe_development_splits",
            f"Mode {mode!r} requires development splits "
            f"{expected_development}, received "
            f"{development.source_splits}.",
        )

    if held_out.source_splits != expected_held_out:
        raise ProbeExtractionError(
            "invalid_probe_held_out_split",
            f"Mode {mode!r} requires held-out split "
            f"{expected_held_out}, received {held_out.source_splits}.",
        )

    if development.data_cache_id != held_out.data_cache_id:
        raise ProbeExtractionError(
            "probe_data_cache_mismatch",
            "Probe development and held-out data came from different "
            "cache identities.",
        )

    return ProbeEvaluationContext(
        mode=mode,
        development_data=probe_split_provenance(development),
        held_out_data=probe_split_provenance(held_out),
    )


def concatenate_probe_representation_sets(
    representations: Sequence[ProbeRepresentationSet],
    *,
    split: str,
) -> ProbeRepresentationSet:
    """Concatenate disjoint probe splits and preserve their identities."""

    parts = tuple(representations)
    if len(parts) < 2:
        raise ProbeExtractionError(
            "probe_development_parts_missing",
            "At least two representation sets are required for concatenation.",
        )

    cache_ids = {part.data_cache_id for part in parts}
    cache_paths = {part.data_cache_path for part in parts}
    sample_seeds = {part.sample_seed for part in parts}
    max_samples = {part.max_samples for part in parts}

    if len(cache_ids) != 1 or len(cache_paths) != 1:
        raise ProbeExtractionError(
            "probe_data_cache_mismatch",
            "Cannot combine probe splits from different data caches.",
        )

    if len(sample_seeds) != 1 or len(max_samples) != 1:
        raise ProbeExtractionError(
            "probe_sampling_protocol_mismatch",
            "Cannot combine probe splits with different sampling protocols.",
        )

    array_names = (
        "latent_logits",
        "latent_sample",
        "reconstructed_data",
        "sensitive_target",
    )
    for array_name in array_names:
        trailing_shapes = {
            getattr(part, array_name).shape[1:]
            for part in parts
        }
        if len(trailing_shapes) != 1:
            raise ProbeExtractionError(
                "probe_representation_shape_mismatch",
                f"Cannot combine {array_name} arrays with trailing "
                f"shapes {sorted(trailing_shapes)}.",
            )

    source_splits = tuple(
        source_split
        for part in parts
        for source_split in part.source_splits
    )
    if len(set(source_splits)) != len(source_splits):
        raise ProbeExtractionError(
            "probe_development_splits_overlap",
            "Cannot combine duplicate probe source splits: "
            f"{source_splits}.",
        )
    manifest = hashlib.sha256()
    for part in parts:
        manifest.update(part.split.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(part.manifest_hash.encode("ascii"))
        manifest.update(
            np.asarray([part.n_events], dtype="<i8").tobytes()
        )

    return ProbeRepresentationSet(
        split=split,
        latent_logits=np.concatenate(
            [part.latent_logits for part in parts],
            axis=0,
        ),
        latent_sample=np.concatenate(
            [part.latent_sample for part in parts],
            axis=0,
        ),
        reconstructed_data=np.concatenate(
            [part.reconstructed_data for part in parts],
            axis=0,
        ),
        sensitive_target=np.concatenate(
            [part.sensitive_target for part in parts],
            axis=0,
        ),
        n_events=sum(part.n_events for part in parts),
        sample_seed=parts[0].sample_seed,
        max_samples=parts[0].max_samples,
        manifest_hash=manifest.hexdigest(),
        data_cache_id=parts[0].data_cache_id,
        data_cache_path=parts[0].data_cache_path,
        source_splits=source_splits,
    )
