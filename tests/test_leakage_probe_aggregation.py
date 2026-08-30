import json
from pathlib import Path

import pytest

from src.evaluation.leakage_probe import (
    LEAKAGE_PROBE_PROTOCOL_VERSION,
    ProbeAggregationError,
    aggregate_paired_seed_leakage,
    write_paired_seed_leakage_aggregate,
)


def write_artifact(
    path: Path,
    *,
    seed: int,
    leakage: float | None,
    configuration_id: str = "configuration-a",
    valid_manifest: str = "valid-manifest",
) -> Path:
    valid = leakage is not None
    payload = {
        "leakage_probe_protocol_version": (
            LEAKAGE_PROBE_PROTOCOL_VERSION
        ),
        "probe_valid": valid,
        "rejection_reason": (
            None if valid else "synthetic_failure"
        ),
        "rejection_message": (
            None if valid else "Synthetic failure."
        ),
        "leakage_worst": leakage,
        "run": {
            "autoencoder_seed": seed,
            "configuration_id": configuration_id,
        },
        "evaluation": {
            "mode": "validation",
            "development_data": (
                {
                    "source_splits": ["train"],
                    "n_events": 100,
                    "sample_seed": 12345,
                    "max_samples": None,
                    "event_manifest_hash": "train-manifest",
                    "data_cache_id": "cache-id",
                }
                if valid
                else None
            ),
            "held_out_data": (
                {
                    "source_splits": ["valid"],
                    "n_events": 40,
                    "sample_seed": 12345,
                    "max_samples": None,
                    "event_manifest_hash": valid_manifest,
                    "data_cache_id": "cache-id",
                }
                if valid
                else None
            ),
        },
    }
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_valid_paired_seeds_report_mean_and_uncertainty(
    tmp_path,
) -> None:
    paths = [
        write_artifact(
            tmp_path / f"seed-{seed}.json",
            seed=seed,
            leakage=leakage,
        )
        for seed, leakage in (
            (10, 0.1),
            (20, 0.2),
            (30, 0.3),
        )
    ]

    aggregate = aggregate_paired_seed_leakage(
        paths,
        expected_autoencoder_seeds=(10, 20, 30),
    )

    assert aggregate["configuration_valid"] is True
    assert aggregate["rejection_reasons"] == []
    assert aggregate["invalid_run_policy"] == (
        "reject_configuration"
    )
    assert aggregate["leakage_worst"]["mean"] == pytest.approx(
        0.2
    )
    assert aggregate["leakage_worst"][
        "sample_std"
    ] == pytest.approx(0.1)
    assert aggregate["leakage_worst"][
        "standard_error"
    ] == pytest.approx(0.1 / (3**0.5))


def test_one_invalid_seed_rejects_configuration_without_fake_score(
    tmp_path,
) -> None:
    paths = [
        write_artifact(
            tmp_path / "seed-10.json",
            seed=10,
            leakage=0.1,
        ),
        write_artifact(
            tmp_path / "seed-20.json",
            seed=20,
            leakage=None,
        ),
    ]

    aggregate = aggregate_paired_seed_leakage(
        paths,
        expected_autoencoder_seeds=(10, 20),
    )

    assert aggregate["configuration_valid"] is False
    assert aggregate["leakage_worst"] is None
    assert aggregate["rejection_reasons"] == [
        "invalid_autoencoder_seeds:[20]"
    ]
    assert aggregate["runs"][1]["leakage_worst"] is None


def test_missing_seed_rejects_configuration(tmp_path) -> None:
    path = write_artifact(
        tmp_path / "seed-10.json",
        seed=10,
        leakage=0.1,
    )

    aggregate = aggregate_paired_seed_leakage(
        [path],
        expected_autoencoder_seeds=(10, 20),
    )

    assert aggregate["configuration_valid"] is False
    assert aggregate["leakage_worst"] is None
    assert aggregate["rejection_reasons"] == [
        "missing_autoencoder_seeds:[20]"
    ]


def test_different_event_manifests_cannot_be_aggregated(
    tmp_path,
) -> None:
    first = write_artifact(
        tmp_path / "seed-10.json",
        seed=10,
        leakage=0.1,
    )
    second = write_artifact(
        tmp_path / "seed-20.json",
        seed=20,
        leakage=0.2,
        valid_manifest="different-valid-manifest",
    )

    with pytest.raises(
        ProbeAggregationError,
        match="different event manifests",
    ):
        aggregate_paired_seed_leakage(
            [first, second],
            expected_autoencoder_seeds=(10, 20),
        )


def test_different_configurations_cannot_be_aggregated(
    tmp_path,
) -> None:
    first = write_artifact(
        tmp_path / "seed-10.json",
        seed=10,
        leakage=0.1,
    )
    second = write_artifact(
        tmp_path / "seed-20.json",
        seed=20,
        leakage=0.2,
        configuration_id="configuration-b",
    )

    with pytest.raises(
        ProbeAggregationError,
        match="different autoencoder configurations",
    ):
        aggregate_paired_seed_leakage(
            [first, second],
            expected_autoencoder_seeds=(10, 20),
        )


def test_aggregate_is_written_as_json(tmp_path) -> None:
    paths = [
        write_artifact(
            tmp_path / f"seed-{seed}.json",
            seed=seed,
            leakage=leakage,
        )
        for seed, leakage in ((10, 0.1), (20, 0.2))
    ]
    output_path = tmp_path / "aggregate" / "leakage.json"

    result_path = write_paired_seed_leakage_aggregate(
        paths,
        expected_autoencoder_seeds=(10, 20),
        output_path=output_path,
    )

    assert result_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["configuration_valid"] is True
