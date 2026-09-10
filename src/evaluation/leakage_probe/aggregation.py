"""Aggregate leakage across a predeclared set of paired AE seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .constants import (
    LEAKAGE_PROBE_INVALID_RUN_POLICY,
    LEAKAGE_PROBE_PROTOCOL_VERSION,
)


class ProbeAggregationError(ValueError):
    """Artifacts cannot be compared under one aggregation protocol."""


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeAggregationError(
            f"Could not read leakage artifact {path}: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ProbeAggregationError(
            f"Leakage artifact {path} must contain a JSON object."
        )
    return payload


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    path: Path,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProbeAggregationError(
            f"Leakage artifact {path} has no {key!r} object."
        )
    return value


def _comparable_provenance(
    evaluation: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    comparable: dict[str, Any] = {}
    for role in ("development_data", "held_out_data"):
        provenance = evaluation.get(role)
        if not isinstance(provenance, dict):
            raise ProbeAggregationError(
                f"Valid leakage artifact {path} has no {role!r} "
                "provenance."
            )
        comparable[role] = {
            "source_splits": provenance.get("source_splits"),
            "n_events": provenance.get("n_events"),
            "sample_seed": provenance.get("sample_seed"),
            "max_samples": provenance.get("max_samples"),
            "event_manifest_hash": provenance.get(
                "event_manifest_hash"
            ),
            "data_cache_id": provenance.get("data_cache_id"),
        }
    return comparable


def aggregate_paired_seed_leakage(
    artifact_paths: Sequence[str | Path],
    *,
    expected_autoencoder_seeds: Sequence[int],
) -> dict[str, Any]:
    """Aggregate one configuration or reject it if any seed is invalid.

    The only supported invalid-run policy is ``reject_configuration``:
    missing or invalid paired seeds remain visible, and no aggregate leakage
    value is emitted for the configuration.
    """

    expected_seeds = tuple(
        int(seed) for seed in expected_autoencoder_seeds
    )
    if len(expected_seeds) < 2:
        raise ProbeAggregationError(
            "Paired-seed aggregation requires at least two expected seeds."
        )
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ProbeAggregationError(
            "Expected autoencoder seeds must be unique."
        )

    records_by_seed: dict[int, dict[str, Any]] = {}
    protocol_versions: set[str] = set()
    configuration_ids: set[str] = set()
    evaluation_modes: set[str] = set()
    valid_provenance: dict[str, Any] | None = None

    for artifact_path in artifact_paths:
        path = Path(artifact_path)
        payload = _read_artifact(path)
        run = _required_mapping(payload, "run", path=path)
        evaluation = _required_mapping(
            payload,
            "evaluation",
            path=path,
        )

        seed = run.get("autoencoder_seed")
        configuration_id = run.get("configuration_id")
        protocol_version = payload.get(
            "leakage_probe_protocol_version"
        )
        evaluation_mode = evaluation.get("mode")
        evaluation_purpose = evaluation.get(
            "purpose",
            "scientific",
        )
        reporting_eligible = evaluation.get(
            "reporting_eligible",
            True,
        )

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ProbeAggregationError(
                f"Leakage artifact {path} has no integer "
                "autoencoder seed."
            )
        if not isinstance(configuration_id, str) or not configuration_id:
            raise ProbeAggregationError(
                f"Leakage artifact {path} has no configuration identity."
            )
        if not isinstance(protocol_version, str):
            raise ProbeAggregationError(
                f"Leakage artifact {path} has no protocol version."
            )
        if not isinstance(evaluation_mode, str):
            raise ProbeAggregationError(
                f"Leakage artifact {path} has no evaluation mode."
            )
        if (
            evaluation_purpose != "scientific"
            or reporting_eligible is not True
        ):
            raise ProbeAggregationError(
                f"Leakage artifact {path} is a non-reportable "
                "smoke-test artifact and cannot be aggregated."
            )
        if seed in records_by_seed:
            raise ProbeAggregationError(
                f"Autoencoder seed {seed} appears more than once."
            )

        probe_valid = payload.get("probe_valid") is True
        leakage_worst = payload.get("leakage_worst")
        if probe_valid:
            if (
                not isinstance(leakage_worst, (int, float))
                or isinstance(leakage_worst, bool)
                or not math.isfinite(float(leakage_worst))
            ):
                raise ProbeAggregationError(
                    f"Valid leakage artifact {path} has no finite "
                    "leakage_worst."
                )
            provenance = _comparable_provenance(
                evaluation,
                path=path,
            )
            if valid_provenance is None:
                valid_provenance = provenance
            elif provenance != valid_provenance:
                raise ProbeAggregationError(
                    "Valid paired seeds use different event manifests, "
                    "cache identities, or sampling protocols."
                )

        records_by_seed[seed] = {
            "autoencoder_seed": seed,
            "artifact_path": str(path),
            "probe_valid": probe_valid,
            "leakage_worst": (
                float(leakage_worst)
                if probe_valid
                else None
            ),
            "rejection_reason": payload.get(
                "rejection_reason"
            ),
            "rejection_message": payload.get(
                "rejection_message"
            ),
        }
        protocol_versions.add(protocol_version)
        configuration_ids.add(configuration_id)
        evaluation_modes.add(evaluation_mode)

    if protocol_versions != {LEAKAGE_PROBE_PROTOCOL_VERSION}:
        raise ProbeAggregationError(
            "All artifacts must use current protocol "
            f"{LEAKAGE_PROBE_PROTOCOL_VERSION!r}; received "
            f"{sorted(protocol_versions)}."
        )
    if len(configuration_ids) != 1:
        raise ProbeAggregationError(
            "Artifacts from different autoencoder configurations "
            "cannot be aggregated."
        )
    if len(evaluation_modes) != 1:
        raise ProbeAggregationError(
            "Validation and final-test artifacts cannot be aggregated "
            "together."
        )

    expected_set = set(expected_seeds)
    observed_set = set(records_by_seed)
    unexpected_seeds = sorted(observed_set - expected_set)
    if unexpected_seeds:
        raise ProbeAggregationError(
            "Artifacts contain undeclared autoencoder seeds: "
            f"{unexpected_seeds}."
        )

    missing_seeds = [
        seed for seed in expected_seeds if seed not in observed_set
    ]
    invalid_seeds = [
        seed
        for seed in expected_seeds
        if seed in records_by_seed
        and not records_by_seed[seed]["probe_valid"]
    ]
    rejection_reasons: list[str] = []
    if missing_seeds:
        rejection_reasons.append(
            f"missing_autoencoder_seeds:{missing_seeds}"
        )
    if invalid_seeds:
        rejection_reasons.append(
            f"invalid_autoencoder_seeds:{invalid_seeds}"
        )

    configuration_valid = not rejection_reasons
    leakage_summary: dict[str, float | int] | None = None
    if configuration_valid:
        values = np.asarray(
            [
                records_by_seed[seed]["leakage_worst"]
                for seed in expected_seeds
            ],
            dtype=np.float64,
        )
        mean = float(np.mean(values))
        sample_std = float(np.std(values, ddof=1))
        standard_error = sample_std / math.sqrt(values.size)
        ci_half_width = 1.96 * standard_error
        leakage_summary = {
            "n_seeds": int(values.size),
            "mean": mean,
            "sample_std": sample_std,
            "standard_error": standard_error,
            "ci95_low": mean - ci_half_width,
            "ci95_high": mean + ci_half_width,
        }

    return {
        "leakage_probe_protocol_version": (
            LEAKAGE_PROBE_PROTOCOL_VERSION
        ),
        "invalid_run_policy": LEAKAGE_PROBE_INVALID_RUN_POLICY,
        "configuration_id": next(iter(configuration_ids)),
        "evaluation_mode": next(iter(evaluation_modes)),
        "expected_autoencoder_seeds": list(expected_seeds),
        "configuration_valid": configuration_valid,
        "rejection_reasons": rejection_reasons,
        "runs": [
            records_by_seed[seed]
            for seed in expected_seeds
            if seed in records_by_seed
        ],
        "data_provenance": valid_provenance,
        "leakage_worst": leakage_summary,
    }


def write_paired_seed_leakage_aggregate(
    artifact_paths: Sequence[str | Path],
    *,
    expected_autoencoder_seeds: Sequence[int],
    output_path: str | Path,
) -> Path:
    """Aggregate paired seeds and write one machine-readable JSON file."""

    payload = aggregate_paired_seed_leakage(
        artifact_paths,
        expected_autoencoder_seeds=expected_autoencoder_seeds,
    )
    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return resolved_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate four-probe leakage across paired AE seeds."
        )
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        help="Per-run leakage_probes.json files.",
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        required=True,
        help="Predeclared paired autoencoder seeds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output leakage_probe_aggregate.json path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_paired_seed_leakage_aggregate(
        args.artifacts,
        expected_autoencoder_seeds=args.expected_seeds,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
