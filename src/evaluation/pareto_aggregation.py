"""Phase 2 collection of paired-seed Pareto-study metrics.

This module deliberately consumes an explicit study map.  Hydra output folders
contain the resolved study manifests whereas checkpoints and compact evaluation
artifacts live elsewhere, so discovering runs from directory names would be
ambiguous and brittle.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.evaluation.leakage_probe.aggregation import (
    ProbeAggregationError,
    aggregate_paired_seed_leakage,
)
from src.evaluation.leakage_probe.constants import (
    LEAKAGE_PROBE_PROTOCOL_VERSION,
)
from src.evaluation.leakage_probe.provenance import (
    leakage_probe_configuration_id,
)


PARETO_METRICS_SCHEMA_VERSION = 1
STUDY_MAP_SCHEMA_VERSION = 1

_ARTIFACT_PATHS = {
    "leakage": Path("plots/val/loss_total/probes/leakage_probes.json"),
    "efficiency": Path("plots/val/loss_total/eff/eff_summary.json"),
    "correlation": Path(
        "plots/val/loss_total/correlation_matrix/normal/mean_correlations.json"
    ),
    "collapse": Path("plots/val/loss_total/latent_collapse/collapse_summary.json"),
    "auroc": Path("plots/val/loss_total/auroc/auroc_summary.json"),
}

_SUMMARY_METRICS = (
    "leakage_worst",
    "residual_correlation",
    "mean_pearson_correlation",
    "mean_spearman_correlation",
    "median_efficiency",
    "min_efficiency",
    "mean_efficiency",
    "cvar25_efficiency",
    "joint_code_entropy_bits",
    "summed_marginal_bit_entropy_bits",
    "effective_code_count",
    "observed_code_count",
    "median_auroc",
    "min_auroc",
    "median_partial_auroc",
    "min_partial_auroc",
)


class ParetoCollectionError(ValueError):
    """The study map, manifest, or a required artifact is invalid."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParetoCollectionError(f"{label} must be a mapping.")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ParetoCollectionError(f"{label}.{key} must be a non-empty string.")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ParetoCollectionError(f"{label}.{key} must be an integer.")
    return int(value)


def _finite_float(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ParetoCollectionError(f"{label} must be a finite number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ParetoCollectionError(f"{label} must be a finite number.")
    return converted


def _finite_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ParetoCollectionError(f"{label} must be an integer.")
    return int(value)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParetoCollectionError(f"Could not read {path}: {error}") from error
    return _require_mapping(payload, label=str(path))


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        config = OmegaConf.load(path)
        payload = OmegaConf.to_container(
            config,
            resolve=True,
            throw_on_missing=True,
        )
    except Exception as error:  # OmegaConf exposes several parse error types.
        raise ParetoCollectionError(
            f"Could not read resolved Pareto manifest {path}: {error}"
        ) from error
    return _require_mapping(payload, label=str(path))


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _expected_seeds(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ParetoCollectionError(f"{label} must be a sequence of seeds.")
    seeds = tuple(
        _finite_int(seed, label=f"{label}[{index}]")
        for index, seed in enumerate(value)
    )
    if len(seeds) < 2:
        raise ParetoCollectionError(f"{label} requires at least two seeds.")
    if len(set(seeds)) != len(seeds):
        raise ParetoCollectionError(f"{label} must contain unique seeds.")
    return seeds


def load_study_map(path: str | Path) -> tuple[Mapping[str, Any], Path]:
    """Load one YAML/JSON study map and return it with its directory."""

    resolved_path = Path(path)
    try:
        payload = OmegaConf.to_container(
            OmegaConf.load(resolved_path),
            resolve=True,
            throw_on_missing=True,
        )
    except Exception as error:
        raise ParetoCollectionError(
            f"Could not read Pareto study map {resolved_path}: {error}"
        ) from error
    return _require_mapping(payload, label=str(resolved_path)), resolved_path.parent


def _validate_study_map(
    study_map: Mapping[str, Any],
) -> tuple[str, str, tuple[int, ...], Sequence[Mapping[str, Any]]]:
    schema_version = _require_int(study_map, "schema_version", label="study map")
    if schema_version != STUDY_MAP_SCHEMA_VERSION:
        raise ParetoCollectionError(
            "Unsupported study-map schema version "
            f"{schema_version}; expected {STUDY_MAP_SCHEMA_VERSION}."
        )

    study_id = _require_string(study_map, "study_id", label="study map")
    protocol_version = _require_string(
        study_map,
        "protocol_version",
        label="study map",
    )
    expected_seeds = _expected_seeds(
        study_map.get("expected_autoencoder_seeds"),
        label="study map.expected_autoencoder_seeds",
    )
    runs = study_map.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or not runs:
        raise ParetoCollectionError("study map.runs must be a non-empty sequence.")
    return study_id, protocol_version, expected_seeds, [
        _require_mapping(run, label=f"study map.runs[{index}]")
        for index, run in enumerate(runs)
    ]


def _manifest_info(
    manifest: Mapping[str, Any],
    *,
    map_study_id: str,
    map_protocol_version: str,
    map_expected_seeds: tuple[int, ...],
    mapped_configuration_id: str,
    mapped_seed: int,
) -> dict[str, Any]:
    """Validate the frozen manifest and return the fields Phase 2 needs."""

    study = _require_mapping(manifest.get("pareto_study"), label="manifest.pareto_study")
    candidate = _require_mapping(study.get("candidate"), label="manifest candidate")
    algorithm = _require_mapping(manifest.get("algorithm"), label="manifest.algorithm")
    encoder = _require_mapping(algorithm.get("encoder"), label="manifest.algorithm.encoder")
    evaluation = _require_mapping(manifest.get("evaluation"), label="manifest.evaluation")
    probes = _require_mapping(
        evaluation.get("leakage_probes"),
        label="manifest.evaluation.leakage_probes",
    )

    if study.get("enabled") is not True:
        raise ParetoCollectionError("Manifest is not an enabled Pareto study.")
    if _require_string(study, "study_id", label="manifest.pareto_study") != map_study_id:
        raise ParetoCollectionError("Manifest study_id does not match the study map.")
    if (
        _require_string(study, "protocol_version", label="manifest.pareto_study")
        != map_protocol_version
    ):
        raise ParetoCollectionError(
            "Manifest protocol_version does not match the study map."
        )

    configuration_id = _require_string(
        study,
        "configuration_id",
        label="manifest.pareto_study",
    )
    if configuration_id != mapped_configuration_id:
        raise ParetoCollectionError(
            "Manifest configuration_id does not match the study-map entry."
        )

    seed = _require_int(candidate, "autoencoder_seed", label="manifest candidate")
    if seed != mapped_seed:
        raise ParetoCollectionError("Manifest seed does not match the study-map entry.")
    if manifest.get("seed") != seed:
        raise ParetoCollectionError("Manifest top-level seed does not match candidate seed.")
    if _expected_seeds(
        study.get("paired_autoencoder_seeds"),
        label="manifest.pareto_study.paired_autoencoder_seeds",
    ) != map_expected_seeds:
        raise ParetoCollectionError(
            "Manifest paired_autoencoder_seeds does not match the study map."
        )
    if manifest.get("test") is not False:
        raise ParetoCollectionError("Phase 2 accepts validation runs only (test=false).")
    if probes.get("enabled") is not True or probes.get("mode") != "validation":
        raise ParetoCollectionError(
            "Manifest must enable validation-mode leakage probes."
        )
    smoke = probes.get("smoke_test")
    if isinstance(smoke, Mapping) and smoke.get("enabled") is True:
        raise ParetoCollectionError("Manifest enables non-reportable smoke probes.")

    gamma = _finite_float(candidate.get("mi_gamma"), label="manifest candidate gamma")
    num_bins = _finite_int(
        candidate.get("mi_sensitive_num_bins"),
        label="manifest candidate bins",
    )
    architecture_id = _require_string(
        candidate,
        "architecture_id",
        label="manifest candidate",
    )
    nodes = candidate.get("encoder_nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        raise ParetoCollectionError("manifest candidate encoder_nodes must be a sequence.")
    encoder_nodes = tuple(
        _finite_int(node, label=f"manifest candidate encoder_nodes[{index}]")
        for index, node in enumerate(nodes)
    )
    if not encoder_nodes or encoder_nodes[-1] != 8:
        raise ParetoCollectionError("Manifest candidate latent width must remain fixed at 8.")
    if _finite_float(algorithm.get("mi_gamma"), label="manifest algorithm gamma") != gamma:
        raise ParetoCollectionError("Manifest algorithm gamma disagrees with candidate.")
    if (
        _finite_int(algorithm.get("mi_sensitive_num_bins"), label="manifest algorithm bins")
        != num_bins
    ):
        raise ParetoCollectionError("Manifest algorithm bins disagree with candidate.")
    algorithm_nodes = encoder.get("nodes")
    if not isinstance(algorithm_nodes, Sequence) or isinstance(
        algorithm_nodes, (str, bytes)
    ):
        raise ParetoCollectionError("Manifest algorithm encoder nodes must be a sequence.")
    if tuple(int(node) for node in algorithm_nodes) != encoder_nodes:
        raise ParetoCollectionError("Manifest algorithm encoder nodes disagree with candidate.")

    sensitive = _require_mapping(
        study.get("sensitive_variable"),
        label="manifest.pareto_study.sensitive_variable",
    )
    raw_target = _require_string(sensitive, "raw_target", label="sensitive variable")
    collapse = _require_mapping(
        study.get("collapse_constraint"),
        label="manifest.pareto_study.collapse_constraint",
    )
    collapse_seed_rule = _require_mapping(
        collapse.get("seed_level_rule"),
        label="manifest collapse seed rule",
    )
    efficiency_constraint = _require_mapping(
        study.get("minimum_efficiency_constraint"),
        label="manifest.pareto_study.minimum_efficiency_constraint",
    )

    return {
        "configuration_id": configuration_id,
        "autoencoder_seed": seed,
        "study_id": map_study_id,
        "protocol_version": map_protocol_version,
        "leakage_algorithm_hash": leakage_probe_configuration_id(algorithm),
        "raw_target": raw_target,
        "candidate": {
            "mi_gamma": gamma,
            "mi_sensitive_num_bins": num_bins,
            "architecture_id": architecture_id,
            "encoder_nodes": list(encoder_nodes),
        },
        "constraints": {
            "minimum_joint_code_entropy_bits": _finite_float(
                collapse_seed_rule.get("minimum_joint_code_entropy_bits"),
                label="manifest collapse minimum entropy",
            ),
            "minimum_fraction_of_paired_gamma_zero_joint_entropy": _finite_float(
                collapse_seed_rule.get(
                    "minimum_fraction_of_paired_gamma_zero_joint_entropy"
                ),
                label="manifest collapse baseline fraction",
            ),
            "max_relative_efficiency_degradation": _finite_float(
                efficiency_constraint.get("max_relative_degradation"),
                label="manifest efficiency degradation",
            ),
        },
    }


def _validate_leakage_artifact(
    payload: Mapping[str, Any],
    *,
    info: Mapping[str, Any],
    seed: int,
) -> tuple[bool, str | None]:
    if payload.get("leakage_probe_protocol_version") != LEAKAGE_PROBE_PROTOCOL_VERSION:
        raise ParetoCollectionError("Leakage artifact has an unsupported protocol version.")
    run = _require_mapping(payload.get("run"), label="leakage artifact.run")
    if _require_int(run, "autoencoder_seed", label="leakage artifact.run") != seed:
        raise ParetoCollectionError("Leakage artifact seed does not match the manifest.")
    if (
        _require_string(run, "configuration_id", label="leakage artifact.run")
        != info["leakage_algorithm_hash"]
    ):
        raise ParetoCollectionError(
            "Leakage artifact algorithm hash does not match the resolved manifest."
        )
    evaluation = _require_mapping(payload.get("evaluation"), label="leakage artifact.evaluation")
    if (
        evaluation.get("mode") != "validation"
        or evaluation.get("purpose") != "scientific"
        or evaluation.get("reporting_eligible") is not True
    ):
        raise ParetoCollectionError("Leakage artifact is not a reportable validation result.")
    valid = payload.get("probe_valid") is True
    reason = payload.get("rejection_reason")
    return valid, reason if isinstance(reason, str) else None


def _validate_efficiency_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("checkpoint") != "loss_total.ckpt" or payload.get("split") != "val":
        raise ParetoCollectionError("Efficiency artifact is not from val/loss_total.ckpt.")
    signal_efficiencies = _require_mapping(
        payload.get("signal_efficiencies"),
        label="efficiency signal_efficiencies",
    )
    if not signal_efficiencies:
        raise ParetoCollectionError("Efficiency artifact contains no signal efficiencies.")
    for dataset, value in signal_efficiencies.items():
        _finite_float(value, label=f"efficiency signal {dataset!r}")
    return {
        "median_efficiency": _finite_float(
            payload.get("median_efficiency"), label="median_efficiency"
        ),
        "min_efficiency": _finite_float(payload.get("min_efficiency"), label="min_efficiency"),
        "mean_efficiency": _finite_float(
            payload.get("mean_efficiency"), label="mean_efficiency"
        ),
        "cvar25_efficiency": _finite_float(
            payload.get("cvar25_efficiency"), label="cvar25_efficiency"
        ),
        "signal_datasets": sorted(str(dataset) for dataset in signal_efficiencies),
    }


def _validate_correlation_artifact(
    payload: Mapping[str, Any], *, raw_target: str
) -> dict[str, float]:
    if payload.get("sensitive_variable") != raw_target:
        raise ParetoCollectionError(
            "Correlation artifact sensitive variable does not match the manifest."
        )
    return {
        "residual_correlation": _finite_float(payload.get("C"), label="correlation C"),
        "mean_pearson_correlation": _finite_float(
            payload.get("mean_pearson_correlation"),
            label="mean_pearson_correlation",
        ),
        "mean_spearman_correlation": _finite_float(
            payload.get("mean_spearman_correlation"),
            label="mean_spearman_correlation",
        ),
    }


def _validate_collapse_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("checkpoint") != "loss_total.ckpt" or payload.get("split") != "val":
        raise ParetoCollectionError("Collapse artifact is not from val/loss_total.ckpt.")
    if payload.get("dataset") != "normal":
        raise ParetoCollectionError("Collapse artifact must evaluate the normal dataset.")
    representation = _require_mapping(payload.get("representation"), label="collapse representation")
    if representation.get("name") != "latent_sample":
        raise ParetoCollectionError("Collapse artifact must evaluate latent_sample.")
    metrics = _require_mapping(payload.get("metrics"), label="collapse metrics")
    decision = _require_mapping(payload.get("decision"), label="collapse decision")
    absolute_pass = decision.get("absolute_entropy_pass")
    if not isinstance(absolute_pass, bool):
        raise ParetoCollectionError("Collapse artifact has no boolean absolute pass decision.")
    return {
        "joint_code_entropy_bits": _finite_float(
            metrics.get("joint_code_entropy_bits"),
            label="joint_code_entropy_bits",
        ),
        "summed_marginal_bit_entropy_bits": _finite_float(
            metrics.get("summed_marginal_bit_entropy_bits"),
            label="summed_marginal_bit_entropy_bits",
        ),
        "effective_code_count": _finite_float(
            metrics.get("effective_code_count"), label="effective_code_count"
        ),
        "observed_code_count": _finite_int(
            metrics.get("observed_code_count"), label="observed_code_count"
        ),
        "absolute_entropy_pass": absolute_pass,
    }


def _validate_auroc_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("checkpoint") != "loss_total.ckpt" or payload.get("split") != "val":
        raise ParetoCollectionError("AUROC artifact is not from val/loss_total.ckpt.")
    if payload.get("score_direction") != "higher_score_is_more_anomalous":
        raise ParetoCollectionError("AUROC artifact has an unexpected score direction.")
    per_signal = _require_mapping(payload.get("per_signal"), label="AUROC per_signal")
    if not per_signal:
        raise ParetoCollectionError("AUROC artifact contains no signal metrics.")
    for dataset, values in per_signal.items():
        signal_values = _require_mapping(values, label=f"AUROC signal {dataset!r}")
        _finite_float(signal_values.get("auroc"), label=f"AUROC {dataset!r}")
        _finite_float(signal_values.get("partial_auroc"), label=f"partial AUROC {dataset!r}")
    summaries = _require_mapping(payload.get("summaries"), label="AUROC summaries")
    return {
        "median_auroc": _finite_float(summaries.get("median_auroc"), label="median_auroc"),
        "min_auroc": _finite_float(summaries.get("min_auroc"), label="min_auroc"),
        "median_partial_auroc": _finite_float(
            summaries.get("median_partial_auroc"), label="median_partial_auroc"
        ),
        "min_partial_auroc": _finite_float(
            summaries.get("min_partial_auroc"), label="min_partial_auroc"
        ),
        "signal_datasets": sorted(str(dataset) for dataset in per_signal),
    }


def _append_reason(record: dict[str, Any], reason: str) -> None:
    if reason not in record["rejection_reasons"]:
        record["rejection_reasons"].append(reason)


def _collect_run(
    entry: Mapping[str, Any],
    *,
    base_dir: Path,
    study_id: str,
    protocol_version: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Collect one mapped seed run without allowing a bad run to abort the study."""

    configuration_id = _require_string(entry, "configuration_id", label="study map run")
    seed = _require_int(entry, "autoencoder_seed", label="study map run")
    manifest_path = _resolve_path(
        _require_string(entry, "manifest_path", label="study map run"), base_dir=base_dir
    )
    checkpoint_run_dir = _resolve_path(
        _require_string(entry, "checkpoint_run_dir", label="study map run"), base_dir=base_dir
    )
    artifact_paths = {
        name: checkpoint_run_dir / relative_path
        for name, relative_path in _ARTIFACT_PATHS.items()
    }
    record: dict[str, Any] = {
        "autoencoder_seed": seed,
        "manifest_path": str(manifest_path),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "rejection_reasons": [],
        "metrics": {},
        "probe_valid": None,
    }

    info: dict[str, Any] | None = None
    try:
        info = _manifest_info(
            _read_manifest(manifest_path),
            map_study_id=study_id,
            map_protocol_version=protocol_version,
            map_expected_seeds=expected_seeds,
            mapped_configuration_id=configuration_id,
            mapped_seed=seed,
        )
        record["manifest"] = info
    except ParetoCollectionError as error:
        _append_reason(record, f"manifest_invalid:{error}")

    payloads: dict[str, Mapping[str, Any]] = {}
    for name, path in artifact_paths.items():
        try:
            payloads[name] = _read_json(path)
        except ParetoCollectionError as error:
            _append_reason(record, f"{name}_artifact_invalid:{error}")

    if info is not None and "leakage" in payloads:
        try:
            probe_valid, probe_reason = _validate_leakage_artifact(
                payloads["leakage"], info=info, seed=seed
            )
            record["probe_valid"] = probe_valid
            if not probe_valid:
                _append_reason(
                    record,
                    "invalid_probe" if probe_reason is None else f"invalid_probe:{probe_reason}",
                )
        except ParetoCollectionError as error:
            _append_reason(record, f"leakage_artifact_invalid:{error}")

    if info is not None and "correlation" in payloads:
        try:
            record["metrics"].update(
                _validate_correlation_artifact(
                    payloads["correlation"], raw_target=info["raw_target"]
                )
            )
        except ParetoCollectionError as error:
            _append_reason(record, f"correlation_artifact_invalid:{error}")

    validators = {
        "efficiency": _validate_efficiency_artifact,
        "collapse": _validate_collapse_artifact,
        "auroc": _validate_auroc_artifact,
    }
    for name, validator in validators.items():
        if name not in payloads:
            continue
        try:
            record["metrics"].update(validator(payloads[name]))
        except ParetoCollectionError as error:
            _append_reason(record, f"{name}_artifact_invalid:{error}")

    record["valid"] = not record["rejection_reasons"]
    return record


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ParetoCollectionError("Metric aggregation requires at least two finite seed values.")
    sample_std = float(np.std(array, ddof=1))
    standard_error = sample_std / math.sqrt(array.size)
    ci_half_width = 1.96 * standard_error
    mean = float(np.mean(array))
    return {
        "n_seeds": int(array.size),
        "mean": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_low": mean - ci_half_width,
        "ci95_high": mean + ci_half_width,
    }


def _configuration_contract(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    manifests = [record.get("manifest") for record in records if record.get("manifest")]
    if not manifests:
        return None
    first = manifests[0]
    fields = ("candidate", "constraints", "leakage_algorithm_hash", "raw_target")
    if any(any(manifest[field] != first[field] for field in fields) for manifest in manifests[1:]):
        return None
    return first


def _collect_configuration(
    configuration_id: str,
    entries: Sequence[Mapping[str, Any]],
    *,
    base_dir: Path,
    study_id: str,
    protocol_version: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    records = [
        _collect_run(
            entry,
            base_dir=base_dir,
            study_id=study_id,
            protocol_version=protocol_version,
            expected_seeds=expected_seeds,
        )
        for entry in entries
    ]
    records.sort(key=lambda record: record["autoencoder_seed"])
    rejection_reasons: list[str] = []

    def add_reason(reason: str) -> None:
        if reason not in rejection_reasons:
            rejection_reasons.append(reason)

    records_by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        seed = record["autoencoder_seed"]
        if seed in records_by_seed:
            add_reason(f"duplicate_study_mapping_seed:{seed}")
            continue
        records_by_seed[seed] = record
        for reason in record["rejection_reasons"]:
            add_reason(f"seed_{seed}:{reason}")

    contract = _configuration_contract(records)
    if contract is None:
        add_reason("manifest_contract_mismatch_or_missing")

    leakage_paths = [
        record["artifact_paths"]["leakage"]
        for record in records
        if Path(record["artifact_paths"]["leakage"]).is_file()
    ]
    leakage_aggregate: dict[str, Any] | None = None
    try:
        leakage_aggregate = aggregate_paired_seed_leakage(
            leakage_paths,
            expected_autoencoder_seeds=expected_seeds,
        )
        for run in leakage_aggregate["runs"]:
            record = records_by_seed.get(run["autoencoder_seed"])
            if record is not None:
                record["metrics"]["leakage_worst"] = run["leakage_worst"]
        if not leakage_aggregate["configuration_valid"]:
            for reason in leakage_aggregate["rejection_reasons"]:
                add_reason(f"leakage_{reason}")
    except ProbeAggregationError as error:
        add_reason(f"leakage_aggregation_failed:{error}")

    if set(records_by_seed) != set(expected_seeds):
        missing = [seed for seed in expected_seeds if seed not in records_by_seed]
        if missing:
            add_reason(f"missing_study_mapping_seeds:{missing}")

    configuration_valid = not rejection_reasons
    aggregates: dict[str, dict[str, float | int]] | None = None
    if configuration_valid:
        try:
            aggregates = {
                metric: _summary(
                    [
                        _finite_float(
                            records_by_seed[seed]["metrics"].get(metric),
                            label=f"seed {seed} {metric}",
                        )
                        for seed in expected_seeds
                    ]
                )
                for metric in _SUMMARY_METRICS
            }
        except ParetoCollectionError as error:
            add_reason(f"metric_aggregation_failed:{error}")
            configuration_valid = False
            aggregates = None

    return {
        "schema_version": PARETO_METRICS_SCHEMA_VERSION,
        "study_id": study_id,
        "protocol_version": protocol_version,
        "configuration_id": configuration_id,
        "expected_autoencoder_seeds": list(expected_seeds),
        "candidate": None if contract is None else contract["candidate"],
        "leakage_algorithm_hash": (
            None if contract is None else contract["leakage_algorithm_hash"]
        ),
        "configuration_valid": configuration_valid,
        "feasible": False,
        "rejection_reasons": rejection_reasons,
        "runs": records,
        "leakage_aggregate": leakage_aggregate,
        "aggregates": aggregates,
        "constraints": None,
    }


def _apply_paired_constraints(configurations: list[dict[str, Any]]) -> None:
    """Apply the frozen gamma-zero comparisons without constructing a Pareto front."""

    by_architecture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for configuration in configurations:
        candidate = configuration.get("candidate")
        if isinstance(candidate, Mapping):
            by_architecture[str(candidate["architecture_id"])].append(configuration)

    for configuration in configurations:
        candidate = configuration.get("candidate")
        if not isinstance(candidate, Mapping) or not configuration["configuration_valid"]:
            configuration["constraints"] = {
                "passed": False,
                "reason": "configuration_invalid_before_constraints",
            }
            continue

        gamma = float(candidate["mi_gamma"])
        architecture_id = str(candidate["architecture_id"])
        baselines = [
            item
            for item in by_architecture[architecture_id]
            if isinstance(item.get("candidate"), Mapping)
            and float(item["candidate"]["mi_gamma"]) == 0.0
            and item["candidate"]["mi_sensitive_num_bins"] == 50
            and item["candidate"]["encoder_nodes"] == candidate["encoder_nodes"]
        ]
        baseline = configuration if gamma == 0.0 else None
        if gamma != 0.0 and len(baselines) == 1:
            baseline = baselines[0]

        if baseline is None:
            reason = "missing_paired_gamma_zero_baseline"
            configuration["constraints"] = {"passed": False, "reason": reason}
            configuration["feasible"] = False
            _append_reason(configuration, reason)
            continue
        if not baseline["configuration_valid"]:
            reason = "paired_gamma_zero_baseline_invalid"
            configuration["constraints"] = {"passed": False, "reason": reason}
            configuration["feasible"] = False
            _append_reason(configuration, reason)
            continue

        policy = configuration["runs"][0]["manifest"]["constraints"]
        entropy_fraction = policy[
            "minimum_fraction_of_paired_gamma_zero_joint_entropy"
        ]
        min_efficiency_fraction = 1.0 - policy["max_relative_efficiency_degradation"]
        candidate_runs = {
            run["autoencoder_seed"]: run for run in configuration["runs"]
        }
        baseline_runs = {run["autoencoder_seed"]: run for run in baseline["runs"]}
        per_seed: list[dict[str, Any]] = []
        all_pass = True

        for seed in configuration["expected_autoencoder_seeds"]:
            candidate_run = candidate_runs[seed]
            baseline_run = baseline_runs[seed]
            candidate_metrics = candidate_run["metrics"]
            baseline_metrics = baseline_run["metrics"]
            candidate_absolute_pass = candidate_metrics["absolute_entropy_pass"]
            baseline_absolute_pass = baseline_metrics["absolute_entropy_pass"]
            relative_entropy_pass = (
                candidate_metrics["joint_code_entropy_bits"]
                >= entropy_fraction * baseline_metrics["joint_code_entropy_bits"]
            )
            minimum_efficiency_pass = (
                candidate_metrics["min_efficiency"]
                >= min_efficiency_fraction * baseline_metrics["min_efficiency"]
            )
            seed_pass = (
                candidate_absolute_pass
                and baseline_absolute_pass
                and relative_entropy_pass
                and minimum_efficiency_pass
            )
            all_pass = all_pass and seed_pass
            per_seed.append(
                {
                    "autoencoder_seed": seed,
                    "candidate_absolute_entropy_pass": candidate_absolute_pass,
                    "baseline_absolute_entropy_pass": baseline_absolute_pass,
                    "relative_entropy_pass": relative_entropy_pass,
                    "minimum_efficiency_pass": minimum_efficiency_pass,
                    "candidate_joint_code_entropy_bits": candidate_metrics[
                        "joint_code_entropy_bits"
                    ],
                    "baseline_joint_code_entropy_bits": baseline_metrics[
                        "joint_code_entropy_bits"
                    ],
                    "candidate_min_efficiency": candidate_metrics["min_efficiency"],
                    "baseline_min_efficiency": baseline_metrics["min_efficiency"],
                }
            )

        failed_seeds = [item["autoencoder_seed"] for item in per_seed if not (
            item["candidate_absolute_entropy_pass"]
            and item["baseline_absolute_entropy_pass"]
            and item["relative_entropy_pass"]
            and item["minimum_efficiency_pass"]
        )]
        configuration["constraints"] = {
            "passed": all_pass,
            "paired_gamma_zero_configuration_id": baseline["configuration_id"],
            "entropy_fraction_required": entropy_fraction,
            "minimum_efficiency_fraction_required": min_efficiency_fraction,
            "per_seed": per_seed,
        }
        configuration["feasible"] = all_pass
        if not all_pass:
            _append_reason(configuration, f"paired_constraints_failed_seeds:{failed_seeds}")


def collect_pareto_study(
    study_map: Mapping[str, Any], *, base_dir: str | Path = "."
) -> dict[str, Any]:
    """Collect a full mapped study into configuration-level Phase 2 records."""

    study_id, protocol_version, expected_seeds, entries = _validate_study_map(study_map)
    grouped_entries: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        configuration_id = _require_string(entry, "configuration_id", label="study map run")
        grouped_entries[configuration_id].append(entry)

    configurations = [
        _collect_configuration(
            configuration_id,
            grouped_entries[configuration_id],
            base_dir=Path(base_dir),
            study_id=study_id,
            protocol_version=protocol_version,
            expected_seeds=expected_seeds,
        )
        for configuration_id in sorted(grouped_entries)
    ]
    _apply_paired_constraints(configurations)
    return {
        "schema_version": PARETO_METRICS_SCHEMA_VERSION,
        "study_id": study_id,
        "protocol_version": protocol_version,
        "expected_autoencoder_seeds": list(expected_seeds),
        "configurations": configurations,
    }


def _flat_study_rows(collection: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for configuration in collection["configurations"]:
        candidate = configuration.get("candidate") or {}
        row: dict[str, Any] = {
            "study_id": collection["study_id"],
            "protocol_version": collection["protocol_version"],
            "configuration_id": configuration["configuration_id"],
            "mi_gamma": candidate.get("mi_gamma"),
            "mi_sensitive_num_bins": candidate.get("mi_sensitive_num_bins"),
            "architecture_id": candidate.get("architecture_id"),
            "encoder_nodes": json.dumps(candidate.get("encoder_nodes")),
            "leakage_algorithm_hash": configuration.get("leakage_algorithm_hash"),
            "configuration_valid": configuration["configuration_valid"],
            "feasible": configuration["feasible"],
            "rejection_reasons": ";".join(configuration["rejection_reasons"]),
        }
        aggregates = configuration.get("aggregates") or {}
        for metric, summary in aggregates.items():
            for name, value in summary.items():
                row[f"{metric}_{name}"] = value
        rows.append(row)
    return rows


def write_pareto_study_outputs(
    collection: Mapping[str, Any], *, output_dir: str | Path
) -> dict[str, Path]:
    """Write the required per-configuration JSON and study-wide CSV/Parquet tables."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for configuration in collection["configurations"]:
        configuration_dir = destination / configuration["configuration_id"]
        configuration_dir.mkdir(parents=True, exist_ok=True)
        (configuration_dir / "pareto_metrics.json").write_text(
            json.dumps(configuration, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    table = pd.DataFrame(_flat_study_rows(collection))
    csv_path = destination / "pareto_metrics.csv"
    parquet_path = destination / "pareto_metrics.parquet"
    table.to_csv(csv_path, index=False)
    table.to_parquet(parquet_path, index=False)
    return {"csv": csv_path, "parquet": parquet_path}


def collect_and_write_pareto_study(
    study_map_path: str | Path, *, output_dir: str | Path
) -> dict[str, Any]:
    """Load a study map, collect it, and write all Phase 2 outputs."""

    study_map, base_dir = load_study_map(study_map_path)
    collection = collect_pareto_study(study_map, base_dir=base_dir)
    collection["output_paths"] = {
        name: str(path)
        for name, path in write_pareto_study_outputs(
            collection, output_dir=output_dir
        ).items()
    }
    return collection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Phase 2 paired-seed Pareto-study metrics."
    )
    parser.add_argument(
        "--study-map",
        type=Path,
        required=True,
        help="Explicit YAML/JSON mapping of configuration/seed runs to artifact paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-configuration JSON and study-wide tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    collection = collect_and_write_pareto_study(
        args.study_map,
        output_dir=args.output_dir,
    )
    print(
        "Collected "
        f"{len(collection['configurations'])} configurations; "
        f"CSV: {collection['output_paths']['csv']}"
    )


if __name__ == "__main__":
    main()
