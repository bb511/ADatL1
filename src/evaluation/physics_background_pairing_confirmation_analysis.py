"""Paired endpoint analysis for the 28-config physics confirmation matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.evaluation.physics_background_pairing_analysis import DOWNSTREAM_DATASETS

SCHEMA_VERSION = 1
ENDPOINTS = ("efficiency_250hz", "auprc")
LEARNED_PAIRINGS = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
    "jetclr",
)
PRACTICAL_THRESHOLDS = {"efficiency_250hz": 0.005, "auprc": 0.01}
DEFAULT_BOOTSTRAP_SEED = 150050165
DEFAULT_BOOTSTRAP_REPLICATES = 100_000


def _content_hash(value: Any) -> str:
    """Hash a JSON-compatible value with the campaign encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _finite_unit(value: Any, label: str) -> float:
    """Return a finite endpoint value in the unit interval."""
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0,1], got {value!r}.")
    return result


def _authenticate_matrix(matrix: dict[str, Any]) -> tuple[list[dict], tuple[int, ...]]:
    """Authenticate the frozen 42-cell/28-config matrix and return its seeds."""
    digest = matrix.get("content_sha256")
    unhashed = {key: value for key, value in matrix.items() if key != "content_sha256"}
    if digest != _content_hash(unhashed):
        raise ValueError("Confirmation matrix content hash differs.")
    if (
        matrix.get("schema_version") != 1
        or matrix.get("kind") != "physics_background_native_confirmation_matrix"
    ):
        raise ValueError("Confirmation matrix identity differs.")
    cells = matrix.get("logical_cells")
    configs = matrix.get("configurations")
    seeds = tuple(int(seed) for seed in matrix.get("training_seeds", []))
    if not isinstance(cells, list) or len(cells) != 42:
        raise ValueError("Confirmation matrix must contain exactly 42 logical cells.")
    if not isinstance(configs, list) or len(configs) != 28:
        raise ValueError("Confirmation matrix must contain exactly 28 configurations.")
    if len(seeds) not in {3, 5} or len(set(seeds)) != len(seeds):
        raise ValueError("Confirmation analysis requires three or five distinct seeds.")
    indices = {int(cell["cell_index"]) for cell in cells}
    if indices != set(range(42)):
        raise ValueError("Logical cell indices must be exactly 0..41.")
    expected = {
        (model, selector, pairing)
        for model in ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
        for selector, pairing in (
            *(("cap", item) for item in (*LEARNED_PAIRINGS, "cdf")),
            ("wasserstein", None),
            ("drift", None),
        )
    }
    observed = {(cell["model"], cell["selector"], cell.get("pairing")) for cell in cells}
    if observed != expected:
        raise ValueError("Confirmation cells do not cover the exact selector matrix.")
    return cells, seeds


def _normalize_rows(
    cells: list[dict], seeds: tuple[int, ...], rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate exact endpoint coverage and same-checkpoint attachment."""
    cells_by_index = {int(cell["cell_index"]): cell for cell in cells}
    normalized, keys = [], set()
    expected_anomalies = set(DOWNSTREAM_DATASETS)
    coverage: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    checkpoint_identity: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    for supplied in rows:
        row = dict(supplied)
        row["cell_index"] = int(row["cell_index"])
        row["configuration_index"] = int(row["configuration_index"])
        row["trial_number"] = int(row["trial_number"])
        row["seed"] = int(row["seed"])
        row["pairing"] = row.get("pairing") or "not_applicable"
        row["value"] = _finite_unit(row["value"], str(row.get("endpoint")))
        cell = cells_by_index.get(row["cell_index"])
        if cell is None or row["seed"] not in seeds:
            raise ValueError("Endpoint row names an unknown cell or seed.")
        expected_fields = {
            "configuration_index": int(cell["configuration_index"]),
            "model": cell["model"],
            "trial_number": int(cell["trial_number"]),
            "score": "native",
            "selector": cell["selector"],
            "pairing": cell.get("pairing") or "not_applicable",
            "logical_objective": cell["logical_objective"],
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Endpoint row {field} differs for cell {row['cell_index']}: "
                    f"{row.get(field)!r} != {expected!r}."
                )
        if row.get("endpoint") not in ENDPOINTS or row.get("anomaly") not in expected_anomalies:
            raise ValueError("Endpoint or anomaly identity differs from the frozen contract.")
        if not row.get("checkpoint_path") or not row.get("checkpoint_sha256"):
            raise ValueError("Endpoint row lacks selected checkpoint identity.")
        key = (row["cell_index"], row["seed"], row["anomaly"], row["endpoint"])
        if key in keys:
            raise ValueError(f"Duplicate confirmation endpoint key: {key}.")
        keys.add(key)
        coverage[(row["cell_index"], row["seed"], row["endpoint"])].add(row["anomaly"])
        checkpoint_identity[(row["cell_index"], row["seed"])].add(
            (str(row["checkpoint_path"]), str(row["checkpoint_sha256"]))
        )
        normalized.append(row)
    expected_rows = 42 * len(seeds) * len(DOWNSTREAM_DATASETS) * len(ENDPOINTS)
    if len(normalized) != expected_rows:
        raise ValueError(
            f"Confirmation endpoint coverage incomplete: expected {expected_rows}, "
            f"found {len(normalized)}."
        )
    expected_groups = {
        (cell, seed, endpoint) for cell in range(42) for seed in seeds for endpoint in ENDPOINTS
    }
    if set(coverage) != expected_groups or any(
        anomalies != expected_anomalies for anomalies in coverage.values()
    ):
        raise ValueError("Confirmation rows lack exact 42×seed×endpoint×anomaly coverage.")
    if any(len(identities) != 1 for identities in checkpoint_identity.values()):
        raise ValueError(
            "Both endpoints must attach to the same selected checkpoint per cell/seed."
        )
    return sorted(
        normalized,
        key=lambda row: (
            row["cell_index"],
            seeds.index(row["seed"]),
            DOWNSTREAM_DATASETS.index(row["anomaly"]),
            ENDPOINTS.index(row["endpoint"]),
        ),
    )


def _endpoint_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize every cell, seed, and endpoint over the same anomaly panel."""
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["cell_index"], row["seed"], row["endpoint"])].append(row)
    summaries = []
    for key in sorted(groups):
        group = groups[key]
        values = sorted(row["value"] for row in group)
        first = group[0]
        summaries.append(
            {
                **{
                    field: first[field]
                    for field in (
                        "cell_index",
                        "configuration_index",
                        "model",
                        "trial_number",
                        "seed",
                        "score",
                        "selector",
                        "pairing",
                        "logical_objective",
                        "checkpoint_path",
                        "checkpoint_sha256",
                        "endpoint",
                    )
                },
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "worst": values[0],
                "bottom_quartile_cvar": statistics.fmean(values[:5]),
            }
        )
    return summaries


def _across_seed_summaries(
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Summarize every logical cell across the paired seed/anomaly panel."""
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["cell_index"], row["endpoint"])].append(row)
    summaries = []
    for (cell_index, endpoint), group in sorted(groups.items()):
        indexed = {(row["seed"], row["anomaly"]): row["value"] for row in group}
        matrix = np.array(
            [[indexed[(seed, anomaly)] for anomaly in DOWNSTREAM_DATASETS] for seed in seeds]
        )
        identity = f"absolute|{cell_index}|{endpoint}"
        derived_seed = bootstrap_seed ^ int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
        low, high = _bootstrap_interval(
            matrix,
            confidence=0.95,
            replicates=bootstrap_replicates,
            seed=derived_seed,
        )
        # Robustness is defined over anomaly types after averaging retraining
        # seeds. Flattening seed × anomaly would incorrectly count each seed as
        # another anomaly and overweight models with reserve-seed extensions.
        anomaly_means = matrix.mean(axis=0)
        values = np.sort(anomaly_means)
        first = group[0]
        summaries.append(
            {
                **{
                    field: first[field]
                    for field in (
                        "cell_index",
                        "configuration_index",
                        "model",
                        "trial_number",
                        "score",
                        "selector",
                        "pairing",
                        "logical_objective",
                        "endpoint",
                    )
                },
                "seeds": len(seeds),
                "anomalies": len(DOWNSTREAM_DATASETS),
                "mean": float(matrix.mean()),
                "median": float(np.median(anomaly_means)),
                "worst": float(values[0]),
                "bottom_quartile_cvar": float(values[:5].mean()),
                "seed_mean_std": float(matrix.mean(axis=1).std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": derived_seed,
            }
        )
    if len(summaries) != 42 * len(ENDPOINTS):
        raise RuntimeError("Across-seed endpoint summary cardinality differs.")
    return summaries


def _normalize_oas_rows(
    cells: list[dict],
    seeds: tuple[int, ...],
    supplied_rows: Iterable[dict[str, Any]],
    native_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Authenticate masked-OAS endpoints on the same native-selected checkpoints."""
    oas_cells = {int(cell["cell_index"]): cell for cell in cells if cell["model"] in {"ae", "vae"}}
    native = {
        (row["cell_index"], row["seed"], row["anomaly"], row["endpoint"]): row
        for row in native_rows
    }
    normalized, keys = [], set()
    coverage: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for supplied in supplied_rows:
        row = dict(supplied)
        row["cell_index"] = int(row["cell_index"])
        row["configuration_index"] = int(row["configuration_index"])
        row["trial_number"] = int(row["trial_number"])
        row["seed"] = int(row["seed"])
        row["pairing"] = row.get("pairing") or "not_applicable"
        row["value"] = _finite_unit(row["value"], str(row.get("endpoint")))
        cell = oas_cells.get(row["cell_index"])
        if cell is None or row["seed"] not in seeds:
            raise ValueError("OAS endpoint row names an unknown AE/VAE cell or seed.")
        expected_fields = {
            "configuration_index": int(cell["configuration_index"]),
            "model": cell["model"],
            "trial_number": int(cell["trial_number"]),
            "score": "residual_oas",
            "selector": cell["selector"],
            "pairing": cell.get("pairing") or "not_applicable",
            "logical_objective": cell["logical_objective"],
        }
        if any(row.get(field) != expected for field, expected in expected_fields.items()):
            raise ValueError(f"OAS endpoint metadata differs for cell {row['cell_index']}.")
        if row.get("endpoint") not in ENDPOINTS or row.get("anomaly") not in DOWNSTREAM_DATASETS:
            raise ValueError("OAS endpoint or anomaly identity differs.")
        key = (row["cell_index"], row["seed"], row["anomaly"], row["endpoint"])
        native_row = native.get(key)
        if key in keys or native_row is None:
            raise ValueError(f"Duplicate or unmatched OAS endpoint key: {key}.")
        if (row.get("checkpoint_path"), row.get("checkpoint_sha256")) != (
            native_row["checkpoint_path"],
            native_row["checkpoint_sha256"],
        ):
            raise ValueError("OAS and native endpoints must use the same selected checkpoint.")
        keys.add(key)
        coverage[(row["cell_index"], row["seed"], row["endpoint"])].add(row["anomaly"])
        normalized.append(row)
    expected_rows = len(oas_cells) * len(seeds) * len(DOWNSTREAM_DATASETS) * len(ENDPOINTS)
    if len(normalized) != expected_rows or any(
        anomalies != set(DOWNSTREAM_DATASETS) for anomalies in coverage.values()
    ):
        raise ValueError(
            f"OAS endpoint coverage incomplete: expected {expected_rows}, found {len(normalized)}."
        )
    return sorted(
        normalized,
        key=lambda row: (
            row["cell_index"],
            seeds.index(row["seed"]),
            DOWNSTREAM_DATASETS.index(row["anomaly"]),
            ENDPOINTS.index(row["endpoint"]),
        ),
    )


def _oas_comparisons(
    native_rows: list[dict[str, Any]],
    oas_rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare masked OAS with native scores on identical selected checkpoints."""
    native = {
        (row["cell_index"], row["seed"], row["anomaly"], row["endpoint"]): row
        for row in native_rows
    }
    paired = []
    for row in oas_rows:
        key = (row["cell_index"], row["seed"], row["anomaly"], row["endpoint"])
        control = native[key]
        paired.append(
            {
                "cell_index": row["cell_index"],
                "configuration_index": row["configuration_index"],
                "model": row["model"],
                "trial_number": row["trial_number"],
                "seed": row["seed"],
                "selector": row["selector"],
                "pairing": row["pairing"],
                "logical_objective": row["logical_objective"],
                "anomaly": row["anomaly"],
                "endpoint": row["endpoint"],
                "contrast": "residual_oas_minus_native",
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "oas_value": row["value"],
                "native_value": control["value"],
                "delta": row["value"] - control["value"],
            }
        )
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(row["cell_index"], row["endpoint"])].append(row)
    summaries = []
    for (cell_index, endpoint), group in sorted(grouped.items()):
        indexed = {(row["seed"], row["anomaly"]): row["delta"] for row in group}
        matrix = np.array(
            [[indexed[(seed, anomaly)] for anomaly in DOWNSTREAM_DATASETS] for seed in seeds]
        )
        identity = f"oas|{cell_index}|{endpoint}"
        derived_seed = bootstrap_seed ^ int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
        low, high = _bootstrap_interval(
            matrix,
            confidence=0.95,
            replicates=bootstrap_replicates,
            seed=derived_seed,
        )
        first = group[0]
        summaries.append(
            {
                **{
                    field: first[field]
                    for field in (
                        "cell_index",
                        "configuration_index",
                        "model",
                        "trial_number",
                        "selector",
                        "pairing",
                        "logical_objective",
                        "endpoint",
                        "contrast",
                    )
                },
                "seeds": len(seeds),
                "anomalies": len(DOWNSTREAM_DATASETS),
                "mean_delta": float(matrix.mean()),
                "median_delta": float(np.median(matrix)),
                "seed_positive": int((matrix.mean(axis=1) > 0).sum()),
                "anomaly_positive": int((matrix.mean(axis=0) > 0).sum()),
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": derived_seed,
            }
        )
    return paired, summaries


def _paired_rows(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Construct exact CDF-baseline and learned-pairing-minus-CDF differences."""
    index = {
        (
            row["model"],
            row["seed"],
            row["anomaly"],
            row["endpoint"],
            row["selector"],
            row["pairing"],
        ): row
        for row in rows
    }
    controlled, pairings = [], []
    models = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
    seeds = sorted({row["seed"] for row in rows})
    for model in models:
        for seed in seeds:
            for anomaly in DOWNSTREAM_DATASETS:
                for endpoint in ENDPOINTS:
                    cdf = index[(model, seed, anomaly, endpoint, "cap", "cdf")]
                    for baseline in ("wasserstein", "drift"):
                        control = index[
                            (model, seed, anomaly, endpoint, baseline, "not_applicable")
                        ]
                        controlled.append(
                            {
                                "model": model,
                                "seed": seed,
                                "anomaly": anomaly,
                                "endpoint": endpoint,
                                "contrast": f"cdf_cap_minus_{baseline}",
                                "treatment_cell_index": cdf["cell_index"],
                                "control_cell_index": control["cell_index"],
                                "treatment_checkpoint_sha256": cdf["checkpoint_sha256"],
                                "control_checkpoint_sha256": control["checkpoint_sha256"],
                                "treatment_value": cdf["value"],
                                "control_value": control["value"],
                                "delta": cdf["value"] - control["value"],
                            }
                        )
                    for pairing in LEARNED_PAIRINGS:
                        treatment = index[(model, seed, anomaly, endpoint, "cap", pairing)]
                        pairings.append(
                            {
                                "model": model,
                                "seed": seed,
                                "anomaly": anomaly,
                                "endpoint": endpoint,
                                "contrast": f"{pairing}_minus_cdf",
                                "pairing": pairing,
                                "treatment_cell_index": treatment["cell_index"],
                                "control_cell_index": cdf["cell_index"],
                                "treatment_checkpoint_sha256": treatment["checkpoint_sha256"],
                                "control_checkpoint_sha256": cdf["checkpoint_sha256"],
                                "treatment_value": treatment["value"],
                                "control_value": cdf["value"],
                                "delta": treatment["value"] - cdf["value"],
                            }
                        )
    return controlled, pairings


def _bootstrap_interval(
    matrix: np.ndarray,
    *,
    confidence: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic crossed seed/anomaly percentile interval."""
    if matrix.ndim != 2 or matrix.shape[1] != len(DOWNSTREAM_DATASETS):
        raise ValueError("Bootstrap matrix must be seed × 20 anomalies.")
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(0, matrix.shape[0], size=(replicates, matrix.shape[0]))
    anomaly_indices = rng.integers(0, matrix.shape[1], size=(replicates, matrix.shape[1]))
    seed_means = matrix[seed_indices].mean(axis=1)
    draws = np.take_along_axis(seed_means, anomaly_indices, axis=1).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, (alpha, 1.0 - alpha))
    return float(low), float(high)


def _classification(low: float, high: float, threshold: float) -> str:
    """Classify an adjusted interval relative to a practical threshold."""
    if low > threshold:
        return "meaningful_improvement"
    if high < -threshold:
        return "meaningful_degradation"
    if low >= -threshold and high <= threshold:
        return "practically_equivalent"
    if low > 0.0:
        return "positive_but_not_meaningful"
    if high < 0.0:
        return "negative_but_not_meaningful"
    return "inconclusive"


def _comparison_summaries(
    controlled: list[dict],
    pairings: list[dict],
    seeds: tuple[int, ...],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict]:
    """Summarize model and fixed-model macro paired contrasts with uncertainty."""
    summaries = []
    all_rows = [("controlled", row) for row in controlled] + [("pairing", row) for row in pairings]
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for family, row in all_rows:
        grouped[(family, row["model"], row["endpoint"], row["contrast"])].append(row)

    def summarize(
        family: str, model: str, endpoint: str, contrast: str, group: list[dict]
    ) -> dict:
        indexed = {(row["seed"], row["anomaly"]): row["delta"] for row in group}
        controls = {(row["seed"], row["anomaly"]): row["control_value"] for row in group}
        matrix = np.array(
            [[indexed[(seed, anomaly)] for anomaly in DOWNSTREAM_DATASETS] for seed in seeds]
        )
        control_matrix = np.array(
            [[controls[(seed, anomaly)] for anomaly in DOWNSTREAM_DATASETS] for seed in seeds]
        )
        identity = f"{family}|{model}|{endpoint}|{contrast}"
        derived_seed = bootstrap_seed ^ int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
        adjusted_confidence = 0.975 if family == "controlled" else 0.9875
        low, high = _bootstrap_interval(
            matrix,
            confidence=0.95,
            replicates=bootstrap_replicates,
            seed=derived_seed,
        )
        adjusted_low, adjusted_high = _bootstrap_interval(
            matrix,
            confidence=adjusted_confidence,
            replicates=bootstrap_replicates,
            seed=derived_seed,
        )
        seed_means = matrix.mean(axis=1)
        anomaly_means = matrix.mean(axis=0)
        threshold = PRACTICAL_THRESHOLDS[endpoint]
        mean_control = float(control_matrix.mean())
        return {
            "scope": "macro" if model == "__macro__" else "model",
            "model": "macro_equal_model" if model == "__macro__" else model,
            "family": family,
            "endpoint": endpoint,
            "contrast": contrast,
            "seeds": len(seeds),
            "anomalies": len(DOWNSTREAM_DATASETS),
            "mean_delta": float(matrix.mean()),
            "median_delta": float(np.median(matrix)),
            "relative_delta_vs_control": (
                float(matrix.mean()) / mean_control if mean_control > 0.0 else None
            ),
            "seed_positive": int((seed_means > 0).sum()),
            "anomaly_positive": int((anomaly_means > 0).sum()),
            "ci95_low": low,
            "ci95_high": high,
            "multiplicity_adjusted_confidence": adjusted_confidence,
            "adjusted_ci_low": adjusted_low,
            "adjusted_ci_high": adjusted_high,
            "practical_threshold": threshold,
            "classification": _classification(adjusted_low, adjusted_high, threshold),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": derived_seed,
        }

    for key in sorted(grouped):
        summaries.append(summarize(*key, grouped[key]))

    models = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
    for family in ("controlled", "pairing"):
        contrasts = sorted(
            {row["contrast"] for row in (controlled if family == "controlled" else pairings)}
        )
        source = controlled if family == "controlled" else pairings
        for endpoint in ENDPOINTS:
            for contrast in contrasts:
                model_groups = {
                    model: [
                        row
                        for row in source
                        if row["model"] == model
                        and row["endpoint"] == endpoint
                        and row["contrast"] == contrast
                    ]
                    for model in models
                }
                macro = []
                for seed in seeds:
                    for anomaly in DOWNSTREAM_DATASETS:
                        selected = [
                            next(
                                row
                                for row in model_groups[model]
                                if row["seed"] == seed and row["anomaly"] == anomaly
                            )
                            for model in models
                        ]
                        macro.append(
                            {
                                **selected[0],
                                "model": "__macro__",
                                "treatment_value": statistics.fmean(
                                    row["treatment_value"] for row in selected
                                ),
                                "control_value": statistics.fmean(
                                    row["control_value"] for row in selected
                                ),
                                "delta": statistics.fmean(row["delta"] for row in selected),
                            }
                        )
                summaries.append(summarize(family, "__macro__", endpoint, contrast, macro))
    return summaries


def build_confirmation_endpoint_analysis(
    matrix: dict[str, Any],
    endpoint_rows: Iterable[dict[str, Any]],
    *,
    oas_endpoint_rows: Iterable[dict[str, Any]] | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate, pair, and summarize a complete confirmation endpoint matrix."""
    if bootstrap_replicates < 1_000:
        raise ValueError("Confirmation analysis requires at least 1,000 bootstrap replicates.")
    cells, seeds = _authenticate_matrix(matrix)
    rows = _normalize_rows(cells, seeds, endpoint_rows)
    endpoint_summaries = _endpoint_summaries(rows)
    across_seed_summaries = _across_seed_summaries(
        rows,
        seeds,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    controlled, pairings = _paired_rows(rows)
    expected_controlled = 6 * 2 * len(seeds) * 20 * 2
    expected_pairings = 6 * 4 * len(seeds) * 20 * 2
    if len(controlled) != expected_controlled or len(pairings) != expected_pairings:
        raise RuntimeError("Internal paired-comparison cardinality differs.")
    comparisons = _comparison_summaries(
        controlled,
        pairings,
        seeds,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    oas_rows: list[dict[str, Any]] = []
    oas_paired: list[dict[str, Any]] = []
    oas_summaries: list[dict[str, Any]] = []
    if oas_endpoint_rows is not None:
        oas_rows = _normalize_oas_rows(cells, seeds, oas_endpoint_rows, rows)
        oas_paired, oas_summaries = _oas_comparisons(
            rows,
            oas_rows,
            seeds,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
    reserve_seed_decisions = []
    for model in ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp"):
        relevant = [
            row
            for row in comparisons
            if row["scope"] == "model"
            and row["model"] == model
            and row["endpoint"] == "efficiency_250hz"
        ]
        triggering = [
            row["contrast"]
            for row in relevant
            if row["adjusted_ci_low"] <= -PRACTICAL_THRESHOLDS["efficiency_250hz"]
            and row["adjusted_ci_high"] >= PRACTICAL_THRESHOLDS["efficiency_250hz"]
        ]
        reserve_seed_decisions.append(
            {
                "model": model,
                "primary_seed_count": 3,
                "observed_seed_count": len(seeds),
                "reserve_seeds": "141421;173205",
                "trigger_rule": "adjusted efficiency CI spans both -0.005 and +0.005",
                "triggered": bool(triggering) if len(seeds) == 3 else False,
                "triggering_contrasts": ";".join(sorted(triggering)),
                "decision": (
                    "add_both_reserve_seeds"
                    if len(seeds) == 3 and triggering
                    else "extension_complete"
                    if len(seeds) == 5
                    else "no_extension"
                ),
            }
        )
    return {
        "method": {
            "schema_version": SCHEMA_VERSION,
            "matrix_content_sha256": matrix["content_sha256"],
            "models": 6,
            "logical_cells": 42,
            "unique_configurations": 28,
            "seeds": list(seeds),
            "anomalies": 20,
            "endpoints": list(ENDPOINTS),
            "endpoint_rows": len(rows),
            "endpoint_summaries": len(endpoint_summaries),
            "across_seed_summaries": len(across_seed_summaries),
            "controlled_paired_rows": len(controlled),
            "pairing_paired_rows": len(pairings),
            "comparison_summaries": len(comparisons),
            "oas_endpoint_rows": len(oas_rows),
            "oas_paired_rows": len(oas_paired),
            "oas_comparison_summaries": len(oas_summaries),
            "primary_selection_is_validation_only": True,
            "bootstrap": {
                "kind": "crossed_paired_seed_anomaly_percentile",
                "replicates": bootstrap_replicates,
                "base_seed": bootstrap_seed,
            },
            "practical_thresholds": PRACTICAL_THRESHOLDS,
        },
        "endpoints_long": rows,
        "endpoints_summary": endpoint_summaries,
        "endpoints_across_seeds": across_seed_summaries,
        "selector_comparison_paired": controlled,
        "pairing_vs_cdf_paired": pairings,
        "comparison_summary": comparisons,
        "oas_endpoints_long": oas_rows,
        "oas_vs_native_paired": oas_paired,
        "oas_vs_native_summary": oas_summaries,
        "reserve_seed_decisions": reserve_seed_decisions,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one rectangular result table atomically and idempotently."""
    if not rows:
        raise ValueError(f"Cannot write empty result table {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"Refusing to replace different analysis table: {path}")
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_confirmation_endpoint_analysis(
    matrix_path: Path,
    endpoint_rows_path: Path,
    endpoint_receipt_path: Path,
    oas_rows_path: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Read JSON/CSV inputs and persist portable authenticated result tables."""
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    receipt = json.loads(endpoint_receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("matrix_content_sha256") != matrix.get("content_sha256")
        or receipt.get("artifacts", {}).get("endpoints_long.csv")
        != hashlib.sha256(endpoint_rows_path.read_bytes()).hexdigest()
        or receipt.get("artifacts", {}).get("oas_diagnostics.csv")
        != hashlib.sha256(oas_rows_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Endpoint extraction receipt or source-table hash differs.")
    with endpoint_rows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with oas_rows_path.open(newline="", encoding="utf-8") as handle:
        oas_rows = list(csv.DictReader(handle))
    result = build_confirmation_endpoint_analysis(
        matrix,
        rows,
        oas_endpoint_rows=oas_rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_names = (
        "endpoints_long",
        "endpoints_summary",
        "endpoints_across_seeds",
        "selector_comparison_paired",
        "pairing_vs_cdf_paired",
        "comparison_summary",
        "oas_endpoints_long",
        "oas_vs_native_paired",
        "oas_vs_native_summary",
        "reserve_seed_decisions",
    )
    for name in table_names:
        _write_csv(output_dir / f"{name}.csv", result[name])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": result["method"],
        "inputs": {
            "matrix": str(matrix_path.resolve()),
            "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "endpoint_rows": str(endpoint_rows_path.resolve()),
            "endpoint_rows_sha256": hashlib.sha256(endpoint_rows_path.read_bytes()).hexdigest(),
            "oas_rows": str(oas_rows_path.resolve()),
            "oas_rows_sha256": hashlib.sha256(oas_rows_path.read_bytes()).hexdigest(),
            "endpoint_receipt": str(endpoint_receipt_path.resolve()),
            "endpoint_receipt_sha256": hashlib.sha256(
                endpoint_receipt_path.read_bytes()
            ).hexdigest(),
        },
        "artifacts": {
            f"{name}.csv": hashlib.sha256((output_dir / f"{name}.csv").read_bytes()).hexdigest()
            for name in table_names
        },
    }
    manifest["content_sha256"] = _content_hash(manifest)
    (output_dir / "confirmation_endpoint_analysis.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "PRACTICAL_THRESHOLDS",
    "build_confirmation_endpoint_analysis",
    "write_confirmation_endpoint_analysis",
]
