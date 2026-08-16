"""Fail-closed joint AUPRC and efficiency analysis for search-50 fronts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.physics_background_pairing_analysis import DOWNSTREAM_DATASETS
from src.evaluation.physics_background_pairing_paper_comparison import MODELS
from src.evaluation.physics_background_pairing_search50_analysis import (
    EXPECTED_FRONTS,
    EXPECTED_MEMBERSHIPS,
    STRATEGIES,
    _expected_families,
)

SCHEMA_VERSION = 1
EXPECTED_ANOMALIES = len(DOWNSTREAM_DATASETS)
EXPECTED_LINKED_ROWS = EXPECTED_MEMBERSHIPS * EXPECTED_ANOMALIES
LEARNED_PAIRINGS = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
    "jetclr",
)

IDENTITY_FIELDS = (
    "membership_index",
    "model",
    "logical_objective",
    "score",
    "metric",
    "strategy",
    "trial_number",
    "checkpoint",
    "checkpoint_sha256",
    "anomaly",
)


def _canonical_hash(value: Any) -> str:
    """Hash JSON data using the campaign's stable canonical encoding."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_unit(value: Any, label: str) -> float:
    """Return a finite probability-like value or reject it."""
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1], got {value!r}.")
    return result


def _normalized_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return the exact membership/checkpoint/anomaly join identity."""
    normalized = dict(row)
    normalized["membership_index"] = int(normalized["membership_index"])
    normalized["trial_number"] = int(normalized["trial_number"])
    normalized["strategy"] = str(normalized.get("strategy") or "")
    return tuple(normalized[field] for field in IDENTITY_FIELDS)


def _stats(values: Iterable[float], endpoint: str) -> dict[str, float]:
    """Summarize one endpoint over the exact downstream anomaly panel."""
    sample = [_finite_unit(value, endpoint) for value in values]
    if len(sample) != EXPECTED_ANOMALIES:
        raise ValueError(f"Expected exactly {EXPECTED_ANOMALIES} {endpoint} values.")
    ordered = sorted(sample)
    return {
        f"mean_{endpoint}": statistics.fmean(sample),
        f"median_{endpoint}": statistics.median(sample),
        f"worst_{endpoint}": ordered[0],
        f"bottom_quartile_cvar_{endpoint}": statistics.fmean(ordered[:5]),
    }


def _validate_aggregate(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Authenticate and return the complete candidate AUPRC payload."""
    if aggregate.get("schema_version") != 1:
        raise ValueError("Expected schema-1 AUPRC aggregate.")
    rows = aggregate.get("candidate_auprc")
    if not isinstance(rows, list) or len(rows) != EXPECTED_LINKED_ROWS:
        found = len(rows) if isinstance(rows, list) else "non-list"
        raise ValueError(
            f"AUPRC aggregate incomplete: expected {EXPECTED_LINKED_ROWS} rows, found {found}."
        )
    if aggregate.get("candidate_auprc_sha256") != _canonical_hash(rows):
        raise ValueError("AUPRC candidate-row hash differs.")
    return rows


def _index_endpoint(
    rows: Iterable[dict[str, Any]], value_field: str, endpoint: str
) -> dict[tuple[Any, ...], float]:
    """Index one endpoint with exact identity and coverage checks."""
    indexed: dict[tuple[Any, ...], float] = {}
    membership_anomalies: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        identity = _normalized_identity(row)
        if identity in indexed:
            raise ValueError(f"Duplicate {endpoint} identity: {identity}.")
        value = _finite_unit(row[value_field], endpoint)
        indexed[identity] = value
        membership_anomalies[int(row["membership_index"])].add(str(row["anomaly"]))
    if len(indexed) != EXPECTED_LINKED_ROWS:
        raise ValueError(f"Expected {EXPECTED_LINKED_ROWS} unique {endpoint} rows.")
    if set(membership_anomalies) != set(range(EXPECTED_MEMBERSHIPS)):
        raise ValueError(f"{endpoint} does not cover all membership indices.")
    expected_anomalies = set(DOWNSTREAM_DATASETS)
    if any(anomalies != expected_anomalies for anomalies in membership_anomalies.values()):
        raise ValueError(f"{endpoint} lacks exact downstream anomaly coverage.")
    return indexed


def _validate_fronts(
    efficiency_index: dict[tuple[Any, ...], float], front_winners: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Validate 56 frozen efficiency-selected front winners from raw rows."""
    grouped: dict[int, list[tuple[tuple[Any, ...], float]]] = defaultdict(list)
    for identity, value in efficiency_index.items():
        grouped[int(identity[0])].append((identity, value))
    summaries: dict[int, dict[str, Any]] = {}
    fronts: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, pairs in grouped.items():
        identities = [identity for identity, _ in pairs]
        invariant = {identity[1:9] for identity in identities}
        if len(invariant) != 1:
            raise ValueError(f"Membership {index} has inconsistent identity.")
        model, objective, score, metric, strategy, trial, checkpoint, digest = invariant.pop()
        stats = _stats((value for _, value in pairs), "efficiency")
        row = {
            "membership_index": index,
            "model": model,
            "logical_objective": objective,
            "score": score,
            "metric": metric,
            "strategy": strategy,
            "trial_number": trial,
            "checkpoint": checkpoint,
            "checkpoint_sha256": digest,
            **stats,
        }
        summaries[index] = row
        fronts[(model, score, metric, strategy)].append(row)
    expected_front_keys = {
        (model, score, metric, strategy)
        for model, score in _expected_families()
        for metric, strategy in STRATEGIES
    }
    if set(fronts) != expected_front_keys or len(fronts) != EXPECTED_FRONTS:
        raise ValueError("Endpoint rows do not cover the exact 56 logical fronts.")
    expected_winners = {
        min(rows, key=lambda row: (-row["mean_efficiency"], row["membership_index"]))[
            "membership_index"
        ]
        for rows in fronts.values()
    }
    supplied = {int(row["membership_index"]): row for row in front_winners}
    if len(front_winners) != EXPECTED_FRONTS or set(supplied) != expected_winners:
        raise ValueError("Supplied front winners differ from frozen efficiency selections.")
    for index, winner in supplied.items():
        source = summaries[index]
        for field in (
            "model",
            "logical_objective",
            "score",
            "metric",
            "trial_number",
            "checkpoint",
            "checkpoint_sha256",
        ):
            observed = int(winner[field]) if field == "trial_number" else winner.get(field)
            if observed != source[field]:
                raise ValueError(f"Front-winner {field} differs for membership {index}.")
        if str(winner.get("strategy") or "") != source["strategy"]:
            raise ValueError(f"Front-winner strategy differs for membership {index}.")
        if not math.isclose(
            float(winner["mean_efficiency"]), source["mean_efficiency"], abs_tol=1e-12
        ):
            raise ValueError(f"Front-winner efficiency differs for membership {index}.")
    return {index: summaries[index] for index in sorted(expected_winners)}


def _comparison_row(row: dict[str, Any], label: str) -> dict[str, Any]:
    """Project one primary front row into a compact comparison record."""
    return {
        "model": row["model"],
        "score": row["score"],
        "comparison_member": label,
        "selector": row["metric"],
        "pairing": row["strategy"] or "not_applicable",
        "membership_index": row["membership_index"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "mean_efficiency": row["mean_efficiency"],
        "mean_auprc": row["mean_auprc"],
    }


def build_endpoint_analysis(
    efficiency_rows: list[dict[str, Any]],
    front_winners: list[dict[str, Any]],
    auprc_aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Build all joined endpoint tables, failing on any incomplete join."""
    auprc_rows = _validate_aggregate(auprc_aggregate)
    efficiency_index = _index_endpoint(efficiency_rows, "efficiency_operational", "efficiency")
    auprc_index = _index_endpoint(auprc_rows, "auprc", "auprc")
    if set(efficiency_index) != set(auprc_index):
        missing = len(set(efficiency_index) - set(auprc_index))
        extra = len(set(auprc_index) - set(efficiency_index))
        raise ValueError(f"AUPRC/efficiency key join differs: missing={missing}, extra={extra}.")

    winners = _validate_fronts(efficiency_index, front_winners)
    combined_long = []
    auprc_by_membership: dict[int, list[float]] = defaultdict(list)
    for identity in sorted(efficiency_index):
        metadata = dict(zip(IDENTITY_FIELDS, identity))
        metadata["selector"] = metadata.pop("metric")
        metadata["pairing"] = metadata.pop("strategy") or "not_applicable"
        for endpoint, value in (
            ("efficiency_250hz", efficiency_index[identity]),
            ("auprc", auprc_index[identity]),
        ):
            combined_long.append({**metadata, "endpoint": endpoint, "value": value})
        auprc_by_membership[int(identity[0])].append(auprc_index[identity])

    primary = []
    for index, row in winners.items():
        primary.append({**row, **_stats(auprc_by_membership[index], "auprc")})
    primary.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            0 if row["score"] == "native" else 1,
            STRATEGIES.index((row["metric"], row["strategy"])),
        )
    )
    lookup = {(row["model"], row["score"], row["metric"], row["strategy"]): row for row in primary}

    controlled, fixed_pairing, best_learned = [], [], []
    for model, score in _expected_families():
        cdf = lookup[(model, score, "cap", "cdf")]
        w1 = lookup[(model, score, "wasserstein", "")]
        drift = lookup[(model, score, "drift", "")]
        for label, row in (("cdf_cap", cdf), ("wasserstein", w1), ("drift", drift)):
            controlled.append(_comparison_row(row, label))
        learned = [lookup[(model, score, "cap", pairing)] for pairing in LEARNED_PAIRINGS]
        for row in learned:
            fixed_pairing.append(
                {
                    **_comparison_row(row, "learned_cap"),
                    "control_membership_index": cdf["membership_index"],
                    "delta_efficiency_vs_cdf": row["mean_efficiency"] - cdf["mean_efficiency"],
                    "delta_auprc_vs_cdf": row["mean_auprc"] - cdf["mean_auprc"],
                }
            )
        learned_winner = min(
            learned,
            key=lambda row: (
                -row["mean_efficiency"],
                -row["bottom_quartile_cvar_efficiency"],
                -row["worst_efficiency"],
                LEARNED_PAIRINGS.index(row["strategy"]),
            ),
        )
        for label, row in (
            ("best_learned_cap", learned_winner),
            ("wasserstein", w1),
            ("drift", drift),
        ):
            best_learned.append(_comparison_row(row, label))

    oas_panel = []
    for model in ("ae", "vae"):
        for metric, strategy in STRATEGIES:
            native = lookup[(model, "native", metric, strategy)]
            oas = lookup[(model, "residual_oas", metric, strategy)]
            oas_panel.append(
                {
                    "model": model,
                    "selector": metric,
                    "pairing": strategy or "not_applicable",
                    "native_membership_index": native["membership_index"],
                    "oas_membership_index": oas["membership_index"],
                    "native_mean_efficiency": native["mean_efficiency"],
                    "oas_mean_efficiency": oas["mean_efficiency"],
                    "delta_efficiency_oas_minus_native": oas["mean_efficiency"]
                    - native["mean_efficiency"],
                    "native_mean_auprc": native["mean_auprc"],
                    "oas_mean_auprc": oas["mean_auprc"],
                    "delta_auprc_oas_minus_native": oas["mean_auprc"] - native["mean_auprc"],
                }
            )

    native_controlled = [row for row in controlled if row["score"] == "native"]
    native_fixed = [row for row in fixed_pairing if row["score"] == "native"]
    macro = {
        "controlled_selectors": {
            label: {
                endpoint: statistics.fmean(
                    row[endpoint] for row in native_controlled if row["comparison_member"] == label
                )
                for endpoint in ("mean_efficiency", "mean_auprc")
            }
            for label in ("cdf_cap", "wasserstein", "drift")
        },
        "fixed_pairing_delta_vs_cdf": {
            pairing: {
                endpoint: statistics.fmean(
                    row[endpoint] for row in native_fixed if row["pairing"] == pairing
                )
                for endpoint in ("delta_efficiency_vs_cdf", "delta_auprc_vs_cdf")
            }
            for pairing in LEARNED_PAIRINGS
        },
    }
    return {
        "method": {
            "schema_version": SCHEMA_VERSION,
            "primary_selection_endpoint": "efficiency_250hz",
            "candidate_memberships": EXPECTED_MEMBERSHIPS,
            "logical_fronts": EXPECTED_FRONTS,
            "anomalies": EXPECTED_ANOMALIES,
            "rows_per_endpoint": EXPECTED_LINKED_ROWS,
            "combined_rows": 2 * EXPECTED_LINKED_ROWS,
            "auprc_aggregate_manifest_content_sha256": auprc_aggregate.get(
                "manifest_content_sha256"
            ),
        },
        "combined_long": combined_long,
        "primary_front_summary": primary,
        "controlled_selector_comparison": controlled,
        "fixed_pairing_vs_cdf": fixed_pairing,
        "best_learned_comparison": best_learned,
        "oas_panel": oas_panel,
        "summary": macro,
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read one UTF-8 CSV as dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a nonempty rectangular CSV atomically enough for analysis artifacts."""
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_endpoint_analysis(
    provisional_dir: Path, auprc_aggregate_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Load authenticated inputs and persist all endpoint analysis tables."""
    aggregate = json.loads(auprc_aggregate_path.read_text(encoding="utf-8"))
    analysis = build_endpoint_analysis(
        _read_csv(provisional_dir / "candidate_efficiencies.csv"),
        _read_csv(provisional_dir / "front_winners.csv"),
        aggregate,
    )
    for key in (
        "combined_long",
        "primary_front_summary",
        "controlled_selector_comparison",
        "fixed_pairing_vs_cdf",
        "best_learned_comparison",
        "oas_panel",
    ):
        _write_csv(output_dir / f"{key}.csv", analysis[key])
    summary = {key: value for key, value in analysis.items() if key != "combined_long"}
    summary["artifact_hashes"] = {
        f"{key}.csv": hashlib.sha256((output_dir / f"{key}.csv").read_bytes()).hexdigest()
        for key in (
            "combined_long",
            "primary_front_summary",
            "controlled_selector_comparison",
            "fixed_pairing_vs_cdf",
            "best_learned_comparison",
            "oas_panel",
        )
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "endpoint_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return analysis
