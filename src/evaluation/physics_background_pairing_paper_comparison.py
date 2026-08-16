"""Authenticated paper baselines and honest campaign-to-paper comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
METHODS = ("semi", "cap", "stable", "wasserstein")
OPERATING_POINTS = ("250hz", "q99")
PAPER_REFERENCE = Path("research/physics_background_pairing/paper_reference.json")


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_authenticated_paper_reference(repository_root: Path) -> tuple[dict[str, Any], str]:
    """Load the paper reference and authenticate every declared notebook source."""
    repository_root = repository_root.resolve()
    reference_path = (repository_root / PAPER_REFERENCE).resolve()
    if not reference_path.is_relative_to(repository_root) or not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("schema_version") != 1:
        raise ValueError("Unknown physics paper-reference schema.")
    sources = reference.get("source_artifacts")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Physics paper reference does not name source artifacts.")
    for source in sources:
        path = (repository_root / str(source.get("path", ""))).resolve()
        if not path.is_relative_to(repository_root) or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != source.get("sha256"):
            raise ValueError(f"Physics paper-reference source changed: {path}.")

    efficiencies = reference.get("efficiency_means_percent", {})
    if set(efficiencies) != set(OPERATING_POINTS):
        raise ValueError("Physics paper reference has incomplete operating-point coverage.")
    for operating_point in OPERATING_POINTS:
        if set(efficiencies[operating_point]) != set(MODELS):
            raise ValueError(f"Incomplete paper model coverage at {operating_point}.")
        for values in efficiencies[operating_point].values():
            if set(values) != set(METHODS) or not all(
                value is None
                or (
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 100.0
                )
                for value in values.values()
            ):
                raise ValueError(f"Invalid paper efficiencies at {operating_point}.")
    correlations = reference.get("epoch_metric_downstream_correlations", {}).get("values", {})
    if set(correlations) != set(MODELS) or any(
        set(values) != set(METHODS)
        or any(
            not math.isfinite(float(value)) or abs(float(value)) > 1.0 for value in values.values()
        )
        for values in correlations.values()
    ):
        raise ValueError("Invalid paper epoch-correlation reference.")
    return reference, _sha256(reference_path)


def _winner_index(winner_rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index the three native campaign methods that match paper semantics."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in winner_rows:
        if row.get("score") != "native":
            continue
        metric = str(row.get("metric"))
        strategy = str(row.get("strategy") or "")
        method = (
            "cap"
            if metric == "cap" and strategy == "cdf"
            else "stable"
            if metric == "drift" and not strategy
            else "wasserstein"
            if metric == "wasserstein" and not strategy
            else None
        )
        if method is None:
            continue
        key = (str(row.get("model")), method)
        if key in result:
            raise ValueError(f"Duplicate paper-comparable campaign winner: {key}.")
        result[key] = row
    return result


def build_paper_comparison(
    reference: dict[str, Any], winner_rows: Iterable[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Build explicit paper comparison rows without manufacturing absent outcomes."""
    winners = None if winner_rows is None else _winner_index(winner_rows)
    rows = []
    for operating_point in OPERATING_POINTS:
        for model in MODELS:
            for method in METHODS:
                paper = reference["efficiency_means_percent"][operating_point][model][method]
                current = None
                delta = None
                if paper is None:
                    status = "paper_value_missing"
                elif method == "semi":
                    status = "no_signal_blind_campaign_equivalent"
                elif operating_point != "250hz":
                    status = "campaign_q99_not_exported"
                elif winners is None:
                    status = "downstream_results_not_available"
                else:
                    winner = winners.get((model, method))
                    if winner is None:
                        raise ValueError(
                            f"Missing paper-comparable campaign winner: {(model, method)}."
                        )
                    value = winner.get("mean_efficiency")
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise ValueError(f"Invalid campaign mean efficiency: {(model, method)}.")
                    current = 100.0 * float(value)
                    delta = current - float(paper)
                    status = "comparable"
                rows.append(
                    {
                        "operating_point": operating_point,
                        "model": model,
                        "paper_method": method,
                        "campaign_method": {
                            "cap": "native__cap__cdf",
                            "stable": "native__drift",
                            "wasserstein": "native__wasserstein",
                        }.get(method, ""),
                        "paper_mean_efficiency_percent": paper,
                        "campaign_mean_efficiency_percent": current,
                        "delta_percentage_points": delta,
                        "comparison_status": status,
                    }
                )
    return rows


def _value(value: Any, digits: int = 2) -> str:
    """Format a finite table value or an explicit missing marker."""
    return "—" if value is None else f"{float(value):.{digits}f}"


def paper_reference_markdown(reference: dict[str, Any], reference_sha256: str) -> list[str]:
    """Render authenticated baseline and epoch-correlation tables."""
    lines = [
        "## Authenticated paper reference",
        "",
        f"Reference artifact: `analysis/paper_reference.json` (SHA-256 `{reference_sha256}`).",
        "The efficiency entries are paper means over 20 downstream anomaly datasets, in percent. "
        "A dash is an explicit missing value, never zero.",
        "",
    ]
    for operating_point, label in (("250hz", "250 Hz"), ("q99", "q99")):
        lines.extend(
            [
                f"### Paper mean efficiency at {label}",
                "",
                "| Model | Semi | CAP | Stable/drift | W1 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model in MODELS:
            values = reference["efficiency_means_percent"][operating_point][model]
            lines.append(
                f"| {model.upper()} | {_value(values['semi'])} | {_value(values['cap'])} | "
                f"{_value(values['stable'])} | {_value(values['wasserstein'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "### Paper epoch-level metric/downstream correlations",
            "",
            "| Model | Semi | CAP | Stable/drift | W1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    correlations = reference["epoch_metric_downstream_correlations"]["values"]
    for model in MODELS:
        values = correlations[model]
        lines.append(
            f"| {model.upper()} | {_value(values['semi'], 3)} | {_value(values['cap'], 3)} | "
            f"{_value(values['stable'], 3)} | {_value(values['wasserstein'], 3)} |"
        )
    lines.extend(
        [
            "",
            reference["epoch_metric_downstream_correlations"]["sign_convention"],
            "These are within-run correlations over common epoch steps, not evidence that a "
            "search-only winner improves downstream performance.",
            "",
        ]
    )
    return lines


def paper_comparison_markdown(rows: Iterable[dict[str, Any]], provisional: bool) -> list[str]:
    """Render a provisional guardrail or authenticated downstream deltas."""
    if provisional:
        return [
            "## Provisional comparison with the paper",
            "",
            "This post-freeze audit contains search objectives and Pareto identities only. "
            "It cannot establish downstream improvement over the paper: no retrained winner "
            "efficiency has been read, and differences between validation metrics are not "
            "surrogates for differences in anomaly efficiency.",
            "",
            "Direct deltas remain `downstream_results_not_available` in "
            "`analysis/paper_comparison.csv`. They may be populated only after authenticated "
            "200-epoch retraining and downstream evaluation.",
            "",
        ]
    comparable = [row for row in rows if row["comparison_status"] == "comparable"]
    return [
        "## Direct comparison with the paper",
        "",
        "The table compares only semantically matched native-score methods: campaign CDF CAP "
        "to paper CAP, forward drift to paper Stable, and W1 to paper W1. Deltas are campaign "
        "minus paper mean efficiency in percentage points.",
        "",
        "| Model | Method | Paper 250 Hz (%) | Campaign 250 Hz (%) | Delta (pp) |",
        "|---|---|---:|---:|---:|",
        *[
            f"| {row['model'].upper()} | {row['paper_method']} | "
            f"{_value(row['paper_mean_efficiency_percent'])} | "
            f"{_value(row['campaign_mean_efficiency_percent'])} | "
            f"{float(row['delta_percentage_points']):+.2f} |"
            for row in comparable
        ],
        "",
        "The campaign does not export a q99 downstream endpoint, so q99 deltas remain "
        "`campaign_q99_not_exported`; they are not inferred from 250 Hz efficiencies. The "
        "signal-informed paper Semi baseline has no signal-blind campaign equivalent. Full "
        "statuses, including explicit paper-missing values, are in "
        "`analysis/paper_comparison.csv`.",
        "",
    ]


__all__ = [
    "build_paper_comparison",
    "load_authenticated_paper_reference",
    "paper_comparison_markdown",
    "paper_reference_markdown",
]
