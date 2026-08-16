"""Pure provisional analysis of frozen 50-epoch search checkpoints."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from src.evaluation.physics_background_pairing_analysis import DOWNSTREAM_DATASETS
from src.evaluation.physics_background_pairing_paper_comparison import (
    METHODS,
    MODELS,
    OPERATING_POINTS,
)

PROVISIONAL_NAMESPACE = "provisional_search50"
EXPECTED_MEMBERSHIPS = 591
EXPECTED_FRONTS = 56

STRATEGIES = (
    ("cap", "flat_physical"),
    ("cap", "physics_summary"),
    ("cap", "typed_sliced_wasserstein"),
    ("cap", "jetclr"),
    ("cap", "cdf"),
    ("wasserstein", ""),
    ("drift", ""),
)


def _expected_families() -> tuple[tuple[str, str], ...]:
    """Return the campaign's eight model/score families in stable order."""
    return tuple(
        (model, score)
        for model in MODELS
        for score in (("native", "residual_oas") if model in {"ae", "vae"} else ("native",))
    )


def _finite(value: Any, label: str) -> float:
    """Return a finite float or reject the labeled value."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}: {value!r}")
    return result


def _summary(values: Iterable[float]) -> dict[str, float]:
    """Compute the fixed downstream summary over exactly 20 anomalies."""
    sample = [_finite(value, "downstream efficiency") for value in values]
    if len(sample) != len(DOWNSTREAM_DATASETS) or any(
        value < 0.0 or value > 1.0 for value in sample
    ):
        raise ValueError("Expected exactly 20 downstream efficiencies in [0, 1].")
    ordered = sorted(sample)
    lower = ordered[: len(ordered) // 4]
    midpoint = len(ordered) // 2
    return {
        "mean_efficiency": statistics.fmean(sample),
        "median_efficiency": statistics.median(sample),
        "worst_efficiency": ordered[0],
        "bottom_quartile_cvar": statistics.fmean(lower),
        "population_std": statistics.pstdev(sample),
        "iqr": statistics.median(ordered[-midpoint:]) - statistics.median(ordered[:midpoint]),
    }


def _objective_id(score: str, metric: str, strategy: str) -> str:
    """Build one canonical logical-objective identifier."""
    suffix = f"{metric}__{strategy}" if strategy else metric
    return f"{score}__{suffix}"


def _validate_manifest(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Validate and index the exact frozen screening membership universe."""
    if (
        manifest.get("schema_version") != 2
        or manifest.get("selection_uses_downstream_anomalies") is not False
    ):
        raise ValueError("Expected a schema-2 signal-blind frozen manifest.")
    memberships = manifest.get("front_memberships", [])
    if len(memberships) != EXPECTED_MEMBERSHIPS:
        raise ValueError(
            f"Expected {EXPECTED_MEMBERSHIPS} frozen memberships, found {len(memberships)}."
        )
    indexed: dict[int, dict[str, Any]] = {}
    fronts = Counter()
    for membership in memberships:
        index = int(membership["membership_index"])
        if index in indexed:
            raise ValueError(f"Duplicate membership index: {index}.")
        model = str(membership["model"])
        score = str(membership["score"])
        metric = str(membership["metric"])
        strategy = str(membership.get("strategy") or "")
        if (model, score) not in _expected_families() or (metric, strategy) not in STRATEGIES:
            raise ValueError(f"Unknown provisional front: {(model, score, metric, strategy)}.")
        expected_id = _objective_id(score, metric, strategy)
        if membership.get("logical_objective") != expected_id:
            raise ValueError(f"Logical-objective identity differs for membership {index}.")
        indexed[index] = membership
        fronts[(model, score, metric, strategy)] += 1
    expected_fronts = {
        (model, score, metric, strategy)
        for model, score in _expected_families()
        for metric, strategy in STRATEGIES
    }
    if set(fronts) != expected_fronts or len(fronts) != EXPECTED_FRONTS:
        raise ValueError("Frozen memberships do not exactly cover the 56 logical fronts.")
    return indexed


def _candidate_summaries(
    memberships: dict[int, dict[str, Any]], efficiency_rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate expanded membership/anomaly rows and summarize every candidate."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_rows = []
    identity_fields = (
        "model",
        "logical_objective",
        "score",
        "metric",
        "strategy",
        "trial_number",
    )
    for supplied in efficiency_rows:
        row = dict(supplied)
        index = int(row["membership_index"])
        if index not in memberships:
            raise ValueError(f"Efficiency row names unknown membership {index}.")
        membership = memberships[index]
        for field in identity_fields:
            if field == "trial_number":
                expected, observed = int(membership[field]), int(row[field])
            else:
                expected = membership.get(field) or ""
                observed = row.get(field) or ""
            if observed != expected:
                raise ValueError(f"Efficiency identity {field} differs for membership {index}.")
        if not row.get("checkpoint") or not row.get("checkpoint_sha256"):
            raise ValueError(f"Missing checkpoint identity for membership {index}.")
        row["efficiency_operational"] = _finite(
            row["efficiency_operational"], "operational efficiency"
        )
        grouped[index].append(row)
        raw_rows.append(row)
    if set(grouped) != set(memberships):
        raise ValueError("Efficiency rows do not exactly cover every frozen membership.")

    front_sizes = Counter(
        (
            row["model"],
            row["score"],
            row["metric"],
            row.get("strategy") or "",
        )
        for row in memberships.values()
    )
    summaries = []
    for index in sorted(memberships):
        membership = memberships[index]
        rows = grouped[index]
        anomalies = [str(row["anomaly"]) for row in rows]
        if len(rows) != len(DOWNSTREAM_DATASETS) or set(anomalies) != set(DOWNSTREAM_DATASETS):
            raise ValueError(f"Membership {index} lacks exact 20-anomaly coverage.")
        if len(anomalies) != len(set(anomalies)):
            raise ValueError(f"Membership {index} has duplicate anomaly rows.")
        checkpoints = {(row["checkpoint"], row["checkpoint_sha256"]) for row in rows}
        if len(checkpoints) != 1:
            raise ValueError(f"Membership {index} has inconsistent checkpoint identity.")
        checkpoint, checkpoint_sha256 = checkpoints.pop()
        summary = _summary(row["efficiency_operational"] for row in rows)
        summaries.append(
            {
                "membership_index": index,
                "model": membership["model"],
                "logical_objective": membership["logical_objective"],
                "score": membership["score"],
                "metric": membership["metric"],
                "strategy": membership.get("strategy") or "",
                "trial_number": int(membership["trial_number"]),
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                **summary,
                "front_size": front_sizes[
                    (
                        membership["model"],
                        membership["score"],
                        membership["metric"],
                        membership.get("strategy") or "",
                    )
                ],
            }
        )
    return summaries, raw_rows


def _front_winners(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one deterministic downstream-mean oracle winner per frozen front."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["model"], row["score"], row["metric"], row["strategy"])].append(row)
    winners = [
        min(rows, key=lambda row: (-row["mean_efficiency"], row["membership_index"]))
        for rows in grouped.values()
    ]
    winners.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            0 if row["score"] == "native" else 1,
            STRATEGIES.index((row["metric"], row["strategy"])),
        )
    )
    if len(winners) != EXPECTED_FRONTS:
        raise ValueError("Expected one winner for each of 56 fronts.")
    return winners


def _cross_combination(
    winners: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank seven front winners and select mean/tail oracles in each family."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in winners:
        grouped[(row["model"], row["score"])].append(row)
    rankings, choices = [], []
    for family in _expected_families():
        rows = grouped[family]
        if len(rows) != len(STRATEGIES):
            raise ValueError(f"Expected seven front winners for {family}.")

        def order(row: dict[str, Any]) -> int:
            return STRATEGIES.index((row["metric"], row["strategy"]))

        mean_order = sorted(
            rows,
            key=lambda row: (
                -row["mean_efficiency"],
                -row["bottom_quartile_cvar"],
                -row["worst_efficiency"],
                order(row),
            ),
        )
        tail_order = sorted(
            rows,
            key=lambda row: (
                -row["bottom_quartile_cvar"],
                -row["mean_efficiency"],
                -row["worst_efficiency"],
                order(row),
            ),
        )
        mean_ranks = {row["logical_objective"]: rank for rank, row in enumerate(mean_order, 1)}
        tail_ranks = {row["logical_objective"]: rank for rank, row in enumerate(tail_order, 1)}
        for row in sorted(rows, key=order):
            rankings.append(
                {
                    **row,
                    "mean_oracle_rank": mean_ranks[row["logical_objective"]],
                    "tail_oracle_rank": tail_ranks[row["logical_objective"]],
                    "is_mean_oracle": row is mean_order[0],
                    "is_tail_oracle": row is tail_order[0],
                }
            )
        for selector, selected, rule in (
            (
                "mean_oracle",
                mean_order[0],
                "max mean; then bottom-quartile CVaR, worst, fixed objective order",
            ),
            (
                "tail_oracle",
                tail_order[0],
                "max bottom-quartile CVaR; then mean, worst, fixed objective order",
            ),
        ):
            choices.append({"selector": selector, "selection_rule": rule, **selected})
    if len(rankings) != EXPECTED_FRONTS or len(choices) != 16:
        raise ValueError("Cross-combination result coverage is incomplete.")
    return rankings, choices


def _paper_comparison(
    paper_reference: dict[str, Any], winners: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build honest provisional paper comparisons for matched native methods."""
    index = {
        (row["model"], row["metric"], row["strategy"]): row
        for row in winners
        if row["score"] == "native"
    }
    rows = []
    for operating_point in OPERATING_POINTS:
        for model in MODELS:
            for method in METHODS:
                paper = paper_reference["efficiency_means_percent"][operating_point][model][method]
                campaign_method = {
                    "cap": ("cap", "cdf", "native__cap__cdf"),
                    "stable": ("drift", "", "native__drift"),
                    "wasserstein": ("wasserstein", "", "native__wasserstein"),
                }.get(method)
                campaign, delta = None, None
                if paper is None:
                    status = "paper_value_missing"
                elif method == "semi":
                    status = "no_signal_blind_campaign_equivalent"
                elif operating_point == "q99":
                    status = "campaign_q99_not_exported"
                else:
                    assert campaign_method is not None
                    winner = index[(model, campaign_method[0], campaign_method[1])]
                    campaign = 100.0 * winner["mean_efficiency"]
                    delta = campaign - float(paper)
                    status = "provisional_50epoch_oracle_delta"
                rows.append(
                    {
                        "operating_point": operating_point,
                        "model": model,
                        "paper_method": method,
                        "campaign_method": "" if campaign_method is None else campaign_method[2],
                        "paper_mean_efficiency_percent": paper,
                        "campaign_mean_efficiency_percent": campaign,
                        "delta_percentage_points": delta,
                        "comparison_status": status,
                    }
                )
    if (
        len(rows) != 48
        or sum(row["comparison_status"] == "provisional_50epoch_oracle_delta" for row in rows)
        != 18
    ):
        raise ValueError("Provisional paper-comparison coverage is incomplete.")
    return rows


def build_provisional_search50_analysis(
    manifest: dict[str, Any],
    efficiency_rows: Iterable[dict[str, Any]],
    paper_reference: dict[str, Any],
) -> dict[str, Any]:
    """Build all deterministic provisional 50-epoch downstream analysis tables."""
    memberships = _validate_manifest(manifest)
    summaries, raw_rows = _candidate_summaries(memberships, efficiency_rows)
    winners = _front_winners(summaries)
    rankings, choices = _cross_combination(winners)
    paper = _paper_comparison(paper_reference, winners)
    return {
        "method": {
            "schema_version": 1,
            "namespace": PROVISIONAL_NAMESPACE,
            "training_epochs": 50,
            "candidate_universe_selected_without_downstream": True,
            "reported_choices_use_downstream_anomalies": True,
            "memberships": len(summaries),
            "logical_fronts": len(winners),
            "cross_combination_choices": len(choices),
        },
        "candidate_efficiencies": raw_rows,
        "candidate_summary": summaries,
        "front_winners": winners,
        "cross_combination_rankings": rankings,
        "cross_combination_choices": choices,
        "paper_comparison": paper,
    }


def render_provisional_search50_report(analysis: dict[str, Any]) -> str:
    """Render a self-contained Markdown report with mandatory bias caveats."""
    method = analysis["method"]
    choices = analysis["cross_combination_choices"]
    mean_choices = [row for row in choices if row["selector"] == "mean_oracle"]
    native_winners = [row for row in analysis["front_winners"] if row["score"] == "native"]
    native_by_strategy = defaultdict(list)
    for row in native_winners:
        native_by_strategy[(row["metric"], row["strategy"])].append(row["mean_efficiency"])
    native_strategy_means = {
        strategy: statistics.fmean(values) for strategy, values in native_by_strategy.items()
    }
    native_flat = native_strategy_means[("cap", "flat_physical")]
    native_cdf = native_strategy_means[("cap", "cdf")]
    ae_native = next(
        row for row in mean_choices if row["model"] == "ae" and row["score"] == "native"
    )
    ae_oas = next(
        row for row in mean_choices if row["model"] == "ae" and row["score"] == "residual_oas"
    )
    vae_native = next(
        row for row in mean_choices if row["model"] == "vae" and row["score"] == "native"
    )
    vae_oas = next(
        row for row in mean_choices if row["model"] == "vae" and row["score"] == "residual_oas"
    )
    strongest = max(mean_choices, key=lambda row: row["mean_efficiency"])
    choice_index = {(row["model"], row["score"], row["selector"]): row for row in choices}
    ae_native_tail = choice_index[("ae", "native", "tail_oracle")]
    dsvae_mean = choice_index[("dsvae", "native", "mean_oracle")]
    dsvae_tail = choice_index[("dsvae", "native", "tail_oracle")]
    realnvp_mean = choice_index[("realnvp", "native", "mean_oracle")]
    realnvp_tail = choice_index[("realnvp", "native", "tail_oracle")]
    paper = [
        row
        for row in analysis["paper_comparison"]
        if row["comparison_status"] == "provisional_50epoch_oracle_delta"
    ]
    lines = [
        "# Provisional 50-epoch search-checkpoint report",
        "",
        f"Namespace: `{method['namespace']}`. This analysis covers "
        f"{method['memberships']} frozen memberships and {method['logical_fronts']} logical "
        "fronts. It does not replace the 200-epoch retraining result.",
        f"The evaluation executed {method.get('physical_checkpoints', 421)} distinct checkpoint "
        f"files and {method.get('checkpoint_score_evaluations', 601)} checkpoint/score passes, "
        "using the frozen validation-derived operating thresholds. Their exact values and "
        "checkpoint hashes are exported in `checkpoint_thresholds.csv`.",
        f"Integrity status: `{method.get('integrity_status', 'pass')}` over "
        f"{method.get('authenticated_tasks', 346)} task markers and "
        f"{method.get('selected_score_anomaly_rows', 11820):,} selected-score anomaly rows.",
        "",
        "Each front winner maximizes mean operational efficiency over the same 20 downstream "
        "anomaly datasets, with exact ties resolved by the lowest frozen membership index. "
        "Cross-combination choices compare the seven front winners within each model/score "
        "family.",
        "",
        "## Descriptive readout",
        "",
        f"The strongest family-level mean choice is {strongest['model'].upper()} "
        f"{strongest['score']} with `{strongest['logical_objective']}` "
        f"({strongest['mean_efficiency']:.4f} mean efficiency). Across the six native-score "
        f"model families, flat-physical CAP averages {native_flat:.4f} across its front winners, "
        f"versus {native_cdf:.4f} for CDF CAP; this small aggregate difference is not uniform "
        "across models.",
        "",
        f"For AE, the best residual-OAS choice reaches {ae_oas['mean_efficiency']:.4f}, versus "
        f"{ae_native['mean_efficiency']:.4f} for the best native choice. For VAE the analogous "
        f"values are {vae_oas['mean_efficiency']:.4f} and {vae_native['mean_efficiency']:.4f}. "
        "Thus masked residual OAS does not beat the native score in this provisional oracle "
        "comparison, although it changes which selection metric is preferred.",
        "",
        f"As a practical shortlist, flat-physical CAP is the strongest broad default, while the "
        f"model-specific mean choices below should be retained. AE's native tail choice "
        f"(`{ae_native_tail['logical_objective']}`) gives up only "
        f"{ae_native['mean_efficiency'] - ae_native_tail['mean_efficiency']:.4f} mean efficiency. "
        f"By contrast, the DSVAE and RealNVP tail choices give up "
        f"{dsvae_mean['mean_efficiency'] - dsvae_tail['mean_efficiency']:.4f} and "
        f"{realnvp_mean['mean_efficiency'] - realnvp_tail['mean_efficiency']:.4f}; prefer their "
        "mean choices unless tail robustness is explicitly predeclared as primary.",
        "",
        "## Per-front 50-epoch downstream-oracle winners",
        "",
        "| Model | Score | Metric | Strategy | Membership | Trial | Mean | Tail CVaR |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['model'].upper()} | {row['score']} | {row['metric']} | "
        f"{row['strategy'] or '—'} | {row['membership_index']} | {row['trial_number']} | "
        f"{row['mean_efficiency']:.4f} | {row['bottom_quartile_cvar']:.4f} |"
        for row in analysis["front_winners"]
    )
    lines.extend(
        [
            "",
            "## Cross-combination exploratory choices",
            "",
            "| Model | Score | Selector | Logical objective | Mean | Bottom-quartile CVaR | Worst |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['model'].upper()} | {row['score']} | {row['selector']} | "
        f"{row['logical_objective']} | {row['mean_efficiency']:.4f} | "
        f"{row['bottom_quartile_cvar']:.4f} | {row['worst_efficiency']:.4f} |"
        for row in choices
    )
    lines.extend(
        [
            "",
            "## Provisional closest-form paper reference deltas at 250 Hz",
            "",
            "Native CDF-CAP, forward drift/Stable, and W1 are the closest metric-form/method-label "
            "matches. They are not like-for-like: validation domains, 50 versus 200 training "
            "epochs, shared high-dimensional NSGA-II versus independent two-objective M-TPE, and "
            "downstream-oracle selection differ. Deltas are campaign minus paper percentage points "
            "and positive values are not evidence of improvement.",
            "",
            "| Model | Method | Paper (%) | Search50 oracle (%) | Delta (pp) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['model'].upper()} | {row['paper_method']} | "
        f"{float(row['paper_mean_efficiency_percent']):.2f} | "
        f"{float(row['campaign_mean_efficiency_percent']):.2f} | "
        f"{float(row['delta_percentage_points']):+.2f} |"
        for row in paper
    )
    lines.extend(
        [
            "",
            "q99 is not exported and is not inferred. No campaign model-selection objective is "
            "equivalent to paper Semi; learned CAP mappings and residual-OAS have no direct "
            "paper match.",
            "",
            "## Interpretation and selection-bias limits",
            "",
            "The frozen candidate universe was selected without downstream anomalies, but every "
            "reported front winner is an oracle chosen on those 20 anomaly labels. Selecting "
            "cross-combination choices adds a second selection layer and further optimism.",
            "",
            "Fronts and checkpoints overlap, anomaly datasets are not independent replicates, "
            "and these 50-epoch checkpoints are not the paper-aligned 200-epoch retrained "
            "endpoint. Paper deltas are descriptive and provisional; they support no "
            "confirmatory confidence interval, p-value, or improvement claim.",
            "",
            "Using these outcomes to choose Stage-2 targets and then reporting Stage-2 on the "
            "same 20 anomalies is target-selection biased. Confirmation requires a preregistered "
            "shortlist or an untouched anomaly holdout.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "PROVISIONAL_NAMESPACE",
    "build_provisional_search50_analysis",
    "render_provisional_search50_report",
]
