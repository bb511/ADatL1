"""Pure deterministic analysis for the physics background-pairing campaign."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

MODEL_ORDER = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
CAP_STRATEGIES = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
    "jetclr",
    "cdf",
)
MAPPING_STRATEGIES = CAP_STRATEGIES[:-1]
DOWNSTREAM_DATASETS = (
    "GluGluHTo2B_Par-MH-125",
    "GluGluHto2G_Par-MH-125",
    "GluGluHto2G_Par-MH-90",
    "GluGluHto2Tau_Par-MH-125",
    "GluGlutoHHto2B2WtoLNu2Q_Par-c2-0-kl-1-kt-1",
    "HHHto4B2Tau_Par-c3-0-d4-0",
    "HHHto6B_Par-c3-0-d4-0",
    "SUSYGluGluToBBHTo2B_Par-M-1200",
    "SUSYGluGluToBBHToBB_Par-M-120",
    "SUSYGluGluToBBHToBB_Par-M-350",
    "SUSYGluGluToBBHToBB_Par-M-600",
    "TTHTo2C_Par-MH-125",
    "TTHto2B_Par-MH-125",
    "VBFHTo2C_Par-MH-125",
    "VBFHto2B_Par-MH-125",
    "VBFHto2Tau_Par-MH-125",
    "WtoTauto3Mu",
    "ggH-suep-decay",
    "smj-case-A",
    "haa-4b-ma15",
)


def _finite(value: Any, label: str) -> float:
    """Return a finite float or reject the labeled value."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}: {value!r}")
    return result


def _stats(values: Iterable[float]) -> dict[str, float]:
    """Compute the campaign's fixed downstream robustness statistics."""
    sample = [_finite(value, "analysis value") for value in values]
    if not sample:
        raise ValueError("Cannot summarize an empty value set.")
    ordered = sorted(sample)
    lower = ordered[: max(1, len(ordered) // 4)]
    midpoint = len(ordered) // 2
    lower_half = ordered[:midpoint]
    upper_half = ordered[-midpoint:] if midpoint else ordered
    return {
        "mean": statistics.fmean(sample),
        "median": statistics.median(sample),
        "worst": min(sample),
        "bottom_quartile_cvar": statistics.fmean(lower),
        "population_std": statistics.pstdev(sample),
        "iqr": statistics.median(upper_half) - statistics.median(lower_half),
    }


def _objective_index(design: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index and validate the 56 logical objectives in the campaign design."""
    if tuple(design.get("models", ())) != MODEL_ORDER:
        raise ValueError("Expected the six campaign models in canonical order.")
    index = {}
    for study in design["studies"]:
        for objective in study["objectives"]:
            key = (study["model"], objective["id"])
            if key in index:
                raise ValueError(f"Duplicate logical objective: {key}")
            index[key] = objective
    if len(index) != 56:
        raise ValueError(f"Expected 56 logical objectives, found {len(index)}.")
    return index


def _efficiencies(candidate: dict[str, Any]) -> dict[str, float]:
    """Validate and canonically order a candidate's 20 anomaly efficiencies."""
    raw = candidate.get("downstream_efficiencies", {})
    if set(raw) != set(DOWNSTREAM_DATASETS):
        raise ValueError("Candidate does not contain the exact 20-anomaly dataset set.")
    values = {name: _finite(raw[name], f"efficiency for {name}") for name in DOWNSTREAM_DATASETS}
    if any(value < 0.0 or value > 1.0 for value in values.values()):
        raise ValueError("Downstream efficiencies must lie in [0, 1].")
    return values


def _family(checkpoint: str, supplied: str | None = None) -> str:
    """Validate an authenticated family against an exact checkpoint identity."""
    if supplied not in {"last", "primary_ema", "stable_native"}:
        raise ValueError(f"Missing or unknown checkpoint family: {supplied}")
    if supplied == "last" and checkpoint != "last":
        raise ValueError(f"Checkpoint {checkpoint!r} is not the last checkpoint.")
    if supplied == "stable_native" and checkpoint != "single/ascore_operational/stable":
        raise ValueError(f"Checkpoint {checkpoint!r} is not the stable-native checkpoint.")
    if supplied == "primary_ema":
        parts = checkpoint.split("/")
        if len(parts) != 3 or parts[0] != "summary" or parts[2] not in {"min", "max"}:
            raise ValueError(f"Checkpoint {checkpoint!r} is not a primary-EMA checkpoint.")
    return supplied


def _candidate_rows(
    objectives: dict[tuple[str, str], dict[str, Any]],
    manifest: dict[str, Any],
    oracle: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join frozen memberships to oracle results and normalize candidate rows."""
    memberships = {int(row["membership_index"]): row for row in manifest["front_memberships"]}
    candidates = {int(row["membership_index"]): row for row in oracle["all_front_memberships"]}
    if len(memberships) != len(manifest["front_memberships"]) or set(candidates) != set(
        memberships
    ):
        raise ValueError("Oracle candidates do not exactly cover unique frozen memberships.")
    winner_indices = {int(row["membership_index"]) for row in oracle["winners"]}
    if len(winner_indices) != 56:
        raise ValueError("Expected exactly one winner for each of 56 fronts.")
    front_counts = Counter(
        (row["model"], row["logical_objective"]) for row in memberships.values()
    )
    rows = []
    for index in sorted(memberships):
        membership = memberships[index]
        candidate = candidates[index]
        if any(candidate.get(key) != value for key, value in membership.items()):
            raise ValueError(f"Candidate differs from frozen membership {index}.")
        objective = objectives[(membership["model"], membership["logical_objective"])]
        search_pair = [
            _finite(value, "search objective") for value in membership["objective_values"]
        ]
        retrained_pair = [
            _finite(value, "retrained objective")
            for value in candidate["retrained_objective_values"]
        ]
        if len(search_pair) != 2 or len(retrained_pair) != 2:
            raise ValueError("Every logical objective must contain exactly two values.")
        efficiencies = _efficiencies(candidate)
        summary = _stats(efficiencies.values())
        recorded_mean = _finite(candidate["mean_downstream_efficiency"], "candidate mean")
        if not math.isclose(recorded_mean, summary["mean"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"Candidate {index} has an incorrect downstream mean.")
        rows.append(
            {
                "membership_index": index,
                "model": membership["model"],
                "logical_objective": membership["logical_objective"],
                "score": membership["score"],
                "metric": membership["metric"],
                "strategy": membership.get("strategy") or "",
                "trial_number": int(membership["trial_number"]),
                "retrain_index": int(membership["retrain_index"]),
                "checkpoint": candidate["optimized_ckpt_name"],
                "search_q_prime": search_pair[0],
                "search_q_double_prime": search_pair[1],
                "retrained_q_prime": retrained_pair[0],
                "retrained_q_double_prime": retrained_pair[1],
                "q_prime_direction": objective["directions"][0],
                "q_double_prime_direction": objective["directions"][1],
                **{f"downstream_{key}": value for key, value in summary.items()},
                "front_size": front_counts[(membership["model"], membership["logical_objective"])],
                "is_winner": index in winner_indices,
                "downstream_efficiencies": efficiencies,
            }
        )
    if sum(row["is_winner"] for row in rows) != 56:
        raise ValueError("Winner membership indices do not cover exactly 56 candidates.")
    return rows


def _authenticate_candidate_checkpoints(
    candidates: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]
) -> None:
    """Attach authenticated retrain checkpoint families to every front candidate."""
    supplied = {}
    for row in checkpoint_rows:
        if row.get("phase") != "retrain":
            continue
        key = (row["model"], int(row["retrain_index"]), row["logical_objective"])
        if key in supplied:
            raise ValueError(f"Duplicate retrain checkpoint identity: {key}")
        checkpoint = str(row.get("checkpoint", row.get("optimized_ckpt_name")))
        _family(checkpoint, row.get("checkpoint_family"))
        supplied[key] = row
    expected = {
        (row["model"], row["retrain_index"], row["logical_objective"]) for row in candidates
    }
    missing = expected - set(supplied)
    if missing:
        raise ValueError("Checkpoint rows do not cover every retrain-candidate selection.")
    for candidate in candidates:
        key = (
            candidate["model"],
            candidate["retrain_index"],
            candidate["logical_objective"],
        )
        checkpoint_row = supplied[key]
        checkpoint = str(
            checkpoint_row.get("checkpoint", checkpoint_row.get("optimized_ckpt_name"))
        )
        if checkpoint != candidate["checkpoint"]:
            raise ValueError(f"Checkpoint identity differs for retrain candidate {key}.")
        candidate["checkpoint_family"] = _family(
            checkpoint, checkpoint_row.get("checkpoint_family")
        )


def _front_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize candidate ranges, winners, and checkpoint families per front."""
    grouped = _group(rows, "model", "logical_objective")
    output = []
    for key in _front_order(grouped):
        members = grouped[key]
        winners = [row for row in members if row["is_winner"]]
        if len(winners) != 1:
            raise ValueError(f"Expected one winner for front {key}.")
        winner = winners[0]
        families = Counter(row["checkpoint_family"] for row in members)
        output.append(
            {
                "model": key[0],
                "logical_objective": key[1],
                "score": winner["score"],
                "metric": winner["metric"],
                "strategy": winner["strategy"],
                "front_size": len(members),
                "unique_trials": len({row["trial_number"] for row in members}),
                "winner_membership_index": winner["membership_index"],
                "winner_trial_number": winner["trial_number"],
                "search_q_prime_min": min(row["search_q_prime"] for row in members),
                "search_q_prime_max": max(row["search_q_prime"] for row in members),
                "search_q_double_prime_min": min(row["search_q_double_prime"] for row in members),
                "search_q_double_prime_max": max(row["search_q_double_prime"] for row in members),
                "last_count": families["last"],
                "primary_ema_count": families["primary_ema"],
                "stable_native_count": families["stable_native"],
            }
        )
    if len(output) != 56:
        raise ValueError("Expected exactly 56 front summaries.")
    return output


def _group(
    rows: Iterable[dict[str, Any]], *fields: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Group rows by an ordered tuple of named fields."""
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return dict(result)


def _front_order(grouped: dict[tuple[str, str], Any]) -> list[tuple[str, str]]:
    """Return front keys in stable campaign model and objective order."""
    return sorted(grouped, key=lambda key: (MODEL_ORDER.index(key[0]), key[1]))


def _front_overlaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure within-model trial overlap for every pair of logical fronts."""
    grouped = _group(rows, "model", "logical_objective")
    output = []
    by_model: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in grouped:
        by_model[key[0]].append(key)
    for model in MODEL_ORDER:
        fronts = sorted(by_model[model], key=lambda key: key[1])
        for left_index, left_key in enumerate(fronts):
            for right_key in fronts[left_index + 1 :]:
                left = grouped[left_key]
                right = grouped[right_key]
                left_trials = {row["trial_number"] for row in left}
                right_trials = {row["trial_number"] for row in right}
                intersection = left_trials & right_trials
                union = left_trials | right_trials
                left_meta, right_meta = left[0], right[0]
                if (
                    left_meta["score"] != right_meta["score"]
                    and left_meta["metric"] == right_meta["metric"]
                    and left_meta["strategy"] == right_meta["strategy"]
                ):
                    comparison = "score"
                elif left_meta["score"] != right_meta["score"]:
                    comparison = "cross_score_unmatched"
                elif left_meta["metric"] == right_meta["metric"] == "cap":
                    comparison = "pairing"
                else:
                    comparison = "metric"
                output.append(
                    {
                        "model": model,
                        "left_objective": left_key[1],
                        "right_objective": right_key[1],
                        "comparison_class": comparison,
                        "intersection": len(intersection),
                        "union": len(union),
                        "jaccard": len(intersection) / len(union),
                        "same_winner": next(row for row in left if row["is_winner"])[
                            "trial_number"
                        ]
                        == next(row for row in right if row["is_winner"])["trial_number"],
                    }
                )
    return output


def _winner_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Index the unique winner for every model, score, metric, and strategy."""
    winners = [row for row in rows if row["is_winner"]]
    index = {(row["model"], row["score"], row["metric"], row["strategy"]): row for row in winners}
    if len(index) != 56:
        raise ValueError("Winner keys are not unique across the 56 logical fronts.")
    return index


def _delta_row(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compute left-minus-right deltas for all downstream summary statistics."""
    return {
        key: left[f"downstream_{key}"] - right[f"downstream_{key}"]
        for key in ("mean", "median", "worst", "bottom_quartile_cvar", "population_std", "iqr")
    }


def _mapping_comparisons(
    winners: dict[tuple[str, str, str, str], dict[str, Any]],
    overlaps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare each learned mapping strategy with its matched CDF control."""
    comparisons, anomaly_rows = [], []
    overlap_index = {
        (row["model"], frozenset((row["left_objective"], row["right_objective"]))): row
        for row in overlaps
        if row["comparison_class"] == "pairing"
    }
    for model in MODEL_ORDER:
        scores = ("native", "residual_oas") if model in {"ae", "vae"} else ("native",)
        for score in scores:
            control = winners[(model, score, "cap", "cdf")]
            for strategy in MAPPING_STRATEGIES:
                mapping = winners[(model, score, "cap", strategy)]
                overlap = overlap_index[
                    (
                        model,
                        frozenset((mapping["logical_objective"], control["logical_objective"])),
                    )
                ]
                deltas = {
                    name: mapping["downstream_efficiencies"][name]
                    - control["downstream_efficiencies"][name]
                    for name in DOWNSTREAM_DATASETS
                }
                signs = Counter(
                    "improved" if value > 0 else "worse" if value < 0 else "tied"
                    for value in deltas.values()
                )
                comparisons.append(
                    {
                        "model": model,
                        "score": score,
                        "mapping_strategy": strategy,
                        "mapping_trial": mapping["trial_number"],
                        "cdf_trial": control["trial_number"],
                        "front_intersection": overlap["intersection"],
                        "front_union": overlap["union"],
                        "front_jaccard": overlap["jaccard"],
                        "same_winner": overlap["same_winner"],
                        **{
                            f"delta_{key}": value
                            for key, value in _delta_row(mapping, control).items()
                        },
                        "anomalies_improved": signs["improved"],
                        "anomalies_tied": signs["tied"],
                        "anomalies_worse": signs["worse"],
                    }
                )
                anomaly_rows.extend(
                    {
                        "model": model,
                        "score": score,
                        "mapping_strategy": strategy,
                        "anomaly": anomaly,
                        "mapping_minus_cdf": deltas[anomaly],
                    }
                    for anomaly in DOWNSTREAM_DATASETS
                )
    summary = []
    grouped = _group(comparisons, "mapping_strategy")
    for strategy in MAPPING_STRATEGIES:
        values = [row["delta_mean"] for row in grouped[(strategy,)]]
        summary.append(
            {
                "mapping_strategy": strategy,
                "groups": len(values),
                "mean_delta": statistics.fmean(values),
                "median_delta": statistics.median(values),
                "groups_improved": sum(value > 0 for value in values),
                "groups_tied": sum(value == 0 for value in values),
                "groups_worse": sum(value < 0 for value in values),
            }
        )
    if len(comparisons) != 32 or len(anomaly_rows) != 640:
        raise ValueError("Mapping comparison coverage is incomplete.")
    return comparisons, anomaly_rows, summary


def _method_name(row: dict[str, Any]) -> str:
    """Return a stable display identifier for a model-selection method."""
    return f"cap__{row['strategy']}" if row["metric"] == "cap" else row["metric"]


def _metric_comparisons(
    winners: dict[tuple[str, str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build matched CAP, Wasserstein, and drift method contrasts."""
    methods, contrasts, anomaly_rows = [], [], []
    for model in MODEL_ORDER:
        scores = ("native", "residual_oas") if model in {"ae", "vae"} else ("native",)
        for score in scores:
            group = [winners[(model, score, "cap", strategy)] for strategy in CAP_STRATEGIES]
            group.extend(
                (winners[(model, score, "wasserstein", "")], winners[(model, score, "drift", "")])
            )
            methods.extend(
                {
                    "model": model,
                    "score": score,
                    "method": _method_name(row),
                    "trial_number": row["trial_number"],
                    **{
                        key.removeprefix("downstream_"): value
                        for key, value in row.items()
                        if key.startswith("downstream_") and key != "downstream_efficiencies"
                    },
                }
                for row in group
            )
            w1, drift = group[-2], group[-1]
            pairs = (
                [(row, w1) for row in group[:5]]
                + [(row, drift) for row in group[:5]]
                + [(w1, drift)]
            )
            for left, right in pairs:
                contrast_id = len(contrasts)
                contrasts.append(
                    {
                        "contrast_index": contrast_id,
                        "model": model,
                        "score": score,
                        "left_method": _method_name(left),
                        "right_method": _method_name(right),
                        **{
                            f"delta_{key}": value for key, value in _delta_row(left, right).items()
                        },
                    }
                )
                anomaly_rows.extend(
                    {
                        "contrast_index": contrast_id,
                        "model": model,
                        "score": score,
                        "left_method": _method_name(left),
                        "right_method": _method_name(right),
                        "anomaly": anomaly,
                        "efficiency_delta": left["downstream_efficiencies"][anomaly]
                        - right["downstream_efficiencies"][anomaly],
                    }
                    for anomaly in DOWNSTREAM_DATASETS
                )
    if len(methods) != 56 or len(contrasts) != 88 or len(anomaly_rows) != 1760:
        raise ValueError("Metric comparison coverage is incomplete.")
    return methods, contrasts, anomaly_rows


def _score_comparisons(
    winners: dict[tuple[str, str, str, str], dict[str, Any]],
    overlaps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare native and residual-OAS winners for AE and VAE methods."""
    overlap_index = {
        (row["model"], frozenset((row["left_objective"], row["right_objective"]))): row
        for row in overlaps
        if row["comparison_class"] == "score"
    }
    output, anomaly_rows = [], []
    for model in ("ae", "vae"):
        methods = [("cap", strategy) for strategy in CAP_STRATEGIES] + [
            ("wasserstein", ""),
            ("drift", ""),
        ]
        for metric, strategy in methods:
            native = winners[(model, "native", metric, strategy)]
            oas = winners[(model, "residual_oas", metric, strategy)]
            overlap = overlap_index[
                (model, frozenset((native["logical_objective"], oas["logical_objective"])))
            ]
            deltas = {
                anomaly: oas["downstream_efficiencies"][anomaly]
                - native["downstream_efficiencies"][anomaly]
                for anomaly in DOWNSTREAM_DATASETS
            }
            signs = Counter(
                "improved" if value > 0 else "worse" if value < 0 else "tied"
                for value in deltas.values()
            )
            output.append(
                {
                    "model": model,
                    "metric": metric,
                    "strategy": strategy,
                    "native_trial": native["trial_number"],
                    "oas_trial": oas["trial_number"],
                    "same_winner": native["trial_number"] == oas["trial_number"],
                    "front_intersection": overlap["intersection"],
                    "front_union": overlap["union"],
                    "front_jaccard": overlap["jaccard"],
                    **{f"delta_{key}": value for key, value in _delta_row(oas, native).items()},
                    "anomalies_improved": signs["improved"],
                    "anomalies_tied": signs["tied"],
                    "anomalies_worse": signs["worse"],
                }
            )
            anomaly_rows.extend(
                {
                    "model": model,
                    "metric": metric,
                    "strategy": strategy,
                    "anomaly": anomaly,
                    "oas_minus_native": deltas[anomaly],
                }
                for anomaly in DOWNSTREAM_DATASETS
            )
    if len(output) != 14 or len(anomaly_rows) != 280:
        raise ValueError("Native/OAS matched comparison coverage is incomplete.")
    return output, anomaly_rows


def _model_summaries(
    winners: dict[tuple[str, str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate native-score outcomes fairly across all six models."""
    summaries, anomaly_rows = [], []
    for model in MODEL_ORDER:
        native = [row for key, row in winners.items() if key[0] == model and key[1] == "native"]
        if len(native) != 7:
            raise ValueError(f"Expected seven native fronts for {model}.")
        pooled = [
            row["downstream_efficiencies"][name] for row in native for name in DOWNSTREAM_DATASETS
        ]
        summaries.append({"model": model, "fronts": 7, **_stats(pooled)})
        for anomaly in DOWNSTREAM_DATASETS:
            values = [row["downstream_efficiencies"][anomaly] for row in native]
            anomaly_rows.append(
                {"model": model, "anomaly": anomaly, "fronts": 7, **_stats(values)}
            )
    if len(anomaly_rows) != 120:
        raise ValueError("Native-only model/anomaly coverage is incomplete.")
    return summaries, anomaly_rows


def _ranks(values: list[float]) -> tuple[list[float], bool, bool]:
    """Return deterministic average ranks plus constant and tie flags."""
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[value].append(index)
    ranks = [0.0] * len(values)
    cursor = 1
    for value in sorted(grouped):
        indices = grouped[value]
        average = (cursor + cursor + len(indices) - 1) / 2.0
        for index in indices:
            ranks[index] = average
        cursor += len(indices)
    return ranks, len(grouped) == 1, any(len(indices) > 1 for indices in grouped.values())


def _spearman(left: list[float], right: list[float]) -> tuple[float | None, bool, bool]:
    """Compute tie-aware Spearman correlation without optional dependencies."""
    left_ranks, left_constant, left_ties = _ranks(left)
    right_ranks, right_constant, right_ties = _ranks(right)
    if left_constant or right_constant:
        return None, left_constant or right_constant, left_ties or right_ties
    left_mean, right_mean = statistics.fmean(left_ranks), statistics.fmean(right_ranks)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_ranks, right_ranks))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left_ranks)
        * sum((y - right_mean) ** 2 for y in right_ranks)
    )
    return numerator / denominator, False, left_ties or right_ties


def _proxy_associations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Relate direction-normalized proxy objectives to downstream performance."""
    grouped = _group(rows, "model", "logical_objective")
    output = []
    for key in _front_order(grouped):
        members = grouped[key]
        downstream = [row["downstream_mean"] for row in members]
        for source, field, direction_field in (
            ("search_q_prime", "search_q_prime", "q_prime_direction"),
            ("search_q_double_prime", "search_q_double_prime", "q_double_prime_direction"),
            ("retrained_q_prime", "retrained_q_prime", "q_prime_direction"),
            ("retrained_q_double_prime", "retrained_q_double_prime", "q_double_prime_direction"),
        ):
            direction = members[0][direction_field]
            sign = 1.0 if direction == "maximize" else -1.0
            desirability = [sign * row[field] for row in members]
            rho, constant, ties = _spearman(desirability, downstream)
            output.append(
                {
                    "model": key[0],
                    "logical_objective": key[1],
                    "source": source,
                    "direction": direction,
                    "n": len(members),
                    "spearman_rho": rho,
                    "constant": constant,
                    "ties": ties,
                }
            )
    if len(output) != 224:
        raise ValueError("Proxy-association coverage is incomplete.")
    return output


def _checkpoint_summary(
    candidates: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Summarize checkpoint-family frequencies across three populations."""
    populations: list[tuple[str, list[dict[str, Any]]]] = [
        ("search", [row for row in checkpoint_rows if row.get("phase") == "search"]),
        ("retrain_candidate", candidates),
        ("winner", [row for row in candidates if row["is_winner"]]),
    ]
    output = []
    for population, rows in populations:
        normalized = []
        for row in rows:
            checkpoint = row.get("checkpoint", row.get("optimized_ckpt_name"))
            if checkpoint is None:
                raise ValueError(f"Missing checkpoint identity in {population} row.")
            normalized.append(
                {
                    "model": row["model"],
                    "score": row["score"],
                    "metric": row["metric"],
                    "strategy": row.get("strategy") or "",
                    "family": _family(str(checkpoint), row.get("checkpoint_family")),
                }
            )
        grouped = _group(normalized, "model", "score", "metric", "strategy")
        for key in sorted(grouped, key=lambda item: (MODEL_ORDER.index(item[0]), item[1:])):
            denominator = len(grouped[key])
            counts = Counter(row["family"] for row in grouped[key])
            for family in ("last", "primary_ema", "stable_native"):
                output.append(
                    {
                        "population": population,
                        "model": key[0],
                        "score": key[1],
                        "metric": key[2],
                        "strategy": key[3],
                        "family": family,
                        "count": counts[family],
                        "denominator": denominator,
                        "fraction": counts[family] / denominator,
                    }
                )
    return output


def _drift_rows(
    winners: dict[tuple[str, str, str, str], dict[str, Any]],
    primary_metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare held-out forward and reverse drift for all drift winners."""
    metrics: dict[tuple[str, str, str], float] = {}
    for row in primary_metric_rows:
        name = row.get("metric_name")
        if name not in {"drift_test", "reverse_drift_posthoc"}:
            continue
        key = (row["model"], row["score"], name)
        if key in metrics:
            raise ValueError(f"Duplicate held-out drift metric: {key}")
        metrics[key] = _finite(row["value"], name)
    output = []
    for model in MODEL_ORDER:
        scores = ("native", "residual_oas") if model in {"ae", "vae"} else ("native",)
        for score in scores:
            winner = winners[(model, score, "drift", "")]
            forward = metrics[(model, score, "drift_test")]
            reverse = metrics[(model, score, "reverse_drift_posthoc")]
            output.append(
                {
                    "model": model,
                    "score": score,
                    "trial_number": winner["trial_number"],
                    "validation_forward_drift": winner["retrained_q_prime"],
                    "test_forward_drift": forward,
                    "test_reverse_drift": reverse,
                    "reverse_minus_forward": reverse - forward,
                    "absolute_asymmetry": abs(reverse - forward),
                }
            )
    if len(output) != 8:
        raise ValueError("Expected eight forward/reverse drift rows.")
    return output


def build_physics_background_pairing_analysis(
    design: dict[str, Any],
    manifest: dict[str, Any],
    oracle: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
    primary_metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build all deterministic scientific-analysis rows from authenticated inputs.

    This function performs no filesystem access. Its callers remain responsible for authenticating
    the supplied campaign artifacts before invoking it.
    """
    objectives = _objective_index(design)
    candidates = _candidate_rows(objectives, manifest, oracle)
    _authenticate_candidate_checkpoints(candidates, checkpoint_rows)
    fronts = _front_summaries(candidates)
    overlaps = _front_overlaps(candidates)
    winners = _winner_index(candidates)
    mapping, mapping_anomalies, mapping_summary = _mapping_comparisons(winners, overlaps)
    metric_methods, metric_contrasts, metric_anomalies = _metric_comparisons(winners)
    score, score_anomalies = _score_comparisons(winners, overlaps)
    model, model_anomalies = _model_summaries(winners)
    proxy = _proxy_associations(candidates)
    checkpoints = _checkpoint_summary(candidates, checkpoint_rows)
    drift = _drift_rows(winners, primary_metric_rows)
    summary = {
        "schema_version": 1,
        "front_candidates": len(candidates),
        "fronts": len(fronts),
        "unique_retrains": len({(row["model"], row["trial_number"]) for row in candidates}),
        "mapping_comparisons": len(mapping),
        "mapping_anomaly_deltas": len(mapping_anomalies),
        "metric_methods": len(metric_methods),
        "metric_contrasts": len(metric_contrasts),
        "metric_anomaly_deltas": len(metric_anomalies),
        "score_comparisons": len(score),
        "score_anomaly_deltas": len(score_anomalies),
        "model_anomaly_rows": len(model_anomalies),
        "proxy_associations": len(proxy),
        "drift_directionality_rows": len(drift),
        "mapping_mean_delta_by_strategy": {
            row["mapping_strategy"]: row["mean_delta"] for row in mapping_summary
        },
        "oas_mean_delta_by_model": {
            model_name: statistics.fmean(
                row["delta_mean"] for row in score if row["model"] == model_name
            )
            for model_name in ("ae", "vae")
        },
    }
    return {
        "front_candidates": candidates,
        "front_summaries": fronts,
        "front_overlaps": overlaps,
        "mapping_comparisons": mapping,
        "mapping_anomaly_deltas": mapping_anomalies,
        "mapping_summary": mapping_summary,
        "metric_methods": metric_methods,
        "metric_contrasts": metric_contrasts,
        "metric_anomaly_deltas": metric_anomalies,
        "score_comparisons": score,
        "score_anomaly_deltas": score_anomalies,
        "model_native_summary": model,
        "model_anomaly_performance": model_anomalies,
        "checkpoint_family_summary": checkpoints,
        "proxy_associations": proxy,
        "drift_directionality": drift,
        "analysis_summary": summary,
    }


__all__ = [
    "CAP_STRATEGIES",
    "DOWNSTREAM_DATASETS",
    "MODEL_ORDER",
    "build_physics_background_pairing_analysis",
]
