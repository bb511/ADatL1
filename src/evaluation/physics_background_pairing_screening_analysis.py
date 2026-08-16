"""Signal-blind convergence diagnostics for frozen background-pairing screens."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any, Iterable


def _dominates(left: tuple[float, float], right: tuple[float, float]) -> bool:
    """Return Pareto dominance for two maximize-oriented points."""
    return left[0] >= right[0] and left[1] >= right[1] and left != right


def _front(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all nondominated rows, retaining tied configurations."""
    candidates = list(rows)
    return [
        row
        for row in candidates
        if not any(
            other["trial_number"] != row["trial_number"]
            and _dominates(other["normalized"], row["normalized"])
            for other in candidates
        )
    ]


def _hypervolume(points: Iterable[tuple[float, float]]) -> float:
    """Return the union area dominated above the fixed origin reference."""
    height = 0.0
    area = 0.0
    for x_value, y_value in sorted(points, reverse=True):
        if y_value > height:
            area += x_value * (y_value - height)
            height = y_value
    return area


def _desirability(value: float, minimum: float, maximum: float, direction: str) -> float:
    """Map one objective to [0, 1], where larger is always preferable."""
    if maximum == minimum:
        return 1.0
    normalized = (value - minimum) / (maximum - minimum)
    return normalized if direction == "maximize" else 1.0 - normalized


def _trial_values(row: dict[str, Any]) -> list[float]:
    """Decode and validate one exported Optuna objective vector."""
    raw = row["values_json"]
    values = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(values, list) or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values
    ):
        raise ValueError(f"Trial {row.get('trial_number')} has invalid objective values.")
    return [float(value) for value in values]


def build_screening_convergence(
    design: dict[str, Any], trial_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Build generation-wise diagnostics from the frozen, signal-blind trial pool.

    Each logical objective is transformed independently. Both axes use min/max scaling
    over that logical objective's complete frozen trial pool; directions are flipped so
    one is always best and zero worst. A constant axis maps to one. Hypervolume is the
    area dominated by the cumulative two-dimensional front relative to the fixed
    normalized reference point ``(0, 0)``.
    """
    target = int(design["target_trials_per_study"])
    selected_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in trial_rows:
        if bool(row.get("selected_for_freeze")):
            selected_by_model.setdefault(str(row["model"]), []).append(row)

    convergence_rows: list[dict[str, Any]] = []
    evolution_rows: list[dict[str, Any]] = []
    for study in design["studies"]:
        model = str(study["model"])
        selected = sorted(
            selected_by_model.get(model, []), key=lambda row: int(row["trial_number"])
        )
        if len(selected) != target:
            raise ValueError(
                f"Expected {target} frozen trials for {model}, found {len(selected)}."
            )
        decoded = []
        for row in selected:
            generation = row.get("nsga_generation")
            if type(generation) is not int or generation < 0:
                raise ValueError(
                    f"Frozen trial {model}/{row.get('trial_number')} lacks an NSGA generation."
                )
            decoded.append((row, generation, _trial_values(row)))
        generations = sorted({generation for _, generation, _ in decoded})
        if generations != list(range(generations[-1] + 1)):
            raise ValueError(f"Frozen generations for {model} are not contiguous: {generations}.")

        for objective in study["objectives"]:
            indices = [int(value) for value in objective["value_indices"]]
            directions = list(objective["directions"])
            if (
                len(indices) != 2
                or len(directions) != 2
                or any(direction not in {"maximize", "minimize"} for direction in directions)
            ):
                raise ValueError(f"Logical objective {model}/{objective['id']} is not 2D.")
            raw_points = [(values[indices[0]], values[indices[1]]) for _, _, values in decoded]
            minima = tuple(min(point[axis] for point in raw_points) for axis in range(2))
            maxima = tuple(max(point[axis] for point in raw_points) for axis in range(2))
            points = []
            for (row, generation, _), raw in zip(decoded, raw_points, strict=True):
                normalized = tuple(
                    _desirability(raw[axis], minima[axis], maxima[axis], directions[axis])
                    for axis in range(2)
                )
                points.append(
                    {
                        "trial_number": int(row["trial_number"]),
                        "generation": generation,
                        "raw": raw,
                        "normalized": normalized,
                    }
                )

            generation_zero = [row for row in points if row["generation"] == 0]
            baseline_q_prime = max(row["normalized"][0] for row in generation_zero)
            baseline_q_double_prime = max(row["normalized"][1] for row in generation_zero)
            baseline_hypervolume = _hypervolume(
                row["normalized"] for row in _front(generation_zero)
            )
            previous_trials: set[int] | None = None
            for generation in generations:
                cumulative = [row for row in points if row["generation"] <= generation]
                front = sorted(_front(cumulative), key=lambda row: row["trial_number"])
                front_trials = {row["trial_number"] for row in front}
                if previous_trials is None:
                    entered = len(front_trials)
                    exited = 0
                    jaccard: float | str = ""
                else:
                    entered = len(front_trials - previous_trials)
                    exited = len(previous_trials - front_trials)
                    union = front_trials | previous_trials
                    jaccard = len(front_trials & previous_trials) / len(union)
                hypervolume = _hypervolume(row["normalized"] for row in front)
                q_prime_best = max(row["normalized"][0] for row in cumulative)
                q_double_prime_best = max(row["normalized"][1] for row in cumulative)
                common = {
                    "model": model,
                    "logical_objective": objective["id"],
                    "score": objective["score"],
                    "metric": objective["metric"],
                    "strategy": objective.get("strategy") or "",
                    "generation": generation,
                }
                convergence_rows.append(
                    {
                        **common,
                        "generation_trials": sum(
                            row["generation"] == generation for row in points
                        ),
                        "cumulative_trials": len(cumulative),
                        "cumulative_nondominated_size": len(front),
                        "front_entered": entered,
                        "front_exited": exited,
                        "previous_front_jaccard": jaccard,
                        "q_prime_best_raw": (
                            max(row["raw"][0] for row in cumulative)
                            if directions[0] == "maximize"
                            else min(row["raw"][0] for row in cumulative)
                        ),
                        "q_double_prime_best_raw": (
                            max(row["raw"][1] for row in cumulative)
                            if directions[1] == "maximize"
                            else min(row["raw"][1] for row in cumulative)
                        ),
                        "q_prime_best_normalized": q_prime_best,
                        "q_double_prime_best_normalized": q_double_prime_best,
                        "q_prime_endpoint_improvement": q_prime_best - baseline_q_prime,
                        "q_double_prime_endpoint_improvement": (
                            q_double_prime_best - baseline_q_double_prime
                        ),
                        "normalized_hypervolume": hypervolume,
                        "hypervolume_improvement": hypervolume - baseline_hypervolume,
                        "q_prime_pool_min": minima[0],
                        "q_prime_pool_max": maxima[0],
                        "q_double_prime_pool_min": minima[1],
                        "q_double_prime_pool_max": maxima[1],
                        "q_prime_direction": directions[0],
                        "q_double_prime_direction": directions[1],
                    }
                )
                evolution_rows.extend(
                    {
                        **common,
                        "trial_number": row["trial_number"],
                        "trial_generation": row["generation"],
                        "q_prime": row["raw"][0],
                        "q_double_prime": row["raw"][1],
                        "q_prime_normalized": row["normalized"][0],
                        "q_double_prime_normalized": row["normalized"][1],
                        "entered_since_previous": (
                            previous_trials is None or row["trial_number"] not in previous_trials
                        ),
                    }
                    for row in front
                )
                previous_trials = front_trials

    return {
        "search_convergence": convergence_rows,
        "front_evolution": evolution_rows,
        "method": {
            "schema_version": 1,
            "signal_blind": True,
            "population": "first target_trials_per_study usable trials named by the freeze",
            "scaling": (
                "Per logical objective and axis, min/max over the complete frozen trial pool; "
                "directions are transformed so 1 is observed-best and 0 observed-worst; "
                "a constant axis maps to 1."
            ),
            "hypervolume_reference": [0.0, 0.0],
            "hypervolume": "Union area dominated by the cumulative 2D front in normalized space.",
            "endpoint_improvement": "Best normalized endpoint minus generation-0 endpoint.",
            "front_churn": "Jaccard and entered/exited trial counts versus the preceding generation.",
            "downstream_outcomes_used": False,
        },
    }


def summarize_screening_convergence(
    analysis: dict[str, Any],
    *,
    final_memberships: int,
    unique_retrains: int,
    milestone_generations: tuple[int, ...] = (3, 6, 9, 11),
    numerical_tolerance: float = 1e-12,
    notable_threshold: float = 1e-4,
    material_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Summarize convergence without consulting downstream outcomes.

    Milestone fractions measure the share of each front's eventual normalized hypervolume gain
    already achieved at that generation. A front with no gain is assigned one because its initial
    front already equals its final front.
    """
    if not 0.0 <= numerical_tolerance < notable_threshold < material_threshold:
        raise ValueError("Convergence thresholds must be positive and strictly ordered.")
    if not 0 < unique_retrains <= final_memberships:
        raise ValueError("Unique retrains must lie between one and the front membership count.")
    rows = analysis.get("search_convergence", [])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["logical_objective"]))].append(row)
    if not grouped:
        raise ValueError("Cannot summarize empty screening convergence rows.")

    ordered: dict[tuple[str, str], list[dict[str, Any]]] = {}
    final_generations = set()
    trials_by_model: dict[str, int] = {}
    for key, front_rows in grouped.items():
        front_rows = sorted(front_rows, key=lambda row: int(row["generation"]))
        generations = [int(row["generation"]) for row in front_rows]
        if generations != list(range(generations[-1] + 1)):
            raise ValueError(f"Convergence generations are not contiguous for {key}.")
        hypervolumes = [float(row["normalized_hypervolume"]) for row in front_rows]
        if any(
            right + numerical_tolerance < left
            for left, right in zip(hypervolumes, hypervolumes[1:], strict=False)
        ):
            raise ValueError(f"Cumulative hypervolume decreases for {key}.")
        final_generations.add(generations[-1])
        final_trials = int(front_rows[-1]["cumulative_trials"])
        prior = trials_by_model.setdefault(key[0], final_trials)
        if prior != final_trials:
            raise ValueError(f"Logical fronts disagree on the frozen trial count for {key[0]}.")
        ordered[key] = front_rows
    if len(final_generations) != 1:
        raise ValueError("Logical fronts do not share one final generation.")
    final_generation = final_generations.pop()

    milestones = []
    for generation in milestone_generations:
        if generation > final_generation:
            continue
        fractions = []
        exact_final = 0
        for key, front_rows in ordered.items():
            row = front_rows[generation]
            baseline = float(front_rows[0]["normalized_hypervolume"])
            final = float(front_rows[-1]["normalized_hypervolume"])
            denominator = final - baseline
            fraction = (
                1.0
                if denominator <= numerical_tolerance
                else (float(row["normalized_hypervolume"]) - baseline) / denominator
            )
            fractions.append(min(1.0, max(0.0, fraction)))
            exact_final += abs(final - float(row["normalized_hypervolume"])) <= (
                numerical_tolerance
            )
        milestones.append(
            {
                "generation": generation,
                "median_fraction_of_final_gain": statistics.median(fractions),
                "fronts_at_final_hypervolume": exact_final,
                "fronts": len(fractions),
            }
        )

    changes = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, logical_objective), front_rows in ordered.items():
        delta = (
            0.0
            if len(front_rows) == 1
            else float(front_rows[-1]["normalized_hypervolume"])
            - float(front_rows[-2]["normalized_hypervolume"])
        )
        row = {
            "model": model,
            "logical_objective": logical_objective,
            "final_generation_hypervolume_gain": delta,
        }
        changes.append(row)
        by_model[model].append(row)
    model_rows = []
    for model in sorted(by_model):
        values = [row["final_generation_hypervolume_gain"] for row in by_model[model]]
        model_rows.append(
            {
                "model": model,
                "fronts": len(values),
                "numerically_changed": sum(value > numerical_tolerance for value in values),
                "notably_changed": sum(value > notable_threshold for value in values),
                "materially_changed": sum(value > material_threshold for value in values),
                "maximum_final_generation_gain": max(values),
            }
        )
    material_flags = sorted(
        (row for row in changes if row["final_generation_hypervolume_gain"] > material_threshold),
        key=lambda row: (
            -row["final_generation_hypervolume_gain"],
            row["model"],
            row["logical_objective"],
        ),
    )
    all_changes = [row["final_generation_hypervolume_gain"] for row in changes]
    return {
        "schema_version": 1,
        "signal_blind": True,
        "models": len(trials_by_model),
        "logical_fronts": len(ordered),
        "trials_per_model": trials_by_model,
        "total_frozen_trials": sum(trials_by_model.values()),
        "final_front_memberships": int(final_memberships),
        "unique_retrains": int(unique_retrains),
        "final_generation": final_generation,
        "milestones": milestones,
        "thresholds": {
            "numerical_tolerance": numerical_tolerance,
            "notable_hypervolume_gain": notable_threshold,
            "material_hypervolume_gain": material_threshold,
        },
        "final_generation_changes": {
            "numerically_changed_fronts": sum(
                value > numerical_tolerance for value in all_changes
            ),
            "notably_changed_fronts": sum(value > notable_threshold for value in all_changes),
            "materially_changed_fronts": sum(value > material_threshold for value in all_changes),
            "maximum_hypervolume_gain": max(all_changes),
        },
        "model_late_gain_summary": model_rows,
        "material_late_gain_flags": material_flags,
        "downstream_outcomes_used": False,
    }


def screening_findings_markdown(summary: dict[str, Any]) -> list[str]:
    """Render a compact findings section solely from a convergence summary."""
    trials = summary["trials_per_model"]
    trial_counts = sorted({int(value) for value in trials.values()})
    trial_description = (
        f"{trial_counts[0]} per model"
        if len(trial_counts) == 1
        else ", ".join(f"{model}={trials[model]}" for model in sorted(trials))
    )
    thresholds = summary["thresholds"]
    changes = summary["final_generation_changes"]
    lines = [
        "## Scientific findings from the frozen screen",
        "",
        "### Completeness",
        "",
        f"The authenticated pool contains {summary['total_frozen_trials']} frozen usable trials "
        "across "
        f"{summary['models']} models ({trial_description}), with "
        f"{summary['logical_fronts']} logical fronts and "
        f"{summary['final_front_memberships']} frozen front memberships deduplicated to "
        f"{summary['unique_retrains']} retraining configurations. All convergence "
        f"series reach generation {summary['final_generation']}.",
        "",
        "### Generation-wise convergence",
        "",
        "| Generation | Median share of eventual HV gain | Fronts already at final HV |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['generation']} | {100 * row['median_fraction_of_final_gain']:.2f}% | "
        f"{row['fronts_at_final_hypervolume']}/{row['fronts']} |"
        for row in summary["milestones"]
    )
    lines.extend(
        [
            "",
            f"From generation {summary['final_generation'] - 1} to the final generation, "
            f"{changes['numerically_changed_fronts']}/{summary['logical_fronts']} fronts changed "
            f"above the numerical tolerance `{thresholds['numerical_tolerance']:.0e}`, "
            f"{changes['notably_changed_fronts']} changed by more than "
            f"`{thresholds['notable_hypervolume_gain']:.0e}`, and "
            f"{changes['materially_changed_fronts']} changed by more than the material flag "
            f"`{thresholds['material_hypervolume_gain']:.0e}`. The largest final-generation "
            f"gain was {changes['maximum_hypervolume_gain']:.6g}.",
            "",
            "| Model | Fronts | Numerical changes | Notable changes | Material changes | Max gain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['model']} | {row['fronts']} | {row['numerically_changed']} | "
        f"{row['notably_changed']} | {row['materially_changed']} | "
        f"{row['maximum_final_generation_gain']:.6g} |"
        for row in summary["model_late_gain_summary"]
    )
    lines.extend(["", "Material late-gain flags:", ""])
    if summary["material_late_gain_flags"]:
        lines.extend(
            f"- `{row['model']} / {row['logical_objective']}`: "
            f"{row['final_generation_hypervolume_gain']:.6g}"
            for row in summary["material_late_gain_flags"]
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "### Interpretation guardrails",
            "",
            "- These diagnostics use search objectives and frozen front identities only; no "
            "downstream anomaly outcome is used.",
            "- Hypervolume is scaled independently within each logical front using the complete "
            "frozen pool. Its trajectory is comparable across generations of that front, not "
            "across models, metrics, scores, or pairing definitions.",
            "- A late-gain flag identifies continued search-front movement; it does not show "
            "better downstream anomaly performance or invalidate the frozen front.",
            "- Front size, endpoint movement, and hypervolume saturation are complementary "
            "diagnostics and are not model rankings.",
        ]
    )
    return lines


__all__ = [
    "build_screening_convergence",
    "screening_findings_markdown",
    "summarize_screening_convergence",
]
