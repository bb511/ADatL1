"""Signal-blind convergence diagnostics for frozen background-pairing screens."""

from __future__ import annotations

import json
import math
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


__all__ = ["build_screening_convergence"]
