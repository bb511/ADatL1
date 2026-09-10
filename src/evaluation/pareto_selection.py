"""Lean, deterministic Phase 3 Pareto-front selection.

This module deliberately consumes only the configuration-level CSV or Parquet
table written by :mod:`src.evaluation.pareto_aggregation`.  It neither reads
individual run artifacts nor reruns any metric, so validation-set selection is
kept separate from training and final-test evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PARETO_SELECTION_SCHEMA_VERSION = 1

CONFIGURATION_ID_COLUMN = "configuration_id"
VALID_COLUMN = "configuration_valid"
FEASIBLE_COLUMN = "feasible"
REJECTION_REASONS_COLUMN = "rejection_reasons"

LEAKAGE_COLUMN = "leakage_worst_mean"
CORRELATION_COLUMN = "residual_correlation_mean"
EFFICIENCY_COLUMN = "median_efficiency_mean"

OBJECTIVES = (
    ("leakage", LEAKAGE_COLUMN, "minimize"),
    ("residual_correlation", CORRELATION_COLUMN, "minimize"),
    ("median_efficiency", EFFICIENCY_COLUMN, "maximize"),
)

_REQUIRED_COLUMNS = {
    "study_id",
    "protocol_version",
    CONFIGURATION_ID_COLUMN,
    VALID_COLUMN,
    FEASIBLE_COLUMN,
    REJECTION_REASONS_COLUMN,
    *(
        column
        for _, mean_column, _ in OBJECTIVES
        for column in (
            mean_column,
            mean_column.replace("_mean", "_ci95_low"),
            mean_column.replace("_mean", "_ci95_high"),
        )
    ),
}

_SELECTION_STATUS_COLUMN = "selection_status"
_PARETO_FRONT_COLUMN = "is_pareto_front"
_RANK_COLUMN = "pareto_rank"
_DISTANCE_COLUMN = "ideal_point_distance"


class ParetoSelectionError(ValueError):
    """The Phase 2 table cannot be used for deterministic selection."""


def _ci_columns(mean_column: str) -> tuple[str, str]:
    return (
        mean_column.replace("_mean", "_ci95_low"),
        mean_column.replace("_mean", "_ci95_high"),
    )


def _as_boolean(series: pd.Series, *, label: str) -> pd.Series:
    """Parse native booleans or the CSV spellings emitted by pandas."""

    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ParetoSelectionError(f"{label} contains missing boolean values.")
        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False}
    if normalized.isna().any() or not normalized.isin(mapping).all():
        raise ParetoSelectionError(
            f"{label} must contain only boolean true/false values."
        )
    return normalized.map(mapping).astype(bool)


def _require_single_string_value(table: pd.DataFrame, column: str) -> str:
    values = table[column].dropna().astype(str).unique()
    if len(values) != 1 or not values[0]:
        raise ParetoSelectionError(
            f"{column} must contain exactly one non-empty value for the input study."
        )
    return str(values[0])


def validate_phase2_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a Phase 2 configuration-level study table.

    Failed configurations may have undefined objective values.  Those values are
    preserved rather than imputed; finite objectives and CIs are required only
    for configurations that Phase 2 marked valid and feasible.
    """

    missing_columns = sorted(_REQUIRED_COLUMNS.difference(table.columns))
    if missing_columns:
        raise ParetoSelectionError(
            "Phase 2 table is missing required columns: " + ", ".join(missing_columns)
        )
    if table.empty:
        raise ParetoSelectionError("Phase 2 table contains no configurations.")

    validated = table.copy()
    _require_single_string_value(validated, "study_id")
    _require_single_string_value(validated, "protocol_version")

    identifiers = validated[CONFIGURATION_ID_COLUMN]
    if identifiers.isna().any() or (identifiers.astype(str).str.strip() == "").any():
        raise ParetoSelectionError("configuration_id must be present and non-empty.")
    validated[CONFIGURATION_ID_COLUMN] = identifiers.astype(str)
    if validated[CONFIGURATION_ID_COLUMN].duplicated().any():
        duplicates = sorted(
            validated.loc[
                validated[CONFIGURATION_ID_COLUMN].duplicated(keep=False),
                CONFIGURATION_ID_COLUMN,
            ].unique()
        )
        raise ParetoSelectionError(
            "configuration_id values must be unique; duplicates: " + ", ".join(duplicates)
        )

    validated[VALID_COLUMN] = _as_boolean(validated[VALID_COLUMN], label=VALID_COLUMN)
    validated[FEASIBLE_COLUMN] = _as_boolean(validated[FEASIBLE_COLUMN], label=FEASIBLE_COLUMN)
    validated[REJECTION_REASONS_COLUMN] = (
        validated[REJECTION_REASONS_COLUMN].fillna("").astype(str)
    )

    eligible = validated[VALID_COLUMN] & validated[FEASIBLE_COLUMN]
    for _, mean_column, _ in OBJECTIVES:
        ci_low_column, ci_high_column = _ci_columns(mean_column)
        for column in (mean_column, ci_low_column, ci_high_column):
            validated[column] = pd.to_numeric(validated[column], errors="coerce")
        values = validated.loc[eligible, [mean_column, ci_low_column, ci_high_column]]
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ParetoSelectionError(
                f"Eligible configurations require finite {mean_column} values and 95% CIs."
            )
        if not values.empty:
            means = values[mean_column]
            lows = values[ci_low_column]
            highs = values[ci_high_column]
            if ((means < 0.0) | (means > 1.0)).any():
                raise ParetoSelectionError(
                    f"Eligible {mean_column} values must lie in [0, 1]."
                )
            if ((lows > means) | (means > highs)).any():
                raise ParetoSelectionError(
                    f"Eligible {mean_column} confidence intervals must contain their mean."
                )

    return validated.sort_values(CONFIGURATION_ID_COLUMN, kind="stable").reset_index(drop=True)


def _dominates(left: pd.Series, right: pd.Series) -> bool:
    """Return whether ``left`` Pareto-dominates ``right`` under the frozen directions."""

    no_worse = (
        left[LEAKAGE_COLUMN] <= right[LEAKAGE_COLUMN]
        and left[CORRELATION_COLUMN] <= right[CORRELATION_COLUMN]
        and left[EFFICIENCY_COLUMN] >= right[EFFICIENCY_COLUMN]
    )
    strictly_better = (
        left[LEAKAGE_COLUMN] < right[LEAKAGE_COLUMN]
        or left[CORRELATION_COLUMN] < right[CORRELATION_COLUMN]
        or left[EFFICIENCY_COLUMN] > right[EFFICIENCY_COLUMN]
    )
    return bool(no_worse and strictly_better)


def pareto_front_mask(eligible: pd.DataFrame) -> pd.Series:
    """Compute a non-dominance mask for feasible configuration means.

    Equal objective triples intentionally remain on the front; the frozen
    configuration-ID tie-break applies only when ranking them afterwards.
    """

    if eligible.empty:
        return pd.Series(False, index=eligible.index, dtype=bool)

    mask = pd.Series(True, index=eligible.index, dtype=bool)
    for right_index, right in eligible.iterrows():
        for left_index, left in eligible.iterrows():
            if left_index != right_index and _dominates(left, right):
                mask.loc[right_index] = False
                break
    return mask


def _ideal_point_distance(row: pd.Series) -> float:
    """Return the frozen equal-weight Euclidean cost distance to (0, 0, 0)."""

    costs = (
        float(row[LEAKAGE_COLUMN]),
        float(row[CORRELATION_COLUMN]),
        1.0 - float(row[EFFICIENCY_COLUMN]),
    )
    return math.sqrt(sum(cost * cost for cost in costs) / len(costs))


def _intervals_overlap(
    left_low: float, left_high: float, right_low: float, right_high: float
) -> bool:
    return max(left_low, right_low) <= min(left_high, right_high)


def _uncertainty_report(front: pd.DataFrame) -> dict[str, Any]:
    """Annotate selected-vs-front-member marginal 95% CI overlap.

    The source table contains a CI per configuration, not a paired-difference
    CI.  This deliberately reports overlap as a warning heuristic and does not
    turn it into a significance claim.
    """

    if front.empty:
        return {
            "policy": "flag_selected_vs_front_members_with_overlapping_marginal_95_percent_cis",
            "selected_configuration_id": None,
            "selection_uncertain": False,
            "comparisons": [],
        }

    selected = front.iloc[0]
    comparisons: list[dict[str, Any]] = []
    for _, alternative in front.iloc[1:].iterrows():
        overlap: dict[str, bool] = {}
        for objective_name, mean_column, _ in OBJECTIVES:
            low_column, high_column = _ci_columns(mean_column)
            overlap[objective_name] = _intervals_overlap(
                float(selected[low_column]),
                float(selected[high_column]),
                float(alternative[low_column]),
                float(alternative[high_column]),
            )
        comparisons.append(
            {
                "configuration_id": str(alternative[CONFIGURATION_ID_COLUMN]),
                "ci95_overlap": overlap,
                "all_objective_intervals_overlap": bool(all(overlap.values())),
            }
        )

    return {
        "policy": "flag_selected_vs_front_members_with_overlapping_marginal_95_percent_cis",
        "selected_configuration_id": str(selected[CONFIGURATION_ID_COLUMN]),
        "selection_uncertain": any(
            comparison["all_objective_intervals_overlap"] for comparison in comparisons
        ),
        "comparisons": comparisons,
    }


def select_pareto_front(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Annotate all candidates, rank the feasible Pareto front, and summarize selection."""

    candidates = validate_phase2_table(table)
    eligible_mask = candidates[VALID_COLUMN] & candidates[FEASIBLE_COLUMN]

    candidates[_SELECTION_STATUS_COLUMN] = np.where(
        ~candidates[VALID_COLUMN],
        "invalid",
        np.where(~candidates[FEASIBLE_COLUMN], "infeasible", "dominated"),
    )
    candidates[_PARETO_FRONT_COLUMN] = False
    candidates[_RANK_COLUMN] = pd.Series(pd.NA, index=candidates.index, dtype="Int64")
    candidates[_DISTANCE_COLUMN] = np.nan
    candidates["normalized_leakage_cost"] = np.nan
    candidates["normalized_residual_correlation_cost"] = np.nan
    candidates["normalized_signal_utility_cost"] = np.nan

    eligible = candidates.loc[eligible_mask].copy()
    front_index = pareto_front_mask(eligible)
    front = eligible.loc[front_index].copy()
    if not front.empty:
        front["normalized_leakage_cost"] = front[LEAKAGE_COLUMN]
        front["normalized_residual_correlation_cost"] = front[CORRELATION_COLUMN]
        front["normalized_signal_utility_cost"] = 1.0 - front[EFFICIENCY_COLUMN]
        front[_DISTANCE_COLUMN] = front.apply(_ideal_point_distance, axis=1)
        front = front.sort_values(
            [_DISTANCE_COLUMN, CONFIGURATION_ID_COLUMN], kind="stable"
        ).reset_index(drop=True)
        front[_RANK_COLUMN] = pd.Series(
            range(1, len(front) + 1), index=front.index, dtype="Int64"
        )

        indexed_front = front.set_index(CONFIGURATION_ID_COLUMN)
        front_ids = indexed_front.index
        candidate_ids = candidates[CONFIGURATION_ID_COLUMN]
        is_front = candidate_ids.isin(front_ids)
        candidates.loc[is_front, _SELECTION_STATUS_COLUMN] = "pareto_front"
        candidates.loc[is_front, _PARETO_FRONT_COLUMN] = True
        for column in (
            _RANK_COLUMN,
            _DISTANCE_COLUMN,
            "normalized_leakage_cost",
            "normalized_residual_correlation_cost",
            "normalized_signal_utility_cost",
        ):
            candidates.loc[is_front, column] = candidate_ids[is_front].map(
                indexed_front[column]
            )

    candidates = candidates.sort_values(CONFIGURATION_ID_COLUMN, kind="stable").reset_index(
        drop=True
    )
    front = candidates.loc[candidates[_PARETO_FRONT_COLUMN]].copy()
    front = front.sort_values(_RANK_COLUMN, kind="stable").reset_index(drop=True)
    uncertainty = _uncertainty_report(front)

    selected_configuration_id = (
        None if front.empty else str(front.iloc[0][CONFIGURATION_ID_COLUMN])
    )
    selection = {
        "schema_version": PARETO_SELECTION_SCHEMA_VERSION,
        "study_id": _require_single_string_value(candidates, "study_id"),
        "protocol_version": _require_single_string_value(candidates, "protocol_version"),
        "objectives": [
            {"name": name, "column": column, "direction": direction}
            for name, column, direction in OBJECTIVES
        ],
        "ideal_point_ranking": {
            "normalized_costs": {
                "leakage": LEAKAGE_COLUMN,
                "residual_correlation": CORRELATION_COLUMN,
                "signal_utility": f"1 - {EFFICIENCY_COLUMN}",
            },
            "ideal_point": [0.0, 0.0, 0.0],
            "distance": "equal_weight_euclidean",
            "weights": {
                "leakage": 1.0 / 3.0,
                "residual_correlation": 1.0 / 3.0,
                "signal_utility": 1.0 / 3.0,
            },
            "tie_break": "configuration_id_ascending",
        },
        "counts": {
            "all_configurations": int(len(candidates)),
            "eligible_configurations": int(eligible_mask.sum()),
            "pareto_front_configurations": int(len(front)),
        },
        "selected_configuration_id": selected_configuration_id,
        "uncertainty": uncertainty,
    }
    return candidates, front, selection


def read_phase2_table(path: str | Path) -> pd.DataFrame:
    """Read the CSV or Parquet table emitted by the Phase 2 collector."""

    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ParetoSelectionError("Input must be a .csv, .parquet, or .pq Phase 2 table.")


def write_pareto_projection(
    candidates: pd.DataFrame, front: pd.DataFrame, *, output_path: str | Path
) -> None:
    """Write the lean L-versus-median-efficiency projection as a PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    eligible = candidates.loc[candidates[VALID_COLUMN] & candidates[FEASIBLE_COLUMN]]
    dominated = eligible.loc[~eligible[_PARETO_FRONT_COLUMN]]
    if not dominated.empty:
        axis.scatter(
            dominated[LEAKAGE_COLUMN],
            dominated[EFFICIENCY_COLUMN],
            color="0.70",
            label="Feasible, dominated",
            zorder=1,
        )

    if not front.empty:
        x_error = np.vstack(
            (
                front[LEAKAGE_COLUMN] - front[LEAKAGE_COLUMN.replace("_mean", "_ci95_low")],
                front[LEAKAGE_COLUMN.replace("_mean", "_ci95_high")] - front[LEAKAGE_COLUMN],
            )
        )
        y_error = np.vstack(
            (
                front[EFFICIENCY_COLUMN]
                - front[EFFICIENCY_COLUMN.replace("_mean", "_ci95_low")],
                front[EFFICIENCY_COLUMN.replace("_mean", "_ci95_high")]
                - front[EFFICIENCY_COLUMN],
            )
        )
        axis.errorbar(
            front[LEAKAGE_COLUMN],
            front[EFFICIENCY_COLUMN],
            xerr=x_error,
            yerr=y_error,
            fmt="none",
            ecolor="0.35",
            capsize=2,
            zorder=2,
        )
        points = axis.scatter(
            front[LEAKAGE_COLUMN],
            front[EFFICIENCY_COLUMN],
            c=front[CORRELATION_COLUMN],
            cmap="viridis",
            edgecolor="black",
            label="Pareto front",
            zorder=3,
        )
        colorbar = figure.colorbar(points, ax=axis)
        colorbar.set_label("Residual correlation E (lower is better)")
        selected = front.iloc[0]
        axis.scatter(
            [selected[LEAKAGE_COLUMN]],
            [selected[EFFICIENCY_COLUMN]],
            marker="*",
            s=180,
            color="crimson",
            edgecolor="black",
            label="Selected",
            zorder=4,
        )

    axis.set_xlabel("Leakage L (lower is better)")
    axis.set_ylabel("Median signal efficiency (higher is better)")
    axis.set_xlim(left=0.0, right=1.0)
    axis.set_ylim(bottom=0.0, top=1.0)
    axis.set_title("Validation Pareto projection")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="best")
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def write_selection_outputs(
    candidates: pd.DataFrame,
    front: pd.DataFrame,
    selection: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist the lean Phase 3 tables, report, and projection."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    candidate_path = destination / "pareto_candidates.csv"
    front_path = destination / "pareto_front.csv"
    report_path = destination / "pareto_selection.json"
    plot_path = destination / "pareto_projection.png"

    candidates.to_csv(candidate_path, index=False)
    front.to_csv(front_path, index=False)
    write_pareto_projection(candidates, front, output_path=plot_path)
    selection["output_paths"] = {
        "candidates_csv": str(candidate_path),
        "front_csv": str(front_path),
        "selection_json": str(report_path),
        "projection_png": str(plot_path),
    }
    report_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "candidates_csv": candidate_path,
        "front_csv": front_path,
        "selection_json": report_path,
        "projection_png": plot_path,
    }


def select_and_write_pareto_front(
    input_path: str | Path, *, output_dir: str | Path
) -> dict[str, Any]:
    """Read one Phase 2 table, select the front, and write the lean artifacts."""

    source = Path(input_path)
    candidates, front, selection = select_pareto_front(read_phase2_table(source))
    selection["input_table"] = str(source.resolve())
    selection["output_paths"] = {
        name: str(path)
        for name, path in write_selection_outputs(
            candidates,
            front,
            selection,
            output_dir=output_dir,
        ).items()
    }
    return selection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct and rank the lean validation Pareto front from a Phase 2 table."
    )
    parser.add_argument(
        "--input-table",
        type=Path,
        required=True,
        help="Phase 2 pareto_metrics.csv or pareto_metrics.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the annotated candidate table, front, report, and PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selection = select_and_write_pareto_front(
        args.input_table,
        output_dir=args.output_dir,
    )
    print(
        "Selected "
        f"{selection['selected_configuration_id'] or 'no configuration'} "
        f"from {selection['counts']['pareto_front_configurations']} Pareto-front candidates."
    )


if __name__ == "__main__":
    main()
