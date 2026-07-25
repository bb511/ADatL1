#!/usr/bin/env python3
"""Select label-free trials and aggregate paper evaluation results.

The pipeline uses two deliberately small, long-form CSV contracts:

Candidate metrics (one row per candidate and validation strategy)::

    dataset,model,seed,candidate_id,strategy,value,params_json

Evaluation results (one row per intervention-level result)::

    dataset,model,strategy,seed,intervention,metric,value

Optional evaluation columns are ``pairing``, ``intervention_family`` and
``strength``.  Keeping these contracts independent of Hydra, Optuna, and the
dataset implementation makes the same commands usable for local smoke tests and
for final cluster runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_FREE_DIRECTIONS = {
    "cap": "maximize",
    "cap_metadata_nearest": "maximize",
    "cap_encoder_nearest": "maximize",
    "cap_random": "maximize",
    "drift": "minimize",
    "wasserstein": "minimize",
}
CANDIDATE_REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "seed",
    "candidate_id",
    "strategy",
    "value",
    "params_json",
}
RESULT_REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "strategy",
    "seed",
    "intervention",
    "metric",
    "value",
}
RESULT_OPTIONAL_COLUMNS = ("pairing", "intervention_family", "strength")
GROUP_COLUMNS = ["dataset", "model", "strategy", "metric"]
COLLECT_MANIFEST_REQUIRED_COLUMNS = {
    "path",
    "dataset",
    "model",
    "strategy",
    "seed",
}
RAW_VALUE_REQUIRED_COLUMNS = {"intervention", "metric", "value"}


def _read_csv(
    path: Path,
    required: set[str],
    *,
    dtype: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Read a finite-valued CSV and enforce its required columns."""
    frame = pd.read_csv(path, dtype=dtype)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite values.")
    return frame


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    """Parse and normalize a JSON object."""
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: params_json is not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{context}: params_json must encode an object.")
    return {str(key): item for key, item in parsed.items()}


def _hydra_value(value: Any) -> str:
    """Render a JSON-compatible value as a Hydra override value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def _candidate_group_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns defining an independently replayed candidate pool."""
    return ["dataset", "model", "seed"]


def validate_shared_candidate_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Verify every label-free strategy replays exactly the same candidate IDs."""
    unknown = sorted(set(frame["strategy"].astype(str)) - set(LABEL_FREE_DIRECTIONS))
    if unknown:
        raise ValueError(
            "Selection is restricted to label-free validation strategies; unsupported "
            f"strategies: {', '.join(unknown)}"
        )

    key = [*_candidate_group_columns(frame), "candidate_id", "strategy"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(
            "Candidate metrics must contain one value per candidate and strategy; "
            f"found {int(duplicates.sum())} duplicate rows."
        )

    audit_rows: list[dict[str, Any]] = []
    for group_values, group in frame.groupby(
        _candidate_group_columns(frame), sort=True, dropna=False
    ):
        strategy_sets = {
            strategy: frozenset(rows["candidate_id"].astype(str))
            for strategy, rows in group.groupby("strategy", sort=True)
        }
        reference = next(iter(strategy_sets.values()))
        inconsistent = {
            strategy: sorted(candidate_ids.symmetric_difference(reference))
            for strategy, candidate_ids in strategy_sets.items()
            if candidate_ids != reference
        }
        if inconsistent:
            labels = dict(zip(_candidate_group_columns(frame), group_values))
            raise ValueError(
                f"Candidate pool was not replayed for all strategies in {labels}: "
                f"{inconsistent}"
            )

        params_by_candidate: dict[str, str] = {}
        for candidate_id, rows in group.groupby("candidate_id", sort=True):
            canonical = {
                json.dumps(
                    _json_object(
                        value,
                        context=f"candidate {candidate_id}",
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for value in rows["params_json"]
            }
            if len(canonical) != 1:
                raise ValueError(
                    f"Candidate {candidate_id} has inconsistent params_json across "
                    "validation strategies."
                )
            params_by_candidate[str(candidate_id)] = canonical.pop()

        audit_rows.append(
            {
                **dict(zip(_candidate_group_columns(frame), group_values)),
                "n_candidates": len(reference),
                "n_strategies": len(strategy_sets),
                "strategies": ",".join(sorted(strategy_sets)),
                "pool_hash": hashlib.sha256(
                    "\n".join(
                        f"{candidate_id}:{params_by_candidate[candidate_id]}"
                        for candidate_id in sorted(params_by_candidate)
                    ).encode("utf-8")
                ).hexdigest(),
                "shared_pool": True,
            }
        )
    return pd.DataFrame(audit_rows)


def select_trials(candidate_metrics: Path, output_dir: Path) -> list[Path]:
    """Select one candidate per seed and strategy using validation metrics only."""
    frame = _read_csv(
        candidate_metrics,
        CANDIDATE_REQUIRED_COLUMNS,
        dtype={"candidate_id": str},
    )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    for column in ("dataset", "model", "candidate_id", "strategy"):
        frame[column] = frame[column].astype(str)

    audit = validate_shared_candidate_pool(frame)
    selected_rows: list[dict[str, Any]] = []
    retrain_rows: list[dict[str, Any]] = []
    group_columns = [*_candidate_group_columns(frame), "strategy"]
    for group_values, group in frame.groupby(group_columns, sort=True, dropna=False):
        dataset, model, seed, strategy = group_values
        direction = LABEL_FREE_DIRECTIONS[strategy]
        ascending = direction == "minimize"
        winner = (
            group.assign(_candidate_sort=group["candidate_id"].astype(str))
            .sort_values(
                ["value", "_candidate_sort"],
                ascending=[ascending, True],
                kind="stable",
            )
            .iloc[0]
        )
        params = _json_object(
            winner["params_json"],
            context=f"candidate {winner['candidate_id']}",
        )
        spec_name = f"{dataset}_{model}_{strategy}"
        candidate_id = str(winner["candidate_id"])
        run_name = f"{strategy}_candidate_{candidate_id}_seed_{seed}"
        selected = {
            "dataset": dataset,
            "model": model,
            "seed": int(seed),
            "strategy": strategy,
            "direction": direction,
            "candidate_id": candidate_id,
            "validation_value": float(winner["value"]),
            "params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
            "spec_name": spec_name,
            "run_name": run_name,
        }
        selected_rows.append(selected)
        retrain_rows.append(
            {
                "spec_name": spec_name,
                "run_name": run_name,
                "seed": int(seed),
                "candidate_id": candidate_id,
                "selection_strategy": strategy,
                "selection_value": float(winner["value"]),
                "overrides": [
                    f"{key}={_hydra_value(value)}" for key, value in sorted(params.items())
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_trials.csv"
    retrain_path = output_dir / "retrain_manifest.json"
    audit_path = output_dir / "candidate_pool_audit.csv"
    provenance_path = output_dir / "selection_provenance.json"
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    pd.DataFrame(retrain_rows).to_json(retrain_path, orient="records", indent=2)
    audit.to_csv(audit_path, index=False)
    provenance_path.write_text(
        json.dumps(
            {
                "candidate_metrics": str(candidate_metrics.resolve()),
                "candidate_metrics_sha256": _sha256(candidate_metrics),
                "allowed_strategies": LABEL_FREE_DIRECTIONS,
                "selection_rule": (
                    "Optimize the named label-free validation metric; break exact ties "
                    "by candidate_id in lexical order."
                ),
                "shared_candidate_pool_required": True,
                "n_selected": len(selected_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [selected_path, retrain_path, audit_path, provenance_path]


def create_checkpoint_manifest(
    selected_trials: Path,
    checkpoints_dir: Path,
    output: Path,
    *,
    checkpoint_name: str | None = None,
) -> Path:
    """Resolve retrained checkpoint paths into ``generation.py``'s input format."""
    frame = pd.read_csv(selected_trials, dtype={"candidate_id": str})
    required = {"spec_name", "run_name", "seed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{selected_trials} is missing required columns: {', '.join(missing)}")

    records: list[dict[str, Any]] = []
    missing_paths: list[Path] = []
    for row in frame.to_dict(orient="records"):
        run_dir = checkpoints_dir / f"{row['spec_name']}_retrain" / str(row["run_name"])
        checkpoint = run_dir / (
            checkpoint_name if checkpoint_name is not None else _selected_checkpoint_path(row)
        )
        if not checkpoint.is_file():
            missing_paths.append(checkpoint)
            continue
        records.append(
            {
                "spec_name": str(row["spec_name"]),
                "run_name": f"{row['run_name']}_evaluate",
                "seed": int(row["seed"]),
                "ckpt_path": str(checkpoint.resolve()),
            }
        )
    if missing_paths:
        preview = "\n".join(str(path) for path in missing_paths[:10])
        raise FileNotFoundError(
            f"Could not resolve {len(missing_paths)} selected checkpoints:\n{preview}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return output


def _selected_checkpoint_path(row: Mapping[str, Any]) -> Path:
    """Return the strategy-selected checkpoint path relative to a run."""
    dataset = str(row.get("dataset", ""))
    strategy = str(row.get("strategy", ""))
    if not dataset or not strategy:
        raise ValueError(
            "selected_trials must contain dataset and strategy when --checkpoint-name "
            "is not provided."
        )
    if strategy.startswith("cap"):
        reference = {
            "physics": "SingleNeutrino_E-10-gun",
            "cifar10": "reference_normal",
            "robustad": "shifted_normal_all",
            "cchamber": "reference_normal",
            "synthetic": "reference_normal",
        }.get(dataset)
        if reference is None:
            raise ValueError(f"Unknown dataset for CAP checkpoint resolution: {dataset}")
        metric = f"cap_ema_normal_vs_{reference}"
        return Path("summary") / metric / "max" / f"{metric}.ckpt"
    if strategy == "drift":
        return Path("summary") / "operational_drift_ema" / "min" / "operational_drift_ema.ckpt"
    if strategy == "wasserstein":
        reference = {
            "physics": "SingleNeutrino_E-10-gun",
            "cifar10": "reference_normal",
            "robustad": "shifted_normal_all",
            "cchamber": "reference_normal",
            "synthetic": "reference_normal",
        }.get(dataset)
        if reference is None:
            raise ValueError(f"Unknown dataset for Wasserstein checkpoint resolution: {dataset}")
        metric = f"w1dist_ema_normal_vs_{reference}"
        return Path("summary") / metric / "min" / f"{metric}.ckpt"
    raise ValueError(f"Unsupported label-free selection strategy: {strategy}")


def collect_results(manifest: Path, output: Path) -> list[Path]:
    """Annotate callback ``values.csv`` files and combine them into one result table."""
    entries = pd.read_csv(manifest)
    missing = sorted(COLLECT_MANIFEST_REQUIRED_COLUMNS - set(entries.columns))
    if missing:
        raise ValueError(f"{manifest} is missing required columns: {', '.join(missing)}")
    if entries.empty:
        raise ValueError(f"{manifest} contains no rows.")

    collected: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    annotation_columns = ["dataset", "model", "strategy", "seed", "pairing"]
    for index, entry in entries.iterrows():
        source = Path(str(entry["path"]))
        if not source.is_absolute():
            source = manifest.parent / source
        if not source.is_file():
            raise FileNotFoundError(f"Manifest row {index}: values file not found: {source}")
        values = _read_csv(source, RAW_VALUE_REQUIRED_COLUMNS)
        values["seed"] = int(entry["seed"])
        for column in ("dataset", "model", "strategy"):
            values[column] = str(entry[column])
        if "pairing" in entries and not pd.isna(entry.get("pairing")):
            values["pairing"] = str(entry["pairing"])
        ordered = [
            *[column for column in annotation_columns if column in values],
            *[
                column
                for column in (
                    "checkpoint",
                    "intervention",
                    "intervention_family",
                    "strength",
                    "metric",
                    "value",
                )
                if column in values
            ],
        ]
        collected.append(values[ordered])
        sources.append(
            {
                "path": str(source.resolve()),
                "sha256": _sha256(source),
                "rows": len(values),
            }
        )

    combined = pd.concat(collected, ignore_index=True)
    missing_result_columns = RESULT_REQUIRED_COLUMNS - set(combined.columns)
    if missing_result_columns:
        raise ValueError(
            "Collected values do not satisfy the result contract; missing: "
            f"{', '.join(sorted(missing_result_columns))}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    provenance = output.with_suffix(".provenance.json")
    provenance.write_text(
        json.dumps(
            {
                "manifest": str(manifest.resolve()),
                "manifest_sha256": _sha256(manifest),
                "sources": sources,
                "rows": len(combined),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [output, provenance]


def _result_group_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns defining one reporting method and metric."""
    return [*GROUP_COLUMNS, *(["pairing"] if "pairing" in frame else [])]


def _bootstrap_interval(values: np.ndarray, *, seed: int = 12345) -> tuple[float, float]:
    """Compute a deterministic percentile bootstrap interval for the mean."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return math.nan, math.nan
    if values.size <= 7:
        grids = np.indices((values.size,) * values.size).reshape(values.size, -1).T
        means = values[grids].mean(axis=1)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, values.size, size=(20_000, values.size))
        means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _aggregate_values(group: pd.DataFrame) -> pd.Series:
    """Summarize one group of scalar measurements."""
    values = group["value"].to_numpy(dtype=float)
    low, high = _bootstrap_interval(values)
    return pd.Series(
        {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else math.nan,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "ci95_low": low,
            "ci95_high": high,
            "n": int(values.size),
        }
    )


def validate_evaluation_coverage(frame: pd.DataFrame) -> None:
    """Require paired seeds and intervention coverage for every compared strategy."""
    for group_values, group in frame.groupby(
        ["dataset", "model", "metric"], sort=True, dropna=False
    ):
        seed_sets = {
            strategy: frozenset(rows["seed"].astype(int))
            for strategy, rows in group.groupby("strategy", sort=True)
        }
        reference_seeds = next(iter(seed_sets.values()))
        if any(seeds != reference_seeds for seeds in seed_sets.values()):
            labels = dict(zip(["dataset", "model", "metric"], group_values))
            raise ValueError(
                f"Strategies do not have paired seed coverage in {labels}: {seed_sets}"
            )

        intervention_sets = {
            (str(strategy), int(seed)): frozenset(rows["intervention"].astype(str))
            for (strategy, seed), rows in group.groupby(["strategy", "seed"], sort=True)
        }
        reference_interventions = next(iter(intervention_sets.values()))
        if any(
            interventions != reference_interventions
            for interventions in intervention_sets.values()
        ):
            labels = dict(zip(["dataset", "model", "metric"], group_values))
            raise ValueError(
                "Strategies and seeds do not have identical intervention coverage in " f"{labels}."
            )


def _pairwise_differences(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute paired strategy differences across shared seeds."""
    rows: list[dict[str, Any]] = []
    pair_groups = ["dataset", "model", "metric"]
    for group_values, group in seed_summary.groupby(pair_groups, sort=True, dropna=False):
        strategies = sorted(group["strategy"].unique())
        for left, right in combinations(strategies, 2):
            a = group.loc[group["strategy"] == left, ["seed", "value"]]
            b = group.loc[group["strategy"] == right, ["seed", "value"]]
            merged = a.merge(b, on="seed", suffixes=("_left", "_right"), validate="one_to_one")
            if merged.empty:
                continue
            difference = merged["value_left"].to_numpy(dtype=float) - merged[
                "value_right"
            ].to_numpy(dtype=float)
            low, high = _bootstrap_interval(difference)
            rows.append(
                {
                    **dict(zip(pair_groups, group_values)),
                    "strategy_left": left,
                    "strategy_right": right,
                    "mean_difference": float(np.mean(difference)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_paired_seeds": int(difference.size),
                }
            )
    return pd.DataFrame(rows)


def aggregate_results(
    results: Path,
    output_dir: Path,
    *,
    main_metric: str = "auprc",
) -> list[Path]:
    """Aggregate interventions within seed, then estimate uncertainty across seeds."""
    frame = _read_csv(results, RESULT_REQUIRED_COLUMNS)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    string_columns = ["dataset", "model", "strategy", "intervention", "metric"]
    string_columns.extend(
        column for column in ("pairing", "intervention_family", "strength") if column in frame
    )
    for column in string_columns:
        frame[column] = frame[column].astype(str)
    duplicate_key = [
        "dataset",
        "model",
        "strategy",
        "seed",
        "intervention",
        "metric",
        *(["pairing"] if "pairing" in frame else []),
    ]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        raise ValueError(
            "Evaluation results must contain one value per seed/intervention/metric; "
            f"found {int(duplicates.sum())} duplicate rows."
        )
    validate_evaluation_coverage(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "results_long.csv"
    frame.sort_values(duplicate_key).to_csv(normalized_path, index=False)

    seed_groups = [*_result_group_columns(frame), "seed"]
    seed_summary = (
        frame.groupby(seed_groups, sort=True, dropna=False)["value"].mean().reset_index()
    )
    seed_summary_path = output_dir / "seed_summary.csv"
    seed_summary.to_csv(seed_summary_path, index=False)

    summary_groups = _result_group_columns(frame)
    summary = (
        seed_summary.groupby(summary_groups, sort=True, dropna=False)
        .apply(_aggregate_values, include_groups=False)
        .reset_index()
    )
    summary = summary.rename(columns={"n": "n_seeds"})
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    intervention_groups = [
        *_result_group_columns(frame),
        "intervention",
        *[column for column in ("intervention_family", "strength") if column in frame],
    ]
    intervention_summary = (
        frame.groupby(intervention_groups, sort=True, dropna=False)
        .apply(_aggregate_values, include_groups=False)
        .reset_index()
        .rename(columns={"n": "n_seeds"})
    )
    intervention_path = output_dir / "intervention_summary.csv"
    intervention_summary.to_csv(intervention_path, index=False)

    pairwise = _pairwise_differences(seed_summary)
    pairwise_path = output_dir / "paired_strategy_differences.csv"
    pairwise.to_csv(pairwise_path, index=False)

    plot_paths = _make_plots(summary, intervention_summary, output_dir, main_metric)
    report_path = output_dir / "report.md"
    _write_report(report_path, frame, summary, pairwise, main_metric, plot_paths)
    provenance_path = output_dir / "aggregation_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "results": str(results.resolve()),
                "results_sha256": _sha256(results),
                "aggregation": (
                    "Arithmetic mean across interventions within each seed, followed "
                    "by mean and deterministic bootstrap 95% interval across seeds."
                ),
                "main_metric": main_metric,
                "n_rows": len(frame),
                "n_seeds": int(frame["seed"].nunique()),
                "n_interventions": int(frame["intervention"].nunique()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        normalized_path,
        seed_summary_path,
        summary_path,
        intervention_path,
        pairwise_path,
        *plot_paths,
        report_path,
        provenance_path,
    ]


def _safe_name(value: str) -> str:
    """Convert a label to a portable filename component."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _make_plots(
    summary: pd.DataFrame,
    intervention_summary: pd.DataFrame,
    output_dir: Path,
    main_metric: str,
) -> list[Path]:
    """Create comparison and intervention plots for the primary metric."""
    paths: list[Path] = []
    for dataset in sorted(summary["dataset"].unique()):
        selected = summary[(summary["dataset"] == dataset) & (summary["metric"] == main_metric)]
        if selected.empty:
            continue
        path = output_dir / f"{_safe_name(dataset)}_{_safe_name(main_metric)}_comparison.png"
        _plot_comparison(selected, path, main_metric)
        paths.append(path)

        intervention_selected = intervention_summary[
            (intervention_summary["dataset"] == dataset)
            & (intervention_summary["metric"] == main_metric)
        ]
        if not intervention_selected.empty:
            heatmap_path = (
                output_dir / f"{_safe_name(dataset)}_{_safe_name(main_metric)}_interventions.png"
            )
            _plot_intervention_heatmap(intervention_selected, heatmap_path, main_metric)
            paths.append(heatmap_path)
    return paths


def _plot_comparison(frame: pd.DataFrame, path: Path, metric: str) -> None:
    """Plot strategy means and seed-level confidence intervals by model."""
    models = sorted(frame["model"].unique())
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(max(5.0, 3.6 * len(models)), 4.2),
        squeeze=False,
        sharey=True,
    )
    for axis, model in zip(axes[0], models):
        part = frame[frame["model"] == model].sort_values("strategy")
        labels = part["strategy"].tolist()
        means = part["mean"].to_numpy(dtype=float)
        low = np.clip(means - part["ci95_low"].to_numpy(dtype=float), 0.0, None)
        high = np.clip(part["ci95_high"].to_numpy(dtype=float) - means, 0.0, None)
        errors = np.nan_to_num(np.vstack([low, high]), nan=0.0)
        axis.errorbar(np.arange(len(part)), means, yerr=errors, fmt="o", capsize=4)
        axis.set_title(model)
        axis.set_xticks(np.arange(len(part)), labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel(metric)
    figure.suptitle(f"{metric}: intervention mean, uncertainty across seeds")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_intervention_heatmap(frame: pd.DataFrame, path: Path, metric: str) -> None:
    """Plot intervention-level means for every model and strategy."""
    plot_frame = frame.copy()
    plot_frame["method"] = plot_frame["model"] + " · " + plot_frame["strategy"]
    matrix = plot_frame.pivot(index="intervention", columns="method", values="mean")
    matrix = matrix.sort_index().sort_index(axis=1)
    figure, axis = plt.subplots(
        figsize=(max(7.0, 0.7 * len(matrix.columns)), max(4.0, 0.25 * len(matrix.index)))
    )
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    axis.set_title(f"{metric} by intervention")
    figure.colorbar(image, ax=axis, label=metric)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    path: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    main_metric: str,
    plot_paths: Sequence[Path],
) -> None:
    """Write a concise Markdown report referencing the generated artifacts."""
    main = summary[summary["metric"] == main_metric].copy()
    lines = [
        "# Paper experiment report",
        "",
        "Results are averaged across interventions within each seed. Confidence intervals "
        "therefore quantify uncertainty across seeds (seed-to-seed variation) rather than "
        "treating interventions as independent training replicates.",
        "",
        "## Coverage",
        "",
        f"- Rows: {len(frame)}",
        f"- Seeds: {frame['seed'].nunique()}",
        f"- Interventions: {frame['intervention'].nunique()}",
        f"- Models: {', '.join(sorted(frame['model'].unique()))}",
        f"- Strategies: {', '.join(sorted(frame['strategy'].unique()))}",
        "",
        f"## {main_metric} summary",
        "",
    ]
    if main.empty:
        lines.append(f"No rows were found for metric `{main_metric}`.")
    else:
        display = main[
            ["dataset", "model", "strategy", "mean", "ci95_low", "ci95_high", "n_seeds"]
        ].copy()
        for column in ("mean", "ci95_low", "ci95_high"):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
        lines.extend(_markdown_table(display))

    if not pairwise.empty:
        selected = pairwise[pairwise["metric"] == main_metric].copy()
        if not selected.empty:
            selected["abs_difference"] = selected["mean_difference"].abs()
            selected = selected.sort_values("abs_difference", ascending=False).head(12)
            lines.extend(["", "## Largest paired strategy differences", ""])
            display = selected[
                [
                    "dataset",
                    "model",
                    "strategy_left",
                    "strategy_right",
                    "mean_difference",
                    "ci95_low",
                    "ci95_high",
                    "n_paired_seeds",
                ]
            ].copy()
            for column in ("mean_difference", "ci95_low", "ci95_high"):
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{value:.4f}"
                )
            lines.extend(_markdown_table(display))

    if plot_paths:
        lines.extend(["", "## Figures", ""])
        lines.extend(f"- [{plot.name}]({plot.name})" for plot in plot_paths)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    """Render a small dataframe as a GitHub-flavored Markdown table."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Label-free selection and paper-result aggregation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select", help="Select trials from a replayed label-free candidate pool."
    )
    select_parser.add_argument("--candidate-metrics", type=Path, required=True)
    select_parser.add_argument("--output-dir", type=Path, required=True)

    checkpoints_parser = subparsers.add_parser(
        "checkpoints", help="Resolve retrained checkpoints for evaluation."
    )
    checkpoints_parser.add_argument("--selected-trials", type=Path, required=True)
    checkpoints_parser.add_argument("--checkpoints-dir", type=Path, required=True)
    checkpoints_parser.add_argument("--output", type=Path, required=True)
    checkpoints_parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Override the strategy-selected checkpoint path (for example last.ckpt).",
    )

    collect_parser = subparsers.add_parser(
        "collect", help="Collect annotated callback values.csv files."
    )
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Aggregate long-form intervention evaluation results."
    )
    aggregate_parser.add_argument("--results", type=Path, required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser.add_argument("--main-metric", default="auprc")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested pipeline stage."""
    args = parse_args(argv)
    if args.command == "select":
        paths = select_trials(args.candidate_metrics, args.output_dir)
    elif args.command == "checkpoints":
        paths = [
            create_checkpoint_manifest(
                args.selected_trials,
                args.checkpoints_dir,
                args.output,
                checkpoint_name=args.checkpoint_name,
            )
        ]
    elif args.command == "collect":
        paths = collect_results(args.manifest, args.output)
    elif args.command == "aggregate":
        paths = aggregate_results(
            args.results,
            args.output_dir,
            main_metric=args.main_metric,
        )
    else:
        raise SystemExit(f"Unsupported command: {args.command}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
