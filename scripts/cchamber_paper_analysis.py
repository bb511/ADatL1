#!/usr/bin/env python3
"""Run the frozen, seed-first Causal Chamber paper analysis.

This module is deliberately separate from campaign execution.  It reads an immutable
campaign root, a prespecified analysis plan, an intervention taxonomy, and an integrity
manifest that pins all three inputs plus the final result table by SHA-256.  It refuses
to write inside the campaign root.

The integrity manifest has this minimal form::

    {
      "schema_version": 1,
      "files": {
        "campaign": {"path": "campaign.json", "sha256": "..."},
        "results": {"path": "paper/results.csv", "sha256": "..."},
        "analysis_plan": {"path": "/frozen/analysis_plan.json", "sha256": "..."},
        "taxonomy": {"path": "/frozen/intervention_taxonomy.csv", "sha256": "..."}
      },
      "optional_artifacts": {
        "candidate_metrics": {"path": "selection/candidate_metrics.csv", "sha256": "..."},
        "pairing_proxy_sensitivity": {
          "path": "selection/pairing_proxy_sensitivity.csv", "sha256": "..."
        }
      }
    }

Relative paths are resolved against ``--campaign-root``.  The analysis plan freezes
models, strategies, reporting seeds, interventions, metrics, and superiority contrasts.
Each contrast must name a ``family`` used for Holm adjustment::

    {
      "schema_version": 1,
      "campaign_id": "...",
      "models": ["ae", "vae", "svdd", "realnvp"],
      "strategies": ["cap_metadata_nearest", "cap_encoder_nearest",
                     "cap_random", "drift", "wasserstein"],
      "reporting_seeds": [1001, 1002],
      "interventions": ["uniform_red_weak", "uniform_red_mid"],
      "metrics": ["auprc", "efficiency_operational"],
      "strength_order": ["weak", "mid", "strong"],
      "contrasts": [
        {"id": "metadata_vs_random", "family": "cap_vs_baselines",
         "left": "cap_metadata_nearest", "right": "cap_random",
         "alternative": "greater"}
      ]
    }

The taxonomy is a CSV with one row per intervention and columns
``intervention,intervention_target,strength,semantic_family,system_group``.
``system_group`` must be either ``process`` or ``measurement``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REQUIRED_RESULT_COLUMNS = {
    "model",
    "strategy",
    "seed",
    "intervention",
    "metric",
    "value",
}
REQUIRED_TAXONOMY_COLUMNS = {
    "intervention",
    "intervention_target",
    "strength",
    "semantic_family",
    "system_group",
}
REQUIRED_INPUTS = ("campaign", "results", "analysis_plan", "taxonomy")
PRIMARY_CLASSIFICATION = "confirmatory"
COMPLEMENTARY_CLASSIFICATION = "complementary_prespecified"
EXPLORATORY_CLASSIFICATION = "exploratory_outcome_selected"
ADMINISTRATIVE_CLASSIFICATION = "provenance_or_status"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    """Write a JSON value atomically without permitting non-finite numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level types."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _resolve_input(campaign_root: Path, record: Mapping[str, Any], name: str) -> Path:
    """Resolve and fingerprint-check one integrity-manifest input."""
    if not {"path", "sha256"}.issubset(record):
        raise ValueError(f"Integrity entry {name!r} must contain path and sha256.")
    path = Path(str(record["path"])).expanduser()
    if not path.is_absolute():
        path = campaign_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _sha256(path)
    expected = str(record["sha256"])
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {name}: observed {observed}, expected {expected}.")
    return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is within ``parent``."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_output_dir(output_dir: Path, campaign_root: Path) -> Path:
    """Create an empty output directory outside the immutable campaign."""
    output_dir = output_dir.expanduser().resolve()
    if _is_relative_to(output_dir, campaign_root):
        raise ValueError("Analysis output must be outside the immutable campaign root.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Analysis output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _validate_sequence(name: str, value: Any) -> list[Any]:
    """Validate one non-empty, duplicate-free analysis-plan sequence."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"Analysis plan {name!r} must be a non-empty JSON list.")
    if len(set(map(str, value))) != len(value):
        raise ValueError(f"Analysis plan {name!r} contains duplicate values.")
    return value


def _validate_plan(plan: Mapping[str, Any], campaign: Mapping[str, Any]) -> None:
    """Validate the frozen analysis plan against campaign metadata."""
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("Analysis plan schema_version must be 1.")
    if str(plan.get("campaign_id")) != str(campaign.get("campaign_id")):
        raise ValueError("Analysis plan campaign_id does not match campaign.json.")
    comparisons = {
        "models": campaign["models"],
        "strategies": campaign["strategies"],
        "reporting_seeds": campaign["reporting_seeds"],
        "interventions": campaign["interventions"],
    }
    for name, campaign_values in comparisons.items():
        plan_values = _validate_sequence(name, plan.get(name))
        if list(plan_values) != list(campaign_values):
            raise ValueError(f"Analysis plan {name} does not exactly match campaign.json.")
    metrics = [str(value) for value in _validate_sequence("metrics", plan.get("metrics"))]
    if metrics != ["auprc", "efficiency_operational"]:
        raise ValueError(
            "Analysis plan metrics must be ['auprc', 'efficiency_operational'] in that order."
        )
    strength_order = [
        str(value) for value in _validate_sequence("strength_order", plan.get("strength_order"))
    ]
    if strength_order != ["weak", "mid", "strong"]:
        raise ValueError("strength_order must be ['weak', 'mid', 'strong'].")

    contrasts = plan.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError("Analysis plan must contain prespecified contrasts.")
    expected_pairs = {
        (cap, baseline)
        for cap in ("cap_metadata_nearest", "cap_encoder_nearest")
        for baseline in ("cap_random", "drift", "wasserstein")
    }
    observed_pairs: set[tuple[str, str]] = set()
    ids: set[str] = set()
    for contrast in contrasts:
        if not isinstance(contrast, dict):
            raise ValueError("Every analysis contrast must be a JSON object.")
        missing = {"id", "family", "left", "right", "alternative"} - set(contrast)
        if missing:
            raise ValueError(f"Analysis contrast is missing: {', '.join(sorted(missing))}.")
        contrast_id = str(contrast["id"])
        if contrast_id in ids:
            raise ValueError(f"Duplicate contrast id: {contrast_id}")
        ids.add(contrast_id)
        left = str(contrast["left"])
        right = str(contrast["right"])
        if left not in plan["strategies"] or right not in plan["strategies"]:
            raise ValueError(f"Contrast {contrast_id} references an unknown strategy.")
        if str(contrast["alternative"]) != "greater":
            raise ValueError(f"Contrast {contrast_id} must use alternative='greater'.")
        observed_pairs.add((left, right))
    if observed_pairs != expected_pairs:
        raise ValueError(
            "Analysis contrasts must contain exactly metadata/encoder CAP versus "
            "random, drift, and Wasserstein."
        )


def _validate_taxonomy(
    taxonomy: pd.DataFrame,
    interventions: Sequence[str],
) -> pd.DataFrame:
    """Validate exact intervention taxonomy coverage and semantic fields."""
    missing = REQUIRED_TAXONOMY_COLUMNS - set(taxonomy.columns)
    if missing:
        raise ValueError(f"Taxonomy is missing columns: {', '.join(sorted(missing))}.")
    taxonomy = taxonomy.loc[:, sorted(REQUIRED_TAXONOMY_COLUMNS)].copy()
    if taxonomy.isna().any().any():
        raise ValueError("Taxonomy fields must be non-null.")
    for column in REQUIRED_TAXONOMY_COLUMNS:
        taxonomy[column] = taxonomy[column].astype(str).str.strip()
    if taxonomy["intervention"].duplicated().any():
        raise ValueError("Taxonomy must contain exactly one row per intervention.")
    if set(taxonomy["intervention"]) != set(map(str, interventions)):
        raise ValueError("Taxonomy intervention coverage does not match the analysis plan.")
    if not set(taxonomy["strength"]).issubset({"weak", "mid", "strong"}):
        raise ValueError("Taxonomy strength must be weak, mid, or strong.")
    if set(taxonomy["system_group"]) != {"process", "measurement"}:
        raise ValueError("Taxonomy must contain both process and measurement system groups.")
    if (taxonomy == "").any().any():
        raise ValueError("Taxonomy fields must be non-empty.")
    return taxonomy


def _artifact_classification(path: Path) -> str:
    """Classify artifacts without treating support outputs as confirmatory."""
    name = path.name
    if name.startswith("exploratory_"):
        return EXPLORATORY_CLASSIFICATION
    if (
        "strength" in name
        or "process_measurement" in name
        or "system_group" in name
        or "pairing_robustness" in name
    ):
        return COMPLEMENTARY_CLASSIFICATION
    if name == "component_status.json":
        return ADMINISTRATIVE_CLASSIFICATION
    return PRIMARY_CLASSIFICATION


def _validate_results(
    results: pd.DataFrame,
    plan: Mapping[str, Any],
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """Validate exact result coverage, bounds, identities, and taxonomy join."""
    missing = REQUIRED_RESULT_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {', '.join(sorted(missing))}.")
    frame = results.loc[:, sorted(REQUIRED_RESULT_COLUMNS)].copy()
    for column in ("model", "strategy", "intervention", "metric"):
        frame[column] = frame[column].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="raise").astype(float)
    if not np.isfinite(frame["value"].to_numpy()).all():
        raise ValueError("Results contain non-finite values.")
    if not frame["value"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("AUPRC and efficiency results must lie in [0, 1].")

    expected = pd.MultiIndex.from_product(
        [
            list(map(str, plan["models"])),
            list(map(str, plan["strategies"])),
            list(map(int, plan["reporting_seeds"])),
            list(map(str, plan["interventions"])),
            list(map(str, plan["metrics"])),
        ],
        names=["model", "strategy", "seed", "intervention", "metric"],
    )
    observed_columns = list(expected.names)
    if frame.duplicated(observed_columns).any():
        raise ValueError("Results contain duplicate model/strategy/seed/intervention/metric rows.")
    observed = pd.MultiIndex.from_frame(frame[observed_columns])
    missing_rows = expected.difference(observed)
    extra_rows = observed.difference(expected)
    if len(missing_rows) or len(extra_rows):
        raise ValueError(
            "Results do not have exact frozen coverage: "
            f"{len(missing_rows)} missing and {len(extra_rows)} extra rows."
        )
    frame = frame.merge(taxonomy, on="intervention", how="left", validate="many_to_one")
    return frame.sort_values(observed_columns, kind="stable").reset_index(drop=True)


def _verify_inputs(
    campaign_root: Path,
    analysis_plan: Path,
    taxonomy_path: Path,
    integrity_manifest: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Path],
]:
    """Verify frozen input hashes before parsing any outcome values."""
    campaign_root = campaign_root.expanduser().resolve()
    integrity_manifest = integrity_manifest.expanduser().resolve()
    integrity = _read_json(integrity_manifest)
    if int(integrity.get("schema_version", -1)) != 1:
        raise ValueError("Integrity manifest schema_version must be 1.")
    files = integrity.get("files")
    if not isinstance(files, dict):
        raise ValueError("Integrity manifest must contain a files object.")
    missing = set(REQUIRED_INPUTS) - set(files)
    if missing:
        raise ValueError(f"Integrity manifest is missing: {', '.join(sorted(missing))}.")
    paths = {name: _resolve_input(campaign_root, files[name], name) for name in REQUIRED_INPUTS}
    if paths["campaign"] != (campaign_root / "campaign.json").resolve():
        raise ValueError("Integrity campaign path must be <campaign-root>/campaign.json.")
    if paths["analysis_plan"] != analysis_plan.expanduser().resolve():
        raise ValueError("--analysis-plan does not match the integrity manifest.")
    if paths["taxonomy"] != taxonomy_path.expanduser().resolve():
        raise ValueError("--taxonomy does not match the integrity manifest.")

    campaign = _read_json(paths["campaign"])
    plan = _read_json(paths["analysis_plan"])
    _validate_plan(plan, campaign)
    taxonomy = _validate_taxonomy(
        pd.read_csv(paths["taxonomy"]),
        plan["interventions"],
    )
    results = _validate_results(pd.read_csv(paths["results"]), plan, taxonomy)
    return campaign, plan, taxonomy, results, integrity, paths


def _mean_interval(values: Sequence[float], confidence: float = 0.95) -> dict[str, float | int]:
    """Return a t interval across independent reporting-seed estimates."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty array.")
    mean = float(array.mean())
    if array.size == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "n_seeds": 1,
        }
    std = float(array.std(ddof=1))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, array.size - 1))
    half_width = critical * std / math.sqrt(array.size)
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "n_seeds": int(array.size),
    }


def _exact_sign_flip_greater(differences: Sequence[float]) -> float:
    """Return an exact one-sided paired sign-flip p-value."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Sign-flip test requires a non-empty one-dimensional sample.")
    if values.size > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 paired seeds.")
    observed = float(values.mean())
    bit_grid = np.arange(1 << values.size, dtype=np.uint64)[:, None]
    positions = np.arange(values.size, dtype=np.uint64)[None, :]
    signs = 2.0 * ((bit_grid >> positions) & 1).astype(float) - 1.0
    permuted = (signs * values[None, :]).mean(axis=1)
    return float(np.mean(permuted >= observed - 1e-15))


def _exact_sign_test_greater(differences: Sequence[float]) -> tuple[float, int, int]:
    """Return a one-sided exact sign-test p-value and nonzero counts."""
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0.0]
    positives = int((nonzero > 0.0).sum())
    if nonzero.size == 0:
        return 1.0, 0, 0
    result = stats.binomtest(positives, int(nonzero.size), p=0.5, alternative="greater")
    return float(result.pvalue), positives, int(nonzero.size)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (values.size - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def _seed_first_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average interventions within seed before summarizing across seeds."""
    seed = (
        frame.groupby(["model", "strategy", "seed", "metric"], sort=True)["value"]
        .mean()
        .reset_index()
    )
    rows = []
    for keys, group in seed.groupby(["model", "strategy", "metric"], sort=True):
        rows.append(
            {
                "model": keys[0],
                "strategy": keys[1],
                "metric": keys[2],
                **_mean_interval(group["value"]),
            }
        )
    return seed, pd.DataFrame(rows)


def _apply_holm(
    frame: pd.DataFrame,
    groups: Sequence[str],
    p_columns: Sequence[str],
) -> pd.DataFrame:
    """Apply Holm correction independently within named table groups."""
    adjusted = frame.copy()
    for p_column in p_columns:
        output_column = f"{p_column}_holm"
        adjusted[output_column] = np.nan
        for _, indices in adjusted.groupby(list(groups), sort=True).groups.items():
            values = adjusted.loc[indices, p_column].astype(float).tolist()
            adjusted.loc[indices, output_column] = _holm_adjust(values)
    return adjusted


def _contrast_rows(
    seed_summary: pd.DataFrame,
    plan: Mapping[str, Any],
) -> pd.DataFrame:
    """Compute prespecified paired strategy contrasts from seed-level estimates."""
    rows: list[dict[str, Any]] = []
    for metric in plan["metrics"]:
        metric_frame = seed_summary[seed_summary["metric"] == metric]
        for model in plan["models"]:
            pivot = metric_frame[metric_frame["model"] == model].pivot(
                index="seed",
                columns="strategy",
                values="value",
            )
            expected_seeds = list(map(int, plan["reporting_seeds"]))
            if list(pivot.index) != expected_seeds:
                raise ValueError(f"Seed pairing failed for {model}/{metric}.")
            for contrast in plan["contrasts"]:
                left = str(contrast["left"])
                right = str(contrast["right"])
                differences = (pivot[left] - pivot[right]).to_numpy(dtype=float)
                sign_p, positives, nonzero = _exact_sign_test_greater(differences)
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "contrast_id": str(contrast["id"]),
                        "test_family": str(contrast["family"]),
                        "strategy_left": left,
                        "strategy_right": right,
                        "alternative": "left_greater_than_right",
                        "mean_difference": float(differences.mean()),
                        "median_difference": float(np.median(differences)),
                        "p_signflip": _exact_sign_flip_greater(differences),
                        "p_sign": sign_p,
                        "positive_seeds": positives,
                        "nonzero_seeds": nonzero,
                        **{
                            key: value
                            for key, value in _mean_interval(differences).items()
                            if key in {"std", "ci_low", "ci_high", "n_seeds"}
                        },
                    }
                )
    result = pd.DataFrame(rows)
    result = _apply_holm(
        result,
        groups=["metric", "test_family"],
        p_columns=["p_signflip", "p_sign"],
    )
    result["reject_signflip_holm_0.05"] = result["p_signflip_holm"] < 0.05
    result["reject_sign_holm_0.05"] = result["p_sign_holm"] < 0.05
    return result


def _strength_outputs(
    frame: pd.DataFrame,
    strength_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build seed-first target-strength summaries and within-target contrasts."""
    target_seed = (
        frame.groupby(
            [
                "model",
                "strategy",
                "metric",
                "intervention_target",
                "strength",
                "seed",
            ],
            sort=True,
        )["value"]
        .mean()
        .reset_index()
    )
    summary_rows = []
    for keys, group in target_seed.groupby(
        ["model", "strategy", "metric", "intervention_target", "strength"],
        sort=True,
    ):
        summary_rows.append(
            {
                "model": keys[0],
                "strategy": keys[1],
                "metric": keys[2],
                "intervention_target": keys[3],
                "strength": keys[4],
                **_mean_interval(group["value"]),
            }
        )
    contrast_rows = []
    rank = {name: index for index, name in enumerate(strength_order)}
    for keys, group in target_seed.groupby(
        ["model", "strategy", "metric", "intervention_target"],
        sort=True,
    ):
        pivot = group.pivot(index="seed", columns="strength", values="value")
        available = sorted(pivot.columns, key=rank.__getitem__)
        for lower_index, lower in enumerate(available):
            for higher in available[lower_index + 1 :]:
                differences = (pivot[higher] - pivot[lower]).dropna().to_numpy(dtype=float)
                if differences.size == 0:
                    continue
                sign_p, positives, nonzero = _exact_sign_test_greater(differences)
                contrast_rows.append(
                    {
                        "model": keys[0],
                        "strategy": keys[1],
                        "metric": keys[2],
                        "intervention_target": keys[3],
                        "higher_strength": higher,
                        "lower_strength": lower,
                        "mean_difference": float(differences.mean()),
                        "p_signflip": _exact_sign_flip_greater(differences),
                        "p_sign": sign_p,
                        "positive_seeds": positives,
                        "nonzero_seeds": nonzero,
                        **{
                            key: value
                            for key, value in _mean_interval(differences).items()
                            if key in {"std", "ci_low", "ci_high", "n_seeds"}
                        },
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    if not contrasts.empty:
        contrasts = _apply_holm(
            contrasts,
            groups=["metric"],
            p_columns=["p_signflip", "p_sign"],
        )
    return target_seed, pd.DataFrame(summary_rows), contrasts


def _system_group_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build seed-first process-versus-measurement summaries."""
    seed = (
        frame.groupby(["model", "strategy", "metric", "system_group", "seed"], sort=True)["value"]
        .mean()
        .reset_index()
    )
    summary_rows = []
    for keys, group in seed.groupby(
        ["model", "strategy", "metric", "system_group"],
        sort=True,
    ):
        summary_rows.append(
            {
                "model": keys[0],
                "strategy": keys[1],
                "metric": keys[2],
                "system_group": keys[3],
                **_mean_interval(group["value"]),
            }
        )
    contrast_rows = []
    for keys, group in seed.groupby(["model", "strategy", "metric"], sort=True):
        pivot = group.pivot(index="seed", columns="system_group", values="value")
        differences = (pivot["process"] - pivot["measurement"]).to_numpy(dtype=float)
        sign_p, positives, nonzero = _exact_sign_test_greater(differences)
        contrast_rows.append(
            {
                "model": keys[0],
                "strategy": keys[1],
                "metric": keys[2],
                "strategy_left": "process",
                "strategy_right": "measurement",
                "mean_difference": float(differences.mean()),
                "p_signflip": _exact_sign_flip_greater(differences),
                "p_sign": sign_p,
                "positive_seeds": positives,
                "nonzero_seeds": nonzero,
                **{
                    key: value
                    for key, value in _mean_interval(differences).items()
                    if key in {"std", "ci_low", "ci_high", "n_seeds"}
                },
            }
        )
    return seed, pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def _top_fraction(values: pd.Series, fraction: float = 0.2) -> set[str]:
    """Return candidate identifiers in the highest-valued fraction."""
    count = max(1, int(math.ceil(len(values) * fraction)))
    return set(values.sort_values(ascending=False, kind="stable").head(count).index.astype(str))


def _pairing_robustness(
    candidate_metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare candidate rankings across frozen pairing sensitivity variants."""
    candidate_required = {"model", "seed", "candidate_id", "strategy", "value"}
    sensitivity_required = {"model", "seed", "candidate_id", "variant", "value"}
    if missing := candidate_required - set(candidate_metrics.columns):
        raise ValueError(f"Candidate metrics are missing: {', '.join(sorted(missing))}.")
    if missing := sensitivity_required - set(sensitivity.columns):
        raise ValueError(f"Pairing sensitivity is missing: {', '.join(sorted(missing))}.")
    primary_map = {
        "cap_random_seed": "cap_random",
        "cap_encoder_seed": "cap_encoder_nearest",
    }
    seed_rows = []
    for keys, variant_rows in sensitivity.groupby(["model", "seed", "variant"], sort=True):
        model, seed, variant = keys
        prefix = next((name for name in primary_map if str(variant).startswith(name)), None)
        if prefix is None:
            raise ValueError(f"Unknown pairing sensitivity variant: {variant}")
        primary_strategy = primary_map[prefix]
        primary = candidate_metrics[
            (candidate_metrics["model"].astype(str) == str(model))
            & (candidate_metrics["seed"].astype(int) == int(seed))
            & (candidate_metrics["strategy"].astype(str) == primary_strategy)
        ][["candidate_id", "value"]]
        secondary = variant_rows[["candidate_id", "value"]]
        merged = primary.merge(
            secondary,
            on="candidate_id",
            suffixes=("_primary", "_variant"),
            validate="one_to_one",
        )
        if len(merged) != len(primary) or len(merged) != len(secondary):
            raise ValueError(f"Candidate coverage mismatch for pairing variant {variant}.")
        correlation = stats.spearmanr(
            merged["value_primary"],
            merged["value_variant"],
        ).statistic
        indexed = merged.set_index(merged["candidate_id"].astype(str))
        primary_top = _top_fraction(indexed["value_primary"])
        variant_top = _top_fraction(indexed["value_variant"])
        union = primary_top | variant_top
        seed_rows.append(
            {
                "model": str(model),
                "seed": int(seed),
                "variant": str(variant),
                "primary_strategy": primary_strategy,
                "n_candidates": len(merged),
                "spearman": float(correlation),
                "winner_agreement": bool(
                    merged.loc[merged["value_primary"].idxmax(), "candidate_id"]
                    == merged.loc[merged["value_variant"].idxmax(), "candidate_id"]
                ),
                "top20_jaccard": len(primary_top & variant_top) / len(union),
            }
        )
    seed_frame = pd.DataFrame(seed_rows)
    summary = (
        seed_frame.groupby(["model", "variant", "primary_strategy"], sort=True)
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_min=("spearman", "min"),
            winner_agreement_rate=("winner_agreement", "mean"),
            top20_jaccard_mean=("top20_jaccard", "mean"),
            n_development_seeds=("seed", "nunique"),
            n_candidates=("n_candidates", "min"),
        )
        .reset_index()
    )
    return seed_frame, summary


def _plot_forest(contrasts: pd.DataFrame, path: Path) -> None:
    """Plot prespecified seed-level contrast estimates and intervals."""
    ordered = contrasts.sort_values(
        ["metric", "contrast_id", "model"],
        kind="stable",
    ).reset_index(drop=True)
    labels = ordered["metric"] + " · " + ordered["model"] + " · " + ordered["contrast_id"]
    y = np.arange(len(ordered))
    mean = ordered["mean_difference"].to_numpy(dtype=float)
    low = mean - ordered["ci_low"].to_numpy(dtype=float)
    high = ordered["ci_high"].to_numpy(dtype=float) - mean
    figure, axis = plt.subplots(figsize=(9.0, max(5.0, 0.28 * len(ordered))))
    axis.errorbar(mean, y, xerr=np.vstack([low, high]), fmt="o", capsize=3)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Paired seed-first mean difference (left − right)")
    axis.set_title("Prespecified strategy contrasts (95% t intervals across seeds)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_ordered_heatmaps(
    seed_summary: pd.DataFrame,
    plan: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Plot explicitly exploratory outcome-ordered strategy differences."""
    paths = []
    for metric in plan["metrics"]:
        selected = seed_summary[seed_summary["metric"] == metric]
        overall = (
            selected.groupby("strategy", sort=True)["value"].mean().sort_values(ascending=False)
        )
        strategy_order = overall.index.tolist()
        figure, axes = plt.subplots(
            1,
            len(plan["models"]),
            figsize=(4.1 * len(plan["models"]), 4.2),
            squeeze=False,
        )
        for axis, model in zip(axes[0], plan["models"]):
            means = (
                selected[selected["model"] == model]
                .groupby("strategy", sort=True)["value"]
                .mean()
                .reindex(strategy_order)
            )
            matrix = means.to_numpy()[:, None] - means.to_numpy()[None, :]
            bound = max(float(np.abs(matrix).max()), 1.0e-6)
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-bound, vmax=bound)
            axis.set_xticks(
                np.arange(len(strategy_order)), strategy_order, rotation=45, ha="right"
            )
            axis.set_yticks(np.arange(len(strategy_order)), strategy_order)
            axis.set_title(str(model))
            figure.colorbar(image, ax=axis, fraction=0.046, label="row − column")
        figure.suptitle(
            f"EXPLORATORY outcome-ordered strategy differences · {metric}",
            fontweight="bold",
        )
        figure.tight_layout()
        path = output_dir / f"exploratory_{metric}_ordered_strategy_differences.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _plot_strength_panels(
    target_seed: pd.DataFrame,
    plan: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Plot complementary seed-first strength summaries."""
    paths = []
    order = list(map(str, plan["strength_order"]))
    for metric in plan["metrics"]:
        selected = target_seed[target_seed["metric"] == metric]
        figure, axes = plt.subplots(
            1,
            len(plan["models"]),
            figsize=(4.1 * len(plan["models"]), 4.0),
            squeeze=False,
            sharey=True,
        )
        for axis, model in zip(axes[0], plan["models"]):
            model_frame = selected[selected["model"] == model]
            descriptive = (
                model_frame.groupby(["strategy", "strength", "seed"], sort=True)["value"]
                .mean()
                .reset_index()
            )
            summary = (
                descriptive.groupby(["strategy", "strength"], sort=True)["value"]
                .mean()
                .unstack("strength")
                .reindex(columns=order)
            )
            for strategy, row in summary.iterrows():
                axis.plot(order, row.to_numpy(dtype=float), marker="o", label=strategy)
            axis.set_title(str(model))
            axis.grid(axis="y", alpha=0.25)
            axis.set_xlabel("Intervention strength")
        axes[0, 0].set_ylabel(metric)
        axes[0, -1].legend(fontsize=7, loc="best")
        figure.suptitle(f"COMPLEMENTARY strength summary (seed first, target averaged) · {metric}")
        figure.tight_layout()
        path = output_dir / f"complementary_{metric}_strength_panels.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _optional_status(
    campaign_root: Path,
    integrity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate declared optional artifacts and mark absent components pending."""
    records = integrity.get("optional_artifacts", {})
    if records is None:
        records = {}
    if not isinstance(records, dict):
        raise ValueError("optional_artifacts must be a JSON object.")
    statuses: dict[str, Any] = {}
    resolved: dict[str, Path] = {}
    for name in (
        "candidate_metrics",
        "pairing_proxy_sensitivity",
        "background_acceptance_diagnostics",
        "candidate_audit_results",
    ):
        if name not in records:
            statuses[name] = {
                "status": "pending",
                "reason": "not declared in the frozen integrity manifest",
            }
            continue
        path = _resolve_input(campaign_root, records[name], name)
        resolved[name] = path
        statuses[name] = {
            "status": "available",
            "path": str(path),
            "sha256": _sha256(path),
        }
    return statuses, resolved


def _write_report(
    path: Path,
    campaign: Mapping[str, Any],
    contrasts: pd.DataFrame,
    statuses: Mapping[str, Any],
) -> None:
    """Write a concise report with inferential and pending-component boundaries."""
    lines = [
        "# Causal Chamber paper analysis",
        "",
        f"- Campaign: `{campaign['campaign_id']}`",
        f"- Reporting seeds: {contrasts['n_seeds'].min()} paired seeds per contrast",
        "- Confirmatory unit: the reporting seed after averaging all interventions.",
        "- Intervention rows are not treated as independent replicates.",
        "",
        "## Prespecified contrasts",
        "",
        (
            "The confirmatory table uses exact paired sign-flip tests with Holm correction. "
            "Exact paired sign tests are reported as a robustness sensitivity."
        ),
        "",
        "## Component status",
        "",
    ]
    for name, record in statuses.items():
        lines.append(f"- `{name}`: **{record['status']}**")
        if record.get("reason"):
            lines.append(f"  - {record['reason']}")
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Outcome-ordered strategy-difference heatmaps are explicitly exploratory.",
            "- Strength and process-versus-measurement summaries are complementary.",
            "- No background-acceptance or candidate-audit conclusion is inferred from absence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    campaign_root: Path,
    analysis_plan: Path,
    taxonomy_path: Path,
    integrity_manifest: Path,
    output_dir: Path,
) -> list[Path]:
    """Validate frozen inputs and write the standalone paper-analysis bundle."""
    campaign_root = campaign_root.expanduser().resolve()
    campaign, plan, _, results, integrity, input_paths = _verify_inputs(
        campaign_root,
        analysis_plan,
        taxonomy_path,
        integrity_manifest,
    )
    statuses, optional_paths = _optional_status(campaign_root, integrity)
    output_dir = _prepare_output_dir(output_dir, campaign_root)

    seed_summary, summary = _seed_first_summary(results)
    contrasts = _contrast_rows(seed_summary, plan)
    target_seed, target_summary, strength_contrasts = _strength_outputs(
        results,
        plan["strength_order"],
    )
    system_seed, system_summary, system_contrasts = _system_group_outputs(results)

    outputs: list[Path] = []
    tables = {
        "seed_first_summary.csv": seed_summary,
        "strategy_summary.csv": summary,
        "prespecified_strategy_contrasts.csv": contrasts,
        "target_strength_seed_summary.csv": target_seed,
        "target_strength_summary.csv": target_summary,
        "within_target_strength_contrasts.csv": strength_contrasts,
        "system_group_seed_summary.csv": system_seed,
        "process_measurement_summary.csv": system_summary,
        "process_measurement_contrasts.csv": system_contrasts,
    }
    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)

    if {
        "candidate_metrics",
        "pairing_proxy_sensitivity",
    }.issubset(optional_paths):
        seed_robustness, robustness = _pairing_robustness(
            pd.read_csv(optional_paths["candidate_metrics"], dtype={"candidate_id": str}),
            pd.read_csv(
                optional_paths["pairing_proxy_sensitivity"],
                dtype={"candidate_id": str},
            ),
        )
        statuses["pairing_robustness_analysis"] = {
            "status": "completed",
            "source": "frozen candidate and pairing-proxy sensitivity tables",
        }
        for name, table in (
            ("pairing_robustness_by_seed.csv", seed_robustness),
            ("pairing_robustness_summary.csv", robustness),
        ):
            path = output_dir / name
            table.to_csv(path, index=False)
            outputs.append(path)
    else:
        statuses["pairing_robustness_analysis"] = {
            "status": "pending",
            "reason": "candidate_metrics and pairing_proxy_sensitivity are both required",
        }

    forest = output_dir / "confirmatory_contrast_forest.png"
    _plot_forest(contrasts, forest)
    outputs.append(forest)
    outputs.extend(_plot_ordered_heatmaps(seed_summary, plan, output_dir))
    outputs.extend(_plot_strength_panels(target_seed, plan, output_dir))

    status_path = output_dir / "component_status.json"
    _atomic_json(status_path, {"schema_version": 1, "components": statuses})
    outputs.append(status_path)
    catalog_rows = [
        {
            "artifact": path.name,
            "classification": _artifact_classification(path),
        }
        for path in outputs
    ]
    catalog = output_dir / "analysis_catalog.csv"
    pd.DataFrame(catalog_rows).to_csv(catalog, index=False)
    outputs.append(catalog)

    report = output_dir / "report.md"
    _write_report(report, campaign, contrasts, statuses)
    outputs.append(report)

    provenance = output_dir / "analysis_provenance.json"
    _atomic_json(
        provenance,
        {
            "schema_version": 1,
            "campaign_id": campaign["campaign_id"],
            "inputs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in input_paths.items()
            },
            "integrity_manifest": {
                "path": str(integrity_manifest.expanduser().resolve()),
                "sha256": _sha256(integrity_manifest.expanduser().resolve()),
            },
            "analysis_contract": {
                "replicate_unit": "reporting_seed",
                "intervention_aggregation": "arithmetic mean within seed before inference",
                "multiplicity": "Holm within metric and prespecified contrast family",
                "sign_sensitivity": "exact paired sign test excluding zero differences",
                "outcome_ordered_views": "exploratory",
            },
            "outputs": {path.name: _sha256(path) for path in outputs},
        },
    )
    outputs.append(provenance)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse standalone analysis command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--integrity-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone analysis command."""
    args = parse_args(argv)
    outputs = analyze(
        args.campaign_root,
        args.analysis_plan,
        args.taxonomy,
        args.integrity_manifest,
        args.output_dir,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
