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
The deployed design may derive this CSV from its frozen target-level JSON, but this
executor requires the derived CSV itself to be integrity-pinned.  Physical displays use
``system_group,semantic_family,intervention_target,strength`` order, with process before
measurement and weak before mid before strong.
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
EXPECTED_MODELS = ("ae", "vae", "svdd", "realnvp")
TARGET_COLUMNS = ("system_group", "semantic_family", "intervention_target")
STRENGTH_ORDER = ("weak", "mid", "strong")
EQUIVALENCE_MARGIN_AUPRC = 0.02
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
    if tuple(map(str, plan["models"])) != EXPECTED_MODELS:
        raise ValueError(
            "Analysis plan models must be ['ae', 'vae', 'svdd', 'realnvp'] "
            "for the prespecified four-model Holm family."
        )
    if len(plan["reporting_seeds"]) < 2:
        raise ValueError("At least two paired reporting seeds are required for inference.")
    metrics = [str(value) for value in _validate_sequence("metrics", plan.get("metrics"))]
    if metrics != ["auprc", "efficiency_operational"]:
        raise ValueError(
            "Analysis plan metrics must be ['auprc', 'efficiency_operational'] in that order."
        )
    strength_order = [
        str(value) for value in _validate_sequence("strength_order", plan.get("strength_order"))
    ]
    if strength_order != list(STRENGTH_ORDER):
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
    taxonomy["_system_order"] = taxonomy["system_group"].map({"process": 0, "measurement": 1})
    taxonomy["_strength_order"] = taxonomy["strength"].map(
        {name: index for index, name in enumerate(STRENGTH_ORDER)}
    )
    taxonomy = taxonomy.sort_values(
        [
            "_system_order",
            "semantic_family",
            "intervention_target",
            "_strength_order",
            "intervention",
        ],
        kind="stable",
    ).reset_index(drop=True)
    taxonomy["taxonomy_order"] = np.arange(len(taxonomy), dtype=int)
    return taxonomy.drop(columns=["_system_order", "_strength_order"])


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
        or "background_acceptance" in name
        or "candidate_rank" in name
        or "family_seed_summary" in name
        or "target_seed_summary" in name
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
    raw_results = pd.read_csv(paths["results"])
    threshold_columns = {
        "manifest_index",
        "checkpoint_sha256",
        "threshold_manifest_sha256",
        "threshold_artifact",
        "threshold_artifact_sha256",
        "threshold_bytes_sha256",
    }
    present_threshold_columns = threshold_columns & set(raw_results.columns)
    if present_threshold_columns and present_threshold_columns != threshold_columns:
        missing_threshold = threshold_columns - set(raw_results.columns)
        raise ValueError(
            "Threshold-safe results omit provenance columns: "
            + ", ".join(sorted(missing_threshold))
        )
    if present_threshold_columns:
        required_sidecar = {"threshold_manifest", "threshold_safe_provenance"}
        missing_sidecar = required_sidecar - set(files)
        if missing_sidecar:
            raise ValueError(
                "Threshold-safe results require integrity entries: "
                + ", ".join(sorted(missing_sidecar))
            )
        for name in sorted(required_sidecar):
            paths[name] = _resolve_input(campaign_root, files[name], name)
        threshold_manifest_sha = _sha256(paths["threshold_manifest"])
        if set(raw_results["threshold_manifest_sha256"].astype(str)) != {threshold_manifest_sha}:
            raise ValueError("Result rows do not reference the pinned threshold manifest.")
        sidecar = _read_json(paths["threshold_safe_provenance"])
        if (
            sidecar.get("threshold_manifest_sha256") != threshold_manifest_sha
            or sidecar.get("results_sha256") != _sha256(paths["results"])
            or int(sidecar.get("expected_records", -1)) != 200
            or int(sidecar.get("expected_result_rows", -1)) != 23_200
            or len(raw_results) != 23_200
        ):
            raise ValueError("Threshold-safe result provenance chain is inconsistent.")
        threshold_manifest = _read_json(paths["threshold_manifest"])
        threshold_records = threshold_manifest.get("records")
        if (
            int(threshold_manifest.get("expected_records", -1)) != 200
            or not isinstance(threshold_records, list)
            or len(threshold_records) != 200
            or set(raw_results["manifest_index"].astype(int)) != set(range(200))
            or not (raw_results.groupby("manifest_index", sort=False).size() == 116).all()
        ):
            raise ValueError("Threshold-safe manifest/result index coverage is not exact.")
        manifest_frame = pd.DataFrame(threshold_records)
        required_manifest_columns = {
            "manifest_index",
            "threshold_artifact",
            "threshold_artifact_sha256",
            "checkpoint_sha256",
            "threshold_bytes_sha256",
        }
        if (
            not required_manifest_columns.issubset(manifest_frame.columns)
            or manifest_frame["manifest_index"].duplicated().any()
        ):
            raise ValueError("Threshold manifest records are not uniquely joinable.")
        row_identity = (
            raw_results.loc[:, sorted(required_manifest_columns)]
            .drop_duplicates()
            .sort_values("manifest_index")
            .reset_index(drop=True)
        )
        manifest_identity = (
            manifest_frame.loc[:, sorted(required_manifest_columns)]
            .sort_values("manifest_index")
            .reset_index(drop=True)
        )
        if not row_identity.equals(manifest_identity):
            raise ValueError("Every result index must exactly join its threshold manifest record.")
    results = _validate_results(raw_results, plan, taxonomy)
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


def _equal_unit_seed_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build seed-first summaries giving each family or target equal weight."""
    keys = ["model", "strategy", "seed", "metric"]
    family_seed = (
        frame.groupby([*keys, "system_group", "semantic_family"], sort=True)["value"]
        .mean()
        .reset_index()
    )
    equal_family_seed = family_seed.groupby(keys, sort=True)["value"].mean().reset_index()
    target_seed = frame.groupby([*keys, *TARGET_COLUMNS], sort=True)["value"].mean().reset_index()
    equal_target_seed = target_seed.groupby(keys, sort=True)["value"].mean().reset_index()
    return {
        "family_seed_summary.csv": family_seed,
        "equal_family_seed_summary.csv": equal_family_seed,
        "target_seed_summary.csv": target_seed,
        "equal_target_seed_summary.csv": equal_target_seed,
    }


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


def _metadata_encoder_equivalence(
    seed_summary: pd.DataFrame,
    plan: Mapping[str, Any],
) -> pd.DataFrame:
    """Run the prespecified paired AUPRC TOST with four-model Holm control."""
    auprc = seed_summary[seed_summary["metric"] == "auprc"]
    expected_seeds = list(map(int, plan["reporting_seeds"]))
    rows: list[dict[str, Any]] = []
    for model in plan["models"]:
        pivot = auprc[auprc["model"] == model].pivot(
            index="seed",
            columns="strategy",
            values="value",
        )
        if list(pivot.index) != expected_seeds:
            raise ValueError(f"Seed pairing failed for {model}/auprc equivalence.")
        differences = (pivot["cap_metadata_nearest"] - pivot["cap_encoder_nearest"]).to_numpy(
            dtype=float
        )
        n_seeds = len(differences)
        mean = float(differences.mean())
        std = float(differences.std(ddof=1))
        standard_error = std / math.sqrt(n_seeds)
        if standard_error == 0.0:
            p_lower = 0.0 if mean > -EQUIVALENCE_MARGIN_AUPRC else 1.0
            p_upper = 0.0 if mean < EQUIVALENCE_MARGIN_AUPRC else 1.0
            ci_low = ci_high = mean
        else:
            degrees_freedom = n_seeds - 1
            p_lower = float(
                stats.t.sf(
                    (mean + EQUIVALENCE_MARGIN_AUPRC) / standard_error,
                    degrees_freedom,
                )
            )
            p_upper = float(
                stats.t.cdf(
                    (mean - EQUIVALENCE_MARGIN_AUPRC) / standard_error,
                    degrees_freedom,
                )
            )
            critical = float(stats.t.ppf(0.95, degrees_freedom))
            ci_low = mean - critical * standard_error
            ci_high = mean + critical * standard_error
        rows.append(
            {
                "model": str(model),
                "metric": "auprc",
                "test_family": "metadata_encoder_equivalence_four_models",
                "strategy_left": "cap_metadata_nearest",
                "strategy_right": "cap_encoder_nearest",
                "difference": "metadata_minus_encoder",
                "equivalence_margin": EQUIVALENCE_MARGIN_AUPRC,
                "mean_difference": mean,
                "std_difference": std,
                "ci90_unadjusted_low": ci_low,
                "ci90_unadjusted_high": ci_high,
                "ci_level": 0.90,
                "ci_multiplicity_adjustment": "none",
                "p_tost_lower_unadjusted": p_lower,
                "p_tost_upper_unadjusted": p_upper,
                "p_tost_unadjusted": max(p_lower, p_upper),
                "equivalent_tost_unadjusted_0.05": max(p_lower, p_upper) < 0.05,
                "n_paired_seeds": n_seeds,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(EXPECTED_MODELS):
        raise ValueError("The equivalence Holm family must contain exactly four models.")
    result["p_tost_holm_four_models"] = _holm_adjust(result["p_tost_unadjusted"])
    result["equivalent_tost_holm_0.05"] = result["p_tost_holm_four_models"] < 0.05
    result["holm_family_size"] = len(EXPECTED_MODELS)
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
                *TARGET_COLUMNS,
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
        ["model", "strategy", "metric", *TARGET_COLUMNS, "strength"],
        sort=True,
    ):
        summary_rows.append(
            {
                "model": keys[0],
                "strategy": keys[1],
                "metric": keys[2],
                "system_group": keys[3],
                "semantic_family": keys[4],
                "intervention_target": keys[5],
                "strength": keys[6],
                **_mean_interval(group["value"]),
            }
        )
    contrast_rows = []
    rank = {name: index for index, name in enumerate(strength_order)}
    for keys, group in target_seed.groupby(
        ["model", "strategy", "metric", *TARGET_COLUMNS],
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
                        "system_group": keys[3],
                        "semantic_family": keys[4],
                        "intervention_target": keys[5],
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


def _strength_panel_outputs(
    target_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select composition-stable target sets and average targets within seed."""
    target_strengths = (
        target_seed.loc[:, [*TARGET_COLUMNS, "strength"]]
        .drop_duplicates()
        .groupby(list(TARGET_COLUMNS), sort=True)["strength"]
        .agg(lambda values: frozenset(map(str, values)))
        .reset_index(name="available_strengths")
    )
    complete = frozenset(STRENGTH_ORDER)
    mid_strong = frozenset(("mid", "strong"))
    if int((target_strengths["available_strengths"] == complete).sum()) != 9:
        raise ValueError(
            "Frozen taxonomy must contain exactly nine complete weak/mid/strong targets."
        )
    eligibility_rows = []
    for record in target_strengths.to_dict("records"):
        values = record["available_strengths"]
        panels = []
        if values == complete:
            panels.append("complete_weak_mid_strong")
        if mid_strong.issubset(values):
            panels.append("mid_strong_all")
        if not panels:
            panels.append("excluded_incomplete")
        for panel in panels:
            eligibility_rows.append(
                {
                    **{column: record[column] for column in TARGET_COLUMNS},
                    "available_strengths": ",".join(
                        strength for strength in STRENGTH_ORDER if strength in values
                    ),
                    "panel": panel,
                }
            )
    eligibility = pd.DataFrame(eligibility_rows)
    if not (eligibility["panel"] == "mid_strong_all").any():
        raise ValueError("Frozen taxonomy must contain targets with both mid and strong levels.")

    included = eligibility[
        eligibility["panel"].isin(["complete_weak_mid_strong", "mid_strong_all"])
    ].drop(columns="available_strengths")
    panel_targets = target_seed.merge(
        included,
        on=list(TARGET_COLUMNS),
        how="inner",
        validate="many_to_many",
    )
    panel_targets = panel_targets[
        (panel_targets["panel"] != "mid_strong_all")
        | panel_targets["strength"].isin(["mid", "strong"])
    ]
    grouping = ["model", "strategy", "metric", "seed", "panel", "strength"]
    panel_seed = (
        panel_targets.groupby(grouping, sort=True)["value"]
        .agg(value="mean", n_targets="size")
        .reset_index()
    )
    expected_counts = included.groupby("panel", sort=True).size().rename("expected_targets")
    panel_seed = panel_seed.merge(
        expected_counts,
        on="panel",
        how="left",
        validate="many_to_one",
    )
    if not (panel_seed["n_targets"] == panel_seed["expected_targets"]).all():
        raise ValueError("Strength panels do not have fixed target composition within seed.")
    panel_seed = panel_seed.drop(columns="expected_targets")

    rows = []
    for keys, group in panel_seed.groupby(
        ["model", "strategy", "metric", "panel", "strength"],
        sort=True,
    ):
        rows.append(
            {
                "model": keys[0],
                "strategy": keys[1],
                "metric": keys[2],
                "panel": keys[3],
                "strength": keys[4],
                "n_targets": int(group["n_targets"].iloc[0]),
                **_mean_interval(group["value"]),
            }
        )
    return eligibility, panel_seed, pd.DataFrame(rows)


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


def _intervention_contrast_outputs(
    frame: pd.DataFrame,
    plan: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute seed-paired CAP-minus-baseline values in frozen taxonomy order."""
    identity = [
        "model",
        "metric",
        "seed",
        "intervention",
        *TARGET_COLUMNS,
        "strength",
        "taxonomy_order",
    ]
    pivot = frame.pivot(index=identity, columns="strategy", values="value").reset_index()
    seed_frames = []
    for contrast in plan["contrasts"]:
        table = pivot.loc[:, identity].copy()
        table["contrast_id"] = str(contrast["id"])
        table["strategy_left"] = str(contrast["left"])
        table["strategy_right"] = str(contrast["right"])
        table["value"] = (pivot[str(contrast["left"])] - pivot[str(contrast["right"])]).to_numpy(
            dtype=float
        )
        seed_frames.append(table)
    seed = pd.concat(seed_frames, ignore_index=True)

    summary_keys = [
        "model",
        "metric",
        "intervention",
        *TARGET_COLUMNS,
        "strength",
        "taxonomy_order",
        "contrast_id",
        "strategy_left",
        "strategy_right",
    ]
    rows = []
    for keys, group in seed.groupby(summary_keys, sort=False):
        interval = _mean_interval(group["value"])
        mean_difference = interval.pop("mean")
        rows.append(
            {name: value for name, value in zip(summary_keys, keys)}
            | {"mean_difference": mean_difference}
            | interval
        )
    summary = pd.DataFrame(rows)
    model_rank = {name: index for index, name in enumerate(plan["models"])}
    metric_rank = {name: index for index, name in enumerate(plan["metrics"])}
    contrast_rank = {
        str(contrast["id"]): index for index, contrast in enumerate(plan["contrasts"])
    }
    for table in (seed, summary):
        table["_model_order"] = table["model"].map(model_rank)
        table["_metric_order"] = table["metric"].map(metric_rank)
        table["_contrast_order"] = table["contrast_id"].map(contrast_rank)
        table.sort_values(
            [
                "_metric_order",
                "_model_order",
                "taxonomy_order",
                "_contrast_order",
                *(["seed"] if "seed" in table.columns else []),
            ],
            kind="stable",
            inplace=True,
        )
        table.drop(
            columns=["_model_order", "_metric_order", "_contrast_order"],
            inplace=True,
        )
        table.reset_index(drop=True, inplace=True)
    return seed, summary


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


def _plot_intervention_heatmaps(
    summary: pd.DataFrame,
    taxonomy: pd.DataFrame,
    plan: Mapping[str, Any],
    path: Path,
) -> None:
    """Plot intervention-level CAP contrasts in frozen physical taxonomy order."""
    ordered_taxonomy = taxonomy.sort_values("taxonomy_order", kind="stable")
    intervention_order = ordered_taxonomy["intervention"].tolist()
    contrast_order = [str(contrast["id"]) for contrast in plan["contrasts"]]
    maximum = max(float(summary["mean_difference"].abs().max()), 1.0e-6)
    figure, axes = plt.subplots(
        len(plan["metrics"]),
        len(plan["models"]),
        figsize=(4.2 * len(plan["models"]), max(12.0, 0.15 * len(taxonomy) * 2)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    image = None
    group_columns = ["system_group", "semantic_family", "intervention_target"]
    group_keys = ordered_taxonomy[group_columns].apply(tuple, axis=1).tolist()
    boundaries = [
        index - 0.5
        for index in range(1, len(group_keys))
        if group_keys[index] != group_keys[index - 1]
    ]
    for row, metric in enumerate(plan["metrics"]):
        for column, model in enumerate(plan["models"]):
            axis = axes[row, column]
            selected = summary[(summary["metric"] == metric) & (summary["model"] == model)]
            matrix = (
                selected.pivot(
                    index="intervention",
                    columns="contrast_id",
                    values="mean_difference",
                )
                .reindex(index=intervention_order, columns=contrast_order)
                .to_numpy(dtype=float)
            )
            if not np.isfinite(matrix).all():
                raise ValueError(f"Incomplete intervention heatmap for {model}/{metric}.")
            image = axis.imshow(
                matrix,
                aspect="auto",
                cmap="coolwarm",
                vmin=-maximum,
                vmax=maximum,
            )
            for boundary in boundaries:
                axis.axhline(boundary, color="black", linewidth=0.25, alpha=0.5)
            axis.set_title(f"{model} · {metric}")
            axis.set_xticks(
                np.arange(len(contrast_order)),
                contrast_order,
                rotation=45,
                ha="right",
            )
            if column == 0:
                axis.set_yticks(
                    np.arange(len(intervention_order)),
                    intervention_order,
                    fontsize=6,
                )
            else:
                axis.tick_params(labelleft=False)
    if image is None:
        raise ValueError("No intervention contrasts are available for plotting.")
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.015,
        pad=0.01,
        label="CAP − baseline, seed-first mean",
    )
    figure.suptitle(
        "Prespecified intervention-level CAP contrasts "
        "(process → measurement; family, target, weak → strong)",
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.16,
        right=0.94,
        bottom=0.12,
        top=0.95,
        hspace=0.12,
        wspace=0.08,
    )
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
    panel_summary: pd.DataFrame,
    plan: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Plot composition-stable within-target strength trajectories."""
    paths = []
    panels = (
        ("complete_weak_mid_strong", list(STRENGTH_ORDER)),
        ("mid_strong_all", ["mid", "strong"]),
    )
    for metric in plan["metrics"]:
        selected = panel_summary[panel_summary["metric"] == metric]
        figure, axes = plt.subplots(
            len(panels),
            len(plan["models"]),
            figsize=(4.1 * len(plan["models"]), 7.5),
            squeeze=False,
            sharey="row",
        )
        for panel_row, (panel, strength_order) in enumerate(panels):
            for model_column, model in enumerate(plan["models"]):
                axis = axes[panel_row, model_column]
                model_frame = selected[(selected["model"] == model) & (selected["panel"] == panel)]
                if model_frame.empty:
                    raise ValueError(f"No strength-panel data for {model}/{metric}/{panel}.")
                n_targets = int(model_frame["n_targets"].iloc[0])
                for strategy in plan["strategies"]:
                    trajectory = (
                        model_frame[model_frame["strategy"] == strategy]
                        .set_index("strength")["mean"]
                        .reindex(strength_order)
                    )
                    if trajectory.isna().any():
                        raise ValueError(
                            f"Incomplete within-target trajectory for "
                            f"{model}/{metric}/{panel}/{strategy}."
                        )
                    axis.plot(
                        strength_order,
                        trajectory.to_numpy(dtype=float),
                        marker="o",
                        label=strategy,
                    )
                label = (
                    "complete weak/mid/strong"
                    if panel == "complete_weak_mid_strong"
                    else "all mid/strong-complete"
                )
                axis.set_title(f"{model} · {label} · {n_targets} targets")
                axis.grid(axis="y", alpha=0.25)
                axis.set_xlabel("Intervention strength")
                if model_column == 0:
                    axis.set_ylabel(metric)
        axes[0, -1].legend(fontsize=7, loc="best")
        figure.suptitle(
            f"COMPLEMENTARY within-target strength trajectories "
            f"(equal target weight within seed) · {metric}"
        )
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
        "candidate_audit_provenance",
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


def _background_acceptance_outputs(
    path: Path,
    plan: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and summarize one operating-point estimate per reporting seed."""
    required = {
        "model",
        "strategy",
        "seed",
        "manifest_index",
        "test_normal_count",
        "triggered_count",
        "achieved_test_normal_acceptance",
        "target_fpr",
        "achieved_minus_target_fpr",
        "wilson_95_ci_low",
        "wilson_95_ci_high",
    }
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Background diagnostics are missing columns: " + ", ".join(sorted(missing))
        )
    frame = frame.loc[:, sorted(required)].copy()
    for column in ("model", "strategy"):
        frame[column] = frame[column].astype(str)
    for column in ("seed", "manifest_index", "test_normal_count", "triggered_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    numeric = sorted(
        required
        - {
            "model",
            "strategy",
            "seed",
            "manifest_index",
            "test_normal_count",
            "triggered_count",
        }
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Background diagnostics contain non-finite values.")
    keys = ["model", "strategy", "seed"]
    expected = pd.MultiIndex.from_product(
        [
            list(map(str, plan["models"])),
            list(map(str, plan["strategies"])),
            list(map(int, plan["reporting_seeds"])),
        ],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(frame[keys])
    if (
        frame.duplicated(keys).any()
        or len(expected.difference(observed))
        or len(observed.difference(expected))
        or frame["manifest_index"].duplicated().any()
    ):
        raise ValueError("Background diagnostics do not have exact model/strategy/seed coverage.")
    if not (
        (frame["test_normal_count"] > 0)
        & (frame["triggered_count"] >= 0)
        & (frame["triggered_count"] <= frame["test_normal_count"])
        & frame["achieved_test_normal_acceptance"].between(0.0, 1.0)
        & frame["target_fpr"].between(0.0, 1.0)
        & frame["wilson_95_ci_low"].between(0.0, 1.0)
        & frame["wilson_95_ci_high"].between(0.0, 1.0)
    ).all():
        raise ValueError("Background diagnostic counts or probabilities are invalid.")
    achieved = frame["triggered_count"] / frame["test_normal_count"]
    if (
        not np.allclose(achieved, frame["achieved_test_normal_acceptance"], atol=1e-12)
        or not np.allclose(
            frame["achieved_test_normal_acceptance"] - frame["target_fpr"],
            frame["achieved_minus_target_fpr"],
            atol=1e-12,
        )
        or not (frame["wilson_95_ci_low"] <= frame["achieved_test_normal_acceptance"]).all()
        or not (frame["achieved_test_normal_acceptance"] <= frame["wilson_95_ci_high"]).all()
    ):
        raise ValueError("Background diagnostic arithmetic or intervals are inconsistent.")

    rows = []
    for (model, strategy), group in frame.groupby(["model", "strategy"], sort=False):
        interval = _mean_interval(group["achieved_test_normal_acceptance"])
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "target_fpr": float(group["target_fpr"].iloc[0]),
                "mean_acceptance": interval["mean"],
                "std_across_seeds": interval["std"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "mean_minus_target_fpr": float(group["achieved_minus_target_fpr"].mean()),
                "max_abs_minus_target_fpr": float(group["achieved_minus_target_fpr"].abs().max()),
                "n_reporting_seeds": interval["n_seeds"],
                "total_test_normal_events": int(group["test_normal_count"].sum()),
                "event_pooling_for_inference": False,
            }
        )
    return frame.sort_values(keys, kind="stable"), pd.DataFrame(rows)


def _candidate_rank_outputs(
    results_path: Path,
    provenance_path: Path,
    plan: Mapping[str, Any],
) -> pd.DataFrame:
    """Authenticate and validate the prespecified candidate-rank association table."""
    provenance = _read_json(provenance_path)
    outputs = provenance.get("outputs")
    if (
        int(provenance.get("schema_version", -1)) != 1
        or not isinstance(outputs, dict)
        or outputs.get(results_path.name) != _sha256(results_path)
        or int(provenance.get("n_permutations", -1)) != 10_000
        or int(provenance.get("n_bootstrap_requested", -1)) != 10_000
    ):
        raise ValueError("Candidate-rank provenance is incomplete or inconsistent.")
    frame = pd.read_csv(results_path, dtype={"model": str, "strategy": str, "metric": str})
    required = {
        "metric",
        "model",
        "strategy",
        "spearman_rho",
        "spearman_permutation_p",
        "spearman_holm_p",
        "kendall_tau_b",
        "top_k",
        "top_k_overlap",
        "top_k_enrichment",
        "top_k_oracle_regret",
        "proxy_best_regret",
        "bootstrap_spearman_ci_low",
        "bootstrap_spearman_ci_high",
        "n_permutations",
        "n_bootstrap_requested",
        "n_bootstrap_effective",
        "n_bootstrap_effective_paired",
        "holm_family_size",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Candidate-rank results are missing columns: " + ", ".join(sorted(missing))
        )
    keys = ["metric", "model", "strategy"]
    expected = pd.MultiIndex.from_product(
        [
            list(map(str, plan["metrics"])),
            list(map(str, plan["models"])),
            list(map(str, plan["strategies"])),
        ],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(frame[keys])
    if (
        frame.duplicated(keys).any()
        or len(expected.difference(observed))
        or len(observed.difference(expected))
    ):
        raise ValueError(
            "Candidate-rank results do not have exact metric/model/strategy coverage."
        )
    numeric = sorted(required - set(keys))
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Candidate-rank results contain non-finite values.")
    if (
        not frame["spearman_rho"].between(-1.0, 1.0).all()
        or not frame["kendall_tau_b"].between(-1.0, 1.0).all()
        or not frame["spearman_permutation_p"].between(0.0, 1.0).all()
        or not frame["spearman_holm_p"].between(0.0, 1.0).all()
        or not (frame["holm_family_size"] == 20).all()
        or not (frame["n_permutations"] == 10_000).all()
        or not (frame["n_bootstrap_requested"] == 10_000).all()
        or not (frame["n_bootstrap_effective"] > 0).all()
        or not (frame["n_bootstrap_effective_paired"] > 0).all()
        or not (frame["bootstrap_spearman_ci_low"] <= frame["bootstrap_spearman_ci_high"]).all()
    ):
        raise ValueError("Candidate-rank statistics violate their frozen contracts.")
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def _write_report(
    path: Path,
    campaign: Mapping[str, Any],
    contrasts: pd.DataFrame,
    equivalence: pd.DataFrame,
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
        (
            "Metadata-versus-encoder AUPRC equivalence uses paired reporting seeds, "
            f"TOST at ±{EQUIVALENCE_MARGIN_AUPRC:.2f}, and Holm decisions across "
            f"{len(equivalence)} models. Its 90% confidence intervals are unadjusted "
            "and labeled as such."
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
            (
                "- The main intervention heatmap follows frozen physical taxonomy order; "
                "it is not outcome ordered."
            ),
            (
                "- Strength panels use fixed within-target compositions and equal target "
                "weight within reporting seed."
            ),
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
    campaign, plan, taxonomy, results, integrity, input_paths = _verify_inputs(
        campaign_root,
        analysis_plan,
        taxonomy_path,
        integrity_manifest,
    )
    statuses, optional_paths = _optional_status(campaign_root, integrity)
    output_dir = _prepare_output_dir(output_dir, campaign_root)

    seed_summary, summary = _seed_first_summary(results)
    equal_unit_tables = _equal_unit_seed_summaries(results)
    contrasts = _contrast_rows(seed_summary, plan)
    equivalence = _metadata_encoder_equivalence(seed_summary, plan)
    target_seed, target_summary, strength_contrasts = _strength_outputs(
        results,
        plan["strength_order"],
    )
    strength_eligibility, strength_panel_seed, strength_panel_summary = _strength_panel_outputs(
        target_seed
    )
    system_seed, system_summary, system_contrasts = _system_group_outputs(results)
    intervention_seed, intervention_summary = _intervention_contrast_outputs(
        results,
        plan,
    )

    outputs: list[Path] = []
    tables = {
        "seed_first_summary.csv": seed_summary,
        "strategy_summary.csv": summary,
        "prespecified_strategy_contrasts.csv": contrasts,
        "prespecified_metadata_encoder_equivalence.csv": equivalence,
        **equal_unit_tables,
        "target_strength_seed_summary.csv": target_seed,
        "target_strength_summary.csv": target_summary,
        "within_target_strength_contrasts.csv": strength_contrasts,
        "strength_target_eligibility.csv": strength_eligibility,
        "strength_panel_equal_target_seed_summary.csv": strength_panel_seed,
        "strength_panel_summary.csv": strength_panel_summary,
        "system_group_seed_summary.csv": system_seed,
        "process_measurement_summary.csv": system_summary,
        "process_measurement_contrasts.csv": system_contrasts,
        "intervention_cap_baseline_seed_contrasts.csv": intervention_seed,
        "intervention_cap_baseline_summary.csv": intervention_summary,
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

    if "background_acceptance_diagnostics" in optional_paths:
        background_seed, background_summary = _background_acceptance_outputs(
            optional_paths["background_acceptance_diagnostics"],
            plan,
        )
        statuses["background_acceptance_analysis"] = {
            "status": "completed",
            "replicate_unit": "reporting seed",
            "event_pooling_for_inference": False,
        }
        for name, table in (
            ("background_acceptance_by_seed.csv", background_seed),
            ("background_acceptance_summary.csv", background_summary),
        ):
            path = output_dir / name
            table.to_csv(path, index=False)
            outputs.append(path)
    else:
        statuses["background_acceptance_analysis"] = {
            "status": "pending",
            "reason": "background_acceptance_diagnostics is required",
        }

    if {
        "candidate_audit_results",
        "candidate_audit_provenance",
    }.issubset(optional_paths):
        candidate_rank = _candidate_rank_outputs(
            optional_paths["candidate_audit_results"],
            optional_paths["candidate_audit_provenance"],
            plan,
        )
        statuses["candidate_rank_analysis"] = {
            "status": "completed",
            "holm_family": "20 model/strategy tests within each metric",
            "candidate_panel_outcome_blind": True,
        }
        path = output_dir / "candidate_rank_associations.csv"
        candidate_rank.to_csv(path, index=False)
        outputs.append(path)
    else:
        statuses["candidate_rank_analysis"] = {
            "status": "pending",
            "reason": ("candidate_audit_results and candidate_audit_provenance are both required"),
        }

    forest = output_dir / "confirmatory_contrast_forest.png"
    _plot_forest(contrasts, forest)
    outputs.append(forest)
    intervention_heatmap = output_dir / "prespecified_intervention_cap_minus_baseline_heatmaps.png"
    _plot_intervention_heatmaps(
        intervention_summary,
        taxonomy,
        plan,
        intervention_heatmap,
    )
    outputs.append(intervention_heatmap)
    outputs.extend(_plot_ordered_heatmaps(seed_summary, plan, output_dir))
    outputs.extend(_plot_strength_panels(strength_panel_summary, plan, output_dir))

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
    _write_report(report, campaign, contrasts, equivalence, statuses)
    outputs.append(report)

    provenance = output_dir / "analysis_provenance.json"
    _atomic_json(
        provenance,
        {
            "schema_version": 1,
            "campaign_id": campaign["campaign_id"],
            "inputs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in {**input_paths, **optional_paths}.items()
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
                "equivalence": {
                    "metric": "auprc",
                    "strategies": [
                        "cap_metadata_nearest",
                        "cap_encoder_nearest",
                    ],
                    "margin": EQUIVALENCE_MARGIN_AUPRC,
                    "test": "paired TOST",
                    "holm_family": "four detector models",
                    "confidence_interval": "90% unadjusted",
                },
                "intervention_heatmap_order": [
                    "system_group",
                    "semantic_family",
                    "intervention_target",
                    "strength",
                ],
                "strength_panels": (
                    "fixed complete and all-mid/strong target sets; "
                    "equal target weight within reporting seed"
                ),
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
