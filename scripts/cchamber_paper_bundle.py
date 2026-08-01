#!/usr/bin/env python3
"""Freeze and launch the post-evaluation Causal Chamber paper analysis.

This builder is intentionally outcome-blind: it validates identities, hashes, and
Cartesian coverage, but it never summarizes or compares metric values.  The emitted
``integrity_manifest.json`` is consumed by ``scripts/cchamber_paper_analysis.py``.

The builder requires the threshold-safe result chain, the label-free candidate/proxy
chain, and one seed-level background diagnostic row per selected checkpoint.  Candidate
rank associations are optional, but their provenance must be supplied with them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cchamber_paper_analysis  # noqa: E402

PAPER_ANALYSIS = REPO_ROOT / "scripts" / "cchamber_paper_analysis.py"
MANIFEST_NAME = "integrity_manifest.json"
LAUNCHER_NAME = "run_paper_analysis.sbatch"
EXPECTED_METRICS = ("auprc", "efficiency_operational")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _record(path: Path) -> dict[str, str]:
    """Build one absolute, hash-pinned integrity record."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": _sha256(path)}


def _require_hash(path: Path, expected: str, label: str) -> None:
    """Require one file to exist with the expected SHA-256."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _sha256(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch: observed {observed}, expected {expected}.")


def _same_path(left: Any, right: Path) -> bool:
    """Compare a serialized path with a resolved path."""
    if not left:
        return False
    return Path(str(left)).expanduser().resolve() == right.expanduser().resolve()


def _write_immutable(path: Path, content: str) -> None:
    """Create a deterministic file or accept a byte-identical existing file."""
    encoded = content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise FileExistsError(f"Refusing to replace a different immutable file: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is within parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_plan_and_taxonomy(
    campaign_path: Path,
    analysis_plan_path: Path,
    taxonomy_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Validate the frozen paper design without reading outcome values."""
    campaign = _read_json(campaign_path)
    plan = _read_json(analysis_plan_path)
    cchamber_paper_analysis._validate_plan(plan, campaign)
    taxonomy = cchamber_paper_analysis._validate_taxonomy(
        pd.read_csv(taxonomy_path),
        plan["interventions"],
    )
    return campaign, plan, taxonomy


def _validate_threshold_manifest(
    threshold_manifest_path: Path,
    expected_records: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Authenticate every frozen threshold artifact and its record identity."""
    manifest = _read_json(threshold_manifest_path)
    records = manifest.get("records")
    if (
        int(manifest.get("schema_version", -1)) != 1
        or manifest.get("test_or_intervention_data_loaded_before_freeze") is not False
        or int(manifest.get("expected_records", -1)) != expected_records
        or not isinstance(records, list)
        or len(records) != expected_records
    ):
        raise ValueError("Threshold manifest freeze/coverage contract is invalid.")
    frame = pd.DataFrame(records)
    required = {
        "manifest_index",
        "threshold_artifact",
        "threshold_artifact_sha256",
        "checkpoint_sha256",
        "threshold_bytes_sha256",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Threshold manifest misses columns: {sorted(required - set(frame.columns))}."
        )
    frame["manifest_index"] = pd.to_numeric(frame["manifest_index"], errors="raise").astype(int)
    if frame["manifest_index"].duplicated().any() or set(frame["manifest_index"]) != set(
        range(expected_records)
    ):
        raise ValueError("Threshold manifest indices are not exact.")
    for row in frame.to_dict("records"):
        artifact_path = Path(str(row["threshold_artifact"])).expanduser().resolve()
        _require_hash(
            artifact_path,
            str(row["threshold_artifact_sha256"]),
            f"threshold artifact {row['manifest_index']}",
        )
        artifact = _read_json(artifact_path)
        if (
            int(artifact.get("manifest_index", -1)) != int(row["manifest_index"])
            or str(artifact.get("checkpoint_sha256")) != str(row["checkpoint_sha256"])
            or str(artifact.get("threshold_float32", {}).get("bytes_sha256"))
            != str(row["threshold_bytes_sha256"])
        ):
            raise ValueError(
                f"Threshold artifact identity changed at index {row['manifest_index']}."
            )
    return manifest, frame.sort_values("manifest_index").reset_index(drop=True)


def _validate_threshold_chain(
    campaign: Mapping[str, Any],
    plan: Mapping[str, Any],
    taxonomy: pd.DataFrame,
    results_path: Path,
    threshold_manifest_path: Path,
    threshold_provenance_path: Path,
    background_diagnostics_path: Path,
) -> None:
    """Validate the complete threshold-safe result and diagnostic provenance chain."""
    models = list(map(str, campaign["models"]))
    strategies = list(map(str, campaign["strategies"]))
    seeds = list(map(int, campaign["reporting_seeds"]))
    expected_records = len(models) * len(strategies) * len(seeds)
    expected_rows = expected_records * len(campaign["interventions"]) * len(EXPECTED_METRICS)
    if expected_records != 240 or expected_rows != 27_840:
        raise ValueError(
            "Paper bundle requires the frozen 240-checkpoint/27,840-row campaign contract."
        )

    manifest, manifest_frame = _validate_threshold_manifest(
        threshold_manifest_path,
        expected_records,
    )
    manifest_sha = _sha256(threshold_manifest_path)
    raw_results = pd.read_csv(results_path)
    cchamber_paper_analysis._validate_results(raw_results, plan, taxonomy)
    provenance_columns = {
        "manifest_index",
        "checkpoint_sha256",
        "threshold_manifest_sha256",
        "threshold_artifact",
        "threshold_artifact_sha256",
        "threshold_bytes_sha256",
    }
    if not provenance_columns.issubset(raw_results.columns):
        raise ValueError(
            "Threshold-safe results miss provenance columns: "
            f"{sorted(provenance_columns - set(raw_results.columns))}."
        )
    if (
        len(raw_results) != expected_rows
        or set(raw_results["threshold_manifest_sha256"].astype(str)) != {manifest_sha}
        or set(pd.to_numeric(raw_results["manifest_index"], errors="raise").astype(int))
        != set(range(expected_records))
        or not (raw_results.groupby("manifest_index", sort=False).size() == 116).all()
    ):
        raise ValueError("Threshold-safe result coverage is not exact.")
    identity_columns = sorted(
        {
            "manifest_index",
            "checkpoint_sha256",
            "threshold_artifact",
            "threshold_artifact_sha256",
            "threshold_bytes_sha256",
        }
    )
    result_identity = (
        raw_results.loc[:, identity_columns]
        .drop_duplicates()
        .sort_values("manifest_index")
        .reset_index(drop=True)
    )
    manifest_identity = (
        manifest_frame.loc[:, identity_columns]
        .sort_values("manifest_index")
        .reset_index(drop=True)
    )
    if not result_identity.equals(manifest_identity):
        raise ValueError("Threshold-safe rows do not exactly join threshold records.")

    provenance = _read_json(threshold_provenance_path)
    results_path = results_path.resolve()
    threshold_manifest_path = threshold_manifest_path.resolve()
    background_diagnostics_path = background_diagnostics_path.resolve()
    if (
        not _same_path(provenance.get("results"), results_path)
        or provenance.get("results_sha256") != _sha256(results_path)
        or not _same_path(provenance.get("threshold_manifest"), threshold_manifest_path)
        or provenance.get("threshold_manifest_sha256") != manifest_sha
        or int(provenance.get("expected_records", -1)) != expected_records
        or int(provenance.get("expected_result_rows", -1)) != expected_rows
        or not _same_path(
            provenance.get("seed_level_summary"),
            background_diagnostics_path,
        )
        or provenance.get("seed_level_summary_sha256") != _sha256(background_diagnostics_path)
    ):
        raise ValueError("Threshold-safe provenance chain is inconsistent.")
    inventory_path = Path(str(provenance.get("inventory", ""))).expanduser().resolve()
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    _require_hash(
        inventory_path,
        str(provenance.get("inventory_sha256")),
        "operating-point inventory",
    )
    if manifest.get("inventory_sha256") != provenance.get("inventory_sha256") or not _same_path(
        manifest.get("inventory"), inventory_path
    ):
        raise ValueError("Threshold manifest and result provenance use different inventories.")

    diagnostics = pd.read_csv(background_diagnostics_path)
    required_diagnostic_columns = {
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
    if not required_diagnostic_columns.issubset(diagnostics.columns):
        raise ValueError(
            "Seed-level background diagnostics miss columns: "
            f"{sorted(required_diagnostic_columns - set(diagnostics.columns))}."
        )
    keys = ["model", "strategy", "seed"]
    diagnostics["seed"] = pd.to_numeric(diagnostics["seed"], errors="raise").astype(int)
    diagnostics["manifest_index"] = pd.to_numeric(
        diagnostics["manifest_index"], errors="raise"
    ).astype(int)
    expected = pd.MultiIndex.from_product(
        [models, strategies, seeds],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(diagnostics[keys])
    numeric = diagnostics.loc[
        :,
        [
            "test_normal_count",
            "triggered_count",
            "achieved_test_normal_acceptance",
            "target_fpr",
            "achieved_minus_target_fpr",
            "wilson_95_ci_low",
            "wilson_95_ci_high",
        ],
    ].apply(pd.to_numeric, errors="raise")
    if (
        len(diagnostics) != expected_records
        or diagnostics.duplicated(keys).any()
        or len(expected.difference(observed))
        or len(observed.difference(expected))
        or set(diagnostics["manifest_index"]) != set(range(expected_records))
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or not np.allclose(numeric["target_fpr"], 0.01)
        or not numeric["achieved_test_normal_acceptance"].between(0.0, 1.0).all()
        or not numeric["wilson_95_ci_low"].between(0.0, 1.0).all()
        or not numeric["wilson_95_ci_high"].between(0.0, 1.0).all()
    ):
        raise ValueError("Seed-level background diagnostic coverage/values are invalid.")


def _validate_candidate_chain(
    campaign_path: Path,
    campaign: Mapping[str, Any],
    candidate_metrics_path: Path,
    candidate_provenance_path: Path,
    pairing_sensitivity_path: Path,
) -> None:
    """Validate label-free candidate metrics and pairing sensitivity as one chain."""
    provenance = _read_json(candidate_provenance_path)
    campaign_path = campaign_path.resolve()
    candidate_metrics_path = candidate_metrics_path.resolve()
    pairing_sensitivity_path = pairing_sensitivity_path.resolve()
    if (
        not _same_path(provenance.get("campaign"), campaign_path)
        or provenance.get("campaign_sha256") != _sha256(campaign_path)
        or not _same_path(provenance.get("candidate_metrics"), candidate_metrics_path)
        or provenance.get("candidate_metrics_sha256") != _sha256(candidate_metrics_path)
        or not _same_path(
            provenance.get("pairing_proxy_sensitivity"),
            pairing_sensitivity_path,
        )
        or provenance.get("pairing_proxy_sensitivity_sha256") != _sha256(pairing_sensitivity_path)
    ):
        raise ValueError("Candidate/pairing provenance chain is inconsistent.")

    models = set(map(str, campaign["models"]))
    strategies = set(map(str, campaign["strategies"]))
    development_seeds = set(map(int, campaign["development_seeds"]))
    candidates = pd.read_csv(candidate_metrics_path, dtype={"candidate_id": str})
    required = {"model", "strategy", "seed", "candidate_id", "value"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"Candidate metrics miss columns: {sorted(required - set(candidates))}.")
    candidates["candidate_id"] = candidates["candidate_id"].astype(str).str.zfill(3)
    candidates["seed"] = pd.to_numeric(candidates["seed"], errors="raise").astype(int)
    candidates["value"] = pd.to_numeric(candidates["value"], errors="raise")
    if (
        set(candidates["model"].astype(str)) != models
        or set(candidates["strategy"].astype(str)) != strategies
        or set(candidates["seed"]) != development_seeds
        or candidates.duplicated(["model", "strategy", "seed", "candidate_id"]).any()
        or not np.isfinite(candidates["value"]).all()
    ):
        raise ValueError("Candidate metric identity/value coverage is invalid.")
    survivor_sets: dict[str, set[str]] = {}
    for model in sorted(models):
        branch = candidates[candidates["model"] == model]
        grouped = {
            (str(strategy), int(seed)): set(group["candidate_id"])
            for (strategy, seed), group in branch.groupby(["strategy", "seed"], sort=True)
        }
        expected_groups = {
            (strategy, seed) for strategy in strategies for seed in development_seeds
        }
        if set(grouped) != expected_groups:
            raise ValueError(f"Candidate groups are incomplete for {model}.")
        reference = next(iter(grouped.values()))
        if not reference or any(values != reference for values in grouped.values()):
            raise ValueError(f"Candidate survivor pool is not globally shared for {model}.")
        survivor_sets[model] = reference
    expected_rows = (
        sum(len(values) for values in survivor_sets.values())
        * len(strategies)
        * len(development_seeds)
    )
    if len(candidates) != expected_rows or int(provenance.get("n_rows", -1)) != expected_rows:
        raise ValueError("Candidate metric row count disagrees with provenance.")

    sensitivity = pd.read_csv(pairing_sensitivity_path, dtype={"candidate_id": str})
    required_sensitivity = {"model", "seed", "candidate_id", "variant", "value"}
    if not required_sensitivity.issubset(sensitivity.columns):
        raise ValueError(
            "Pairing sensitivity misses columns: "
            f"{sorted(required_sensitivity - set(sensitivity.columns))}."
        )
    sensitivity["candidate_id"] = sensitivity["candidate_id"].astype(str).str.zfill(3)
    sensitivity["seed"] = pd.to_numeric(sensitivity["seed"], errors="raise").astype(int)
    sensitivity["value"] = pd.to_numeric(sensitivity["value"], errors="raise")
    expected_variants = set(
        map(
            str,
            campaign.get("sensitivity_design", {}).get("metric_definitions", {}).keys(),
        )
    )
    if not expected_variants:
        raise ValueError("Campaign has no frozen pairing-sensitivity variants.")
    if (
        set(sensitivity["model"].astype(str)) != models
        or set(sensitivity["seed"]) != development_seeds
        or set(sensitivity["variant"].astype(str)) != expected_variants
        or sensitivity.duplicated(["model", "seed", "candidate_id", "variant"]).any()
        or not np.isfinite(sensitivity["value"]).all()
    ):
        raise ValueError("Pairing-sensitivity identity/value coverage is invalid.")
    for model in sorted(models):
        branch = sensitivity[sensitivity["model"] == model]
        grouped = {
            (int(seed), str(variant)): set(group["candidate_id"])
            for (seed, variant), group in branch.groupby(["seed", "variant"], sort=True)
        }
        expected_groups = {
            (seed, variant) for seed in development_seeds for variant in expected_variants
        }
        if set(grouped) != expected_groups or any(
            values != survivor_sets[model] for values in grouped.values()
        ):
            raise ValueError(f"Pairing-sensitivity candidate coverage changed for {model}.")


def _validate_rank_chain(
    campaign: Mapping[str, Any],
    candidate_metrics_path: Path,
    associations_path: Path,
    provenance_path: Path,
) -> None:
    """Validate optional candidate-rank associations and their upstream hashes."""
    provenance = _read_json(provenance_path)
    associations_path = associations_path.resolve()
    candidate_metrics_path = candidate_metrics_path.resolve()
    output_hash = provenance.get("outputs", {}).get(associations_path.name)
    if (
        output_hash != _sha256(associations_path)
        or not _same_path(provenance.get("candidate_metrics"), candidate_metrics_path)
        or provenance.get("candidate_metrics_sha256") != _sha256(candidate_metrics_path)
    ):
        raise ValueError("Candidate-rank association provenance is inconsistent.")
    for path_key, hash_key in (
        ("outcomes", "outcomes_sha256"),
        ("outcome_provenance", "outcome_provenance_sha256"),
    ):
        source = Path(str(provenance.get(path_key, ""))).expanduser().resolve()
        _require_hash(source, str(provenance.get(hash_key)), f"rank {path_key}")
    outcome_provenance_path = Path(str(provenance["outcome_provenance"])).resolve()
    outcome_provenance = _read_json(outcome_provenance_path)
    if not _same_path(
        outcome_provenance.get("combined"), Path(str(provenance["outcomes"]))
    ) or outcome_provenance.get("combined_sha256") != provenance.get("outcomes_sha256"):
        raise ValueError("Candidate-rank sealed-outcome chain is inconsistent.")

    frame = pd.read_csv(associations_path)
    required = {
        "metric",
        "model",
        "strategy",
        "spearman_rho",
        "spearman_permutation_p",
        "spearman_holm_p",
        "holm_family_size",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Rank associations miss columns: {sorted(required - set(frame))}.")
    expected = pd.MultiIndex.from_product(
        [
            EXPECTED_METRICS,
            list(map(str, campaign["models"])),
            list(map(str, campaign["strategies"])),
        ],
        names=["metric", "model", "strategy"],
    )
    observed = pd.MultiIndex.from_frame(frame[list(expected.names)])
    numeric = frame[
        [
            "spearman_rho",
            "spearman_permutation_p",
            "spearman_holm_p",
            "holm_family_size",
        ]
    ].apply(pd.to_numeric, errors="raise")
    if (
        len(frame) != len(expected)
        or frame.duplicated(list(expected.names)).any()
        or len(expected.difference(observed))
        or len(observed.difference(expected))
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or not numeric["spearman_rho"].between(-1.0, 1.0).all()
        or not numeric["spearman_permutation_p"].between(0.0, 1.0).all()
        or not numeric["spearman_holm_p"].between(0.0, 1.0).all()
        or set(numeric["holm_family_size"].astype(int))
        != {len(campaign["models"]) * len(campaign["strategies"])}
    ):
        raise ValueError("Candidate-rank association coverage/values are invalid.")


def _git_commit() -> str:
    """Return the current commit after requiring a clean analysis checkout."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required to freeze the analysis-code revision.")
    commit = subprocess.check_output(  # nosec B603
        [git, "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    dirty = subprocess.check_output(  # nosec B603
        [git, "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Paper analysis must be bundled from a clean repository checkout.")
    return commit


def _launcher_text(
    *,
    account: str,
    bundle_dir: Path,
    analysis_output: Path,
    analysis_plan: Path,
    taxonomy: Path,
    campaign_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    uv_path: Path,
    code_commit: str,
    paper_analysis_sha256: str,
) -> str:
    """Render a CPU-only Clariden debug launcher."""
    quote = shlex.quote
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=cchamber-paper
#SBATCH --account={account}
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output={bundle_dir / "slurm-%j.out"}
#SBATCH --error={bundle_dir / "slurm-%j.err"}

set -euo pipefail

REPOSITORY={quote(str(REPO_ROOT))}
EXPECTED_COMMIT={quote(code_commit)}
ANALYSIS_SCRIPT={quote(str(PAPER_ANALYSIS))}
EXPECTED_ANALYSIS_SHA256={quote(paper_analysis_sha256)}
EXPECTED_MANIFEST_SHA256={quote(manifest_sha256)}

cd "$REPOSITORY"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$ANALYSIS_SCRIPT" | awk '{{print $1}}')" = "$EXPECTED_ANALYSIS_SHA256"
test "$(sha256sum {quote(str(manifest_path))} | awk '{{print $1}}')" = "$EXPECTED_MANIFEST_SHA256"
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${{SLURM_TMPDIR:-/tmp}}/cchamber-matplotlib-${{SLURM_JOB_ID}}"
mkdir -p "$MPLCONFIGDIR"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \\
  {quote(str(uv_path))} run --frozen --no-sync python "$ANALYSIS_SCRIPT" \\
  --campaign-root {quote(str(campaign_root))} \\
  --analysis-plan {quote(str(analysis_plan))} \\
  --taxonomy {quote(str(taxonomy))} \\
  --integrity-manifest {quote(str(manifest_path))} \\
  --output-dir {quote(str(analysis_output))}
"""


def build_bundle(
    *,
    campaign_root: Path,
    analysis_plan: Path,
    taxonomy: Path,
    threshold_results: Path,
    threshold_manifest: Path,
    threshold_provenance: Path,
    candidate_metrics: Path,
    candidate_metrics_provenance: Path,
    pairing_sensitivity: Path,
    background_diagnostics: Path,
    bundle_dir: Path,
    analysis_output: Path,
    rank_associations: Path | None = None,
    rank_provenance: Path | None = None,
    account: str = "a0166",
) -> tuple[Path, Path]:
    """Validate all frozen inputs and emit the integrity manifest and Slurm launcher."""
    campaign_root = campaign_root.expanduser().resolve()
    campaign_path = campaign_root / "campaign.json"
    analysis_plan = analysis_plan.expanduser().resolve()
    taxonomy = taxonomy.expanduser().resolve()
    threshold_results = threshold_results.expanduser().resolve()
    threshold_manifest = threshold_manifest.expanduser().resolve()
    threshold_provenance = threshold_provenance.expanduser().resolve()
    candidate_metrics = candidate_metrics.expanduser().resolve()
    candidate_metrics_provenance = candidate_metrics_provenance.expanduser().resolve()
    pairing_sensitivity = pairing_sensitivity.expanduser().resolve()
    background_diagnostics = background_diagnostics.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    analysis_output = analysis_output.expanduser().resolve()
    if bool(rank_associations) != bool(rank_provenance):
        raise ValueError("--rank-associations and --rank-provenance must be supplied together.")
    rank_associations = None if rank_associations is None else rank_associations.resolve()
    rank_provenance = None if rank_provenance is None else rank_provenance.resolve()
    for path in (
        campaign_path,
        analysis_plan,
        taxonomy,
        threshold_results,
        threshold_manifest,
        threshold_provenance,
        candidate_metrics,
        candidate_metrics_provenance,
        pairing_sensitivity,
        background_diagnostics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _is_relative_to(bundle_dir, campaign_root) or _is_relative_to(
        analysis_output, campaign_root
    ):
        raise ValueError("Bundle and analysis outputs must be outside the immutable campaign.")
    if analysis_output.exists() and (
        not analysis_output.is_dir() or any(analysis_output.iterdir())
    ):
        raise FileExistsError(f"Analysis output is not empty: {analysis_output}")

    campaign, plan, taxonomy_frame = _validate_plan_and_taxonomy(
        campaign_path,
        analysis_plan,
        taxonomy,
    )
    _validate_threshold_chain(
        campaign,
        plan,
        taxonomy_frame,
        threshold_results,
        threshold_manifest,
        threshold_provenance,
        background_diagnostics,
    )
    _validate_candidate_chain(
        campaign_path,
        campaign,
        candidate_metrics,
        candidate_metrics_provenance,
        pairing_sensitivity,
    )
    if rank_associations is not None and rank_provenance is not None:
        _validate_rank_chain(
            campaign,
            candidate_metrics,
            rank_associations,
            rank_provenance,
        )

    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required to generate the paper-analysis launcher.")
    code_commit = _git_commit()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_dir / MANIFEST_NAME
    launcher_path = bundle_dir / LAUNCHER_NAME
    files = {
        "campaign": _record(campaign_path),
        "results": _record(threshold_results),
        "analysis_plan": _record(analysis_plan),
        "taxonomy": _record(taxonomy),
        "threshold_manifest": _record(threshold_manifest),
        "threshold_safe_provenance": _record(threshold_provenance),
    }
    optional_artifacts = {
        "candidate_metrics": _record(candidate_metrics),
        "candidate_metrics_provenance": _record(candidate_metrics_provenance),
        "pairing_proxy_sensitivity": _record(pairing_sensitivity),
        "background_acceptance_diagnostics": _record(background_diagnostics),
    }
    if rank_associations is not None and rank_provenance is not None:
        optional_artifacts.update(
            {
                "candidate_audit_results": _record(rank_associations),
                "candidate_audit_provenance": _record(rank_provenance),
            }
        )
    manifest = {
        "schema_version": 1,
        "manifest_type": "cchamber_paper_analysis_integrity",
        "campaign_id": campaign["campaign_id"],
        "files": files,
        "optional_artifacts": optional_artifacts,
        "bundle_contract": {
            "required_result_rows": 27_840,
            "required_selected_checkpoints": 240,
            "metrics": list(EXPECTED_METRICS),
            "rank_analysis": "available" if rank_associations else "not_supplied",
            "outcome_values_summarized_or_compared_by_builder": False,
        },
        "execution": {
            "repository": str(REPO_ROOT),
            "git_commit": code_commit,
            "paper_analysis": _record(PAPER_ANALYSIS),
            "bundle_builder": _record(Path(__file__)),
            "slurm_account": account,
            "slurm_partition": "debug",
            "accelerator": "cpu",
            "analysis_output": str(analysis_output),
        },
    }
    manifest_text = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    _write_immutable(manifest_path, manifest_text)
    launcher = _launcher_text(
        account=account,
        bundle_dir=bundle_dir,
        analysis_output=analysis_output,
        analysis_plan=analysis_plan,
        taxonomy=taxonomy,
        campaign_root=campaign_root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        uv_path=Path(uv).resolve(),
        code_commit=code_commit,
        paper_analysis_sha256=_sha256(PAPER_ANALYSIS),
    )
    _write_immutable(launcher_path, launcher)
    launcher_path.chmod(0o755)
    bash = shutil.which("bash")
    if bash is None:
        raise FileNotFoundError("bash is required to validate the Slurm launcher.")
    subprocess.run([bash, "-n", str(launcher_path)], check=True)  # nosec B603
    return manifest_path, launcher_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the bundle-builder CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--campaign-root", type=Path, required=True)
    build.add_argument("--analysis-plan", type=Path, required=True)
    build.add_argument("--taxonomy", type=Path, required=True)
    build.add_argument("--threshold-results", type=Path, required=True)
    build.add_argument("--threshold-manifest", type=Path, required=True)
    build.add_argument("--threshold-provenance", type=Path, required=True)
    build.add_argument("--candidate-metrics", type=Path, required=True)
    build.add_argument("--candidate-metrics-provenance", type=Path, required=True)
    build.add_argument("--pairing-sensitivity", type=Path, required=True)
    build.add_argument("--background-diagnostics", type=Path, required=True)
    build.add_argument("--rank-associations", type=Path)
    build.add_argument("--rank-provenance", type=Path)
    build.add_argument("--bundle-dir", type=Path, required=True)
    build.add_argument("--analysis-output", type=Path, required=True)
    build.add_argument("--account", default="a0166")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the requested immutable paper-analysis handoff."""
    args = parse_args(argv)
    manifest, launcher = build_bundle(
        campaign_root=args.campaign_root,
        analysis_plan=args.analysis_plan,
        taxonomy=args.taxonomy,
        threshold_results=args.threshold_results,
        threshold_manifest=args.threshold_manifest,
        threshold_provenance=args.threshold_provenance,
        candidate_metrics=args.candidate_metrics,
        candidate_metrics_provenance=args.candidate_metrics_provenance,
        pairing_sensitivity=args.pairing_sensitivity,
        background_diagnostics=args.background_diagnostics,
        bundle_dir=args.bundle_dir,
        analysis_output=args.analysis_output,
        rank_associations=args.rank_associations,
        rank_provenance=args.rank_provenance,
        account=args.account,
    )
    print(manifest)
    print(launcher)
    print(f"Submit with: sbatch {launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
