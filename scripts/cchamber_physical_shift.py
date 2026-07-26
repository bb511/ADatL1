#!/usr/bin/env python3
"""Characterize frozen Causal Chamber physical shifts after label-free selection.

The tool is deliberately separate from training, selection, and performance analysis. It will not
read intervention CSVs until the campaign, estimand, target catalog, and label-free selection
artifacts have passed their identity and SHA-256 checks.  Outputs must be written outside the
immutable campaign root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.components.causal_chamber import (  # noqa: E402
    META_COLUMNS,
    READOUT_FEATURES,
    CausalChamberNormalizer,
    parse_intervention_name,
)
from src.utils.pairing.utils import one_to_one_nearest_pairs  # noqa: E402

TRAIN_FRACTION = 0.6
VALID_FRACTION = 0.2
REFERENCE_FRACTION = 0.5
SIGNAL_VALID_FRACTION = 0.6
ROBUST_QUANTILES = (0.05, 0.95)
CLIP_VALUE = 10.0
CI_QUANTILES = (0.025, 0.975)
PHYSICAL_CLASS_ORDER = {
    "process_or_actuator": 0,
    "measurement_chain": 1,
}
STRENGTH_ORDER = {"weak": 0, "mid": 1, "strong": 2}
ENERGY_FORMULA = "2*mean(||X-Y||)-mean(||X-X'||)-mean(||Y-Y'||), " "including zero diagonal terms"
LOCATION_ESTIMAND = (
    "Hedges-corrected standardized mean difference using the pooled "
    "reference/intervention standard deviation"
)
SCALE_ESTIMAND = "log of intervention standard deviation divided by reference standard deviation"
ZERO_VARIANCE_RULE = (
    "Report null only when both groups are identical constants; otherwise "
    "signed infinity is invalid and must raise."
)
ZERO_VARIANCE_AMENDMENT = (
    "When either group has zero variance, report the raw mean difference and "
    "mark Hedges g and the log-SD ratio undefined; never replace infinity with "
    "a finite value."
)
BOOTSTRAP_CI_AMENDMENT = (
    "When a finite point estimate has fewer than 80% finite deterministic "
    "bootstrap replicates because a resample is degenerate, retain the point "
    "estimate, report the finite replicate count, and mark its interval "
    "undefined; never fabricate or clip interval bounds."
)
DESCENDANT_ESTIMAND = (
    "Within intervention, compare absolute location and scale effects on "
    "graph-expected descendants against non-descendant readouts."
)
BOOTSTRAP_PATTERN = re.compile(
    r"(?P<repetitions>[1-9][0-9]*) deterministic stratified bootstrap resamples "
    r"within reference and intervention, seed (?P<seed>[0-9]+) plus intervention index"
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    """Return the campaign-compatible canonical JSON representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Any) -> None:
    """Write strict JSON atomically."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is inside ``parent``."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_path(output_dir: Path, campaign_root: Path) -> Path:
    """Validate an empty output destination outside the campaign without creating it."""
    output_dir = output_dir.expanduser().resolve()
    if _is_relative_to(output_dir, campaign_root):
        raise ValueError("Physical-shift output must be outside the immutable campaign root.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    return output_dir


def _require_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    """Require a fixed set of object keys."""
    missing = keys - set(value)
    if missing:
        raise ValueError(f"{name} is missing: {', '.join(sorted(missing))}.")


def _parse_bootstrap(plan: Mapping[str, Any]) -> tuple[int, int]:
    """Parse the frozen deterministic-bootstrap contract."""
    uncertainty = str(plan["joint_shift"].get("uncertainty", ""))
    match = BOOTSTRAP_PATTERN.fullmatch(uncertainty)
    if match is None:
        raise ValueError("joint_shift.uncertainty does not match the frozen bootstrap contract.")
    return int(match.group("repetitions")), int(match.group("seed"))


def _validate_frozen_design(
    campaign_root: Path,
    shift_plan_path: Path,
    target_catalog_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]:
    """Validate campaign, estimand, and target-catalog identities and contracts."""
    campaign_path = campaign_root / "campaign.json"
    if not campaign_path.is_file():
        raise FileNotFoundError(campaign_path)
    shift_plan_path = shift_plan_path.expanduser().resolve()
    target_catalog_path = target_catalog_path.expanduser().resolve()
    for path in (shift_plan_path, target_catalog_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    campaign = _read_json(campaign_path)
    plan = _read_json(shift_plan_path)
    catalog = _read_json(target_catalog_path)
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("Physical-shift plan schema_version must be 1.")
    if int(catalog.get("schema_version", -1)) != 1:
        raise ValueError("Physical target catalog schema_version must be 1.")
    if plan.get("frozen_before_intervention_outcomes") is not True:
        raise ValueError("Physical-shift plan was not frozen before intervention outcomes.")
    if catalog.get("frozen_before_intervention_outcomes") is not True:
        raise ValueError("Physical target catalog was not frozen before intervention outcomes.")

    campaign_hash = _sha256(campaign_path)
    campaign_id = str(campaign.get("campaign_id"))
    if (
        str(plan.get("campaign_id")) != campaign_id
        or str(catalog.get("campaign_id")) != campaign_id
    ):
        raise ValueError("Campaign identity differs across the campaign, plan, and catalog.")
    if str(plan.get("campaign_manifest_sha256")) != campaign_hash:
        raise ValueError("Physical-shift plan campaign hash does not match campaign.json.")
    if str(catalog.get("campaign_manifest_sha256")) != campaign_hash:
        raise ValueError("Physical target catalog campaign hash does not match campaign.json.")
    if str(plan.get("physical_catalog_sha256")) != _sha256(target_catalog_path):
        raise ValueError("Physical target catalog hash does not match the frozen shift plan.")

    data_contract = plan.get("data_contract")
    if not isinstance(data_contract, dict):
        raise ValueError("Physical-shift plan must contain a data_contract object.")
    normalization = data_contract.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Physical-shift normalization contract is missing.")
    if int(data_contract.get("data_seed", -1)) != int(campaign.get("data_seed", -2)):
        raise ValueError("Physical-shift data seed differs from campaign.json.")
    if data_contract.get("reference_experiment") != "uniform_reference":
        raise ValueError("Physical-shift reference experiment must be uniform_reference.")
    if data_contract.get("reference_split") != "test.normal":
        raise ValueError("Physical-shift reference split must be test.normal.")
    if data_contract.get("signal_split") != "test":
        raise ValueError("Physical-shift signal split must be test.")
    if data_contract.get("feature_set") != "readouts":
        raise ValueError("Physical-shift feature_set must be readouts.")
    if campaign.get("feature_set") != "readouts" or int(campaign.get("n_features", -1)) != 11:
        raise ValueError("Campaign did not freeze the eleven-readout feature contract.")
    if normalization != {
        "fit_split": "uniform_reference.train",
        "fit_count": int(normalization.get("fit_count", -1)),
        "method": "subtract per-feature median and divide by q95-q05",
        "clip": [-10.0, 10.0],
    }:
        raise ValueError("Physical-shift robust-normalization method or clipping differs.")
    if int(data_contract.get("signal_count", -1)) != len(campaign.get("interventions", [])):
        raise ValueError("Physical-shift intervention count differs from campaign.json.")

    joint = plan.get("joint_shift")
    if not isinstance(joint, dict):
        raise ValueError("Physical-shift plan must contain joint_shift.")
    if joint.get("name") != "biased_multivariate_energy_distance":
        raise ValueError("Unexpected physical joint-shift estimand.")
    if joint.get("formula") != ENERGY_FORMULA:
        raise ValueError("Physical joint-shift formula differs from the frozen estimand.")
    if joint.get("finite_sample_rule") != "Clamp negative floating-point roundoff to zero.":
        raise ValueError("Unexpected physical joint-shift finite-sample rule.")
    repetitions, bootstrap_seed = _parse_bootstrap(plan)

    effects = plan.get("readout_effects")
    if not isinstance(effects, dict):
        raise ValueError("Physical-shift plan must contain readout_effects.")
    _require_keys(
        effects,
        {"location", "scale", "zero_variance_rule", "descendant_summary", "use"},
        "readout_effects",
    )
    if effects["location"] != LOCATION_ESTIMAND:
        raise ValueError("Readout location estimand differs from the frozen contract.")
    if effects["scale"] != SCALE_ESTIMAND:
        raise ValueError("Readout scale estimand differs from the frozen contract.")
    if effects["zero_variance_rule"] != ZERO_VARIANCE_RULE:
        raise ValueError("Readout zero-variance rule differs from the frozen contract.")
    if effects["descendant_summary"] != DESCENDANT_ESTIMAND:
        raise ValueError("Expected-descendant estimand differs from the frozen contract.")
    performance = plan.get("performance_association")
    if not isinstance(performance, dict):
        raise ValueError("Frozen performance-association contract is missing.")
    _require_keys(
        performance,
        {"unit", "methods", "prohibited"},
        "performance_association",
    )
    if not performance["methods"] or not performance["prohibited"]:
        raise ValueError("Frozen performance-association methods/prohibitions are empty.")

    if catalog.get("model_readouts") != list(READOUT_FEATURES):
        raise ValueError("Target catalog readouts differ from the repository readout contract.")
    if str(catalog.get("protocol_source", {}).get("archive_md5")) != str(
        campaign.get("dataset_archive_md5")
    ):
        raise ValueError("Target catalog protocol archive differs from campaign.json.")
    targets = catalog.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Target catalog must contain target records.")
    target_names: set[str] = set()
    for record in targets:
        if not isinstance(record, dict):
            raise ValueError("Every target catalog record must be an object.")
        _require_keys(
            record,
            {
                "target",
                "physical_class",
                "semantic_family",
                "knob",
                "expected_readout_descendants",
            },
            "target catalog record",
        )
        target = str(record["target"])
        if target in target_names:
            raise ValueError(f"Duplicate physical target: {target}")
        target_names.add(target)
        if str(record["physical_class"]) not in PHYSICAL_CLASS_ORDER:
            raise ValueError(f"Unknown physical class for target {target}.")
        descendants = list(map(str, record["expected_readout_descendants"]))
        if len(descendants) != len(set(descendants)) or not set(descendants).issubset(
            READOUT_FEATURES
        ):
            raise ValueError(f"Invalid expected descendants for target {target}.")

    intervention_targets = {
        str(parse_intervention_name(str(name))["target"])
        for name in campaign.get("interventions", [])
    }
    if intervention_targets != target_names:
        raise ValueError("Target catalog coverage differs from campaign interventions.")
    return campaign, plan, catalog, repetitions, bootstrap_seed


def _validate_selection_frozen(
    campaign_root: Path,
    campaign: Mapping[str, Any],
    selection_provenance_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Require complete, hash-pinned, label-free selection before data loading."""
    expected_path = (campaign_root / "selection" / "selection_provenance.json").resolve()
    selection_provenance_path = selection_provenance_path.expanduser().resolve()
    if selection_provenance_path != expected_path:
        raise ValueError(
            "Selection provenance must be <campaign-root>/selection/selection_provenance.json."
        )
    if not selection_provenance_path.is_file():
        raise FileNotFoundError(selection_provenance_path)
    observed_hash = _sha256(selection_provenance_path)
    if observed_hash != str(expected_sha256):
        raise ValueError(
            f"Selection provenance SHA-256 mismatch: observed {observed_hash}, "
            f"expected {expected_sha256}."
        )
    provenance = _read_json(selection_provenance_path)
    if provenance.get("intervention_labels_used") is not False:
        raise ValueError("Selection provenance does not certify label-free selection.")
    if list(provenance.get("development_seeds", [])) != list(
        campaign.get("development_seeds", [])
    ):
        raise ValueError("Selection development seeds differ from campaign.json.")

    candidate_metrics = (campaign_root / "selection" / "candidate_metrics.csv").resolve()
    selected_path = (campaign_root / "selection" / "selected_trials.csv").resolve()
    retrain_path = (campaign_root / "selection" / "retrain_manifest.json").resolve()
    if Path(str(provenance.get("candidate_metrics", ""))).resolve() != candidate_metrics:
        raise ValueError("Selection provenance candidate_metrics path is not campaign-local.")
    artifacts = {
        "candidate_metrics": (
            candidate_metrics,
            str(provenance.get("candidate_metrics_sha256", "")),
        ),
        "selected_trials": (
            selected_path,
            str(provenance.get("selected_trials_sha256", "")),
        ),
        "retrain_manifest": (
            retrain_path,
            str(provenance.get("retrain_manifest_sha256", "")),
        ),
    }
    for name, (path, expected_hash) in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != expected_hash:
            raise ValueError(f"Frozen selection artifact hash mismatch: {name}.")

    selected = pd.read_csv(selected_path, dtype={"candidate_id": str})
    required = {"model", "strategy", "candidate_id", "pool_sha256", "git_commit"}
    if missing := required - set(selected.columns):
        raise ValueError(f"selected_trials.csv is missing: {', '.join(sorted(missing))}.")
    expected_pairs = {
        (str(model), str(strategy))
        for model in campaign.get("models", [])
        for strategy in campaign.get("strategies", [])
    }
    observed_pairs = set(zip(selected["model"].astype(str), selected["strategy"].astype(str)))
    if observed_pairs != expected_pairs or selected.duplicated(["model", "strategy"]).any():
        raise ValueError("Frozen selected trials do not have exact model/strategy coverage.")
    for row in selected.itertuples(index=False):
        if str(row.git_commit) != str(campaign.get("git_commit")):
            raise ValueError("Selected trial git commit differs from campaign.json.")
        if str(row.pool_sha256) != str(campaign["pool_sha256"][str(row.model)]):
            raise ValueError("Selected trial pool hash differs from campaign.json.")

    retrain = json.loads(retrain_path.read_text(encoding="utf-8"))
    if not isinstance(retrain, list):
        raise ValueError("retrain_manifest.json must contain a list.")
    expected_retrains = {
        (model, strategy, int(seed))
        for model, strategy in expected_pairs
        for seed in campaign.get("reporting_seeds", [])
    }
    observed_retrains = {
        (str(row.get("model")), str(row.get("strategy")), int(row.get("seed", -1)))
        for row in retrain
    }
    if observed_retrains != expected_retrains or len(retrain) != len(expected_retrains):
        raise ValueError("Frozen retrain manifest does not have exact reporting coverage.")
    selected_candidates = {
        (str(row.model), str(row.strategy)): str(row.candidate_id)
        for row in selected.itertuples(index=False)
    }
    if any(
        str(row.get("candidate_id"))
        != selected_candidates[(str(row.get("model")), str(row.get("strategy")))]
        for row in retrain
    ):
        raise ValueError("Frozen retrain candidates differ from selected_trials.csv.")
    if int(provenance.get("n_selected", -1)) != len(expected_pairs):
        raise ValueError("Selection provenance n_selected is incomplete.")
    if int(provenance.get("n_retrains", -1)) != len(expected_retrains):
        raise ValueError("Selection provenance n_retrains is incomplete.")
    return {
        "selection_provenance": {
            "path": str(selection_provenance_path),
            "sha256": observed_hash,
        },
        **{
            name: {"path": str(path), "sha256": expected_hash}
            for name, (path, expected_hash) in artifacts.items()
        },
        "intervention_labels_used": False,
        "n_selected": len(expected_pairs),
        "n_retrains": len(expected_retrains),
    }


def _dataset_paths(
    campaign: Mapping[str, Any],
) -> tuple[Path, dict[str, Path], list[dict[str, Any]]]:
    """Verify the frozen dataset tree and return exact experiment paths."""
    records = campaign.get("dataset_files")
    if not isinstance(records, list):
        raise ValueError("campaign.json does not contain frozen dataset_files.")
    expected_names = {
        "uniform_reference",
        *map(str, campaign.get("interventions", [])),
    }
    observed: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every campaign dataset file record must be an object.")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix != ".csv" or path.stem in paths:
            raise ValueError(f"Invalid or duplicate campaign dataset file: {path}")
        observed_record = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if observed_record != {
            "path": str(Path(str(record["path"])).expanduser().resolve()),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }:
            raise ValueError(f"Campaign dataset fingerprint changed: {path}")
        paths[path.stem] = path
        observed.append(observed_record)
    if set(paths) != expected_names:
        raise ValueError("Campaign dataset file coverage is not exact.")
    if len({path.parent for path in paths.values()}) != 1:
        raise ValueError("Campaign dataset CSVs do not share one frozen directory.")
    tree_hash = hashlib.sha256(_canonical_json(observed).encode("utf-8")).hexdigest()
    if tree_hash != str(campaign.get("dataset_tree_sha256")):
        raise ValueError("Campaign dataset tree hash changed.")
    return next(iter(paths.values())).parent, paths, observed


def _load_readouts(path: Path) -> np.ndarray:
    """Load the exact eleven finite readouts as repository-compatible float32."""
    frame = pd.read_csv(path, usecols=list(READOUT_FEATURES))
    frame = frame.loc[:, list(READOUT_FEATURES)]
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite or non-numeric readouts found in {path}.")
    return values


def _load_reference_table(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load readouts and the exact default metadata-pairing feature matrix."""
    frame = pd.read_csv(path)
    missing = [name for name in READOUT_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"Reference table is missing readouts: {missing}")
    readouts = (
        frame.loc[:, list(READOUT_FEATURES)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    all_numeric = [
        column
        for column in frame.columns
        if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    pairing_names = [name for name in all_numeric if name not in READOUT_FEATURES]
    if not pairing_names:
        raise ValueError("Frozen metadata-nearest reference split has no pairing features.")
    pairing = (
        frame.loc[:, pairing_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    if not np.isfinite(readouts).all() or not np.isfinite(pairing).all():
        raise ValueError("Reference readouts or pairing features contain non-finite values.")
    return readouts, pairing, pairing_names


def _reference_split_indices(
    n_total: int,
    seed: int,
    pairing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct train/valid/test and metadata-matched test.normal ordering."""
    if pairing.ndim != 2 or pairing.shape[0] != n_total or pairing.shape[1] == 0:
        raise ValueError("Reference pairing matrix has invalid shape.")
    n_train = int(round(TRAIN_FRACTION * n_total))
    n_valid = int(round(VALID_FRACTION * n_total))
    n_test = n_total - n_train - n_valid
    if min(n_train, n_valid, n_test) <= 1:
        raise ValueError("Reference data are too small for the frozen split.")
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(n_total, generator=generator)
    train = permutation[:n_train]
    valid = permutation[n_train : n_train + n_valid]
    test = permutation[n_train + n_valid :]
    n_normal = min(
        max(1, int(round(REFERENCE_FRACTION * test.numel()))),
        test.numel() // 2,
    )
    pair_generator = torch.Generator().manual_seed(int(seed) + test.numel())
    pair_order = torch.randperm(test.numel(), generator=pair_generator)
    normal_pool = test[pair_order[:n_normal]]
    reference_pool = test[pair_order[n_normal : 2 * n_normal]]
    pairing_tensor = torch.as_tensor(pairing, dtype=torch.float32)
    x1 = pairing_tensor[normal_pool]
    x2 = pairing_tensor[reference_pool]
    combined = torch.cat([x1, x2], dim=0)
    center = combined.mean(dim=0)
    scale = combined.std(dim=0).clamp_min(1.0e-6)
    pairs = one_to_one_nearest_pairs(
        (x1 - center) / scale,
        (x2 - center) / scale,
        k=None,
    )
    if pairs.idx_1.numel() != n_normal:
        raise ValueError("Metadata-nearest reference pairing did not achieve full coverage.")
    normal = normal_pool[pairs.idx_1]
    return (
        train.numpy(),
        valid.numpy(),
        test.numpy(),
        normal.numpy(),
    )


def _signal_test_indices(n_total: int, seed: int) -> np.ndarray:
    """Reconstruct the exact intervention test split."""
    if n_total <= 1:
        raise ValueError("Intervention data require at least two rows.")
    n_valid = int(round(SIGNAL_VALID_FRACTION * n_total))
    n_valid = min(max(1, n_valid), n_total - 1)
    generator = torch.Generator().manual_seed(int(seed) + n_total)
    permutation = torch.randperm(n_total, generator=generator)
    return permutation[n_valid:].numpy()


def _hash_indices(indices: np.ndarray) -> str:
    """Fingerprint an integer index vector."""
    contiguous = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _distance_matrices(
    reference: np.ndarray,
    intervention: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Euclidean distance matrices in float64."""
    reference64 = np.asarray(reference, dtype=np.float64)
    intervention64 = np.asarray(intervention, dtype=np.float64)
    return (
        cdist(reference64, reference64, metric="euclidean"),
        cdist(intervention64, intervention64, metric="euclidean"),
        cdist(reference64, intervention64, metric="euclidean"),
    )


def _biased_energy_from_distances(
    reference_distances: np.ndarray,
    intervention_distances: np.ndarray,
    cross_distances: np.ndarray,
) -> float:
    """Compute biased multivariate energy distance including diagonal zeros."""
    value = (
        2.0 * float(cross_distances.mean())
        - float(reference_distances.mean())
        - float(intervention_distances.mean())
    )
    return max(0.0, value)


def _bootstrap_weights(
    rng: np.random.Generator,
    repetitions: int,
    n_rows: int,
) -> np.ndarray:
    """Generate exact multinomial bootstrap weights."""
    indices = rng.integers(0, n_rows, size=(repetitions, n_rows), endpoint=False)
    counts = np.zeros((repetitions, n_rows), dtype=np.float64)
    rows = np.repeat(np.arange(repetitions), n_rows)
    np.add.at(counts, (rows, indices.ravel()), 1.0)
    return counts / float(n_rows)


def _quadratic_forms(
    weights: np.ndarray,
    distances: np.ndarray,
    chunk_size: int = 64,
) -> np.ndarray:
    """Compute diagonal bootstrap quadratic forms in bounded memory."""
    values = np.empty(weights.shape[0], dtype=np.float64)
    for start in range(0, weights.shape[0], chunk_size):
        stop = min(start + chunk_size, weights.shape[0])
        chunk = weights[start:stop]
        values[start:stop] = np.einsum(
            "bi,bi->b",
            chunk @ distances,
            chunk,
            optimize=True,
        )
    return values


def _cross_forms(
    left_weights: np.ndarray,
    distances: np.ndarray,
    right_weights: np.ndarray,
    chunk_size: int = 64,
) -> np.ndarray:
    """Compute bootstrap cross-distance means in bounded memory."""
    values = np.empty(left_weights.shape[0], dtype=np.float64)
    for start in range(0, left_weights.shape[0], chunk_size):
        stop = min(start + chunk_size, left_weights.shape[0])
        left = left_weights[start:stop]
        right = right_weights[start:stop]
        values[start:stop] = np.einsum(
            "bi,bi->b",
            left @ distances,
            right,
            optimize=True,
        )
    return values


def _hedges_correction(n_reference: int, n_intervention: int) -> float:
    """Return the standard small-sample Hedges correction."""
    degrees_freedom = n_reference + n_intervention - 2
    if degrees_freedom <= 0:
        raise ValueError("Hedges g requires positive pooled degrees of freedom.")
    return 1.0 - 3.0 / (4.0 * degrees_freedom - 1.0)


def _readout_effects(
    reference: np.ndarray,
    intervention: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized effects and both zero-variance indicators."""
    n_reference, n_intervention = len(reference), len(intervention)
    reference_mean = reference.mean(axis=0, dtype=np.float64)
    intervention_mean = intervention.mean(axis=0, dtype=np.float64)
    reference_sd = reference.std(axis=0, ddof=1, dtype=np.float64)
    intervention_sd = intervention.std(axis=0, ddof=1, dtype=np.float64)
    identical_constant = (
        (reference_sd == 0.0) & (intervention_sd == 0.0) & (reference_mean == intervention_mean)
    )
    undefined_constant = ((reference_sd == 0.0) | (intervention_sd == 0.0)) & ~identical_constant
    pooled_variance = (
        (n_reference - 1) * reference_sd**2 + (n_intervention - 1) * intervention_sd**2
    ) / (n_reference + n_intervention - 2)
    pooled_sd = np.sqrt(pooled_variance)
    hedges = np.full(reference.shape[1], np.nan, dtype=np.float64)
    log_sd = np.full(reference.shape[1], np.nan, dtype=np.float64)
    valid = ~(identical_constant | undefined_constant)
    hedges[valid] = (
        _hedges_correction(n_reference, n_intervention)
        * (intervention_mean[valid] - reference_mean[valid])
        / pooled_sd[valid]
    )
    log_sd[valid] = np.log(intervention_sd[valid] / reference_sd[valid])
    if not np.isfinite(hedges[valid]).all() or not np.isfinite(log_sd[valid]).all():
        raise ValueError("Non-finite readout effect encountered.")
    return hedges, log_sd, identical_constant, undefined_constant


def _bootstrap_effects(
    reference: np.ndarray,
    intervention: np.ndarray,
    reference_weights: np.ndarray,
    intervention_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute readout effects for every deterministic bootstrap resample."""

    def moments(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = weights @ values
        second = weights @ (values**2)
        variances = (
            len(values)
            / (len(values) - 1)
            * np.maximum(
                0.0,
                second - means**2,
            )
        )
        return means, np.sqrt(variances)

    reference_mean, reference_sd = moments(reference, reference_weights)
    intervention_mean, intervention_sd = moments(intervention, intervention_weights)
    pooled_sd = np.sqrt(
        ((len(reference) - 1) * reference_sd**2 + (len(intervention) - 1) * intervention_sd**2)
        / (len(reference) + len(intervention) - 2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        hedges = (
            _hedges_correction(len(reference), len(intervention))
            * (intervention_mean - reference_mean)
            / pooled_sd
        )
        log_sd = np.log(intervention_sd / reference_sd)
    return hedges, log_sd


def _percentile_interval(values: np.ndarray, repetitions: int) -> tuple[float, float, int]:
    """Return a finite 95% percentile interval and effective bootstrap count."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < max(2, math.ceil(0.8 * repetitions)):
        raise ValueError("Too few finite deterministic bootstrap replicates for a CI.")
    low, high = np.quantile(finite, CI_QUANTILES)
    return float(low), float(high), int(len(finite))


def _optional_percentile_interval(
    values: np.ndarray,
    repetitions: int,
) -> tuple[float | None, float | None, int, bool]:
    """Return a descriptive CI, explicitly undefined after bootstrap degeneracy."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    finite_count = int(len(finite))
    if finite_count < max(2, math.ceil(0.8 * repetitions)):
        return None, None, finite_count, True
    low, high = np.quantile(finite, CI_QUANTILES)
    return float(low), float(high), finite_count, False


def _ordered_interventions(
    campaign: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the outcome-independent physical plotting order."""
    targets = {str(record["target"]): record for record in catalog["targets"]}
    rows = []
    for campaign_index, intervention in enumerate(map(str, campaign["interventions"])):
        parsed = parse_intervention_name(intervention)
        target = str(parsed["target"])
        strength = str(parsed["strength"])
        if strength not in STRENGTH_ORDER:
            raise ValueError(f"Intervention lacks frozen strength: {intervention}")
        target_record = targets[target]
        rows.append(
            {
                "intervention": intervention,
                "target": target,
                "strength": strength,
                "physical_class": str(target_record["physical_class"]),
                "semantic_family": str(target_record["semantic_family"]),
                "knob": str(target_record["knob"]),
                "campaign_intervention_index": campaign_index,
                "expected_readout_descendants": list(
                    map(str, target_record["expected_readout_descendants"])
                ),
            }
        )
    order = pd.DataFrame(rows)
    order["_physical_order"] = order["physical_class"].map(PHYSICAL_CLASS_ORDER)
    order["_strength_order"] = order["strength"].map(STRENGTH_ORDER)
    order = order.sort_values(
        [
            "_physical_order",
            "semantic_family",
            "target",
            "_strength_order",
            "intervention",
        ],
        kind="stable",
    ).reset_index(drop=True)
    order["intervention_order"] = np.arange(len(order), dtype=int)
    return order.drop(columns=["_physical_order", "_strength_order"])


def _compute_characterization(
    campaign: Mapping[str, Any],
    plan: Mapping[str, Any],
    catalog: Mapping[str, Any],
    paths: Mapping[str, Path],
    repetitions: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load frozen splits and compute shift/effect tables."""
    data_seed = int(plan["data_contract"]["data_seed"])
    reference_raw, reference_pairing, pairing_names = _load_reference_table(
        paths["uniform_reference"]
    )
    train_idx, valid_idx, test_idx, normal_idx = _reference_split_indices(
        len(reference_raw),
        data_seed,
        reference_pairing,
    )
    contract = plan["data_contract"]
    if len(train_idx) != int(contract["normalization"]["fit_count"]):
        raise ValueError("Reconstructed reference training count differs from the frozen plan.")
    if len(normal_idx) != int(contract["reference_count"]):
        raise ValueError("Reconstructed test.normal count differs from the frozen plan.")
    normalizer = CausalChamberNormalizer.fit(
        torch.as_tensor(reference_raw[train_idx], dtype=torch.float32),
        normalize=True,
        robust_quantiles=ROBUST_QUANTILES,
        clip_value=CLIP_VALUE,
    )
    reference = normalizer.transform(
        torch.as_tensor(reference_raw[normal_idx], dtype=torch.float32)
    ).numpy()
    order = _ordered_interventions(campaign, catalog)

    magnitude_rows = []
    effect_rows = []
    signal_index_hashes = {}
    for record in order.to_dict("records"):
        intervention = str(record["intervention"])
        signal_raw = _load_readouts(paths[intervention])
        signal_idx = _signal_test_indices(len(signal_raw), data_seed)
        if len(signal_idx) != int(contract["signal_count_per_intervention"]):
            raise ValueError(f"Reconstructed signal test count differs for {intervention}.")
        signal_index_hashes[intervention] = _hash_indices(signal_idx)
        signal = normalizer.transform(
            torch.as_tensor(signal_raw[signal_idx], dtype=torch.float32)
        ).numpy()

        reference_distances, signal_distances, cross_distances = _distance_matrices(
            reference,
            signal,
        )
        energy = _biased_energy_from_distances(
            reference_distances,
            signal_distances,
            cross_distances,
        )
        seed = bootstrap_seed + int(record["campaign_intervention_index"])
        rng = np.random.default_rng(seed)
        reference_weights = _bootstrap_weights(rng, repetitions, len(reference))
        signal_weights = _bootstrap_weights(rng, repetitions, len(signal))
        energy_bootstrap = np.maximum(
            0.0,
            2.0 * _cross_forms(reference_weights, cross_distances, signal_weights)
            - _quadratic_forms(reference_weights, reference_distances)
            - _quadratic_forms(signal_weights, signal_distances),
        )
        energy_low, energy_high, energy_repetitions = _percentile_interval(
            energy_bootstrap,
            repetitions,
        )
        magnitude_rows.append(
            {
                key: record[key]
                for key in (
                    "intervention",
                    "target",
                    "strength",
                    "physical_class",
                    "semantic_family",
                    "knob",
                    "campaign_intervention_index",
                    "intervention_order",
                )
            }
            | {
                "n_reference": len(reference),
                "n_intervention": len(signal),
                "biased_energy_distance": energy,
                "bootstrap_ci95_low": energy_low,
                "bootstrap_ci95_high": energy_high,
                "bootstrap_repetitions": repetitions,
                "bootstrap_finite_repetitions": energy_repetitions,
                "bootstrap_seed": seed,
            }
        )

        reference_mean = reference.mean(axis=0, dtype=np.float64)
        signal_mean = signal.mean(axis=0, dtype=np.float64)
        hedges, log_sd, identical_constant, undefined_constant = _readout_effects(
            reference,
            signal,
        )
        hedges_bootstrap, log_sd_bootstrap = _bootstrap_effects(
            reference,
            signal,
            reference_weights,
            signal_weights,
        )
        descendants = set(record["expected_readout_descendants"])
        for index, readout in enumerate(READOUT_FEATURES):
            if identical_constant[index] or undefined_constant[index]:
                hedges_low = hedges_high = None
                log_low = log_high = None
                hedges_finite = log_finite = 0
                hedges_ci_undefined = log_ci_undefined = True
                hedges_value = log_value = None
            else:
                (
                    hedges_low,
                    hedges_high,
                    hedges_finite,
                    hedges_ci_undefined,
                ) = _optional_percentile_interval(
                    hedges_bootstrap[:, index],
                    repetitions,
                )
                (
                    log_low,
                    log_high,
                    log_finite,
                    log_ci_undefined,
                ) = _optional_percentile_interval(
                    log_sd_bootstrap[:, index],
                    repetitions,
                )
                hedges_value = float(hedges[index])
                log_value = float(log_sd[index])
            effect_rows.append(
                {
                    key: record[key]
                    for key in (
                        "intervention",
                        "target",
                        "strength",
                        "physical_class",
                        "semantic_family",
                        "campaign_intervention_index",
                        "intervention_order",
                    )
                }
                | {
                    "readout": readout,
                    "readout_order": index,
                    "expected_descendant": readout in descendants,
                    "reference_mean": float(reference_mean[index]),
                    "intervention_mean": float(signal_mean[index]),
                    "mean_difference": float(signal_mean[index] - reference_mean[index]),
                    "hedges_g": hedges_value,
                    "hedges_g_bootstrap_ci95_low": hedges_low,
                    "hedges_g_bootstrap_ci95_high": hedges_high,
                    "hedges_g_bootstrap_finite_repetitions": hedges_finite,
                    "hedges_g_bootstrap_ci_undefined": hedges_ci_undefined,
                    "log_sd_ratio": log_value,
                    "log_sd_ratio_bootstrap_ci95_low": log_low,
                    "log_sd_ratio_bootstrap_ci95_high": log_high,
                    "log_sd_ratio_bootstrap_finite_repetitions": log_finite,
                    "log_sd_ratio_bootstrap_ci_undefined": log_ci_undefined,
                    "identical_constant_null": bool(identical_constant[index]),
                    "undefined_zero_variance_effect": bool(undefined_constant[index]),
                    "bootstrap_seed": seed,
                    "bootstrap_repetitions": repetitions,
                }
            )

    magnitude = pd.DataFrame(magnitude_rows)
    effects = pd.DataFrame(effect_rows)
    descendant_rows = []
    for intervention, group in effects.groupby("intervention", sort=False):
        metadata = group.iloc[0]
        descendants = group[group["expected_descendant"]]
        non_descendants = group[~group["expected_descendant"]]
        row = {
            "intervention": intervention,
            "target": metadata["target"],
            "strength": metadata["strength"],
            "physical_class": metadata["physical_class"],
            "semantic_family": metadata["semantic_family"],
            "intervention_order": int(metadata["intervention_order"]),
            "n_expected_descendants": len(descendants),
            "n_non_descendants": len(non_descendants),
            "n_undefined_zero_variance_effects": int(
                group["undefined_zero_variance_effect"].sum()
            ),
        }
        for column in ("hedges_g", "log_sd_ratio"):
            descendant_values = descendants[column].abs().dropna()
            non_descendant_values = non_descendants[column].abs().dropna()
            descendant_mean = None if descendant_values.empty else float(descendant_values.mean())
            non_descendant_mean = (
                None if non_descendant_values.empty else float(non_descendant_values.mean())
            )
            row[f"mean_abs_{column}_expected_descendants"] = descendant_mean
            row[f"mean_abs_{column}_non_descendants"] = non_descendant_mean
            row[f"mean_abs_{column}_descendant_minus_non"] = (
                None
                if descendant_mean is None or non_descendant_mean is None
                else descendant_mean - non_descendant_mean
            )
        descendant_rows.append(row)
    descendants = pd.DataFrame(descendant_rows)
    split_contract = {
        "data_seed": data_seed,
        "reference_total": len(reference_raw),
        "reference_train_count": len(train_idx),
        "reference_valid_base_count": len(valid_idx),
        "reference_test_base_count": len(test_idx),
        "reference_test_normal_count": len(normal_idx),
        "reference_train_indices_sha256": _hash_indices(train_idx),
        "reference_valid_indices_sha256": _hash_indices(valid_idx),
        "reference_test_indices_sha256": _hash_indices(test_idx),
        "reference_test_normal_indices_sha256": _hash_indices(normal_idx),
        "signal_test_indices_sha256": signal_index_hashes,
        "features": list(READOUT_FEATURES),
        "pairing_features": pairing_names,
        "test_normal_pairing": "repository metadata_nearest ordering",
        "normalizer": normalizer.to_contract(),
    }
    return magnitude, effects, descendants, split_contract


def _plot_readout_effects(effects: pd.DataFrame, path: Path) -> None:
    """Plot ordered location and scale effects."""
    intervention_order = (
        effects[["intervention_order", "intervention"]]
        .drop_duplicates()
        .sort_values("intervention_order")["intervention"]
        .tolist()
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, max(12.0, 0.24 * len(intervention_order))),
        sharey=True,
    )
    for axis, (column, title) in zip(
        axes,
        (
            ("hedges_g", "Hedges g (intervention − reference)"),
            ("log_sd_ratio", "log(SD intervention / SD reference)"),
        ),
    ):
        matrix = (
            effects.pivot(index="intervention", columns="readout", values=column)
            .reindex(index=intervention_order, columns=READOUT_FEATURES)
            .to_numpy(dtype=float)
        )
        bound = max(float(np.nanmax(np.abs(matrix))), 1.0e-6)
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap="coolwarm",
            vmin=-bound,
            vmax=bound,
        )
        image.cmap.set_bad(color="lightgray")
        axis.set_xticks(
            np.arange(len(READOUT_FEATURES)),
            READOUT_FEATURES,
            rotation=45,
            ha="right",
        )
        axis.set_yticks(
            np.arange(len(intervention_order)),
            intervention_order,
            fontsize=6,
        )
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    figure.suptitle("Outcome-independent physical readout effects in frozen catalog order")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_shift_magnitude(magnitude: pd.DataFrame, path: Path) -> None:
    """Plot ordered biased energy distances and deterministic bootstrap intervals."""
    ordered = magnitude.sort_values("intervention_order", kind="stable")
    y = np.arange(len(ordered))
    values = ordered["biased_energy_distance"].to_numpy(dtype=float)
    low = ordered["bootstrap_ci95_low"].to_numpy(dtype=float)
    high = ordered["bootstrap_ci95_high"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(10.0, max(12.0, 0.24 * len(ordered))))
    axis.hlines(y, low, high, linewidth=1.2)
    axis.scatter(values, y, marker="o", s=18)
    axis.set_yticks(y, ordered["intervention"], fontsize=6)
    axis.invert_yaxis()
    axis.set_xlabel("Biased multivariate energy distance")
    axis.set_title("Physical shift magnitude (95% deterministic bootstrap CI)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(
    campaign_root: Path,
    shift_plan_path: Path,
    target_catalog_path: Path,
    selection_provenance_path: Path,
    selection_provenance_sha256: str,
    output_dir: Path,
) -> list[Path]:
    """Run the gated, outcome-independent physical-shift characterization."""
    campaign_root = campaign_root.expanduser().resolve()
    output_dir = _validate_output_path(output_dir, campaign_root)
    campaign, plan, catalog, repetitions, bootstrap_seed = _validate_frozen_design(
        campaign_root,
        shift_plan_path,
        target_catalog_path,
    )
    selection_contract = _validate_selection_frozen(
        campaign_root,
        campaign,
        selection_provenance_path,
        selection_provenance_sha256,
    )
    dataset_dir, dataset_paths, dataset_records = _dataset_paths(campaign)
    magnitude, effects, descendants, split_contract = _compute_characterization(
        campaign,
        plan,
        catalog,
        dataset_paths,
        repetitions,
        bootstrap_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, table in (
        ("physical_shift_magnitude.csv", magnitude),
        ("readout_effects.csv", effects),
        ("expected_descendant_summary.csv", descendants),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)

    readout_figure = output_dir / "ordered_readout_effects.png"
    magnitude_figure = output_dir / "ordered_shift_magnitude.png"
    _plot_readout_effects(effects, readout_figure)
    _plot_shift_magnitude(magnitude, magnitude_figure)
    outputs.extend([readout_figure, magnitude_figure])

    status_path = output_dir / "component_status.json"
    _atomic_json(
        status_path,
        {
            "schema_version": 1,
            "components": {
                "physical_shift_characterization": {
                    "status": "completed",
                    "selection_blind": True,
                },
                "performance_association": {
                    "status": "pending",
                    "reason": (
                        "No performance table is accepted by this outcome-independent tool; "
                        "join only after final reporting-seed aggregation."
                    ),
                    "frozen_methods": plan["performance_association"]["methods"],
                    "prohibited": plan["performance_association"]["prohibited"],
                },
            },
        },
    )
    outputs.append(status_path)

    provenance_path = output_dir / "physical_shift_provenance.json"
    _atomic_json(
        provenance_path,
        {
            "schema_version": 1,
            "campaign_id": campaign["campaign_id"],
            "inputs": {
                "campaign": {
                    "path": str((campaign_root / "campaign.json").resolve()),
                    "sha256": _sha256(campaign_root / "campaign.json"),
                },
                "shift_plan": {
                    "path": str(shift_plan_path.expanduser().resolve()),
                    "sha256": _sha256(shift_plan_path.expanduser().resolve()),
                },
                "target_catalog": {
                    "path": str(target_catalog_path.expanduser().resolve()),
                    "sha256": _sha256(target_catalog_path.expanduser().resolve()),
                },
                **selection_contract,
            },
            "dataset": {
                "directory": str(dataset_dir),
                "tree_sha256": campaign["dataset_tree_sha256"],
                "files": dataset_records,
            },
            "split_and_normalization_contract": split_contract,
            "estimands": {
                "joint": {
                    "name": plan["joint_shift"]["name"],
                    "formula": plan["joint_shift"]["formula"],
                    "negative_roundoff": "clamped to zero",
                },
                "readout_location": plan["readout_effects"]["location"],
                "readout_scale": plan["readout_effects"]["scale"],
                "zero_variance_protocol_amendment": {
                    "trigger": (
                        "Observed non-identical zero-variance readouts made the frozen "
                        "standardized estimands mathematically undefined."
                    ),
                    "policy": ZERO_VARIANCE_AMENDMENT,
                    "selection_impact": "none; analysis ran only after selection was frozen",
                },
                "bootstrap_degeneracy_protocol_amendment": {
                    "trigger": (
                        "Some finite readout point estimates had fewer than 80% finite "
                        "bootstrap replicates because resampled groups had zero variance."
                    ),
                    "policy": BOOTSTRAP_CI_AMENDMENT,
                    "selection_impact": "none; analysis ran only after selection was frozen",
                },
                "bootstrap": {
                    "repetitions": repetitions,
                    "base_seed": bootstrap_seed,
                    "strata": ["reference", "intervention"],
                    "interval": "95% percentile, unadjusted descriptive",
                },
            },
            "repository_sources": {
                "causal_chamber_data_component": {
                    "path": str((REPO_ROOT / "src/data/components/causal_chamber.py").resolve()),
                    "sha256": _sha256(REPO_ROOT / "src/data/components/causal_chamber.py"),
                }
            },
            "performance_association": {
                "status": "pending",
                "frozen_contract": plan["performance_association"],
            },
            "outputs": {path.name: _sha256(path) for path in outputs},
        },
    )
    outputs.append(provenance_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--shift-plan", type=Path, required=True)
    parser.add_argument("--target-catalog", type=Path, required=True)
    parser.add_argument("--selection-provenance", type=Path, required=True)
    parser.add_argument("--selection-provenance-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line physical-shift analysis."""
    args = parse_args(argv)
    outputs = analyze(
        args.campaign_root,
        args.shift_plan,
        args.target_catalog,
        args.selection_provenance,
        args.selection_provenance_sha256,
        args.output_dir,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
