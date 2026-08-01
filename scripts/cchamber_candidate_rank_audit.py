#!/usr/bin/env python3
"""Frozen, outcome-blind secondary candidate-rank audit for Causal Chamber."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from mlflow import MlflowClient
from scipy import stats
from torchmetrics.classification import BinaryAveragePrecision

from src.utils.pairing.io import compose_config
from src.utils.pairing.table import load_pair_table

PANEL_PATH = Path(os.environ.get("CCHAMBER_RANK_PANEL", "/nonexistent/candidate_audit_panel.json"))
PANEL_SHA256 = os.environ.get("CCHAMBER_RANK_PANEL_SHA256", "")
CONTRACT_PATH = Path(
    os.environ.get("CCHAMBER_RANK_CONTRACT", "/nonexistent/candidate_audit_contract.json")
)
CONTRACT_SHA256 = os.environ.get("CCHAMBER_RANK_CONTRACT_SHA256", "")
CAMPAIGN_COMMIT = os.environ.get(
    "CCHAMBER_RANK_CAMPAIGN_COMMIT", "63b941a287c48c84e2537d0cfbd07c2240435c0e"
)
MODELS = ("ae", "vae", "svdd", "realnvp")
STRATEGIES = (
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "cap_cdf",
    "cap_random",
    "drift",
    "wasserstein",
)
DIRECTIONS = {
    "cap_metadata_nearest": "maximize",
    "cap_encoder_nearest": "maximize",
    "cap_cdf": "maximize",
    "cap_random": "maximize",
    "drift": "minimize",
    "wasserstein": "minimize",
}
MONITORS = {
    "cap_metadata_nearest": "val/summary/cap_metadata_nearest_ema_normal_vs_reference_normal",
    "cap_encoder_nearest": "val/summary/cap_encoder_nearest_ema_normal_vs_reference_normal",
    "cap_cdf": "val/summary/cap_cdf_ema_normal_vs_reference_normal",
    "cap_random": "val/summary/cap_random_ema_normal_vs_reference_normal",
    "drift": "val/summary/operational_drift_ema",
    "wasserstein": "val/summary/w1dist_ema_normal_vs_reference_normal",
}
METRICS = ("auprc", "efficiency_operational")
DEVELOPMENT_SEEDS = (101, 202, 303, 404, 505)
EXPECTED_TRAJECTORIES = 192
EXPECTED_CHECKPOINTS = 1_152
EXPECTED_INTERVENTIONS = 58
EXPECTED_ROWS = 133_632
PANEL_CANDIDATE_IDS = (
    "000",
    "001",
    "006",
    "010",
    "015",
    "019",
    "024",
    "028",
    "033",
    "037",
    "042",
    "046",
    "051",
    "055",
    "060",
    "064",
)
PANEL_REPORTING_SEEDS = (1001, 1002, 1003)


def _set_design_identity(
    panel: Path, panel_sha256: str, contract: Path, contract_sha256: str, commit: str
) -> None:
    """Set the immutable design identity for this process."""
    global PANEL_PATH, PANEL_SHA256, CONTRACT_PATH, CONTRACT_SHA256, CAMPAIGN_COMMIT
    PANEL_PATH = panel.resolve()
    PANEL_SHA256 = panel_sha256
    CONTRACT_PATH = contract.resolve()
    CONTRACT_SHA256 = contract_sha256
    CAMPAIGN_COMMIT = commit


def _sha256(path: Path) -> str:
    """Hash one file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    """Serialize strict JSON deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json(path: Path) -> Any:
    """Load a required JSON artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any, *, create: bool = False) -> None:
    """Atomically write JSON, optionally refusing to replace an existing marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        if create:
            os.link(temporary, path)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_hash(path: Path, expected: str, label: str) -> None:
    """Require an immutable artifact hash."""
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}, expected {expected}: {path}")


def _git(*args: str) -> str:
    """Run a fixed git inspection command."""
    return subprocess.check_output(  # nosec B603 B607
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _audit_revision(*, require_clean: bool = True) -> tuple[str, str]:
    """Return current audit-code revision and optionally enforce a clean tree."""
    if require_clean and _git("status", "--porcelain"):
        raise RuntimeError("Candidate-rank audit execution requires a clean worktree.")
    return _git("rev-parse", "HEAD"), _git("branch", "--show-current")


def _attempt_id() -> str:
    """Return a sortable retry identifier."""
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"_j{os.environ.get('SLURM_JOB_ID', 'local')}"
        + f"_a{os.environ.get('SLURM_ARRAY_TASK_ID', 'none')}_p{os.getpid()}"
    )


def _hydra_value(value: Any) -> str:
    """Render a JSON-compatible value as a Hydra override."""
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(value, separators=(",", ":"))


def _validate_frozen_design() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and load only the two authorized frozen design inputs."""
    _require_hash(PANEL_PATH, PANEL_SHA256, "candidate audit panel")
    _require_hash(CONTRACT_PATH, CONTRACT_SHA256, "candidate audit execution contract")
    panel = _load_json(PANEL_PATH)
    contract = _load_json(CONTRACT_PATH)
    if (
        panel["campaign_git_commit"] != CAMPAIGN_COMMIT
        or contract["candidate_audit_panel_sha256"] != PANEL_SHA256
        or panel["campaign_id"] != contract["campaign_id"]
        or tuple(panel["models"]) != MODELS
        or tuple(contract["models"]) != MODELS
        or panel["candidate_ids"] != contract["panel_candidate_ids"]
        or panel["reporting_seeds"] != contract["reporting_seeds"]
        or int(contract["training_unit"]["count"]) != EXPECTED_TRAJECTORIES
        or int(contract["sealed_evaluation"]["expected_rows"]) != EXPECTED_ROWS
    ):
        raise ValueError("Frozen candidate-rank design contract changed.")
    branches = contract["training_unit"]["checkpoint_branches"]
    if {str(branch["strategy"]): str(branch["direction"]) for branch in branches} != DIRECTIONS:
        raise ValueError("Frozen checkpoint branch contract changed.")
    return dict(panel), dict(contract)


def _require_external_root(root: Path, campaign_manifest: Path) -> Path:
    """Require the sidecar root to remain outside the primary campaign tree."""
    root = root.resolve()
    campaign_root = campaign_manifest.resolve().parent
    if root == campaign_root or campaign_root in root.parents:
        raise ValueError("Candidate-rank audit root must be outside the campaign tree.")
    return root


def _validate_campaign(
    path: Path, expected_sha256: str, *, full_data_tree: bool
) -> dict[str, Any]:
    """Validate the label-free campaign manifest and frozen raw-data identities."""
    _require_hash(path, expected_sha256, "campaign manifest")
    campaign = _load_json(path)
    if (
        (CAMPAIGN_COMMIT and campaign.get("git_commit") != CAMPAIGN_COMMIT)
        or len(campaign.get("interventions", ())) != EXPECTED_INTERVENTIONS
        or len(set(campaign["interventions"])) != EXPECTED_INTERVENTIONS
    ):
        raise ValueError("Campaign identity or intervention catalog changed.")
    sources = campaign.get("dataset_files")
    if not isinstance(sources, list) or len(sources) != 59:
        raise ValueError("Campaign must pin exactly 59 raw Causal Chamber CSVs.")
    normalized = []
    found_uniform_reference = False
    for source in sources:
        source_path = Path(source["path"]).resolve()
        is_uniform_reference = source_path.name == "uniform_reference.csv"
        found_uniform_reference = found_uniform_reference or is_uniform_reference
        if full_data_tree or is_uniform_reference:
            _require_hash(source_path, str(source["sha256"]), "raw dataset source")
            if source_path.stat().st_size != int(source["size"]):
                raise ValueError(f"Raw dataset source size changed: {source_path}")
        normalized.append(
            {
                "path": str(source_path),
                "size": int(source["size"]),
                "sha256": str(source["sha256"]),
            }
        )
    if not found_uniform_reference:
        raise ValueError("Campaign dataset tree has no uniform_reference.csv source.")
    tree_hash = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
    if tree_hash != campaign.get("dataset_tree_sha256"):
        raise ValueError("Campaign dataset-tree identity changed.")
    return dict(campaign)


def _validate_pairing_provenance(
    pairing_manifest_path: Path,
    pairing_manifest_sha256: str,
    encoder_pair_table: Path,
    encoder_pair_table_sha256: str,
    campaign: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the primary seed-123 validation table and its encoder."""
    pairing_manifest_path = pairing_manifest_path.resolve()
    encoder_pair_table = encoder_pair_table.resolve()
    _require_hash(pairing_manifest_path, pairing_manifest_sha256, "pairing manifest")
    pairing = _load_json(pairing_manifest_path)
    if (
        pairing.get("campaign_id") != campaign["campaign_id"]
        or int(pairing.get("primary_encoder_seed", -1)) != 123
        or Path(pairing.get("primary_validation_table", "")).resolve() != encoder_pair_table
        or pairing.get("primary_validation_table_sha256") != encoder_pair_table_sha256
    ):
        raise ValueError("Primary pairing-manifest identity changed.")
    runs = pairing.get("encoder_runs")
    if not isinstance(runs, list):
        raise ValueError("Pairing manifest has no encoder-run provenance.")
    expected_encoder_seeds = set(map(int, campaign["pair_encoder_seeds"]))
    if (
        len(runs) != len(expected_encoder_seeds)
        or {int(run["encoder_seed"]) for run in runs} != expected_encoder_seeds
    ):
        raise ValueError("Pairing encoder-seed provenance is not exact.")
    primary_runs = [run for run in runs if int(run["encoder_seed"]) == 123]
    if len(primary_runs) != 1:
        raise ValueError("Pairing manifest must contain one primary seed-123 encoder.")
    primary = dict(primary_runs[0])
    checkpoint = Path(primary["encoder_checkpoint"]).resolve()
    if (
        primary.get("campaign_id") != campaign["campaign_id"]
        or primary.get("git_commit") != CAMPAIGN_COMMIT
        or int(primary.get("data_seed", -1)) != int(campaign["data_seed"])
        or Path(primary["validation_table"]).resolve() != encoder_pair_table
        or primary.get("validation_table_sha256") != encoder_pair_table_sha256
    ):
        raise ValueError("Primary pairing encoder provenance changed.")
    _require_hash(checkpoint, str(primary["encoder_checkpoint_sha256"]), "pairing encoder")
    _require_hash(encoder_pair_table, encoder_pair_table_sha256, "encoder validation pair table")
    table = load_pair_table(
        encoder_pair_table,
        expected_dataset_1="normal",
        expected_dataset_2="reference_normal",
        expected_split="validate",
    )
    metadata = table["metadata"]
    if (
        Path(table["encoder_ckpt"]).resolve() != checkpoint
        or metadata["encoder_checkpoint_sha256"] != primary["encoder_checkpoint_sha256"]
        or metadata.get("pairing_mode") != "one_to_one_nearest"
        or metadata.get("normalized") is not True
        or int(metadata.get("data_seed", -1)) != int(campaign["data_seed"])
        or int(metadata.get("n_pairs", -1)) != 1_000
        or int(metadata.get("n_dataset_1", -1)) != 1_000
        or int(metadata.get("n_dataset_2", -1)) != 1_000
    ):
        raise ValueError("Primary validation pair-table metadata/provenance changed.")
    return dict(pairing), primary


def _validate_candidate_metrics_provenance(
    provenance_path: Path,
    provenance_sha256: str,
    candidate_metrics_path: Path,
    candidate_metrics_sha256: str,
    campaign_manifest_path: Path,
    campaign_manifest_sha256: str,
    campaign: Mapping[str, Any],
    primary_pair_table_sha256: str,
) -> pd.DataFrame:
    """Authenticate the label-free candidate CSV and its globally shared pools."""
    provenance_path = provenance_path.resolve()
    candidate_metrics_path = candidate_metrics_path.resolve()
    campaign_manifest_path = campaign_manifest_path.resolve()
    _require_hash(provenance_path, provenance_sha256, "candidate metrics provenance")
    provenance = _load_json(provenance_path)
    if (
        Path(provenance.get("candidate_metrics", "")).resolve() != candidate_metrics_path
        or provenance.get("candidate_metrics_sha256") != candidate_metrics_sha256
        or Path(provenance.get("campaign", "")).resolve() != campaign_manifest_path
        or provenance.get("campaign_sha256") != campaign_manifest_sha256
    ):
        raise ValueError("Candidate-metrics provenance identity changed.")
    _require_hash(candidate_metrics_path, candidate_metrics_sha256, "candidate metrics")
    if (
        campaign.get("git_commit") != CAMPAIGN_COMMIT
        or tuple(campaign.get("models", ())) != MODELS
        or set(campaign.get("strategies", ())) != set(STRATEGIES)
        or tuple(map(int, campaign.get("development_seeds", ()))) != DEVELOPMENT_SEEDS
    ):
        raise ValueError("Candidate search model/strategy/development-seed design changed.")
    frame = pd.read_csv(candidate_metrics_path, dtype={"candidate_id": str})
    required = {
        "dataset",
        "model",
        "seed",
        "candidate_id",
        "strategy",
        "value",
        "params_json",
        "pool_sha256",
        "pair_table_sha256",
        "git_commit",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Candidate metrics miss provenance columns: {sorted(required - set(frame))}"
        )
    frame["candidate_id"] = frame["candidate_id"].astype(str).str.zfill(3)
    frame["seed"] = frame["seed"].astype(int)
    if (
        set(frame["model"]) != set(MODELS)
        or set(frame["strategy"]) != set(STRATEGIES)
        or set(frame["seed"]) != set(DEVELOPMENT_SEEDS)
        or set(frame["dataset"]) != {"cchamber"}
        or set(frame["git_commit"]) != {CAMPAIGN_COMMIT}
        or set(frame["pair_table_sha256"]) != {primary_pair_table_sha256}
    ):
        raise ValueError("Candidate metrics identities are not exact.")
    excluded_by_model = provenance.get("globally_excluded_candidates")
    survivor_counts = provenance.get("surviving_candidates_per_model")
    if (
        not isinstance(excluded_by_model, Mapping)
        or set(excluded_by_model) != set(MODELS)
        or not isinstance(survivor_counts, Mapping)
        or set(survivor_counts) != set(MODELS)
    ):
        raise ValueError("Global candidate survivor provenance is incomplete.")
    for model in MODELS:
        pool_path = campaign_manifest_path.parent / "design" / f"{model}_candidates.json"
        expected_pool_sha256 = str(campaign["pool_sha256"][model])
        _require_hash(pool_path, expected_pool_sha256, f"{model} candidate pool")
        pool = _load_json(pool_path)
        if (
            not isinstance(pool, list)
            or len(pool) != int(campaign["n_candidates_per_model"])
            or {str(row["model"]) for row in pool} != {model}
        ):
            raise ValueError(f"Candidate pool structure changed for {model}.")
        pool_by_id = {str(row["candidate_id"]).zfill(3): row for row in pool}
        if len(pool_by_id) != len(pool):
            raise ValueError(f"Candidate pool IDs are duplicated for {model}.")
        excluded = {str(value).zfill(3) for value in excluded_by_model[model]}
        if not excluded.issubset(pool_by_id):
            raise ValueError(f"Excluded candidate is not in the {model} pool.")
        expected_survivors = set(pool_by_id) - excluded
        model_frame = frame[frame["model"] == model]
        if set(model_frame["pool_sha256"]) != {expected_pool_sha256}:
            raise ValueError(f"Candidate pool hash changed in metrics for {model}.")
        grouped_sets = {
            (str(strategy), int(seed)): set(group["candidate_id"])
            for (strategy, seed), group in model_frame.groupby(["strategy", "seed"])
        }
        expected_groups = set(product(STRATEGIES, DEVELOPMENT_SEEDS))
        if set(grouped_sets) != expected_groups or any(
            candidates != expected_survivors for candidates in grouped_sets.values()
        ):
            raise ValueError(f"Globally surviving shared candidate pool changed for {model}.")
        if int(survivor_counts[model]) != len(expected_survivors):
            raise ValueError(f"Candidate survivor count changed for {model}.")
        for candidate_id, rows in model_frame.groupby("candidate_id"):
            expected_params = _canonical_json(pool_by_id[str(candidate_id)]["params"])
            if set(rows["params_json"].astype(str)) != {expected_params}:
                raise ValueError(f"Candidate parameters changed for {model}/{candidate_id}.")
    expected_rows = (
        sum(int(survivor_counts[model]) for model in MODELS)
        * len(DEVELOPMENT_SEEDS)
        * len(STRATEGIES)
    )
    if (
        len(frame) != expected_rows
        or int(provenance.get("n_rows", -1)) != expected_rows
        or int(provenance.get("expected_rows_after_global_exclusion", -1)) != expected_rows
        or frame.groupby(["model", "strategy", "seed", "candidate_id"]).size().ne(1).any()
    ):
        raise ValueError("Candidate metrics row/survivor coverage is not exact.")
    return frame


def _candidate_parameters(
    frame: pd.DataFrame,
    panel: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Extract one exact parameter set for each fixed model/candidate pair."""
    required = {"model", "candidate_id", "strategy", "seed", "value", "params_json"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Candidate metrics missing columns: {sorted(required - set(frame))}")
    frame = frame.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str).str.zfill(3)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for model, candidate_id in product(panel["models"], panel["candidate_ids"]):
        rows = frame[
            (frame["model"].astype(str) == str(model))
            & (frame["candidate_id"] == str(candidate_id))
        ]
        if rows.empty:
            raise ValueError(
                f"Frozen panel candidate is unavailable and must not be replaced: "
                f"{model}/{candidate_id}."
            )
        values = set(rows["params_json"].astype(str))
        if len(values) != 1:
            raise ValueError(f"Candidate parameters are inconsistent: {model}/{candidate_id}.")
        params = json.loads(next(iter(values)))
        output[(str(model), str(candidate_id))] = dict(params)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write non-empty rows with stable insertion-order columns."""
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _tracking_uri(root: Path) -> str:
    """Return the sidecar-only MLflow tracking URI."""
    return f"file:{(root / 'mlflow' / 'mlruns').resolve()}"


def design(
    root: Path,
    campaign_manifest: Path,
    campaign_manifest_sha256: str,
    candidate_metrics: Path,
    candidate_metrics_sha256: str,
    candidate_metrics_provenance: Path,
    candidate_metrics_provenance_sha256: str,
    pairing_manifest: Path,
    pairing_manifest_sha256: str,
    encoder_pair_table: Path,
    encoder_pair_table_sha256: str,
) -> dict[str, Any]:
    """Freeze the exact 192 shared trajectories before any sealed evaluation."""
    campaign_manifest = campaign_manifest.resolve()
    _require_hash(campaign_manifest, campaign_manifest_sha256, "campaign manifest")
    global CAMPAIGN_COMMIT
    CAMPAIGN_COMMIT = str(_load_json(campaign_manifest)["git_commit"])
    campaign = _validate_campaign(
        campaign_manifest,
        campaign_manifest_sha256,
        full_data_tree=True,
    )
    root = _require_external_root(root, campaign_manifest)
    if root.exists():
        raise FileExistsError(f"Audit root already exists: {root}")
    root.mkdir(parents=True)
    panel_path = root / "design" / "candidate_audit_panel.json"
    panel = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "campaign_git_commit": campaign["git_commit"],
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "outcome_status_at_lock": "No intervention performance outcome was used.",
        "selection_independence": "Outcome-independent secondary audit; cannot revise selection.",
        "panel_rule": "Baseline plus 15 fixed approximately evenly spaced Sobol IDs.",
        "candidate_ids": list(PANEL_CANDIDATE_IDS),
        "models": list(MODELS),
        "reporting_seeds": list(PANEL_REPORTING_SEEDS),
        "planned_estimands": ["rank association", "top-k enrichment", "panel regret"],
    }
    _atomic_json(panel_path, panel)
    panel_sha256 = _sha256(panel_path)
    branches = [
        {"strategy": strategy, "monitor": MONITORS[strategy], "direction": DIRECTIONS[strategy]}
        for strategy in STRATEGIES
    ]
    contract_path = root / "design" / "candidate_audit_execution_contract_v1.json"
    contract = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "candidate_audit_panel_sha256": panel_sha256,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "intervention_outcomes_inspected_before_freeze": False,
        "panel_candidate_ids": list(PANEL_CANDIDATE_IDS),
        "models": list(MODELS),
        "reporting_seeds": list(PANEL_REPORTING_SEEDS),
        "training_unit": {
            "count": EXPECTED_TRAJECTORIES,
            "checkpoint_branches": branches,
        },
        "sealed_evaluation": {
            "expected_rows": EXPECTED_ROWS,
            "metrics": list(METRICS),
        },
        "estimands": {"top_k": 4},
        "inference": {"auprc_holm_family": 24, "efficiency_holm_family": 24},
    }
    _atomic_json(contract_path, contract)
    contract_sha256 = _sha256(contract_path)
    _set_design_identity(
        panel_path, panel_sha256, contract_path, contract_sha256, str(campaign["git_commit"])
    )
    panel, contract = _validate_frozen_design()
    candidate_metrics = candidate_metrics.resolve()
    candidate_metrics_provenance = candidate_metrics_provenance.resolve()
    pairing_manifest = pairing_manifest.resolve()
    encoder_pair_table = encoder_pair_table.resolve()
    pairing, primary_encoder = _validate_pairing_provenance(
        pairing_manifest,
        pairing_manifest_sha256,
        encoder_pair_table,
        encoder_pair_table_sha256,
        campaign,
    )
    frame = _validate_candidate_metrics_provenance(
        candidate_metrics_provenance,
        candidate_metrics_provenance_sha256,
        candidate_metrics,
        candidate_metrics_sha256,
        campaign_manifest,
        str(contract["campaign_manifest_sha256"]),
        campaign,
        str(pairing["primary_validation_table_sha256"]),
    )
    forbidden = {"intervention", "auprc", "efficiency_operational", "test_outcome"}
    if forbidden & set(frame.columns):
        raise ValueError("Candidate audit design source contains sealed outcome columns.")
    params = _candidate_parameters(frame, panel)
    commit, branch = _audit_revision()
    trajectories = []
    for index, (model, candidate_id, seed) in enumerate(
        product(panel["models"], panel["candidate_ids"], panel["reporting_seeds"])
    ):
        parameters = params[(str(model), str(candidate_id))]
        trajectories.append(
            {
                "trajectory_index": index,
                "model": str(model),
                "candidate_id": str(candidate_id),
                "reporting_seed": int(seed),
                "params": parameters,
                "params_sha256": hashlib.sha256(
                    _canonical_json(parameters).encode("utf-8")
                ).hexdigest(),
            }
        )
    if len(trajectories) != EXPECTED_TRAJECTORIES:
        raise AssertionError("Trajectory construction did not produce exactly 192 records.")
    trajectory_path = root / "design" / "trajectories.json"
    _atomic_json(trajectory_path, trajectories)
    audit = {
        "schema_version": 1,
        "campaign_id": panel["campaign_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outcome_blind_at_design": True,
        "campaign_training_commit": CAMPAIGN_COMMIT,
        "audit_code_commit": commit,
        "audit_code_branch": branch,
        "panel": str(PANEL_PATH),
        "panel_sha256": PANEL_SHA256,
        "execution_contract": str(CONTRACT_PATH),
        "execution_contract_sha256": CONTRACT_SHA256,
        "campaign_manifest": str(campaign_manifest),
        "campaign_manifest_sha256": contract["campaign_manifest_sha256"],
        "candidate_metrics": str(candidate_metrics),
        "candidate_metrics_sha256": candidate_metrics_sha256,
        "candidate_metrics_provenance": str(candidate_metrics_provenance),
        "candidate_metrics_provenance_sha256": candidate_metrics_provenance_sha256,
        "pairing_manifest": str(pairing_manifest),
        "pairing_manifest_sha256": pairing_manifest_sha256,
        "primary_pair_encoder_seed": 123,
        "primary_pair_encoder_checkpoint": str(
            Path(primary_encoder["encoder_checkpoint"]).resolve()
        ),
        "primary_pair_encoder_checkpoint_sha256": primary_encoder["encoder_checkpoint_sha256"],
        "encoder_validation_pair_table": str(encoder_pair_table),
        "encoder_validation_pair_table_sha256": encoder_pair_table_sha256,
        "data_dir": str(Path(campaign["dataset_archive"]).resolve().parent.parent),
        "interventions": list(campaign["interventions"]),
        "trajectory_manifest": str(trajectory_path.resolve()),
        "trajectory_manifest_sha256": _sha256(trajectory_path),
        "expected_trajectories": EXPECTED_TRAJECTORIES,
        "expected_checkpoints": EXPECTED_CHECKPOINTS,
        "expected_rows": EXPECTED_ROWS,
        "tracking_uri": _tracking_uri(root),
    }
    _atomic_json(root / "audit.json", audit)
    _write_slurm_scripts(root)
    return audit


def _write_slurm_scripts(root: Path) -> None:
    """Write guarded canary, production, freeze, collection, and analysis jobs."""
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required to generate launch scripts.")
    audit = root / "audit.json"
    common = (
        "set -euo pipefail\n"
        f"REPO={shlex.quote(str(REPO_ROOT))}\n"
        f"AUDIT_ROOT={shlex.quote(str(root.resolve()))}\n"
        f"AUDIT_MANIFEST={shlex.quote(str(audit.resolve()))}\n"
        f"AUDIT_SHA256=$(sha256sum \"$AUDIT_MANIFEST\" | awk '{{print $1}}')\n"
        'export CCHAMBER_RANK_PANEL=$(python -c \'import json,sys; print(json.load(open(sys.argv[1]))["panel"])\' "$AUDIT_MANIFEST")\n'
        'export CCHAMBER_RANK_PANEL_SHA256=$(python -c \'import json,sys; print(json.load(open(sys.argv[1]))["panel_sha256"])\' "$AUDIT_MANIFEST")\n'
        'export CCHAMBER_RANK_CONTRACT=$(python -c \'import json,sys; print(json.load(open(sys.argv[1]))["execution_contract"])\' "$AUDIT_MANIFEST")\n'
        'export CCHAMBER_RANK_CONTRACT_SHA256=$(python -c \'import json,sys; print(json.load(open(sys.argv[1]))["execution_contract_sha256"])\' "$AUDIT_MANIFEST")\n'
        'export CCHAMBER_RANK_CAMPAIGN_COMMIT=$(python -c \'import json,sys; print(json.load(open(sys.argv[1]))["campaign_training_commit"])\' "$AUDIT_MANIFEST")\n'
        'cd "$REPO"\n'
        f"UV=({shlex.quote(uv)} run --frozen --no-sync python)\n"
    )

    def packed(stage: str) -> str:
        freeze = (
            'CHECKPOINT_SHA256=$(sha256sum "$AUDIT_ROOT/checkpoint_manifest.json" '
            "| awk '{print $1}')\n"
            if stage == "evaluate"
            else ""
        )
        freeze_arg = (
            '--checkpoint-manifest-sha256 "$CHECKPOINT_SHA256" ' if stage == "evaluate" else ""
        )
        return (
            "#!/usr/bin/env bash\n"
            "#SBATCH --account=a0166\n#SBATCH --partition=normal\n"
            "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=4\n"
            "#SBATCH --cpus-per-task=72\n"
            "#SBATCH --gpus-per-node=4\n#SBATCH --mem=440G\n"
            "#SBATCH --time=04:00:00\n"
            "#SBATCH --array=0-47%16\n"
            f"#SBATCH --job-name=cch-rank-{stage}\n"
            + common
            + freeze
            + "pids=()\n"
            + "for slot in 0 1 2 3; do\n"
            + "  index=$((SLURM_ARRAY_TASK_ID * 4 + slot))\n"
            + "  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 "
            + "--gpus-per-node=1 --mem=110G "
            + '"${UV[@]}" scripts/cchamber_candidate_rank_audit.py '
            + f'run-{stage} --root "$AUDIT_ROOT" --audit-sha256 "$AUDIT_SHA256" '
            + freeze_arg
            + '--trajectory-index "$index" &\n'
            + '  pids+=("$!")\n'
            + "done\n"
            + 'status=0\nfor pid in "${pids[@]}"; do wait "$pid" || status=1; done\n'
            + 'exit "$status"\n'
        )

    scripts = root / "slurm"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "train_packed.sh").write_text(packed("train"), encoding="utf-8")
    (scripts / "evaluate_packed.sh").write_text(packed("evaluate"), encoding="utf-8")
    canary = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --account=a0166\n#SBATCH --partition=debug\n"
        "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=72\n"
        "#SBATCH --gpus-per-node=1\n#SBATCH --mem=110G\n"
        "#SBATCH --time=00:30:00\n#SBATCH --job-name=cch-rank-canary\n"
        + common
        + "srun --nodes=1 --ntasks=1 --cpus-per-task=72 "
        + '--gpus-per-node=1 --mem=110G "${UV[@]}" '
        + "scripts/cchamber_candidate_rank_audit.py run-canary "
        + '--root "$AUDIT_ROOT" --audit-sha256 "$AUDIT_SHA256" --trajectory-index 0\n'
    )
    (scripts / "debug_fingerprint_canary.sh").write_text(canary, encoding="utf-8")

    timing_canary = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --account=a0166\n#SBATCH --partition=normal\n"
        "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=72\n"
        "#SBATCH --gpus-per-node=1\n#SBATCH --mem=110G\n"
        "#SBATCH --time=04:00:00\n#SBATCH --job-name=cch-rank-timing\n"
        + common
        + "srun --nodes=1 --ntasks=1 --cpus-per-task=72 "
        + '--gpus-per-node=1 --mem=110G "${UV[@]}" '
        + "scripts/cchamber_candidate_rank_audit.py run-train "
        + '--root "$AUDIT_ROOT" --audit-sha256 "$AUDIT_SHA256" --trajectory-index 0\n'
    )
    (scripts / "production_timing_canary.sh").write_text(timing_canary, encoding="utf-8")

    freeze = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --account=a0166\n#SBATCH --partition=normal\n"
        "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=4\n"
        "#SBATCH --mem=32G\n#SBATCH --time=02:00:00\n"
        "#SBATCH --job-name=cch-rank-freeze\n"
        + common
        + 'srun --nodes=1 --ntasks=1 --cpus-per-task=4 "${UV[@]}" '
        + "scripts/cchamber_candidate_rank_audit.py freeze-checkpoints "
        + '--root "$AUDIT_ROOT" --audit-sha256 "$AUDIT_SHA256"\n'
    )
    (scripts / "freeze_checkpoints.sh").write_text(freeze, encoding="utf-8")

    checkpoint_manifest = (
        'CHECKPOINT_MANIFEST="$AUDIT_ROOT/checkpoint_manifest.json"\n'
        'test -f "$CHECKPOINT_MANIFEST"\n'
        "CHECKPOINT_SHA256=$(sha256sum \"$CHECKPOINT_MANIFEST\" | awk '{print $1}')\n"
    )
    collect = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --account=a0166\n#SBATCH --partition=normal\n"
        "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=4\n"
        "#SBATCH --mem=64G\n#SBATCH --time=02:00:00\n"
        "#SBATCH --job-name=cch-rank-collect\n"
        + common
        + checkpoint_manifest
        + 'srun --nodes=1 --ntasks=1 --cpus-per-task=4 "${UV[@]}" '
        + "scripts/cchamber_candidate_rank_audit.py collect "
        + '--root "$AUDIT_ROOT" --audit-sha256 "$AUDIT_SHA256" '
        + '--checkpoint-manifest-sha256 "$CHECKPOINT_SHA256"\n'
    )
    (scripts / "collect.sh").write_text(collect, encoding="utf-8")

    analyze = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --account=a0166\n#SBATCH --partition=normal\n"
        "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=72\n"
        "#SBATCH --mem=110G\n#SBATCH --time=04:00:00\n"
        "#SBATCH --job-name=cch-rank-analyze\n"
        + common
        + checkpoint_manifest
        + 'srun --nodes=1 --ntasks=1 --cpus-per-task=72 "${UV[@]}" '
        + "scripts/cchamber_candidate_rank_audit.py analyze "
        + '--root "$AUDIT_ROOT" --audit-sha256 "$AUDIT_SHA256" '
        + '--checkpoint-manifest-sha256 "$CHECKPOINT_SHA256" '
        + "--n-permutations 10000 --n-bootstrap 10000\n"
    )
    (scripts / "analyze.sh").write_text(analyze, encoding="utf-8")

    workflow = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)\n'
        'canary_job=$(sbatch --parsable "$SCRIPT_DIR/debug_fingerprint_canary.sh")\n'
        'timing_job=$(sbatch --parsable --dependency="afterok:${canary_job}" '
        '"$SCRIPT_DIR/production_timing_canary.sh")\n'
        'training_job=$(sbatch --parsable --dependency="afterok:${timing_job}" '
        '"$SCRIPT_DIR/train_packed.sh")\n'
        'freeze_job=$(sbatch --parsable --dependency="afterok:${training_job}" '
        '"$SCRIPT_DIR/freeze_checkpoints.sh")\n'
        'evaluation_job=$(sbatch --parsable --dependency="afterok:${freeze_job}" '
        '"$SCRIPT_DIR/evaluate_packed.sh")\n'
        'collect_job=$(sbatch --parsable --dependency="afterok:${evaluation_job}" '
        '"$SCRIPT_DIR/collect.sh")\n'
        'analysis_job=$(sbatch --parsable --dependency="afterok:${collect_job}" '
        '"$SCRIPT_DIR/analyze.sh")\n'
        'printf "canary=%s\\ntiming=%s\\ntraining=%s\\nfreeze=%s\\n" '
        '"$canary_job" "$timing_job" "$training_job" "$freeze_job"\n'
        'printf "evaluation=%s\\ncollect=%s\\nanalysis=%s\\n" '
        '"$evaluation_job" "$collect_job" "$analysis_job"\n'
    )
    (scripts / "submit_workflow.sh").write_text(workflow, encoding="utf-8")
    for path in scripts.glob("*.sh"):
        path.chmod(0o750)


def _load_audit(root: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load an immutable audit design and revalidate every label-free source hash."""
    root = root.resolve()
    audit_path = root / "audit.json"
    _require_hash(audit_path, expected_sha256, "audit design")
    audit = _load_json(audit_path)
    _set_design_identity(
        Path(audit["panel"]),
        str(audit["panel_sha256"]),
        Path(audit["execution_contract"]),
        str(audit["execution_contract_sha256"]),
        str(audit["campaign_training_commit"]),
    )
    if (
        audit.get("panel_sha256") != PANEL_SHA256
        or audit.get("execution_contract_sha256") != CONTRACT_SHA256
        or audit.get("campaign_training_commit") != CAMPAIGN_COMMIT
        or int(audit.get("expected_trajectories", -1)) != EXPECTED_TRAJECTORIES
    ):
        raise ValueError("Audit design identity changed.")
    _validate_frozen_design()
    campaign = _validate_campaign(
        Path(audit["campaign_manifest"]),
        str(audit["campaign_manifest_sha256"]),
        full_data_tree=False,
    )
    pairing, primary_encoder = _validate_pairing_provenance(
        Path(audit["pairing_manifest"]),
        str(audit["pairing_manifest_sha256"]),
        Path(audit["encoder_validation_pair_table"]),
        str(audit["encoder_validation_pair_table_sha256"]),
        campaign,
    )
    _validate_candidate_metrics_provenance(
        Path(audit["candidate_metrics_provenance"]),
        str(audit["candidate_metrics_provenance_sha256"]),
        Path(audit["candidate_metrics"]),
        str(audit["candidate_metrics_sha256"]),
        Path(audit["campaign_manifest"]),
        str(audit["campaign_manifest_sha256"]),
        campaign,
        str(pairing["primary_validation_table_sha256"]),
    )
    if (
        int(audit.get("primary_pair_encoder_seed", -1)) != 123
        or Path(audit["primary_pair_encoder_checkpoint"]).resolve()
        != Path(primary_encoder["encoder_checkpoint"]).resolve()
        or audit.get("primary_pair_encoder_checkpoint_sha256")
        != primary_encoder["encoder_checkpoint_sha256"]
    ):
        raise ValueError("Frozen primary pairing-encoder identity changed.")
    trajectory_path = Path(audit["trajectory_manifest"])
    _require_hash(trajectory_path, str(audit["trajectory_manifest_sha256"]), "trajectory manifest")
    trajectories = _load_json(trajectory_path)
    if (
        not isinstance(trajectories, list)
        or len(trajectories) != EXPECTED_TRAJECTORIES
        or [int(row["trajectory_index"]) for row in trajectories]
        != list(range(EXPECTED_TRAJECTORIES))
    ):
        raise ValueError("Trajectory manifest coverage changed.")
    commit, _ = _audit_revision()
    if commit != audit["audit_code_commit"]:
        raise RuntimeError(
            "Current code commit does not match the frozen sidecar audit-code commit."
        )
    return dict(audit), [dict(row) for row in trajectories]


def _require_slurm_gpu() -> None:
    """Refuse training/evaluation work outside a GPU Slurm allocation."""
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("GPU audit execution must run inside Slurm.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU audit execution requires an available CUDA device.")


def _find_mlflow_run(client: MlflowClient, experiment_name: str, run_name: str):
    """Find the newest finished retry with one exact MLflow run name."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment is missing: {experiment_name}")
    escaped = run_name.replace("'", "\\'")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{escaped}'",
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    finished = [run for run in runs if run.info.status == "FINISHED"]
    if not finished:
        raise RuntimeError(f"No finished MLflow run found for {run_name}.")
    return finished[0]


def _training_command(
    audit: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    attempt_dir: Path,
    run_name: str,
    *,
    epochs: int,
    control: bool,
) -> tuple[list[str], dict[str, str]]:
    """Build one sealed shared-trajectory training command and environment."""
    model = str(trajectory["model"])
    tags = {
        "campaign_id": audit["campaign_id"],
        "stage": "candidate_rank_canary" if epochs < 200 else "candidate_rank_train",
        "trajectory_index": int(trajectory["trajectory_index"]),
        "model": model,
        "candidate_id": trajectory["candidate_id"],
        "reporting_seed": int(trajectory["reporting_seed"]),
        "audit_code_commit": audit["audit_code_commit"],
        "campaign_training_commit": CAMPAIGN_COMMIT,
        "control": str(bool(control)).lower(),
    }
    tag_override = "{" + ",".join(f"{key}:{value}" for key, value in tags.items()) + "}"
    command = [
        sys.executable,
        "src/train.py",
        f"experiment=cchamber/{model}_candidate_rank_audit",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.deterministic=true",
        f"trainer.min_epochs={epochs}",
        f"trainer.max_epochs={epochs}",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        "data.signal_experiments=[]",
        "train=true",
        "test=false",
        f"experiment_name={audit['campaign_id']}_candidate_rank_{model}",
        f"run_name={run_name}",
        f"paths.base_data_dir={audit['data_dir']}",
        f"paths.log_dir={Path(audit['tracking_uri'].removeprefix('file:')).parents[1]}",
        f"paths.checkpoints_dir={attempt_dir / 'checkpoints'}",
        f"hydra.run.dir={attempt_dir / 'hydra'}",
        f"logger.mlflow.tags={tag_override}",
        "extras.print_config=false",
        *[f"{name}={_hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    if model == "svdd":
        command.extend(
            [
                f"algorithm.pretrained_encoder_ckpt={audit['primary_pair_encoder_checkpoint']}",
                "algorithm.pretrained_encoder_strict=true",
                "algorithm.enforce_architecture_constraints=true",
            ]
        )
    if epochs < 200:
        command.extend(
            [
                "+trainer.limit_train_batches=2",
                "trainer.num_sanity_val_steps=0",
            ]
        )
    if control:
        command.extend(
            [
                "callbacks.audit_ckpt_cap_metadata_nearest=null",
                "callbacks.audit_ckpt_cap_encoder_nearest=null",
                "callbacks.audit_ckpt_cap_cdf=null",
                "callbacks.audit_ckpt_cap_random=null",
                "callbacks.audit_ckpt_drift=null",
                "callbacks.audit_ckpt_wasserstein=null",
                "callbacks.audit_checkpoint_manifest=null",
            ]
        )
    environment = os.environ.copy()
    environment["CCHAMBER_VALID_PAIR_TABLE"] = str(audit["encoder_validation_pair_table"])
    environment["CCHAMBER_AUDIT_CHECKPOINT_MANIFEST"] = str(
        attempt_dir / "checkpoint_branches.json"
    )
    environment["CCHAMBER_AUDIT_TRAJECTORY_FINGERPRINT"] = str(
        attempt_dir / "trajectory_fingerprint.json"
    )
    environment["LOG_DIR"] = str(Path(audit["tracking_uri"].removeprefix("file:")).parents[1])
    return command, environment


def _validate_branch_manifest(
    path: Path,
    *,
    expected_epochs: int,
    client: MlflowClient | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Validate all branch checkpoints and the earliest-equal tie rule."""
    manifest = _load_json(path)
    branches = manifest.get("branches")
    if not isinstance(branches, list) or len(branches) != len(STRATEGIES):
        raise ValueError("Checkpoint manifest does not cover every strategy exactly once.")
    by_strategy = {str(row["strategy"]): dict(row) for row in branches}
    if set(by_strategy) != set(STRATEGIES):
        raise ValueError("Checkpoint manifest strategy coverage is not exact.")
    output = []
    for strategy in STRATEGIES:
        row = by_strategy[strategy]
        checkpoint = Path(row["checkpoint"])
        if row["monitor"] != MONITORS[strategy]:
            raise ValueError(f"Checkpoint monitor changed for {strategy}.")
        _require_hash(checkpoint, str(row["checkpoint_sha256"]), "branch checkpoint")
        epoch = int(row["selected_epoch"])
        score = float(row["monitor_value"])
        if not 0 <= epoch < expected_epochs or not math.isfinite(score):
            raise ValueError(f"Invalid selected branch state for {strategy}.")
        if client is not None and run_id is not None:
            history = sorted(
                client.get_metric_history(run_id, MONITORS[strategy]),
                key=lambda item: (item.timestamp, item.step),
            )[-expected_epochs:]
            values = [float(item.value) for item in history]
            if len(values) != expected_epochs or not np.isfinite(values).all():
                raise ValueError(f"Incomplete monitor trajectory for {strategy}.")
            best = max(values) if DIRECTIONS[strategy] == "maximize" else min(values)
            earliest = next(index for index, value in enumerate(values) if value == best)
            if score != best or epoch != earliest:
                raise ValueError(f"Earliest-equal checkpoint rule failed for {strategy}.")
        output.append(row)
    return output


def _resume_training_marker(
    marker_path: Path,
    audit: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return one completed training marker."""
    marker = _load_json(marker_path)
    expected = {
        "campaign_id": audit["campaign_id"],
        "audit_code_commit": audit["audit_code_commit"],
        "trajectory_index": int(trajectory["trajectory_index"]),
        "model": trajectory["model"],
        "candidate_id": trajectory["candidate_id"],
        "reporting_seed": int(trajectory["reporting_seed"]),
        "params_sha256": trajectory["params_sha256"],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"Training resume marker mismatch for {key!r}.")
    branch_manifest = Path(marker["branch_manifest"])
    fingerprint = Path(marker["trajectory_fingerprint"])
    _require_hash(branch_manifest, marker["branch_manifest_sha256"], "branch manifest")
    _require_hash(fingerprint, marker["trajectory_fingerprint_sha256"], "fingerprint")
    validated_branches = _validate_branch_manifest(branch_manifest, expected_epochs=200)
    if marker.get("branches") != validated_branches:
        raise ValueError("Training marker branches differ from the validated branch manifest.")
    return dict(marker)


def run_train(root: Path, audit_sha256: str, trajectory_index: int) -> dict[str, Any]:
    """Train or safely resume one exact shared trajectory with five branches."""
    _require_slurm_gpu()
    audit, trajectories = _load_audit(root, audit_sha256)
    _validate_canary(root.resolve(), audit_sha256, audit)
    if not 0 <= int(trajectory_index) < EXPECTED_TRAJECTORIES:
        raise IndexError("trajectory-index must be in [0, 191].")
    trajectory = trajectories[int(trajectory_index)]
    marker_path = root.resolve() / "training" / f"{int(trajectory_index):03d}.json"
    if marker_path.is_file():
        return _resume_training_marker(marker_path, audit, trajectory)
    attempt = _attempt_id()
    attempt_dir = root.resolve() / "attempts" / "training" / f"{trajectory_index:03d}" / attempt
    run_name = f"rank_train_t{trajectory_index:03d}_{attempt}"
    command, environment = _training_command(
        audit, trajectory, attempt_dir, run_name, epochs=200, control=False
    )
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)  # nosec B603
    branch_manifest = attempt_dir / "checkpoint_branches.json"
    fingerprint = attempt_dir / "trajectory_fingerprint.json"
    client = MlflowClient(tracking_uri=str(audit["tracking_uri"]))
    experiment_name = f"{audit['campaign_id']}_candidate_rank_{trajectory['model']}"
    run = _find_mlflow_run(client, experiment_name, run_name)
    branches = _validate_branch_manifest(
        branch_manifest, expected_epochs=200, client=client, run_id=run.info.run_id
    )
    if not fingerprint.is_file():
        raise FileNotFoundError(fingerprint)
    marker = {
        "schema_version": 1,
        "campaign_id": audit["campaign_id"],
        "campaign_training_commit": CAMPAIGN_COMMIT,
        "audit_code_commit": audit["audit_code_commit"],
        **{key: trajectory[key] for key in trajectory if key != "params"},
        "attempt_id": attempt,
        "branch_manifest": str(branch_manifest.resolve()),
        "branch_manifest_sha256": _sha256(branch_manifest),
        "trajectory_fingerprint": str(fingerprint.resolve()),
        "trajectory_fingerprint_sha256": _sha256(fingerprint),
        "branches": branches,
        "mlflow_tracking_uri": audit["tracking_uri"],
        "mlflow_experiment": experiment_name,
        "mlflow_run_id": run.info.run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    try:
        _atomic_json(marker_path, marker, create=True)
    except FileExistsError:
        return _resume_training_marker(marker_path, audit, trajectory)
    client.log_artifact(run.info.run_id, str(marker_path), artifact_path="rank_audit")
    client.set_tag(run.info.run_id, "training_marker_sha256", _sha256(marker_path))
    return marker


def freeze_checkpoints(root: Path, audit_sha256: str) -> tuple[Path, str]:
    """Freeze all branch checkpoint hashes before intervention evaluation."""
    audit, trajectories = _load_audit(root, audit_sha256)
    _validate_canary(root.resolve(), audit_sha256, audit)
    records = []
    source_markers = []
    for index, trajectory in enumerate(trajectories):
        marker_path = root.resolve() / "training" / f"{index:03d}.json"
        marker = _resume_training_marker(marker_path, audit, trajectory)
        source_markers.append(
            {
                "trajectory_index": index,
                "training_marker": str(marker_path.resolve()),
                "training_marker_sha256": _sha256(marker_path),
            }
        )
        for branch in marker["branches"]:
            records.append(
                {
                    "trajectory_index": index,
                    "model": trajectory["model"],
                    "candidate_id": trajectory["candidate_id"],
                    "reporting_seed": trajectory["reporting_seed"],
                    "params_sha256": trajectory["params_sha256"],
                    **branch,
                }
            )
    if len(records) != EXPECTED_CHECKPOINTS:
        raise ValueError(f"Expected {EXPECTED_CHECKPOINTS} frozen checkpoints.")
    manifest = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "outcomes_inspected_before_freeze": False,
        "audit": str(root.resolve() / "audit.json"),
        "audit_sha256": audit_sha256,
        "audit_code_commit": audit["audit_code_commit"],
        "campaign_training_commit": CAMPAIGN_COMMIT,
        "expected_trajectories": EXPECTED_TRAJECTORIES,
        "expected_checkpoints": EXPECTED_CHECKPOINTS,
        "training_markers": source_markers,
        "checkpoints": records,
    }
    output = root.resolve() / "checkpoint_manifest.json"
    if output.exists():
        existing = _load_json(output)
        comparable_existing = dict(existing)
        comparable_existing.pop("frozen_at", None)
        comparable_new = dict(manifest)
        comparable_new.pop("frozen_at", None)
        if comparable_existing != comparable_new:
            raise FileExistsError("Refusing to replace a different checkpoint freeze.")
    else:
        _atomic_json(output, manifest, create=True)
    return output, _sha256(output)


def _validate_canary(root: Path, audit_sha256: str, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Require the checkpoint/no-checkpoint trajectory-equivalence gate."""
    path = root / "canary" / "trajectory_fingerprint_canary.json"
    marker = _load_json(path)
    if (
        marker.get("audit_sha256") != audit_sha256
        or marker.get("audit_code_commit") != audit["audit_code_commit"]
        or marker.get("equivalent") is not True
    ):
        raise ValueError("Trajectory fingerprint canary gate is invalid.")
    for key in ("control_fingerprint", "checkpoint_fingerprint"):
        artifact = Path(marker[key])
        _require_hash(artifact, marker[f"{key}_sha256"], key)
    if _load_json(Path(marker["control_fingerprint"])) != _load_json(
        Path(marker["checkpoint_fingerprint"])
    ):
        raise ValueError("Checkpoint callbacks changed the shared trajectory fingerprint.")
    return dict(marker)


def run_canary(root: Path, audit_sha256: str, trajectory_index: int = 0) -> dict[str, Any]:
    """Compare a short checkpoint-enabled trajectory with a no-checkpoint control."""
    _require_slurm_gpu()
    audit, trajectories = _load_audit(root, audit_sha256)
    if not 0 <= int(trajectory_index) < EXPECTED_TRAJECTORIES:
        raise IndexError("trajectory-index must be in [0, 191].")
    trajectory = trajectories[int(trajectory_index)]
    output = root.resolve() / "canary" / "trajectory_fingerprint_canary.json"
    if output.is_file():
        return _validate_canary(root.resolve(), audit_sha256, audit)
    attempt = _attempt_id()
    fingerprints = {}
    run_ids = {}
    for mode, control in (("control", True), ("checkpoint", False)):
        attempt_dir = root.resolve() / "attempts" / "canary" / attempt / mode
        run_name = f"rank_canary_t{trajectory_index:03d}_{mode}_{attempt}"
        command, environment = _training_command(
            audit, trajectory, attempt_dir, run_name, epochs=2, control=control
        )
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)  # nosec B603
        fingerprint = attempt_dir / "trajectory_fingerprint.json"
        if not fingerprint.is_file():
            raise FileNotFoundError(fingerprint)
        if not control:
            _validate_branch_manifest(attempt_dir / "checkpoint_branches.json", expected_epochs=2)
        fingerprints[mode] = fingerprint
        client = MlflowClient(tracking_uri=str(audit["tracking_uri"]))
        experiment_name = f"{audit['campaign_id']}_candidate_rank_{trajectory['model']}"
        run_ids[mode] = _find_mlflow_run(client, experiment_name, run_name).info.run_id
    control_value = _load_json(fingerprints["control"])
    checkpoint_value = _load_json(fingerprints["checkpoint"])
    if control_value != checkpoint_value:
        raise RuntimeError(
            "Checkpoint callbacks changed model state, metric trajectory, RNG, "
            "initialization, or minibatch order."
        )
    marker = {
        "schema_version": 1,
        "audit_sha256": audit_sha256,
        "audit_code_commit": audit["audit_code_commit"],
        "trajectory_index": int(trajectory_index),
        "model": trajectory["model"],
        "candidate_id": trajectory["candidate_id"],
        "reporting_seed": trajectory["reporting_seed"],
        "epochs": 2,
        "limit_train_batches": 2,
        "validation_batches": "full_pair_table_contract",
        "control_fingerprint": str(fingerprints["control"].resolve()),
        "control_fingerprint_sha256": _sha256(fingerprints["control"]),
        "checkpoint_fingerprint": str(fingerprints["checkpoint"].resolve()),
        "checkpoint_fingerprint_sha256": _sha256(fingerprints["checkpoint"]),
        "control_mlflow_run_id": run_ids["control"],
        "checkpoint_mlflow_run_id": run_ids["checkpoint"],
        "equivalent": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, marker, create=True)
    client = MlflowClient(tracking_uri=str(audit["tracking_uri"]))
    client.log_artifact(run_ids["checkpoint"], str(output), artifact_path="rank_audit_canary")
    client.set_tag(
        run_ids["checkpoint"],
        "trajectory_fingerprint_canary_sha256",
        _sha256(output),
    )
    return marker


class _SealedMetricsCallback(pl.Callback):
    """Emit AUPRC and 1%-operating-point efficiency without plots."""

    def __init__(
        self,
        interventions: Sequence[str],
        output_path: Path,
        context: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.interventions = tuple(map(str, interventions))
        self.output_path = output_path
        self.context = dict(context)

    def on_test_start(self, trainer, pl_module) -> None:
        """Validate exact loader and threshold contracts."""
        expected = {"normal", *self.interventions}
        if set(trainer.test_dataloaders) != expected:
            raise ValueError("Sealed evaluation loader coverage is not exact.")
        threshold = getattr(pl_module, "thres_operational", None)
        if threshold is None or not torch.isfinite(threshold):
            raise ValueError("Branch checkpoint lacks a finite validation threshold.")
        self.threshold = float(threshold.detach().cpu().item())

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Reset score buffers."""
        del trainer, pl_module
        self.scores: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("normal", *self.interventions)
        }

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        """Collect finite per-event anomaly scores for one named stream."""
        del pl_module, batch, batch_idx
        name = list(trainer.test_dataloaders)[dataloader_idx]
        score = outputs["ascore/full"].detach().view(-1).cpu()
        if not torch.isfinite(score).all():
            raise ValueError(f"Non-finite sealed evaluation scores for {name}.")
        self.scores[name].append(score)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Write exactly 58 interventions by two metrics."""
        del trainer, pl_module
        normal = torch.cat(self.scores["normal"]).float()
        rows = []
        for intervention in self.interventions:
            signal = torch.cat(self.scores[intervention]).float()
            prediction = torch.cat([normal, signal])
            target = torch.cat(
                [
                    torch.zeros(normal.numel(), dtype=torch.long),
                    torch.ones(signal.numel(), dtype=torch.long),
                ]
            )
            # TorchMetrics can overshoot a probability metric by one float32 ULP
            # (for example, 1.000000119).  Persist the mathematical [0, 1]
            # quantity rather than rejecting an otherwise valid sealed result.
            auprc = min(1.0, max(0.0, float(BinaryAveragePrecision()(prediction, target).item())))
            efficiency = min(
                1.0,
                max(0.0, float((signal >= self.threshold).float().mean().item())),
            )
            for metric, value in (
                ("auprc", auprc),
                ("efficiency_operational", efficiency),
            ):
                rows.append(
                    {
                        **self.context,
                        "intervention": intervention,
                        "metric": metric,
                        "value": value,
                        "validation_derived_threshold": self.threshold,
                    }
                )
        if len(rows) != EXPECTED_INTERVENTIONS * len(METRICS):
            raise AssertionError("Sealed callback row contract failed.")
        _write_csv(self.output_path, rows)


def _load_checkpoint_state(model, checkpoint: Path) -> None:
    """Strictly restore one branch, including dynamically registered buffers."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no Lightning state_dict: {checkpoint}")
    for name, value in state.items():
        if (
            "." not in name
            and name.startswith("thres_")
            and name not in dict(model.named_buffers())
        ):
            model.register_buffer(name, value.detach().clone(), persistent=True)
    if "center" in state and getattr(model, "center", "absent") is None:
        model.center = torch.zeros_like(state["center"])
    model.load_state_dict(state, strict=True)
    model._ckpt_path = checkpoint


def _compose_evaluation(audit: Mapping[str, Any], trajectory: Mapping[str, Any]):
    """Compose the exact model and all 58 sealed intervention loaders."""
    os.environ["CCHAMBER_VALID_PAIR_TABLE"] = str(audit["encoder_validation_pair_table"])
    interventions = list(audit["interventions"])
    overrides = [
        f"experiment=cchamber/{trajectory['model']}_candidate_rank_audit",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        f"paths.base_data_dir={audit['data_dir']}",
        f"data.signal_experiments={_hydra_value(interventions)}",
        "logger=none",
        *[f"{name}={_hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("test")
    all_loaders = datamodule.test_dataloader()
    loaders = {"normal": all_loaders["normal"]}
    loaders.update({name: all_loaders[name] for name in interventions})
    loaders["normal"].loader = datamodule.loader
    return cfg, datamodule, loaders


def _load_checkpoint_freeze(
    root: Path,
    audit_sha256: str,
    checkpoint_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    """Validate the pre-outcome 960-checkpoint freeze gate."""
    path = root.resolve() / "checkpoint_manifest.json"
    _require_hash(path, checkpoint_manifest_sha256, "checkpoint freeze")
    manifest = _load_json(path)
    if (
        manifest.get("audit_sha256") != audit_sha256
        or manifest.get("audit_code_commit")
        != _load_json(root.resolve() / "audit.json").get("audit_code_commit")
        or manifest.get("outcomes_inspected_before_freeze") is not False
        or int(manifest.get("expected_checkpoints", -1)) != EXPECTED_CHECKPOINTS
    ):
        raise ValueError("Checkpoint freeze gate identity changed.")
    records = manifest.get("checkpoints")
    if not isinstance(records, list) or len(records) != EXPECTED_CHECKPOINTS:
        raise ValueError(f"Checkpoint freeze must contain exactly {EXPECTED_CHECKPOINTS} records.")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        checkpoint = Path(record["checkpoint"])
        _require_hash(checkpoint, record["checkpoint_sha256"], "frozen checkpoint")
        grouped.setdefault(int(record["trajectory_index"]), []).append(dict(record))
    if set(grouped) != set(range(EXPECTED_TRAJECTORIES)) or any(
        {row["strategy"] for row in values} != set(STRATEGIES) for values in grouped.values()
    ):
        raise ValueError("Checkpoint freeze trajectory/branch coverage is not exact.")
    return dict(manifest), grouped


def _validate_evaluation_rows(
    path: Path,
    trajectory: Mapping[str, Any],
    interventions: Sequence[str],
    frozen_branches: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Validate one complete trajectory evaluation table."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(STRATEGIES) * len(interventions) * len(METRICS):
        raise ValueError(f"Evaluation row count is not exact: {path}")
    expected = set(product(STRATEGIES, interventions, METRICS))
    actual = {(row["strategy"], row["intervention"], row["metric"]) for row in rows}
    if actual != expected:
        raise ValueError("Evaluation branch/intervention/metric coverage is not exact.")
    if set(frozen_branches) != set(STRATEGIES):
        raise ValueError("Frozen branch identities are not exact.")
    for row in rows:
        branch = frozen_branches[row["strategy"]]
        if (
            int(row["trajectory_index"]) != int(trajectory["trajectory_index"])
            or row["model"] != trajectory["model"]
            or row["candidate_id"] != trajectory["candidate_id"]
            or int(row["reporting_seed"]) != int(trajectory["reporting_seed"])
            or Path(row["checkpoint"]).resolve() != Path(branch["checkpoint"]).resolve()
            or row["checkpoint_sha256"] != branch["checkpoint_sha256"]
            or row["monitor"] != branch["monitor"]
            or int(row["selected_epoch"]) != int(branch["selected_epoch"])
            or float(row["monitor_value"]) != float(branch["monitor_value"])
        ):
            raise ValueError("Evaluation trajectory/frozen-checkpoint identity mismatch.")
        value = float(row["value"])
        threshold = float(row["validation_derived_threshold"])
        if not 0.0 <= value <= 1.0 or not math.isfinite(threshold):
            raise ValueError("Evaluation metrics/thresholds must be finite and bounded.")
    return rows


def run_evaluate(
    root: Path,
    audit_sha256: str,
    checkpoint_manifest_sha256: str,
    trajectory_index: int,
) -> dict[str, Any]:
    """Evaluate all five frozen branches for one trajectory on 58 interventions."""
    _require_slurm_gpu()
    audit, trajectories = _load_audit(root, audit_sha256)
    _validate_campaign(
        Path(audit["campaign_manifest"]),
        str(audit["campaign_manifest_sha256"]),
        full_data_tree=True,
    )
    if not 0 <= int(trajectory_index) < EXPECTED_TRAJECTORIES:
        raise IndexError("trajectory-index must be in [0, 191].")
    trajectory = trajectories[int(trajectory_index)]
    _, grouped = _load_checkpoint_freeze(root, audit_sha256, checkpoint_manifest_sha256)
    branches = {row["strategy"]: row for row in grouped[int(trajectory_index)]}
    marker_path = root.resolve() / "evaluation" / f"{trajectory_index:03d}.json"
    if marker_path.is_file():
        marker = _load_json(marker_path)
        if (
            marker.get("audit_sha256") != audit_sha256
            or marker.get("checkpoint_manifest_sha256") != checkpoint_manifest_sha256
            or int(marker.get("trajectory_index", -1)) != int(trajectory_index)
            or marker.get("model") != trajectory["model"]
            or marker.get("candidate_id") != trajectory["candidate_id"]
            or int(marker.get("reporting_seed", -1)) != int(trajectory["reporting_seed"])
            or marker.get("params_sha256") != trajectory["params_sha256"]
        ):
            raise ValueError("Evaluation resume marker identity mismatch.")
        output = Path(marker["evaluation_rows"])
        _require_hash(output, marker["evaluation_rows_sha256"], "evaluation rows")
        _validate_evaluation_rows(output, trajectory, audit["interventions"], branches)
        return dict(marker)
    attempt = _attempt_id()
    attempt_dir = root.resolve() / "attempts" / "evaluation" / f"{trajectory_index:03d}" / attempt
    cfg, datamodule, loaders = _compose_evaluation(audit, trajectory)
    branch_rows = []
    tracking_uri = str(audit["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"{audit['campaign_id']}_candidate_rank_evaluate"
    mlflow.set_experiment(experiment_name)
    tags = {
        "campaign_id": audit["campaign_id"],
        "stage": "candidate_rank_evaluate",
        "trajectory_index": str(trajectory_index),
        "model": trajectory["model"],
        "candidate_id": trajectory["candidate_id"],
        "reporting_seed": str(trajectory["reporting_seed"]),
        "audit_sha256": audit_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
    }
    with mlflow.start_run(
        run_name=f"rank_evaluate_t{trajectory_index:03d}_{attempt}", tags=tags
    ) as run:
        for strategy in STRATEGIES:
            branch = branches[strategy]
            checkpoint = Path(branch["checkpoint"])
            pl.seed_everything(int(trajectory["reporting_seed"]), workers=True)
            model = hydra.utils.instantiate(cfg.algorithm)
            _load_checkpoint_state(model, checkpoint)
            branch_output = attempt_dir / f"{strategy}.csv"
            callback = _SealedMetricsCallback(
                audit["interventions"],
                branch_output,
                {
                    "trajectory_index": int(trajectory_index),
                    "model": trajectory["model"],
                    "candidate_id": trajectory["candidate_id"],
                    "reporting_seed": int(trajectory["reporting_seed"]),
                    "strategy": strategy,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": branch["checkpoint_sha256"],
                    "selected_epoch": int(branch["selected_epoch"]),
                    "monitor": branch["monitor"],
                    "monitor_value": float(branch["monitor_value"]),
                },
            )
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                logger=False,
                callbacks=[callback],
                enable_checkpointing=False,
                enable_progress_bar=False,
                deterministic=True,
                inference_mode=True,
            )
            trainer.split = "test"
            trainer.test(model=model, dataloaders=loaders, verbose=False)
            with branch_output.open("r", encoding="utf-8", newline="") as handle:
                branch_rows.extend(csv.DictReader(handle))
        output = attempt_dir / "evaluation_rows.csv"
        _write_csv(output, branch_rows)
        _validate_evaluation_rows(output, trajectory, audit["interventions"], branches)
        mlflow.log_artifact(str(output), artifact_path="rank_audit")
        run_id = run.info.run_id
    datamodule.teardown("test")
    marker = {
        "schema_version": 1,
        "audit_sha256": audit_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        **{key: trajectory[key] for key in trajectory if key != "params"},
        "attempt_id": attempt,
        "evaluation_rows": str(output.resolve()),
        "evaluation_rows_sha256": _sha256(output),
        "n_checkpoint_evaluations": len(STRATEGIES),
        "n_rows": len(branch_rows),
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_experiment": experiment_name,
        "mlflow_run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    try:
        _atomic_json(marker_path, marker, create=True)
    except FileExistsError:
        return run_evaluate(
            root,
            audit_sha256,
            checkpoint_manifest_sha256,
            trajectory_index,
        )
    client = MlflowClient(tracking_uri=tracking_uri)
    client.log_artifact(run_id, str(marker_path), artifact_path="rank_audit")
    client.set_tag(run_id, "evaluation_marker_sha256", _sha256(marker_path))
    return marker


def collect_rows(
    root: Path,
    audit_sha256: str,
    checkpoint_manifest_sha256: str,
) -> tuple[Path, Path]:
    """Validate and combine exactly 111,360 immutable evaluation rows."""
    audit, trajectories = _load_audit(root, audit_sha256)
    _, frozen_by_trajectory = _load_checkpoint_freeze(
        root, audit_sha256, checkpoint_manifest_sha256
    )
    rows = []
    sources = []
    for index, trajectory in enumerate(trajectories):
        marker_path = root.resolve() / "evaluation" / f"{index:03d}.json"
        marker = _load_json(marker_path)
        if (
            marker.get("audit_sha256") != audit_sha256
            or marker.get("checkpoint_manifest_sha256") != checkpoint_manifest_sha256
            or int(marker.get("trajectory_index", -1)) != index
        ):
            raise ValueError(f"Evaluation marker identity mismatch: {marker_path}")
        source = Path(marker["evaluation_rows"])
        _require_hash(source, marker["evaluation_rows_sha256"], "evaluation rows")
        frozen_branches = {row["strategy"]: row for row in frozen_by_trajectory[index]}
        rows.extend(
            _validate_evaluation_rows(
                source,
                trajectory,
                audit["interventions"],
                frozen_branches,
            )
        )
        sources.append(
            {
                "trajectory_index": index,
                "evaluation_marker": str(marker_path.resolve()),
                "evaluation_marker_sha256": _sha256(marker_path),
                "evaluation_rows": str(source.resolve()),
                "evaluation_rows_sha256": _sha256(source),
                "mlflow_run_id": marker["mlflow_run_id"],
            }
        )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}.")
    frame = pd.DataFrame(rows)
    group_sizes = frame.groupby(
        ["model", "candidate_id", "reporting_seed", "strategy", "metric"]
    ).size()
    if (
        len(group_sizes) != EXPECTED_TRAJECTORIES * len(STRATEGIES) * len(METRICS)
        or not (group_sizes == EXPECTED_INTERVENTIONS).all()
    ):
        raise ValueError("Combined sealed-result coverage is not exact.")
    output = root.resolve() / "results" / "sealed_candidate_outcomes.csv"
    _write_csv(output, rows)
    provenance_path = root.resolve() / "results" / "sealed_candidate_outcomes_provenance.json"
    provenance = {
        "schema_version": 1,
        "audit": str(root.resolve() / "audit.json"),
        "audit_sha256": audit_sha256,
        "checkpoint_manifest": str(root.resolve() / "checkpoint_manifest.json"),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "combined": str(output.resolve()),
        "combined_sha256": _sha256(output),
        "expected_rows": EXPECTED_ROWS,
        "sources": sources,
    }
    if provenance_path.exists() and _load_json(provenance_path) != provenance:
        raise FileExistsError("Refusing to replace different sealed-result provenance.")
    if not provenance_path.exists():
        _atomic_json(provenance_path, provenance, create=True)
    return output, provenance_path


def compute_search_utility(
    candidate_metrics: pd.DataFrame,
    development_seeds: Sequence[int],
) -> pd.DataFrame:
    """Compute frozen full-pool direction-aware utility before panel filtering."""
    required = {"model", "strategy", "candidate_id", "seed", "value"}
    if not required.issubset(candidate_metrics.columns):
        raise ValueError(
            f"Candidate utility source misses {sorted(required - set(candidate_metrics))}."
        )
    frame = candidate_metrics.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str).str.zfill(3)
    frame["seed"] = frame["seed"].astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if not np.isfinite(frame["value"]).all():
        raise ValueError("Candidate utility values must be finite.")
    supplied_seeds = tuple(map(int, development_seeds))
    if len(supplied_seeds) != len(DEVELOPMENT_SEEDS) or set(supplied_seeds) != set(
        DEVELOPMENT_SEEDS
    ):
        raise ValueError("Search utility requires the exact five frozen development seeds.")
    if set(frame["model"].astype(str)) != set(MODELS):
        raise ValueError("Search utility requires the exact frozen model identities.")
    if set(frame["strategy"].astype(str)) != set(STRATEGIES):
        raise ValueError("Search utility requires the exact frozen strategy identities.")
    expected_seeds = set(DEVELOPMENT_SEEDS)
    rows = []
    model_survivors: dict[str, set[str]] = {}
    for (model, strategy), group in frame.groupby(["model", "strategy"], sort=True):
        strategy = str(strategy)
        if set(group["seed"]) != expected_seeds:
            raise ValueError(f"Development-seed coverage changed for {model}/{strategy}.")
        duplicates = group.groupby(["seed", "candidate_id"]).size()
        if not (duplicates == 1).all():
            raise ValueError(f"Candidate utility rows are duplicated for {model}/{strategy}.")
        candidate_sets = [
            set(seed_group["candidate_id"]) for _, seed_group in group.groupby("seed", sort=True)
        ]
        if any(values != candidate_sets[0] for values in candidate_sets[1:]):
            raise ValueError(f"Candidate pool is not shared across seeds for {model}/{strategy}.")
        survivors = candidate_sets[0]
        if str(model) in model_survivors and model_survivors[str(model)] != survivors:
            raise ValueError(
                f"Candidate survivor pool is not shared across strategies for {model}."
            )
        model_survivors[str(model)] = survivors
        ascending = DIRECTIONS[strategy] == "minimize"
        group = group.copy()
        group["within_seed_rank"] = group.groupby("seed")["value"].rank(
            method="average", ascending=ascending
        )
        aggregate = (
            group.groupby("candidate_id", sort=True)
            .agg(
                mean_rank=("within_seed_rank", "mean"),
                mean_proxy_value=("value", "mean"),
                n_development_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        aggregate["search_utility"] = -aggregate["mean_rank"]
        aggregate["model"] = str(model)
        aggregate["strategy"] = strategy
        aggregate["direction"] = DIRECTIONS[strategy]
        rows.extend(aggregate.to_dict("records"))
    output = pd.DataFrame(rows)
    expected_groups = len(MODELS) * len(STRATEGIES)
    if output.groupby(["model", "strategy"]).ngroups != expected_groups:
        raise ValueError("Full-pool search utility lacks one or more model/proxy groups.")
    return output


def _holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    n = len(values)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _positive_spearman_permutation(
    utility: np.ndarray,
    outcome: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute Spearman rho and a one-sided candidate-label permutation p-value."""
    observed = float(stats.spearmanr(utility, outcome).statistic)
    if not math.isfinite(observed):
        raise ValueError("Observed Spearman association is undefined.")
    exceedances = 0
    for _ in range(int(n_permutations)):
        permuted = rng.permutation(outcome)
        value = float(stats.spearmanr(utility, permuted).statistic)
        exceedances += int(value >= observed)
    return observed, (exceedances + 1.0) / (int(n_permutations) + 1.0)


def rank_analysis(
    outcome_frame: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    panel: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    random_seed: int = 904_021,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute all frozen rank estimands and paired hierarchical bootstrap draws."""
    if n_permutations <= 0 or n_bootstrap <= 0:
        raise ValueError("Permutation and bootstrap counts must be positive.")
    utility = compute_search_utility(
        candidate_metrics, development_seeds=(101, 202, 303, 404, 505)
    )
    panel_ids = tuple(map(str, panel["candidate_ids"]))
    seeds = tuple(map(int, panel["reporting_seeds"]))
    outcomes = outcome_frame.copy()
    outcomes["candidate_id"] = outcomes["candidate_id"].astype(str).str.zfill(3)
    outcomes["reporting_seed"] = outcomes["reporting_seed"].astype(int)
    outcomes["value"] = pd.to_numeric(outcomes["value"], errors="raise")
    candidate_seed = (
        outcomes.groupby(
            ["model", "strategy", "candidate_id", "reporting_seed", "metric"],
            sort=True,
        )["value"]
        .mean()
        .reset_index()
    )
    expected_seed_rows = len(MODELS) * len(STRATEGIES) * len(panel_ids) * len(seeds) * len(METRICS)
    if len(candidate_seed) != expected_seed_rows:
        raise ValueError("Candidate/seed outcome coverage is not exact.")

    rng = np.random.default_rng(random_seed)
    association_rows = []
    seed_rows = []
    bootstrap_rows = []
    top_k = int(contract["estimands"]["top_k"])
    for metric in METRICS:
        metric_indices = []
        for model in MODELS:
            model_draws = {strategy: [] for strategy in STRATEGIES}
            model_association_indices = []
            candidate_samples = rng.integers(0, len(panel_ids), size=(n_bootstrap, len(panel_ids)))
            seed_samples = rng.integers(0, len(seeds), size=(n_bootstrap, len(seeds)))
            for strategy in STRATEGIES:
                util = utility[
                    (utility["model"] == model)
                    & (utility["strategy"] == strategy)
                    & (utility["candidate_id"].isin(panel_ids))
                ].set_index("candidate_id")
                if set(util.index) != set(panel_ids):
                    raise ValueError(f"Panel utility coverage failed for {model}/{strategy}.")
                util = util.loc[list(panel_ids)]
                branch = candidate_seed[
                    (candidate_seed["metric"] == metric)
                    & (candidate_seed["model"] == model)
                    & (candidate_seed["strategy"] == strategy)
                ]
                matrix = (
                    branch.pivot(index="candidate_id", columns="reporting_seed", values="value")
                    .loc[list(panel_ids), list(seeds)]
                    .to_numpy(dtype=float)
                )
                mean_outcome = matrix.mean(axis=1)
                rho, pvalue = _positive_spearman_permutation(
                    util["search_utility"].to_numpy(dtype=float),
                    mean_outcome,
                    n_permutations=n_permutations,
                    rng=rng,
                )
                kendall = float(
                    stats.kendalltau(
                        util["search_utility"].to_numpy(dtype=float),
                        mean_outcome,
                        variant="b",
                    ).statistic
                )
                seed_rhos = []
                for seed_index, seed in enumerate(seeds):
                    seed_rho = float(
                        stats.spearmanr(
                            util["search_utility"].to_numpy(dtype=float),
                            matrix[:, seed_index],
                        ).statistic
                    )
                    seed_rhos.append(seed_rho)
                    seed_rows.append(
                        {
                            "metric": metric,
                            "model": model,
                            "strategy": strategy,
                            "reporting_seed": seed,
                            "spearman_rho": seed_rho,
                        }
                    )
                ordered_proxy = util.reset_index().sort_values(
                    ["search_utility", "mean_proxy_value", "candidate_id"],
                    ascending=[
                        False,
                        DIRECTIONS[strategy] == "minimize",
                        True,
                    ],
                    kind="mergesort",
                )
                proxy_top = list(ordered_proxy["candidate_id"].head(top_k))
                outcome_map = dict(zip(panel_ids, mean_outcome))
                oracle_top = sorted(
                    panel_ids, key=lambda candidate: (-outcome_map[candidate], candidate)
                )[:top_k]
                overlap = len(set(proxy_top) & set(oracle_top))
                proxy_top_mean = float(np.mean([outcome_map[value] for value in proxy_top]))
                oracle_top_mean = float(np.mean([outcome_map[value] for value in oracle_top]))
                panel_mean = float(mean_outcome.mean())
                proxy_best = proxy_top[0]
                oracle_best = max(mean_outcome)
                draws = []
                utility_values = util["search_utility"].to_numpy(dtype=float)
                for replicate in range(n_bootstrap):
                    candidate_index = candidate_samples[replicate]
                    seed_index = seed_samples[replicate]
                    sampled_outcome = matrix[:, seed_index].mean(axis=1)
                    draw = float(
                        stats.spearmanr(
                            utility_values[candidate_index],
                            sampled_outcome[candidate_index],
                        ).statistic
                    )
                    draws.append(draw)
                    model_draws[strategy].append(draw)
                finite_draws = np.asarray(draws, dtype=float)
                finite_draws = finite_draws[np.isfinite(finite_draws)]
                if finite_draws.size == 0:
                    raise ValueError("All hierarchical bootstrap correlations are undefined.")
                association_rows.append(
                    {
                        "metric": metric,
                        "model": model,
                        "strategy": strategy,
                        "spearman_rho": rho,
                        "spearman_permutation_p": pvalue,
                        "kendall_tau_b": kendall,
                        "seed_spearman_min": float(min(seed_rhos)),
                        "seed_spearman_max": float(max(seed_rhos)),
                        "top_k": top_k,
                        "top_k_overlap": overlap,
                        "top_k_jaccard": overlap / (2 * top_k - overlap),
                        "top_k_enrichment": proxy_top_mean - panel_mean,
                        "top_k_oracle_regret": oracle_top_mean - proxy_top_mean,
                        "proxy_best_candidate": proxy_best,
                        "panel_oracle_candidate": min(
                            candidate
                            for candidate, value in outcome_map.items()
                            if value == oracle_best
                        ),
                        "proxy_best_regret": oracle_best - outcome_map[proxy_best],
                        "bootstrap_spearman_median": float(np.median(finite_draws)),
                        "bootstrap_spearman_ci_low": float(np.quantile(finite_draws, 0.025)),
                        "bootstrap_spearman_ci_high": float(np.quantile(finite_draws, 0.975)),
                        "n_permutations": int(n_permutations),
                        "n_bootstrap_requested": int(n_bootstrap),
                        "n_bootstrap_effective": int(finite_draws.size),
                    }
                )
                association_index = len(association_rows) - 1
                metric_indices.append(association_index)
                model_association_indices.append(association_index)
            paired_finite = np.all(
                np.isfinite(np.column_stack([model_draws[strategy] for strategy in STRATEGIES])),
                axis=1,
            )
            n_paired_effective = int(paired_finite.sum())
            if n_paired_effective == 0:
                raise ValueError("All paired hierarchical bootstrap draws are undefined.")
            for index in model_association_indices:
                association_rows[index]["n_bootstrap_effective_paired"] = n_paired_effective
            for replicate in range(n_bootstrap):
                bootstrap_rows.append(
                    {
                        "metric": metric,
                        "model": model,
                        "replicate": replicate,
                        "paired_draw_finite": bool(paired_finite[replicate]),
                        "n_bootstrap_requested": int(n_bootstrap),
                        "n_bootstrap_effective_paired": n_paired_effective,
                        **{
                            f"spearman_{strategy}": model_draws[strategy][replicate]
                            for strategy in STRATEGIES
                        },
                    }
                )
        pvalues = [association_rows[index]["spearman_permutation_p"] for index in metric_indices]
        family_size = len(MODELS) * len(STRATEGIES)
        if len(pvalues) != family_size:
            raise AssertionError(f"Each Holm family must contain exactly {family_size} tests.")
        for index, adjusted in zip(metric_indices, _holm_adjust(pvalues)):
            association_rows[index]["spearman_holm_p"] = adjusted
            association_rows[index]["holm_family_size"] = family_size
    return (
        pd.DataFrame(association_rows),
        pd.DataFrame(seed_rows),
        pd.DataFrame(bootstrap_rows),
    )


def analyze(
    root: Path,
    audit_sha256: str,
    checkpoint_manifest_sha256: str,
    *,
    n_permutations: int,
    n_bootstrap: int,
) -> list[Path]:
    """Run the frozen rank analysis after exact sealed-result collection."""
    audit, _ = _load_audit(root, audit_sha256)
    panel, contract = _validate_frozen_design()
    outcome_path, outcome_provenance = collect_rows(root, audit_sha256, checkpoint_manifest_sha256)
    outcomes = pd.read_csv(outcome_path, dtype={"candidate_id": str})
    candidate_metrics = pd.read_csv(audit["candidate_metrics"], dtype={"candidate_id": str})
    associations, seeds, bootstrap = rank_analysis(
        outcomes,
        candidate_metrics,
        panel,
        contract,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
    )
    output_dir = root.resolve() / "analysis"
    paths = [
        output_dir / "rank_associations.csv",
        output_dir / "seed_correlations.csv",
        output_dir / "paired_hierarchical_bootstrap.csv",
    ]
    for frame, path in zip((associations, seeds, bootstrap), paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    provenance = {
        "schema_version": 1,
        "audit_sha256": audit_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "outcomes": str(outcome_path.resolve()),
        "outcomes_sha256": _sha256(outcome_path),
        "outcome_provenance": str(outcome_provenance.resolve()),
        "outcome_provenance_sha256": _sha256(outcome_provenance),
        "candidate_metrics": audit["candidate_metrics"],
        "candidate_metrics_sha256": audit["candidate_metrics_sha256"],
        "panel_sha256": PANEL_SHA256,
        "execution_contract_sha256": CONTRACT_SHA256,
        "n_permutations": n_permutations,
        "n_bootstrap_requested": n_bootstrap,
        "n_bootstrap_effective_min": int(associations["n_bootstrap_effective"].min()),
        "n_bootstrap_effective_paired_min": int(
            associations["n_bootstrap_effective_paired"].min()
        ),
        "outputs": {path.name: _sha256(path) for path in paths},
    }
    provenance_path = output_dir / "rank_analysis_provenance.json"
    _atomic_json(provenance_path, provenance)
    return [*paths, provenance_path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse design, execution, freeze, collection, and analysis commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    design_parser = sub.add_parser("design")
    design_parser.add_argument("--root", type=Path, required=True)
    design_parser.add_argument("--campaign-manifest", type=Path, required=True)
    design_parser.add_argument("--campaign-manifest-sha256", required=True)
    design_parser.add_argument("--candidate-metrics", type=Path, required=True)
    design_parser.add_argument("--candidate-metrics-sha256", required=True)
    design_parser.add_argument("--candidate-metrics-provenance", type=Path, required=True)
    design_parser.add_argument("--candidate-metrics-provenance-sha256", required=True)
    design_parser.add_argument("--pairing-manifest", type=Path, required=True)
    design_parser.add_argument("--pairing-manifest-sha256", required=True)
    design_parser.add_argument("--encoder-pair-table", type=Path, required=True)
    design_parser.add_argument("--encoder-pair-table-sha256", required=True)

    for command in ("run-canary", "run-train"):
        run = sub.add_parser(command)
        run.add_argument("--root", type=Path, required=True)
        run.add_argument("--audit-sha256", required=True)
        run.add_argument("--trajectory-index", type=int, required=True)

    freeze = sub.add_parser("freeze-checkpoints")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--audit-sha256", required=True)

    evaluate = sub.add_parser("run-evaluate")
    evaluate.add_argument("--root", type=Path, required=True)
    evaluate.add_argument("--audit-sha256", required=True)
    evaluate.add_argument("--checkpoint-manifest-sha256", required=True)
    evaluate.add_argument("--trajectory-index", type=int, required=True)

    collection = sub.add_parser("collect")
    collection.add_argument("--root", type=Path, required=True)
    collection.add_argument("--audit-sha256", required=True)
    collection.add_argument("--checkpoint-manifest-sha256", required=True)

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--root", type=Path, required=True)
    analysis.add_argument("--audit-sha256", required=True)
    analysis.add_argument("--checkpoint-manifest-sha256", required=True)
    analysis.add_argument("--n-permutations", type=int, default=10_000)
    analysis.add_argument("--n-bootstrap", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the frozen candidate-rank audit CLI."""
    args = parse_args(argv)
    if args.command == "design":
        audit = design(
            args.root,
            args.campaign_manifest,
            args.campaign_manifest_sha256,
            args.candidate_metrics,
            args.candidate_metrics_sha256,
            args.candidate_metrics_provenance,
            args.candidate_metrics_provenance_sha256,
            args.pairing_manifest,
            args.pairing_manifest_sha256,
            args.encoder_pair_table,
            args.encoder_pair_table_sha256,
        )
        path = args.root.resolve() / "audit.json"
        print(path)
        print(_sha256(path))
        print(audit["trajectory_manifest"])
    elif args.command == "run-canary":
        marker = run_canary(args.root, args.audit_sha256, args.trajectory_index)
        print(marker["checkpoint_fingerprint"])
    elif args.command == "run-train":
        marker = run_train(args.root, args.audit_sha256, args.trajectory_index)
        print(marker["branch_manifest"])
    elif args.command == "freeze-checkpoints":
        path, digest = freeze_checkpoints(args.root, args.audit_sha256)
        print(path)
        print(digest)
    elif args.command == "run-evaluate":
        marker = run_evaluate(
            args.root,
            args.audit_sha256,
            args.checkpoint_manifest_sha256,
            args.trajectory_index,
        )
        print(marker["evaluation_rows"])
    elif args.command == "collect":
        for path in collect_rows(args.root, args.audit_sha256, args.checkpoint_manifest_sha256):
            print(path)
    elif args.command == "analyze":
        for path in analyze(
            args.root,
            args.audit_sha256,
            args.checkpoint_manifest_sha256,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
        ):
            print(path)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
