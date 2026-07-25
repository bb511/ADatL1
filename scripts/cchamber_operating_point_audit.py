#!/usr/bin/env python3
"""Threshold-safe sidecar evaluation for the frozen Causal Chamber campaign.

The sidecar authenticates every label-free campaign input, calibrates one
immutable threshold per checkpoint using only ``validation.normal``, freezes all
200 thresholds, then evaluates ``test.normal`` and all 58 interventions in one
pass. It never invokes the legacy campaign evaluator or mutates a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import struct
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from mlflow import MlflowClient
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import BinaryAveragePrecision

from scripts import generation
from src.callbacks.metrics.rate import AnomalyRate as ValidationAnomalyRate
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback
from src.utils.pairing.io import compose_config
from src.utils.pairing.table import load_pair_table, sha256_tensor

CAMPAIGN_TRAINING_COMMIT = "63b941a287c48c84e2537d0cfbd07c2240435c0e"
TARGET_FPR = 0.01
EXPECTED_RECORDS = 200
EXPECTED_NORMAL_COUNT = 1_000
EXPECTED_INTERVENTIONS = 58
EXPECTED_RESULT_ROWS = 23_200
THRESHOLD_SCHEMA_VERSION = 1
DEVELOPMENT_SEEDS = (101, 202, 303, 404, 505)
PAIR_ENCODER_SEEDS = (123, 456, 789, 101112, 131415)
WILSON_Z_95 = 1.959963984540054
MODELS = ("ae", "vae", "svdd", "realnvp")
STRATEGIES = (
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "cap_random",
    "drift",
    "wasserstein",
)


class _ValidationThresholdCallback(Callback):
    """Derive and persist the operational threshold from validation.normal only."""

    def __init__(self, output_name: str = "ascore/full") -> None:
        super().__init__()
        self.output_name = output_name
        self.scores: list[torch.Tensor] = []
        self.inputs: list[torch.Tensor] = []
        self.sample_ids: list[torch.Tensor] = []
        self.validation_normal_count = 0

    def on_validation_start(self, trainer, pl_module) -> None:
        """Validate the direct false-positive-rate and one-loader contract."""
        if set(trainer.val_dataloaders) != {"normal"}:
            raise ValueError("Threshold calibration requires exactly validation.normal.")
        if (
            float(pl_module.hparams.target_rate) != TARGET_FPR
            or pl_module.hparams.base_rate is not None
        ):
            raise ValueError("Threshold calibration requires direct target rate 0.01.")

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Reset score state before calibration."""
        self.scores = []
        self.inputs = []
        self.sample_ids = []

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        """Accumulate finite validation-normal anomaly scores."""
        del trainer, pl_module, batch_idx, dataloader_idx
        scores = outputs[self.output_name].detach().view(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("Threshold calibration requires finite validation scores.")
        if not isinstance(batch, Mapping) or "x" not in batch or "sample_id" not in batch:
            raise ValueError("Calibration batches require x and sample_id provenance.")
        inputs = batch["x"].detach().cpu()
        sample_ids = batch["sample_id"].detach().view(-1).cpu()
        if inputs.shape[0] != scores.numel() or sample_ids.numel() != scores.numel():
            raise ValueError("Calibration score/input/sample-id counts differ.")
        self.scores.append(scores)
        self.inputs.append(inputs)
        self.sample_ids.append(sample_ids)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Set the higher-quantile operational buffer used by training calibration."""
        del trainer
        if not self.scores:
            raise ValueError("Threshold calibration received no validation-normal scores.")
        scores = torch.cat(self.scores).float()
        inputs = torch.cat(self.inputs).float()
        sample_ids = torch.cat(self.sample_ids).long()
        if scores.numel() != EXPECTED_NORMAL_COUNT:
            raise ValueError(
                f"Threshold calibration requires exactly {EXPECTED_NORMAL_COUNT} "
                f"validation.normal samples, found {scores.numel()}."
            )
        if torch.unique(sample_ids).numel() != sample_ids.numel():
            raise ValueError("validation.normal sample IDs must be unique.")
        rate = ValidationAnomalyRate(TARGET_FPR, None).to(pl_module.device)
        rate.set_threshold(scores)
        threshold = rate.threshold.detach().clone().float()
        if "thres_operational" not in dict(pl_module.named_buffers()):
            pl_module.register_buffer("thres_operational", threshold, persistent=True)
        else:
            pl_module.thres_operational.data.copy_(threshold)
        self.validation_normal_count = int(scores.numel())
        self.validation_scores = scores.detach().cpu()
        self.validation_inputs_sha256 = sha256_tensor(inputs)
        self.validation_sample_ids = sample_ids.tolist()
        self.validation_sample_ids_sha256 = sha256_tensor(sample_ids)


def _sha256(path: Path) -> str:
    """Hash one file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    """Serialize a value deterministically for identity hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Any) -> None:
    """Atomically write one strict JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_json_create(path: Path, value: Any) -> None:
    """Atomically create a JSON marker without replacing a concurrent result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json_after_finished_run(
    path: Path,
    value: Mapping[str, Any],
    tracking_uri: str,
    run_id: str,
    expected_tags: Mapping[str, str],
) -> bool:
    """Publish a canonical marker only after its MLflow run is authenticated."""
    _require_finished_run(tracking_uri, run_id, expected_tags)
    try:
        _atomic_json_create(path, value)
    except FileExistsError:
        return False
    return True


def _load_json(path: Path) -> Any:
    """Load an existing JSON artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str, label: str) -> None:
    """Require that an artifact retains its frozen SHA-256 identity."""
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch for {path}: {actual}, expected {expected}.")


def _git(*args: str) -> str:
    """Run a fixed git inspection command inside this repository."""
    return subprocess.check_output(  # nosec B603 B607
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _audit_revision() -> tuple[str, str]:
    """Return audit commit/branch, requiring a reproducible clean deployment."""
    dirty = _git("status", "--porcelain")
    if dirty:
        raise RuntimeError("Operating-point execution requires a clean deployment worktree.")
    return _git("rev-parse", "HEAD"), _git("branch", "--show-current")


def _require_deployment(inventory: Mapping[str, Any]) -> tuple[str, str]:
    """Require a clean deployment at the exact inventory code revision."""
    commit, branch = _audit_revision()
    if commit != inventory.get("audit_code_commit") or branch != inventory.get(
        "audit_code_branch"
    ):
        raise RuntimeError(
            "Deployment revision differs from the frozen operating-point inventory."
        )
    return commit, branch


def _require_slurm_gpu(accelerator: str) -> None:
    """Refuse GPU calibration/evaluation outside a Slurm GPU allocation."""
    if accelerator != "gpu":
        return
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("GPU threshold-sidecar execution must run inside Slurm.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU threshold-sidecar execution requires CUDA.")


def _expected_identities(campaign: Mapping[str, Any]) -> set[tuple[str, str, int]]:
    """Return the frozen 4-by-5-by-10 campaign identity set."""
    models = tuple(map(str, campaign.get("models", ())))
    strategies = tuple(map(str, campaign.get("strategies", ())))
    seeds = tuple(int(value) for value in campaign.get("reporting_seeds", ()))
    if models != MODELS or strategies != STRATEGIES or len(seeds) != 10:
        raise ValueError("Campaign model/strategy/reporting-seed contract has changed.")
    expected = {
        (model, strategy, seed) for model in models for strategy in strategies for seed in seeds
    }
    if len(expected) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} campaign identities, found {len(expected)}."
        )
    return expected


def _validate_dataset_sources(campaign: Mapping[str, Any], *, full_tree: bool) -> tuple[Path, str]:
    """Validate the consumed normal CSV, and optionally the complete frozen tree."""
    frozen = campaign.get("dataset_files")
    if not isinstance(frozen, list) or len(frozen) != 59:
        raise ValueError("Campaign must pin exactly 59 Causal Chamber CSV files.")
    current: list[dict[str, Any]] = []
    normal: tuple[Path, str] | None = None
    for source in frozen:
        source_path = Path(str(source["path"])).resolve()
        if source_path.name == "uniform_reference.csv":
            normal = (source_path, str(source["sha256"]))
        if full_tree or source_path.name == "uniform_reference.csv":
            _require_hash(source_path, str(source["sha256"]), "campaign dataset file")
            if source_path.stat().st_size != int(source["size"]):
                raise ValueError(f"Campaign dataset size changed: {source_path}")
        current.append(
            {
                "path": str(source_path),
                "size": int(source["size"]),
                "sha256": str(source["sha256"]),
            }
        )
    if normal is None:
        raise ValueError("Campaign does not pin uniform_reference.csv.")
    tree_hash = hashlib.sha256(_canonical_json(current).encode("utf-8")).hexdigest()
    if tree_hash != campaign.get("dataset_tree_sha256"):
        raise ValueError("Campaign dataset-tree identity is invalid.")
    return normal


def _validate_campaign(
    path: Path, expected_sha256: str, *, full_data_tree: bool = False
) -> dict[str, Any]:
    """Validate the immutable campaign hash and scientific identity."""
    _require_hash(path, expected_sha256, "campaign")
    campaign = _load_json(path)
    if campaign.get("git_commit") != CAMPAIGN_TRAINING_COMMIT:
        raise ValueError(
            "Operating-point audit only accepts the frozen campaign trained at "
            f"{CAMPAIGN_TRAINING_COMMIT}."
        )
    if campaign.get("dataset") != "lt_interventions_standard_v1":
        raise ValueError("Unexpected Causal Chamber dataset identity.")
    if campaign.get("feature_set") != "readouts" or int(campaign.get("n_features", -1)) != 11:
        raise ValueError("Unexpected Causal Chamber feature contract.")
    _expected_identities(campaign)
    _validate_dataset_sources(campaign, full_tree=full_data_tree)
    return dict(campaign)


def _require_external_output(output: Path, campaign_root: Path) -> Path:
    """Reject any audit write path inside the immutable campaign tree."""
    resolved = output.resolve()
    root = campaign_root.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(f"Audit output must be outside the campaign tree: {resolved}")
    return resolved


def _artifact_record(path: Path, label: str) -> dict[str, Any]:
    """Return an authenticated file record."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}


def _require_artifact(record: Mapping[str, Any], label: str) -> Path:
    """Revalidate one authenticated file record."""
    path = Path(str(record["path"])).resolve()
    _require_hash(path, str(record["sha256"]), label)
    if path.stat().st_size != int(record["size"]):
        raise ValueError(f"{label} size changed: {path}")
    return path


def _float32_identity(value: float | torch.Tensor) -> dict[str, Any]:
    """Encode one finite threshold by its exact IEEE-754 float32 bytes."""
    scalar = float(torch.as_tensor(value, dtype=torch.float32).item())
    if not torch.isfinite(torch.tensor(scalar)):
        raise ValueError("Threshold must be finite.")
    raw = struct.pack("<f", scalar)
    return {
        "value": scalar,
        "little_endian_hex": raw.hex(),
        "uint32": struct.unpack("<I", raw)[0],
        "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": "float32",
    }


def _decode_float32_identity(record: Mapping[str, Any]) -> torch.Tensor:
    """Validate and decode an exact float32 identity."""
    raw = bytes.fromhex(str(record["little_endian_hex"]))
    if len(raw) != 4 or hashlib.sha256(raw).hexdigest() != record.get("bytes_sha256"):
        raise ValueError("Threshold byte identity is invalid.")
    value = struct.unpack("<f", raw)[0]
    if struct.unpack("<I", raw)[0] != int(record["uint32"]):
        raise ValueError("Threshold uint32 identity is invalid.")
    if _float32_identity(value) != dict(record):
        raise ValueError("Threshold float32 identity is internally inconsistent.")
    return torch.tensor(value, dtype=torch.float32)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write a non-empty, rectangular CSV."""
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError("CSV rows do not share one exact schema.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson binomial proportion interval."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= positive trials.")
    proportion = successes / trials
    z2 = WILSON_Z_95**2
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    radius = (
        WILSON_Z_95
        * ((proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials**2)) ** 0.5)
        / denominator
    )
    return center - radius, center + radius


def _validate_pairing_inventory(
    campaign_root: Path, campaign: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the primary seed-123 validation and test pair tables."""
    path = campaign_root / "pairing" / "comparison" / "pairing_manifest.json"
    manifest = _load_json(path)
    encoder_runs = manifest.get("encoder_runs")
    if (
        manifest.get("campaign_id") != campaign["campaign_id"]
        or int(manifest.get("primary_encoder_seed", -1)) != 123
        or not isinstance(encoder_runs, list)
        or len(encoder_runs) != len(PAIR_ENCODER_SEEDS)
        or {int(row.get("encoder_seed", -1)) for row in encoder_runs} != set(PAIR_ENCODER_SEEDS)
    ):
        raise ValueError("Pairing manifest does not preserve the primary seed-123 contract.")
    for row in encoder_runs:
        if (
            row.get("campaign_id") != campaign["campaign_id"]
            or row.get("git_commit") != CAMPAIGN_TRAINING_COMMIT
            or int(row.get("data_seed", -1)) != 314159
        ):
            raise ValueError("Pairing encoder run campaign/commit/data-seed changed.")
    primary = next(
        (row for row in manifest["encoder_runs"] if int(row.get("encoder_seed", -1)) == 123),
        None,
    )
    if primary is None:
        raise ValueError("Pairing manifest has no primary seed-123 encoder record.")
    records = {"manifest": _artifact_record(path, "pairing manifest")}
    for split, expected_split in (("validation", "validate"), ("test", "test")):
        table_path = Path(str(primary[f"{split}_table"])).resolve()
        top_level_path = Path(str(manifest[f"primary_{split}_table"])).resolve()
        if table_path != top_level_path:
            raise ValueError(f"Primary {split} pair-table path changed.")
        expected_sha = str(primary[f"{split}_table_sha256"])
        _require_hash(table_path, expected_sha, f"primary {split} pair table")
        table = load_pair_table(
            table_path,
            expected_dataset_1="normal",
            expected_dataset_2="reference_normal",
            expected_split=expected_split,
            n_dataset_1=EXPECTED_NORMAL_COUNT,
            n_dataset_2=EXPECTED_NORMAL_COUNT,
        )
        if (
            Path(str(table["encoder_ckpt"])).resolve()
            != Path(str(primary["encoder_checkpoint"])).resolve()
        ):
            raise ValueError(f"Primary {split} table encoder-checkpoint path changed.")
        records[f"{split}_table"] = {
            **_artifact_record(table_path, f"primary {split} pair table"),
            "source_1_sha256": table["metadata"]["source_1_sha256"],
            "source_2_sha256": table["metadata"]["source_2_sha256"],
            "encoder_checkpoint_sha256": table["metadata"]["encoder_checkpoint_sha256"],
        }
    encoder = Path(str(primary["encoder_checkpoint"])).resolve()
    _require_hash(encoder, str(primary["encoder_checkpoint_sha256"]), "pairing encoder checkpoint")
    records["encoder_checkpoint"] = _artifact_record(encoder, "pairing encoder checkpoint")
    if (
        records["validation_table"]["sha256"] != manifest["primary_validation_table_sha256"]
        or records["test_table"]["sha256"] != manifest["primary_test_table_sha256"]
        or records["validation_table"]["encoder_checkpoint_sha256"]
        != records["encoder_checkpoint"]["sha256"]
        or records["test_table"]["encoder_checkpoint_sha256"]
        != records["encoder_checkpoint"]["sha256"]
    ):
        raise ValueError("Primary pairing table/checkpoint provenance is inconsistent.")
    return records


def _validate_selection_inventory(campaign_root: Path, campaign_sha256: str) -> dict[str, Any]:
    """Authenticate candidate metrics, selection, and retrain inputs as one chain."""
    selection = campaign_root / "selection"
    names = {
        "candidate_metrics": selection / "candidate_metrics.csv",
        "candidate_metrics_provenance": selection / "candidate_metrics_provenance.json",
        "selected_trials": selection / "selected_trials.csv",
        "selection_provenance": selection / "selection_provenance.json",
        "retrain_manifest": selection / "retrain_manifest.json",
    }
    records = {name: _artifact_record(path, name) for name, path in names.items()}
    candidate_provenance = _load_json(names["candidate_metrics_provenance"])
    selection_provenance = _load_json(names["selection_provenance"])
    candidate_path = names["candidate_metrics"].resolve()
    campaign_path = (campaign_root / "campaign.json").resolve()
    if (
        Path(str(candidate_provenance.get("campaign", ""))).resolve() != campaign_path
        or candidate_provenance.get("campaign_sha256") != campaign_sha256
        or Path(str(candidate_provenance.get("candidate_metrics", ""))).resolve() != candidate_path
        or candidate_provenance.get("candidate_metrics_sha256")
        != records["candidate_metrics"]["sha256"]
        or Path(str(selection_provenance.get("candidate_metrics", ""))).resolve() != candidate_path
        or selection_provenance.get("candidate_metrics_sha256")
        != records["candidate_metrics"]["sha256"]
        or selection_provenance.get("selected_trials_sha256")
        != records["selected_trials"]["sha256"]
        or selection_provenance.get("retrain_manifest_sha256")
        != records["retrain_manifest"]["sha256"]
        or int(selection_provenance.get("n_selected", -1)) != len(MODELS) * len(STRATEGIES)
        or int(selection_provenance.get("n_retrains", -1)) != EXPECTED_RECORDS
        or tuple(map(int, selection_provenance.get("development_seeds", ()))) != DEVELOPMENT_SEEDS
        or selection_provenance.get("intervention_labels_used") is not False
    ):
        raise ValueError("Selection provenance chain is incomplete or inconsistent.")
    return records


def build_inventory(
    campaign_root: Path,
    campaign_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Freeze all 200 label-free source identities and hashes."""
    campaign_root = campaign_root.resolve()
    output = _require_external_output(output, campaign_root)
    audit_commit, audit_branch = _audit_revision()
    campaign_path = campaign_root / "campaign.json"
    campaign = _validate_campaign(campaign_path, campaign_sha256, full_data_tree=True)
    normal_source, normal_source_sha256 = _validate_dataset_sources(campaign, full_tree=False)
    selection_inventory = _validate_selection_inventory(campaign_root, campaign_sha256)
    pairing_inventory = _validate_pairing_inventory(campaign_root, campaign)
    manifest_path = Path(selection_inventory["retrain_manifest"]["path"])
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, list) or len(manifest) != EXPECTED_RECORDS:
        raise ValueError(f"Retrain manifest must contain exactly {EXPECTED_RECORDS} records.")

    expected = _expected_identities(campaign)
    seen: set[tuple[str, str, int]] = set()
    records: list[dict[str, Any]] = []
    for index, selected in enumerate(manifest):
        marker_path = campaign_root / "retrain_results" / f"{index:03d}.json"
        marker = _load_json(marker_path)
        identity = (
            str(selected["model"]),
            str(selected["strategy"]),
            int(selected["seed"]),
        )
        if identity not in expected or identity in seen:
            raise ValueError(
                f"Invalid or duplicate retrain identity at index {index}: {identity}."
            )
        seen.add(identity)
        required = {
            "campaign_id": campaign["campaign_id"],
            "git_commit": CAMPAIGN_TRAINING_COMMIT,
            "manifest_index": index,
            "model": identity[0],
            "strategy": identity[1],
            "seed": identity[2],
            "candidate_id": selected["candidate_id"],
            "pool_sha256": selected["pool_sha256"],
        }
        for key, value in required.items():
            if marker.get(key) != value:
                raise ValueError(
                    f"Retrain marker {marker_path} has {key}={marker.get(key)!r}, "
                    f"expected {value!r}."
                )
        if marker.get("params") != selected.get("params"):
            raise ValueError(f"Retrain marker parameters differ at index {index}.")
        if (
            marker.get("valid_pair_table_sha256")
            != pairing_inventory["validation_table"]["sha256"]
            or marker.get("test_pair_table_sha256") != pairing_inventory["test_table"]["sha256"]
        ):
            raise ValueError(f"Retrain marker pairing provenance differs at index {index}.")
        checkpoint = Path(str(marker["checkpoint"])).resolve()
        _require_hash(checkpoint, str(marker["checkpoint_sha256"]), "checkpoint")
        records.append(
            {
                **required,
                "params": selected["params"],
                "params_sha256": hashlib.sha256(
                    _canonical_json(selected["params"]).encode("utf-8")
                ).hexdigest(),
                "retrain_marker": str(marker_path.resolve()),
                "retrain_marker_sha256": _sha256(marker_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": str(marker["checkpoint_sha256"]),
                "valid_pair_table_sha256": marker.get("valid_pair_table_sha256"),
                "test_pair_table_sha256": marker.get("test_pair_table_sha256"),
                "training_mlflow_run_id": marker.get("mlflow_run_id"),
            }
        )
    if seen != expected:
        raise ValueError("Retrain manifest does not provide exact Cartesian coverage.")

    inventory = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outcome_blind": True,
        "campaign": str(campaign_path.resolve()),
        "campaign_sha256": campaign_sha256,
        "campaign_id": campaign["campaign_id"],
        "campaign_training_commit": CAMPAIGN_TRAINING_COMMIT,
        "audit_code_commit": audit_commit,
        "audit_code_branch": audit_branch,
        "test_normal_source": str(normal_source),
        "test_normal_source_sha256": normal_source_sha256,
        "data_dir": str(normal_source.parent.parent),
        "retrain_manifest": str(manifest_path.resolve()),
        "retrain_manifest_sha256": _sha256(manifest_path),
        "selection_inventory": selection_inventory,
        "pairing_inventory": pairing_inventory,
        "expected_records": EXPECTED_RECORDS,
        "tracking_uri": _tracking_uri(output.parent),
        "records": records,
    }
    if output.exists():
        existing = _load_json(output)
        comparable = dict(existing)
        comparable.pop("created_at", None)
        proposed = dict(inventory)
        proposed.pop("created_at", None)
        if comparable != proposed:
            raise FileExistsError(f"Refusing to replace non-matching inventory: {output}")
        _write_slurm_scripts(output, output.parent)
        return dict(existing)
    _atomic_json(output, inventory)
    _write_slurm_scripts(output, output.parent)
    return inventory


def load_inventory(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and fully validate a frozen 200-record audit inventory."""
    path = path.resolve()
    _require_hash(path, expected_sha256, "audit inventory")
    inventory = _load_json(path)
    if (
        inventory.get("schema_version") != 1
        or inventory.get("outcome_blind") is not True
        or inventory.get("campaign_training_commit") != CAMPAIGN_TRAINING_COMMIT
        or not inventory.get("audit_code_commit")
        or not inventory.get("audit_code_branch")
        or int(inventory.get("expected_records", -1)) != EXPECTED_RECORDS
    ):
        raise ValueError("Invalid operating-point inventory contract.")
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORDS:
        raise ValueError(f"Audit inventory must contain exactly {EXPECTED_RECORDS} records.")
    campaign_path = Path(str(inventory["campaign"]))
    campaign = _validate_campaign(campaign_path, str(inventory["campaign_sha256"]))
    normal_source, normal_source_sha256 = _validate_dataset_sources(campaign, full_tree=False)
    if (
        inventory.get("test_normal_source") != str(normal_source)
        or inventory.get("test_normal_source_sha256") != normal_source_sha256
        or inventory.get("data_dir") != str(normal_source.parent.parent)
    ):
        raise ValueError("Audit inventory test-normal source identity changed.")
    manifest_path = Path(str(inventory["retrain_manifest"]))
    _require_hash(manifest_path, str(inventory["retrain_manifest_sha256"]), "retrain manifest")
    for name, record in inventory.get("selection_inventory", {}).items():
        _require_artifact(record, f"selection {name}")
    for name, record in inventory.get("pairing_inventory", {}).items():
        _require_artifact(record, f"pairing {name}")
    if set(inventory.get("selection_inventory", ())) != {
        "candidate_metrics",
        "candidate_metrics_provenance",
        "selected_trials",
        "selection_provenance",
        "retrain_manifest",
    } or set(inventory.get("pairing_inventory", ())) != {
        "manifest",
        "validation_table",
        "test_table",
        "encoder_checkpoint",
    }:
        raise ValueError("Audit inventory omits authenticated selection/pairing inputs.")
    expected = _expected_identities(campaign)
    identities = {(str(row["model"]), str(row["strategy"]), int(row["seed"])) for row in records}
    if identities != expected or [int(row["manifest_index"]) for row in records] != list(
        range(EXPECTED_RECORDS)
    ):
        raise ValueError("Audit inventory record coverage is not exact.")
    return dict(inventory)


def _validate_source_record(record: Mapping[str, Any], campaign_id: str) -> None:
    """Revalidate a retrain marker and checkpoint immediately before use."""
    marker_path = Path(str(record["retrain_marker"]))
    _require_hash(marker_path, str(record["retrain_marker_sha256"]), "retrain marker")
    marker = _load_json(marker_path)
    for key in (
        "manifest_index",
        "model",
        "strategy",
        "seed",
        "candidate_id",
        "pool_sha256",
        "checkpoint",
        "checkpoint_sha256",
    ):
        if marker.get(key) != record.get(key):
            raise ValueError(f"Frozen retrain marker identity changed for field {key!r}.")
    if (
        marker.get("campaign_id") != campaign_id
        or marker.get("git_commit") != CAMPAIGN_TRAINING_COMMIT
    ):
        raise ValueError("Frozen retrain marker campaign identity changed.")
    _require_hash(Path(str(record["checkpoint"])), str(record["checkpoint_sha256"]), "checkpoint")


def _compose_model_and_validation_loader(record: Mapping[str, Any], data_dir: str | Path):
    """Restore one selected model and expose only its validation-normal loader."""
    spec = generation.make_experiment_specification(
        dataset=generation.Dataset.CCHAMBER,
        model=generation.Model(str(record["model"])),
        strategy=generation.Strategy(str(record["strategy"])),
        n_trials=1,
        seeds=(int(record["seed"]),),
    )
    overrides = [
        f"experiment={spec.experiment}",
        *spec.fixed_overrides,
        *[
            f"{key}={json.dumps(value, separators=(',', ':'))}"
            for key, value in record["params"].items()
        ],
        f"data.data_dir={json.dumps(str(Path(data_dir).resolve()))}",
        "data.signal_experiments=[]",
        "data.max_val_batches=-1",
        "logger=none",
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("validate")
    loaders = datamodule.val_dataloader()
    if "normal" not in loaders:
        raise ValueError("Causal Chamber datamodule did not expose validation.normal.")
    normal_loader = loaders["normal"]
    # Preserve the feature-map discovery used by the algorithms without handing
    # Lightning any second loader.
    normal_loader.loader = datamodule.loader

    model = hydra.utils.instantiate(cfg.algorithm)
    checkpoint = Path(str(record["checkpoint"]))
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
    return cfg, datamodule, model, normal_loader


def _diagnostic_context(
    inventory_path: Path,
    inventory_sha256: str,
    inventory: Mapping[str, Any],
    record: Mapping[str, Any],
    audit_commit: str,
    audit_branch: str,
) -> dict[str, Any]:
    """Build exact source and code identities for one numeric row."""
    return {
        "campaign_id": inventory["campaign_id"],
        "manifest_index": int(record["manifest_index"]),
        "model": record["model"],
        "strategy": record["strategy"],
        "seed": int(record["seed"]),
        "candidate_id": record["candidate_id"],
        "campaign_training_commit": CAMPAIGN_TRAINING_COMMIT,
        "audit_code_commit": audit_commit,
        "audit_code_branch": audit_branch,
        "campaign_path": inventory["campaign"],
        "campaign_sha256": inventory["campaign_sha256"],
        "retrain_manifest_path": inventory["retrain_manifest"],
        "retrain_manifest_sha256": inventory["retrain_manifest_sha256"],
        "retrain_marker_path": record["retrain_marker"],
        "retrain_marker_sha256": record["retrain_marker_sha256"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "audit_inventory_path": str(inventory_path.resolve()),
        "audit_inventory_sha256": inventory_sha256,
        "params_sha256": record["params_sha256"],
        "valid_pair_table_sha256": record.get("valid_pair_table_sha256"),
        "test_pair_table_sha256": record.get("test_pair_table_sha256"),
        "training_mlflow_run_id": record.get("training_mlflow_run_id"),
        "test_normal_source_path": inventory["test_normal_source"],
        "test_normal_source_sha256": inventory["test_normal_source_sha256"],
    }


def _validate_diagnostic(path: Path, context: Mapping[str, Any]) -> dict[str, str]:
    """Validate one diagnostic row against identities and exact count arithmetic."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one diagnostic row.")
    row = rows[0]
    for key, value in context.items():
        expected = "" if value is None else str(value)
        if row.get(key) != expected:
            raise ValueError(f"{path}: context mismatch for {key!r}.")
    if float(row["target_fpr"]) != TARGET_FPR:
        raise ValueError(f"{path}: target false-positive rate changed.")
    if row["threshold_source"] != "audit_validation.normal:thres_operational":
        raise ValueError(f"{path}: threshold is not validation-derived.")
    if not torch.isfinite(torch.tensor(float(row["validation_derived_threshold"]))):
        raise ValueError(f"{path}: validation-derived threshold is not finite.")
    count = int(row["test_normal_count"])
    finite = int(row["test_normal_finite_count"])
    triggered = int(row["triggered_count"])
    if count <= 0 or finite != count or not 0 <= triggered <= count:
        raise ValueError(f"{path}: invalid test-normal counts.")
    if abs(float(row["achieved_test_normal_acceptance"]) - triggered / count) > 1e-12:
        raise ValueError(f"{path}: achieved acceptance does not match exact counts.")
    if abs(float(row["finite_sample_granularity"]) - 1.0 / count) > 1e-12:
        raise ValueError(f"{path}: finite-sample granularity is invalid.")
    if float(row["nominal_test_normal_acceptance"]) != TARGET_FPR:
        raise ValueError(f"{path}: nominal test-normal acceptance changed.")
    nearest = round(TARGET_FPR * count) / count
    if abs(float(row["nearest_attainable_test_normal_acceptance"]) - nearest) > 1e-12:
        raise ValueError(f"{path}: nearest attainable acceptance is invalid.")
    validation_count = int(row["validation_normal_count"])
    if validation_count <= 0 or (
        abs(float(row["validation_threshold_granularity"]) - 1.0 / validation_count) > 1e-12
    ):
        raise ValueError(f"{path}: validation threshold granularity is invalid.")
    return row


def _tracking_uri(output_root: Path) -> str:
    """Return the audit-only MLflow file-store URI."""
    return f"file:{(output_root / 'mlflow' / 'mlruns').resolve()}"


def _write_slurm_scripts(inventory_path: Path, output_root: Path) -> None:
    """Generate the guarded debug, packed, freeze, collect, and dependency workflow."""
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required to generate threshold-sidecar launchers.")
    scripts = output_root.resolve() / "slurm"
    scripts.mkdir(parents=True, exist_ok=True)
    common = (
        "set -euo pipefail\n"
        f"REPO={shlex.quote(str(REPO_ROOT))}\n"
        f"INVENTORY={shlex.quote(str(inventory_path.resolve()))}\n"
        f"OUTPUT_ROOT={shlex.quote(str(output_root.resolve()))}\n"
        "INVENTORY_SHA256=$(sha256sum \"$INVENTORY\" | awk '{print $1}')\n"
        'cd "$REPO"\n'
        f"UV=({shlex.quote(uv)} run --frozen --no-sync python)\n"
    )

    def packed(stage: str) -> str:
        threshold = (
            'THRESHOLD_MANIFEST="$OUTPUT_ROOT/threshold_manifest.json"\n'
            'test -f "$THRESHOLD_MANIFEST"\n'
            "THRESHOLD_SHA256=$(sha256sum \"$THRESHOLD_MANIFEST\" | awk '{print $1}')\n"
            if stage == "evaluate"
            else ""
        )
        threshold_args = (
            '--threshold-manifest "$THRESHOLD_MANIFEST" '
            '--threshold-manifest-sha256 "$THRESHOLD_SHA256" '
            if stage == "evaluate"
            else ""
        )
        command = "evaluate-index" if stage == "evaluate" else "calibrate-index"
        return (
            "#!/usr/bin/env bash\n"
            "#SBATCH --account=a0166\n#SBATCH --partition=normal\n"
            "#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=4\n"
            "#SBATCH --cpus-per-task=72\n"
            "#SBATCH --gpus-per-node=4\n#SBATCH --mem=440G\n"
            "#SBATCH --time=04:00:00\n#SBATCH --array=0-49%16\n"
            f"#SBATCH --job-name=cch-threshold-{stage}\n"
            + common
            + threshold
            + "pids=()\n"
            + "for slot in 0 1 2 3; do\n"
            + "  index=$((SLURM_ARRAY_TASK_ID * 4 + slot))\n"
            + "  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 "
            + "--gpus-per-node=1 --mem=110G "
            + '"${UV[@]}" scripts/cchamber_operating_point_audit.py '
            + f'{command} --inventory "$INVENTORY" '
            + '--inventory-sha256 "$INVENTORY_SHA256" '
            + threshold_args
            + '--output-root "$OUTPUT_ROOT" --manifest-index "$index" &\n'
            + '  pids+=("$!")\n'
            + "done\n"
            + 'status=0\nfor pid in "${pids[@]}"; do wait "$pid" || status=1; done\n'
            + 'exit "$status"\n'
        )

    debug = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --account=a0166\n#SBATCH --partition=debug\n"
        "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=72\n"
        "#SBATCH --gpus-per-node=1\n"
        "#SBATCH --mem=110G\n#SBATCH --time=00:30:00\n"
        "#SBATCH --array=0-3\n#SBATCH --job-name=cch-threshold-canary\n"
        + common
        + "indices=(0 50 100 150)\n"
        + 'index="${indices[$SLURM_ARRAY_TASK_ID]}"\n'
        + "srun --nodes=1 --ntasks=1 --cpus-per-task=72 "
        + '--gpus-per-node=1 --mem=110G "${UV[@]}" '
        + "scripts/cchamber_operating_point_audit.py calibrate-index "
        + '--inventory "$INVENTORY" --inventory-sha256 "$INVENTORY_SHA256" '
        + '--output-root "$OUTPUT_ROOT" --manifest-index "$index"\n'
    )
    freeze = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --account=a0166\n#SBATCH --partition=debug\n"
        "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --mem=16G\n"
        "#SBATCH --time=00:30:00\n#SBATCH --job-name=cch-threshold-freeze\n"
        + common
        + '"${UV[@]}" scripts/cchamber_operating_point_audit.py freeze-thresholds '
        + '--inventory "$INVENTORY" --inventory-sha256 "$INVENTORY_SHA256" '
        + '--output-root "$OUTPUT_ROOT"\n'
    )
    collect_script = (
        "#!/usr/bin/env bash\n"
        "#SBATCH --account=a0166\n#SBATCH --partition=debug\n"
        "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --mem=32G\n"
        "#SBATCH --time=00:30:00\n#SBATCH --job-name=cch-threshold-collect\n"
        + common
        + 'THRESHOLD_MANIFEST="$OUTPUT_ROOT/threshold_manifest.json"\n'
        + 'test -f "$THRESHOLD_MANIFEST"\n'
        + "THRESHOLD_SHA256=$(sha256sum \"$THRESHOLD_MANIFEST\" | awk '{print $1}')\n"
        + '"${UV[@]}" scripts/cchamber_operating_point_audit.py collect '
        + '--inventory "$INVENTORY" --inventory-sha256 "$INVENTORY_SHA256" '
        + '--threshold-manifest "$THRESHOLD_MANIFEST" '
        + '--threshold-manifest-sha256 "$THRESHOLD_SHA256" '
        + '--output-root "$OUTPUT_ROOT"\n'
    )
    workflow = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)\n'
        'calibration_job=$(sbatch --parsable "$SCRIPT_DIR/calibrate_packed.sh")\n'
        'freeze_job=$(sbatch --parsable --dependency="afterok:${calibration_job}" '
        '"$SCRIPT_DIR/freeze_thresholds.sh")\n'
        'evaluation_job=$(sbatch --parsable --dependency="afterok:${freeze_job}" '
        '"$SCRIPT_DIR/evaluate_packed.sh")\n'
        'collect_job=$(sbatch --parsable --dependency="afterok:${evaluation_job}" '
        '"$SCRIPT_DIR/collect.sh")\n'
        'printf "calibration=%s\\nfreeze=%s\\nevaluation=%s\\ncollect=%s\\n" '
        '"$calibration_job" "$freeze_job" "$evaluation_job" "$collect_job"\n'
    )
    content = {
        "debug_four_models.sh": debug,
        "calibrate_packed.sh": packed("calibrate"),
        "freeze_thresholds.sh": freeze,
        "evaluate_packed.sh": packed("evaluate"),
        "collect.sh": collect_script,
        "submit_workflow.sh": workflow,
    }
    for name, value in content.items():
        path = scripts / name
        path.write_text(value, encoding="utf-8")
        path.chmod(0o750)


def _run_tags(
    stage: str,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
    record: Mapping[str, Any],
    **extra: Any,
) -> dict[str, str]:
    """Build the exact immutable MLflow tag set for one sidecar task."""
    tags = {
        "stage": stage,
        "campaign_id": str(inventory["campaign_id"]),
        "manifest_index": str(record["manifest_index"]),
        "model": str(record["model"]),
        "strategy": str(record["strategy"]),
        "seed": str(record["seed"]),
        "candidate_id": str(record["candidate_id"]),
        "campaign_training_commit": CAMPAIGN_TRAINING_COMMIT,
        "audit_code_commit": str(inventory["audit_code_commit"]),
        "audit_code_branch": str(inventory["audit_code_branch"]),
        "inventory_sha256": inventory_sha256,
        "checkpoint_sha256": str(record["checkpoint_sha256"]),
    }
    tags.update({key: str(value) for key, value in extra.items()})
    return tags


def _require_finished_run(
    tracking_uri: str, run_id: str, expected_tags: Mapping[str, str]
) -> None:
    """Require a FINISHED MLflow run with every exact immutable task tag."""
    run = MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run is not FINISHED: {run_id}")
    for key, value in expected_tags.items():
        if run.data.tags.get(key) != str(value):
            raise ValueError(f"MLflow run tag mismatch for {key!r}: {run_id}")


def run_index(
    inventory_path: Path,
    inventory_sha256: str,
    output_root: Path,
    manifest_index: int,
    accelerator: str = "gpu",
    devices: int = 1,
) -> dict[str, Any]:
    """Run or safely resume one outcome-blind test-normal audit task."""
    inventory = load_inventory(inventory_path, inventory_sha256)
    campaign_root = Path(str(inventory["campaign"])).parent
    output_root = _require_external_output(output_root, campaign_root)
    if not 0 <= int(manifest_index) < EXPECTED_RECORDS:
        raise IndexError(f"manifest-index must be in [0, {EXPECTED_RECORDS - 1}].")
    record = dict(inventory["records"][int(manifest_index)])
    _validate_source_record(record, str(inventory["campaign_id"]))
    audit_commit, audit_branch = _audit_revision()
    context = _diagnostic_context(
        inventory_path,
        inventory_sha256,
        inventory,
        record,
        audit_commit,
        audit_branch,
    )

    marker_path = output_root / "records" / f"{int(manifest_index):03d}.json"
    if marker_path.is_file():
        marker = _load_json(marker_path)
        for key, value in context.items():
            if marker.get(key) != value:
                raise ValueError(f"Resume marker identity mismatch for {key!r}: {marker_path}")
        diagnostics = Path(str(marker["diagnostics_csv"]))
        _require_hash(diagnostics, str(marker["diagnostics_csv_sha256"]), "diagnostics")
        resume_context = {
            **context,
            "validation_normal_count": marker["validation_normal_count"],
            "validation_threshold_granularity": marker["validation_threshold_granularity"],
        }
        _validate_diagnostic(diagnostics, resume_context)
        return dict(marker)

    attempt = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"_j{os.environ.get('SLURM_JOB_ID', 'local')}"
        + f"_a{os.environ.get('SLURM_ARRAY_TASK_ID', 'none')}_p{os.getpid()}"
    )
    attempt_dir = output_root / "attempts" / f"{int(manifest_index):03d}" / attempt
    diagnostics = attempt_dir / "operating_point_diagnostics.csv"
    tracking_uri = _tracking_uri(output_root)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"cchamber_{inventory['campaign_id']}_operating_point_audit"
    mlflow.set_experiment(experiment_name)
    tags = {
        "stage": "operating_point_audit",
        "outcome_blind": "true",
        "campaign_id": str(inventory["campaign_id"]),
        "manifest_index": str(manifest_index),
        "model": str(record["model"]),
        "strategy": str(record["strategy"]),
        "seed": str(record["seed"]),
        "campaign_training_commit": CAMPAIGN_TRAINING_COMMIT,
        "audit_code_commit": audit_commit,
        "checkpoint_sha256": str(record["checkpoint_sha256"]),
        "retrain_marker_sha256": str(record["retrain_marker_sha256"]),
        "audit_inventory_sha256": inventory_sha256,
        "test_normal_source_sha256": str(inventory["test_normal_source_sha256"]),
        "threshold_source": "audit_validation.normal",
    }
    with mlflow.start_run(
        run_name=f"operating_point_{int(manifest_index):03d}_{attempt}", tags=tags
    ) as run:
        pl.seed_everything(int(record["seed"]), workers=True)
        _, datamodule, model, validation_normal_loader = _compose_model_and_validation_loader(
            record, str(inventory["data_dir"])
        )
        threshold_callback = _ValidationThresholdCallback()
        callback = AnomalyEfficiencyCallback(
            output_name="ascore/full",
            ds=[],
            target_rates=None,
            base_rate=None,
            pure_thres=False,
            log_raw_mlflow=False,
            name="operating_point",
            operating_point_diagnostics_path=diagnostics,
            operating_point_context=context,
            operating_point_only=True,
            operating_point_threshold_source="audit_validation.normal:thres_operational",
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=False,
            callbacks=[threshold_callback, callback],
            enable_checkpointing=False,
            enable_progress_bar=False,
            inference_mode=True,
            deterministic=True,
        )
        trainer.validate(
            model=model,
            dataloaders={"normal": validation_normal_loader},
            verbose=False,
        )
        context["validation_normal_count"] = threshold_callback.validation_normal_count
        context["validation_threshold_granularity"] = (
            1.0 / threshold_callback.validation_normal_count
        )
        callback.operating_point_context = dict(context)
        datamodule.setup("test")
        test_loaders = datamodule.test_dataloader()
        normal_test_loader = test_loaders["normal"]
        normal_test_loader.loader = datamodule.loader
        trainer.split = "test"
        trainer.test(
            model=model,
            dataloaders={"normal": normal_test_loader},
            verbose=False,
        )
        diagnostic_row = _validate_diagnostic(diagnostics, context)
        mlflow.log_artifact(str(diagnostics), artifact_path="operating_point")
        mlflow.log_params(
            {
                "target_fpr": TARGET_FPR,
                "test_stream": "normal",
                "manifest_index": int(manifest_index),
            }
        )
        mlflow.log_metrics(
            {
                "validation_derived_threshold": float(
                    diagnostic_row["validation_derived_threshold"]
                ),
                "test_normal_count": float(diagnostic_row["test_normal_count"]),
                "triggered_count": float(diagnostic_row["triggered_count"]),
                "achieved_test_normal_acceptance": float(
                    diagnostic_row["achieved_test_normal_acceptance"]
                ),
                "finite_sample_granularity": float(diagnostic_row["finite_sample_granularity"]),
            }
        )
        run_id = run.info.run_id
        datamodule.teardown("test")

    marker = {
        "schema_version": 1,
        **context,
        "attempt_id": attempt,
        "diagnostics_csv": str(diagnostics.resolve()),
        "diagnostics_csv_sha256": _sha256(diagnostics),
        "audit_mlflow_tracking_uri": tracking_uri,
        "audit_mlflow_experiment": experiment_name,
        "audit_mlflow_run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    try:
        _atomic_json_create(marker_path, marker)
    except FileExistsError:
        concurrent = _load_json(marker_path)
        for key, value in context.items():
            if concurrent.get(key) != value:
                raise ValueError(f"Concurrent completion marker identity mismatch: {marker_path}")
        marker = dict(concurrent)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.log_artifact(run_id, str(marker_path), artifact_path="operating_point")
    client.set_tag(run_id, "audit_record_sha256", _sha256(marker_path))
    return marker


def collect(
    inventory_path: Path,
    inventory_sha256: str,
    output_root: Path,
) -> tuple[Path, Path]:
    """Validate and combine exactly 200 immutable operating-point records."""
    inventory = load_inventory(inventory_path, inventory_sha256)
    campaign_root = Path(str(inventory["campaign"])).parent
    output_root = _require_external_output(output_root, campaign_root)
    rows: list[dict[str, str]] = []
    sources: list[dict[str, Any]] = []
    for index, record in enumerate(inventory["records"]):
        _validate_source_record(record, str(inventory["campaign_id"]))
        marker_path = output_root.resolve() / "records" / f"{index:03d}.json"
        marker = _load_json(marker_path)
        if (
            marker.get("manifest_index") != index
            or marker.get("retrain_marker_sha256") != record["retrain_marker_sha256"]
            or marker.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or marker.get("campaign_training_commit") != CAMPAIGN_TRAINING_COMMIT
        ):
            raise ValueError(f"Audit record identity mismatch: {marker_path}")
        diagnostics = Path(str(marker["diagnostics_csv"]))
        _require_hash(diagnostics, str(marker["diagnostics_csv_sha256"]), "diagnostics")
        context = {
            key: marker[key]
            for key in (
                "campaign_id",
                "manifest_index",
                "model",
                "strategy",
                "seed",
                "candidate_id",
                "campaign_training_commit",
                "audit_code_commit",
                "audit_code_branch",
                "campaign_path",
                "campaign_sha256",
                "retrain_manifest_path",
                "retrain_manifest_sha256",
                "retrain_marker_path",
                "retrain_marker_sha256",
                "checkpoint_sha256",
                "audit_inventory_path",
                "audit_inventory_sha256",
                "params_sha256",
                "valid_pair_table_sha256",
                "test_pair_table_sha256",
                "training_mlflow_run_id",
                "test_normal_source_path",
                "test_normal_source_sha256",
                "validation_normal_count",
                "validation_threshold_granularity",
            )
        }
        rows.append(dict(_validate_diagnostic(diagnostics, context)))
        sources.append(
            {
                "manifest_index": index,
                "record": str(marker_path.resolve()),
                "record_sha256": _sha256(marker_path),
                "diagnostics_csv": str(diagnostics.resolve()),
                "diagnostics_csv_sha256": _sha256(diagnostics),
                "audit_mlflow_run_id": marker["audit_mlflow_run_id"],
            }
        )

    identities = {(row["model"], row["strategy"], int(row["seed"])) for row in rows}
    audit_commits = {row["audit_code_commit"] for row in rows}
    campaign = _load_json(Path(str(inventory["campaign"])))
    if (
        len(rows) != EXPECTED_RECORDS
        or identities != _expected_identities(campaign)
        or len(audit_commits) != 1
    ):
        raise ValueError("Collected operating-point coverage is not exact.")
    combined = output_root / "operating_point_diagnostics.csv"
    combined.parent.mkdir(parents=True, exist_ok=True)
    temporary = combined.with_name(f".{combined.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(combined)
    provenance_path = output_root / "operating_point_diagnostics_provenance.json"
    provenance = {
        "schema_version": 1,
        "outcome_blind": True,
        "campaign_training_commit": CAMPAIGN_TRAINING_COMMIT,
        "audit_code_commit": next(iter(audit_commits)),
        "audit_inventory": str(inventory_path.resolve()),
        "audit_inventory_sha256": inventory_sha256,
        "combined_csv": str(combined.resolve()),
        "combined_csv_sha256": _sha256(combined),
        "expected_records": EXPECTED_RECORDS,
        "records": sources,
    }
    if provenance_path.exists():
        existing = _load_json(provenance_path)
        if existing != provenance:
            raise FileExistsError(
                f"Refusing to replace non-matching collection: {provenance_path}"
            )
    else:
        _atomic_json(provenance_path, provenance)
    return combined, provenance_path


def _compose_for_stage(
    record: Mapping[str, Any],
    data_dir: str | Path,
    *,
    stage: str,
    interventions: Sequence[str],
):
    """Compose one frozen model and only the requested data stage."""
    spec = generation.make_experiment_specification(
        dataset=generation.Dataset.CCHAMBER,
        model=generation.Model(str(record["model"])),
        strategy=generation.Strategy(str(record["strategy"])),
        n_trials=1,
        seeds=(int(record["seed"]),),
    )
    overrides = [
        f"experiment={spec.experiment}",
        *spec.fixed_overrides,
        *[
            f"{key}={json.dumps(value, separators=(',', ':'))}"
            for key, value in record["params"].items()
        ],
        f"data.data_dir={json.dumps(str(Path(data_dir).resolve()))}",
        f"data.signal_experiments={json.dumps(list(interventions), separators=(',', ':'))}",
        "data.max_val_batches=-1",
        "logger=none",
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage)
    model = hydra.utils.instantiate(cfg.algorithm)
    return cfg, datamodule, model


def _load_checkpoint_strict(model, checkpoint: Path) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Strictly restore the original checkpoint before any threshold injection."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no Lightning state_dict: {checkpoint}")
    checkpoint_threshold = state.get("thres_operational")
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
    model._ckpt_path = checkpoint.resolve()
    return payload, (
        None
        if checkpoint_threshold is None
        else checkpoint_threshold.detach().float().cpu().reshape(())
    )


def _threshold_artifact_path(output_root: Path, manifest_index: int) -> Path:
    """Return the canonical immutable threshold-artifact path for one index."""
    return output_root.resolve() / "thresholds" / f"{int(manifest_index):03d}.json"


@lru_cache(maxsize=None)
def _load_checkpoint_threshold_identity(
    checkpoint: str, checkpoint_sha256: str
) -> dict[str, Any] | None:
    """Read one already authenticated checkpoint's optional threshold once."""
    path = Path(checkpoint).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no Lightning state_dict: {path}")
    value = state.get("thres_operational")
    return None if value is None else _float32_identity(torch.as_tensor(value).reshape(()))


def _checkpoint_threshold_identity(
    checkpoint: str, checkpoint_sha256: str
) -> dict[str, Any] | None:
    """Authenticate a checkpoint before reusing its cached threshold identity."""
    path = Path(checkpoint).resolve()
    _require_hash(path, checkpoint_sha256, "threshold checkpoint")
    return _load_checkpoint_threshold_identity(str(path), checkpoint_sha256)


def _validate_threshold_artifact(
    path: Path,
    inventory_path: Path,
    inventory: Mapping[str, Any],
    inventory_sha256: str,
    record: Mapping[str, Any],
    *,
    validate_run: bool = True,
) -> dict[str, Any]:
    """Validate one immutable validation-only threshold artifact."""
    artifact = _load_json(path)
    expected_fields = {
        "schema_version",
        "created_at",
        "calibration_only",
        "campaign_id",
        "manifest_index",
        "model",
        "strategy",
        "seed",
        "candidate_id",
        "inventory",
        "inventory_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "audit_code_commit",
        "audit_code_branch",
        "split",
        "dataset",
        "data_seed",
        "sample_count",
        "sample_ids",
        "sample_ids_sha256",
        "normalized_tensor_sha256",
        "pair_table",
        "pair_table_source_1_sha256",
        "pair_table_sha256",
        "quantile",
        "interpolation",
        "comparator",
        "quantile_rank_zero_based",
        "quantile_rank_one_based",
        "threshold_float32",
        "checkpoint_threshold_float32",
        "checkpoint_threshold_present",
        "count_below_threshold",
        "count_equal_threshold",
        "count_above_threshold",
        "triggered_count",
        "achieved_validation_acceptance",
        "tie_count_at_threshold",
        "finite_sample_granularity",
        "audit_mlflow_tracking_uri",
        "audit_mlflow_run_id",
        "audit_mlflow_tags",
    }
    if set(artifact) != expected_fields:
        missing = sorted(expected_fields - set(artifact))
        extra = sorted(set(artifact) - expected_fields)
        raise ValueError(
            f"Threshold artifact schema fields changed: missing={missing}, extra={extra}."
        )
    validation_table = inventory["pairing_inventory"]["validation_table"]
    expected_tags = _run_tags("threshold_calibration", inventory, inventory_sha256, record)
    integer_fields = (
        "schema_version",
        "manifest_index",
        "seed",
        "data_seed",
        "sample_count",
        "quantile_rank_zero_based",
        "quantile_rank_one_based",
    )
    if any(
        isinstance(artifact[field], bool) or not isinstance(artifact[field], int)
        for field in integer_fields
    ):
        raise ValueError("Threshold artifact integer field types are invalid.")
    expected = {
        "schema_version": THRESHOLD_SCHEMA_VERSION,
        "calibration_only": True,
        "campaign_id": inventory["campaign_id"],
        "manifest_index": int(record["manifest_index"]),
        "model": record["model"],
        "strategy": record["strategy"],
        "seed": int(record["seed"]),
        "candidate_id": record["candidate_id"],
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": inventory_sha256,
        "checkpoint": str(Path(record["checkpoint"]).resolve()),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "audit_code_commit": inventory["audit_code_commit"],
        "audit_code_branch": inventory["audit_code_branch"],
        "split": "validate",
        "dataset": "normal",
        "data_seed": 314159,
        "sample_count": EXPECTED_NORMAL_COUNT,
        "pair_table": str(Path(validation_table["path"]).resolve()),
        "pair_table_source_1_sha256": validation_table["source_1_sha256"],
        "pair_table_sha256": validation_table["sha256"],
        "quantile": 0.99,
        "interpolation": "higher",
        "comparator": ">=",
        "quantile_rank_zero_based": 990,
        "quantile_rank_one_based": 991,
        "finite_sample_granularity": 1.0 / EXPECTED_NORMAL_COUNT,
        "audit_mlflow_tracking_uri": inventory["tracking_uri"],
        "audit_mlflow_tags": expected_tags,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"Threshold artifact mismatch for {key!r}: {path}")
    try:
        created_at = datetime.fromisoformat(str(artifact["created_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Threshold artifact created_at is not ISO-8601.") from error
    if created_at.tzinfo is None:
        raise ValueError("Threshold artifact created_at must be timezone-aware.")
    threshold_identity = dict(artifact["threshold_float32"])
    _decode_float32_identity(threshold_identity)
    actual_checkpoint_identity = _checkpoint_threshold_identity(
        str(Path(record["checkpoint"]).resolve()),
        str(record["checkpoint_sha256"]),
    )
    checkpoint_present = artifact["checkpoint_threshold_present"]
    if not isinstance(checkpoint_present, bool) or checkpoint_present != (
        actual_checkpoint_identity is not None
    ):
        raise ValueError("Checkpoint-threshold presence declaration is invalid.")
    if artifact["checkpoint_threshold_float32"] != actual_checkpoint_identity:
        raise ValueError("Checkpoint-threshold float32 identity is invalid.")
    if actual_checkpoint_identity is not None:
        _decode_float32_identity(actual_checkpoint_identity)
        if actual_checkpoint_identity["bytes_sha256"] != threshold_identity["bytes_sha256"]:
            raise ValueError("Checkpoint and calibrated threshold identities differ.")
    ids = artifact.get("sample_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != EXPECTED_NORMAL_COUNT
        or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
        or len(set(ids)) != EXPECTED_NORMAL_COUNT
        or sha256_tensor(torch.tensor(ids, dtype=torch.int64)) != artifact.get("sample_ids_sha256")
    ):
        raise ValueError("Threshold artifact validation sample IDs are invalid.")
    if artifact.get("normalized_tensor_sha256") != validation_table["source_1_sha256"]:
        raise ValueError("Validation tensor fingerprint does not match the primary pair table.")
    count_keys = (
        "count_below_threshold",
        "count_equal_threshold",
        "count_above_threshold",
        "triggered_count",
        "tie_count_at_threshold",
    )
    if any(
        isinstance(artifact[key], bool) or not isinstance(artifact[key], int) or artifact[key] < 0
        for key in count_keys
    ):
        raise ValueError("Threshold rank/tie counts must be non-negative integers.")
    below = artifact["count_below_threshold"]
    equal = artifact["count_equal_threshold"]
    above = artifact["count_above_threshold"]
    triggered = artifact["triggered_count"]
    rank = artifact["quantile_rank_zero_based"]
    achieved = float(artifact["achieved_validation_acceptance"])
    if (
        below + equal + above != EXPECTED_NORMAL_COUNT
        or triggered != equal + above
        or artifact["tie_count_at_threshold"] != equal
        or not (below <= rank < below + equal)
        or not torch.isfinite(torch.tensor(achieved))
        or abs(achieved - triggered / EXPECTED_NORMAL_COUNT) > 1e-15
    ):
        raise ValueError("Threshold rank/tie diagnostics are inconsistent.")
    run_id = artifact.get("audit_mlflow_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Threshold artifact has no calibration MLflow run ID.")
    if validate_run:
        _require_finished_run(
            str(inventory["tracking_uri"]),
            run_id,
            expected_tags,
        )
    return dict(artifact)


def _calibration_payload(
    inventory_path: Path,
    inventory_sha256: str,
    inventory: Mapping[str, Any],
    record: Mapping[str, Any],
    manifest_index: int,
    accelerator: str,
    devices: int,
) -> dict[str, Any]:
    """Run validation.normal and return one threshold payload before persistence."""
    pl.seed_everything(int(record["seed"]), workers=True)
    _, datamodule, model = _compose_for_stage(
        record, inventory["data_dir"], stage="validate", interventions=()
    )
    _, checkpoint_threshold = _load_checkpoint_strict(model, Path(record["checkpoint"]))
    loaders = datamodule.val_dataloader()
    if set(loaders) != {"normal", "reference_normal"}:
        raise ValueError("Validation composition changed before sealed calibration.")
    normal_loader = loaders["normal"]
    normal_loader.loader = datamodule.loader
    callback = _ValidationThresholdCallback()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        callbacks=[callback],
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        inference_mode=True,
    )
    trainer.validate(model=model, dataloaders={"normal": normal_loader}, verbose=False)
    threshold_identity = _float32_identity(model.thres_operational)
    if checkpoint_threshold is not None:
        checkpoint_identity = _float32_identity(checkpoint_threshold)
        if checkpoint_identity["bytes_sha256"] != threshold_identity["bytes_sha256"]:
            raise ValueError(
                "Checkpoint operational threshold disagrees bitwise with validation recalibration."
            )
    else:
        checkpoint_identity = None
    scores = callback.validation_scores
    threshold = _decode_float32_identity(threshold_identity)
    below = int((scores < threshold).sum().item())
    equal = int((scores == threshold).sum().item())
    above = int((scores > threshold).sum().item())
    triggered = equal + above
    rank_zero_based = int(torch.ceil(torch.tensor(0.99 * (scores.numel() - 1))).item())
    artifact = {
        "schema_version": THRESHOLD_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_only": True,
        "campaign_id": inventory["campaign_id"],
        "manifest_index": int(manifest_index),
        "model": record["model"],
        "strategy": record["strategy"],
        "seed": int(record["seed"]),
        "candidate_id": record["candidate_id"],
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": inventory_sha256,
        "checkpoint": str(Path(record["checkpoint"]).resolve()),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "audit_code_commit": inventory["audit_code_commit"],
        "audit_code_branch": inventory["audit_code_branch"],
        "split": "validate",
        "dataset": "normal",
        "data_seed": 314159,
        "sample_count": EXPECTED_NORMAL_COUNT,
        "sample_ids": callback.validation_sample_ids,
        "sample_ids_sha256": callback.validation_sample_ids_sha256,
        "normalized_tensor_sha256": callback.validation_inputs_sha256,
        "pair_table": inventory["pairing_inventory"]["validation_table"]["path"],
        "pair_table_source_1_sha256": inventory["pairing_inventory"]["validation_table"][
            "source_1_sha256"
        ],
        "pair_table_sha256": inventory["pairing_inventory"]["validation_table"]["sha256"],
        "quantile": 0.99,
        "interpolation": "higher",
        "comparator": ">=",
        "quantile_rank_zero_based": rank_zero_based,
        "quantile_rank_one_based": rank_zero_based + 1,
        "threshold_float32": threshold_identity,
        "checkpoint_threshold_float32": checkpoint_identity,
        "checkpoint_threshold_present": checkpoint_identity is not None,
        "count_below_threshold": below,
        "count_equal_threshold": equal,
        "count_above_threshold": above,
        "triggered_count": triggered,
        "achieved_validation_acceptance": triggered / EXPECTED_NORMAL_COUNT,
        "tie_count_at_threshold": equal,
        "finite_sample_granularity": 1.0 / EXPECTED_NORMAL_COUNT,
    }
    datamodule.teardown("validate")
    return artifact


def calibrate_index(
    inventory_path: Path,
    inventory_sha256: str,
    output_root: Path,
    manifest_index: int,
    accelerator: str = "gpu",
    devices: int = 1,
) -> dict[str, Any]:
    """Calibrate and immutably persist one validation.normal-only threshold."""
    inventory = load_inventory(inventory_path, inventory_sha256)
    _require_deployment(inventory)
    _require_slurm_gpu(accelerator)
    campaign_root = Path(str(inventory["campaign"])).parent
    output_root = _require_external_output(output_root, campaign_root)
    if not 0 <= int(manifest_index) < EXPECTED_RECORDS:
        raise IndexError(f"manifest-index must be in [0, {EXPECTED_RECORDS - 1}].")
    record = dict(inventory["records"][int(manifest_index)])
    _validate_source_record(record, str(inventory["campaign_id"]))
    path = _threshold_artifact_path(output_root, manifest_index)
    if path.is_file():
        return _validate_threshold_artifact(
            path, inventory_path, inventory, inventory_sha256, record
        )

    tracking_uri = str(inventory["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{inventory['campaign_id']}_threshold_sidecar")
    tags = _run_tags("threshold_calibration", inventory, inventory_sha256, record)
    with mlflow.start_run(
        run_name=f"threshold_calibrate_{int(manifest_index):03d}", tags=tags
    ) as run:
        artifact = _calibration_payload(
            inventory_path,
            inventory_sha256,
            inventory,
            record,
            manifest_index,
            accelerator,
            devices,
        )
        artifact["audit_mlflow_tracking_uri"] = tracking_uri
        artifact["audit_mlflow_run_id"] = run.info.run_id
        artifact["audit_mlflow_tags"] = tags
        staged = (
            output_root
            / "attempts"
            / "calibration"
            / f"{int(manifest_index):03d}"
            / run.info.run_id
            / "threshold.json"
        )
        _atomic_json(staged, artifact)
        mlflow.log_artifact(str(staged), artifact_path="threshold_calibration")
        mlflow.log_metrics(
            {
                "threshold_float32": artifact["threshold_float32"]["value"],
                "validation_normal_count": artifact["sample_count"],
                "validation_triggered_count": artifact["triggered_count"],
                "validation_acceptance": artifact["achieved_validation_acceptance"],
                "threshold_tie_count": artifact["tie_count_at_threshold"],
            }
        )
        mlflow.log_params(
            {
                "quantile": artifact["quantile"],
                "interpolation": artifact["interpolation"],
                "comparator": artifact["comparator"],
            }
        )
    _publish_json_after_finished_run(path, artifact, tracking_uri, run.info.run_id, tags)
    return _validate_threshold_artifact(path, inventory_path, inventory, inventory_sha256, record)


def freeze_threshold_manifest(
    inventory_path: Path,
    inventory_sha256: str,
    output_root: Path,
) -> tuple[Path, str]:
    """Freeze all 200 calibrated thresholds before any test stream is loaded."""
    inventory = load_inventory(inventory_path, inventory_sha256)
    _require_deployment(inventory)
    campaign_root = Path(str(inventory["campaign"])).parent
    output_root = _require_external_output(output_root, campaign_root)
    records = []
    for index, record in enumerate(inventory["records"]):
        path = _threshold_artifact_path(output_root, index)
        artifact = _validate_threshold_artifact(
            path, inventory_path, inventory, inventory_sha256, record
        )
        records.append(
            {
                "manifest_index": index,
                "threshold_artifact": str(path.resolve()),
                "threshold_artifact_sha256": _sha256(path),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "threshold_bytes_sha256": artifact["threshold_float32"]["bytes_sha256"],
            }
        )
    manifest = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "test_or_intervention_data_loaded_before_freeze": False,
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": inventory_sha256,
        "audit_code_commit": inventory["audit_code_commit"],
        "audit_code_branch": inventory["audit_code_branch"],
        "expected_records": EXPECTED_RECORDS,
        "records": records,
    }
    path = output_root / "threshold_manifest.json"
    if path.exists():
        existing = _load_json(path)
        old = dict(existing)
        new = dict(manifest)
        old.pop("frozen_at", None)
        new.pop("frozen_at", None)
        if old != new:
            raise FileExistsError("Refusing to replace a different threshold manifest.")
    else:
        _atomic_json_create(path, manifest)
    return path, _sha256(path)


def _load_threshold_manifest(
    path: Path,
    expected_sha256: str,
    inventory_sha256: str,
    inventory: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Validate the 200-threshold pre-test freeze gate."""
    _require_hash(path, expected_sha256, "threshold manifest")
    manifest = _load_json(path)
    records = manifest.get("records")
    if (
        manifest.get("test_or_intervention_data_loaded_before_freeze") is not False
        or manifest.get("inventory_sha256") != inventory_sha256
        or (
            inventory is not None
            and (
                manifest.get("audit_code_commit") != inventory["audit_code_commit"]
                or manifest.get("audit_code_branch") != inventory["audit_code_branch"]
            )
        )
        or int(manifest.get("expected_records", -1)) != EXPECTED_RECORDS
        or not isinstance(records, list)
        or len(records) != EXPECTED_RECORDS
    ):
        raise ValueError("Threshold manifest freeze gate is invalid.")
    by_index = {int(row["manifest_index"]): dict(row) for row in records}
    if set(by_index) != set(range(EXPECTED_RECORDS)):
        raise ValueError("Threshold manifest index coverage is not exact.")
    for index, record in by_index.items():
        artifact = Path(str(record["threshold_artifact"])).resolve()
        _require_hash(artifact, str(record["threshold_artifact_sha256"]), "threshold artifact")
        value = _load_json(artifact)
        if (
            int(value.get("manifest_index", -1)) != index
            or value.get("checkpoint_sha256") != record.get("checkpoint_sha256")
            or value.get("threshold_float32", {}).get("bytes_sha256")
            != record.get("threshold_bytes_sha256")
            or (
                inventory is not None
                and (
                    value.get("audit_code_commit") != inventory["audit_code_commit"]
                    or value.get("audit_code_branch") != inventory["audit_code_branch"]
                )
            )
        ):
            raise ValueError("Threshold manifest record does not match its artifact.")
    return dict(manifest), by_index


def apply_threshold_artifact(model, artifact: Mapping[str, Any]) -> str:
    """Inject one validated float32 threshold after strict checkpoint restore."""
    threshold = _decode_float32_identity(artifact["threshold_float32"])
    if "thres_operational" not in dict(model.named_buffers()):
        model.register_buffer("thres_operational", threshold.clone(), persistent=True)
    else:
        model.thres_operational.data.copy_(threshold)
    observed = _float32_identity(model.thres_operational)["bytes_sha256"]
    expected = artifact["threshold_float32"]["bytes_sha256"]
    if observed != expected:
        raise ValueError("Injected threshold bytes changed.")
    return observed


class _ThresholdSafeEvaluationCallback(Callback):
    """Compute AUPRC, efficiency, and normal diagnostics in one sealed pass."""

    def __init__(
        self,
        interventions: Sequence[str],
        results_path: Path,
        diagnostics_path: Path,
        context: Mapping[str, Any],
        validation_sample_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self.interventions = tuple(map(str, interventions))
        self.results_path = results_path
        self.diagnostics_path = diagnostics_path
        self.context = dict(context)
        self.validation_sample_ids = set(map(int, validation_sample_ids))

    def on_test_start(self, trainer, pl_module) -> None:
        """Require exact loader coverage and the frozen threshold bytes."""
        expected = {"normal", *self.interventions}
        if set(trainer.test_dataloaders) != expected:
            raise ValueError("Threshold-safe evaluation loader coverage is not exact.")
        observed = _float32_identity(pl_module.thres_operational)["bytes_sha256"]
        if observed != self.context["threshold_bytes_sha256"]:
            raise ValueError("Model threshold differs from the frozen threshold artifact.")
        self.threshold = float(pl_module.thres_operational.detach().cpu().item())

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Reset all score and test-normal provenance buffers."""
        del trainer, pl_module
        self.scores = {name: [] for name in ("normal", *self.interventions)}
        self.normal_inputs: list[torch.Tensor] = []
        self.normal_sample_ids: list[torch.Tensor] = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Collect finite scores and test-normal identities from one batch."""
        del pl_module, batch_idx
        name = list(trainer.test_dataloaders)[dataloader_idx]
        scores = outputs["ascore/full"].detach().view(-1).float().cpu()
        if not torch.isfinite(scores).all():
            raise ValueError(f"Non-finite sealed scores for {name}.")
        self.scores[name].append(scores)
        if name == "normal":
            if not isinstance(batch, Mapping) or "x" not in batch or "sample_id" not in batch:
                raise ValueError("test.normal batches require x and sample_id provenance.")
            self.normal_inputs.append(batch["x"].detach().float().cpu())
            self.normal_sample_ids.append(batch["sample_id"].detach().long().view(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Write complete intervention metrics and test-normal diagnostics."""
        del trainer
        normal = torch.cat(self.scores["normal"])
        inputs = torch.cat(self.normal_inputs)
        ids = torch.cat(self.normal_sample_ids)
        if (
            normal.numel() != EXPECTED_NORMAL_COUNT
            or ids.numel() != EXPECTED_NORMAL_COUNT
            or torch.unique(ids).numel() != EXPECTED_NORMAL_COUNT
        ):
            raise ValueError("test.normal must contain exactly 1,000 unique samples.")
        overlap = self.validation_sample_ids & set(map(int, ids.tolist()))
        if overlap:
            raise ValueError("validation.normal and test.normal sample IDs overlap.")
        tensor_sha = sha256_tensor(inputs)
        id_sha = sha256_tensor(ids)
        if tensor_sha != self.context["test_pair_table_source_1_sha256"]:
            raise ValueError("test.normal tensor fingerprint differs from the primary pair table.")
        shared = {
            **self.context,
            "test_normal_count": EXPECTED_NORMAL_COUNT,
            "test_normal_sample_ids_sha256": id_sha,
            "test_normal_tensor_sha256": tensor_sha,
        }
        rows: list[dict[str, Any]] = []
        for intervention in self.interventions:
            signal = torch.cat(self.scores[intervention])
            prediction = torch.cat((normal, signal))
            target = torch.cat(
                (
                    torch.zeros(normal.numel(), dtype=torch.long),
                    torch.ones(signal.numel(), dtype=torch.long),
                )
            )
            auprc = float(BinaryAveragePrecision()(prediction, target).item())
            efficiency = float((signal >= self.threshold).float().mean().item())
            for metric, value in (
                ("auprc", auprc),
                ("efficiency_operational", efficiency),
            ):
                rows.append(
                    {
                        **shared,
                        "intervention": intervention,
                        "metric": metric,
                        "value": value,
                    }
                )
        if len(rows) != EXPECTED_INTERVENTIONS * 2:
            raise ValueError("Sealed intervention result coverage is not exact.")
        triggered = int((normal >= self.threshold).sum().item())
        wilson_low, wilson_high = _wilson_interval(triggered, EXPECTED_NORMAL_COUNT)
        diagnostics = {
            **shared,
            "target_fpr": TARGET_FPR,
            "comparator": ">=",
            "triggered_count": triggered,
            "achieved_test_normal_acceptance": triggered / EXPECTED_NORMAL_COUNT,
            "achieved_minus_target_fpr": triggered / EXPECTED_NORMAL_COUNT - TARGET_FPR,
            "wilson_95_ci_low": wilson_low,
            "wilson_95_ci_high": wilson_high,
            "finite_sample_granularity": 1.0 / EXPECTED_NORMAL_COUNT,
            "validation_test_sample_overlap_count": 0,
        }
        _write_csv(self.results_path, rows)
        _write_csv(self.diagnostics_path, [diagnostics])
        observed = _float32_identity(pl_module.thres_operational)["bytes_sha256"]
        if observed != self.context["threshold_bytes_sha256"]:
            raise ValueError("Threshold changed during test evaluation.")


def _validate_final_rows(
    path: Path,
    record: Mapping[str, Any],
    interventions: Sequence[str],
    threshold_record: Mapping[str, Any],
    expected_context: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Validate one exact 58-intervention-by-two-metric result table."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(product(map(str, interventions), ("auprc", "efficiency_operational")))
    actual = {(row["intervention"], row["metric"]) for row in rows}
    if len(rows) != EXPECTED_INTERVENTIONS * 2 or actual != expected:
        raise ValueError("Final threshold-safe result coverage is not exact.")
    for row in rows:
        if (
            int(row["manifest_index"]) != int(record["manifest_index"])
            or row["checkpoint_sha256"] != record["checkpoint_sha256"]
            or row["threshold_artifact_sha256"] != threshold_record["threshold_artifact_sha256"]
            or row["threshold_bytes_sha256"] != threshold_record["threshold_bytes_sha256"]
            or not 0.0 <= float(row["value"]) <= 1.0
        ):
            raise ValueError("Final threshold-safe result identity/value is invalid.")
        if expected_context is not None:
            for key, value in expected_context.items():
                if row.get(key) != str(value):
                    raise ValueError(f"Final result context mismatch for {key!r}.")
        if expected_context is not None and (
            row["threshold_comparator"] != ">="
            or float(row["threshold_value_float32"])
            != float(expected_context["threshold_value_float32"])
            or row["test_normal_tensor_sha256"] != row["test_pair_table_source_1_sha256"]
            or len(row["test_normal_sample_ids_sha256"]) != 64
        ):
            raise ValueError("Final result threshold/test fingerprint contract is invalid.")
    return rows


def _validate_test_diagnostics(
    path: Path,
    record: Mapping[str, Any],
    threshold_record: Mapping[str, Any],
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Validate one test.normal diagnostic against the same frozen provenance."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Expected exactly one test.normal diagnostic row.")
    row = rows[0]
    if (
        int(row["manifest_index"]) != int(record["manifest_index"])
        or row["checkpoint_sha256"] != record["checkpoint_sha256"]
        or row["threshold_artifact_sha256"] != threshold_record["threshold_artifact_sha256"]
        or row["threshold_bytes_sha256"] != threshold_record["threshold_bytes_sha256"]
        or int(row["test_normal_count"]) != EXPECTED_NORMAL_COUNT
        or int(row["validation_test_sample_overlap_count"]) != 0
        or row["comparator"] != ">="
        or float(row["target_fpr"]) != TARGET_FPR
    ):
        raise ValueError("Test.normal diagnostic identity/contract is invalid.")
    if expected_context is not None:
        for key, value in expected_context.items():
            if row.get(key) != str(value):
                raise ValueError(f"Test.normal diagnostic context mismatch for {key!r}.")
    if expected_context is not None and (
        row["threshold_comparator"] != ">="
        or float(row["threshold_value_float32"])
        != float(expected_context["threshold_value_float32"])
        or row["test_normal_tensor_sha256"] != row["test_pair_table_source_1_sha256"]
        or len(row["test_normal_sample_ids_sha256"]) != 64
    ):
        raise ValueError("Test.normal threshold/fingerprint contract is invalid.")
    triggered = int(row["triggered_count"])
    if (
        not 0 <= triggered <= EXPECTED_NORMAL_COUNT
        or abs(float(row["achieved_test_normal_acceptance"]) - triggered / EXPECTED_NORMAL_COUNT)
        > 1e-12
    ):
        raise ValueError("Test.normal diagnostic count arithmetic is invalid.")
    if abs(float(row["finite_sample_granularity"]) - 1.0 / EXPECTED_NORMAL_COUNT) > 1e-12:
        raise ValueError("Test.normal finite-sample granularity is invalid.")
    expected_low, expected_high = _wilson_interval(triggered, EXPECTED_NORMAL_COUNT)
    if (
        abs(float(row["achieved_minus_target_fpr"]) - (triggered / EXPECTED_NORMAL_COUNT - 0.01))
        > 1e-12
        or abs(float(row["wilson_95_ci_low"]) - expected_low) > 1e-12
        or abs(float(row["wilson_95_ci_high"]) - expected_high) > 1e-12
    ):
        raise ValueError("Test.normal Wilson/difference diagnostics are invalid.")
    return row


def _evaluation_context(
    inventory_path: Path,
    inventory_sha256: str,
    inventory: Mapping[str, Any],
    record: Mapping[str, Any],
    threshold_manifest_path: Path,
    threshold_manifest_sha256: str,
    threshold_path: Path,
    threshold_record: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build every immutable identity copied to result and diagnostic rows."""
    return {
        "campaign_id": inventory["campaign_id"],
        "manifest_index": int(record["manifest_index"]),
        "model": record["model"],
        "strategy": record["strategy"],
        "seed": int(record["seed"]),
        "candidate_id": record["candidate_id"],
        "audit_code_commit": inventory["audit_code_commit"],
        "audit_code_branch": inventory["audit_code_branch"],
        "checkpoint": str(Path(record["checkpoint"]).resolve()),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": inventory_sha256,
        "threshold_manifest": str(threshold_manifest_path.resolve()),
        "threshold_manifest_sha256": threshold_manifest_sha256,
        "threshold_artifact": str(threshold_path.resolve()),
        "threshold_artifact_sha256": threshold_record["threshold_artifact_sha256"],
        "threshold_bytes_sha256": threshold_record["threshold_bytes_sha256"],
        "threshold_value_float32": artifact["threshold_float32"]["value"],
        "threshold_comparator": ">=",
        "validation_normal_count": EXPECTED_NORMAL_COUNT,
        "validation_sample_ids_sha256": artifact["sample_ids_sha256"],
        "validation_normal_tensor_sha256": artifact["normalized_tensor_sha256"],
        "validation_pair_table_sha256": inventory["pairing_inventory"]["validation_table"][
            "sha256"
        ],
        "validation_pair_table_source_1_sha256": inventory["pairing_inventory"][
            "validation_table"
        ]["source_1_sha256"],
        "test_pair_table_sha256": inventory["pairing_inventory"]["test_table"]["sha256"],
        "test_pair_table_source_1_sha256": inventory["pairing_inventory"]["test_table"][
            "source_1_sha256"
        ],
    }


def evaluate_index(
    inventory_path: Path,
    inventory_sha256: str,
    threshold_manifest_path: Path,
    threshold_manifest_sha256: str,
    output_root: Path,
    manifest_index: int,
    accelerator: str = "gpu",
    devices: int = 1,
) -> dict[str, Any]:
    """Strict-load, inject a frozen threshold, and run one complete sealed test pass."""
    inventory = load_inventory(inventory_path, inventory_sha256)
    _require_deployment(inventory)
    _require_slurm_gpu(accelerator)
    campaign = _load_json(Path(inventory["campaign"]))
    interventions = tuple(map(str, campaign["interventions"]))
    if len(interventions) != EXPECTED_INTERVENTIONS or len(set(interventions)) != len(
        interventions
    ):
        raise ValueError("Campaign intervention coverage is not exactly 58.")
    _, threshold_records = _load_threshold_manifest(
        threshold_manifest_path, threshold_manifest_sha256, inventory_sha256, inventory
    )
    if not 0 <= int(manifest_index) < EXPECTED_RECORDS:
        raise IndexError(f"manifest-index must be in [0, {EXPECTED_RECORDS - 1}].")
    record = dict(inventory["records"][int(manifest_index)])
    _validate_source_record(record, str(inventory["campaign_id"]))
    frozen = threshold_records[int(manifest_index)]
    threshold_path = Path(frozen["threshold_artifact"])
    _require_hash(threshold_path, frozen["threshold_artifact_sha256"], "threshold artifact")
    artifact = _validate_threshold_artifact(
        threshold_path, inventory_path, inventory, inventory_sha256, record
    )
    context = _evaluation_context(
        inventory_path,
        inventory_sha256,
        inventory,
        record,
        threshold_manifest_path,
        threshold_manifest_sha256,
        threshold_path,
        frozen,
        artifact,
    )
    output_root = _require_external_output(output_root, Path(str(inventory["campaign"])).parent)
    marker_path = output_root / "evaluation" / f"{int(manifest_index):03d}.json"
    run_tags = _run_tags(
        "threshold_evaluation",
        inventory,
        inventory_sha256,
        record,
        threshold_manifest_sha256=threshold_manifest_sha256,
        threshold_artifact_sha256=frozen["threshold_artifact_sha256"],
        threshold_bytes_sha256=frozen["threshold_bytes_sha256"],
    )
    if marker_path.is_file():
        marker = _load_json(marker_path)
        if any(marker.get(key) != value for key, value in context.items()) or (
            marker.get("threshold_manifest_sha256") != threshold_manifest_sha256
            or marker.get("threshold_artifact_sha256") != frozen["threshold_artifact_sha256"]
            or marker.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or int(marker.get("n_rows", -1)) != EXPECTED_INTERVENTIONS * 2
        ):
            raise ValueError("Evaluation resume marker identity changed.")
        _require_finished_run(
            str(inventory["tracking_uri"]),
            str(marker["audit_mlflow_run_id"]),
            run_tags,
        )
        results = Path(marker["results_csv"])
        diagnostics = Path(marker["diagnostics_csv"])
        _require_hash(results, marker["results_csv_sha256"], "threshold-safe results")
        _require_hash(diagnostics, marker["diagnostics_csv_sha256"], "normal diagnostics")
        _validate_final_rows(results, record, interventions, frozen, context)
        diagnostic = _validate_test_diagnostics(diagnostics, record, frozen, context)
        with results.open("r", encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle))
        for key in ("test_normal_sample_ids_sha256", "test_normal_tensor_sha256"):
            if first[key] != diagnostic[key]:
                raise ValueError(f"Resume result/diagnostic mismatch for {key!r}.")
        return dict(marker)

    pl.seed_everything(int(record["seed"]), workers=True)
    _, datamodule, model = _compose_for_stage(
        record, inventory["data_dir"], stage="test", interventions=interventions
    )
    _load_checkpoint_strict(model, Path(record["checkpoint"]))
    threshold_bytes_sha256 = apply_threshold_artifact(model, artifact)
    loaders_all = datamodule.test_dataloader()
    loaders = {"normal": loaders_all["normal"]}
    loaders.update({name: loaders_all[name] for name in interventions})
    loaders["normal"].loader = datamodule.loader
    attempt = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"_j{os.environ.get('SLURM_JOB_ID', 'local')}_p{os.getpid()}"
    )
    attempt_dir = output_root / "attempts" / "evaluation" / f"{manifest_index:03d}" / attempt
    results = attempt_dir / "threshold_safe_results.csv"
    diagnostics = attempt_dir / "test_normal_diagnostics.csv"
    if context["threshold_bytes_sha256"] != threshold_bytes_sha256:
        raise ValueError("Injected threshold differs from the evaluation context.")
    callback = _ThresholdSafeEvaluationCallback(
        interventions, results, diagnostics, context, artifact["sample_ids"]
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        callbacks=[callback],
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        inference_mode=True,
    )
    trainer.split = "test"
    tracking_uri = str(inventory["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{inventory['campaign_id']}_threshold_sidecar")
    run = mlflow.start_run(run_name=f"threshold_evaluate_{int(manifest_index):03d}", tags=run_tags)
    try:
        trainer.test(model=model, dataloaders=loaders, verbose=False)
        rows = _validate_final_rows(results, record, interventions, frozen, context)
        diagnostic = _validate_test_diagnostics(diagnostics, record, frozen, context)
        marker = {
            "schema_version": 1,
            **context,
            "attempt_id": attempt,
            "audit_code_commit": inventory["audit_code_commit"],
            "audit_code_branch": inventory["audit_code_branch"],
            "audit_mlflow_tracking_uri": tracking_uri,
            "audit_mlflow_run_id": run.info.run_id,
            "audit_mlflow_tags": run_tags,
            "results_csv": str(results.resolve()),
            "results_csv_sha256": _sha256(results),
            "diagnostics_csv": str(diagnostics.resolve()),
            "diagnostics_csv_sha256": _sha256(diagnostics),
            "n_rows": len(rows),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        staged_marker = attempt_dir / "evaluation_marker.json"
        _atomic_json(staged_marker, marker)
        mlflow.log_artifact(str(results), artifact_path="threshold_evaluation")
        mlflow.log_artifact(str(diagnostics), artifact_path="threshold_evaluation")
        mlflow.log_artifact(str(staged_marker), artifact_path="threshold_evaluation")
        mlflow.log_metrics(
            {
                "test_normal_count": EXPECTED_NORMAL_COUNT,
                "test_normal_triggered_count": float(diagnostic["triggered_count"]),
                "test_normal_acceptance": float(diagnostic["achieved_test_normal_acceptance"]),
                "test_normal_acceptance_minus_target": float(
                    diagnostic["achieved_minus_target_fpr"]
                ),
                "test_normal_wilson_95_low": float(diagnostic["wilson_95_ci_low"]),
                "test_normal_wilson_95_high": float(diagnostic["wilson_95_ci_high"]),
            }
        )
    except Exception:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")
    datamodule.teardown("test")
    published = _publish_json_after_finished_run(
        marker_path, marker, tracking_uri, run.info.run_id, run_tags
    )
    if published:
        return marker
    concurrent = _load_json(marker_path)
    if (
        any(concurrent.get(key) != value for key, value in context.items())
        or int(concurrent.get("n_rows", -1)) != EXPECTED_INTERVENTIONS * 2
    ):
        raise ValueError("Concurrent evaluation marker identity changed.")
    _require_finished_run(tracking_uri, str(concurrent["audit_mlflow_run_id"]), run_tags)
    concurrent_results = Path(str(concurrent["results_csv"]))
    concurrent_diagnostics = Path(str(concurrent["diagnostics_csv"]))
    _require_hash(
        concurrent_results,
        str(concurrent["results_csv_sha256"]),
        "concurrent threshold-safe results",
    )
    _require_hash(
        concurrent_diagnostics,
        str(concurrent["diagnostics_csv_sha256"]),
        "concurrent normal diagnostics",
    )
    concurrent_rows = _validate_final_rows(
        concurrent_results, record, interventions, frozen, context
    )
    concurrent_diagnostic = _validate_test_diagnostics(
        concurrent_diagnostics, record, frozen, context
    )
    for key in ("test_normal_sample_ids_sha256", "test_normal_tensor_sha256"):
        if concurrent_rows[0][key] != concurrent_diagnostic[key]:
            raise ValueError(f"Concurrent result/diagnostic mismatch for {key!r}.")
    return dict(concurrent)


def collect_threshold_safe(
    inventory_path: Path,
    inventory_sha256: str,
    threshold_manifest_path: Path,
    threshold_manifest_sha256: str,
    output_root: Path,
) -> tuple[Path, Path, Path, Path]:
    """Collect exactly 200 diagnostics and 23,200 threshold-safe result rows."""
    inventory = load_inventory(inventory_path, inventory_sha256)
    _require_deployment(inventory)
    campaign = _load_json(Path(inventory["campaign"]))
    interventions = tuple(map(str, campaign["interventions"]))
    _, threshold_records = _load_threshold_manifest(
        threshold_manifest_path, threshold_manifest_sha256, inventory_sha256, inventory
    )
    output_root = _require_external_output(output_root, Path(str(inventory["campaign"])).parent)
    all_rows: list[dict[str, str]] = []
    diagnostics_rows: list[dict[str, str]] = []
    sources = []
    for index, record in enumerate(inventory["records"]):
        marker_path = output_root / "evaluation" / f"{index:03d}.json"
        marker = _load_json(marker_path)
        frozen = threshold_records[index]
        threshold_path = Path(frozen["threshold_artifact"])
        artifact = _validate_threshold_artifact(
            threshold_path, inventory_path, inventory, inventory_sha256, record
        )
        context = _evaluation_context(
            inventory_path,
            inventory_sha256,
            inventory,
            record,
            threshold_manifest_path,
            threshold_manifest_sha256,
            threshold_path,
            frozen,
            artifact,
        )
        if any(marker.get(key) != value for key, value in context.items()) or (
            int(marker.get("manifest_index", -1)) != index
            or marker.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or marker.get("threshold_manifest_sha256") != threshold_manifest_sha256
            or marker.get("threshold_artifact_sha256") != frozen["threshold_artifact_sha256"]
            or int(marker.get("n_rows", -1)) != EXPECTED_INTERVENTIONS * 2
        ):
            raise ValueError(f"Evaluation marker identity mismatch: {marker_path}")
        _require_finished_run(
            str(inventory["tracking_uri"]),
            str(marker["audit_mlflow_run_id"]),
            _run_tags(
                "threshold_evaluation",
                inventory,
                inventory_sha256,
                record,
                threshold_manifest_sha256=threshold_manifest_sha256,
                threshold_artifact_sha256=frozen["threshold_artifact_sha256"],
                threshold_bytes_sha256=frozen["threshold_bytes_sha256"],
            ),
        )
        results = Path(marker["results_csv"])
        diagnostics = Path(marker["diagnostics_csv"])
        _require_hash(results, marker["results_csv_sha256"], "threshold-safe results")
        _require_hash(diagnostics, marker["diagnostics_csv_sha256"], "normal diagnostics")
        validated_rows = _validate_final_rows(results, record, interventions, frozen, context)
        diagnostic = _validate_test_diagnostics(diagnostics, record, frozen, context)
        for key in ("test_normal_sample_ids_sha256", "test_normal_tensor_sha256"):
            if validated_rows[0][key] != diagnostic[key]:
                raise ValueError(f"Result/diagnostic mismatch for {key!r}.")
        all_rows.extend(validated_rows)
        diagnostics_rows.append(diagnostic)
        sources.append(
            {
                "manifest_index": index,
                "marker": str(marker_path.resolve()),
                "marker_sha256": _sha256(marker_path),
                "results_sha256": _sha256(results),
                "diagnostics_sha256": _sha256(diagnostics),
            }
        )
    if len(all_rows) != EXPECTED_RESULT_ROWS or len(diagnostics_rows) != EXPECTED_RECORDS:
        raise ValueError("Threshold-safe collection counts are not exact.")
    identities = {
        (
            row["model"],
            row["strategy"],
            int(row["seed"]),
            row["intervention"],
            row["metric"],
        )
        for row in all_rows
    }
    if len(identities) != EXPECTED_RESULT_ROWS:
        raise ValueError("Threshold-safe collection Cartesian coverage is not exact.")
    results_path = output_root / "results" / "threshold_safe_results.csv"
    diagnostics_path = output_root / "results" / "test_normal_diagnostics.csv"
    seed_summary_path = output_root / "results" / "seed_level_operating_point.csv"
    _write_csv(results_path, all_rows)
    _write_csv(diagnostics_path, diagnostics_rows)
    seed_rows = [
        {
            "model": row["model"],
            "strategy": row["strategy"],
            "seed": int(row["seed"]),
            "manifest_index": int(row["manifest_index"]),
            "test_normal_count": int(row["test_normal_count"]),
            "triggered_count": int(row["triggered_count"]),
            "achieved_test_normal_acceptance": float(row["achieved_test_normal_acceptance"]),
            "target_fpr": TARGET_FPR,
            "achieved_minus_target_fpr": float(row["achieved_minus_target_fpr"]),
            "wilson_95_ci_low": float(row["wilson_95_ci_low"]),
            "wilson_95_ci_high": float(row["wilson_95_ci_high"]),
        }
        for row in diagnostics_rows
    ]
    if len(seed_rows) != EXPECTED_RECORDS:
        raise ValueError("Seed-level operating-point coverage is not exact.")
    _write_csv(seed_summary_path, seed_rows)
    provenance_path = output_root / "results" / "threshold_safe_provenance.json"
    provenance = {
        "schema_version": 1,
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": inventory_sha256,
        "threshold_manifest": str(threshold_manifest_path.resolve()),
        "threshold_manifest_sha256": threshold_manifest_sha256,
        "results": str(results_path.resolve()),
        "results_sha256": _sha256(results_path),
        "diagnostics": str(diagnostics_path.resolve()),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "seed_level_summary": str(seed_summary_path.resolve()),
        "seed_level_summary_sha256": _sha256(seed_summary_path),
        "seed_level_estimand": (
            "One achieved_test_normal_acceptance - 0.01 estimate per frozen "
            "model/strategy/reporting-seed checkpoint; no event pooling."
        ),
        "expected_records": EXPECTED_RECORDS,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "sources": sources,
    }
    if provenance_path.exists() and _load_json(provenance_path) != provenance:
        raise FileExistsError("Refusing to replace a different threshold-safe collection.")
    if not provenance_path.exists():
        _atomic_json_create(provenance_path, provenance)
    return results_path, diagnostics_path, seed_summary_path, provenance_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse inventory, calibration, threshold-freeze, evaluation, and collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="Freeze all 200 source hashes.")
    inventory.add_argument("--campaign-root", type=Path, required=True)
    inventory.add_argument("--campaign-sha256", required=True)
    inventory.add_argument("--output", type=Path, required=True)

    calibrate = sub.add_parser(
        "calibrate-index", help="Calibrate one validation.normal-only threshold."
    )
    calibrate.add_argument("--inventory", type=Path, required=True)
    calibrate.add_argument("--inventory-sha256", required=True)
    calibrate.add_argument("--output-root", type=Path, required=True)
    calibrate.add_argument("--manifest-index", type=int, required=True)
    calibrate.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    calibrate.add_argument("--devices", type=int, default=1)

    freeze = sub.add_parser(
        "freeze-thresholds", help="Freeze all 200 thresholds before test evaluation."
    )
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--inventory-sha256", required=True)
    freeze.add_argument("--output-root", type=Path, required=True)

    evaluate = sub.add_parser(
        "evaluate-index", help="Evaluate one checkpoint using its frozen threshold."
    )
    evaluate.add_argument("--inventory", type=Path, required=True)
    evaluate.add_argument("--inventory-sha256", required=True)
    evaluate.add_argument("--threshold-manifest", type=Path, required=True)
    evaluate.add_argument("--threshold-manifest-sha256", required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--manifest-index", type=int, required=True)
    evaluate.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    evaluate.add_argument("--devices", type=int, default=1)

    collection = sub.add_parser("collect", help="Combine 200 diagnostics and 23,200 rows.")
    collection.add_argument("--inventory", type=Path, required=True)
    collection.add_argument("--inventory-sha256", required=True)
    collection.add_argument("--threshold-manifest", type=Path, required=True)
    collection.add_argument("--threshold-manifest-sha256", required=True)
    collection.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the operating-point audit CLI."""
    args = parse_args(argv)
    if args.command == "inventory":
        build_inventory(args.campaign_root, args.campaign_sha256, args.output)
        print(args.output.resolve())
        print(_sha256(args.output.resolve()))
    elif args.command == "calibrate-index":
        marker = calibrate_index(
            args.inventory,
            args.inventory_sha256,
            args.output_root,
            args.manifest_index,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        print(_threshold_artifact_path(args.output_root, args.manifest_index))
        print(marker["threshold_float32"]["bytes_sha256"])
    elif args.command == "freeze-thresholds":
        path, digest = freeze_threshold_manifest(
            args.inventory, args.inventory_sha256, args.output_root
        )
        print(path)
        print(digest)
    elif args.command == "evaluate-index":
        marker = evaluate_index(
            args.inventory,
            args.inventory_sha256,
            args.threshold_manifest,
            args.threshold_manifest_sha256,
            args.output_root,
            args.manifest_index,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        print(Path(str(marker["results_csv"])))
    elif args.command == "collect":
        for path in collect_threshold_safe(
            args.inventory,
            args.inventory_sha256,
            args.threshold_manifest,
            args.threshold_manifest_sha256,
            args.output_root,
        ):
            print(path)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
