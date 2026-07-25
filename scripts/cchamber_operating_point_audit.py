#!/usr/bin/env python3
"""Outcome-blind operating-point audit for the frozen Causal Chamber campaign.

The audit is a sidecar: it never writes into the campaign directory and never
reads intervention outcomes.  ``inventory`` freezes the exact campaign,
retraining-marker, manifest, and checkpoint hashes.  Array tasks then calibrate
on ``validation.normal`` and evaluate only ``test.normal``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
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

from scripts import generation
from src.callbacks.metrics.rate import AnomalyRate as ValidationAnomalyRate
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback
from src.utils.pairing.io import compose_config

CAMPAIGN_TRAINING_COMMIT = "63b941a287c48c84e2537d0cfbd07c2240435c0e"
TARGET_FPR = 0.01
EXPECTED_RECORDS = 200
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
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        scores = outputs[self.output_name].detach().view(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("Threshold calibration requires finite validation scores.")
        self.scores.append(scores)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Set the higher-quantile operational buffer used by training calibration."""
        del trainer
        if not self.scores:
            raise ValueError("Threshold calibration received no validation-normal scores.")
        scores = torch.cat(self.scores)
        rate = ValidationAnomalyRate(TARGET_FPR, None).to(pl_module.device)
        rate.set_threshold(scores)
        threshold = rate.threshold.detach().clone()
        if "thres_operational" not in dict(pl_module.named_buffers()):
            pl_module.register_buffer("thres_operational", threshold, persistent=True)
        else:
            pl_module.thres_operational.data.copy_(threshold)
        self.validation_normal_count = int(scores.numel())


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


def build_inventory(
    campaign_root: Path,
    campaign_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Freeze all 200 label-free source identities and hashes."""
    campaign_root = campaign_root.resolve()
    output = _require_external_output(output, campaign_root)
    campaign_path = campaign_root / "campaign.json"
    campaign = _validate_campaign(campaign_path, campaign_sha256, full_data_tree=True)
    normal_source, normal_source_sha256 = _validate_dataset_sources(campaign, full_tree=False)
    manifest_path = campaign_root / "selection" / "retrain_manifest.json"
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
        "test_normal_source": str(normal_source),
        "test_normal_source_sha256": normal_source_sha256,
        "data_dir": str(normal_source.parent.parent),
        "retrain_manifest": str(manifest_path.resolve()),
        "retrain_manifest_sha256": _sha256(manifest_path),
        "expected_records": EXPECTED_RECORDS,
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
        return dict(existing)
    _atomic_json(output, inventory)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse inventory, one-index execution, and collection commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="Freeze all 200 source hashes.")
    inventory.add_argument("--campaign-root", type=Path, required=True)
    inventory.add_argument("--campaign-sha256", required=True)
    inventory.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run-index", help="Audit one exact manifest index.")
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--inventory-sha256", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--manifest-index", type=int, required=True)
    run.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    run.add_argument("--devices", type=int, default=1)

    collection = sub.add_parser("collect", help="Validate and combine all 200 rows.")
    collection.add_argument("--inventory", type=Path, required=True)
    collection.add_argument("--inventory-sha256", required=True)
    collection.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the operating-point audit CLI."""
    args = parse_args(argv)
    if args.command == "inventory":
        build_inventory(args.campaign_root, args.campaign_sha256, args.output)
        print(args.output.resolve())
        print(_sha256(args.output.resolve()))
    elif args.command == "run-index":
        marker = run_index(
            args.inventory,
            args.inventory_sha256,
            args.output_root,
            args.manifest_index,
            accelerator=args.accelerator,
            devices=args.devices,
        )
        print(Path(str(marker["diagnostics_csv"])))
    elif args.command == "collect":
        for path in collect(args.inventory, args.inventory_sha256, args.output_root):
            print(path)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
