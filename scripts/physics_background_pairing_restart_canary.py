#!/usr/bin/env python3
"""Isolated Clariden canary for the production exact-epoch restart contract.

This is deliberately not a configurable version of the physics runner.  It trains a
small synthetic model, but attaches the production CAP, Wasserstein, drift, and stable
checkpoint callbacks.  A Slurm ``USR1`` must interrupt the first allocation; the
requeued allocation archives the resulting HPC checkpoint as scheduler evidence, then
restores the latest authenticated end-of-epoch checkpoint. Its terminal parameters
must exactly match an uninterrupted, nonzero-learning-rate reference run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess  # nosec B404 -- fixed git executable and argv only
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mlflow.tracking import MlflowClient
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.callbacks.cap import CAPCallback
from src.callbacks.checkpointing.criterion import Stable
from src.callbacks.efficiency import AnomalyEfficiencyCallback
from src.callbacks.exact_retrain_rng import ExactRetrainRNGState
from src.callbacks.model_checkpoint import SingleDatasetModelCheckpoint
from src.callbacks.thres_drift import ThresholdDriftCallback
from src.callbacks.wasserstein import WassersteinCallback

SCHEMA_VERSION = 2
EXPERIMENT_NAME = "physics_bgpair_restart_canary"
RUN_NAME = "slurm_hpc_resume"
DATASET_NAMES = ("normal", "background0", "background1", "signal")
MAX_EPOCHS = 6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any, *, create: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create and path.exists():
        raise RuntimeError(f"Refusing to replace existing canary evidence: {path}")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(*args: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git")
    return subprocess.run(  # nosec B603 -- resolved git executable and fixed commands
        [git, *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_root(root: Path) -> Path:
    """Keep destructive/stale-state checks outside every scientific campaign."""
    root = root.expanduser().resolve()
    if root.name.startswith("retrain_restart_canary_") is False:
        raise ValueError("Canary root basename must start with 'retrain_restart_canary_'.")
    if "campaigns" in root.parts:
        raise ValueError("Restart canary root must not be inside a campaign directory.")
    return root


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _parameter_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.named_parameters()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _expected_tags(root: Path, commit: str) -> dict[str, str]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "commit": commit,
        "experiment_name": EXPERIMENT_NAME,
        "run_name": RUN_NAME,
    }
    return {
        "mlflow.runName": RUN_NAME,
        "mlflow.source.git.commit": commit,
        "physics_bgpair_restart_canary": "true",
        "physics_bgpair_restart_canary_identity_sha256": _json_sha256(identity),
    }


def _create_or_authenticate_run(
    root: Path, commit: str, persisted_run_id: str | None
) -> tuple[MlflowClient, str, str]:
    tracking_uri = f"file:{(root / 'mlflow' / 'mlruns').resolve()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        client.create_experiment(EXPERIMENT_NAME)
        if experiment is None
        else experiment.experiment_id
    )
    tags = _expected_tags(root, commit)
    matches = client.search_runs(
        [experiment_id],
        filter_string=(
            "tags.physics_bgpair_restart_canary_identity_sha256 = "
            f"'{tags['physics_bgpair_restart_canary_identity_sha256']}'"
        ),
        max_results=2,
    )
    if len(matches) > 1:
        raise RuntimeError("More than one MLflow run claims the canary identity.")
    discovered = None if not matches else matches[0].info.run_id
    if persisted_run_id is not None and discovered != persisted_run_id:
        raise RuntimeError("Persisted canary MLflow run differs from tagged MLflow state.")
    if discovered is None:
        if persisted_run_id is not None:
            raise RuntimeError("Persisted canary MLflow run is missing.")
        discovered = client.create_run(experiment_id, tags=tags).info.run_id
    run = client.get_run(discovered)
    if run.info.experiment_id != experiment_id:
        raise RuntimeError("Canary MLflow run belongs to the wrong experiment.")
    for name, expected in tags.items():
        if run.data.tags.get(name) != expected:
            raise RuntimeError(f"Canary MLflow tag {name!r} is not authenticated.")
    return client, str(discovered), tracking_uri


class _SyntheticModule(LightningModule):
    """A deterministic GPU workload with meaningful parameter updates."""

    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters({"target_rate": 10.0, "base_rate": 40_000.0})
        self.layer = torch.nn.Linear(64, 64, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(64))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        python_scale = random.uniform(0.95, 1.05)  # nosec B311 -- stochastic test oracle
        numpy_shift = float(np.random.uniform(-0.02, 0.02))
        prediction = self.layer(x) * python_scale + numpy_shift + torch.randn_like(x) * 0.01
        loss = prediction.square().mean()
        self.log("canary/train_loss", loss, on_step=False, on_epoch=True)
        # This delay exists only in this isolated canary.  It keeps allocation zero
        # alive until Slurm delivers its configured signal, while resumed work is fast.
        if int(os.environ.get("SLURM_RESTART_COUNT", "0")) == 0:
            time.sleep(12)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        score = (self.layer(x) - 0.25 * x).square().mean(dim=1)
        dataset = DATASET_NAMES[dataloader_idx]
        self.log(
            f"val/{dataset}/anomaly_score",
            score.mean(),
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=x.shape[0],
        )
        return {"score": score}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _RestartProof(Callback):
    """Assert restored production callback state before resumed training mutates it."""

    def __init__(
        self,
        root: Path,
        callbacks: dict[str, Callback],
        expected_states: dict[str, Any] | None,
        restart_count: int,
        client: MlflowClient,
        run_id: str,
    ) -> None:
        super().__init__()
        self.root = root
        self.production_callbacks = callbacks
        self.expected_states = expected_states
        self.restart_count = restart_count
        self.client = client
        self.run_id = run_id
        self.validation_epochs = 0

    @property
    def state_key(self) -> str:
        return "physics-background-pairing-restart-proof"

    def state_dict(self) -> dict[str, Any]:
        return {"validation_epochs": int(self.validation_epochs)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validation_epochs = int(state_dict["validation_epochs"])

    def on_fit_start(self, trainer, pl_module) -> None:
        self.client.log_metric(
            self.run_id,
            "canary/allocation_restart_count",
            float(self.restart_count),
            step=self.restart_count,
        )
        if self.restart_count == 0:
            if self.expected_states is not None:
                raise RuntimeError("Fresh canary allocation unexpectedly has resume state.")
            return
        if self.expected_states is None:
            raise RuntimeError("Requeued canary allocation has no exact-epoch callback state.")

        restored = {
            name: _jsonable(callback.state_dict())
            for name, callback in self.production_callbacks.items()
        }
        if restored != self.expected_states:
            raise RuntimeError("Production callback state differs after exact-epoch restore.")

        cap_state = restored["cap"]
        w1_state = restored["wasserstein"]
        drift_state = restored["drift"]
        stable_state = restored["stable"]
        efficiency_state = restored["efficiency"]
        if cap_state.get("cap_ema") is None or w1_state.get("w1dist_ema") is None:
            raise RuntimeError("CAP/W1 EMA was not populated before preemption.")
        if not drift_state.get("drift_ema"):
            raise RuntimeError("Drift EMA was not populated before preemption.")
        if not efficiency_state.get("cvar25_ema") or not efficiency_state.get("cvar10_ema"):
            raise RuntimeError("Efficiency EMA was not populated before preemption.")
        threshold = getattr(pl_module, "thres_operational", None)
        if threshold is None or not torch.isfinite(threshold):
            raise RuntimeError("Operational threshold buffer was not restored before fit start.")
        records = stable_state.get("checkpoints", {}).get("background0", [])
        window = stable_state.get("ds_criterion", {}).get("background0", {}).get("window", [])
        if not records or not window:
            raise RuntimeError("Stable checkpoint bookkeeping/window was not restored.")
        retained = []
        for record in records:
            path = Path(record["fpath"]).resolve()
            if not path.is_file():
                raise RuntimeError(f"Retained stable checkpoint disappeared: {path}")
            retained.append({"path": str(path), "sha256": _sha256(path)})
        _write_json(
            self.root / "evidence" / f"restore_{self.restart_count:03d}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "restored_at": _utc_now(),
                "restart_count": self.restart_count,
                "callback_states": restored,
                "retained_checkpoints": retained,
            },
            create=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch = int(trainer.current_epoch)
        if epoch != self.validation_epochs:
            raise RuntimeError(
                "Canary epoch sequence is not contiguous: "
                f"observed {epoch}, expected {self.validation_epochs}."
            )
        # This mirrors the production `epoch_idx` contract while making the canary's
        # expected MLflow step explicit. Restored `validation_epochs` makes any
        # repeated or skipped completed epoch fail immediately after a requeue.
        self.client.log_metric(
            self.run_id,
            "epoch_idx",
            float(epoch),
            step=epoch,
        )
        self.validation_epochs += 1


def _load_callback_states(
    checkpoint: Path | None, callbacks: dict[str, Callback]
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = payload.get("callbacks", {})
    expected = {}
    for name, callback in callbacks.items():
        if callback.state_key not in stored:
            raise RuntimeError(f"Checkpoint has no state for {name}: {callback.state_key}")
        expected[name] = _jsonable(stored[callback.state_key])
    return expected


def _scheduler_log_evidence(text: str) -> dict[str, Any]:
    """Validate append-mode evidence for the two allocations and child signal."""
    required = [
        "CANARY_ATTEMPT_START restart=0",
        "CANARY_ATTEMPT_START restart=1",
        "CANARY_LIGHTNING_CHILD restart=0",
        "CANARY_LIGHTNING_CHILD restart=1",
        "Handling auto-requeue signal",
    ]
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise RuntimeError(f"Scheduler log is missing restart evidence: {missing}")
    return {needle: text.count(needle) for needle in required}


def _make_loaders() -> tuple[DataLoader, dict[str, DataLoader]]:
    generator = torch.Generator().manual_seed(123)
    train_x = torch.randn(80, 64, generator=generator)
    normal = torch.linspace(-0.8, 0.8, 32 * 64).reshape(32, 64)
    background0 = torch.linspace(-1.0, 1.0, 32 * 64).reshape(32, 64)
    background1 = background0 * 1.1 + 0.05
    signal = background0 * 2.0 + 1.0
    zeros_train = torch.zeros(len(train_x), dtype=torch.long)
    zeros_val = torch.zeros(len(normal), dtype=torch.long)
    background_labels = -torch.ones(len(background0), dtype=torch.long)
    signal_labels = torch.ones(len(signal), dtype=torch.long)
    train = DataLoader(TensorDataset(train_x, zeros_train), batch_size=16, shuffle=False)
    validation = {
        "normal": DataLoader(TensorDataset(normal, zeros_val), batch_size=32, shuffle=False),
        "background0": DataLoader(
            TensorDataset(background0, background_labels), batch_size=32, shuffle=False
        ),
        "background1": DataLoader(
            TensorDataset(background1, background_labels), batch_size=32, shuffle=False
        ),
        "signal": DataLoader(TensorDataset(signal, signal_labels), batch_size=32, shuffle=False),
    }
    return train, validation


class _CanaryDataModule(LightningDataModule):
    """Synthetic data with the same dedicated train-shuffler contract as L1AD."""

    def __init__(self) -> None:
        super().__init__()
        self.shuffler = torch.Generator().manual_seed(123)

    def train_dataloader(self):
        train, _ = _make_loaders()
        return DataLoader(
            train.dataset,
            batch_size=16,
            shuffle=True,
            generator=self.shuffler,
            num_workers=0,
        )

    def val_dataloader(self):
        _, validation = _make_loaders()
        return validation


def _build_callbacks(root: Path) -> dict[str, Callback]:
    efficiency = AnomalyEfficiencyCallback("score", beta=0.9)
    cap = CAPCallback(
        "score",
        "background0",
        "background1",
        "cdf",
        {
            "beta0": 1.0,
            "normalization_type": "sigmoid",
            "energy_type": "adaptive",
            "energy_params": {"scale": 0.5},
            "regularization_type": "none",
            "binary": True,
            "lr": 0.01,
            "n_epochs": 1,
            "batch_size": 32,
            "normalize_gradients": True,
        },
        metric_name="canary",
        beta=0.9,
    )
    wasserstein = WassersteinCallback(
        "score", "background0", "background1", metric_name="canary", beta=0.9
    )
    drift = ThresholdDriftCallback(
        "score",
        dataset_1="background0",
        dataset_2="background1",
        metric_name="canary",
        beta=0.9,
    )
    stable = SingleDatasetModelCheckpoint(
        monitor="anomaly_score",
        criterion=Stable(top_k=1, threshold=0.05, patience=1),
        ds=["background0"],
        dirpath=str(root / "retained"),
        weights_only=False,
    )
    return {
        "exact_rng": ExactRetrainRNGState(require_datamodule_shuffler=True),
        "efficiency": efficiency,
        "cap": cap,
        "wasserstein": wasserstein,
        "drift": drift,
        "stable": stable,
    }


def _build_exact_epoch_checkpoint(root: Path) -> ModelCheckpoint:
    """Mirror the production last-checkpoint behavior at every completed epoch."""
    # Lightning 2.6 writes ``save_last`` during validation only after a primary
    # checkpoint was saved at the same global step. With ``save_top_k=0`` it waits
    # until ``on_train_end``, which an intentionally preempted allocation never
    # reaches. Production uses the same one-primary-plus-last shape.
    return ModelCheckpoint(
        dirpath=root / "exact_epoch",
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )


def _archive_hpc(root: Path, restart_count: int) -> dict[str, Any] | None:
    """Move mid-epoch state out of Lightning discovery before constructing Trainer."""
    checkpoints = sorted((root / "lightning").glob("hpc_ckpt_*.ckpt"))
    if restart_count == 0:
        if checkpoints:
            raise RuntimeError("Fresh canary allocation found an unauthenticated HPC checkpoint.")
        return None
    if not checkpoints:
        raise RuntimeError("Requeued canary allocation found no Lightning HPC checkpoint.")
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one requeue HPC checkpoint, found {len(checkpoints)}.")
    source = checkpoints[0].resolve()
    archive = root / "evidence" / f"hpc_input_{restart_count:03d}.ckpt"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise RuntimeError(f"HPC evidence already exists: {archive}")
    source.replace(archive)
    if list((root / "lightning").glob("hpc_ckpt_*.ckpt")):
        raise RuntimeError("HPC checkpoint remains discoverable after archival.")
    return {
        "path": str(archive.resolve()),
        "sha256": _sha256(archive),
        "resume_eligible": False,
    }


def _exact_epoch_resume(
    root: Path,
    restart_count: int,
    client: MlflowClient,
    run_id: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Authenticate and freeze the completed-epoch checkpoint used for rollback."""
    checkpoint = root / "exact_epoch" / "last.ckpt"
    history = sorted(client.get_metric_history(run_id, "epoch_idx"), key=lambda row: row.step)
    if restart_count == 0:
        if checkpoint.exists() or history:
            raise RuntimeError("Fresh canary allocation found prior completed-epoch state.")
        return None, None
    if not checkpoint.is_file() or not history:
        raise RuntimeError("Requeued canary has no authenticated completed-epoch state.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed_epochs = int(payload.get("epoch", -1)) + 1
    steps = [int(row.step) for row in history]
    values = [float(row.value) for row in history]
    expected = list(range(completed_epochs))
    if (
        completed_epochs <= 0
        or completed_epochs >= MAX_EPOCHS
        or steps != expected
        or values != [float(epoch) for epoch in expected]
    ):
        raise RuntimeError(
            "Completed-epoch checkpoint differs from contiguous MLflow history: "
            f"checkpoint_epochs={completed_epochs}, steps={steps}, values={values}."
        )
    frozen = root / "evidence" / f"exact_resume_{restart_count:03d}.ckpt"
    if frozen.exists():
        raise RuntimeError(f"Exact resume evidence already exists: {frozen}")
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, frozen)
    return frozen, {
        "path": str(frozen.resolve()),
        "sha256": _sha256(frozen),
        "completed_epochs": completed_epochs,
        "mlflow_epoch_steps": steps,
        "mlflow_epoch_values": values,
    }


def _uninterrupted_reference_hash(root: Path) -> str:
    """Train the same optimization trajectory once without scheduler interruption."""
    seed_everything(123, workers=True)
    model = _SyntheticModule()
    callbacks = _build_callbacks(root / "reference")
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        logger=False,
        callbacks=list(callbacks.values()),
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=_CanaryDataModule())
    return _parameter_sha256(model)


def run(root: Path, scheduler_log: Path, expected_commit: str) -> Path:
    root = _validate_root(root)
    actual_commit = _git("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"Canary commit differs: expected {expected_commit}, got {actual_commit}"
        )
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Restart canary requires a clean immutable execution worktree.")
    if not torch.cuda.is_available():
        raise RuntimeError("Restart canary requires a real CUDA allocation.")
    if os.environ.get("SLURM_JOB_ID") is None:
        raise RuntimeError("Restart canary must run inside Slurm.")

    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    print(
        f"CANARY_LIGHTNING_CHILD restart={restart_count} pid={os.getpid()} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    complete = root / "COMPLETE.json"
    if complete.is_file():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        if payload.get("code_commit") != expected_commit:
            raise RuntimeError("Existing canary completion belongs to another commit.")
        return complete

    metadata_path = root / "state.json"
    persisted = None
    if metadata_path.is_file():
        persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
        if persisted.get("code_commit") != expected_commit:
            raise RuntimeError("Canary state belongs to another commit.")
    elif root.is_dir() and any(root.iterdir()):
        raise RuntimeError("Fresh canary root is not empty and has no authenticated state file.")
    client, run_id, tracking_uri = _create_or_authenticate_run(
        root, expected_commit, None if persisted is None else persisted["mlflow_run_id"]
    )
    if persisted is None:
        persisted = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "code_commit": expected_commit,
            "mlflow_run_id": run_id,
            "tracking_uri": tracking_uri,
        }
        _write_json(metadata_path, persisted, create=True)

    seed_everything(123, workers=True)
    model = _SyntheticModule()
    initial_state_sha256 = _parameter_sha256(model)
    callbacks = _build_callbacks(root)
    hpc_evidence = _archive_hpc(root, restart_count)
    exact_checkpoint, exact_resume_evidence = _exact_epoch_resume(
        root, restart_count, client, run_id
    )
    expected_states = _load_callback_states(exact_checkpoint, callbacks)
    proof = _RestartProof(root, callbacks, expected_states, restart_count, client, run_id)
    exact_checkpointing = _build_exact_epoch_checkpoint(root)
    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        tracking_uri=tracking_uri,
        run_id=run_id,
        tags=_expected_tags(root, expected_commit),
        log_model=False,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[*callbacks.values(), proof, exact_checkpointing],
        logger=logger,
        default_root_dir=str(root / "lightning"),
        deterministic=True,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(
        model,
        datamodule=_CanaryDataModule(),
        ckpt_path=exact_checkpoint,
    )

    # Allocation zero must never finish: doing so would not test a restart.
    if restart_count == 0:
        raise RuntimeError("Training finished without the required Slurm requeue.")
    if int(trainer.current_epoch) != MAX_EPOCHS:
        raise RuntimeError(
            f"Resumed canary stopped at epoch {trainer.current_epoch}, expected {MAX_EPOCHS}."
        )
    final_state_sha256 = _parameter_sha256(model)
    uninterrupted_reference_state_sha256 = _uninterrupted_reference_hash(root)
    parameters_match_reference = final_state_sha256 == uninterrupted_reference_state_sha256
    if not parameters_match_reference:
        raise RuntimeError(
            "Exact-epoch resumed parameters differ from uninterrupted training: "
            f"resumed={final_state_sha256}, reference={uninterrupted_reference_state_sha256}."
        )
    restore_path = root / "evidence" / f"restore_{restart_count:03d}.json"
    if not restore_path.is_file() or hpc_evidence is None or exact_resume_evidence is None:
        raise RuntimeError("Resumed fit did not emit exact-epoch and scheduler evidence.")

    logger.finalize("success")
    allocation_history = client.get_metric_history(run_id, "canary/allocation_restart_count")
    allocation_steps = sorted({int(item.step) for item in allocation_history})
    if 0 not in allocation_steps or not any(step > 0 for step in allocation_steps):
        raise RuntimeError("One MLflow run does not contain both allocation histories.")
    epoch_history = sorted(client.get_metric_history(run_id, "epoch_idx"), key=lambda x: x.step)
    epoch_steps = [int(item.step) for item in epoch_history]
    epoch_values = [float(item.value) for item in epoch_history]
    expected_epochs = list(range(MAX_EPOCHS))
    if epoch_steps != expected_epochs or epoch_values != [float(x) for x in expected_epochs]:
        raise RuntimeError(
            "Resumed MLflow epoch history has a duplicate or gap: "
            f"steps={epoch_steps}, values={epoch_values}."
        )
    if proof.validation_epochs != MAX_EPOCHS:
        raise RuntimeError(
            f"Proof callback observed {proof.validation_epochs} epochs, expected {MAX_EPOCHS}."
        )
    run_record = client.get_run(run_id)
    for name, expected in _expected_tags(root, expected_commit).items():
        if run_record.data.tags.get(name) != expected:
            raise RuntimeError(f"Final MLflow tag {name!r} changed.")

    scheduler_log = scheduler_log.resolve()
    if not scheduler_log.is_file():
        raise RuntimeError(f"Scheduler log does not exist: {scheduler_log}")
    scheduler_evidence = _scheduler_log_evidence(scheduler_log.read_text(encoding="utf-8"))
    scheduler_snapshot = root / "evidence" / "scheduler_log_snapshot.txt"
    if scheduler_snapshot.exists():
        raise RuntimeError(f"Scheduler-log evidence already exists: {scheduler_snapshot}")
    shutil.copy2(scheduler_log, scheduler_snapshot)
    restore = json.loads(restore_path.read_text(encoding="utf-8"))
    retained = restore["retained_checkpoints"]
    for row in retained:
        path = Path(row["path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"Retained stable checkpoint changed after restore: {path}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": _utc_now(),
        "code_commit": expected_commit,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_restart_count": restart_count,
        "cuda_device": torch.cuda.get_device_name(0),
        "hpc_checkpoint_evidence": hpc_evidence,
        "exact_resume_checkpoint": exact_resume_evidence,
        "restore_evidence": {
            "path": str(restore_path),
            "sha256": _sha256(restore_path),
        },
        "retained_checkpoints": retained,
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "allocation_steps": allocation_steps,
            "epoch_steps": epoch_steps,
            "epoch_values": epoch_values,
            "tags": _expected_tags(root, expected_commit),
        },
        "scheduler_log": {
            "source_path": str(scheduler_log),
            "snapshot_path": str(scheduler_snapshot.resolve()),
            "snapshot_sha256": _sha256(scheduler_snapshot),
            "sentinels": scheduler_evidence,
        },
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "uninterrupted_reference_state_sha256": uninterrupted_reference_state_sha256,
        "parameters_match_uninterrupted_reference": parameters_match_reference,
        "final_epoch": int(trainer.current_epoch),
    }
    _write_json(complete, payload, create=True)
    print(f"CANARY_COMPLETE path={complete} sha256={_sha256(complete)}", flush=True)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scheduler-log", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.root, args.scheduler_log, args.expected_commit)


if __name__ == "__main__":
    main()
