"""Outcome-blind callbacks used by the frozen candidate-rank sidecar audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


def _sha256(path: Path) -> str:
    """Hash one file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    """Hash a tensor without changing model or random-generator state."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_hash(module: torch.nn.Module) -> str:
    """Hash a module state dict in deterministic key order."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_hash(value).encode("ascii"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    """Atomically write a strict JSON callback artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class CheckpointBranchManifest(Callback):
    """Freeze the five ModelCheckpoint branch identities after one shared fit."""

    def __init__(self, output_path: str, expected_monitors: dict[str, str]) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.expected_monitors = dict(expected_monitors)

    def on_fit_end(self, trainer, pl_module) -> None:
        """Write paths, hashes, selected epochs, and monitor values for all branches."""
        del pl_module
        checkpoints = [
            callback
            for callback in trainer.callbacks
            if isinstance(callback, ModelCheckpoint) and callback.monitor is not None
        ]
        if len(checkpoints) != len(self.expected_monitors):
            raise ValueError(
                f"Expected {len(self.expected_monitors)} monitored checkpoint callbacks, "
                f"found {len(checkpoints)}."
            )
        by_monitor = {str(callback.monitor): callback for callback in checkpoints}
        if set(by_monitor) != set(self.expected_monitors.values()):
            raise ValueError(
                "Checkpoint monitor contract mismatch: "
                f"{sorted(by_monitor)} != {sorted(self.expected_monitors.values())}."
            )
        branches = []
        for strategy, monitor in self.expected_monitors.items():
            callback = by_monitor[monitor]
            checkpoint = Path(callback.best_model_path).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            epoch = int(payload["epoch"])
            score = callback.best_model_score
            if score is None or not torch.isfinite(score):
                raise ValueError(f"Non-finite checkpoint monitor for {strategy}.")
            branches.append(
                {
                    "strategy": strategy,
                    "monitor": monitor,
                    "monitor_value": float(score.detach().cpu().item()),
                    "selected_epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            )
        _atomic_json(
            self.output_path,
            {
                "schema_version": 1,
                "tie_rule": "earliest epoch for equal monitor values",
                "branches": branches,
            },
        )


class TrajectoryFingerprint(Callback):
    """Record state, metric, RNG, initialization, and minibatch-order fingerprints."""

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.train_batch_hashes: list[str] = []
        self.epochs: list[dict[str, Any]] = []

    @staticmethod
    def _rng() -> dict[str, Any]:
        """Hash CPU and per-device CUDA random-generator states."""
        value: dict[str, Any] = {"cpu": _tensor_hash(torch.random.get_rng_state())}
        value["cuda"] = (
            [_tensor_hash(state) for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )
        return value

    def on_fit_start(self, trainer, pl_module) -> None:
        """Record initialization and initial random-generator identities."""
        del trainer
        self.initial_model_state_sha256 = _state_hash(pl_module)
        self.initial_rng = self._rng()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        """Fingerprint the realized minibatch sequence without retaining examples."""
        del trainer, pl_module, batch_idx
        if isinstance(batch, dict):
            tensor = batch["x"]
        elif isinstance(batch, (list, tuple)):
            tensor = batch[0]
        else:
            raise TypeError(f"Unsupported fingerprint batch type: {type(batch)}")
        self.train_batch_hashes.append(_tensor_hash(tensor))

    def on_validation_end(self, trainer, pl_module) -> None:
        """Record state, RNG, and finite scalar metrics after each validation."""
        metrics = {}
        for name, value in sorted(trainer.callback_metrics.items()):
            if torch.is_tensor(value) and value.numel() == 1 and torch.isfinite(value):
                metrics[str(name)] = float(value.detach().cpu().item())
            elif isinstance(value, (int, float)):
                metrics[str(name)] = float(value)
        self.epochs.append(
            {
                "epoch": int(trainer.current_epoch),
                "model_state_sha256": _state_hash(pl_module),
                "rng": self._rng(),
                "metrics": metrics,
            }
        )

    def on_fit_end(self, trainer, pl_module) -> None:
        """Write the complete deterministic trajectory fingerprint."""
        del trainer
        _atomic_json(
            self.output_path,
            {
                "schema_version": 1,
                "initial_model_state_sha256": self.initial_model_state_sha256,
                "initial_rng": self.initial_rng,
                "train_batch_sha256": self.train_batch_hashes,
                "epochs": self.epochs,
                "final_model_state_sha256": _state_hash(pl_module),
                "final_rng": self._rng(),
            },
        )
