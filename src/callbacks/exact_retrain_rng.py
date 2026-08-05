"""Exact random-state continuation for segmented retraining."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer


class ExactRetrainRNGState(Callback):
    """Checkpoint and restore every RNG stream used by physics retraining.

    Lightning restores model, optimizer, scheduler, loop, and callback state, but it does not
    checkpoint process RNG state. The L1 iterable training dataset also owns a dedicated datamodule
    shuffler generator. Restoring these states at fit start (after process/model setup and before
    the first iterator is constructed) makes a segmented run follow the same stochastic trajectory
    as one uninterrupted fit.
    """

    STATE_SCHEMA_VERSION = 1

    def __init__(self, require_datamodule_shuffler: bool = True) -> None:
        super().__init__()
        self.require_datamodule_shuffler = bool(require_datamodule_shuffler)
        self._trainer: Trainer | None = None
        self._pending_state: dict[str, Any] | None = None

    @property
    def state_key(self) -> str:
        return "physics-background-pairing-exact-retrain-rng-v1"

    def setup(self, trainer: Trainer, pl_module, stage: str) -> None:
        self._trainer = trainer

    def _shuffler(self, trainer: Trainer) -> torch.Generator | None:
        shuffler = getattr(trainer.datamodule, "shuffler", None)
        if shuffler is not None and not isinstance(shuffler, torch.Generator):
            raise RuntimeError("datamodule.shuffler is not a torch.Generator.")
        if shuffler is None and self.require_datamodule_shuffler:
            raise RuntimeError("Exact retraining requires datamodule.shuffler.")
        return shuffler

    def state_dict(self) -> dict[str, Any]:
        if self._trainer is None:
            raise RuntimeError("Exact RNG callback was checkpointed before Trainer setup.")
        shuffler = self._shuffler(self._trainer)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        if self._pending_state is not None:
            python_state = self._pending_state["python"]
            pending_numpy = self._pending_state["numpy"]
            numpy_state = (
                str(pending_numpy["algorithm"]),
                pending_numpy["keys"].cpu().numpy().astype(np.uint32, copy=False),
                int(pending_numpy["position"]),
                int(pending_numpy["has_gauss"]),
                float(pending_numpy["cached_gaussian"]),
            )
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "python": python_state,
            "numpy": {
                "algorithm": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
            "datamodule_shuffler": None if shuffler is None else shuffler.get_state().clone(),
        }

    @classmethod
    def validate_state(
        cls,
        state_dict: dict[str, Any],
        *,
        require_datamodule_shuffler: bool,
        expected_cuda_states: int,
    ) -> None:
        """Fail closed unless a checkpoint contains a complete safe RNG snapshot."""
        required = {
            "schema_version",
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "datamodule_shuffler",
        }
        python_state = state_dict.get("python") if isinstance(state_dict, dict) else None
        torch_cpu = state_dict.get("torch_cpu") if isinstance(state_dict, dict) else None
        cuda_states = state_dict.get("torch_cuda") if isinstance(state_dict, dict) else None
        if (
            not isinstance(state_dict, dict)
            or set(state_dict) != required
            or state_dict.get("schema_version") != cls.STATE_SCHEMA_VERSION
            or not isinstance(python_state, tuple)
            or len(python_state) != 3
            or python_state[0] != 3
            or not isinstance(python_state[1], tuple)
            or len(python_state[1]) != 625
            or not all(type(value) is int for value in python_state[1])
            or not 0 <= python_state[1][-1] <= 624
            or (
                python_state[2] is not None
                and (not isinstance(python_state[2], float) or not np.isfinite(python_state[2]))
            )
            or not isinstance(torch_cpu, torch.Tensor)
            or torch_cpu.dtype != torch.uint8
            or torch_cpu.ndim != 1
            or torch_cpu.numel() == 0
            or not isinstance(cuda_states, list)
            or len(cuda_states) != expected_cuda_states
            or not all(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.ndim == 1
                and value.numel() > 0
                for value in cuda_states
            )
        ):
            raise RuntimeError("Checkpoint contains invalid exact retrain RNG state.")
        numpy_state = state_dict.get("numpy")
        if (
            not isinstance(numpy_state, dict)
            or set(numpy_state)
            != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
            or numpy_state.get("algorithm") != "MT19937"
            or not isinstance(numpy_state.get("keys"), torch.Tensor)
            or numpy_state["keys"].dtype != torch.uint32
            or tuple(numpy_state["keys"].shape) != (624,)
            or type(numpy_state.get("position")) is not int
            or not 0 <= numpy_state["position"] <= 624
            or type(numpy_state.get("has_gauss")) is not int
            or numpy_state["has_gauss"] not in {0, 1}
            or not isinstance(numpy_state.get("cached_gaussian"), float)
            or not np.isfinite(numpy_state["cached_gaussian"])
        ):
            raise RuntimeError("Checkpoint contains invalid NumPy RNG state.")
        shuffler_state = state_dict.get("datamodule_shuffler")
        if shuffler_state is not None and (
            not isinstance(shuffler_state, torch.Tensor)
            or shuffler_state.dtype != torch.uint8
            or shuffler_state.ndim != 1
            or shuffler_state.numel() == 0
        ):
            raise RuntimeError("Checkpoint contains invalid datamodule shuffler state.")
        if require_datamodule_shuffler and shuffler_state is None:
            raise RuntimeError("Checkpoint contains no L1 datamodule shuffler state.")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state(
            state_dict,
            require_datamodule_shuffler=self.require_datamodule_shuffler,
            expected_cuda_states=(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        )
        self._pending_state = state_dict

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        if int(trainer.world_size) != 1:
            raise RuntimeError("Exact retrain RNG continuation currently requires world_size=1.")
        if self._pending_state is None:
            if trainer.ckpt_path is not None:
                raise RuntimeError("Resume checkpoint contains no exact retrain RNG state.")
            # Validate the required shuffler even on the first segment.
            self._shuffler(trainer)
            return
        state = self._pending_state
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_states = state["torch_cuda"]
        if torch.cuda.is_available():
            if len(cuda_states) != torch.cuda.device_count():
                raise RuntimeError(
                    "Checkpoint CUDA RNG count differs from visible CUDA device count."
                )
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
        elif cuda_states:
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        shuffler = self._shuffler(trainer)
        if shuffler is not None:
            shuffler.set_state(state["datamodule_shuffler"].cpu())

    def on_train_batch_start(
        self, trainer: Trainer, pl_module, batch: Any, batch_idx: int
    ) -> None:
        """Restore host RNGs immediately before the first resumed training step."""
        if self._pending_state is None:
            return
        state = self._pending_state
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        self._pending_state = None
