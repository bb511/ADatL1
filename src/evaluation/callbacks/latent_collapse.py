"""Latent-code collapse diagnostics for the frozen Pareto validation protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from src.data.utils import unpack_batch


class LatentCollapseDiagnosticsCallback(Callback):
    """Measure diversity of deterministic hard Bernoulli codes on normal validation.

    The callback is deliberately checkpoint-specific: it only runs for the selected
    ``loss_total.ckpt`` and records the hard ``latent_sample`` consumed by the decoder.
    It does not use continuous latent logits as a collapse proxy.
    """

    def __init__(
        self,
        *,
        protocol_version: str,
        configuration_id: str,
        autoencoder_seed: int,
        architecture_id: str,
        minimum_joint_code_entropy_bits: float,
        minimum_fraction_of_paired_gamma_zero_joint_entropy: float,
        dataset: str = "normal",
        evaluation_split: str = "val",
        source_split: str = "valid",
        ckpts: dict | None = None,
        name: str = "latent_collapse",
    ) -> None:
        super().__init__()

        if not protocol_version:
            raise ValueError("protocol_version must be a non-empty string.")
        if not configuration_id:
            raise ValueError("configuration_id must be a non-empty string.")
        if not architecture_id:
            raise ValueError("architecture_id must be a non-empty string.")
        if isinstance(autoencoder_seed, bool) or not isinstance(
            autoencoder_seed,
            Integral,
        ):
            raise ValueError("autoencoder_seed must be an integer.")
        if float(minimum_joint_code_entropy_bits) < 0.0:
            raise ValueError("minimum_joint_code_entropy_bits must be non-negative.")
        if not 0.0 <= float(
            minimum_fraction_of_paired_gamma_zero_joint_entropy
        ) <= 1.0:
            raise ValueError(
                "minimum_fraction_of_paired_gamma_zero_joint_entropy must be "
                "in [0, 1]."
            )

        self.protocol_version = protocol_version
        self.configuration_id = configuration_id
        self.autoencoder_seed = int(autoencoder_seed)
        self.architecture_id = architecture_id
        self.minimum_joint_code_entropy_bits = float(
            minimum_joint_code_entropy_bits
        )
        self.minimum_fraction_of_paired_gamma_zero_joint_entropy = float(
            minimum_fraction_of_paired_gamma_zero_joint_entropy
        )
        self.dataset = dataset
        self.evaluation_split = evaluation_split
        self.source_split = source_split
        self.ckpts = ckpts or {"loss_total": True}
        self.name = name
        self._active = False

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Initialise streaming entropy and provenance state for one checkpoint."""
        self._active = (
            str(getattr(trainer, "split", "")) == self.evaluation_split
            and self._should_run_for_current_ckpt(trainer)
        )
        if not self._active:
            return
        if bool(pl_module.training):
            raise RuntimeError(
                "Latent-collapse diagnostics require evaluation mode."
            )

        self._bit_sums: np.ndarray | None = None
        self._latent_width: int | None = None
        self._code_counts: Counter[bytes] = Counter()
        self._n_events = 0
        self._verified_determinism = False
        self._event_component_manifests: dict[str, Any] = {}
        self._event_layouts: dict[str, tuple[Any, ...]] = {}

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate code counts and validate deterministic binary sampling."""
        if not self._active:
            return

        dataset = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dataset != self.dataset:
            return

        batch_view = unpack_batch(batch)
        identity_data = (
            batch_view.control_x
            if batch_view.control_x is not None
            else batch_view.x
        )
        identity_mask = (
            batch_view.control_mask
            if batch_view.control_mask is not None
            else batch_view.mask
        )
        self._update_event_manifest("cached_data", identity_data)
        self._update_event_manifest("cached_mask", identity_mask)
        self._update_event_manifest("l1bit", batch_view.l1bit)

        model_input = torch.flatten(batch_view.x, start_dim=1)
        with torch.inference_mode():
            representations = pl_module.forward_with_representations(model_input)
        latent_sample = self._binary_sample(
            representations,
            expected_batch_size=model_input.shape[0],
        )

        if not self._verified_determinism:
            with torch.inference_mode():
                repeated = pl_module.forward_with_representations(model_input)
            repeated_sample = self._binary_sample(
                repeated,
                expected_batch_size=model_input.shape[0],
            )
            if not torch.equal(latent_sample, repeated_sample):
                raise RuntimeError(
                    "Evaluation-time latent_sample is not deterministic on the "
                    "first normal validation batch."
                )
            self._verified_determinism = True

        self._accumulate_latent_sample(latent_sample)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Persist a finite, machine-readable collapse decision."""
        if not self._active:
            return
        if self._n_events == 0:
            raise RuntimeError(
                "No normal validation events were available for latent-collapse "
                "diagnostics."
            )
        if not self._verified_determinism:
            raise RuntimeError(
                "Latent-collapse diagnostics did not verify deterministic sampling."
            )

        ckpt_path = Path(pl_module._ckpt_path)
        ckpt_name = ckpt_path.stem
        output_folder = (
            ckpt_path.parent
            / "plots"
            / str(trainer.split)
            / ckpt_name
            / self.name
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        self._write_summary(
            output_folder / "collapse_summary.json",
            checkpoint=self._checkpoint_identity(ckpt_path),
            trainer=trainer,
            pl_module=pl_module,
        )

    def _accumulate_latent_sample(self, latent_sample: torch.Tensor) -> None:
        """Update entropy sufficient statistics without retaining all event codes."""
        values = latent_sample.detach().to(device="cpu", dtype=torch.uint8).numpy()
        n_events, latent_width = values.shape

        if self._latent_width is None:
            self._latent_width = int(latent_width)
            self._bit_sums = np.zeros(latent_width, dtype=np.int64)
        elif latent_width != self._latent_width:
            raise RuntimeError(
                "latent_sample width changed between normal validation batches."
            )

        assert self._bit_sums is not None
        self._bit_sums += values.sum(axis=0, dtype=np.int64)
        self._n_events += int(n_events)

        packed_codes = np.packbits(values, axis=1, bitorder="little")
        unique_codes, counts = np.unique(packed_codes, axis=0, return_counts=True)
        for code, count in zip(unique_codes, counts):
            self._code_counts[code.tobytes()] += int(count)

    @staticmethod
    def _binary_sample(
        representations: Any,
        *,
        expected_batch_size: int,
    ) -> torch.Tensor:
        if not isinstance(representations, Mapping):
            raise RuntimeError("forward_with_representations must return a mapping.")
        if "latent_sample" not in representations:
            raise RuntimeError(
                "forward_with_representations did not return latent_sample."
            )

        sample = representations["latent_sample"]
        if not isinstance(sample, torch.Tensor):
            raise RuntimeError("latent_sample must be a torch.Tensor.")
        if sample.ndim == 0:
            raise RuntimeError("latent_sample cannot be a scalar.")
        if sample.ndim == 1:
            sample = sample.unsqueeze(1)
        else:
            sample = torch.flatten(sample, start_dim=1)

        if sample.shape[0] != expected_batch_size:
            raise RuntimeError(
                "latent_sample and normal-validation input have different event "
                "counts."
            )
        if sample.shape[1] < 1:
            raise RuntimeError("latent_sample must contain at least one bit.")
        if not bool(torch.isfinite(sample).all().item()):
            raise RuntimeError("latent_sample contains NaN or infinity.")
        if not bool(torch.all((sample == 0) | (sample == 1)).item()):
            raise RuntimeError(
                "Evaluation-time latent_sample must contain only hard zero/one codes."
            )
        return sample

    def _update_event_manifest(
        self,
        name: str,
        tensor: torch.Tensor | None,
    ) -> None:
        """Hash cached event content independent of evaluation batch boundaries."""
        manifest = self._event_component_manifests.setdefault(
            name,
            hashlib.sha256(),
        )
        if tensor is None:
            layout = ("none",)
            values = None
        else:
            values = tensor.detach().cpu().contiguous().numpy()
            layout = (str(values.dtype), *values.shape[1:])

        previous_layout = self._event_layouts.get(name)
        if previous_layout is not None and previous_layout != layout:
            raise RuntimeError(
                f"Event identity component {name!r} changed layout."
            )
        if previous_layout is None:
            self._event_layouts[name] = layout
            if values is None:
                manifest.update(b"none")
            else:
                manifest.update(str(values.dtype).encode("ascii"))
                manifest.update(
                    np.asarray(values.shape[1:], dtype="<i8").tobytes()
                )

        if values is not None:
            manifest.update(values.tobytes(order="C"))

    def _event_manifest(self) -> tuple[str, dict[str, str]]:
        """Combine stable component digests into the validation-event identity."""
        if not self._event_component_manifests:
            raise RuntimeError("The validation-event manifest has no components.")

        manifest = hashlib.sha256()
        manifest.update(b"latent-collapse-event-manifest-v1\0")
        manifest.update(self.source_split.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(np.asarray([self._n_events], dtype="<i8").tobytes())
        component_hashes: dict[str, str] = {}
        for name in sorted(self._event_component_manifests):
            component_digest = self._event_component_manifests[name].digest()
            component_hashes[name] = component_digest.hex()
            manifest.update(name.encode("utf-8"))
            manifest.update(b"\0")
            manifest.update(component_digest)

        return manifest.hexdigest(), component_hashes

    @staticmethod
    def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
        """Return exact binary entropies in bits, including zero for p=0 or p=1."""
        entropy = np.zeros_like(probabilities, dtype=np.float64)
        nonzero = probabilities > 0.0
        below_one = probabilities < 1.0
        entropy[nonzero] -= probabilities[nonzero] * np.log2(probabilities[nonzero])
        entropy[below_one] -= (
            (1.0 - probabilities[below_one])
            * np.log2(1.0 - probabilities[below_one])
        )
        return entropy

    def _metrics(self) -> dict[str, Any]:
        assert self._bit_sums is not None
        assert self._latent_width is not None

        probabilities = self._bit_sums.astype(np.float64) / self._n_events
        bit_entropies = self._binary_entropy(probabilities)
        counts = np.fromiter(self._code_counts.values(), dtype=np.float64)
        code_probabilities = counts / self._n_events
        joint_entropy = float(-np.sum(code_probabilities * np.log2(code_probabilities)))

        return {
            "latent_width": self._latent_width,
            "bits": [
                {
                    "index": index,
                    "activation_probability": float(probabilities[index]),
                    "binary_entropy_bits": float(bit_entropies[index]),
                }
                for index in range(self._latent_width)
            ],
            "summed_marginal_bit_entropy_bits": float(bit_entropies.sum()),
            "joint_code_entropy_bits": joint_entropy,
            "observed_code_count": int(len(self._code_counts)),
            "effective_code_count": float(math.exp2(joint_entropy)),
        }

    def _write_summary(
        self,
        output_path: Path,
        *,
        checkpoint: dict[str, Any],
        trainer: Any,
        pl_module: Any,
    ) -> None:
        metrics = self._metrics()
        event_manifest_hash, event_manifest_components = self._event_manifest()
        absolute_pass = (
            metrics["joint_code_entropy_bits"]
            >= self.minimum_joint_code_entropy_bits
        )
        bernoulli_threshold = self._bernoulli_threshold(pl_module)
        payload = {
            "schema_version": 1,
            "protocol_version": self.protocol_version,
            "run": {
                "configuration_id": self.configuration_id,
                "autoencoder_seed": self.autoencoder_seed,
                "architecture_id": self.architecture_id,
            },
            "checkpoint": checkpoint,
            "evaluation": {
                "mode": "validation",
                "trainer_split": str(trainer.split),
                "dataset": self.dataset,
                "source_split": self.source_split,
                "n_events": self._n_events,
                "event_manifest_hash": event_manifest_hash,
                "event_manifest_components": event_manifest_components,
            },
            "representation": {
                "name": "latent_sample",
                "sampling": "deterministic_evaluation_time_hard_bernoulli",
                "bernoulli_probability_threshold": bernoulli_threshold,
            },
            "thresholds": {
                "entropy_unit": "bits",
                "minimum_joint_code_entropy_bits": self.minimum_joint_code_entropy_bits,
                "minimum_fraction_of_paired_gamma_zero_joint_entropy": (
                    self.minimum_fraction_of_paired_gamma_zero_joint_entropy
                ),
                "paired_reference": "same_architecture_and_autoencoder_seed",
            },
            "metrics": metrics,
            "decision": {
                "pass": absolute_pass,
                "pass_scope": "absolute_joint_code_entropy_only",
                "absolute_joint_entropy_pass": absolute_pass,
                "reason": (
                    "passed_minimum_joint_code_entropy"
                    if absolute_pass
                    else "joint_code_entropy_below_minimum"
                ),
                "paired_baseline_comparison": "deferred_to_phase_2_aggregation",
                "configuration_policy": "every_expected_seed_must_pass",
            },
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _bernoulli_threshold(pl_module: Any) -> float:
        bernoulli = getattr(pl_module, "bernoulli", None)
        threshold = getattr(bernoulli, "threshold", None)
        if not isinstance(threshold, torch.Tensor) or threshold.numel() != 1:
            raise RuntimeError(
                "The model does not expose a scalar Bernoulli threshold."
            )
        value = float(threshold.detach().cpu().item())
        if not 0.0 <= value <= 1.0:
            raise RuntimeError("The Bernoulli threshold must be in [0, 1].")
        return value

    @staticmethod
    def _checkpoint_identity(ckpt_path: Path) -> dict[str, Any]:
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Latent-collapse checkpoint does not exist: {ckpt_path}."
            )

        digest = hashlib.sha256()
        with ckpt_path.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "name": ckpt_path.name,
            "path": str(ckpt_path.resolve()),
            "sha256": digest.hexdigest(),
            "size_bytes": ckpt_path.stat().st_size,
            "selection_metric": "val/loss_total",
        }

    def _should_run_for_current_ckpt(self, trainer: Any) -> bool:
        strategy = getattr(trainer, "strat_name", None)
        metric = getattr(trainer, "metric_name", None)
        criterion = getattr(trainer, "criterion_name", None)
        if strategy is None:
            return False

        configured = self.ckpts.get(strategy)
        if isinstance(configured, bool):
            return configured
        if not isinstance(configured, Mapping) or metric is None or criterion is None:
            return False

        allowed_criteria = configured.get(metric)
        return (
            isinstance(allowed_criteria, Sequence)
            and not isinstance(allowed_criteria, (str, bytes))
            and criterion in allowed_criteria
        )
