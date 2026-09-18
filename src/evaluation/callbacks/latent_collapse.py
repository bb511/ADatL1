"""Latent-code collapse diagnostics for the frozen Pareto validation protocol."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from src.data.utils import unpack_batch


class LatentCollapseDiagnosticsCallback(Callback):
    """Measure deterministic hard-Bernoulli code diversity on normal validation.

    The callback runs only for the selected loss-total checkpoint. It measures
    latent_sample, rather than continuous logits, because that is the representation
    consumed by the decoder. The absolute entropy gate is decided per run; the paired
    gamma-zero comparison is deliberately left for the Phase 2 collector.
    """

    def __init__(
        self,
        *,
        architecture_id: str,
        minimum_joint_code_entropy_bits: float,
        minimum_fraction_of_paired_gamma_zero_joint_entropy: float,
        dataset: str = "normal",
        evaluation_split: str = "val",
        ckpts: dict | None = None,
        name: str = "latent_collapse",
    ) -> None:
        super().__init__()
        if not architecture_id:
            raise ValueError("architecture_id must be a non-empty string.")
        if float(minimum_joint_code_entropy_bits) < 0.0:
            raise ValueError("minimum_joint_code_entropy_bits must be non-negative.")
        if not 0.0 <= float(
            minimum_fraction_of_paired_gamma_zero_joint_entropy
        ) <= 1.0:
            raise ValueError(
                "minimum_fraction_of_paired_gamma_zero_joint_entropy must be "
                "in [0, 1]."
            )

        self.architecture_id = architecture_id
        self.minimum_joint_code_entropy_bits = float(
            minimum_joint_code_entropy_bits
        )
        self.minimum_fraction_of_paired_gamma_zero_joint_entropy = float(
            minimum_fraction_of_paired_gamma_zero_joint_entropy
        )
        self.dataset = dataset
        self.evaluation_split = evaluation_split
        self.ckpts = ckpts or {"loss_total": True}
        self.name = name
        self._active = False

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._active = (
            str(getattr(trainer, "split", "")) == self.evaluation_split
            and self._should_run_for_current_ckpt(trainer)
        )
        if not self._active:
            return
        if bool(pl_module.training):
            raise RuntimeError("Latent-collapse diagnostics require evaluation mode.")

        self._bit_sums: np.ndarray | None = None
        self._latent_width: int | None = None
        self._code_counts: Counter[bytes] = Counter()
        self._n_events = 0
        self._verified_determinism = False

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._active:
            return

        dataset = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dataset != self.dataset:
            return

        model_input = torch.flatten(unpack_batch(batch).x, start_dim=1)
        with torch.inference_mode():
            representations = pl_module.forward_with_representations(model_input)
        latent_sample = self._binary_sample(
            representations,
            expected_batch_size=model_input.shape[0],
        )

        if not self._verified_determinism:
            with torch.inference_mode():
                repeated = pl_module.forward_with_representations(model_input)
            if not torch.equal(
                latent_sample,
                self._binary_sample(
                    repeated,
                    expected_batch_size=model_input.shape[0],
                ),
            ):
                raise RuntimeError(
                    "Evaluation-time latent_sample is not deterministic on the "
                    "first normal validation batch."
                )
            self._verified_determinism = True

        self._accumulate_latent_sample(latent_sample)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
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
        output_folder = (
            ckpt_path.parent
            / "plots"
            / str(trainer.split)
            / ckpt_path.stem
            / self.name
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        self._write_summary(
            output_folder / "collapse_summary.json",
            checkpoint_name=ckpt_path.name,
            split=str(trainer.split),
            bernoulli_threshold=self._bernoulli_threshold(pl_module),
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
        sample = representations.get("latent_sample")
        if not isinstance(sample, torch.Tensor):
            raise RuntimeError(
                "forward_with_representations must return a tensor latent_sample."
            )
        if sample.ndim == 0:
            raise RuntimeError("latent_sample cannot be a scalar.")
        sample = (
            sample.unsqueeze(1)
            if sample.ndim == 1
            else torch.flatten(sample, start_dim=1)
        )
        if sample.shape[0] != expected_batch_size:
            raise RuntimeError(
                "latent_sample and normal-validation input have different event "
                "counts."
            )
        if sample.shape[1] < 1:
            raise RuntimeError("latent_sample must contain at least one bit.")
        if not bool(torch.isfinite(sample).all().item()):
            raise RuntimeError("latent_sample contains NaN or infinity.")
        binary = (sample == 0) | (sample == 1)
        if not bool(torch.all(binary).item()):
            # Report the offending values. Without them this failure is
            # undiagnosable after the fact: it aborts a run that has already
            # spent its full epoch budget, and the batch sandbox holding the
            # model is deleted when the job ends. Values in {0, 0.1, ..., 1.0}
            # mean the module was left in training mode; values a few ulps off
            # 0 or 1 mean straight-through rounding.
            offending = sample[~binary]
            examples = ", ".join(
                f"{float(v):.17g}" for v in offending.flatten()[:8].tolist()
            )
            raise RuntimeError(
                "Evaluation-time latent_sample must contain only hard zero/one "
                f"codes. dtype={sample.dtype}, shape={tuple(sample.shape)}, "
                f"non-binary={offending.numel()} of {sample.numel()}, "
                f"min={float(sample.min()):.17g}, max={float(sample.max()):.17g}, "
                f"distinct non-binary values={int(torch.unique(offending).numel())}, "
                f"examples=[{examples}]"
            )
        return sample

    @staticmethod
    def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
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
        code_probabilities = (
            np.fromiter(self._code_counts.values(), dtype=np.float64) / self._n_events
        )
        joint_entropy = float(
            -np.sum(code_probabilities * np.log2(code_probabilities))
        )

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
        checkpoint_name: str,
        split: str,
        bernoulli_threshold: float,
    ) -> None:
        metrics = self._metrics()
        absolute_pass = (
            metrics["joint_code_entropy_bits"]
            >= self.minimum_joint_code_entropy_bits
        )
        payload = {
            "schema_version": 1,
            "checkpoint": checkpoint_name,
            "split": split,
            "dataset": self.dataset,
            "representation": {
                "name": "latent_sample",
                "sampling": "deterministic_evaluation_time_hard_bernoulli",
                "bernoulli_probability_threshold": bernoulli_threshold,
            },
            "thresholds": {
                "entropy_unit": "bits",
                "minimum_joint_code_entropy_bits": self.minimum_joint_code_entropy_bits,
            },
            "paired_baseline": {
                "architecture_id": self.architecture_id,
                "minimum_fraction_of_paired_gamma_zero_joint_entropy": (
                    self.minimum_fraction_of_paired_gamma_zero_joint_entropy
                ),
                "paired_reference": "same_architecture_and_autoencoder_seed",
                "status": "requires_phase_2_aggregation",
            },
            "metrics": metrics,
            "decision": {
                "absolute_entropy_pass": absolute_pass,
                "paired_baseline_entropy_pass": None,
                "configuration_eligible": False if not absolute_pass else None,
                "reason": (
                    "joint_code_entropy_below_minimum"
                    if not absolute_pass
                    else "awaiting_paired_gamma_zero_comparison"
                ),
            },
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _bernoulli_threshold(pl_module: Any) -> float:
        threshold = getattr(getattr(pl_module, "bernoulli", None), "threshold", None)
        if not isinstance(threshold, torch.Tensor) or threshold.numel() != 1:
            raise RuntimeError(
                "The model does not expose a scalar Bernoulli threshold."
            )
        value = float(threshold.detach().cpu().item())
        if not 0.0 <= value <= 1.0:
            raise RuntimeError("The Bernoulli threshold must be in [0, 1].")
        return value

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
