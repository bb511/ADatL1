"""Threshold-independent ROC diagnostics for the Pareto validation protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils
from src.evaluation.artifact_provenance import (
    EventManifest,
    checkpoint_identity_from_trainer,
    data_cache_identity_from_trainer,
    evaluation_mode_from_trainer_split,
    make_artifact_provenance,
    normalize_artifact_identity,
)
from src.plot import horizontal_bar


class AnomalyAUROCCallback(Callback):
    """Store per-signal AUROC and low-FPR partial-AUROC diagnostics.

    Normal events are the negative class and each configured signal dataset is
    independently the positive class.  Scores must increase with anomalousness.
    The partial AUROC is the raw ROC area from zero through ``max_false_positive_rate``
    (with a linearly interpolated endpoint), divided by that FPR as frozen in the
    FET.Et Pareto-study manifest.
    """

    def __init__(
        self,
        output_name: str,
        ds: list[str],
        max_false_positive_rate: float,
        pure_normal: bool = False,
        score_direction: str = "higher_score_is_more_anomalous",
        ckpts: dict | None = None,
        log_raw_mlflow: bool = True,
        name: str = "auroc",
        artifact_provenance: dict | None = None,
    ) -> None:
        super().__init__()

        if not ds:
            raise ValueError("AUROC diagnostics require at least one signal dataset.")
        if len(set(ds)) != len(ds):
            raise ValueError("AUROC signal datasets must be unique.")
        if not 0.0 < float(max_false_positive_rate) <= 1.0:
            raise ValueError(
                "max_false_positive_rate must be in (0, 1], got "
                f"{max_false_positive_rate!r}."
            )
        if score_direction != "higher_score_is_more_anomalous":
            raise ValueError(
                "Only higher_score_is_more_anomalous is supported by the "
                "frozen Pareto protocol."
            )

        self.output_name = output_name
        self.ds = tuple(ds)
        self.max_false_positive_rate = float(max_false_positive_rate)
        self.pure_normal = bool(pure_normal)
        self.score_direction = score_direction
        self.ckpts = ckpts or {"loss_total": True}
        self.log_raw_mlflow = log_raw_mlflow
        self.name = name
        self.artifact_provenance = normalize_artifact_identity(artifact_provenance)
        self._active = False

    def on_test_start(self, trainer, pl_module) -> None:
        """Ensure normal events define the shared negative-class distribution."""
        if list(trainer.test_dataloaders.keys())[0] != "normal":
            raise ValueError("AUROC callback needs normal data first in the data dict!")

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Initialise one score buffer per dataset for an enabled checkpoint."""
        self._active = self._should_run_for_current_ckpt(trainer)
        if not self._active:
            return

        self._normal_score_chunks: list[np.ndarray] = []
        self._signal_score_chunks = {dataset: [] for dataset in self.ds}
        self._event_manifest = (
            EventManifest(
                "valid"
                if evaluation_mode_from_trainer_split(trainer.split) == "validation"
                else "test"
            )
            if self.artifact_provenance is not None
            else None
        )

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect finite score vectors without retaining GPU tensors."""
        if not self._active:
            return

        dataset = list(trainer.test_dataloaders.keys())[dataloader_idx]
        scores = self._scores_to_numpy(outputs[self.output_name], dataset)

        if dataset == "normal":
            batch_view = unpack_batch(batch)
            if self.pure_normal:
                if batch_view.l1bit is None:
                    raise ValueError(
                        "pure_normal=True requires l1bit in normal batches."
                    )
                keep = ~batch_view.l1bit.detach().cpu().numpy().astype(bool)
                scores = scores[keep]
                if self._event_manifest is not None:
                    self._event_manifest.update_batch(dataset, batch, rows=keep)
            elif self._event_manifest is not None:
                self._event_manifest.update_batch(dataset, batch)
            if scores.size:
                self._normal_score_chunks.append(scores)
            return

        if dataset not in self._signal_score_chunks:
            return

        labels = unpack_batch(batch).y
        if labels is None or not bool(torch.all(labels > 0).item()):
            raise ValueError(
                "AUROC callback requires every configured signal loader to contain "
                f"only positive labels; dataset={dataset!r}."
            )
        if self._event_manifest is not None:
            self._event_manifest.update_batch(dataset, batch)
        if scores.size:
            self._signal_score_chunks[dataset].append(scores)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Write one machine-readable diagnostic artifact for the checkpoint."""
        if not self._active:
            return

        normal_scores = self._concatenate_scores(
            self._normal_score_chunks,
            dataset="normal",
        )
        per_signal = {
            dataset: self._signal_metrics(
                normal_scores,
                self._concatenate_scores(chunks, dataset=dataset),
            )
            for dataset, chunks in self._signal_score_chunks.items()
        }

        ckpt_path = Path(pl_module._ckpt_path)
        ckpt_name = ckpt_path.stem
        split = str(trainer.split)
        plot_folder = ckpt_path.parent / "plots" / split / ckpt_name / self.name
        plot_folder.mkdir(parents=True, exist_ok=True)

        self._write_summary(
            plot_folder / "auroc_summary.json",
            checkpoint_name=ckpt_path.name,
            split=split,
            normal_event_count=int(normal_scores.size),
            per_signal=per_signal,
            provenance=self._provenance_payload(trainer, pl_module),
        )

        self._plot(
            {name: values["auroc"] for name, values in per_signal.items()},
            "AUROC per signal",
            plot_folder,
        )
        self._plot(
            {
                name: values["partial_auroc"]
                for name, values in per_signal.items()
            },
            "Normalized partial AUROC per signal",
            plot_folder,
        )
        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            self.name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=self.name,
        )

    @staticmethod
    def _scores_to_numpy(scores: torch.Tensor, dataset: str) -> np.ndarray:
        if not isinstance(scores, torch.Tensor):
            raise TypeError(
                f"AUROC score {dataset!r} must be a torch.Tensor, got {type(scores)!r}."
            )

        values = scores.detach().float().cpu().numpy().reshape(-1).copy()
        if not np.isfinite(values).all():
            raise ValueError(f"AUROC scores for dataset {dataset!r} are not finite.")
        return values

    @staticmethod
    def _concatenate_scores(chunks: list[np.ndarray], *, dataset: str) -> np.ndarray:
        if not chunks:
            raise ValueError(
                f"No usable AUROC scores were accumulated for {dataset!r}."
            )
        values = np.concatenate(chunks, axis=0)
        if not values.size:
            raise ValueError(
                f"No usable AUROC scores were accumulated for {dataset!r}."
            )
        return values

    def _signal_metrics(
        self,
        normal_scores: np.ndarray,
        signal_scores: np.ndarray,
    ) -> dict[str, float | int]:
        labels = np.concatenate(
            [
                np.zeros(normal_scores.size, dtype=np.int8),
                np.ones(signal_scores.size, dtype=np.int8),
            ]
        )
        scores = np.concatenate([normal_scores, signal_scores])
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
        raw_partial_auroc = self._raw_partial_auroc(fpr, tpr)

        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "partial_auroc_raw": raw_partial_auroc,
            "partial_auroc": raw_partial_auroc / self.max_false_positive_rate,
            "signal_event_count": int(signal_scores.size),
        }

    def _raw_partial_auroc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Integrate the ROC through the configured FPR with an exact endpoint."""
        max_fpr = self.max_false_positive_rate
        right = int(np.searchsorted(fpr, max_fpr, side="right"))
        partial_fpr = fpr[:right].copy()
        partial_tpr = tpr[:right].copy()

        if partial_fpr.size == 0:
            raise ValueError("ROC curve does not contain its zero-FPR origin.")

        if partial_fpr[-1] < max_fpr:
            if right >= fpr.size:
                raise ValueError(
                    "ROC curve ended before the configured maximum false-positive rate."
                )
            lower_fpr = partial_fpr[-1]
            upper_fpr = fpr[right]
            if upper_fpr <= lower_fpr:
                raise ValueError("ROC FPR values must increase after duplicate points.")
            endpoint_tpr = partial_tpr[-1] + (
                (max_fpr - lower_fpr)
                * (tpr[right] - partial_tpr[-1])
                / (upper_fpr - lower_fpr)
            )
            partial_fpr = np.append(partial_fpr, max_fpr)
            partial_tpr = np.append(partial_tpr, endpoint_tpr)

        return float(np.trapz(partial_tpr, partial_fpr))

    def _write_summary(
        self,
        output_path: Path,
        *,
        checkpoint_name: str,
        split: str,
        normal_event_count: int,
        per_signal: dict[str, dict[str, float | int]],
        provenance: dict | None = None,
    ) -> None:
        aurocs = np.fromiter(
            (float(values["auroc"]) for values in per_signal.values()),
            dtype=float,
        )
        partial_aurocs = np.fromiter(
            (float(values["partial_auroc"]) for values in per_signal.values()),
            dtype=float,
        )
        payload = {
            "schema_version": 2,
            "checkpoint": checkpoint_name,
            "split": split,
            "anomaly_score": self.output_name,
            "score_direction": self.score_direction,
            "roc_convention": {
                "negative_class": "normal",
                "positive_class": "signal",
                "threshold_convention": "threshold-independent ROC sweep",
                "partial_auroc_max_false_positive_rate": self.max_false_positive_rate,
                "partial_auroc_raw_definition": (
                    "integral of TPR as a function of FPR from 0 through the "
                    "configured maximum FPR"
                ),
                "partial_auroc_endpoint_rule": (
                    "linearly interpolate TPR at maximum FPR"
                ),
                "partial_auroc_normalization": (
                    "raw_partial_auc / max_false_positive_rate"
                ),
            },
            "normal_event_count": normal_event_count,
            "num_signal_datasets": len(per_signal),
            "per_signal": {name: per_signal[name] for name in sorted(per_signal)},
            "summaries": {
                "median_auroc": float(np.median(aurocs)),
                "min_auroc": float(np.min(aurocs)),
                "median_partial_auroc": float(np.median(partial_aurocs)),
                "min_partial_auroc": float(np.min(partial_aurocs)),
            },
            "metric_contract": {
                "auroc": {
                    "definition": "Area under the full signal-versus-normal ROC curve.",
                    "unit": "dimensionless",
                    "range": [0.0, 1.0],
                },
                "partial_auroc": {
                    "definition": (
                        "Raw ROC area through the configured maximum false-positive "
                        "rate, divided by that rate."
                    ),
                    "unit": "dimensionless",
                    "range": [0.0, 1.0],
                },
            },
        }
        if provenance is not None:
            payload["provenance"] = provenance
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _provenance_payload(self, trainer, pl_module) -> dict | None:
        if self.artifact_provenance is None:
            return None
        if self._event_manifest is None:
            raise RuntimeError("AUROC provenance event manifest was not initialized.")
        return make_artifact_provenance(
            self.artifact_provenance,
            checkpoint=checkpoint_identity_from_trainer(trainer, pl_module._ckpt_path),
            evaluation_mode=evaluation_mode_from_trainer_split(trainer.split),
            data_cache=data_cache_identity_from_trainer(trainer),
            event_manifest=self._event_manifest.payload(),
        )

    @staticmethod
    def _plot(
        values: dict[str, float],
        xlabel: str,
        plot_folder: Path,
    ) -> None:
        horizontal_bar.plot_yright(
            values,
            values,
            xlabel,
            " ",
            plot_folder,
            percent=False,
        )

    def _should_run_for_current_ckpt(self, trainer) -> bool:
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
