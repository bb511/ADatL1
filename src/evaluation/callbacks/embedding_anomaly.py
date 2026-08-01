from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from src.utils.pairing.table import atomic_json_dump


class EmbeddingAnomalyMetrics(Callback):
    """Measure validation anomaly utility of a frozen encoder with cosine kNN."""

    def __init__(
        self,
        output_name: str = "pairing_rep_data",
        reference_dataset: str = "normal",
        background_datasets: list[str] | None = None,
        reference_size: int = 8192,
        max_query_events: int = 8192,
        k: int = 10,
        target_fprs: tuple[float, ...] = (1e-2, 1e-3),
        name: str = "embedding_anomaly",
    ) -> None:
        super().__init__()
        self.output_name = output_name
        self.reference_dataset = reference_dataset
        self.background_datasets = set(background_datasets or ["SingleNeutrino_E-10-gun"])
        self.reference_size = int(reference_size)
        self.max_query_events = int(max_query_events)
        self.k = int(k)
        self.target_fprs = tuple(float(value) for value in target_fprs)
        if self.reference_size <= 0 or self.max_query_events <= 0 or self.k <= 0:
            raise ValueError("Embedding anomaly sizes and k must be positive.")
        if any(not 0 < value < 1 for value in self.target_fprs):
            raise ValueError("Embedding anomaly target_fprs must lie strictly between 0 and 1.")
        self.name = name
        self.summary = defaultdict(float)
        self.last_metrics: dict = {}

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        names = list(trainer.test_dataloaders.keys())
        if self.reference_dataset not in names:
            raise ValueError(
                f"Embedding anomaly reference {self.reference_dataset!r} is missing; "
                f"available datasets are {names}."
            )
        self.reference: list[torch.Tensor] | torch.Tensor = []
        self.normal_scores: list[torch.Tensor] = []
        self.signal_scores: dict[str, list[torch.Tensor]] = {
            name: []
            for name in names
            if name != self.reference_dataset and name not in self.background_datasets
        }

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        dataset = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dataset in self.background_datasets:
            return
        if self.output_name not in outputs:
            raise KeyError(f"Embedding anomaly output {self.output_name!r} is missing.")
        embedding = outputs[self.output_name].detach()

        if dataset == self.reference_dataset:
            embedding = self._take_reference(embedding)
            if not embedding.numel():
                return
            remaining = self._remaining(self.normal_scores)
            if remaining > 0:
                self.normal_scores.append(self._score(embedding[:remaining]).cpu())
            return

        remaining = self._remaining(self.signal_scores[dataset])
        if remaining > 0:
            self.signal_scores[dataset].append(self._score(embedding[:remaining]).cpu())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not self.normal_scores:
            raise RuntimeError(
                "Embedding anomaly evaluation collected no normal scoring events. "
                "Increase the normal evaluation cap beyond reference_size."
            )
        normal = torch.cat(self.normal_scores).float()
        per_dataset = {}
        for dataset, chunks in self.signal_scores.items():
            if not chunks:
                continue
            signal = torch.cat(chunks).float()
            scores = torch.cat((normal, signal))
            labels = torch.cat(
                (
                    torch.zeros(normal.shape[0], dtype=torch.long),
                    torch.ones(signal.shape[0], dtype=torch.long),
                )
            )
            metrics = {
                "auroc": BinaryAUROC()(scores, labels).item(),
                "auprc": BinaryAveragePrecision()(scores, labels).item(),
                "n_signal": int(signal.shape[0]),
            }
            for target_fpr in self.target_fprs:
                threshold = torch.quantile(normal, 1.0 - target_fpr)
                metrics[f"tpr_at_fpr_{target_fpr:g}"] = (signal > threshold).float().mean().item()
            per_dataset[dataset] = metrics
        if not per_dataset:
            raise RuntimeError("Embedding anomaly evaluation collected no signal datasets.")

        aurocs = torch.tensor([metrics["auroc"] for metrics in per_dataset.values()])
        auprcs = torch.tensor([metrics["auprc"] for metrics in per_dataset.values()])
        worst_count = max(1, math.ceil(0.25 * aurocs.numel()))
        metrics = {
            "macro_median_auroc": aurocs.median().item(),
            "macro_mean_auroc": aurocs.mean().item(),
            "worst_quartile_mean_auroc": aurocs.sort().values[:worst_count].mean().item(),
            "macro_median_auprc": auprcs.median().item(),
            "n_normal": int(normal.shape[0]),
            "n_signal_datasets": len(per_dataset),
            "per_dataset": per_dataset,
        }
        ckpt_name = Path(getattr(pl_module, "_ckpt_path", "last")).stem
        self.last_metrics = metrics
        self.summary[ckpt_name] = metrics["macro_median_auroc"]
        root = self._artifact_dir(trainer, ckpt_name)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(metrics, root / "embedding_anomaly.json", overwrite=True)

    def get_optimized_metric(self, ckpt_name: str | None = None):
        if ckpt_name is not None:
            return ckpt_name, self.summary[ckpt_name]
        if not self.summary:
            return None, None
        best = max(self.summary, key=self.summary.get)
        return best, self.summary[best]

    def clear_crit_summary(self) -> None:
        self.summary.clear()

    def _take_reference(self, embedding: torch.Tensor) -> torch.Tensor:
        if isinstance(self.reference, torch.Tensor):
            return embedding
        current = sum(chunk.shape[0] for chunk in self.reference)
        needed = max(self.reference_size - current, 0)
        if needed:
            self.reference.append(embedding[:needed].cpu())
            embedding = embedding[needed:]
        if sum(chunk.shape[0] for chunk in self.reference) >= self.reference_size:
            self.reference = F.normalize(
                torch.cat(self.reference)[: self.reference_size].float(), dim=1
            )
        return embedding

    def _remaining(self, values: list[torch.Tensor]) -> int:
        return max(self.max_query_events - sum(value.shape[0] for value in values), 0)

    @torch.no_grad()
    def _score(self, embedding: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.reference, torch.Tensor):
            raise RuntimeError("Embedding anomaly reference must be collected before queries.")
        query = F.normalize(embedding.float(), dim=1)
        reference = self.reference.to(
            device=query.device, dtype=query.dtype, non_blocking=True
        )
        k = min(self.k, reference.shape[0])
        similarities = query @ reference.T
        return 1.0 - torch.topk(similarities, k=k, dim=1).values.mean(dim=1)

    @staticmethod
    def _artifact_dir(trainer, ckpt_name: str) -> Path:
        if getattr(trainer, "default_root_dir", None):
            return Path(trainer.default_root_dir) / "metrics" / "embedding_anomaly" / ckpt_name
        if getattr(trainer, "logger", None) is not None and hasattr(trainer.logger, "save_dir"):
            return Path(trainer.logger.save_dir) / "embedding_anomaly" / ckpt_name
        return Path("outputs") / "embedding_anomaly" / ckpt_name
