from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import torch
from pytorch_lightning.callbacks import Callback

from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils as eval_utils
from src.utils.pairing.utils import (
    closure_metrics,
    mutual_nearest_pairs,
    standardized_mean_differences,
)


class PairingDiagnostics(Callback):
    """Evaluate whether a frozen encoder can support validation pair tables."""

    def __init__(
        self,
        output_name: str = "pairing_rep_data",
        view1_name: str = "pairing_view1_data",
        view2_name: str = "pairing_view2_data",
        dataset_1: str = "normal",
        dataset_2: str = "reference_normal",
        closure_dataset: str | None = None,
        k: int = 20,
        caliper_quantile: float | None = 0.95,
        name: str = "pairing_diagnostics",
    ):
        super().__init__()
        self.output_name = output_name
        self.view1_name = view1_name
        self.view2_name = view2_name
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.closure_dataset = closure_dataset or dataset_1
        self.k = int(k)
        self.caliper_quantile = (
            None if caliper_quantile is None else float(caliper_quantile)
        )
        self.name = name
        self.summary = defaultdict(float)
        self.last_metrics = {}

    def on_test_epoch_start(self, trainer, pl_module):
        self.reps = {self.dataset_1: [], self.dataset_2: []}
        self.raw = {self.dataset_1: [], self.dataset_2: []}
        self.closure_1 = []
        self.closure_2 = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name not in {self.dataset_1, self.dataset_2, self.closure_dataset}:
            return

        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1).detach().cpu()

        if dset_name in self.reps:
            self.reps[dset_name].append(outputs[self.output_name].detach().cpu())
            self.raw[dset_name].append(x)

        if dset_name == self.closure_dataset:
            self.closure_1.append(outputs[self.view1_name].detach().cpu())
            self.closure_2.append(outputs[self.view2_name].detach().cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        ckpt_name = Path(getattr(pl_module, "_ckpt_path", "last")).stem
        metrics = self._compute_metrics()
        self.last_metrics = metrics
        self.summary[ckpt_name] = metrics["selection_score"]

        root = self._artifact_dir(trainer, ckpt_name)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "pairing_diagnostics.json").open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    def _compute_metrics(self) -> dict[str, float]:
        z1 = torch.cat(self.reps[self.dataset_1], dim=0)
        z2 = torch.cat(self.reps[self.dataset_2], dim=0)
        x1 = torch.cat(self.raw[self.dataset_1], dim=0)
        x2 = torch.cat(self.raw[self.dataset_2], dim=0)
        c1 = torch.cat(self.closure_1, dim=0)
        c2 = torch.cat(self.closure_2, dim=0)

        close = closure_metrics(c1, c2)
        caliper = None
        if self.caliper_quantile is not None:
            caliper = torch.quantile(
                1.0
                - torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(c1.float(), dim=1),
                    torch.nn.functional.normalize(c2.float(), dim=1),
                    dim=1,
                ),
                self.caliper_quantile,
            ).item()
        pairs = mutual_nearest_pairs(z1, z2, k=self.k, caliper=caliper)

        smd_before = standardized_mean_differences(x1, x2)
        smd_after = standardized_mean_differences(x1, x2, pairs.idx_1, pairs.idx_2)
        coverage = pairs.idx_1.numel() / max(min(z1.shape[0], z2.shape[0]), 1)
        mean_smd_after = smd_after.mean().item() if smd_after.numel() else float("inf")

        selection_score = (
            close.get("closure_recall_at_10", 0.0)
            * coverage
            / (1.0 + max(mean_smd_after, 0.0))
        )

        return {
            **close,
            "mnn_pairs": int(pairs.idx_1.numel()),
            "mnn_coverage": float(coverage),
            "caliper": None if caliper is None else float(caliper),
            "pair_distance_mean": pairs.distance.mean().item()
            if pairs.distance.numel()
            else float("nan"),
            "pair_distance_p95": torch.quantile(pairs.distance, 0.95).item()
            if pairs.distance.numel()
            else float("nan"),
            "smd_before_mean": smd_before.mean().item()
            if smd_before.numel()
            else float("nan"),
            "smd_before_max": smd_before.max().item()
            if smd_before.numel()
            else float("nan"),
            "smd_after_mean": mean_smd_after,
            "smd_after_max": smd_after.max().item()
            if smd_after.numel()
            else float("nan"),
            "selection_score": float(selection_score),
        }

    def get_optimized_metric(self, ckpt_name: str | None = None):
        if ckpt_name is not None:
            return ckpt_name, self.summary[ckpt_name]
        if not self.summary:
            return None, None
        best = max(self.summary, key=self.summary.get)
        return best, self.summary[best]

    def clear_crit_summary(self):
        self.summary.clear()

    @staticmethod
    def _artifact_dir(trainer, ckpt_name: str) -> Path:
        if getattr(trainer, "logger", None) is not None and hasattr(trainer.logger, "save_dir"):
            return Path(trainer.logger.save_dir) / "pairing_diagnostics" / ckpt_name
        return Path("outputs") / "pairing_diagnostics" / ckpt_name
