"""Evaluation-only multi-seed random-bijection CAP diagnostic."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.metrics.cap.binary.pairing import mapping
from src.callbacks.metrics.cap.metric import ApproximationCapacity
from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils
from src.utils.pairing.random_tables import validate_frozen_random_pairing_tables
from src.utils.pairing.table import load_pair_table, sha256_file, sha256_tensor


class RandomBijectionCAP(Callback):
    """Evaluate five frozen random CAP mappings from one collected score stream.

    This callback deliberately has no ``get_optimized_metric`` method and cannot select a
    checkpoint. It is intended only for offline evaluation of already sealed checkpoints.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_manifest_path: str,
        cap_metric_config: dict,
        score: str,
        name: str,
    ) -> None:
        super().__init__()
        self.output_name = output_name
        self.dataset_1_name = dataset_1
        self.dataset_2_name = dataset_2
        self.pairing_manifest_path = Path(pairing_manifest_path).expanduser().resolve()
        self.cap_metric_config = dict(cap_metric_config)
        self.score = score
        self.name = name
        self.device = None

    def on_test_start(self, trainer, pl_module) -> None:
        """Authenticate the table universe and require both source loaders."""
        self.device = pl_module.device
        if self.score == "residual_oas":
            readiness = None
            for name in ("residual_oas_ready", "residual_score_ready"):
                if hasattr(pl_module, name):
                    readiness = bool(getattr(pl_module, name))
                    break
            if readiness is not True:
                raise RuntimeError(
                    "Random-CAP residual_oas requires an authenticated OAS-ready checkpoint."
                )
        if not hasattr(self, "manifest"):
            self.manifest = validate_frozen_random_pairing_tables(
                self.pairing_manifest_path.parent.parent
            )
        if (
            self.pairing_manifest_path
            != (
                self.pairing_manifest_path.parent.parent / "random_pairing/manifest.json"
            ).resolve()
        ):
            raise ValueError("Random CAP manifest must be the campaign's frozen manifest.")
        loaders = set(trainer.test_dataloaders)
        expected = {self.dataset_1_name, self.dataset_2_name}
        if not expected <= loaders:
            raise ValueError(f"Random CAP is missing source loaders: {sorted(expected-loaders)}")

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Initialize one shared source/score cache for all five mappings."""
        del pl_module
        self.split = "validate" if trainer.split in {"val", "validate"} else trainer.split
        if self.split not in {"validate", "test"}:
            raise ValueError(f"Random CAP does not support split {self.split!r}.")
        self.dataset_1_scores: list[torch.Tensor] = []
        self.dataset_2_scores: list[torch.Tensor] = []
        self.dataset_1_inputs: list[torch.Tensor] = []
        self.dataset_2_inputs: list[torch.Tensor] = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Collect each source input and score once, independent of seed count."""
        del pl_module, batch_idx
        dataset = list(trainer.test_dataloaders)[dataloader_idx]
        if dataset not in {self.dataset_1_name, self.dataset_2_name}:
            return
        inputs = torch.flatten(unpack_batch(batch).x, start_dim=1).detach().cpu()
        scores = outputs[self.output_name].detach().reshape(-1)
        if len(inputs) != len(scores):
            raise ValueError(f"Random CAP input/score row count differs for {dataset!r}.")
        if dataset == self.dataset_1_name:
            self.dataset_1_inputs.append(inputs)
            self.dataset_1_scores.append(scores)
        else:
            self.dataset_2_inputs.append(inputs)
            self.dataset_2_scores.append(scores)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Run production CAP once per frozen seed and persist explicit scalar rows."""
        del trainer
        if not self.dataset_1_scores or not self.dataset_2_scores:
            raise RuntimeError("Random CAP did not collect both source score streams.")
        score_1 = torch.cat(self.dataset_1_scores)
        score_2 = torch.cat(self.dataset_2_scores)
        source_1_sha256 = sha256_tensor(torch.cat(self.dataset_1_inputs))
        source_2_sha256 = sha256_tensor(torch.cat(self.dataset_2_inputs))
        rows = []
        tables = sorted(
            (row for row in self.manifest["tables"] if row["split"] == self.split),
            key=lambda row: int(row["replicate"]),
        )
        if len(tables) != 5 or [int(row["replicate"]) for row in tables] != list(range(5)):
            raise RuntimeError(f"Random CAP table coverage differs for {self.split}.")
        for table_row in tables:
            path = Path(table_row["path"]).resolve(strict=True)
            if sha256_file(path) != table_row["file_sha256"]:
                raise RuntimeError(f"Random CAP table file changed: {path}")
            table = load_pair_table(
                path,
                expected_dataset_1=self.dataset_1_name,
                expected_dataset_2=self.dataset_2_name,
                expected_split=self.split,
                n_dataset_1=len(score_1),
                n_dataset_2=len(score_2),
                source_1_sha256=source_1_sha256,
                source_2_sha256=source_2_sha256,
            )
            idx_1, idx_2 = mapping(score_1, score_2, table["map_0_to_1"])
            metric = ApproximationCapacity(
                **copy.deepcopy(self.cap_metric_config), device=self.device
            ).to(self.device)
            with torch.inference_mode(False), torch.enable_grad():
                metric.update(
                    score_1[idx_1].clone().requires_grad_(True),
                    score_2[idx_2].clone().requires_grad_(True),
                )
            value = float(metric.compute())
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Random CAP produced a nonfinite value for seed {table_row['seed']}."
                )
            rows.append(
                {
                    "checkpoint": Path(pl_module._ckpt_path).stem,
                    "split": self.split,
                    "score": self.score,
                    "replicate": int(table_row["replicate"]),
                    "seed": int(table_row["seed"]),
                    "metric": "cap",
                    "value": value,
                    "pair_table_sha256": table_row["file_sha256"],
                    "permutation_sha256": table_row["permutation_sha256"],
                }
            )
            metric.to("cpu")
            del metric
        checkpoint_dir = Path(pl_module._ckpt_path).parent
        output = (
            checkpoint_dir
            / "plots"
            / ("val" if self.split == "validate" else "test")
            / Path(pl_module._ckpt_path).stem
            / self.name
            / "values.csv"
        )
        utils.misc.write_metric_values(output, rows)


__all__ = ["RandomBijectionCAP"]
