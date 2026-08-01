from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils as eval_utils
from src.utils.pairing.table import atomic_json_dump
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
        max_events_per_dataset: int | None = 32768,
        closure_chunk_size: int = 512,
        min_active_fraction: float = 0.8,
        min_effective_rank: float = 6.0,
        min_participation_rank: float = 4.5,
        max_top_pc_fraction: float = 0.5,
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
        if self.k <= 0:
            raise ValueError("PairingDiagnostics k must be positive.")
        self.caliper_quantile = None if caliper_quantile is None else float(caliper_quantile)
        if self.caliper_quantile is not None and not 0.0 <= self.caliper_quantile <= 1.0:
            raise ValueError("PairingDiagnostics caliper_quantile must be between 0 and 1.")
        self.max_events_per_dataset = (
            None if max_events_per_dataset is None else int(max_events_per_dataset)
        )
        if self.max_events_per_dataset is not None and self.max_events_per_dataset <= 0:
            raise ValueError("PairingDiagnostics max_events_per_dataset must be positive.")
        self.closure_chunk_size = int(closure_chunk_size)
        if self.closure_chunk_size <= 0:
            raise ValueError("PairingDiagnostics closure_chunk_size must be positive.")
        self.min_active_fraction = float(min_active_fraction)
        self.min_effective_rank = float(min_effective_rank)
        self.min_participation_rank = float(min_participation_rank)
        self.max_top_pc_fraction = float(max_top_pc_fraction)
        if not 0.0 <= self.min_active_fraction <= 1.0:
            raise ValueError("min_active_fraction must be between zero and one.")
        if self.min_effective_rank <= 0.0 or self.min_participation_rank <= 0.0:
            raise ValueError("Rank thresholds must be positive.")
        if not 0.0 < self.max_top_pc_fraction <= 1.0:
            raise ValueError("max_top_pc_fraction must be in (0, 1].")
        self.name = name
        self.summary = defaultdict(float)
        self.last_metrics = {}

    def on_test_epoch_start(self, trainer, pl_module):
        self.reps = {self.dataset_1: [], self.dataset_2: []}
        self.raw = {self.dataset_1: [], self.dataset_2: []}
        self.raw_mask = {self.dataset_1: [], self.dataset_2: []}
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
        mask = torch.flatten(b.mask, start_dim=1).detach().cpu().bool()

        if dset_name in self.reps:
            if self.output_name not in outputs:
                raise KeyError(
                    f"Pairing diagnostics output {self.output_name!r} is missing for "
                    f"dataset {dset_name!r}. Available outputs: {sorted(outputs)}"
                )
            remaining = self._remaining(self.reps[dset_name])
            if remaining > 0:
                self.reps[dset_name].append(outputs[self.output_name][:remaining].detach().cpu())
                self.raw[dset_name].append(x[:remaining])
                self.raw_mask[dset_name].append(mask[:remaining])

        if dset_name == self.closure_dataset:
            missing = [name for name in (self.view1_name, self.view2_name) if name not in outputs]
            if missing:
                raise KeyError(
                    f"Pairing closure outputs are missing {missing}. "
                    f"Available outputs: {sorted(outputs)}"
                )
            remaining = self._remaining(self.closure_1)
            if remaining > 0:
                self.closure_1.append(outputs[self.view1_name][:remaining].detach().cpu())
                self.closure_2.append(outputs[self.view2_name][:remaining].detach().cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        ckpt_name = Path(getattr(pl_module, "_ckpt_path", "last")).stem
        metrics = self._compute_metrics()
        self.last_metrics = metrics
        self.summary[ckpt_name] = metrics["selection_score"]

        root = self._artifact_dir(trainer, ckpt_name)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(metrics, root / "pairing_diagnostics.json", overwrite=True)

    def _compute_metrics(self) -> dict[str, object]:
        missing = [
            name
            for name, values in (
                (self.dataset_1, self.reps.get(self.dataset_1, [])),
                (self.dataset_2, self.reps.get(self.dataset_2, [])),
                (f"{self.closure_dataset}:view1", self.closure_1),
                (f"{self.closure_dataset}:view2", self.closure_2),
            )
            if not values
        ]
        if missing:
            raise RuntimeError(f"Pairing diagnostics collected no outputs for: {missing}")
        z1 = torch.cat(self.reps[self.dataset_1], dim=0)
        z2 = torch.cat(self.reps[self.dataset_2], dim=0)
        x1 = torch.cat(self.raw[self.dataset_1], dim=0)
        x2 = torch.cat(self.raw[self.dataset_2], dim=0)
        m1 = torch.cat(self.raw_mask[self.dataset_1], dim=0)
        m2 = torch.cat(self.raw_mask[self.dataset_2], dim=0)
        c1 = torch.cat(self.closure_1, dim=0)
        c2 = torch.cat(self.closure_2, dim=0)

        close = closure_metrics(c1, c2, chunk_size=self.closure_chunk_size)
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
        value_smd_before = self._masked_value_smd(x1, m1, x2, m2)
        occupancy_smd_before = standardized_mean_differences(m1.float(), m2.float())
        coverage = pairs.idx_1.numel() / max(min(z1.shape[0], z2.shape[0]), 1)
        embedding = self._embedding_statistics(
            z1,
            min_active_fraction=self.min_active_fraction,
            min_effective_rank=self.min_effective_rank,
            min_participation_rank=self.min_participation_rank,
            max_top_pc_fraction=self.max_top_pc_fraction,
        )

        if pairs.idx_1.numel():
            smd_after = standardized_mean_differences(x1, x2, pairs.idx_1, pairs.idx_2)
            value_smd_after = self._masked_value_smd(
                x1[pairs.idx_1],
                m1[pairs.idx_1],
                x2[pairs.idx_2],
                m2[pairs.idx_2],
                jointly_valid=True,
            )
            occupancy_smd_after = standardized_mean_differences(
                m1.float(), m2.float(), pairs.idx_1, pairs.idx_2
            )
            mean_smd_after = smd_after.mean().item()
            raw_selection_score = (
                close.get("closure_recall_at_10", 0.0)
                * coverage
                / (1.0 + max(mean_smd_after, 0.0))
            )
        else:
            smd_after = torch.empty(0)
            value_smd_after = torch.empty(0)
            occupancy_smd_after = torch.empty(0)
            mean_smd_after = None
            raw_selection_score = 0.0
            embedding["collapse_pass"] = False
            embedding["collapse_failures"] = [
                *embedding["collapse_failures"],
                "no_mutual_nearest_pairs",
            ]

        selection_score = raw_selection_score if embedding["collapse_pass"] else 0.0

        return {
            **close,
            "mnn_pairs": int(pairs.idx_1.numel()),
            "mnn_coverage": float(coverage),
            "caliper": None if caliper is None else float(caliper),
            "pair_distance_mean": pairs.distance.mean().item() if pairs.distance.numel() else None,
            "pair_distance_p95": (
                torch.quantile(pairs.distance, 0.95).item() if pairs.distance.numel() else None
            ),
            "smd_before_mean": smd_before.mean().item() if smd_before.numel() else None,
            "smd_before_max": smd_before.max().item() if smd_before.numel() else None,
            "smd_after_mean": mean_smd_after,
            "smd_after_max": smd_after.max().item() if smd_after.numel() else None,
            "value_smd_before_mean": self._finite_mean(value_smd_before),
            "value_smd_after_mean": (
                self._finite_mean(value_smd_after) if value_smd_after.numel() else None
            ),
            "occupancy_smd_before_mean": self._finite_mean(occupancy_smd_before),
            "occupancy_smd_after_mean": (
                self._finite_mean(occupancy_smd_after) if occupancy_smd_after.numel() else None
            ),
            "raw_selection_score": float(raw_selection_score),
            "selection_score": float(selection_score),
            "n_dataset_1": int(z1.shape[0]),
            "n_dataset_2": int(z2.shape[0]),
            **embedding,
        }

    def _remaining(self, values: list[torch.Tensor]) -> int:
        if self.max_events_per_dataset is None:
            return 2**63 - 1
        return max(self.max_events_per_dataset - sum(value.shape[0] for value in values), 0)

    @staticmethod
    def _finite_mean(values: torch.Tensor) -> float:
        finite = values[torch.isfinite(values)]
        return finite.mean().item() if finite.numel() else float("nan")

    @staticmethod
    def _masked_value_smd(
        x1: torch.Tensor,
        m1: torch.Tensor,
        x2: torch.Tensor,
        m2: torch.Tensor,
        jointly_valid: bool = False,
    ) -> torch.Tensor:
        values = []
        for feature in range(x1.shape[1]):
            if jointly_valid:
                valid = m1[:, feature] & m2[:, feature]
                left = x1[valid, feature].float()
                right = x2[valid, feature].float()
            else:
                left = x1[m1[:, feature], feature].float()
                right = x2[m2[:, feature], feature].float()
            if left.numel() < 2 or right.numel() < 2:
                values.append(float("nan"))
                continue
            pooled = torch.sqrt((left.var(unbiased=False) + right.var(unbiased=False)) / 2)
            values.append(float((left.mean() - right.mean()).abs() / pooled.clamp_min(1e-8)))
        return torch.tensor(values)

    @staticmethod
    def _embedding_statistics(
        z: torch.Tensor,
        min_active_fraction: float = 0.8,
        min_effective_rank: float = 6.0,
        min_participation_rank: float = 4.5,
        max_top_pc_fraction: float = 0.5,
    ) -> dict[str, float | bool | list[str]]:
        z = z.float()
        finite_fraction = torch.isfinite(z).float().mean().item()
        safe = torch.nan_to_num(z)
        std = safe.std(dim=0, unbiased=False)
        centered = safe - safe.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(centered.shape[0], 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        variance = eigenvalues.sum().clamp_min(1e-12)
        probabilities = eigenvalues / variance
        nonzero = probabilities > 0
        effective_rank = torch.exp(
            -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        ).item()
        participation_rank = (
            variance.square() / eigenvalues.square().sum().clamp_min(1e-12)
        ).item()
        top_pc_fraction = (eigenvalues.max() / variance).item()
        active_fraction = (std > 1e-3).float().mean().item()

        failures = []
        if finite_fraction < 1.0:
            failures.append("nonfinite")
        if active_fraction < min_active_fraction:
            failures.append("inactive_dimensions")
        if effective_rank < min_effective_rank:
            failures.append("low_effective_rank")
        if participation_rank < min_participation_rank:
            failures.append("low_participation_rank")
        if top_pc_fraction > max_top_pc_fraction:
            failures.append("dominant_principal_component")

        sample = torch.nn.functional.normalize(safe[: min(2048, safe.shape[0])], dim=1)
        cosine = sample @ sample.T
        offdiag = ~torch.eye(sample.shape[0], dtype=torch.bool)
        mean_offdiag_cosine = cosine[offdiag].mean().item() if offdiag.any() else float("nan")
        return {
            "embedding_finite_fraction": finite_fraction,
            "embedding_mean_feature_std": std.mean().item(),
            "embedding_active_fraction": active_fraction,
            "embedding_effective_rank": effective_rank,
            "embedding_participation_rank": participation_rank,
            "embedding_top_pc_fraction": top_pc_fraction,
            "embedding_mean_norm": safe.norm(dim=1).mean().item(),
            "embedding_mean_offdiag_cosine": mean_offdiag_cosine,
            "collapse_min_active_fraction": float(min_active_fraction),
            "collapse_min_effective_rank": float(min_effective_rank),
            "collapse_min_participation_rank": float(min_participation_rank),
            "collapse_max_top_pc_fraction": float(max_top_pc_fraction),
            "collapse_pass": not failures,
            "collapse_failures": failures,
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
        if getattr(trainer, "default_root_dir", None):
            return Path(trainer.default_root_dir) / "metrics" / "pairing_diagnostics" / ckpt_name
        if getattr(trainer, "logger", None) is not None and hasattr(trainer.logger, "save_dir"):
            return Path(trainer.logger.save_dir) / "pairing_diagnostics" / ckpt_name
        return Path("outputs") / "pairing_diagnostics" / ckpt_name
