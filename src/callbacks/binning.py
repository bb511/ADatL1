"""Lightweight training diagnostics for fixed MI-sensitive binning."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback

from src.plot.histogram import (
    plot_categorical_bin_counts,
    plot_minibatch_scalar_histogram,
)


class BinningDiagnosticsCallback(Callback):
    """Plot fixed-bin occupancy and raw-value diversity during selected epochs."""

    def __init__(
        self,
        enabled: bool = False,
        epochs: Sequence[int] = (0,),
        batch_indices: Sequence[int] = (0, 100, 400, 764),
        output_subdir: str = "mi_diagnostics",
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.epochs = frozenset(self._validate_indices(epochs, "epochs"))
        self.batch_indices = frozenset(
            self._validate_indices(batch_indices, "batch_indices")
        )

        subdir = Path(output_subdir)
        if subdir.is_absolute() or ".." in subdir.parts:
            raise ValueError(
                "output_subdir must be a relative path within the run output."
            )
        self.output_subdir = subdir

        self._unique_counts: list[int] = []
        self._nominal_batch_size: int | None = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Reset scalar collection and validate fixed-binner state when scheduled."""
        self._unique_counts = []
        self._nominal_batch_size = None

        if not self._is_scheduled_epoch(trainer) or not self._is_global_zero(trainer):
            return

        self._validate_fitted_binner(pl_module)
        self._nominal_batch_size = self._get_nominal_batch_size(trainer)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Collect one diversity scalar and render selected occupancy plots."""
        if not self._is_scheduled_epoch(trainer) or not self._is_global_zero(trainer):
            return

        binner, num_effective_bins, fit_counts = self._validate_fitted_binner(
            pl_module
        )

        with torch.no_grad():
            raw_values = pl_module.extract_sensitive_values(batch).detach().flatten()
            finite_values = raw_values[torch.isfinite(raw_values)]
            self._unique_counts.append(int(torch.unique(finite_values).numel()))

            if self._nominal_batch_size is None:
                self._nominal_batch_size = int(raw_values.numel())

            if int(batch_idx) not in self.batch_indices:
                return

            bin_ids = binner.transform_values(raw_values).flatten()
            observed = torch.bincount(bin_ids, minlength=num_effective_bins)
            if observed.numel() != num_effective_bins:
                raise RuntimeError(
                    "Sensitive bin IDs exceeded the fitted effective-bin range."
                )

            minibatch_size = int(bin_ids.numel())
            observed_counts = observed.cpu().numpy()

        fit_total = float(fit_counts.sum())
        expected_counts = fit_counts / fit_total * minibatch_size
        epoch = int(trainer.current_epoch)
        global_step = int(trainer.global_step)
        title = (
            "Fixed MI bin occupancy | "
            f"epoch={epoch}, batch={int(batch_idx)}, step={global_step}, "
            f"effective bins={num_effective_bins}, "
            f"minibatch size={minibatch_size}"
        )
        filename = (
            f"bin_occupancy_epoch{epoch:04d}_batch{int(batch_idx):06d}_"
            f"step{global_step:09d}_bins{num_effective_bins:04d}_"
            f"n{minibatch_size:06d}.png"
        )

        plot_categorical_bin_counts(
            observed_counts,
            self._output_dir(trainer) / filename,
            title=title,
            expected_counts=expected_counts,
            expected_label="Expected from full training-set proportions",
            xlabel="Bin ID",
            ylabel="Number of events in minibatch",
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Render the per-minibatch raw-value-diversity distribution."""
        if not self._is_scheduled_epoch(trainer) or not self._is_global_zero(trainer):
            return
        if not self._unique_counts:
            return

        epoch = int(trainer.current_epoch)
        num_minibatches = len(self._unique_counts)
        nominal_batch_size = self._nominal_batch_size
        if nominal_batch_size is None:
            nominal_batch_size = 0

        title = (
            f"Finite raw {pl_module.sensitive_binner.variable} diversity | "
            f"epoch={epoch}, minibatches={num_minibatches}, "
            f"nominal minibatch size={nominal_batch_size}"
        )
        filename = (
            f"raw_value_diversity_epoch{epoch:04d}_"
            f"batches{num_minibatches:06d}_nominal_n{nominal_batch_size:06d}.png"
        )

        plot_minibatch_scalar_histogram(
            self._unique_counts,
            self._output_dir(trainer) / filename,
            title=title,
            xlabel=(
                f"Number of unique finite raw "
                f"{pl_module.sensitive_binner.variable} values per minibatch"
            ),
            ylabel="Number of minibatches",
        )
        self._unique_counts = []

    def _is_scheduled_epoch(self, trainer) -> bool:
        return self.enabled and int(trainer.current_epoch) in self.epochs

    @staticmethod
    def _is_global_zero(trainer) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def _output_dir(self, trainer) -> Path:
        return Path(trainer.default_root_dir) / self.output_subdir

    @staticmethod
    def _get_nominal_batch_size(trainer) -> int | None:
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None

        per_device = getattr(datamodule, "batch_size_per_device", None)
        if per_device is not None:
            return int(per_device)

        hparams = getattr(datamodule, "hparams", None)
        configured = getattr(hparams, "batch_size", None)
        return int(configured) if configured is not None else None

    @staticmethod
    def _validate_fitted_binner(pl_module):
        binner = getattr(pl_module, "sensitive_binner", None)
        if binner is None:
            raise RuntimeError(
                "BinningDiagnosticsCallback requires pl_module.sensitive_binner."
            )
        if not binner.is_fitted:
            raise RuntimeError(
                "BinningDiagnosticsCallback only uses an already fitted sensitive "
                "binner; no fixed bin edges are available."
            )

        stats = binner.fit_stats
        num_effective_bins = int(stats.get("num_bins_effective", 0))
        edge_bins = int(binner.bin_edges.numel()) + 1
        fit_counts = np.asarray(stats.get("counts", []), dtype=float)

        if num_effective_bins != edge_bins:
            raise RuntimeError(
                "fit_stats['num_bins_effective'] does not match the fixed bin edges."
            )
        if fit_counts.shape != (num_effective_bins,):
            raise RuntimeError(
                "fit_stats['counts'] must contain one count per effective bin."
            )
        if not np.all(np.isfinite(fit_counts)) or np.any(fit_counts < 0):
            raise RuntimeError("fit_stats['counts'] must be finite and non-negative.")
        if fit_counts.sum() <= 0:
            raise RuntimeError("fit_stats['counts'] must have a positive total.")

        return binner, num_effective_bins, fit_counts

    @staticmethod
    def _validate_indices(values: Sequence[int], name: str) -> tuple[int, ...]:
        indices = tuple(int(value) for value in values)
        if any(value < 0 for value in indices):
            raise ValueError(f"{name} must contain only non-negative integers.")
        return indices
