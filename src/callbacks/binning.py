"""Lightweight training diagnostics for fixed MI-sensitive binning."""

import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers import MLFlowLogger

from src.callbacks.utils import mlflow_plot_gallery
from src.plot.histogram import (
    plot_categorical_bin_counts,
    plot_fixed_bin_widths,
    plot_histogram_counts,
    plot_minibatch_scalar_histogram,
    save_plot_data_csv,
)


class BinningDiagnosticsCallback(Callback):
    """Plot fixed-bin occupancy and raw-value diversity during selected epochs."""

    def __init__(
        self,
        enabled: bool = False,
        epochs: Sequence[int] = (0,),
        include_last_epoch: bool = False,
        batch_indices: Sequence[int] = (0, 100, 400, 764),
        output_root_dir: str | Path | None = None,
        output_subdir: str = "mi_diagnostics",
        checkpoint_root_dir: str | Path | None = None,
        log_to_mlflow: bool = False,
        mlflow_artifact_path: str | None = None,
        raw_histogram_bins: int = 50,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.epochs = frozenset(self._validate_indices(epochs, "epochs"))
        self.include_last_epoch = bool(include_last_epoch)
        self.batch_indices = frozenset(
            self._validate_indices(batch_indices, "batch_indices")
        )

        subdir = Path(output_subdir)
        if subdir.is_absolute() or ".." in subdir.parts:
            raise ValueError(
                "output_subdir must be a relative path within the output root."
            )
        self.output_root_dir = (
            Path(output_root_dir) if output_root_dir is not None else None
        )
        self.output_subdir = subdir
        self.checkpoint_root_dir = (
            Path(checkpoint_root_dir) if checkpoint_root_dir is not None else None
        )
        self.log_to_mlflow = bool(log_to_mlflow)
        self.raw_histogram_bins = int(raw_histogram_bins)
        if self.raw_histogram_bins < 1:
            raise ValueError("raw_histogram_bins must be at least 1.")
        self.mlflow_artifact_path = (
            mlflow_artifact_path
            if mlflow_artifact_path is not None
            else self.output_subdir.as_posix()
        )

        self._unique_counts: list[int] = []
        self._nominal_batch_size: int | None = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Reset scalar collection and validate fixed-binner state when scheduled."""
        self._unique_counts = []
        self._nominal_batch_size = None

        if not self._is_scheduled_epoch(trainer) or not self._is_global_zero(trainer):
            return

        binner, _, _ = self._validate_fitted_binner(pl_module)
        self._nominal_batch_size = self._get_nominal_batch_size(trainer)

        epoch = int(trainer.current_epoch)
        bin_widths = self._fixed_bin_widths(binner)
        filename = f"mi_bin_widths_epoch{epoch:04d}.png"
        output_path = plot_fixed_bin_widths(
            bin_widths,
            self._output_dir(trainer) / filename,
            title=f"Fixed MI bin widths: epoch = {epoch}",
            ylabel=f"Bin width Δ{binner.variable}",
        )
        self._publish_artifact(trainer, output_path)

        csv_path = save_plot_data_csv(
            {
                "epoch": np.full(bin_widths.size, epoch),
                "bin_id": np.arange(bin_widths.size),
                "bin_width": bin_widths,
            },
            self._data_dir(trainer) / Path(filename).with_suffix(".csv"),
        )
        self._publish_artifact(trainer, csv_path)

        if epoch == 0:
            self._plot_full_training_distribution(trainer, binner)

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

            unique_values = torch.unique(finite_values)
            unique_bin_ids = binner.transform_values(unique_values).flatten()
            unique_counts = torch.bincount(
                unique_bin_ids,
                minlength=num_effective_bins,
            )
            if unique_counts.numel() != num_effective_bins:
                raise RuntimeError(
                    "Unique sensitive values exceeded the fitted "
                    "effective-bin range."
                )
            unique_counts_per_bin = unique_counts.cpu().numpy()

            (
                raw_histogram_counts,
                raw_histogram_edges,
                raw_histogram_stats,
            ) = self._raw_histogram(finite_values)

        fit_total = float(fit_counts.sum())
        expected_counts = fit_counts / fit_total * minibatch_size
        epoch = int(trainer.current_epoch)
        global_step = int(trainer.global_step)
        title = f"MI bin occupancy: batch = {int(batch_idx)}"
        metadata = {
            "Epoch": epoch,
            "Global step": global_step,
            "Effective bins": num_effective_bins,
            "Minibatch size": minibatch_size,
        }
        filename = (
            f"mi_bin_occupancy_batch_{int(batch_idx)}_"
            f"epoch{epoch:04d}.png"
        )

        output_path = plot_categorical_bin_counts(
            observed_counts,
            self._output_dir(trainer) / filename,
            title=title,
            expected_counts=expected_counts,
            expected_label="Expected from full training-set proportions",
            xlabel="Bin ID",
            ylabel="Number of events in minibatch",
            metadata=metadata,
        )
        self._publish_artifact(trainer, output_path)

        occupancy_csv_path = save_plot_data_csv(
            {
                "epoch": np.full(num_effective_bins, epoch),
                "batch_index": np.full(num_effective_bins, int(batch_idx)),
                "global_step": np.full(num_effective_bins, global_step),
                "bin_id": np.arange(num_effective_bins),
                "observed_count": observed_counts,
                "expected_count": expected_counts,
            },
            self._data_dir(trainer) / Path(filename).with_suffix(".csv"),
        )
        self._publish_artifact(trainer, occupancy_csv_path)

        variable = binner.variable
        unique_filename = (
            f"mi_bin_unique_values_batch_{int(batch_idx)}_"
            f"epoch{epoch:04d}.png"
        )
        unique_output_path = plot_categorical_bin_counts(
            unique_counts_per_bin,
            self._output_dir(trainer) / unique_filename,
            title=(
                f"Unique {variable} values per MI bin: "
                f"batch = {int(batch_idx)}"
            ),
            observed_label=f"Unique finite {variable} values in minibatch",
            xlabel="Bin ID",
            ylabel=f"Number of unique {variable} values",
            metadata=metadata,
            y_axis_max=20,
            annotate_clipped_values=True,
            integer_y_ticks=True,
        )
        self._publish_artifact(trainer, unique_output_path)

        unique_csv_path = save_plot_data_csv(
            {
                "epoch": np.full(num_effective_bins, epoch),
                "batch_index": np.full(num_effective_bins, int(batch_idx)),
                "global_step": np.full(num_effective_bins, global_step),
                "bin_id": np.arange(num_effective_bins),
                "unique_finite_value_count": unique_counts_per_bin,
            },
            self._data_dir(trainer) / Path(unique_filename).with_suffix(".csv"),
        )
        self._publish_artifact(trainer, unique_csv_path)

        raw_filename = (
            f"raw_fet_et_histogram_batch_{int(batch_idx)}_"
            f"epoch{epoch:04d}.png"
        )
        raw_output_path = plot_histogram_counts(
            raw_histogram_counts,
            raw_histogram_edges,
            self._output_dir(trainer) / raw_filename,
            title=f"Raw {variable} distribution: batch = {int(batch_idx)}",
            xlabel=f"Raw {variable} value",
            ylabel="Number of events",
            metadata={
                **metadata,
                "Finite values": raw_histogram_stats["finite_values"],
                "Histogram bins": self.raw_histogram_bins,
                "Min": raw_histogram_stats["min"],
                "Max": raw_histogram_stats["max"],
                "Mean": raw_histogram_stats["mean"],
                "Std": raw_histogram_stats["std"],
            },
        )
        self._publish_artifact(trainer, raw_output_path)

        num_histogram_bins = raw_histogram_counts.size
        raw_csv_path = save_plot_data_csv(
            {
                "epoch": np.full(num_histogram_bins, epoch),
                "batch_index": np.full(
                    num_histogram_bins,
                    int(batch_idx),
                ),
                "global_step": np.full(num_histogram_bins, global_step),
                "bin_left": raw_histogram_edges[:-1],
                "bin_right": raw_histogram_edges[1:],
                "count": raw_histogram_counts,
                "finite_values": np.full(
                    num_histogram_bins,
                    raw_histogram_stats["finite_values"],
                ),
                "min": np.full(
                    num_histogram_bins,
                    raw_histogram_stats["min"],
                ),
                "max": np.full(
                    num_histogram_bins,
                    raw_histogram_stats["max"],
                ),
                "mean": np.full(
                    num_histogram_bins,
                    raw_histogram_stats["mean"],
                ),
                "std": np.full(
                    num_histogram_bins,
                    raw_histogram_stats["std"],
                ),
            },
            self._data_dir(trainer) / Path(raw_filename).with_suffix(".csv"),
        )
        self._publish_artifact(trainer, raw_csv_path)

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

        variable = pl_module.sensitive_binner.variable
        title = f"Raw {variable} diversity: epoch {epoch}"
        filename = (
            f"raw_value_diversity_epoch{epoch:04d}_"
            f"batches{num_minibatches:06d}_nominal_n{nominal_batch_size:06d}.png"
        )

        output_path = plot_minibatch_scalar_histogram(
            self._unique_counts,
            self._output_dir(trainer) / filename,
            title=title,
            xlabel="Minibatch number",
            ylabel=f"Number of unique {variable} values",
        )
        self._publish_artifact(trainer, output_path)

        diversity_csv_path = save_plot_data_csv(
            {
                "epoch": np.full(num_minibatches, epoch),
                "minibatch_index": np.arange(num_minibatches),
                "unique_finite_value_count": np.asarray(self._unique_counts),
                "num_minibatches": np.full(num_minibatches, num_minibatches),
                "nominal_minibatch_size": np.full(
                    num_minibatches,
                    nominal_batch_size,
                ),
            },
            self._data_dir(trainer) / Path(filename).with_suffix(".csv"),
        )
        self._publish_artifact(trainer, diversity_csv_path)
        self._log_mlflow_gallery(trainer)
        self._unique_counts = []

    def _is_scheduled_epoch(self, trainer) -> bool:
        if not self.enabled:
            return False

        epoch = int(trainer.current_epoch)
        if epoch in self.epochs:
            return True
        if not self.include_last_epoch:
            return False

        max_epochs = getattr(trainer, "max_epochs", None)
        if max_epochs is None:
            return False
        max_epochs = int(max_epochs)
        return max_epochs > 0 and epoch == max_epochs - 1

    @staticmethod
    def _is_global_zero(trainer) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def _output_dir(self, trainer) -> Path:
        return self._output_root(trainer) / self._epoch_subdir(trainer)

    def _data_dir(self, trainer) -> Path:
        return self._output_root(trainer) / "data" / self._epoch_subdir(trainer)

    def _output_root(self, trainer) -> Path:
        output_root = self.output_root_dir
        if output_root is None:
            output_root = Path(trainer.default_root_dir)
        return output_root / self.output_subdir

    def _checkpoint_output_root(self) -> Path | None:
        if self.checkpoint_root_dir is None:
            return None
        return self.checkpoint_root_dir / self.output_subdir

    @staticmethod
    def _epoch_subdir(trainer) -> Path:
        return Path(f"epoch_{int(trainer.current_epoch):04d}")

    def _publish_artifact(self, trainer, output_path: Path) -> None:
        """Copy a diagnostic artifact to checkpoints and upload it to MLflow."""
        relative_path = output_path.relative_to(self._output_root(trainer))
        artifact_source = output_path
        checkpoint_output_root = self._checkpoint_output_root()
        if checkpoint_output_root is not None:
            artifact_source = checkpoint_output_root / relative_path
            artifact_source.parent.mkdir(parents=True, exist_ok=True)
            if output_path.resolve() != artifact_source.resolve():
                shutil.copy2(output_path, artifact_source)

        if not self.log_to_mlflow:
            return

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        mlflow_logger.experiment.log_artifact(
            run_id=mlflow_logger.run_id,
            local_path=str(artifact_source),
            artifact_path=str(
                Path(self.mlflow_artifact_path) / relative_path.parent
            ),
        )

    def _log_mlflow_gallery(self, trainer) -> None:
        """Upload an HTML gallery containing all diagnostics generated so far."""
        if not self.log_to_mlflow:
            return

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        plot_dir = self._checkpoint_output_root() or self._output_root(trainer)
        html_gallery = mlflow_plot_gallery.build_html(
            plot_dir,
            title="MI Binning Diagnostics",
        )
        mlflow_logger.experiment.log_text(
            run_id=mlflow_logger.run_id,
            text=html_gallery,
            artifact_file=str(Path(self.mlflow_artifact_path) / "index.html"),
        )

    @staticmethod
    def _get_mlflow_logger(trainer) -> MLFlowLogger | None:
        logger = getattr(trainer, "logger", None)
        if isinstance(logger, MLFlowLogger):
            return logger

        for candidate in getattr(trainer, "loggers", []) or []:
            if isinstance(candidate, MLFlowLogger):
                return candidate

        return None

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

    def _plot_full_training_distribution(self, trainer, binner) -> None:
        """Render the fitted full-training raw-value histogram once at epoch 0."""
        stats = binner.fit_stats
        counts = np.asarray(stats.get("raw_histogram_counts", []), dtype=float)
        edges = np.asarray(stats.get("raw_histogram_edges", []), dtype=float)
        if counts.ndim != 1 or counts.size == 0:
            raise RuntimeError(
                "fit_stats must contain non-empty 'raw_histogram_counts'."
            )
        if edges.shape != (counts.size + 1,):
            raise RuntimeError(
                "fit_stats['raw_histogram_edges'] must contain one more "
                "value than 'raw_histogram_counts'."
            )

        epoch = int(trainer.current_epoch)
        variable = binner.variable
        filename = f"full_fet_et_histogram_epoch{epoch:04d}.png"
        metadata = {
            "Epoch": epoch,
            "Training values": int(stats["num_values"]),
            "Histogram bins": int(counts.size),
            "Min": f"{float(stats['min']):.6g}",
            "Max": f"{float(stats['max']):.6g}",
            "Mean": f"{float(stats['mean']):.6g}",
            "Std": f"{float(stats['std']):.6g}",
        }
        output_path = plot_histogram_counts(
            counts,
            edges,
            self._output_dir(trainer) / filename,
            title=f"Full training {variable} distribution",
            xlabel=f"Raw {variable} value",
            ylabel="Number of training events",
            metadata=metadata,
        )
        self._publish_artifact(trainer, output_path)

        num_histogram_bins = counts.size
        csv_path = save_plot_data_csv(
            {
                "epoch": np.full(num_histogram_bins, epoch),
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "count": counts,
                "training_values": np.full(
                    num_histogram_bins,
                    metadata["Training values"],
                ),
                "min": np.full(num_histogram_bins, metadata["Min"]),
                "max": np.full(num_histogram_bins, metadata["Max"]),
                "mean": np.full(num_histogram_bins, metadata["Mean"]),
                "std": np.full(num_histogram_bins, metadata["Std"]),
            },
            self._data_dir(trainer) / Path(filename).with_suffix(".csv"),
        )
        self._publish_artifact(trainer, csv_path)

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

    @staticmethod
    def _fixed_bin_widths(binner) -> np.ndarray:
        """Return finite bin widths bounded by the fitted training range."""
        stats = binner.fit_stats
        try:
            fitted_min = float(stats["min"])
            fitted_max = float(stats["max"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "fit_stats must contain finite 'min' and 'max' values "
                "to plot fixed MI bin widths."
            ) from error

        internal_edges = binner.bin_edges.detach().cpu().numpy().astype(float)
        boundaries = np.concatenate(
            ([fitted_min], internal_edges, [fitted_max])
        )
        widths = np.diff(boundaries)
        if not np.all(np.isfinite(widths)) or np.any(widths < 0):
            raise RuntimeError(
                "Fitted training bounds and fixed bin edges must be finite "
                "and monotonically increasing."
            )
        return widths

    def _raw_histogram(
        self,
        finite_values: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
        """Compute raw-value histogram counts on-device and return small arrays."""
        if finite_values.numel() == 0:
            stats: dict[str, int | str] = {
                "finite_values": 0,
                "min": "n/a",
                "max": "n/a",
                "mean": "n/a",
                "std": "n/a",
            }
            counts = torch.zeros(
                self.raw_histogram_bins,
                device=finite_values.device,
                dtype=torch.float32,
            )
            edges = torch.linspace(
                0.0,
                1.0,
                steps=self.raw_histogram_bins + 1,
                device=finite_values.device,
            )
        else:
            histogram_values = finite_values.float()
            value_min = float(histogram_values.min().item())
            value_max = float(histogram_values.max().item())
            stats = {
                "finite_values": int(histogram_values.numel()),
                "min": f"{value_min:.6g}",
                "max": f"{value_max:.6g}",
                "mean": f"{float(histogram_values.mean().item()):.6g}",
                "std": (
                    f"{float(histogram_values.std(unbiased=False).item()):.6g}"
                ),
            }
            if value_min == value_max:
                half_width = max(abs(value_min) * 1e-3, 1e-6)
                value_min -= half_width
                value_max += half_width

            counts = torch.histc(
                histogram_values,
                bins=self.raw_histogram_bins,
                min=value_min,
                max=value_max,
            )
            edges = torch.linspace(
                value_min,
                value_max,
                steps=self.raw_histogram_bins + 1,
                device=finite_values.device,
                dtype=histogram_values.dtype,
            )

        return counts.cpu().numpy(), edges.cpu().numpy(), stats
