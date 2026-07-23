# Plot the training loss histories together and log the PNG to MLflow.
from pathlib import Path

import numpy as np
from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks import utils
from src.plot import scatter
from src.plot.lossplotter import LOSS_COLORS
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class LossesCallback(Callback):
    """Plot MI, reconstruction, and total training losses together.

    At the end of training, the callback reads ``train/loss_mi``,
    ``train/loss_reco``, and ``train/loss`` from the active MLflow run. Each history
    is plotted independently normalized to [0, 1]. A second figure shows the original
    reconstruction loss and the gamma-weighted MI contribution, ``gamma * loss_mi``,
    on the left y-axis, with total loss on a separate right y-axis. The line-only
    figures are saved as PNGs in the run's checkpoint plot folder and logged as
    MLflow run artifacts. ``train/loss`` is labelled ``Loss_total`` because it is the
    total objective used for backpropagation.

    :param include_loss_mi: Include the mutual-information loss.
    :param include_loss_reco: Include the reconstruction loss.
    :param include_loss_total: Include the total training objective.
    :param gamma: Gamma value shown in the title. If omitted, read ``mi_gamma`` from
        the Lightning module.
    :param name: Callback output folder and MLflow artifact folder.
    :param log_raw_mlflow: Whether to upload the generated PNG to MLflow.
    """

    def __init__(
        self,
        include_loss_mi: bool = True,
        include_loss_reco: bool = True,
        include_loss_total: bool = True,
        gamma: float | None = None,
        name: str = "losses",
        log_raw_mlflow: bool = True,
    ):
        super().__init__()
        self.include_loss_mi = include_loss_mi
        self.include_loss_reco = include_loss_reco
        self.include_loss_total = include_loss_total
        self.gamma = gamma
        self.name = name
        self.log_raw_mlflow = log_raw_mlflow

        if not any(
            (self.include_loss_mi, self.include_loss_reco, self.include_loss_total)
        ):
            raise ValueError("At least one loss must be enabled for LossesCallback.")

    def on_train_end(self, trainer, pl_module) -> None:
        """Create and log the combined loss plot after the training loop finishes."""
        if not trainer.is_global_zero:
            return

        mlflow_logger = utils.mlflow.get_mlflow_logger(trainer)
        if mlflow_logger is None:
            log.warning(
                "LossesCallback requires an MLflow logger to read the completed "
                "metric histories; skipping the combined loss plot."
            )
            return

        gamma = self._resolve_gamma(pl_module)
        gamma_title = f"{gamma:g}"
        raw_histories = {}
        normalized_histories = {}
        raw_colors = {}
        normalized_colors = {}
        for metric_name, label, enabled in self._configured_metrics():
            if not enabled:
                continue

            metric_history = mlflow_logger.experiment.get_metric_history(
                mlflow_logger.run_id,
                f"train/{metric_name}",
            )
            if not metric_history:
                raise RuntimeError(
                    f"MLflow metric 'train/{metric_name}' has no history. "
                    "Cannot create the combined training-loss plot."
                )

            values = self._history_values(metric_history)
            normalized = self._normalize(values)
            raw_label = "Loss_mi * gamma" if metric_name == "loss_mi" else label
            raw_values = values * gamma if metric_name == "loss_mi" else values
            raw_histories[raw_label] = {
                epoch: float(value) for epoch, value in enumerate(raw_values, start=1)
            }
            normalized_histories[label] = {
                epoch: float(value)
                for epoch, value in enumerate(normalized, start=1)
            }
            raw_colors[raw_label] = LOSS_COLORS[label]
            normalized_colors[label] = LOSS_COLORS[label]

        plot_folder = self._resolve_plot_folder(trainer)
        raw_component_histories = {
            label: history
            for label, history in raw_histories.items()
            if label != "Loss_total"
        }
        raw_total_history = {
            label: history
            for label, history in raw_histories.items()
            if label == "Loss_total"
        }
        plot_paths = [
            scatter.plot_lines(
                data=normalized_histories,
                xlabel="Epoch",
                ylabel="Normalized loss",
                title=f"Normalized training losses (gamma = {gamma_title})",
                save_dir=plot_folder,
                colors=normalized_colors,
                filename="training_losses.png",
                alphas={"Loss_total": 0.8},
            ),
            scatter.plot_lines(
                data=raw_component_histories,
                xlabel="Epoch",
                ylabel="Loss_mi * gamma & Loss_reco",
                title=f"Unnormalized training losses (gamma = {gamma_title})",
                save_dir=plot_folder,
                colors=raw_colors,
                filename="training_losses_unnormalized.png",
                right_axis_data=raw_total_history,
                right_ylabel="Loss_total",
                alphas={"Loss_total": 0.8},
            ),
        ]

        if self.log_raw_mlflow:
            for plot_path in plot_paths:
                mlflow_logger.experiment.log_artifact(
                    run_id=mlflow_logger.run_id,
                    local_path=str(plot_path),
                    artifact_path=self.name,
                )

    def _configured_metrics(self) -> tuple[tuple[str, str, bool], ...]:
        """Map stored MLflow metric names to the labels shown in the plot."""
        return (
            ("loss_mi", "Loss_mi", self.include_loss_mi),
            ("loss_reco", "Loss_reco", self.include_loss_reco),
            ("loss", "Loss_total", self.include_loss_total),
        )

    @staticmethod
    def _history_values(metric_history) -> np.ndarray:
        """Sort MLflow history by step and keep the last value for each step."""
        values_by_step = {}
        for point in sorted(metric_history, key=lambda item: (item.step, item.timestamp)):
            values_by_step[point.step] = point.value
        return np.asarray(list(values_by_step.values()), dtype=float)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        """Normalize a loss history to [0, 1], including constant histories."""
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Loss histories must contain finite numeric values.")

        minimum = values.min()
        span = values.max() - minimum
        if span == 0:
            return np.zeros_like(values)
        return (values - minimum) / span

    def _resolve_gamma(self, pl_module) -> float:
        """Resolve gamma for the weighted MI curve and plot titles."""
        gamma = self.gamma
        if gamma is None:
            gamma = getattr(pl_module, "mi_gamma", None)
        if gamma is None:
            gamma = getattr(getattr(pl_module, "hparams", None), "mi_gamma", None)
        if gamma is None:
            raise RuntimeError(
                "LossesCallback could not resolve gamma from its configuration or "
                "the Lightning module."
            )
        return float(gamma)

    def _resolve_plot_folder(self, trainer) -> Path:
        """Place the plot under the current run's checkpoint plot directory."""
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        checkpoint_dir = getattr(checkpoint_callback, "dirpath", None)
        if checkpoint_dir is None:
            checkpoint_dir = trainer.default_root_dir

        plot_folder = Path(checkpoint_dir) / "plots" / self.name
        plot_folder.mkdir(parents=True, exist_ok=True)
        return plot_folder
