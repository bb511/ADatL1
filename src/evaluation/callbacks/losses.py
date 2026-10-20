# Plot the training loss histories together and log the PNG to MLflow.
from pathlib import Path

from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks import utils
from src.plot import loss_curves
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class LossesCallback(Callback):
    """Plot and summarize MI, reconstruction, and total training losses.

    At the end of training, the callback reads ``train/loss_mi``,
    ``train/loss_reco``, and ``train/loss`` from the active MLflow run. Each history
    is plotted independently normalized to [0, 1]. A second figure shows the original
    reconstruction loss and the gamma-weighted MI contribution, ``gamma * loss_mi``,
    on the left y-axis, with total loss on a separate right y-axis. The line-only
    figures are saved as PNGs in the run's checkpoint plot folder and logged as
    MLflow run artifacts. ``train/loss`` is labelled ``Loss_total`` because it is the
    total objective used for backpropagation. The callback also logs the minimum MI
    loss and its same-step reconstruction loss, plus the minimum reconstruction loss
    and its same-step MI loss, as scalar MLflow metrics. The figures themselves are
    drawn by ``src.plot.loss_curves``, which ``redo_loss_plots.py`` also uses.

    :param include_loss_mi: Include the mutual-information loss.
    :param include_loss_reco: Include the reconstruction loss.
    :param include_loss_total: Include the total training objective.
    :param gamma: Gamma value shown in the title. If omitted, read ``mi_gamma`` from
        the Lightning module.
    :param include_mi_null: Also plot the MI loss against its permutation noise
        floor (``loss_mi_permuted``) and the analytic floor, for train and the
        normal validation split, when those metrics were logged. Skipped
        silently if the AE ran with ``mi_num_permutations=0``.
    :param name: Callback output folder and MLflow artifact folder.
    :param log_raw_mlflow: Whether to upload the generated PNG to MLflow.
    """

    def __init__(
        self,
        include_loss_mi: bool = True,
        include_loss_reco: bool = True,
        include_loss_total: bool = True,
        include_mi_null: bool = True,
        gamma: float | None = None,
        name: str = "losses",
        log_raw_mlflow: bool = True,
    ):
        super().__init__()
        self.include_loss_mi = include_loss_mi
        self.include_loss_reco = include_loss_reco
        self.include_loss_total = include_loss_total
        self.include_mi_null = include_mi_null
        self.gamma = gamma
        self.name = name
        self.log_raw_mlflow = log_raw_mlflow

        if not any(
            (self.include_loss_mi, self.include_loss_reco, self.include_loss_total)
        ):
            raise ValueError("At least one loss must be enabled for LossesCallback.")

    def on_train_end(self, trainer, pl_module) -> None:
        """Log loss summaries and plots after the training loop finishes."""
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
        step_histories = {}
        for metric_name, _, enabled in self._configured_metrics():
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
            step_histories[metric_name] = self._history_by_step(metric_history)

        if self.include_loss_mi and self.include_loss_reco:
            minimum_metrics, final_step = self._paired_minimum_metrics(
                mi_by_step=step_histories["loss_mi"],
                reco_by_step=step_histories["loss_reco"],
            )
            mlflow_logger.log_metrics(minimum_metrics, step=final_step)

        plot_folder = self._resolve_plot_folder(trainer)
        plot_paths = loss_curves.plot_training_losses(
            step_histories, gamma, plot_folder
        )

        if self.include_mi_null:
            mi_null_path = self._plot_mi_null(mlflow_logger, plot_folder)
            if mi_null_path is not None:
                plot_paths.append(mi_null_path)

        if self.log_raw_mlflow:
            for plot_path in plot_paths:
                mlflow_logger.experiment.log_artifact(
                    run_id=mlflow_logger.run_id,
                    local_path=str(plot_path),
                    artifact_path=self.name,
                )

    # Legend label -> (MLflow metric key, colour, line opacity)
    MI_NULL_SERIES = loss_curves.MI_NULL_SERIES

    def _plot_mi_null(self, mlflow_logger, plot_folder: Path) -> Path | None:
        """Plot MI next to its permutation noise floor; None if not logged."""
        path = loss_curves.plot_mi_null(
            lambda key: self._optional_history(mlflow_logger, key),
            plot_folder,
        )
        if path is None:
            log.info(
                "No 'train/loss_mi_permuted' history (mi_num_permutations=0?); "
                "skipping the MI noise-floor plot."
            )
        return path

    def _optional_history(self, mlflow_logger, key: str) -> dict[int, float] | None:
        """Step-aligned history of ``key``, or None if absent or not finite."""
        history = mlflow_logger.experiment.get_metric_history(
            mlflow_logger.run_id,
            key,
        )
        if not history:
            return None
        try:
            return self._history_by_step(history)
        except ValueError:
            log.warning(f"MLflow metric '{key}' has non-finite values; not plotted.")
            return None

    def _configured_metrics(self) -> tuple[tuple[str, str, bool], ...]:
        """Map stored MLflow metric names to the labels shown in the plot."""
        return (
            ("loss_mi", "Loss_mi", self.include_loss_mi),
            ("loss_reco", "Loss_reco", self.include_loss_reco),
            ("loss", "Loss_total", self.include_loss_total),
        )

    _history_by_step = staticmethod(loss_curves.history_by_step)

    @staticmethod
    def _paired_minimum_metrics(
        mi_by_step: dict[int, float],
        reco_by_step: dict[int, float],
    ) -> tuple[dict[str, float], int]:
        """Compute both minima and counterpart losses at the exact same steps."""
        mi_steps = set(mi_by_step)
        reco_steps = set(reco_by_step)
        if mi_steps != reco_steps:
            missing_reco = sorted(mi_steps - reco_steps)
            missing_mi = sorted(reco_steps - mi_steps)
            raise RuntimeError(
                "MLflow training-loss histories are not step-aligned; cannot select "
                "corresponding losses exactly. "
                f"Missing loss_reco steps: {missing_reco}; "
                f"missing loss_mi steps: {missing_mi}."
            )
        if not mi_steps:
            raise ValueError("Training-loss histories must not be empty.")

        ordered_steps = sorted(mi_steps)
        min_mi_step = min(ordered_steps, key=lambda step: mi_by_step[step])
        min_reco_step = min(ordered_steps, key=lambda step: reco_by_step[step])

        return (
            {
                "train/min_loss_mi": mi_by_step[min_mi_step],
                "train/loss_reco_at_min_loss_mi": reco_by_step[min_mi_step],
                "train/min_loss_reco": reco_by_step[min_reco_step],
                "train/loss_mi_at_min_loss_reco": mi_by_step[min_reco_step],
            },
            ordered_steps[-1],
        )

    _normalize = staticmethod(loss_curves.normalize)

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
