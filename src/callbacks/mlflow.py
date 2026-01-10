# Log data set metadata to artifacts in mlflow.
from pathlib import Path

import optuna
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger, Logger
from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from .utils import mlflow_plot_gallery

from src.utils import pylogger
from colorama import Fore, Back, Style

log = pylogger.RankedLogger(__name__)


class LogDataInfoMlFlow(Callback):
    """Logs the figures pertaining to the data feature distributions to mlflow.

    This logs the feature distributions for each of the used data sets as artifacts
    in MLFlow. The data is logged at each step, i.e., extraction, processing, mlready.
    See the data module for more details about what each of these steps does.
    """

    def __init__(self):
        super().__init__()

    def _get_mlflow_logger(self, trainer: Trainer) -> MLFlowLogger:
        """Extract the MLFlow logger from the trainer, if it is being used."""
        logger = trainer.logger
        if isinstance(logger, MLFlowLogger):
            return logger

        if isinstance(logger, Logger):
            return None

        for logger in getattr(trainer, "loggers", []) or []:
            if isinstance(logger, MLFlowLogger):
                return logger

        return None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the data information into MLFlow at the start of the training."""
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            log.warn(Fore.YELLOW + "No MLFlow logger but using LogDataMlFlow callback!")
            return

        self._setup_data_paths(trainer)
        self._log_data_distriubtions(mlflow_logger)

    def _setup_data_paths(self, trainer: Trainer):
        """Get the paths to where plots of the data have been saved."""
        data_module = getattr(trainer, "datamodule", None)
        if data_module is None:
            raise ValueError(Fore.RED + "Data module not found in trainer!")

        self.mlready_path = data_module.hparams.data_mlready.cache_folder

    def _log_data_distriubtions(self, mlflow_logger: MLFlowLogger) -> None:
        """Log the input data distributions plotted with the data module."""
        mlready_data_plot_paths = self.mlready_path.rglob("PLOTS")
        run_id = mlflow_logger.run_id

        for path in mlready_data_plot_paths:
            arti_path = path.relative_to(self.mlready_path)
            arti_path = "data" / arti_path.parent
            mlflow_logger.experiment.log_artifact(run_id, path, artifact_path=arti_path)

            html_gallery = mlflow_plot_gallery.build_html(path)
            gallery_path = arti_path / "PLOTS"
            mlflow_logger.experiment.log_text(
                run_id, html_gallery, artifact_file=gallery_path / "index.html"
            )
