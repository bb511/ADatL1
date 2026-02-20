# Different ways of doing model checkpointing.
import os
from typing import Optional
from collections import defaultdict
from pathvalidate import sanitize_filename
import importlib.util

from weakref import proxy
from pathlib import Path
import copy
import shutil
import numpy as np

from pytorch_lightning.callbacks import Callback

from src.callbacks.checkpointing.criterion import Criterion
from src.plot import horizontal_bar
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class DatasetAwareModelCheckpoint(Callback):
    """ModelCheckpoint class to store dataset-specific checkpoints.

    :param monitor: String name of the metric key in the metrics log dictionary.
    :param mode: Callable method that takes the value of the metric at one epoch and
        decides whether to create a checkpoint or not.
    :param dirpath: String to dir path where to save the checkpoints.
    :param filename: String of the checkpoint filename template.
    :param ds: List of strings specifying which datasets to checkpoint on.
    """

    def __init__(
        self,
        monitor: str,
        criterion: Criterion,
        ds: list[str],
        dirpath: Optional[str] = "checkpoints",
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.criterion = criterion
        self.topk = len(criterion.top_k_values)
        self.ds = ds
        self.save_last = self.criterion.name == "last"

        self.checkpoints = defaultdict(list)
        self.ds_criterion = None

    def on_validation_start(self, trainer, pl_module):
        """Initialize the criterion object for each dataset at the start of training."""
        if not self.ds_criterion is None:
            return
        if getattr(trainer, "sanity_checking", False):
            return

        all_valid_datasets = set(getattr(trainer, "val_dataloaders").keys())
        if not set(self.ds).issubset(all_valid_datasets):
            raise ValueError(
                "Given ds to checkpoint on are not in the available ds:\n"
                f"Given ds: {list(self.ds)}.\n"
                f"Available ds: {list(all_valid_datasets)}."
            )

        self.ds_criterion = {}
        for ds_name in self.ds:
            self.ds_criterion[ds_name] = copy.deepcopy(self.criterion)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Get the metric at that epoch and decide whether to set a checkpoint there."""
        ds_metrics = {}
        for ds_name in self.ds:
            ds_metrics[ds_name] = self._get_metric(trainer.callback_metrics, ds_name)

        self._process_metric_across_datasets(trainer, pl_module, ds_metrics)

    def _metric_key(self, dataset_name: str) -> str:
        return f"val/{dataset_name}/{self.monitor}"

    def _get_metric(self, available_metrics: dict, dataset_name: str):
        """Cache the value of the specified metric for one given data set name.

        This is expecting the format of the metric name 'val/[dataset_name]/metric_name'
        make sure that this is how the callback metrics are named.
        """
        metric_key = self._metric_key(dataset_name)
        if metric_key in available_metrics:
            metric_value = available_metrics[metric_key]
            if hasattr(metric_value, "detach"):
                return metric_value.detach().cpu().float().item()
            else:
                return float(metric_value)

        log.warn(
            f"Checkpoint for metric '{self.monitor}' not found in logged metrics. "
            f"Did you call `log('{self.monitor}')` in the LightningModule "
            f"for {dataset_name}?",
        )
        return np.nan

    def _process_metric_across_datasets(self, trainer, pl_module, ds_metrics: dict):
        """Compute a summary metric across datasets, like leave one out."""
        raise NotImplementedError("Subclasses must implement _process_dataset_losses")

    def save_top_k(self, trainer, pl_module, dataset_name: str, metric_value: float):
        """Save a checkpoint if it's among the top_k best for the given dataset.

        How many top_k should be saved is configured in the criterion directly.
        """
        should_save = self.ds_criterion[dataset_name].check(metric_value)

        if should_save:
            filepath = self._configure_filepath(trainer, dataset_name, metric_value)
            self.checkpoints[dataset_name].append(
                {
                    "value": metric_value,
                    "epoch": trainer.current_epoch,
                    "fpath": filepath,
                }
            )
            self._save_checkpoint(trainer, pl_module, filepath)
            self._clean_checkpoints(dataset_name)

    def _save_checkpoint(self, trainer, pl_module, filepath: Path):
        """Save a checkpoint with a custom key and metric value."""
        if trainer.is_global_zero:
            trainer.save_checkpoint(filepath)

        # Save keras model separately if it is present.
        self._maybe_save_keras_models(pl_module, Path(filepath))

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    def _clean_checkpoints(self, dataset_name: str):
        """Makes sure that the saved checkpoints are just the top_k ones."""
        files_to_remove = [
            ckpt["fpath"]
            for ckpt in self.checkpoints[dataset_name]
            if (
                (not ckpt["value"] in self.ds_criterion[dataset_name].top_k_values)
                and (os.path.exists(ckpt["fpath"]))
            )
        ]

        self.checkpoints[dataset_name] = [
            ckpt
            for ckpt in self.checkpoints[dataset_name]
            if ckpt["value"] in self.ds_criterion[dataset_name].top_k_values
        ]

        for filepath in files_to_remove:
            os.remove(filepath)

    def create_checkpoint_dir(self):
        """Create the directory where the checkpoints are saved."""
        shutil.rmtree(self.dirpath, ignore_errors=True)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_fit_end(self, trainer, pl_module):
        """Plot results. Save the models obtained at the end of the training."""
        self._plot_epochs()
        if not self.save_last:
            return

        self._ckpt_last_epoch(trainer, pl_module)

    def _ckpt_last_epoch(self, trainer, pl_module):
        """Checkpoint the model at the last epoch of the training."""
        ds_metrics = {}
        for ds_name in self.ds:
            metric_value = self._get_metric(trainer.callback_metrics, ds_name)
            ds_metrics[ds_name] = metric_value

        plot_folder = self.dirpath / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)
        xlabel = f"{self.monitor}"
        horizontal_bar.plot(ds_metrics, xlabel, plot_folder)

        self._save_checkpoint(trainer, pl_module, self.dirpath / "last_epoch.ckpt")

    def _configure_filepath(self, trainer, ds_name: str, metric_val: float) -> Path:
        """Construct the filename out of the given properties of the checkpoint."""
        filename = "ds={dataset_name}__metric={metric_name}__value={metric_value:.6f}"
        filename = filename + "__epoch={epoch:02d}"

        filename = filename.format(
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            dataset_name=ds_name,
            metric_name=self.monitor,
            metric_value=metric_val,
        )

        filename = sanitize_filename(filename)
        filename = filename + ".ckpt"
        return self.dirpath / filename

    def _plot_epochs(self):
        """Plots the epochs at which the best checkpoints are saved.

        Best checkpoints are defined in the criterion, which is specified by the user.
        """
        plot_folder = self.dirpath / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)
        epochs = defaultdict(int)
        values = defaultdict(float)

        for k in range(self.topk):
            for dataset_name in self.checkpoints.keys():
                if not self.checkpoints[dataset_name]:
                    log.warn(
                        f"No checkpoints made for  {self.criterion.__dict__} "
                        f"on dataset {dataset_name}."
                    )
                    continue
                epochs[dataset_name] = self.checkpoints[dataset_name][k]["epoch"]
                values[dataset_name] = self.checkpoints[dataset_name][k]["value"]

            if not epochs:
                continue

            xlabel = f"top {k+1} epoch"
            ylabel = f"metric value"
            horizontal_bar.plot_yright(epochs, values, xlabel, ylabel, plot_folder)

    def _maybe_save_keras_models(self, pl_module, ckpt_path: Path) -> None:
        """Checkpoint keras model if it is found in the model definition."""
        if importlib.util.find_spec("keras") is None:
            return

        import keras  # safe now

        keras_model_cls = getattr(keras, "Model", None)
        if keras_model_cls is None:
            return

        for _, v in vars(pl_module).items():
            if isinstance(v, keras_model_cls):
                v.save(str(ckpt_path.with_suffix(".keras")))
            elif isinstance(v, dict):
                for name, vv in v.items():
                    if isinstance(vv, keras_model_cls):
                        vv.save(
                            str(ckpt_path.with_suffix("")) + f"__component={name}.keras"
                        )
