# Different ways of doing model checkpointing.
from typing import Optional
from pathlib import Path
import os
import yaml
import shutil

import copy
from pytorch_lightning.callbacks import Callback

from src.callbacks.checkpointing.dataset_aware import DatasetAwareModelCheckpoint


class SingleDatasetModelCheckpoint(DatasetAwareModelCheckpoint):
    """ModelCheckpoint that saves checkpoints based on individual dataset performance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dirpath = self.dirpath / "single" / self.monitor / self.criterion.name
        self.create_checkpoint_dir()

    def _process_metric_across_datasets(self, trainer, pl_module, ds_metrics: dict):
        """Process the metric for each data set separately."""
        for dataset_name, metric_value in ds_metrics.items():
            self.save_top_k(trainer, pl_module, dataset_name, metric_value)


class LeaveOneOutModelCheckpoint(DatasetAwareModelCheckpoint):
    """ModelCheckpoint that saves checkpoints based on leave-one-out dataset performance.

    For each dataset, this callback computes the mean metric on all OTHER datasets
    (leave-one-out) and saves the checkpoint that performs best according to this metric.
    This helps avoid overfitting to specific anomalies in any single dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dirpath = self.dirpath / "loo" / self.monitor / self.criterion.name
        self.create_checkpoint_dir()

    def _process_metric_across_datasets(self, trainer, pl_module, ds_metrics: dict):
        """Process the metric according to a leave one out strategy."""
        for dataset_name in ds_metrics.keys():
            leave_one_out_sum = sum(
                val for name, val in ds_metrics.items() if name != dataset_name
            )
            self.save_top_k(trainer, pl_module, dataset_name, leave_one_out_sum)


class LeaveKOutModelCheckpoint(DatasetAwareModelCheckpoint):
    """
    ModelCheckpoint that saves checkpoints based on leave-k-out dataset performance.

    For each dataset group, this callback computes the mean loss on the remaining datasets
    (leave-k-out) and saves the checkpoint that performs best according to this metric.
    """

    def __init__(self, selected_datasets: list[str], **kwargs):
        super().__init__(**kwargs)
        self.selected_datasets = selected_datasets
        self.dirpath = self.dirpath / "lko" / self.monitor / self.criterion.name
        self.create_checkpoint_dir()

    def on_train_start(self, trainer, pl_module):
        """Initialize the criterion object for each dataset at the start of training."""
        self.ds_criterion = {
            "selected_ds": copy.deepcopy(self.criterion),
            "left_out_ds": copy.deepcopy(self.criterion),
            "all_in": copy.deepcopy(self.criterion),
        }

    def _process_metric_across_datasets(self, trainer, pl_module, ds_metrics: dict):
        """Process dataset losses using the leave-k-out strategy."""
        selected_sum = sum(
            val for name, val in ds_metrics.items() if name in self.selected_datasets
        )
        left_out_sum = sum(
            val
            for name, val in ds_metrics.items()
            if not name in self.selected_datasets
        )

        self.save_top_k(trainer, pl_module, "all_in", selected_sum + left_out_sum)
        self.save_top_k(trainer, pl_module, "selected_ds", selected_sum)
        self.save_top_k(trainer, pl_module, "left_out_ds", left_out_sum)

        fpath = self.dirpath / "selected_ds.yaml"
        if not os.path.exists(fpath):
            return

        with open(fpath, "w", encoding="utf-8") as file:
            yaml.safe_dump(list(self.selected_datasets), file)
