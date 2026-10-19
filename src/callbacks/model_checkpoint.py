# Checkpoint callback that keeps the top-k epochs of each data set separately.
from src.callbacks.checkpointing.dataset_aware import DatasetAwareModelCheckpoint


class SingleDatasetModelCheckpoint(DatasetAwareModelCheckpoint):
    """ModelCheckpoint that saves checkpoints based on individual dataset
    performance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dirpath = self.dirpath / "single" / self.monitor / self.criterion.name

    def _process_metric_across_datasets(self, trainer, pl_module, ds_metrics: dict):
        """Process the metric for each data set separately."""
        for dataset_name, metric_value in ds_metrics.items():
            self.save_top_k(trainer, pl_module, dataset_name, metric_value)
