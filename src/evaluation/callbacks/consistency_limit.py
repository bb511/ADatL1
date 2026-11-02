from pathlib import Path

from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks.components.cap.metric import consistency_limit
from src.evaluation.callbacks.components.paired_scores import PairedScores


class ConsistencyLimit(Callback):
    """Evaluate the consistency limit of two test sets at every checkpoint.

    The scores of the two sets are paired as by the CAP callback, and
    `consistency_limit` of the pairs is logged to MLflow under the callback's name and
    kept per checkpoint. The checkpoint of the largest value is the optimised one, as
    for CAP; the evaluator selects it as the main metric with
    `optimized_metric_config.main_metric.callback.name` set to the name.

    Args:
        output_name: Key of the per-event anomaly scores in the outputs of a batch.
        dataset_1: Name of the first test dataloader.
        dataset_2: Name of the second test dataloader.
        pairing_type: Name of the pairing, as `get_pairing_fn` takes it.
        cap_metric_config: The CAP configuration, of which the normalisation and the
            energy are read.
        log_raw_mlflow: Unused, accepted for config compatibility.
        name: Name under which the evaluator finds the callback and the values are
            logged.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
        cap_metric_config: dict,
        log_raw_mlflow: bool = True,
        name: str = "consistency",
    ):
        super().__init__()
        self.cap_metric_config = cap_metric_config
        self.name = name
        self.pairs = PairedScores(
            name, "Consistency", output_name, dataset_1, dataset_2, pairing_type
        )

    def on_test_epoch_start(self, trainer, pl_module):
        self.pairs.reset()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.pairs.collect(trainer, outputs, dataloader_idx)

    def on_test_epoch_end(self, trainer, pl_module):
        value = consistency_limit(*self.pairs.paired(), **self.cap_metric_config)
        self.pairs.record(trainer, pl_module, value)

    def plot_summary(self, trainer, root_folder: Path):
        self.pairs.plot_summary(trainer, root_folder)

    def get_optimized_metric(self):
        return self.pairs.best()

    def clear_crit_summary(self):
        self.pairs.summary.clear()
