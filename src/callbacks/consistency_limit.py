from pytorch_lightning import Callback

from src.callbacks.components.cap.metric import consistency_limit
from src.callbacks.components.paired_scores import LOG_KWARGS, PairedScores
from src.callbacks.components.selection import EMA_WEIGHT


class ConsistencyLimitCallback(Callback):
    """Log the moving average of the consistency limit of two validation sets.

    At every validation epoch after the sanity pass, the scores of the two sets are
    paired as by CAPCallback, and `consistency_limit` of the pairs enters the moving
    average logged as `val/summary/consistency_ema_<dataset_1>_vs_<dataset_2>`, which
    is maximised; a non-finite value leaves the average unchanged.

    Args:
        output_name: Key of the per-event anomaly scores in the outputs of a batch.
        dataset_1: Name of the first validation dataloader.
        dataset_2: Name of the second validation dataloader.
        pairing_type: Name of the pairing, as `get_pairing_fn` takes it.
        cap_metric_config: The CAP configuration, of which the normalisation and the
            energy are read.
        ema_weight: Weight of the previous value in the moving average.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
        cap_metric_config: dict,
        ema_weight: float = EMA_WEIGHT,
    ):
        super().__init__()
        self.cap_metric_config = cap_metric_config
        self.pairs = PairedScores(
            "consistency", output_name, dataset_1, dataset_2, pairing_type, ema_weight
        )

    def on_fit_start(self, trainer, pl_module):
        self.pairs.start()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.pairs.reset(trainer)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.pairs.collect(trainer, outputs, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = consistency_limit(*self.pairs.paired(), **self.cap_metric_config)
        metrics = {self.pairs.key("ema"): self.pairs.average(value)}
        pl_module.log_dict(metrics, **LOG_KWARGS)
