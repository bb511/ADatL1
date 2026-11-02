"""The paired validation scores that the CAP and consistency callbacks share."""

import math

import torch
from torch import Tensor

from src.callbacks.components.cap.binary import get_pairing_fn
from src.callbacks.components.selection import ema_update
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

# Every value is logged once per epoch, as ModelCheckpoint reads it.
LOG_KWARGS = dict(
    prog_bar=False,
    on_step=False,
    on_epoch=True,
    logger=True,
    sync_dist=True,
    add_dataloader_idx=False,
)


class PairedScores:
    """The scores of two validation sets over an epoch, paired, and the moving average
    of one value per epoch computed from the pairs.

    The moving average leaves out a non-finite value and is NaN until the first finite
    one. A quantity q is logged under `val/summary/<name>_<q>_<pair_name>`, where
    `pair_name` is `<dataset_1>_vs_<dataset_2>` with every '/' replaced by '_'.

    Args:
        name: Name of the value, which opens its logged keys.
        output_name: Key of the per-event anomaly scores in the outputs of a batch.
        dataset_1: Name of the first validation dataloader.
        dataset_2: Name of the second validation dataloader.
        pairing_type: Name of the pairing, as `get_pairing_fn` takes it.
        ema_weight: Weight of the previous value in the moving average.
    """

    def __init__(
        self,
        name: str,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
        ema_weight: float,
    ):
        self.name, self.output_name = name, output_name
        self.datasets = (dataset_1, dataset_2)
        self.pairing_fn = get_pairing_fn(pairing_type)
        self.ema_weight = ema_weight
        self.pair_name = "_vs_".join(d.replace("/", "_") for d in self.datasets)

    def key(self, quantity: str) -> str:
        return f"val/summary/{self.name}_{quantity}_{self.pair_name}"

    def start(self) -> None:
        """Restart the moving average, at the start of a fit."""
        self.ema = math.nan

    def reset(self, trainer) -> None:
        """Check that both sets are served and drop the scores of the last epoch."""
        served = list(trainer.val_dataloaders.keys())
        missing = [dataset for dataset in self.datasets if dataset not in served]
        if missing:
            raise ValueError(
                f"The {self.name} of {self.pair_name} needs the validation dataloaders "
                f"{missing}. Available validation dataloaders: {served}"
            )
        self.scores = {dataset: [] for dataset in self.datasets}

    def collect(self, trainer, outputs: dict, dataloader_idx: int) -> None:
        """Keep the scores of a batch of either set."""
        dataset = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dataset not in self.scores:
            return
        scores = outputs[self.output_name]
        if scores.ndim == 0:
            raise ValueError(f"outputs['{self.output_name}'] is scalar. Need a tensor.")
        self.scores[dataset].append(scores.detach().view(-1))

    def paired(self) -> tuple[Tensor, Tensor]:
        """Return the epoch's scores of the two sets, paired and cut to equal length."""
        for dataset, scores in self.scores.items():
            if not scores:
                raise RuntimeError(
                    f"No validation scores were collected for '{dataset}'."
                )
        scores_1, scores_2 = (torch.cat(self.scores[d]) for d in self.datasets)
        index_1, index_2 = self.pairing_fn(scores_1, scores_2)
        paired_1, paired_2 = scores_1[index_1], scores_2[index_2]
        n = min(len(paired_1), len(paired_2))
        if n == 0:
            raise RuntimeError(f"No pairs were formed for the {self.name}.")
        return paired_1[:n], paired_2[:n]

    def average(self, value: float) -> float:
        """Return the moving average updated with the epoch's value."""
        if not math.isfinite(value):
            log.warning(f"The {self.name} {value} is not finite: left out of its EMA.")
        self.ema = ema_update(self.ema, value, self.ema_weight, skip_nonfinite=True)
        return self.ema
