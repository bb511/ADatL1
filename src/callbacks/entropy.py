"""The loss entropy of EntropyStop as a training callback (Huang et al., KDD 2024).

The logged keys, the EMA and the skipped sanity pass are those of
src.callbacks.criterion.CriterionCallback; the criterion is
``label_free.loss_entropy``, minimised.
"""

from torch.utils.data import DataLoader, Subset

from src.callbacks.criterion import CriterionCallback, row_subset
from src.callbacks.metrics import label_free
from src.callbacks.metrics.selection import EMA_WEIGHT
from src.callbacks.score_dump import score_in_eval_mode


class LossEntropyCallback(CriterionCallback):
    """Entropy of the per-sample training loss, from an eval-mode pass every epoch over
    the rows of ``trainer.datamodule.data_train``, unshuffled.

    The entropy is NaN, with a warning, when a loss is not positive, as the negative
    log-likelihood of RealNVP can be.

    :param output_name: Key of the per-sample loss in the module's outputs.
    :param max_rows: Number of training rows, a subset drawn once by ``row_subset``;
        None takes every row.
    :param row_seed: Seed of the subset.
    :param batch_size: Batch size of the pass over the training rows.
    """

    name = "entropy"

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "loss/full",
        ema_weight: float = EMA_WEIGHT,
        max_rows: int | None = None,
        row_seed: int = 0,
        batch_size: int = 1024,
    ):
        super().__init__(name, output_name, ema_weight)
        self.max_rows = max_rows
        self.row_seed = row_seed
        self.batch_size = batch_size

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        rows = trainer.datamodule.data_train
        subset = row_subset(len(rows), self.max_rows, self.row_seed)
        rows = rows if subset is None else Subset(rows, subset.tolist())
        self.loaders = {"train": DataLoader(rows, batch_size=self.batch_size)}

    def criterion(self, trainer, pl_module) -> float:
        keys = [self.output_name]
        losses = score_in_eval_mode(trainer, pl_module, self.loaders, keys).scores()
        return label_free.loss_entropy(losses["train"][self.output_name])
