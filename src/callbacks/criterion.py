"""The base of the label-free criterion callbacks.

A criterion callback computes one value per validation epoch from the scores of
validation ``normal`` and of any extra set it scores itself, and logs it under
``val/summary/<name>``, with its exponential moving average under
``val/summary/<name>_ema``. The sanity pass is skipped, so that the average starts at
epoch 0, and a non-finite value propagates into it, as in the drift and Wasserstein
callbacks. The extra sets are scored in eval mode under forked RNG streams, so that the
training trajectory is that of a run without the callback. The criteria themselves are
the functions of src.callbacks.metrics.label_free.
"""

import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.metrics.selection import EMA_WEIGHT, ema_update
from src.callbacks.score_dump import aux_inputs, input_moments, score_in_eval_mode

LOG_KWARGS = dict(
    prog_bar=False,
    on_step=False,
    on_epoch=True,
    logger=True,
    sync_dist=True,
    add_dataloader_idx=False,
    batch_size=1,
)


def row_subset(n: int, max_rows: int | None, seed: int) -> torch.Tensor | None:
    """Sorted indices of max_rows of n rows drawn without replacement, or None for every
    row when max_rows is None or at least n."""
    if max_rows is None or max_rows >= n:
        return None
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:max_rows].sort().values


def _loader(x: torch.Tensor, batch_size: int) -> DataLoader:
    # The labels are placeholders: model_step reads only the inputs.
    labels = torch.zeros(len(x), dtype=torch.long)
    return DataLoader(TensorDataset(x, labels), batch_size=batch_size)


def score_inputs(
    trainer, pl_module, inputs: dict[str, torch.Tensor], key: str, batch_size: int
) -> dict[str, torch.Tensor]:
    """Score input sets, (n, *s) each, in eval mode under forked RNG streams.

    Return the ``key`` output of every set, (n,), on CPU; an empty set is left out.
    """
    loaders = {name: _loader(x, batch_size) for name, x in inputs.items()}
    scores = score_in_eval_mode(trainer, pl_module, loaders, [key]).scores()
    return {name: set_scores[key] for name, set_scores in scores.items()}


class CriterionCallback(Callback):
    """Base of the label-free criteria: ``criterion`` returns the epoch's value.

    A subclass sets the class attribute ``name`` and implements ``criterion``, which
    reads the epoch's validation normal scores from ``normal()``.

    :param name: Name in the logged keys; None keeps the class's.
    :param output_name: Key of the per-event score in the module's outputs.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

    name: str

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/full",
        ema_weight: float = EMA_WEIGHT,
    ):
        super().__init__()
        if name is not None:
            self.name = name
        self.output_name = output_name
        self.ema_weight = ema_weight

    def on_fit_start(self, trainer, pl_module):
        self.ema = None

    def on_validation_epoch_start(self, trainer, pl_module):
        if "normal" not in trainer.val_dataloaders:
            raise ValueError(f"{type(self).__name__} needs a 'normal' val loader.")
        self.normal_scores = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if list(trainer.val_dataloaders)[dataloader_idx] == "normal":
            score = torch.as_tensor(outputs[self.output_name]).detach()
            self.normal_scores.append(score.reshape(-1))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = float(self.criterion(trainer, pl_module))
        self.ema = ema_update(self.ema, value, self.ema_weight)
        pl_module.log_dict(self.logged(value), **LOG_KWARGS)

    def normal(self) -> torch.Tensor:
        """The epoch's ``output_name`` of validation normal, (n,), on CPU."""
        return torch.cat(self.normal_scores).cpu()

    def criterion(self, trainer, pl_module) -> float:
        raise NotImplementedError

    def logged(self, value: float) -> dict[str, float]:
        key = f"val/summary/{self.name}"
        return {key: value, f"{key}_ema": self.ema}


class AuxSetCallback(CriterionCallback):
    """Base of a criterion that also reads the scores of the auxiliary set ``aux_set``.

    The auxiliary sets are those of the score dump, ``aux_inputs`` of the moments of
    the training inputs, so that the dump and the callback score the same points. They
    are drawn once at fit start and kept in ``aux`` with the moments; the set
    ``aux_set`` is scored every epoch.

    :param aux_seed: Seed of the auxiliary sets, as ScoreDumpCallback's.
    :param aux_size: Number of points in each auxiliary set.
    :param batch_size: Batch size of the pass over the auxiliary set.
    """

    aux_set: str

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/full",
        ema_weight: float = EMA_WEIGHT,
        aux_seed: int = 0,
        aux_size: int = 10000,
        batch_size: int = 1024,
    ):
        super().__init__(name, output_name, ema_weight)
        self.aux_seed = aux_seed
        self.aux_size = aux_size
        self.batch_size = batch_size

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        moments = input_moments(trainer.datamodule.data_train.tensors[0])
        meta = {"aux_seed": self.aux_seed, "aux_size": self.aux_size, **moments}
        self.aux = {**aux_inputs(meta), **moments}

    def aux_scores(self, trainer, pl_module) -> torch.Tensor:
        """The epoch's ``output_name`` of the set ``aux_set``, (aux_size,), on CPU."""
        inputs = {self.aux_set: self.aux[self.aux_set]}
        scores = score_inputs(
            trainer, pl_module, inputs, self.output_name, self.batch_size
        )
        return scores[self.aux_set]
