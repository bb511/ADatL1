"""The loss entropy of EntropyStop (Y. Huang et al., KDD 2024,
doi:10.1145/3637528.3671943), minimised, as a function of the per-sample losses, its
offline stopping rule, and the training callback.

Both functions follow EntropyEarlyStop.py of github.com/goldenNormal/EntropyStop-KDD2024
(commit b1e0774). The callback logs the entropy under ``val/summary/<name>`` and the
exponential moving average under ``val/summary/<name>_ema``, from the first epoch after
the sanity pass.
"""

import math
import warnings

import numpy as np
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, Subset

from src.callbacks.components.selection import EMA_WEIGHT, ema_update
from src.callbacks.components.utils import as_scores, row_subset
from src.callbacks.score_dump import score_in_eval_mode


def loss_entropy(losses) -> float:
    """Loss entropy H_L of EntropyStop, Eqs. (9)-(10), p. 4, in nats.

    The per-sample losses are normalised to a distribution u and H_L = -sum u log u,
    without the 1e-7 that the code's ``cal_entropy`` adds inside the log, which lowers
    its value by about 1e-7 per sample. The criterion assumes positive losses (p. 3):
    with a loss that is not positive it is undefined, and NaN is returned with a
    warning.
    """
    losses = as_scores(losses)
    if not (losses > 0).all():
        warnings.warn("loss_entropy needs positive losses; returning NaN.")
        return math.nan
    u = losses / losses.sum()
    return float(-(u * np.log(u)).sum())


def entropy_stop(curve, k: int = 100, r_down: float = 0.01) -> tuple[int, int]:
    """Offline EntropyStop: ``getStopOffline`` of EntropyEarlyStop.py, ported.

    A value becomes the accepted minimum when it lies below the current one by more
    than ``r_down`` times the total variation of the curve since that minimum was set
    (Sec. 4.3 and Algorithm 1, pp. 5-6). Return the index of the last accepted minimum
    and the index at which the patience reached ``k``, or the last index when it never
    did. The defaults are the code's; the paper sets k = 100 and R_down = 0.1 (App.
    B.2, p. 13).
    """
    curve = [float(value) for value in curve]
    minimum, best, patience, variation = None, 0, 0, 0.0
    for i, value in enumerate(curve):
        if minimum is None:
            minimum = value
            continue
        variation += abs(value - curve[i - 1])
        if value < minimum and (minimum - value) / variation > r_down:
            minimum, best, patience, variation = value, i, 0, 0.0
        else:
            patience += 1
        if patience == k:
            break
    return best, i


class LossEntropyCallback(Callback):
    """Compute the entropy of the per-sample training loss.

    The losses come from a pass every epoch over the rows of
    ``trainer.datamodule.data_train``, unshuffled, in eval mode under forked RNG
    streams, so that the training trajectory is that of a run without the callback. The
    entropy is NaN, with a warning, when a loss is not positive, as the negative
    log-likelihood of RealNVP can be.

    :param name: Name in the logged keys; None takes "entropy".
    :param output_name: Key of the per-sample loss in the module's outputs.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    :param max_rows: Number of training rows, a seeded subset drawn once without
        replacement; None takes every row.
    :param row_seed: Seed of the subset.
    :param batch_size: Batch size of the pass over the training rows.
    """

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "loss/full",
        ema_weight: float = EMA_WEIGHT,
        max_rows: int | None = None,
        row_seed: int = 0,
        batch_size: int = 1024,
    ):
        super().__init__()
        self.name = "entropy" if name is None else name
        self.output_name = output_name
        self.ema_weight = ema_weight
        self.max_rows = max_rows
        self.row_seed = row_seed
        self.batch_size = batch_size

        self.log_kwargs = dict(
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
            batch_size=1,
        )

    def on_fit_start(self, trainer, pl_module):
        """Instantiate useful quantities and the loader of the training rows."""
        self.entropy_ema = None
        rows = trainer.datamodule.data_train
        subset = row_subset(len(rows), self.max_rows, self.row_seed)
        rows = rows if subset is None else Subset(rows, subset.tolist())
        self.loaders = {"train": DataLoader(rows, batch_size=self.batch_size)}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the loss entropy and its EMA."""
        if trainer.sanity_checking:
            return

        keys = [self.output_name]
        losses = score_in_eval_mode(trainer, pl_module, self.loaders, keys).scores()
        entropy = loss_entropy(losses["train"][self.output_name])
        self.entropy_ema = ema_update(self.entropy_ema, entropy, self.ema_weight)

        key = f"val/summary/{self.name}"
        metrics = {key: entropy, f"{key}_ema": self.entropy_ema}
        pl_module.log_dict(metrics, **self.log_kwargs)
