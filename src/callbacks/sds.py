"""The SDS error of self-adaptive data shifting as a training callback (Wang et al.,
Pattern Recognit. 74, 2018).

The logged keys, the EMA and the skipped sanity pass are those of
src.callbacks.criterion.CriterionCallback; the criterion is ``label_free.sds_error``,
minimised, and ``sds_inputs`` builds its pseudo-sets.
"""

import math
import warnings

import torch

from src.callbacks.criterion import CriterionCallback, row_subset, score_inputs
from src.callbacks.metrics import label_free
from src.callbacks.metrics.rate import AnomalyRate
from src.callbacks.metrics.selection import EMA_WEIGHT
from src.data.utils import unpack_batch

SOURCES = ("train", "normal")


def sds_inputs(
    x: torch.Tensor, max_rows: int | None, row_seed: int
) -> dict[str, torch.Tensor]:
    """Pseudo-targets and pseudo-outliers of ``label_free.sds_pseudo_sets`` from the
    rows of x, (n, *s).

    The rows are flattened and deduplicated, since sds.m divides by the distance to
    every neighbour; of the distinct rows, in the sorted order of ``torch.unique``,
    ``row_subset`` with ``row_seed`` keeps at most ``max_rows``. Return ``sds_targets``
    and ``sds_outliers``, (m, *s) each, in the dtype of x.
    """
    rows = torch.unique(x.reshape(len(x), -1), dim=0)
    subset = row_subset(len(rows), max_rows, row_seed)
    rows = rows if subset is None else rows[subset]
    sets = label_free.sds_pseudo_sets(rows.double().numpy())
    return {
        name: torch.from_numpy(points).to(x.dtype).reshape(-1, *x.shape[1:])
        for name, points in zip(("sds_targets", "sds_outliers"), sets)
    }


def _operational_cut(pl_module, normal: torch.Tensor) -> tuple[float, float]:
    """Threshold and tie weight of AnomalyRate at the module's target rate, set on the
    validation normal scores as AnomalyEfficiencyCallback sets them."""
    hparams = pl_module.hparams
    rate = AnomalyRate(hparams.target_rate, getattr(hparams, "base_rate", None))
    rate.set_threshold(normal)
    return rate.threshold.item(), rate.tie_weight.item()


class SDSCallback(CriterionCallback):
    """SDS error of the detector cut at the epoch's operational threshold.

    The pseudo-sets of ``sds_inputs`` are built once, from the training inputs at fit
    start (``source="train"``) or from the validation normal inputs of the first epoch
    after the sanity pass (``source="normal"``), and are scored every epoch.

    :param source: "train" or "normal", the rows the pseudo-sets are built from.
    :param max_rows: Largest number of distinct rows the pseudo-sets are built from.
    :param row_seed: Seed of the subset of rows.
    :param batch_size: Batch size of the pass over the pseudo-sets.
    """

    name = "sds"

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/full",
        ema_weight: float = EMA_WEIGHT,
        source: str = "train",
        max_rows: int | None = 10000,
        row_seed: int = 0,
        batch_size: int = 1024,
    ):
        super().__init__(name, output_name, ema_weight)
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got '{source}'.")
        self.source = source
        self.max_rows = max_rows
        self.row_seed = row_seed
        self.batch_size = batch_size

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self.inputs, self.normal_inputs = None, []
        if self.source == "train":
            x_train = trainer.datamodule.data_train.tensors[0]
            self.inputs = sds_inputs(x_train, self.max_rows, self.row_seed)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        name = list(trainer.val_dataloaders)[dataloader_idx]
        if name == "normal" and self.inputs is None and not trainer.sanity_checking:
            self.normal_inputs.append(unpack_batch(batch).x.detach().cpu())

    def criterion(self, trainer, pl_module) -> float:
        if self.inputs is None:
            self._build_from_normal_inputs()
        if not all(len(points) for points in self.inputs.values()):
            warnings.warn("SDS found an empty pseudo-set; returning NaN.")
            return math.nan
        scores = score_inputs(
            trainer, pl_module, self.inputs, self.output_name, self.batch_size
        )
        threshold, tie_weight = _operational_cut(pl_module, self.normal())
        a_targets, a_outliers = scores["sds_targets"], scores["sds_outliers"]
        return label_free.sds_error(a_targets, a_outliers, threshold, tie_weight)

    def _build_from_normal_inputs(self) -> None:
        x = torch.cat(self.normal_inputs)
        self.inputs = sds_inputs(x, self.max_rows, self.row_seed)
        self.normal_inputs = []
