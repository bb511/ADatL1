"""The SDS error of self-adaptive data shifting (S. Wang et al., Pattern Recognit. 74,
198-211, 2018, doi:10.1016/j.patcog.2017.09.012), minimised: the pseudo-sets, the error
of a thresholded score, and the training callback.

The paper's text is not open; the functions follow sds.m and main.m of
github.com/homarques/SDS-hyperparameter-selection (commit 8bdbe4a), the implementation
behind Marques et al., DMKD 37 (2023), doi:10.1007/s10618-023-00931-x. The callback
logs the error under ``val/summary/<name>`` and the exponential moving average under
``val/summary/<name>_ema``, from the first epoch after the sanity pass.
"""

import math
import warnings

import numpy as np
import torch
from pytorch_lightning import Callback
from scipy.spatial import cKDTree

from src.callbacks.components.rate import AnomalyRate
from src.callbacks.components.selection import EMA_WEIGHT, ema_update
from src.callbacks.components.utils import as_scores, placeholder_loader, row_subset
from src.callbacks.score_dump import score_in_eval_mode
from src.data.utils import unpack_batch

# sds.m's default edge threshold; the Python port
# bzantium/OCSVM-hyperparameter-selection defaults to 0.01 with a strict inequality.
SDS_THRESHOLD = 0.1
SOURCES = ("train", "normal")


def _neighbours(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours of every row, itself excluded, (n, k)."""
    distances, indices = cKDTree(x).query(x, k=k + 1)
    if (distances[:, 1:] == 0).any():
        raise ValueError("SDS needs distinct rows: sds.m divides by zero distances.")
    return indices[:, 1:]


def _edge_patterns(v: np.ndarray, threshold: float) -> np.ndarray:
    """Edge flags of sds.m, from the unit offsets v (n, k, d) of every row."""
    theta = np.einsum("ikd,id->ik", v, v.sum(axis=1))
    return 1 / v.shape[1] * (theta >= 0).sum(axis=1) >= 1 - threshold


def _pseudo_targets(x: np.ndarray, unit: np.ndarray, offsets: np.ndarray):
    # lambda_ij = (-n_i) . (x_j - x_i); a point with no positive lambda yields no
    # pseudo-target, as in sds.m.
    lam = np.einsum("id,ikd->ik", unit, offsets)
    shift = np.where(lam > 0, lam, np.inf).min(axis=1)
    keep = np.isfinite(shift)
    return x[keep] - shift[keep, None] * unit[keep]


def sds_pseudo_sets(
    x, k: int | None = None, threshold: float = SDS_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-targets and pseudo-outliers of self-adaptive data shifting, as sds.m.

    ``x`` (n, d) holds distinct rows, the data a detector is trained on. Per row, with
    k = ceil(5 log10 n) neighbours: v_ij = (x_i - x_j) / norm(x_i - x_j) and
    n_i = sum_j v_ij, and the row is an edge pattern when the fraction of j with
    v_ij . n_i >= 0 is at least 1 - threshold. Pseudo-outliers shift every edge pattern
    along n_i / norm(n_i) by the mean, over the edge patterns, of their mean neighbour
    distance. Pseudo-targets shift every row along -n_i / norm(n_i) by the smallest
    positive projection of its neighbour offsets on that direction. The generation is
    deterministic. Return (targets, outliers), (n_t, d) and (n_o, d).
    """
    x = np.asarray(x, dtype=np.float64)
    k = math.ceil(5 * math.log10(len(x))) if k is None else k
    offsets = x[:, None, :] - x[_neighbours(x, k)]
    distances = np.linalg.norm(offsets, axis=2)
    v = offsets / distances[:, :, None]
    normal = v.sum(axis=1)
    unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    edge = _edge_patterns(v, threshold)
    outliers = x[edge] + unit[edge] * distances[edge].mean(axis=1).mean()
    return _pseudo_targets(x, unit, offsets), outliers


def _rejected(scores, threshold: float, tie_weight: float) -> float:
    s = as_scores(scores)
    return float(((s > threshold).sum() + tie_weight * (s == threshold).sum()) / s.size)


def sds_error(
    a_targets, a_outliers, threshold: float, tie_weight: float = 1.0
) -> float:
    """SDS error of a detector cut at ``threshold``: pseudo-targets rejected plus
    pseudo-outliers accepted, each as a fraction of its set.

    A score above the threshold is rejected, and one equal to it with weight
    ``tie_weight``, as ``AnomalyRate`` counts exceedances. The two fractions are
    summed as in main.m (``err_t_sds(1) + err_o_sds(2)``); the Python port averages
    them, which orders models identically.
    """
    accepted = 1.0 - _rejected(a_outliers, threshold, tie_weight)
    return _rejected(a_targets, threshold, tie_weight) + accepted


def sds_inputs(
    x: torch.Tensor, max_rows: int | None, row_seed: int
) -> dict[str, torch.Tensor]:
    """Pseudo-targets and pseudo-outliers of ``sds_pseudo_sets`` from the rows of x, (n,
    *s).

    The rows are flattened and deduplicated, since sds.m divides by the distance to
    every neighbour; of the distinct rows, in the sorted order of ``torch.unique``, at
    most ``max_rows`` are kept, drawn without replacement with ``row_seed``. Return
    ``sds_targets`` and ``sds_outliers``, (m, *s) each, in the dtype of x.
    """
    rows = torch.unique(x.reshape(len(x), -1), dim=0)
    subset = row_subset(len(rows), max_rows, row_seed)
    rows = rows if subset is None else rows[subset]
    sets = sds_pseudo_sets(rows.double().numpy())
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


class SDSCallback(Callback):
    """Compute the SDS error of the detector cut at the epoch's operational threshold.

    The pseudo-sets of ``sds_inputs`` are built once, from the training inputs at fit
    start (``source="train"``) or from the validation normal inputs of the first epoch
    after the sanity pass (``source="normal"``). They are scored every epoch in eval
    mode under forked RNG streams, so that the training trajectory is that of a run
    without the callback, and cut at the threshold and tie weight of AnomalyRate at the
    module's target rate, set on the epoch's validation normal scores.

    :param name: Name in the logged keys; None takes "sds".
    :param output_name: Key in outputs dict containing per-event anomaly scores.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    :param source: "train" or "normal", the rows the pseudo-sets are built from.
    :param max_rows: Largest number of distinct rows the pseudo-sets are built from.
    :param row_seed: Seed of the subset of rows.
    :param batch_size: Batch size of the pass over the pseudo-sets.
    """

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
        super().__init__()
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got '{source}'.")
        self.name = "sds" if name is None else name
        self.output_name = output_name
        self.ema_weight = ema_weight
        self.source = source
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
        """Instantiate useful quantities and, for source "train", build the pseudo-sets
        from the training inputs."""
        self.sds_ema = None
        self.loaders, self.normal_inputs = None, []
        if self.source == "train":
            self._build_pseudo_sets(trainer.datamodule.data_train.tensors[0])

    def on_validation_epoch_start(self, trainer, pl_module):
        """Make sure the normal data is in the used data sets."""
        dset_names = list(trainer.val_dataloaders.keys())
        if "normal" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'normal'. "
                f"Available validation dataloaders: {dset_names}"
            )

        self.normal_scores = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Aggregate the anomaly scores of the normal validation data set, and its
        inputs until the pseudo-sets are built."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        self.normal_scores.append(score.reshape(-1))
        if self.loaders is None and not trainer.sanity_checking:
            self.normal_inputs.append(unpack_batch(batch).x.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the SDS error and its EMA."""
        if trainer.sanity_checking:
            return

        if self.loaders is None:
            self._build_pseudo_sets(torch.cat(self.normal_inputs))
            self.normal_inputs = []
        sds_value = self._compute_sds(trainer, pl_module)
        self.sds_ema = ema_update(self.sds_ema, sds_value, self.ema_weight)

        key = f"val/summary/{self.name}"
        metrics = {key: sds_value, f"{key}_ema": self.sds_ema}
        pl_module.log_dict(metrics, **self.log_kwargs)

    def _build_pseudo_sets(self, x: torch.Tensor):
        """Build the loaders of the pseudo-sets of ``sds_inputs`` from the rows of x."""
        inputs = sds_inputs(x, self.max_rows, self.row_seed)
        self.loaders = {
            n: placeholder_loader(p, self.batch_size) for n, p in inputs.items()
        }

    def _compute_sds(self, trainer, pl_module) -> float:
        """Compute the SDS error of the pseudo-sets at the operational cut."""
        if not all(len(loader.dataset) for loader in self.loaders.values()):
            warnings.warn("SDS found an empty pseudo-set; returning NaN.")
            return math.nan
        keys = [self.output_name]
        scores = score_in_eval_mode(trainer, pl_module, self.loaders, keys).scores()
        normal = torch.cat(self.normal_scores).cpu()
        threshold, tie_weight = _operational_cut(pl_module, normal)
        a_targets = scores["sds_targets"][self.output_name]
        a_outliers = scores["sds_outliers"][self.output_name]
        return sds_error(a_targets, a_outliers, threshold, tie_weight)
