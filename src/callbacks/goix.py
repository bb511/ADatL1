"""Goix's criteria: the excess-mass area (maximised) and the mass-volume area
(minimised), N. Goix, arXiv:1607.01152, as functions of scores and as training
callbacks.

The functions follow em.py and em_bench.py of github.com/ngoix/EMMV_benchmarks (commit
490d423) line for line, so that the two can be compared; they take anomaly scores,
higher meaning more anomalous, as anything ``np.asarray`` accepts, and compute in
float64. Both read a set drawn uniformly over the per-feature bounds of the training
inputs (the dump's ``uniform`` set) and are NaN, with a warning, when the volume of
that box is not finite and positive, as for the 117 physics features. Each callback
logs its criterion of the validation normal scores under ``val/summary/<name>`` and
the exponential moving average under ``val/summary/<name>_ema``, from the first epoch
after the sanity pass.
"""

import math
import warnings

import numpy as np
import torch
from pytorch_lightning import Callback
from sklearn.metrics import auc
from torch.utils.data import DataLoader

from src.callbacks.components.selection import EMA_WEIGHT, ema_update
from src.callbacks.components.utils import as_scores, placeholder_loader
from src.callbacks.score_dump import aux_inputs, input_moments, score_in_eval_mode

# The EM level t_max and the alpha grid of em_bench.py.
EM_T_MAX = 0.9
MV_ALPHAS = np.arange(0.9, 0.999, 0.0001)


def box_volume(lower, upper) -> float:
    """Volume of the axis-aligned box [lower, upper], in float64; inf on overflow."""
    with np.errstate(over="ignore"):
        sides = np.asarray(upper, np.float64) - np.asarray(lower, np.float64)
        return float(np.prod(sides))


def _is_volume(volume: float) -> bool:
    if math.isfinite(volume) and volume > 0:
        return True
    warnings.warn(f"The box volume {volume} is not finite and positive; returning NaN.")
    return False


def _em(t: np.ndarray, t_max: float, volume: float, s_unif, s_x) -> tuple[float, bool]:
    """``em`` of em.py on normality scores: (area, t_max reached).

    The counts #{s > u} come from sorted copies, which gives the integers of
    ``(s > u).sum()``.
    """
    x_sorted, unif_sorted = np.sort(s_x), np.sort(s_unif)
    em_t = np.zeros(t.shape[0])
    em_t[0] = 1.0
    for u in np.unique(s_x):
        above_x = s_x.size - np.searchsorted(x_sorted, u, side="right")
        above_unif = s_unif.size - np.searchsorted(unif_sorted, u, side="right")
        mass = 1.0 / s_x.size * above_x
        em_t = np.maximum(em_t, mass - t * above_unif / s_unif.size * volume)
    amax = np.argmax(em_t <= t_max) + 1
    # The code's amax = -1 when t_max is never reached: the area runs over t[:-1].
    reached = bool(amax != 1)
    amax = amax if reached else -1
    return float(auc(t[:amax], em_t[:amax])), reached


def excess_mass(
    a_normal, a_uniform, volume: float, t_max: float = EM_T_MAX
) -> tuple[float, bool]:
    """Excess-mass area, Goix (2016) Eqs. (3) and (5), as em.py computes it.

    The normality score is s = -a. ``a_uniform`` scores points drawn uniformly over an
    axis-aligned box of volume ``volume``, which estimates Leb(s > u) as volume
    #{s(U) > u} / n_uniform. The conventions are the code's: the strict ">" in both
    counts (the paper has ">="), EM(0) = 1, the grid t = arange(0, 100 / volume,
    0.01 / volume) of em_bench.py, and the area truncated at the first t with
    EM(t) <= t_max. Return the area and whether t_max was reached; (NaN, False) with a
    warning when the volume is not finite and positive.
    """
    if not _is_volume(volume):
        return math.nan, False
    t = np.arange(0, 100 / volume, 0.01 / volume)
    return _em(t, t_max, volume, -as_scores(a_uniform), -as_scores(a_normal))


def _mv(axis_alpha: np.ndarray, volume: float, s_unif, s_x) -> float:
    """``mv`` of em.py on normality scores."""
    order, unif_sorted = s_x.argsort(), np.sort(s_unif)
    mass, cpt, u = 0, 0, s_x[order[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_x[order[-cpt]]
            mass = 1.0 / s_x.size * cpt
        at_or_above = s_unif.size - np.searchsorted(unif_sorted, u, side="left")
        mv[i] = float(at_or_above) / s_unif.size * volume
    return float(auc(axis_alpha, mv))


def mass_volume(a_normal, a_uniform, volume: float, alphas=MV_ALPHAS) -> float:
    """Mass-volume area, Goix (2016) Eqs. (4) and (6), as em.py computes it.

    MV(alpha) is the estimated volume of the smallest upper level set of s = -a that
    holds a fraction alpha of ``a_normal``, with Leb(s >= u) estimated as volume
    #{s(U) >= u} / n_uniform (the code's ">=", as in the paper); the curve is
    integrated over ``alphas``, by default em_bench.py's arange(0.9, 0.999, 0.0001).
    NaN is returned with a warning when the volume is not finite and positive.
    """
    if not _is_volume(volume):
        return math.nan
    alphas = np.asarray(alphas, dtype=np.float64)
    return _mv(alphas, volume, -as_scores(a_uniform), -as_scores(a_normal))


def _uniform_box(
    trainer, aux_seed: int, aux_size: int, batch_size: int
) -> tuple[dict[str, DataLoader], float]:
    """Loaders of the auxiliary set ``uniform`` of the score dump, drawn by
    ``aux_inputs`` over the per-feature bounds of the training inputs, and the volume of
    that box."""
    moments = input_moments(trainer.datamodule.data_train.tensors[0])
    meta = {"aux_seed": aux_seed, "aux_size": aux_size, **moments}
    loaders = {"uniform": placeholder_loader(aux_inputs(meta)["uniform"], batch_size)}
    return loaders, box_volume(moments["aux_lower"], moments["aux_upper"])


class ExcessMassCallback(Callback):
    """Compute the excess-mass area of the validation normal scores.

    The uniform set is drawn once at fit start, so that the dump and the callback score
    the same points, and scored every epoch in eval mode under forked RNG streams, so
    that the training trajectory is that of a run without the callback.

    :param name: Name in the logged keys; None takes "em".
    :param output_name: Key in outputs dict containing per-event anomaly scores.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    :param aux_seed: Seed of the auxiliary sets, as ScoreDumpCallback's.
    :param aux_size: Number of points in each auxiliary set.
    :param batch_size: Batch size of the pass over the auxiliary set.
    """

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/full",
        ema_weight: float = EMA_WEIGHT,
        aux_seed: int = 0,
        aux_size: int = 10000,
        batch_size: int = 1024,
    ):
        super().__init__()
        self.name = "em" if name is None else name
        self.output_name = output_name
        self.ema_weight = ema_weight
        self.aux_seed = aux_seed
        self.aux_size = aux_size
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
        """Instantiate useful quantities and draw the uniform set."""
        self.em_ema = None
        self.loaders, self.volume = _uniform_box(
            trainer, self.aux_seed, self.aux_size, self.batch_size
        )

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
        """Aggregate the anomaly scores of the normal validation data set."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        self.normal_scores.append(score.reshape(-1))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the excess-mass area and its EMA."""
        if trainer.sanity_checking:
            return

        normal = torch.cat(self.normal_scores).cpu()
        aux = score_in_eval_mode(trainer, pl_module, self.loaders, [self.output_name])
        uniform = aux.scores()["uniform"][self.output_name]
        em_value = excess_mass(normal, uniform, self.volume)[0]
        self.em_ema = ema_update(self.em_ema, em_value, self.ema_weight)

        key = f"val/summary/{self.name}"
        metrics = {key: em_value, f"{key}_ema": self.em_ema}
        pl_module.log_dict(metrics, **self.log_kwargs)


class MassVolumeCallback(Callback):
    """Compute the mass-volume area of the validation normal scores.

    The uniform set is drawn once at fit start, so that the dump and the callback score
    the same points, and scored every epoch in eval mode under forked RNG streams, so
    that the training trajectory is that of a run without the callback.

    :param name: Name in the logged keys; None takes "mv".
    :param output_name: Key in outputs dict containing per-event anomaly scores.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    :param aux_seed: Seed of the auxiliary sets, as ScoreDumpCallback's.
    :param aux_size: Number of points in each auxiliary set.
    :param batch_size: Batch size of the pass over the auxiliary set.
    """

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/full",
        ema_weight: float = EMA_WEIGHT,
        aux_seed: int = 0,
        aux_size: int = 10000,
        batch_size: int = 1024,
    ):
        super().__init__()
        self.name = "mv" if name is None else name
        self.output_name = output_name
        self.ema_weight = ema_weight
        self.aux_seed = aux_seed
        self.aux_size = aux_size
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
        """Instantiate useful quantities and draw the uniform set."""
        self.mv_ema = None
        self.loaders, self.volume = _uniform_box(
            trainer, self.aux_seed, self.aux_size, self.batch_size
        )

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
        """Aggregate the anomaly scores of the normal validation data set."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        self.normal_scores.append(score.reshape(-1))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the mass-volume area and its EMA."""
        if trainer.sanity_checking:
            return

        normal = torch.cat(self.normal_scores).cpu()
        aux = score_in_eval_mode(trainer, pl_module, self.loaders, [self.output_name])
        uniform = aux.scores()["uniform"][self.output_name]
        mv_value = mass_volume(normal, uniform, self.volume)
        self.mv_ema = ema_update(self.mv_ema, mv_value, self.ema_weight)

        key = f"val/summary/{self.name}"
        metrics = {key: mv_value, f"{key}_ema": self.mv_ema}
        pl_module.log_dict(metrics, **self.log_kwargs)
