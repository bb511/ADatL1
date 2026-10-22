"""AutoUAD's criteria: the normalised pseudo discrepancy, the relative top-median and
the expected anomaly gap (W. Dai and J. Fan, ICLR 2025, OpenReview ErQPdaD5wJ), all
maximised, as functions of scores and as training callbacks.

The functions take anomaly scores, higher meaning more anomalous, as anything
``np.asarray`` accepts, and compute in float64; their docstrings fix the conventions
the paper leaves open. Each callback logs its criterion of the validation normal scores
under ``val/summary/<name>`` and the exponential moving average under
``val/summary/<name>_ema``, from the first epoch after the sanity pass.
"""

import numpy as np
import torch
from pytorch_lightning import Callback

from src.callbacks.components.selection import EMA_WEIGHT, ema_update
from src.callbacks.components.utils import as_scores, drawn_masks, placeholder_loader
from src.callbacks.score_dump import aux_inputs, input_moments, score_in_eval_mode
from src.data.utils import unpack_batch

# AutoUAD, App. G, p. 23: "For all eps, we set it as 1 x 10^-9".
EPS = 1e-9
# tau = 5 for RTM (Sec. 3.1, p. 5); rho = 0.2 for EAG (Assumption 2, App. A).
RTM_TOP_PERCENT = 5
EAG_TOP_PERCENT = 20


def _top_count(percent: int, n: int) -> int:
    # Integer ceil(percent n / 100), free of the rounding of percent / 100.
    return -(-percent * n // 100)


def npd(normal, noise, eps: float = EPS) -> float:
    """Normalised pseudo discrepancy, AutoUAD Def. 6 and Eq. (7), p. 6.

    ``normal`` scores held-out normal data (X_val) and ``noise`` draws from N(mu_trn,
    diag sigma_trn^2), the moments of the training inputs. The variances are population
    variances (ddof 0), as in the only public implementation (github.com/
    OliverHennhoefer/label-free); the paper does not state the estimator. The squared
    gap ignores the sign of the mean difference.
    """
    s_val, s_gen = as_scores(normal), as_scores(noise)
    gap = s_gen.mean() - s_val.mean()
    return float(gap**2 / (2 * (s_gen.var() + s_val.var()) + eps))


def rtm(normal, eps: float = EPS) -> float:
    """Relative top-median, AutoUAD Def. 3 and Eq. (4), p. 5, with tau = 5.

    Eq. (4) averages the scores at or above s_(tau N / 100) of the ascending order,
    which read literally is the top 95 %; the text ("the top tau %", p. 5) and the name
    RTM_5% mean the tau % largest scores, which are taken here. tau N / 100 is rounded
    up, and scores tied with the threshold enter, as the ">=" of Eq. (4) has it. The
    ratio is not translation-invariant: with a median at or below zero (a negative
    log-likelihood can be negative) its sign carries no meaning.
    """
    s = as_scores(normal)
    threshold = np.sort(s)[-_top_count(RTM_TOP_PERCENT, s.size)]
    median = np.median(s)
    return float((s[s >= threshold].mean() - median) / (median + eps))


def _prefix_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population variance of x[:j] for every j = 1, ..., n, (n,) each.

    The sum of squared deviations grows by Welford's term (j - 1) / j (x_j -
    mean_{j-1})^2, which is non-negative, so its cumulative sum does not cancel as the
    difference of the sums of x^2 and of x does when the scores are large against their
    spread.
    """
    j = np.arange(1, x.size + 1)
    mean = np.cumsum(x) / j
    terms = j[:-1] / j[1:] * (x[1:] - mean[:-1]) ** 2
    return mean, np.concatenate([[0.0], np.cumsum(terms)]) / j


def eag(normal, eps: float = EPS) -> float:
    """Expected anomaly gap, AutoUAD Defs. 4-5 and Eqs. (5)-(6), pp. 5-6.

    The discrete form of Sec. 3.2 averages the anomaly gap over the thresholds at or
    above the 80th percentile. As in Algorithms 1 and 2 (App. A, pp. 14-15), the scores
    are sorted in descending order and the k-th threshold splits them by position into
    the k largest and the rest, for k = 1, ..., ceil(0.2 N) (at most N - 1). Each gap
    is Eq. (5), w0 w1 (mu0 - mu1)^2 / (w0 var0 + w1 var1 + eps), with the population
    variances of the empirical distribution that Eq. (5) implies. Algorithm 1 adds eps
    as eps / N in the units of Eq. (5), and Algorithm 2 uses sample variances; both
    differ from Eq. (5) only for small k or near-constant groups. The moments of every
    split come from running moments of the descending scores (the k largest) and of
    the ascending ones (the N - k smallest), in O(N log N).
    """
    ascending = np.sort(as_scores(normal))
    n = ascending.size
    k = np.arange(1, min(_top_count(EAG_TOP_PERCENT, n), n - 1) + 1)
    mean_high, var_high = (m[k - 1] for m in _prefix_moments(ascending[::-1]))
    mean_low, var_low = (m[n - k - 1] for m in _prefix_moments(ascending))
    w1, w0 = k / n, (n - k) / n
    gaps = w0 * w1 * (mean_low - mean_high) ** 2
    return float(np.mean(gaps / (w0 * var_low + w1 * var_high + eps)))


class NPDCallback(Callback):
    """Compute the normalised pseudo discrepancy of the validation normal scores.

    The pseudo-anomalies are the auxiliary set ``autouad`` of the score dump, N(mu, diag
    sigma^2) with the per-feature moments of the training inputs, padding included. It
    is drawn once at fit start by ``aux_inputs``, so that the dump and the callback
    score the same points, and scored every epoch in eval mode under forked RNG streams,
    so that the training trajectory is that of a run without the callback. When the
    training set carries a data mask (the L1AD data), every point takes the mask of a
    training row drawn with ``aux_seed`` and is zero where it is padded.

    :param name: Name in the logged keys; None takes "npd".
    :param output_name: Key in outputs dict containing per-event anomaly scores.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    :param aux_seed: Seed of the auxiliary sets, as ScoreDumpCallback's, and of the
        training rows whose masks the points take.
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
        self.name = "npd" if name is None else name
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
        """Instantiate useful quantities and draw the autouad set."""
        self.npd_ema = None
        train = unpack_batch(trainer.datamodule.data_train.tensors)
        moments = input_moments(train.x)
        meta = {"aux_seed": self.aux_seed, "aux_size": self.aux_size, **moments}
        autouad = aux_inputs(meta)["autouad"]
        mask = drawn_masks(train.mask, len(autouad), self.aux_seed)
        loader = placeholder_loader(autouad, self.batch_size, mask)
        self.loaders = {"autouad": loader}

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
        """Compute and log the NPD against the autouad set, and its EMA."""
        if trainer.sanity_checking:
            return

        normal = torch.cat(self.normal_scores).cpu()
        aux = score_in_eval_mode(trainer, pl_module, self.loaders, [self.output_name])
        npd_value = npd(normal, aux.scores()["autouad"][self.output_name])
        self.npd_ema = ema_update(self.npd_ema, npd_value, self.ema_weight)

        key = f"val/summary/{self.name}"
        metrics = {key: npd_value, f"{key}_ema": self.npd_ema}
        pl_module.log_dict(metrics, **self.log_kwargs)


class RTMCallback(Callback):
    """Compute the relative top-median of the validation normal scores.

    :param name: Name in the logged keys; None takes "rtm".
    :param output_name: Key in outputs dict containing per-event anomaly scores.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/full",
        ema_weight: float = EMA_WEIGHT,
    ):
        super().__init__()
        self.name = "rtm" if name is None else name
        self.output_name = output_name
        self.ema_weight = ema_weight

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
        """Instantiate useful quantities."""
        self.rtm_ema = None

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
        """Compute and log the RTM and its EMA."""
        if trainer.sanity_checking:
            return

        rtm_value = rtm(torch.cat(self.normal_scores).cpu())
        self.rtm_ema = ema_update(self.rtm_ema, rtm_value, self.ema_weight)

        key = f"val/summary/{self.name}"
        metrics = {key: rtm_value, f"{key}_ema": self.rtm_ema}
        pl_module.log_dict(metrics, **self.log_kwargs)


class EAGCallback(Callback):
    """Compute the expected anomaly gap of the validation normal scores.

    :param name: Name in the logged keys; None takes "eag".
    :param output_name: Key in outputs dict containing per-event anomaly scores.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/full",
        ema_weight: float = EMA_WEIGHT,
    ):
        super().__init__()
        self.name = "eag" if name is None else name
        self.output_name = output_name
        self.ema_weight = ema_weight

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
        """Instantiate useful quantities."""
        self.eag_ema = None

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
        """Compute and log the EAG and its EMA."""
        if trainer.sanity_checking:
            return

        eag_value = eag(torch.cat(self.normal_scores).cpu())
        self.eag_ema = ema_update(self.eag_ema, eag_value, self.ema_weight)

        key = f"val/summary/{self.name}"
        metrics = {key: eag_value, f"{key}_ema": self.eag_ema}
        pl_module.log_dict(metrics, **self.log_kwargs)
