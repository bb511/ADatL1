import math
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers import MLFlowLogger

from src.callbacks.metrics.cap.binary import get_pairing_fn
from src.callbacks.metrics.cap.metric import ApproximationCapacity
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class CAPCallback(Callback):
    """Compute a validation approximation capacity metric between two datasets.

    This callback collects all scores from two validation dataloaders, pairs the
    samples according to the selected pairing function, and computes the CAP
    metric on the paired score samples from:
        - dataset_1
        - dataset_2

    This is appropriate as a validation-side proxy objective for HPO.

    Beside the EMA, every epoch logs the maximising beta and whether it sits at the
    upper bound of the beta grid, and adds its CAP(beta) curve to
    `<default_root_dir>/cap_curves_<dataset_1>_vs_<dataset_2>.npz` (arrays `betas`,
    `epochs`, `curves`), which is logged as an MLflow artifact at the end of the fit.
    Metrics without a beta curve, such as PosteriorConsistency, log only the EMA.

    Subclasses select a different metric by overriding `metric_cls`, and a
    different logged key by overriding `metric_key`; see
    src.callbacks.consistency.PosteriorConsistencyCallback.

    :param output_name: Key in outputs dict containing per-event anomaly scores / losses.
    :param dataset_1: Name of the first validation dataloader.
    :param dataset_2: Name of the second validation dataloader.
    :param pairing_type: Pairing scheme name passed to get_pairing_fn.
    :param cap_metric_config: Keyword arguments used to instantiate
        ApproximationCapacity.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

    metric_cls = ApproximationCapacity
    metric_key = "cap"

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
        cap_metric_config: dict,
        ema_weight: float = 0.9,
    ):
        super().__init__()
        self.device = None
        self.output_name = output_name
        self.dataset_1_name = dataset_1
        self.dataset_2_name = dataset_2
        ds1 = dataset_1.replace("/", "_")
        ds2 = dataset_2.replace("/", "_")
        self.pair_name = f"{ds1}_vs_{ds2}"
        self.cap_metric_config = cap_metric_config
        self.pairing_fn = get_pairing_fn(pairing_type)
        self.ema_weight = ema_weight

        self.log_kwargs = dict(
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def on_fit_start(self, trainer, pl_module):
        """Instantiate useful quantities."""
        self.cap_ema = math.nan
        self.curve_epochs = []
        self.curves = []

    def on_validation_epoch_start(self, trainer, pl_module):
        """Set device, validate dataloaders, and initialize metric state."""
        self.device = pl_module.device

        dset_names = list(trainer.val_dataloaders.keys())
        missing = [
            dset_name
            for dset_name in (self.dataset_1_name, self.dataset_2_name)
            if dset_name not in dset_names
        ]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires validation dataloaders {missing}. "
                f"Available validation dataloaders: {dset_names}"
            )

        self.dataset_1_scores = []
        self.dataset_2_scores = []

        self.capmetric = self.metric_cls(**self.cap_metric_config)
        self.capmetric.to(self.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Aggregate anomaly scores from the two designated validation data sets."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name not in (self.dataset_1_name, self.dataset_2_name):
            return

        loss = outputs[self.output_name]
        if loss.ndim == 0:
            raise ValueError(f"outputs['{self.output_name}'] is scalar. Need a tensor.")

        loss = loss.detach().view(-1)

        if dset_name == self.dataset_1_name:
            self.dataset_1_scores.append(loss)

        if dset_name == self.dataset_2_name:
            self.dataset_2_scores.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the metric across the two data sets."""
        if not self.dataset_1_scores:
            raise RuntimeError(
                f"No validation scores were collected for '{self.dataset_1_name}'."
            )
        if not self.dataset_2_scores:
            raise RuntimeError(
                f"No validation scores were collected for '{self.dataset_2_name}'."
            )

        cap_value = self._compute_cap()
        self._compute_cap_ema(cap_value)

        metrics = {f"val/summary/{self.metric_key}_ema_{self.pair_name}": self.cap_ema}
        if self.capmetric.curve is not None:
            metrics |= self._store_curve(trainer)
        pl_module.log_dict(metrics, **self.log_kwargs)

    def on_fit_end(self, trainer, pl_module):
        """Log the file of CAP(beta) curves to MLflow, if the run logs there."""
        if self.curves and isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_artifact(
                trainer.logger.run_id, str(self._curves_path(trainer))
            )

    def _curves_path(self, trainer) -> Path:
        name = f"{self.metric_key}_curves_{self.pair_name}.npz"
        return Path(trainer.default_root_dir) / name

    def _store_curve(self, trainer) -> dict:
        """Rewrite the curve file with this epoch's curve; return what to log."""
        self.curve_epochs.append(trainer.current_epoch)
        self.curves.append(self.capmetric.curve.numpy())
        np.savez(
            self._curves_path(trainer),
            betas=self.capmetric.betas.cpu().numpy(),
            epochs=np.array(self.curve_epochs),
            curves=np.stack(self.curves),
        )
        prefix = f"val/summary/{self.metric_key}"
        return {
            f"{prefix}_beta_star_{self.pair_name}": self.capmetric.beta_star,
            f"{prefix}_at_bound_{self.pair_name}": float(self.capmetric.at_bound),
        }

    def _compute_cap(self):
        """Compute the metric between the two data sets."""
        dataset_1_scores = torch.cat(self.dataset_1_scores, dim=0).view(-1)
        dataset_2_scores = torch.cat(self.dataset_2_scores, dim=0).view(-1)

        idxs1, idxs2 = self.pairing_fn(dataset_1_scores, dataset_2_scores)
        ds1_scores = dataset_1_scores[idxs1]
        ds2_scores = dataset_2_scores[idxs2]

        n = min(len(ds1_scores), len(ds2_scores))
        if n <= 0:
            raise RuntimeError("No paired validation samples available for CAP.")

        ds1_scores = ds1_scores[:n]
        ds2_scores = ds2_scores[:n]

        with torch.inference_mode(False):
            with torch.enable_grad():
                ds1 = ds1_scores.clone().requires_grad_(True)
                ds2 = ds2_scores.clone().requires_grad_(True)
                self.capmetric.update(ds1, ds2)

        cap_value = self.capmetric.compute()
        if isinstance(cap_value, torch.Tensor):
            cap_value = cap_value.detach().item()

        return float(cap_value)

    def _compute_cap_ema(self, cap: float):
        """Update the exponential moving average, which skips a non-finite value."""
        if not math.isfinite(cap):
            log.warning(f"Non-finite {self.metric_key} {cap} left out of its EMA.")
        elif math.isnan(self.cap_ema):
            self.cap_ema = float(cap)
        else:
            self.cap_ema = self.ema_weight * self.cap_ema + (
                1 - self.ema_weight
            ) * float(cap)
