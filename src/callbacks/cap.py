from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers import MLFlowLogger

from src.callbacks.components.cap.metric import ApproximationCapacity
from src.callbacks.components.paired_scores import LOG_KWARGS, PairedScores


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

    :param output_name: Key in outputs dict containing per-event anomaly scores / losses.
    :param dataset_1: Name of the first validation dataloader.
    :param dataset_2: Name of the second validation dataloader.
    :param pairing_type: Pairing scheme name passed to get_pairing_fn.
    :param cap_metric_config: Keyword arguments used to instantiate
        ApproximationCapacity.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

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
        self.cap_metric_config = cap_metric_config
        self.pairs = PairedScores(
            "cap", output_name, dataset_1, dataset_2, pairing_type, ema_weight
        )

    def on_fit_start(self, trainer, pl_module):
        """Instantiate useful quantities."""
        self.pairs.start()
        self.curve_epochs = []
        self.curves = []

    def on_validation_epoch_start(self, trainer, pl_module):
        """Validate dataloaders and initialize metric state."""
        self.pairs.reset(trainer)
        self.capmetric = ApproximationCapacity(**self.cap_metric_config)
        self.capmetric.to(pl_module.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Aggregate anomaly scores from the two designated validation data sets."""
        self.pairs.collect(trainer, outputs, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the metric across the two data sets."""
        if trainer.sanity_checking:
            return
        metrics = {self.pairs.key("ema"): self.pairs.average(self._compute_cap())}
        pl_module.log_dict(metrics | self._store_curve(trainer), **LOG_KWARGS)

    def on_fit_end(self, trainer, pl_module):
        """Log the file of CAP(beta) curves to MLflow, if the run logs there."""
        if self.curves and isinstance(trainer.logger, MLFlowLogger):
            trainer.logger.experiment.log_artifact(
                trainer.logger.run_id, str(self._curves_path(trainer))
            )

    def _curves_path(self, trainer) -> Path:
        name = f"cap_curves_{self.pairs.pair_name}.npz"
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
        return {
            self.pairs.key("beta_star"): self.capmetric.beta_star,
            self.pairs.key("at_bound"): float(self.capmetric.at_bound),
        }

    def _compute_cap(self):
        """Compute the metric between the two data sets."""
        ds1_scores, ds2_scores = self.pairs.paired()

        with torch.inference_mode(False):
            with torch.enable_grad():
                ds1 = ds1_scores.clone().requires_grad_(True)
                ds2 = ds2_scores.clone().requires_grad_(True)
                self.capmetric.update(ds1, ds2)

        cap_value = self.capmetric.compute()
        if isinstance(cap_value, torch.Tensor):
            cap_value = cap_value.detach().item()

        return float(cap_value)
