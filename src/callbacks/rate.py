# Callback that calculates the rate of anomalies.
from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback

from src.models.metrics.rate import AnomalyRate


class AnomalyRateCallback(Callback):
    """Calculates the how many anomalies are detected given a certain trigger rate.

    :target_rate: Float specifying the target rate of anomalies.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total rate of events that are processed by
        the L1 trigger.
    """
    def __init__(self, target_rates: list[int], bc_rate: int):
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.z_mean_main = []
        self.metrics = {}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        dset_key = list(trainer.val_dataloaders.keys())[dataloader_idx]
        self._check_valid_dict_order(dataloader_idx, dset_key)

        if dset_key == 'main_val':
            self.z_mean_main.append(outputs['z_mean'].detach().cpu())
        else:
            self._setup_metrics_for_dset(dset_key, batch_idx)
            for rate in self.target_rates:
                self.metrics[f"{dset_key}/rate_{rate}"].update(outputs['z_mean'].detach().cpu())

        del outputs['z_mean']

    def _check_valid_dict_order(self, dataloader_idx: int, dset_key: str):
        """Check if main_val is first in the validation data dictionary.

        Need to compute the threshold on the full main validation latent space. Hence,
        this validation dataset requires special treatment.
        """
        if dataloader_idx == 0 and dset_key != "main_val":
            raise ValueError("Rate callback requires main_val first in the val dict!")

    def _setup_metrics_for_dset(self, dset_key: str, batch_idx: int):
        """Initializes the rate metric for a dataset for each given target rate."""
        if batch_idx == 0:
            z_mean_main = torch.cat(self.z_mean_main)
            for target_rate in self.target_rates:
                metric = AnomalyRate(target_rate, self.bc_rate)
                metric.set_threshold(z_mean_main)
                self.metrics.update({f"{dset_key}/rate_{target_rate}": metric})

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Get the rate of anomalies on the main validation data, i.e., on ZB.
        # This is treated as FPR.
        for target_rate in self.target_rates:
            metric = AnomalyRate(target_rate, self.bc_rate)
            metric.set_threshold(torch.cat(self.z_mean_main))
            metric.update(torch.cat(self.z_mean_main))
            pl_module.log_dict(
                {f"val/main_val/rate_{target_rate}": metric.compute()},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        # Get the rate of anomalies on each of the simulations.
        # This is treated as TPR.
        for metric_name, metric in self.metrics.items():
            pl_module.log_dict(
                {f"val/{metric_name}": metric.compute()},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        # Reset array that contains the mean latent space values on the main val data.
        self.z_mean_main = []
