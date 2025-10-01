# Callback that calculates the rate of anomalies.
from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback

from src.models.metrics.rate import AnomalyRate
from src.models.metrics.rate import ClassifierRate


class AnomalyRateCallback(Callback):
    """Calculates the how many anomalies are detected given a certain trigger rate.

    :target_rate: Float specifying the target rate of anomalies.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total rate of events that are processed by
        the L1 trigger.
    """

    def __init__(self, target_rates: list[int], bc_rate: int, metric_names: list[str]):
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.rates = {}

        self.metric_names = metric_names
        self.metrics_main = {}

    def on_validation_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        # Check if 'main_val' dataset is the first dataset in the validation dictionary.
        # This is required to compute the thresholds at first.
        first_val_dset_key = list(trainer.val_dataloaders.keys())[0]
        if first_val_dset_key != "main_val":
            raise ValueError("Rate callback requires main_val first in the val dict!")

    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        for mname in self.metrics_main.keys():
            self.metrics_main[mname] = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        dset_key = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_key == "main_val":
            self._cache_main_output(outputs, batch_idx)
        else:
            self._setup_metrics_for_dset(dset_key, batch_idx)
            self._update_dset_rate(dset_key, outputs)

    def _cache_main_output(self, outputs: dict, batch_idx: int):
        """Cache the output for a batch to a dictionary defined in the init."""
        for mname in self.metric_names:
            if batch_idx == 0:
                self.metrics_main[mname] = outputs[mname].detach().cpu()
            else:
                self.metrics_main[mname] = torch.cat(
                    [self.metrics_main[mname], outputs[mname].detach().cpu()]
                )

    def _update_dset_rate(self, dset_key: str, outputs: dict):
        """Update the rate for the metric of each data set."""
        for mname in self.metric_names:
            for target_rate in self.target_rates:
                tracking_name = f"{mname.replace('/', '_')}_rate{target_rate}"
                self.rates[f"{dset_key}/{tracking_name}"].update(
                    outputs[mname].detach().cpu()
                )

    def _setup_metrics_for_dset(self, dset_key: str, batch_idx: int):
        """Initializes the rate metric for a dataset for each given target rate."""
        if batch_idx == 0:
            for mname in self.metrics_main.keys():
                for target_rate in self.target_rates:
                    rate = AnomalyRate(target_rate, self.bc_rate)
                    rate.set_threshold(self.metrics_main[mname])
                    tracking_name = f"{mname.replace('/', '_')}_rate{target_rate}"
                    self.rates.update({f"{dset_key}/{tracking_name}": rate})

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Get the rate of anomalies on the main validation data, i.e., on ZB.
        # This is treated as FPR.
        for mname in self.metric_names:
            for target_rate in self.target_rates:
                rate = AnomalyRate(target_rate, self.bc_rate)
                rate.set_threshold(self.metrics_main[mname])
                rate.update(self.metrics_main[mname])
                tracking_name = f"{mname.replace('/', '_')}_rate{target_rate}"
                pl_module.log_dict(
                    {f"val/main_val/{tracking_name}": rate.compute()},
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        # Get the rate of anomalies on each of the simulations.
        # This is treated as TPR.
        for rate_name, rate in self.rates.items():
            pl_module.log_dict(
                {f"val/{rate_name}": rate.compute()},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )


class ClassifierRateCallback(Callback):
    """Calculates the how many anomalies are detected given a certain trigger rate.

    :target_rate: Float specifying the target rate of anomalies.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total rate of events that are processed by
        the L1 trigger.
    """

    def __init__(self, target_rates: list[int], bc_rate: int, metric_names: list[str]):
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.rates = {}

        self.metric_names = metric_names
        self.metrics_main = {}

    def on_validation_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        # Check if 'main_val' dataset is the first dataset in the validation dictionary.
        # This is required to compute the thresholds at first.
        first_val_dset_key = list(trainer.val_dataloaders.keys())[0]
        if first_val_dset_key != "main_val":
            raise ValueError("Rate callback requires main_val first in the val dict!")

    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        for mname in self.metrics_main.keys():
            self.metrics_main[mname] = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        dset_key = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_key == "main_val":
            self._cache_main_output(outputs, batch_idx)
        else:
            self._setup_metrics_for_dset(dset_key, batch_idx)
            self._update_dset_rate(dset_key, outputs)

    def _cache_main_output(self, outputs: dict, batch_idx: int):
        """Cache the output for a batch to a dictionary defined in the init."""
        for mname in self.metric_names:
            if batch_idx == 0:
                self.metrics_main[mname] = outputs[mname].detach().cpu()
            else:
                self.metrics_main[mname] = torch.cat(
                    [self.metrics_main[mname], outputs[mname].detach().cpu()]
                )

    def _update_dset_rate(self, dset_key: str, outputs: dict):
        """Update the rate for the metric of each data set."""
        for mname in self.metric_names:
            for target_rate in self.target_rates:
                tracking_name = f"{mname.replace('/', '_')}_rate{target_rate}"
                self.rates[f"{dset_key}/{tracking_name}"].update(
                    outputs[mname].detach().cpu()
                )

    def _setup_metrics_for_dset(self, dset_key: str, batch_idx: int):
        """Initializes the rate metric for a dataset for each given target rate."""
        if batch_idx == 0:
            for mname in self.metrics_main.keys():
                for target_rate in self.target_rates:
                    rate = ClassifierRate(target_rate, self.bc_rate)
                    rate.set_threshold(self.metrics_main[mname])
                    tracking_name = f"{mname.replace('/', '_')}_rate{target_rate}"
                    self.rates.update({f"{dset_key}/{tracking_name}": rate})

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Get the rate of anomalies on the main validation data, i.e., on ZB.
        # This is treated as FPR.
        for mname in self.metric_names:
            for target_rate in self.target_rates:
                rate = ClassifierRate(target_rate, self.bc_rate)
                rate.set_threshold(self.metrics_main[mname])
                rate.update(self.metrics_main[mname])
                tracking_name = f"{mname.replace('/', '_')}_rate{target_rate}"
                pl_module.log_dict(
                    {f"val/main_val/{tracking_name}": rate.compute()},
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        # Get the rate of anomalies on each of the simulations.
        # This is treated as TPR.
        for rate_name, rate in self.rates.items():
            pl_module.log_dict(
                {f"val/{rate_name}": rate.compute()},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )
