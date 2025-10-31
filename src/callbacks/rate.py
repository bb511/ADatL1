# Callback that computes the anomaly rate during training.
from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.metrics.rate import AnomalyRate


class AnomalyRateCallback(Callback):
    """Calculates the how many anomalies are detected given a certain trigger rate.

    :target_rate: Float specifying the target rate of anomalies.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total rate of events that are processed by
        the L1 trigger.
    :materic_names: Which model metrics to use as the anomaly score to calculate the
        rate on.
    """

    def __init__(self, target_rates: list[int], bc_rate: int, metric_names: list[str]):
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.rates = {}

        self.metric_names = metric_names
        self.mainval_score_data = defaultdict(list)

    def on_validation_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        # Check if 'main_val' dataset is the first dataset in the validation dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        first_val_dset_key = list(trainer.val_dataloaders.keys())[0]
        if first_val_dset_key != "main_val":
            raise ValueError("Rate callback requires main_val first in the val dict!")

    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        for mname in self.mainval_score_data.keys():
            self.mainval_score_data[mname] = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Determine the rate for every given metric for every validation data set.

        First, the desired metrics are compute on the mainval_data and accummulated
        across batches. The full metric distribution is used to set a threshold that
        will give a given rate or set of rates, specified by the user.
        These thresholds are then applied on all other data sets used for validation
        to determine, by using the threshold computed on mainval, what would the rate
        be on these other data sets.
        """
        self.dataset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        self.total_batches = trainer.num_val_batches[dataloader_idx]

        for metric_name in self.metric_names:
            if self.dataset_name == "main_val":
                self._accumulate_mainval_output(outputs, batch_idx, metric_name)
            else:
                if batch_idx == 0:
                    self._initialize_rate_metric(batch_idx, metric_name)
                self._compute_batch_rate(outputs, metric_name)


    def _accumulate_mainval_output(self, outputs: dict, batch_idx: int, mname: str):
        """Accummulates the specified metric data across batches.

        Used if currently processing the main_val data set, accummulate the values of
        each metric across batches. The whole metric output distribution is needed to
        set a treshold that gives a certain rate.
        """
        batch_output = outputs[mname].detach().to('cpu')
        self.mainval_score_data[mname].append(batch_output)

        if batch_idx == self.total_batches - 1:
            self.mainval_score_data[mname] = torch.cat(
                self.mainval_score_data[mname], dim=0
            )
            self._compute_mainval_rate(mname)

    def _compute_mainval_rate(self, mname: str):
        """Computes the desired rates on the main validation data set.

        This is a sanity check. The threshold computed on the mainval and applied to the
        mainval data should return the rate for which this threshold was computed.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate)
            rate.set_threshold(self.mainval_score_data[mname])
            rate.update(self.mainval_score_data[mname])
            rate_name = f"{mname.replace('/', '_')}_rate{target_rate}"
            self.rates.update({f"{self.dataset_name}/{rate_name}": rate})

    def _compute_batch_rate(self, outputs: dict, mname: str):
        """Done after knowing the rate thresholds.

        For all the other validation data sets, except the main one, this calculates
        which events pass the rate threshold for each batch and updates the total.
        """
        for target_rate in self.target_rates:
            rate_name = f"{mname.replace('/', '_')}_rate{target_rate}"
            self.rates[f"{self.dataset_name}/{rate_name}"].update(
                outputs[mname].detach().cpu()
            )

    def _initialize_rate_metric(self, batch_idx: int, mname: str):
        """Initializes the rate metric for a dataset for each given target rate.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the main_val metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from main_val.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate)
            rate.set_threshold(self.mainval_score_data[mname])

            rate_name = f"{mname.replace('/', '_')}_rate{target_rate}"
            self.rates.update({f"{self.dataset_name}/{rate_name}": rate})

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
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
