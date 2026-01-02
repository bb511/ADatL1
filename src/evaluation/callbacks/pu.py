# Plots the correlation of the anomaly rate with the pileup.
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningDataModule

from src.evaluation.callbacks.metrics.rate import AnomalyRate
from src.evaluation.callbacks import utils
from src.data.components.normalization import L1DataNormalizer
from src.plot import scatter


class BkgRatePileup(Callback):
    """Plots the correlation between the background data sets rate and the pileup.

    :param target_rate: Float specifying the target rate of anomalies.
    :param bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total efficiency of events that are processed by
        the L1 trigger.
    :param datamodule: The datamodule that was used during the training of the model.
    :param materic_name: String speicfying the model metric to use as anomaly score.
        Needs to be in the outdict returned by the model.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(
        self,
        target_rates: list[int],
        bc_rate: int,
        metric_name: str,
        datamodule: LightningDataModule,
        log_raw_mlflow: bool = True
    ):
        super().__init__()
        self.device = None
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.metric_name = metric_name
        self.log_raw_mlflow = log_raw_mlflow
        self.pu_per_ds = self._get_pileup_data(datamodule)

    def on_test_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        # Check if 'main_test' dataset is the first dataset in the test dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        self.device = pl_module.device
        if first_test_dset_key != "main_test":
            raise ValueError("PU callback requires main_test first in the data dict!")

    def _get_pileup_data(self, datamodule: LightningDataModule):
        """Get the pileup data for each test data set.

        Get the nPV_True feature of the event_info object, put it in the subfolder of
        the data in 'callbacks_pileup' and don't normalize it.
        """
        extra_feats = {"event_info": ["nPV_True"]}
        flag = "callbacks_pileup"
        normalizer = L1DataNormalizer("unnormalized", {})

        _, _, test_data = datamodule.get_extra(normalizer, extra_feats, flag)

        return test_data

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.rates_per_pu = defaultdict(dict)
        self.maintest_score_data = []
        self.maintest_pu_data = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Determine the rate for every given metric for the simulated bkg data.

        First, the desired metrics are computed on the main_test data and accummulated
        across batches. The full metric distribution is used to set a threshold that
        will give a target rate or set of rates, specified by the user.
        These thresholds are then applied on the bkg simulation data sets, i.e., the
        data sets that have target == -1, to determine what the rate on this other data
        would be at that threshold.
        """
        self.dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        self.total_batches = trainer.num_test_batches[dataloader_idx]
        _, labels = batch

        if self.dataset_name == "main_test":
            self._accumulate_maintest_output(outputs, batch_idx)
        elif (labels < 0).all().item():
            self._compute_batch_rate(outputs, batch_idx)
        else:
            continue

    def _accumulate_maintest_output(self, outputs: dict, batch_idx: int):
        """Accummulates the specified metric data across batches.

        Used if currently processing the main_test data set, accummulate the values of
        each metric across batches. The whole metric output distribution is needed to
        set a treshold that gives a certain rate.
        """
        batch_output = outputs[self.metric_name]
        self.maintest_score_data.append(batch_output)

        pu, _ = self.pu_per_ds["main_test"][batch_idx]
        pu = pu.round().int().to("cpu").view(-1)
        self.maintest_pu_data.append(pu)

        if batch_idx == self.total_batches - 1:
            self.maintest_score_data = torch.cat(self.maintest_score_data, dim=0)
            self.maintest_pu_data = torch.cat(self.maintest_pu_data, dim=0)
            self._compute_maintest_rate(self.metric_name)

    def _compute_maintest_rate(self):
        """Computes the desired rates on the main test data set.

        This is a sanity check. The threshold computed on the maintest and applied to the
        maintest data should return the rate for which this threshold was computed.
        """
        unique_pu, inv = self.maintest_pu_data.unique(sorted=True, return_inverse=True)
        scores_by_pu = {
            int(pu.item()): self.maintest_score_data[inv == i]
            for i, pu in enumerate(unique_pu)
        }
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.maintest_score_data)
            rate_name = f"{self.dataset_name}/{target_rate}"
            for pu, scores in scores_by_pu.items():
                rate.update(scores)
                self.rates_per_pu[rate_name][pu] = rate

    def _compute_batch_rate(self, outputs: dict, batch_idx: int):
        """Done after knowing the rate thresholds.

        For all the other test data sets, except the main one, this calculates
        which events pass the rate threshold for each batch and updates the total.
        """
        pus, _ = self.pu_per_ds[self.dataset_name][batch_idx]
        pus = pus.round().int().to("cpu").view(-1)
        pus, inv = pus.unique(sorted=True, return_inverse=True)

        scores_by_pu = {
            int(pu.item()): outputs[self.metric_name].detach().to("cpu").float()[inv == i]
            for i, pu in enumerate(pus)
        }
        for target_rate in self.target_rates:
            rate_name = f"{self.dataset_name}/{target_rate}"
            for pu, scores in scores_by_pu.items():
                if not pu in self.rates_per_pu[rate_name].keys():
                    self._initialize_rate_metric(self.metric_name, target_rate, pu)
                self.rates_per_pu[rate_name][pu].update(scores)

    def _initialize_rate_metric(self, target_rate: float, pu: int):
        """Initialize the rate metric for a certain target rate and pu.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the main_test metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from main_test.
        """
        rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
        rate.set_threshold(self.maintest_score_data)
        rate_name = f"{self.dataset_name}/{target_rate}"
        self.rates_per_pu[rate_name][pu] = rate

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / "bkgpileup"
        plot_folder.mkdir(parents=True, exist_ok=True)

        for rate_name, pu_rates in self.rates_per_pu.items():
            target_rate = rate_name.split("/")[1]
            dataset_name = self._get_dsname(rate_name)
            pu_rates = self._compute_rates(pu_rates)

            xlabel = f"pileup"
            ylabel = f"{self.metric_name} rate (kHz) @ {target_rate} kHz"
            scatter.plot_connected(pu_rates, xlabel, ylabel, dataset_name, plot_folder)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name, "pus",
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f'pus_{self.metric_name}'
        )

    def _get_dsname(self, rate_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = rate_name.split("/")[0]
        return dataset_name

    def _compute_rates(self, pu_rates: dict):
        """Compute the rates in the pileup_rates dictionary."""
        for pu_value, anomaly_counter in pu_rates.items():
            pu_rates[pu_value] = anomaly_counter.compute("rate").item()

        return pu_rates
