# Callback that computes the anomaly rate.
from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger, Logger
from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from src.evaluation.callbacks.metrics.rate import AnomalyRate
from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class AnomalyRateCallback(Callback):
    """Calculates the how many anomalies are detected given a certain trigger rate.

    The checkpoints given to this callback can be saved on metrics different than
    the ones that his callback is instantiated with. For example, you can evaluate
    a model checkpointed on the minimum of the total loss by using the KL div as the
    anomaly score, rather than the total loss, in this callback.

    :param target_rate: Float specifying the target rate of anomalies.
    :param bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total rate of events that are processed by
        the L1 trigger.
    :param materic_name: String speicfying the model metric to use as anomaly score.
        Needs to be in the outdict returned by the model.
    :param pure_thres: Whether to set the threshold on the pure data sets, i.e.,
        the data that is not triggered by any L1 trigger algorithm.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(
        self,
        target_rates: list[int],
        bc_rate: float,
        metric_name: str,
        ds: list[str],
        pure_thres: bool = False,
        log_raw_mlflow: bool = True,
    ):
        super().__init__()
        self.device = None
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.metric_name = metric_name
        self.pure_thres = pure_thres
        self.ds = set(ds)
        self.log_raw_mlflow = log_raw_mlflow
        self.rate_summary = defaultdict(lambda: defaultdict(float))

    def on_test_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""
        self.device = pl_module.device

        # Check if 'main_test' dataset is the first dataset in the test dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        if first_test_dset_key != "main_test":
            raise ValueError("rate callback requires main_test first in the data dict!")

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.main_rate = {}
        self.sig_rates = {}
        self.bkg_rates = {}
        self.maintest_score_data = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Determine the rate for every given metric for every test data set.

        First, the desired metrics are computed on the main_test data and accummulated
        across batches. The full metric distribution is used to set a threshold that
        will give a target rate or set of rates, specified by the user.
        These thresholds are then applied on all other data sets used for test
        to determine, by using the threshold computed on main_test, what would the rate
        be on these other data sets.
        """
        self.dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        self.total_batches = trainer.num_test_batches[dataloader_idx]
        _, _, l1bit, labels = batch

        if self.dataset_name == "main_test":
            self._accumulate_maintest_output(outputs, batch_idx, l1bit)
        else:
            if self.dataset_name not in self.ds:
                return
            if batch_idx == 0:
                self._initialize_rate_metric(labels)
            self._compute_batch_rate(outputs, labels)

    def _accumulate_maintest_output(
        self, outputs: dict, batch_idx: int, l1bit: torch.Tensor
    ):
        """Accummulates the specified metric data across batches.

        Used if currently processing the main_test data set, accummulate the values of
        each metric across batches. The whole metric output distribution is needed to
        set a treshold that gives a certain rate.
        """
        batch_output = outputs[self.metric_name].detach()
        if self.pure_thres:
            batch_output = batch_output[~l1bit]
        self.maintest_score_data.append(batch_output)

        if batch_idx == self.total_batches - 1:
            self.maintest_score_data = torch.cat(self.maintest_score_data, dim=0)
            self._compute_maintest_rate()

    def _compute_maintest_rate(self):
        """Computes the desired rates on the main test data set.

        This is a sanity check. The threshold computed on the maintest and applied to the
        maintest data should return the rate for which this threshold was computed.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.maintest_score_data)
            rate.update(self.maintest_score_data)
            self.main_rate.update({f"{self.dataset_name}/{target_rate}": rate})

    def _initialize_rate_metric(self, labels: torch.Tensor):
        """Initialise the rate metric for signal or bkg data.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the main_test metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from main_test.
        """
        if bool(torch.all(labels >= 1)):
            self._initialize_sig_rate_metric()
        elif bool(torch.all(labels <= -1)):
            self._initialize_bkg_rate_metric()
        else:
            raise ValueError(
                "The rate test callback requires that the labels of the signal "
                "data are all >= 1 and labels of background data are all <= -1."
            )

    def _initialize_sig_rate_metric(self):
        """Initializes the rate metric for a sig dataset for each given target rate."""
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.maintest_score_data)
            self.sig_rates.update({f"{self.dataset_name}/{target_rate}": rate})

    def _initialize_bkg_rate_metric(self):
        """Initializes the rate metric for a bkg dataset for each given target rate."""
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.maintest_score_data)
            self.bkg_rates.update({f"{self.dataset_name}/{target_rate}": rate})

    def _compute_batch_rate(self, outputs: dict, labels: torch.Tensor):
        """Done after knowing the rate thresholds.

        For all the other test data sets, except the main one, this calculates
        which events pass the rate threshold for each batch and updates the total.
        """
        if bool(torch.all(labels >= 1)):
            self._compute_sig_batch_rate(outputs)
        elif bool(torch.all(labels <= -1)):
            self._compute_bkg_batch_rate(outputs)
        else:
            raise ValueError(
                "The rate test callback requires that the labels of the signal "
                "data are all >= 1 and labels of background data are all <= -1."
            )

    def _compute_sig_batch_rate(self, outputs: dict):
        """Compute batch rate of a signal dataset."""
        for target_rate in self.target_rates:
            rate_name = f"{self.dataset_name}/{target_rate}"
            self.sig_rates[rate_name].update(outputs[self.metric_name].detach())

    def _compute_bkg_batch_rate(self, outputs: dict):
        """Compute batch rate of a background data set."""
        for target_rate in self.target_rates:
            rate_name = f"{self.dataset_name}/{target_rate}"
            self.bkg_rates[rate_name].update(outputs[self.metric_name].detach())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        rate_name = "rates_pure" if self.pure_thres else "rates"
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / rate_name
        plot_folder.mkdir(parents=True, exist_ok=True)

        for trate in self.target_rates:
            # Construct rate per data set for specific metric name and target rate.
            sig_rates = self._compute_rate(trate, self.sig_rates)
            bkg_rates = self._compute_rate(trate, self.bkg_rates)
            main_rate = self._compute_rate(trate, self.main_rate)
            rates = sig_rates | bkg_rates | main_rate

            trate_name = f"{trate} kHz"
            ascore = f"anomaly score: {self.metric_name}"
            xlabel = f"rate at threshold: {trate_name}\n{ascore}"
            self._plot(rates, xlabel, plot_folder, percent=False)
            self._store_summary(sig_rates, bkg_rates, ckpt_name, trate)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            rate_name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{rate_name}_{self.metric_name.replace('/', '_')}",
        )

    def _compute_rate(self, relevant_target_rate: float, rate_dict: dict):
        """Compute the rate in given rate dictionary."""
        rates = {
            self._get_dsname(rate_name): val.compute("rate").item()
            for rate_name, val in rate_dict.items()
            if str(relevant_target_rate) in rate_name
        }
        return rates

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the summary metrics accummulated in rate_summary and reset this attr."""
        rate_name = "rates_pure" if self.pure_thres else "rates"
        plot_folder = root_folder / "plots" / (rate_name + "_summary")
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        for target_rate in self.rate_summary.keys():
            # Get the summary metric per checkpoint.
            smet = self.rate_summary[target_rate]

            # Configure plot.
            trate = f"{target_rate} kHz"
            ascore = f"anomaly score: {self.metric_name}"
            xlabel = f"mu^2/sigma at threshold: {trate}\n{ascore}"
            self._plot(smet, xlabel, plot_folder)

        utils.mlflow.log_plots_to_mlflow(trainer, None, rate_name, plot_folder)

    def clear_crit_summary(self):
        self.rate_summary.clear()

    def get_optimized_metric(self, target_rate: float):
        """Get one number that one should optimize on this callback.

        Here, it's the maximum of the summary metric across checkpoints corresponding
        to a certain checkpointing criterion.
        """
        if not target_rate in list(self.rate_summary.keys()):
            raise ValueError(
                f"rate callback for metric {self.metric_name} did not calculate "
                f"rate at target rate {target_rate}. Choose {self.rate_summary.keys()}."
            )

        metric_across_ckpts = self.rate_summary[target_rate]
        max_ckpt_ds = max(metric_across_ckpts, key=metric_across_ckpts.get)
        max_metric_value = metric_across_ckpts[max_ckpt_ds]
        return max_ckpt_ds, max_metric_value

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        """Plot the rate per data set for an anomaly metric at target rate."""
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def _store_summary(self, sig_rates: dict, bkg_rates: dict, ckpt: str, trate: float):
        """Store the summary statistic for the rate for one checkpoint.

        We compute the CVaR robustness metric by collecting the TPR values across all
        simulated anomaly types and averaging the worst fraction of them (e.g., the
        lowest 20%). Concretely, we select the smallest subset of TPR values
        corresponding to the chosen tail level and compute their mean. This tail-average
        summarizes detection performance on the most challenging anomaly scenarios.
        """
        sig_data = np.fromiter(sig_effs.values(), dtype=float)
        bkg_data = np.fromiter(bkg_effs.values(), dtype=float)

        bkg_mean = bkg_data.mean() if bkg_data.size else 0.0

        alpha = 0.25
        if sig_data.size:
            k = max(1, int(np.ceil(alpha * sig_data.size)))
            worst = np.partition(sig_data, k - 1)[:k]
            sig_cvar = worst.mean()
        else:
            sig_cvar = 0.0

        summary_metric = sig_cvar - bkg_mean

        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt)
        self.rate_summary[trate][ckpt_ds] = summary_metric

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.rate_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_dsname(self, rate_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = rate_name.split("/")[0]
        return dataset_name
