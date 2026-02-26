# Callback that computes the anomaly efficiency.
from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer

from src.evaluation.callbacks.metrics.rate import AnomalyRate
from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class AnomalyEfficiencyCallback(Callback):
    """Calculates the fraction of anomalies detected given a bkg rate.

    The checkpoints given to this callback can be saved on metrics different than
    the ones that his callback is instantiated with. For example, you can evaluate
    a model checkpointed on the minimum of the total loss by using the KL div as the
    anomaly score, rather than the total loss, in this callback.

    :param target_rate: Float specifying the target rate of bkg data.
    :param bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total efficiency of events that are processed by
        the L1 trigger.
    :param materic_name: String speicfying the model metric to use as anomaly score.
        Needs to be in the outdict returned by the model.
    :param pure_thres: Whether to set the threshold on the pure data sets, i.e.,
        the data that is not triggered by any L1 trigger algorithm.
    :param ds: List of strings containing data set names to compute the efficiencies on.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    :param name: String specifying the name of the callback for identification in
        later methods that manipulate callbacks.
    """
    def __init__(
        self,
        target_rates: list[int],
        bc_rate: float,
        metric_name: str,
        ds: list[str],
        pure_thres: bool = False,
        log_raw_mlflow: bool = True,
        name: str = 'eff'
    ):
        super().__init__()
        self.device = None
        self.name = name

        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.metric_name = metric_name
        self.pure_thres = pure_thres
        self.ds = set(ds)

        self.log_raw_mlflow = log_raw_mlflow
        self.eff_summary = defaultdict(lambda: defaultdict(float))
        self.eff_min = defaultdict(lambda: defaultdict(float))
        self.eff_med = defaultdict(lambda: defaultdict(float))

    def on_test_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""
        self.device = pl_module.device
        self.thresholds = defaultdict(float)
        for target_rate in self.target_rates:
            trate_name = f'{target_rate}'.replace(".", "_")
            self.thresholds[target_rate] = getattr(pl_module, f'thres_{trate_name}kHz')

        # Check if 'main_test' dataset is the first dataset in the test dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        if first_test_dset_key != "main_test":
            raise ValueError("Eff callback requires main_test first in the data dict!")

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.main_rate = defaultdict(lambda: defaultdict(AnomalyRate))
        self.sig_rates = defaultdict(lambda: defaultdict(AnomalyRate))
        self.bkg_rates = defaultdict(lambda: defaultdict(AnomalyRate))
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

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        eff_name = "effs_pure" if self.pure_thres else "effs"
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / eff_name
        plot_folder.mkdir(parents=True, exist_ok=True)

        for trate in self.target_rates:
            # Construct eff per data set for specific metric name and target rate.
            main_eff = self._compute_eff(self.main_rate, trate)
            sig_effs = self._compute_eff(self.sig_rates, trate)
            bkg_effs = self._compute_eff(self.bkg_rates, trate)
            effs = sig_effs | bkg_effs | main_eff

            ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
            sig_data = np.fromiter(sig_effs.values(), dtype=float)
            self.eff_med[trate][ckpt_ds] = np.median(sig_data)
            self.eff_min[trate][ckpt_ds] = min(sig_data)

            trate_name = f"{trate} kHz"
            ascore = f"anomaly score: {self.metric_name}"
            xlabel = f"efficiency at threshold: {trate_name}\n{ascore}"
            self._plot(effs, xlabel, plot_folder, percent=True)
            self._store_summary(sig_effs, bkg_effs, ckpt_name, trate)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            eff_name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{eff_name}_{self.metric_name.replace('/', '_')}",
        )

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the summary metrics accummulated in eff_summary and reset this attr."""
        eff_name = "effs_pure" if self.pure_thres else "effs"
        plot_folder = root_folder / "plots" / (eff_name + "_summary")
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        for target_rate in self.eff_summary.keys():
            # Get the summary metric per checkpoint.
            smet = self.eff_summary[target_rate]

            # Configure plot.
            trate = f"{target_rate} kHz"
            ascore = f"anomaly score: {self.metric_name}"
            xlabel = f"25% CVaR at threshold: {trate}\n{ascore}"
            self._plot(smet, xlabel, plot_folder, True)
            xlabel = f"med eff at:{trate}\n{ascore}"
            self._plot(self.eff_med[target_rate], xlabel, plot_folder, True)
            xlabel = f"min eff at:{trate}\n{ascore}"
            self._plot(self.eff_min[target_rate], xlabel, plot_folder, True)

        utils.mlflow.log_plots_to_mlflow(trainer, None, eff_name, plot_folder)

    def clear_crit_summary(self):
        self.eff_summary.clear()
        self.eff_med.clear()
        self.eff_min.clear()

    def get_optimized_metric(self, target_rate: float):
        """Get one number that one should optimize on this callback.

        Here, it's the maximum of the summary metric across checkpoints corresponding
        to a certain checkpointing criterion. See how the summary metric is defined in
        self._store_summary.
        """
        if not target_rate in list(self.eff_summary.keys()):
            raise ValueError(
                f"Efficiency callback for metric {self.metric_name} did not calculate "
                f"eff at target rate {target_rate}. Choose {self.eff_summary.keys()}."
            )

        metric_across_ckpts = self.eff_summary[target_rate]
        max_ckpt_ds = max(metric_across_ckpts, key=metric_across_ckpts.get)
        max_metric_value = metric_across_ckpts[max_ckpt_ds]
        return max_ckpt_ds, max_metric_value

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        """Plot the efficiency per data set for an anomaly metric at target rate."""
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def _store_summary(self, sig_effs: dict, bkg_effs: dict, ckpt: str, trate: float):
        """Store the summary statistic for the efficiency for one checkpoint.

        We compute the CVaR robustness metric by collecting the TPR values across all
        simulated anomaly types and averaging the worst fraction of them (e.g., the
        lowest 25%). Concretely, we select the smallest subset of TPR values
        corresponding to the chosen tail level and compute their mean. This tail-average
        summarizes detection performance on the most challenging anomaly datasets.
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

        summary_metric = 1e3 * (sig_cvar - bkg_mean)

        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt)
        self.eff_summary[trate][ckpt_ds] = summary_metric

    def _accumulate_maintest_output(
        self, outputs: dict, batch_idx: int, l1bit: torch.Tensor
    ):
        """Accummulates the specified metric data across batches.

        Used if currently processing the main_test data set, accummulate the values of
        each metric across batches. The whole metric output distribution is needed to
        set a treshold that gives a certain rate.
        """
        batch_output = outputs[self.metric_name]
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
            rate.apply_threshold(self.thresholds[target_rate])
            rate.update(self.maintest_score_data)
            self.main_rate[target_rate][self.dataset_name] = rate

    def _initialize_rate_metric(self, labels: torch.Tensor):
        """Initializes the rate metric for a sig dataset for each given target rate."""
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.apply_threshold(self.thresholds[target_rate])
            if torch.all(labels < 0):
                self.bkg_rates[target_rate][self.dataset_name] = rate
            if torch.all(labels > 0):
                self.sig_rates[target_rate][self.dataset_name] = rate

    def _compute_batch_rate(self, outputs: dict, labels: torch.Tensor):
        """Compute batch rate of a background data set."""
        for tr in self.target_rates:
            if torch.all(labels < 0):
                self.bkg_rates[tr][self.dataset_name].update(outputs[self.metric_name])
            if torch.all(labels > 0):
                self.sig_rates[tr][self.dataset_name].update(outputs[self.metric_name])

    def _compute_eff(self, rates: dict, target_rate: float):
        """Compute the efficiency in given rate dictionary."""
        effs = defaultdict(float)
        clean_metric_name = self.metric_name.replace('/', '_')
        for ds_name, rate in rates[target_rate].items():
            logging_name = f"{ds_name}"
            effs[logging_name] = rate.compute('efficiency').item()

        return effs

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.eff_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
