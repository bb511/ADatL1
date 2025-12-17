# Callback that computes the anomaly rate during training.
from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger, Logger
from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from src.evaluation.callbacks.metrics.rate import AnomalyCounter
from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class AnomalyEfficiencyCallback(Callback):
    """Calculates the how many anomalies are detected given a certain trigger rate.

    The checkpoints given to this callback can be saved on metrics different than
    the ones that his callback is instantiated with. For example, you can evaluate
    a model checkpointed on the minimum of the total loss by using the KL div as the
    anomaly score, rather than the total loss, in this callback.

    :target_rate: Float specifying the target rate of anomalies.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total efficiency of events that are processed by
        the L1 trigger.
    :materic_names: Which model metrics to use as the anomaly score to calculate the
        efficiency on.
    """

    def __init__(self, target_rates: list[int], bc_rate: int, metric_names: list[str]):
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.metric_names = metric_names
        self.eff_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def on_test_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        # Check if 'main_test' dataset is the first dataset in the test dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        if first_test_dset_key != "main_test":
            raise ValueError("Eff callback requires main_test first in the data dict!")

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.rates = {}
        self.maintest_score_data = defaultdict(list)

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

        for metric_name in self.metric_names:
            if self.dataset_name == "main_test":
                self._accumulate_maintest_output(outputs, batch_idx, metric_name)
            else:
                if batch_idx == 0:
                    self._initialize_rate_metric(metric_name)
                self._compute_batch_rate(outputs, metric_name)

    def _accumulate_maintest_output(self, outputs: dict, batch_idx: int, mname: str):
        """Accummulates the specified metric data across batches.

        Used if currently processing the main_test data set, accummulate the values of
        each metric across batches. The whole metric output distribution is needed to
        set a treshold that gives a certain rate.
        """
        batch_output = outputs[mname]
        self.maintest_score_data[mname].append(batch_output)

        if batch_idx == self.total_batches - 1:
            self.maintest_score_data[mname] = torch.cat(
                self.maintest_score_data[mname], dim=0
            )
            self._compute_maintest_rate(mname)

    def _compute_maintest_rate(self, mname: str):
        """Computes the desired rates on the main test data set.

        This is a sanity check. The threshold computed on the maintest and applied to the
        maintest data should return the rate for which this threshold was computed.
        """
        for target_rate in self.target_rates:
            rate = AnomalyCounter(target_rate, self.bc_rate)
            rate.set_threshold(self.maintest_score_data[mname])
            rate.update(self.maintest_score_data[mname])
            rate_name = f"{mname.replace('/', '_')}_eff{target_rate}"
            self.rates.update({f"{self.dataset_name}/{rate_name}": rate})

    def _compute_batch_rate(self, outputs: dict, mname: str):
        """Done after knowing the rate thresholds.

        For all the other test data sets, except the main one, this calculates
        which events pass the rate threshold for each batch and updates the total.
        """
        for target_rate in self.target_rates:
            rate_name = f"{mname.replace('/', '_')}_eff{target_rate}"
            self.rates[f"{self.dataset_name}/{rate_name}"].update(outputs[mname])

    def _initialize_rate_metric(self, mname: str):
        """Initializes the rate metric for a dataset for each given target rate.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the main_test metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from main_test.
        """
        for target_rate in self.target_rates:
            rate = AnomalyCounter(target_rate, self.bc_rate)
            rate.set_threshold(self.maintest_score_data[mname])

            rate_name = f"{mname.replace('/', '_')}_eff{target_rate}"
            self.rates.update({f"{self.dataset_name}/{rate_name}": rate})

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / 'plots' / ckpt_name / 'efficiencies'
        plot_folder.mkdir(parents=True, exist_ok=True)

        for metric_name in self.metric_names:
            for target_rate in self.target_rates:
                # Construct eff per data set for specific metric name and target rate.
                name = f"{metric_name.replace('/', '_')}_eff{target_rate}"
                efficiencies = {
                    self._get_dsname(rate_name): round(val.compute('efficiency').item(), 4)
                    for rate_name, val in self.rates.items()
                    if name in rate_name
                }
                self._plot(efficiencies, metric_name, target_rate, plot_folder)
                self._store_summary(efficiencies, ckpt_name, metric_name, target_rate)

        self._log_plots_to_mlflow(trainer, ckpt_name, plot_folder)

    def _get_dsname(self, rate_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = rate_name.split('/')[0]
        return dataset_name

    def _plot(self, effs: dict, mname: str, trate: float, plot_folder: Path):
        """Plot the efficiency per data set for an anomaly metric at target rate."""
        xlabel = (
            f"efficiency at threshold: {trate} kHz\n"
            f"anomaly score: {mname}"
        )
        ylabel = ' '
        horizontal_bar.plot_yright(effs, effs, xlabel, ylabel, plot_folder)

    def _store_summary(self, effs: dict, ckpt_name: str, metric: str, trate: float):
        """Store the summary statistic for the efficiency for one checkpoint.

        Here, we compute the variance of efficiency over the data sets and divide by the
        mean to deduce the stability of the checkpoint across data sets.
        Then, we divide the mean by the previous result to get something that is also
        dependend on the absolute efficiency across data sets.
        This is what we want: a checkpoint that is both stable and achieves the best
        performance.
        """
        data = np.fromiter(effs.values(), dtype=float)
        mean = data.mean()
        var = data.var()

        eps = 1e-14
        summary_metric = mean**2/(var + eps)
        ckpt_ds = ckpt_name.split("ds=")
        if len(ckpt_ds) > 1:
            ckpt_ds = ckpt_ds[1].split("__")[0]
        else:
            ckpt_ds = ckpt_ds[0]

        self.eff_summary[metric][trate][ckpt_ds] = summary_metric

    def _log_plots_to_mlflow(self, trainer, ckpt_name: str, plot_folder: Path):
        """Logs the plots generated by this callback to MLFlow."""
        mlflow_logger = utils.mlflow.get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        arti_ckpt_dir = self._resolve_arti_dir(trainer, ckpt_name)
        gallery_dir = arti_ckpt_dir.parent

        # Log each image in the given plot_folder as an artifact.
        IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        for img_path in sorted(plot_folder.glob('*')):
            if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
                mlflow_logger.experiment.log_artifact(
                    run_id=mlflow_logger.run_id,
                    local_path=str(img_path),
                    artifact_path=str(arti_ckpt_dir),
                )

        # Generate an html gallery of the logs plots in the parent dir of the arti.
        _ = utils.mlflow.make_gall(
            mlflow_logger, plot_folder, gallery_dir, ckpt_name, 'index'
        )

    def _resolve_arti_dir(self, trainer, ckpt_name: str = None):
        """Resolve the artifacts directory where the plots will be stored in mlflow."""
        arti_dir = Path()
        if trainer.strat_name:
            arti_dir = arti_dir / trainer.strat_name / 'efficiencies'
        if trainer.metric_name:
            arti_dir = arti_dir / trainer.metric_name
        if trainer.criterion_name:
            arti_dir = arti_dir / trainer.criterion_name

        if arti_dir == Path():
            arti_dir = arti_dir / 'efficiencies'

        if ckpt_name is None or ckpt_name in arti_dir.parts:
            return arti_dir

        return arti_dir / ckpt_name

    def get_optimized_metric(self, metric_name: str, target_rate: float):
        """Get one number that one should optimize on this callback."""
        available_metrics = list(self.eff_summary.keys())
        if not metric_name in available_metrics:
            raise ValueError(
                "Optimized metric name not in summary metrics. "
                f"Choose between {available_metrics}"
            )
        elif not target_rate in list(self.eff_summary[metric_name].keys()):
            raise ValueError(
                "Optimized metric target rate not in summary metrics. "
                f"Choose between {list(self.eff_summary[metric_name].keys())}"
            )

        metric_across_ckpts = self.eff_summary[metric_name][target_rate]
        return max(metric_across_ckpts.values())

    def plot_summary_reset(self, trainer, root_folder: Path):
        """Plot the summary metrics accummulated in eff_summary and reset this attr."""
        plot_folder = root_folder / 'plots'
        plot_folder.mkdir(parents=True, exist_ok=True)
        with open(plot_folder / 'summary.pkl', 'wb') as f:
            plain_dict = self._to_dict(self.eff_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        for metric_name in self.eff_summary.keys():
            for target_rate in self.eff_summary[metric_name].keys():
                # Get the summary metric per checkpoint.
                smet = self.eff_summary[metric_name][target_rate]
                xlabel = (
                    f"mu^2/sigma at threshold: {target_rate} kHz\n"
                    f"anomaly score: {metric_name}"
                )
                ylabel = ' '
                horizontal_bar.plot_yright(smet, smet, xlabel, ylabel, plot_folder)

        mlflow_logger = utils.mlflow.get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        arti_dir = self._resolve_arti_dir(trainer)
        IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        for img_path in sorted(plot_folder.glob('*')):
            if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
                mlflow_logger.experiment.log_artifact(
                    run_id=mlflow_logger.run_id,
                    local_path=str(img_path),
                    artifact_path=str(arti_dir),
                )

        self.eff_summary.clear()

    def _to_dict(self, d):
        if isinstance(d, defaultdict):
            return {k: self._to_dict(v) for k, v in d.items()}
        return d
