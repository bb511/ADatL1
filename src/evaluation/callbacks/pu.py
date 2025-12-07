# Plots the correlation of the anomaly score with the pileup.

from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger, Logger
from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from src.evaluation.callbacks.metrics.rate import AnomalyCounter
from src.evaluation.callbacks import utils
from src.data.components.normalization import L1DataNormalizer
from src.plot import horizontal_bar


class BkgRatePileup(Callback):
    """Plots the correlation between the background data sets rate and the pileup.

    :target_rate: Float specifying the target rate of anomalies.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total efficiency of events that are processed by
        the L1 trigger.
    :datamodule: The datamodule that was used during the training of the model.
    :materic_names: Which model metrics to use as the anomaly score to calculate the
        rate on.
    """

    def __init__(
        self,
        target_rates: list[int],
        bc_rate: int,
        metric_names: list[str],
        datamodule: LightningDataModule
    ):
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.datamodule = datamodule
        self.metric_names = metric_names

        self.rates = {}
        self.maintest_score_data = defaultdict(list)

    def on_test_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        # Check if 'main_test' dataset is the first dataset in the test dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        if first_test_dset_key != "main_test":
            raise ValueError("Eff callback requires main_test first in the data dict!")

        self.put_per_dataset = self._get_pileup_data()

    def _get_pileup_data(self):
        """Get the pileup data for each test data set.

        # Get the nPV_True feature of the event_info object, put it in the subfolder of
        # the data in 'test_plots_data' and don't normalize it.
        """
        extra_feats = {'event_info': ['nPV_True']}
        flag = 'test_plots_data'
        normalizer = L1DataNormalizer('unnormalized', {})

        _, _, test_data = self.datamodule.get_extra(normalizer, extra_feats, flag)

        print(test_data)
        exit(1)
        return test_data

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        for mname in self.maintest_score_data.keys():
            self.maintest_score_data[mname] = []

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
        x, y = batch

        for metric_name in self.metric_names:
            if self.dataset_name == "main_test":
                self._accumulate_maintest_output(outputs, batch_idx, metric_name)
            elif (y == -1).all().item():
                if batch_idx == 0:
                    self._initialize_rate_metric(batch_idx, metric_name)
                self._compute_batch_rate(outputs, metric_name)
            else:
                continue

    def _accumulate_maintest_output(self, outputs: dict, batch_idx: int, mname: str):
        """Accummulates the specified metric data across batches.

        Used if currently processing the main_test data set, accummulate the values of
        each metric across batches. The whole metric output distribution is needed to
        set a treshold that gives a certain rate.
        """
        batch_output = outputs[mname].detach().to('cpu')
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
            rate_name = f"{mname.replace('/', '_')}_rate{target_rate}"
            self.rates.update({f"{self.dataset_name}/{rate_name}": rate})

    def _compute_batch_rate(self, outputs: dict, mname: str):
        """Done after knowing the rate thresholds.

        For all the other test data sets, except the main one, this calculates
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
        plot_folder = ckpts_dir / 'plots' / ckpt_name / 'bkgpileup'
        plot_folder.mkdir(parents=True, exist_ok=True)

        for metric_name in self.metric_names:
            for target_rate in self.target_rates:
                # Get the rates per dataset for a specific target rate
                # and for a specific metric_name used as the anomaly score.
                name = f"{metric_name.replace('/', '_')}_rate{target_rate}"
                rates_per_dataset = {
                    self._get_dsname(rate_name): round(val.compute('efficiency').item(), 4)
                    for rate_name, val in self.rates.items()
                    if name in rate_name
                }

                xlabel = (
                    f"efficiency at threshold: {target_rate} kHz\n"
                    f"anomaly score: {metric_name}"
                )
                ylabel = ' '
                horizontal_bar.plot_yright(
                    rates_per_dataset, rates_per_dataset, xlabel, ylabel, plot_folder
                )

            self._log_plots_to_mlflow(trainer, ckpt_name, plot_folder)

    def _get_dsname(self, rate_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = rate_name.split('/')[0]
        return dataset_name

    def _log_plots_to_mlflow(self, trainer, ckpt_name: str, plot_folder: Path):
        """Logs the plots generated by this callback to MLFlow."""
        mlflow_logger = utils.mlflow.get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        run_id = mlflow_logger.run_id
        arti_ckpt_dir = self._resolve_arti_dir(trainer, ckpt_name)

        # Log each image in the given plot_folder as an artifact.
        img_paths = sorted(plot_folder.glob('*.jpg'))
        for img_path in img_paths:
            mlflow_logger.experiment.log_artifact(
                run_id=run_id,
                local_path=str(img_path),
                artifact_path=str(arti_ckpt_dir),
            )

        # Generate an html gallery at the parent directory level.
        gallery_dir = arti_ckpt_dir.parent
        html_gallery = utils.mlflow.build_html(
            mlflow_logger, plot_folder, gallery_dir, arti_ckpt_dir
        )
        mlflow_logger.experiment.log_text(
            run_id, html_gallery, artifact_file=gallery_dir / 'index.html'
        )

    def _resolve_arti_dir(self, trainer, ckpt_name: str):
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

        if ckpt_name in arti_dir.parts:
            return arti_dir

        return arti_dir / ckpt_name
