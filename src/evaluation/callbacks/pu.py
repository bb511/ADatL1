# Plots the correlation of the anomaly rate with the pileup.

from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger, Logger
from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from src.evaluation.callbacks.metrics.rate import AnomalyCounter
from src.evaluation.callbacks import utils
from src.data.components.normalization import L1DataNormalizer
from src.plot import scatter


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
        datamodule: LightningDataModule,
    ):
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.metric_names = metric_names
        self.pu_per_ds = self._get_pileup_data(datamodule)

    def on_test_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        # Check if 'main_test' dataset is the first dataset in the test dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        if first_test_dset_key != "main_test":
            raise ValueError("PU callback requires main_test first in the data dict!")

    def _get_pileup_data(self, datamodule: LightningDataModule):
        """Get the pileup data for each test data set.

        Get the nPV_True feature of the event_info object, put it in the subfolder of
        the data in 'callbacks_pileup' and don't normalize it.
        """
        extra_feats = {'event_info': ['nPV_True']}
        flag = 'callbacks_pileup'
        normalizer = L1DataNormalizer('unnormalized', {})

        _, _, test_data = datamodule.get_extra(normalizer, extra_feats, flag)

        return test_data

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.rates_per_pu = defaultdict(dict)
        self.maintest_score_data = defaultdict(list)
        self.maintest_pu_data = defaultdict(list)

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

        for metric_name in self.metric_names:
            if self.dataset_name == "main_test":
                self._accumulate_maintest_output(outputs, batch_idx, metric_name)
            elif (labels < 0).all().item():
                self._compute_batch_rate(outputs, batch_idx, metric_name)
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

        pu, _ = self.pu_per_ds['main_test'][batch_idx]
        pu = pu.round().int().to('cpu').view(-1)
        self.maintest_pu_data[mname].append(pu)

        if batch_idx == self.total_batches - 1:
            self.maintest_score_data[mname] = torch.cat(
                self.maintest_score_data[mname], dim=0
            )
            self.maintest_pu_data[mname] = torch.cat(
                self.maintest_pu_data[mname], dim=0
            )
            self._compute_maintest_rate(mname)

    def _compute_maintest_rate(self, mname: str):
        """Computes the desired rates on the main test data set.

        This is a sanity check. The threshold computed on the maintest and applied to the
        maintest data should return the rate for which this threshold was computed.
        """
        unique_pu, inv = self.maintest_pu_data[mname].unique(
            sorted=True, return_inverse=True
        )
        scores_by_pu = {
            int(pu.item()): self.maintest_score_data[mname][inv == i]
            for i, pu in enumerate(unique_pu)
        }
        for target_rate in self.target_rates:
            rate = AnomalyCounter(target_rate, self.bc_rate)
            rate.set_threshold(self.maintest_score_data[mname])
            rate_name = f"{mname.replace('/', '_')}_rate{target_rate}"
            rate_name = f"{self.dataset_name}/{rate_name}"
            for pu, scores in scores_by_pu.items():
                rate.update(scores)
                self.rates_per_pu[rate_name][pu] = rate

    def _compute_batch_rate(self, outputs: dict, batch_idx: int, mname: str):
        """Done after knowing the rate thresholds.

        For all the other test data sets, except the main one, this calculates
        which events pass the rate threshold for each batch and updates the total.
        """
        pus, _ = self.pu_per_ds[self.dataset_name][batch_idx]
        pus = pus.round().int().to('cpu').view(-1)
        pus, inv = pus.unique(sorted=True, return_inverse=True)

        scores_by_pu = {
            int(pu.item()): outputs[mname].detach().to('cpu')[inv == i]
            for i, pu in enumerate(pus)
        }
        for target_rate in self.target_rates:
            rate_name = f"{mname.replace('/', '_')}_rate{target_rate}"
            rate_name = f"{self.dataset_name}/{rate_name}"
            for pu, scores in scores_by_pu.items():
                if not pu in self.rates_per_pu[rate_name].keys():
                    self._initialize_rate_metric(mname, target_rate, pu)
                self.rates_per_pu[rate_name][pu].update(scores)

    def _initialize_rate_metric(self, mname: str, target_rate: float, pu: int):
        """Initialize the rate metric for a certain target rate and pu.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the main_test metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from main_test.
        """
        rate = AnomalyCounter(target_rate, self.bc_rate)
        rate.set_threshold(self.maintest_score_data[mname])
        rate_name = f"{mname.replace('/', '_')}_rate{target_rate}"
        rate_name = f"{self.dataset_name}/{rate_name}"
        self.rates_per_pu[rate_name][pu] = rate

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / 'plots' / ckpt_name / 'bkgpileup'
        plot_folder.mkdir(parents=True, exist_ok=True)

        for rate_name, pu_rates in self.rates_per_pu.items():
            target_rate = rate_name.split('_')[-1].replace('rate', '')
            dataset_name = self._get_dsname(rate_name)
            pu_rates = self._compute_rates(pu_rates)
            xlabel = f"pileup"
            ylabel = f"rate (kHz) @ {target_rate} kHz"
            scatter.plot_connected(pu_rates, xlabel, ylabel, dataset_name, plot_folder)

        self._log_plots_to_mlflow(trainer, ckpt_name, plot_folder)

    def _get_dsname(self, rate_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = rate_name.split('/')[0]
        return dataset_name

    def _compute_rates(self, pu_rates: dict):
        """Compute the rates in the pileup_rates dictionary."""
        for pu_value, anomaly_counter in pu_rates.items():
            pu_rates[pu_value] = anomaly_counter.compute('rate').item()

        return pu_rates

    def _log_plots_to_mlflow(self, trainer, ckpt_name: str, plot_folder: Path):
        """Logs the plots generated by this callback to MLFlow."""
        mlflow_logger = utils.mlflow.get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        run_id = mlflow_logger.run_id
        arti_ckpt_dir = self._resolve_arti_dir(trainer, ckpt_name)

        # Log each image in the given plot_folder as an artifact.
        for img_path in sorted(
            p for p in plot_folder.glob("*.jpg")
            if p.is_file() and not p.name.startswith("._")
        ):
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
            arti_dir = arti_dir / trainer.strat_name / 'pileup'
        if trainer.metric_name:
            arti_dir = arti_dir / trainer.metric_name
        if trainer.criterion_name:
            arti_dir = arti_dir / trainer.criterion_name

        if arti_dir == Path():
            arti_dir = arti_dir / 'pileup'

        if ckpt_name in arti_dir.parts:
            return arti_dir

        return arti_dir / ckpt_name
