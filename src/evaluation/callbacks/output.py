# Callback that plots histograms of the model output.
from pathlib import Path
from collections import defaultdict
import pickle

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics.aggregation import MeanMetric

from src.evaluation.callbacks import utils
from src.plot.histogram import plot_streamed
from src.plot import horizontal_bar
from src.plot import matrix
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class OutputCallback(Callback):
    """Evaluate the output on the test data sets. Plot the result in a histogram.

    :param output_name: String specifying which output value, returned in the model
        outdict should be used in this callback.
    :param ds: List of strings specifying which datasets to get the output on.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    :param name: String specifying the name of the callback for identification in
        later methods that manipulate callbacks.
    """

    def __init__(
        self,
        output_name: str,
        ds: list[str],
        log_raw_mlflow: bool = True,
        name: str = "loss",
    ):
        self.device = None
        self.output_name = output_name
        self.ds = ds
        self.log_raw_mlflow = log_raw_mlflow
        self.name = name
        self.loss_summary = defaultdict(dict)

    def on_test_epoch_start(self, trainer, pl_module):
        """Initialise variable where to accummulate test losses over batches."""
        self.device = pl_module.device
        dset_names = list(trainer.test_dataloaders.keys())
        self.dset_losses = defaultdict(float)
        for dset_name in dset_names:
            self.dset_losses[dset_name] = MeanMetric().to(self.device)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Accummulate the mean loss of a batch, over batches.

        CAREFUL: this expects the batch_output to be a torch tensor of dimension 1,ie,
        the mean loss over that batch.
        """
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        batch_output = outputs[self.output_name]

        self.dset_losses[dset_name].update(batch_output)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        split = trainer.split
        plot_folder = ckpts_dir / "plots" / split / ckpt_name / self.name
        plot_folder.mkdir(parents=True, exist_ok=True)

        losses = {
            dset_name: loss.compute().item()
            for dset_name, loss in self.dset_losses.items()
            if dset_name in self.ds
        }
        xlabel = f"{self.output_name}"
        ylabel = " "
        horizontal_bar.plot_yright(losses, losses, xlabel, ylabel, plot_folder)
        self._store_summary(losses, ckpt_name)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            self.name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{self.name}",
        )

    def _store_summary(self, losses: dict, ckpt_name: str):
        """Store a summary statistic for the losses of one checkpoint.

        Here, we store losses computed on all the test data sets. At the end, this is
        plotted as a matrix.
        """
        ckpt_ds_name = utils.misc.get_ckpt_ds_name(ckpt_name)
        self.loss_summary[ckpt_ds_name] = losses

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the summary metrics accummulated in eff_summary and reset this attr."""
        split = trainer.split
        plot_folder = root_folder / "plots" / split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        matrix.plot(self.loss_summary, self.output_name.replace("/", "_"), plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def clear_crit_summary(self):
        self.loss_summary.clear()

    def get_optimized_metric(self, ckpt_name: str, test_ds: str):
        """Get one number that one should optimize on this callback.

        Here it is the value of the self.output_name at the given ckpt_name which is
        evaluated at the give test_ds.
        If the loss_summary dictionary only has 'last' as a key, then the criterion is
        'last', i.e., only checkpoint on the last epoch. This is trated as a special
        case and the ckpt_name that is given to this method is returned, along with the
        loss in the last epoch.
        """
        if not ckpt_name in self.loss_summary.keys():
            log.warn(
                f"Checkpoint with name '{ckpt_name}' not found for this strategy. "
                f"Available ckpt names: {list(self.loss_summary.keys())}"
            )
            return ckpt_name, None

        optimized_loss = self.loss_summary[ckpt_name][test_ds]
        return ckpt_name, optimized_loss

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.loss_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


class HistogramOutputCallback(Callback):
    """Stream model outputs into histograms with automatic bin detection.

    :param output_name: String specifying the key in the output dictionary of the
        output you'd like to plot.
    :param ds: List of strings specifying the datasets that you'd like to make the
        histogram for.
    :param ckpts: Dictionary of the checkpoints that you'd like to plot the histograms
        for.
    :param bins: Int with the number of bins that the histogram should have.
    :param warmup_batches: Int specifying the number of batches used to determine the
        range of the histogram. If this is a float, it should be a float between 0 and
        1, since then this is interpreted as the fraction of batches that should be used
        for warmup.
    :param name: String specifying the name of the callback and the name of the gallery.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(
        self,
        output_name: str,
        ds: list[str],
        ckpts: dict,
        bins: int = 10,
        warmup_batches: int | float = 0.2,
        log: bool = False,
        name: str = "outputs_hist",
        log_raw_mlflow: bool = True,
    ):
        self.output_name = output_name
        self.ds = ds
        self.ckpts = ckpts
        self.bins = bins
        self.warmup_batches = warmup_batches
        self.name = name
        self.log = log
        self.log_raw_mlflow = log_raw_mlflow
        self._active = False

    def on_test_epoch_start(self, trainer, pl_module):
        """Determine if this callback should be ran on the current checkpoint.

        If yes, initialise required quantities, and then determine the number of warmup
        batches to use when deducing the range of the histograms.
        """
        self._active = self._should_run_for_current_ckpt(trainer)
        if not self._active:
            return

        self.buffer = {d: [] for d in self.ds}
        self.hist = {d: None for d in self.ds}
        self.edges = {d: None for d in self.ds}
        self.batch_count = {d: 0 for d in self.ds}

        if isinstance(self.warmup_batches, float):
            self._warmup_batches = {}

            for name, loader in trainer.test_dataloaders.items():
                if name not in self.ds:
                    continue

                total_batches = len(loader)
                self._warmup_batches[name] = max(
                    1, int(self.warmup_batches * total_batches)
                )
        else:
            self._warmup_batches = {d: self.warmup_batches for d in self.ds}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Stream the selected output to a histogram."""
        if not self._active:
            return

        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name not in self.ds:
            return

        x = outputs[self.output_name]
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        # warmup phase
        if self.edges[dset_name] is None:
            S = 2048
            if x.size > S:
                x = np.random.choice(x, size=S, replace=False)
            self.buffer[dset_name].append(x)
            self.batch_count[dset_name] += 1

            if self.batch_count[dset_name] >= self.warmup_batches:
                vals = np.concatenate(self.buffer[dset_name])
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    return

                edges = np.histogram_bin_edges(vals, bins=self.bins)

                self.edges[dset_name] = edges
                counts, _ = np.histogram(vals, bins=edges)
                self.hist[dset_name] = counts.astype(np.float64)

                self.buffer[dset_name] = []
        else:
            counts, _ = np.histogram(x, bins=self.edges[dset_name])
            self.hist[dset_name] += counts

    def on_test_epoch_end(self, trainer, pl_module):
        """Plot the accummulated histogram for each relevant data set."""
        if not self._active:
            return

        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem

        split = trainer.split
        plot_folder = ckpts_dir / "plots" / split / ckpt_name / self.name
        plot_folder.mkdir(parents=True, exist_ok=True)

        for dset_name in self.ds:
            if self.hist[dset_name] is None:
                continue

            plot_streamed(
                counts=self.hist[dset_name],
                edges=self.edges[dset_name],
                obj_name=dset_name,
                feat_name=self.output_name,
                save_dir=plot_folder,
                log=self.log,
            )

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            self.name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{self.name}",
        )

    def _should_run_for_current_ckpt(self, trainer) -> bool:
        """Determine whether this callback should run for the current ckpt."""
        strat = getattr(trainer, "strat_name", None)
        metric = getattr(trainer, "metric_name", None)
        crit = getattr(trainer, "criterion_name", None)

        if strat is None:
            return False

        if strat == "last":
            return bool(self.ckpts.get("last", False))

        strat_cfg = self.ckpts.get(strat, None)
        if not isinstance(strat_cfg, dict) or metric is None or crit is None:
            return False

        allowed_criteria = strat_cfg.get(metric, None)
        if not isinstance(allowed_criteria, (list, tuple)):
            return False

        return crit in allowed_criteria
