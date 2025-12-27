# Callback that computes the loss of a given checkpoint on all test data sets.
from pathlib import Path
from collections import defaultdict
import pickle

from pytorch_lightning.callbacks import Callback
from torchmetrics.aggregation import MeanMetric

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar
from src.plot import matrix


class LossCallback(Callback):
    """Evaluate the loss on the test data sets.

    :param loss_names: List of strings specifying the names of the losses that this
        callback tracks and creates plots of.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(self, loss_names: list[str], log_raw_mlflow: bool = True):
        self.device = None
        self.loss_names = loss_names
        self.log_raw_mlflow = log_raw_mlflow
        self.loss_summary = defaultdict(lambda: defaultdict(dict))

    def on_test_epoch_start(self, trainer, pl_module):
        """Initialise variable where to accummulate test losses over batches."""
        self.device = pl_module.device
        dset_names = list(trainer.test_dataloaders.keys())
        self.dset_losses = defaultdict(lambda: defaultdict(float))
        for loss_name in self.loss_names:
            for dset_name in dset_names:
                self.dset_losses[loss_name][dset_name] = MeanMetric().to(self.device)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Accummulate the mean loss of a batch, over batches.

        CAREFUL: this expects the batch_output to be a torch tensor of dimension 1,ie,
        the mean loss over that batch.
        """
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        for loss_name in self.loss_names:
            batch_output = outputs[loss_name]
            self.dset_losses[loss_name][dset_name].update(batch_output)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        plot_folder = ckpts_dir / "plots" / ckpt_name / "losses"
        plot_folder.mkdir(parents=True, exist_ok=True)

        for loss_name in self.loss_names:
            losses = {
                dset_name: loss.compute().item()
                for dset_name, loss in self.dset_losses[loss_name].items()
            }
            xlabel = f"{loss_name}"
            ylabel = " "
            horizontal_bar.plot_yright(losses, losses, xlabel, ylabel, plot_folder)
            self._store_summary(loss_name, losses, ckpt_name)

        utils.mlflow.log_plots_to_mlflow(
            trainer, ckpt_name, "losses", plot_folder, log_raw=self.log_raw_mlflow, gallery_name='losses'
        )

    def _store_summary(self, loss_name: str, losses: dict, ckpt_name: str):
        """Store a summary statistic for the losses of one checkpoint.

        Here, we store losses computed on all the test data sets. At the end, this is
        plotted as a matrix.
        """
        ckpt_ds_name = utils.misc.get_ckpt_ds_name(ckpt_name)
        self.loss_summary[loss_name][ckpt_ds_name] = losses

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the summary metrics accummulated in eff_summary and reset this attr."""
        plot_folder = root_folder / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        for loss_name in self.loss_summary.keys():
            # Get the summary metric per checkpoint.
            losses_per_ckpt = self.loss_summary[loss_name]
            matrix.plot(losses_per_ckpt, loss_name, plot_folder)

        utils.mlflow.log_plots_to_mlflow(trainer, None, "losses", plot_folder)

    def clear_crit_summary(self):
        self.loss_summary.clear()

    def get_optimized_metric(self, loss_name: str, ckpt_ds: str, test_ds: str):
        """Get one number that one should optimize on this callback.

        Here it is the value of the given loss_name that a model checkpointed on a given
        checkponint dataset and evaluated on a given test data set.
        """
        available_losses = list(self.loss_summary.keys())
        if not loss_name in available_losses:
            raise ValueError(f"Choose {available_losses}")

        optimized_loss = self.loss_summary[loss_name][ckpt_ds][test_ds]
        return optimized_loss

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.loss_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
