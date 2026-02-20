# Compute the approximation capacity metric.
from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np

from pytorch_lightning.callbacks import Callback

from capmetric.metric import ApproximationCapacity
from capmetric.binary import get_pairing_fn

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class CAP(Callback):
    """Evaluate the approximation capacity on the selected test data sets.

    :param metric_name: String specifying the model metric to use as an anomaly score.
    :param ds: List of strings with the names of the datasets to compute this on.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(
        self,
        metric_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
        cap_metric_config: dict,
        log_raw_mlflow: bool = True,
    ):
        super().__init__()
        self.device = None
        self.metric_name = metric_name
        self.dataset_1_name = dataset_1
        self.dataset_2_name = dataset_2
        self.cap_metric_config = cap_metric_config
        self.pairing_fn = get_pairing_fn(pairing_type)
        self.log_raw_mlflow = log_raw_mlflow
        self.cap_summary = defaultdict(float)

    def on_test_start(self, trainer, pl_module):
        """Set the right device at start of testing."""
        self.device = pl_module.device

    def on_test_epoch_start(self, trainer, pl_module):
        """Initialise useful quantities."""
        self.dataset_1_scores = []
        self.dataset_2_scores = []
        self.capmetric = ApproximationCapacity(**self.cap_metric_config)
        self.capmetric.to(self.device)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the data to compute approximation capacity on.

        The two data sets need to be paired and going through two data sets at once in not
        compatible with the lightning workflow, and hence we cache the whole data sets here.
        """
        self.dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if self.dataset_name == self.dataset_1_name:
            self.dataset_1_scores.append(outputs[self.metric_name])
        if self.dataset_name == self.dataset_2_name:
            self.dataset_2_scores.append(outputs[self.metric_name])

    def _compute_cap(self):
        """Compute the cap metric between the two given data sets."""
        self.dataset_1_scores = torch.cat(self.dataset_1_scores, dim=0)
        self.dataset_2_scores = torch.cat(self.dataset_2_scores, dim=0)
        idxs1, idxs2 = self.pairing_fn(self.dataset_1_scores, self.dataset_2_scores)
        ds1_scores = self.dataset_1_scores[idxs1]
        ds2_scores = self.dataset_2_scores[idxs2]

        n = min(len(ds1_scores), len(ds2_scores))

        with torch.inference_mode(False):
            with torch.enable_grad():
                # If capmetric needs gradients w.r.t. these tensors, they must require grad
                ds1 = ds1_scores[:n].detach().clone().requires_grad_(True)
                ds2 = ds2_scores[:n].detach().clone().requires_grad_(True)
                self.capmetric.update(ds1, ds2)

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute the CAP metric on the designated dataset."""
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        self._compute_cap()

        cap_metric_value = self.capmetric.compute()
        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
        self._store_summary(cap_metric_value, ckpt_ds)

    def _store_summary(self, cap_metric_value: float, ckpt_ds: str):
        """Store the summary statistic for the cap for one checkpoint.

        Here, it is the metric itself, but this is implemented to be consistent with
        the other evaluation callbacks.
        """
        self.cap_summary[ckpt_ds] = abs(cap_metric_value)

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        """Plot the efficiency per data set for an anomaly metric at target rate."""
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the summary metrics accummulated in eff_summary and reset this attr."""
        plot_folder = root_folder / "plots" / "cap_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        # Configure plot.
        xlabel = f"CAP({self.dataset_1_name},\n{self.dataset_2_name})"
        self._plot(self.cap_summary, xlabel, plot_folder)

        utils.mlflow.log_plots_to_mlflow(trainer, None, "cap", plot_folder)

    def get_optimized_metric(self):
        """Get one number that one should optimize on this callback.

        Here, it's the maximum of the summary metric across checkpoints corresponding
        to a certain checkpointing criterion.
        """
        max_ckpt_ds = max(self.cap_summary, key=self.cap_summary.get)
        max_metric_value = self.cap_summary[max_ckpt_ds]
        return max_ckpt_ds, max_metric_value

    def clear_crit_summary(self):
        self.cap_summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.cap_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
