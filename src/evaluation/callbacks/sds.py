"""The SDS error as an evaluation callback, the mirror of src/callbacks/sds.py over the
``sds_targets`` and ``sds_outliers`` loaders of a test loop.

The cut is ``pl_module.thres_operational`` with ``pl_module.tie_operational``, the
threshold and tie weight set on the epoch's validation normal scores at the module's
target rate. The callback keeps the error of every checkpoint in ``summary``, keyed by
the data set name of the checkpoint, and logs it with ``log_metrics_to_mlflow``;
nothing is plotted per checkpoint.
"""

import pickle
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.sds import sds_error
from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class SDS(Callback):
    """Evaluate the SDS error of the pseudo-sets at the operational cut, minimised.

    :param output_name: Name of the model output used as anomaly score.
    :param name: Callback name; None takes "sds".
    """

    def __init__(self, output_name: str = "ascore/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = "sds" if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that both required dataloaders are present and reset the buffers."""
        dset_names = list(trainer.test_dataloaders.keys())
        missing = [n for n in ("sds_targets", "sds_outliers") if n not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires test dataloaders {missing}. "
                f"Available test dataloaders: {dset_names}"
            )

        self.target_scores = []
        self.outlier_scores = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the pseudo-targets and the pseudo-outliers."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name not in ("sds_targets", "sds_outliers"):
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        if dset_name == "sds_targets":
            self.target_scores.append(score.reshape(-1).cpu())
        if dset_name == "sds_outliers":
            self.outlier_scores.append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the SDS error of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        threshold = torch.as_tensor(pl_module.thres_operational).item()
        tie_weight = torch.as_tensor(pl_module.tie_operational).item()
        a_targets = torch.cat(self.target_scores)
        a_outliers = torch.cat(self.outlier_scores)
        sds_value = sds_error(a_targets, a_outliers, threshold, tie_weight)
        self._store_summary(sds_value, utils.misc.get_ckpt_ds_name(ckpt_name))
        utils.mlflow.log_metrics_to_mlflow(
            trainer, {"sds": sds_value}, ckpt_name=ckpt_name, cb_name=self.name
        )

    def _store_summary(self, sds_value: float, ckpt_ds: str):
        """Store the SDS error of one checkpoint."""
        self.summary[ckpt_ds] = sds_value

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        horizontal_bar.plot_yright(self.summary, self.summary, "sds", " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with minimum SDS error."""
        min_ckpt_ds = min(self.summary, key=self.summary.get)
        return min_ckpt_ds, self.summary[min_ckpt_ds]

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)
