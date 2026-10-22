"""The SDS error as an evaluation callback, the mirror of src/callbacks/sds.py over the
``sds_targets`` and ``sds_outliers`` loaders of a test loop.

Without a target rate, the cut is ``pl_module.thres_operational`` with
``pl_module.tie_operational``, the threshold and tie weight set on the epoch's
validation normal scores at the module's target rate. With one, the cut is set at that
rate on the ``normal`` scores of the same test loop, which must then hold the
validation normal set. The callback keeps the error of every checkpoint in ``summary``,
keyed by the data set name of the checkpoint, and logs it with
``log_metrics_to_mlflow``; nothing is plotted per checkpoint.
"""

import pickle
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.sds import rate_cut, sds_error, tiered_name
from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class SDS(Callback):
    """Evaluate the SDS error of the pseudo-sets at the cut of a rate, minimised.

    :param output_name: Name of the model output used as anomaly score.
    :param name: Callback name; None takes "sds". With ``target_rate`` set, the name
        carries its rate token, as ``tiered_name`` forms it.
    :param target_rate: Rate of the cut, in the unit of the module's target rate (kHz
        on the LHC); None takes the module's operational cut.
    """

    def __init__(
        self,
        output_name: str = "ascore/full",
        name: str | None = None,
        target_rate: float | None = None,
    ):
        super().__init__()
        self.output_name = output_name
        self.name = tiered_name("sds" if name is None else name, target_rate)
        self.target_rate = target_rate
        self.sets = ("sds_targets", "sds_outliers")
        self.sets += () if target_rate is None else ("normal",)
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that the required dataloaders are present and reset the buffers."""
        dset_names = list(trainer.test_dataloaders.keys())
        missing = [n for n in self.sets if n not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires test dataloaders {missing}. "
                f"Available test dataloaders: {dset_names}"
            )

        self.scores = {name: [] for name in self.sets}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the pseudo-targets, the pseudo-outliers and, for a cut at
        a target rate, the normal set."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name not in self.sets:
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        self.scores[dset_name].append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the SDS error of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        threshold, tie_weight = self._cut(pl_module)
        a_targets = torch.cat(self.scores["sds_targets"])
        a_outliers = torch.cat(self.scores["sds_outliers"])
        sds_value = sds_error(a_targets, a_outliers, threshold, tie_weight)
        self._store_summary(sds_value, utils.misc.get_ckpt_ds_name(ckpt_name))
        utils.mlflow.log_metrics_to_mlflow(
            trainer, {"sds": sds_value}, ckpt_name=ckpt_name, cb_name=self.name
        )

    def _cut(self, pl_module) -> tuple[float, float]:
        """Threshold and tie weight of the cut."""
        if self.target_rate is None:
            threshold = torch.as_tensor(pl_module.thres_operational).item()
            return threshold, torch.as_tensor(pl_module.tie_operational).item()
        return rate_cut(pl_module, torch.cat(self.scores["normal"]), self.target_rate)

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
