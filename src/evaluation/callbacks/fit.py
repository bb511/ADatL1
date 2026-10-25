"""The fit Q'' of validation normal as an evaluation callback, the mirror of
``FitCallback`` in src/callbacks/operational.py over the ``normal`` loader of a test
loop.

The callback keeps the fit of every checkpoint in ``summary``, keyed by the data set
name of the checkpoint, and logs the scalars ``normal`` (the fit) and ``normal_se``
(its standard error, which the gate of the lattice step reads) with
``log_metrics_to_mlflow``; nothing is plotted per checkpoint.
"""

import math
import pickle
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class Fit(Callback):
    """Evaluate the mean of the per-event fit over the ``normal`` loader, minimised.

    The mean and its standard error s / sqrt(n) are taken in float64 over every row, as
    the training callback takes the mean, so that neither depends on a batch size.

    :param output_name: Key of the per-event fit in the module's outputs.
    :param name: Callback name; None takes "fit".
    """

    def __init__(self, output_name: str = "fit/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = "fit" if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that the normal dataloader is present and reset the buffer."""
        dset_names = list(trainer.test_dataloaders.keys())
        if "normal" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'normal'. "
                f"Available test dataloaders: {dset_names}"
            )

        self.normal_fits = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the per-event fit of the normal data set."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        fit = torch.as_tensor(outputs[self.output_name]).detach()
        self.normal_fits.append(fit.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the fit of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        fits = torch.cat(self.normal_fits).double()
        fit, se = fits.mean().item(), fits.std().item() / math.sqrt(fits.numel())
        self.summary[utils.misc.get_ckpt_ds_name(ckpt_name)] = fit
        metrics = {"normal": fit, "normal_se": se}
        utils.mlflow.log_metrics_to_mlflow(
            trainer, metrics, ckpt_name=ckpt_name, cb_name=self.name
        )

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        summary = self.summary
        horizontal_bar.plot_yright(summary, summary, "fit", " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self, ckpt_name: str):
        """Return ``ckpt_name`` and its fit, None for a checkpoint not evaluated."""
        return ckpt_name, self.summary.get(ckpt_name)

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)
