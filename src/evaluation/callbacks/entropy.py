"""The loss entropy of EntropyStop as an evaluation callback, the mirror of
src/callbacks/entropy.py over the ``train`` loader of a test loop.

The callback keeps the entropy of every checkpoint in ``summary``, keyed by the data
set name of the checkpoint, and logs it with ``log_metrics_to_mlflow``; nothing is
plotted per checkpoint.
"""

import pickle
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.entropy import loss_entropy
from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class LossEntropy(Callback):
    """Evaluate the entropy of the per-sample loss of the ``train`` loader, minimised;
    NaN, with a warning, when a loss is not positive.

    :param output_name: Key of the per-sample loss in the module's outputs.
    :param name: Callback name; None takes "entropy".
    """

    def __init__(self, output_name: str = "loss/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = "entropy" if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that the train dataloader is present and reset the buffer."""
        dset_names = list(trainer.test_dataloaders.keys())
        if "train" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'train'. "
                f"Available test dataloaders: {dset_names}"
            )

        self.train_losses = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the per-sample losses of the train data set."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name != "train":
            return

        loss = torch.as_tensor(outputs[self.output_name]).detach()
        self.train_losses.append(loss.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the loss entropy of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        entropy = loss_entropy(torch.cat(self.train_losses))
        self._store_summary(entropy, utils.misc.get_ckpt_ds_name(ckpt_name))
        utils.mlflow.log_metrics_to_mlflow(
            trainer, {"entropy": entropy}, ckpt_name=ckpt_name, cb_name=self.name
        )

    def _store_summary(self, entropy: float, ckpt_ds: str):
        """Store the loss entropy of one checkpoint."""
        self.summary[ckpt_ds] = entropy

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        summary = self.summary
        horizontal_bar.plot_yright(summary, summary, "entropy", " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with minimum loss entropy."""
        min_ckpt_ds = min(self.summary, key=self.summary.get)
        return min_ckpt_ds, self.summary[min_ckpt_ds]

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)
