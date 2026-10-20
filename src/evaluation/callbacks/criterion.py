"""The base of the label-free criterion mirrors, the evaluation callbacks that compute a
criterion of src.callbacks.metrics.label_free over the named loaders of a test loop."""

import pickle
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class CriterionSummary(Callback):
    """Base of the criterion mirrors: per checkpoint, ``values`` maps the scores of the
    test loaders named in ``sets`` to the logged scalars, the criterion under ``key``.

    A subclass sets ``key``, ``mode`` ("min" or "max", the better direction) and
    ``sets``. Every scalar is logged with ``log_metrics_to_mlflow`` under the callback's
    name; the criterion of every checkpoint is kept in ``summary``, keyed by the data
    set name of the checkpoint. Nothing is plotted per checkpoint.

    :param output_name: Key of the per-event score in the module's outputs.
    :param name: Callback name; None takes ``key``.
    """

    key: str
    mode: str
    sets: tuple[str, ...] = ("normal",)

    def __init__(self, output_name: str = "ascore/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = self.key if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that every loader read is present."""
        missing = [name for name in self.sets if name not in trainer.test_dataloaders]
        if missing:
            raise ValueError(f"{type(self).__name__} needs the test loaders {missing}.")
        self.scores = {name: [] for name in self.sets}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        name = list(trainer.test_dataloaders)[dataloader_idx]
        if name in self.scores:
            score = torch.as_tensor(outputs[self.output_name]).detach()
            self.scores[name].append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the scalars of the current checkpoint."""
        scores = {name: torch.cat(chunks) for name, chunks in self.scores.items()}
        values = self.values(pl_module, scores)
        ckpt_name = Path(pl_module._ckpt_path).stem
        self.summary[utils.misc.get_ckpt_ds_name(ckpt_name)] = values[self.key]
        utils.mlflow.log_metrics_to_mlflow(
            trainer, values, ckpt_name=ckpt_name, cb_name=self.name
        )

    def values(self, pl_module, scores: dict[str, torch.Tensor]) -> dict[str, float]:
        raise NotImplementedError

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache the criterion of every checkpoint."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        summary = self.summary
        horizontal_bar.plot_yright(summary, summary, self.key, " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with the best criterion and its value."""
        best = max if self.mode == "max" else min
        ckpt_ds = best(self.summary, key=self.summary.get)
        return ckpt_ds, self.summary[ckpt_ds]

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)
