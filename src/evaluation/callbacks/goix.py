"""Goix's criteria as evaluation callbacks: excess mass and mass volume, the mirrors of
src/callbacks/goix.py over the named loaders of a test loop.

The volume of the uniform box comes from ``pl_module.aux_lower`` and
``pl_module.aux_upper``, the per-feature bounds of the training inputs that the
``uniform`` loader is drawn over. Each callback keeps its criterion of every checkpoint
in ``summary``, keyed by the data set name of the checkpoint, and logs it with
``log_metrics_to_mlflow``; nothing is plotted per checkpoint.
"""

import pickle
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.goix import box_volume, excess_mass, mass_volume
from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


def _volume(pl_module) -> float:
    return box_volume(pl_module.aux_lower.cpu(), pl_module.aux_upper.cpu())


class ExcessMass(Callback):
    """Evaluate the excess-mass area of ``normal`` against ``uniform``, maximised; also
    logs ``em_reached``, 1 when the area reached the level t_max.

    :param output_name: Name of the model output used as anomaly score.
    :param name: Callback name; None takes "em".
    """

    def __init__(self, output_name: str = "ascore/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = "em" if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that both required dataloaders are present and reset the buffers."""
        dset_names = list(trainer.test_dataloaders.keys())
        missing = [n for n in ("normal", "uniform") if n not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires test dataloaders {missing}. "
                f"Available test dataloaders: {dset_names}"
            )

        self.normal_scores = []
        self.uniform_scores = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the normal and uniform data sets."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name not in ("normal", "uniform"):
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        if dset_name == "normal":
            self.normal_scores.append(score.reshape(-1).cpu())
        if dset_name == "uniform":
            self.uniform_scores.append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the excess-mass area of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        normal, uniform = torch.cat(self.normal_scores), torch.cat(self.uniform_scores)
        area, reached = excess_mass(normal, uniform, _volume(pl_module))
        self._store_summary(area, utils.misc.get_ckpt_ds_name(ckpt_name))
        utils.mlflow.log_metrics_to_mlflow(
            trainer,
            {"em": area, "em_reached": float(reached)},
            ckpt_name=ckpt_name,
            cb_name=self.name,
        )

    def _store_summary(self, area: float, ckpt_ds: str):
        """Store the excess-mass area of one checkpoint."""
        self.summary[ckpt_ds] = area

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        horizontal_bar.plot_yright(self.summary, self.summary, "em", " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with maximum excess-mass area."""
        max_ckpt_ds = max(self.summary, key=self.summary.get)
        return max_ckpt_ds, self.summary[max_ckpt_ds]

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)


class MassVolume(Callback):
    """Evaluate the mass-volume area of ``normal`` against ``uniform``, minimised.

    :param output_name: Name of the model output used as anomaly score.
    :param name: Callback name; None takes "mv".
    """

    def __init__(self, output_name: str = "ascore/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = "mv" if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that both required dataloaders are present and reset the buffers."""
        dset_names = list(trainer.test_dataloaders.keys())
        missing = [n for n in ("normal", "uniform") if n not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires test dataloaders {missing}. "
                f"Available test dataloaders: {dset_names}"
            )

        self.normal_scores = []
        self.uniform_scores = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the normal and uniform data sets."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name not in ("normal", "uniform"):
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        if dset_name == "normal":
            self.normal_scores.append(score.reshape(-1).cpu())
        if dset_name == "uniform":
            self.uniform_scores.append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the mass-volume area of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        normal, uniform = torch.cat(self.normal_scores), torch.cat(self.uniform_scores)
        area = mass_volume(normal, uniform, _volume(pl_module))
        self._store_summary(area, utils.misc.get_ckpt_ds_name(ckpt_name))
        utils.mlflow.log_metrics_to_mlflow(
            trainer, {"mv": area}, ckpt_name=ckpt_name, cb_name=self.name
        )

    def _store_summary(self, area: float, ckpt_ds: str):
        """Store the mass-volume area of one checkpoint."""
        self.summary[ckpt_ds] = area

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        horizontal_bar.plot_yright(self.summary, self.summary, "mv", " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with minimum mass-volume area."""
        min_ckpt_ds = min(self.summary, key=self.summary.get)
        return min_ckpt_ds, self.summary[min_ckpt_ds]

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)
