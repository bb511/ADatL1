"""AutoUAD's criteria as evaluation callbacks: NPD, RTM and EAG, the mirrors of
src/callbacks/autouad.py over the named loaders of a test loop.

Each callback keeps its criterion of every checkpoint in ``summary``, keyed by the data
set name of the checkpoint, and logs it with ``log_metrics_to_mlflow``; nothing is
plotted per checkpoint.
"""

import pickle
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.autouad import eag, npd, rtm
from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class NPD(Callback):
    """Evaluate the normalised pseudo discrepancy of ``normal`` against ``autouad``,
    maximised.

    :param output_name: Name of the model output used as anomaly score.
    :param name: Callback name; None takes "npd".
    """

    def __init__(self, output_name: str = "ascore/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = "npd" if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that both required dataloaders are present and reset the buffers."""
        dset_names = list(trainer.test_dataloaders.keys())
        missing = [n for n in ("normal", "autouad") if n not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires test dataloaders {missing}. "
                f"Available test dataloaders: {dset_names}"
            )

        self.normal_scores = []
        self.autouad_scores = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the normal and autouad data sets."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name not in ("normal", "autouad"):
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        if dset_name == "normal":
            self.normal_scores.append(score.reshape(-1).cpu())
        if dset_name == "autouad":
            self.autouad_scores.append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the NPD of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        normal, autouad = torch.cat(self.normal_scores), torch.cat(self.autouad_scores)
        npd_value = npd(normal, autouad)
        self._store_summary(npd_value, utils.misc.get_ckpt_ds_name(ckpt_name))
        utils.mlflow.log_metrics_to_mlflow(
            trainer, {"npd": npd_value}, ckpt_name=ckpt_name, cb_name=self.name
        )

    def _store_summary(self, npd_value: float, ckpt_ds: str):
        """Store the NPD of one checkpoint."""
        self.summary[ckpt_ds] = npd_value

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        horizontal_bar.plot_yright(self.summary, self.summary, "npd", " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with maximum NPD."""
        max_ckpt_ds = max(self.summary, key=self.summary.get)
        return max_ckpt_ds, self.summary[max_ckpt_ds]

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)


class RTM(Callback):
    """Evaluate the relative top-median of ``normal``, maximised.

    :param output_name: Name of the model output used as anomaly score.
    :param name: Callback name; None takes "rtm".
    """

    def __init__(self, output_name: str = "ascore/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = "rtm" if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that the normal dataloader is present and reset the buffer."""
        dset_names = list(trainer.test_dataloaders.keys())
        if "normal" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'normal'. "
                f"Available test dataloaders: {dset_names}"
            )

        self.normal_scores = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the normal data set."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        self.normal_scores.append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the RTM of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        rtm_value = rtm(torch.cat(self.normal_scores))
        self._store_summary(rtm_value, utils.misc.get_ckpt_ds_name(ckpt_name))
        utils.mlflow.log_metrics_to_mlflow(
            trainer, {"rtm": rtm_value}, ckpt_name=ckpt_name, cb_name=self.name
        )

    def _store_summary(self, rtm_value: float, ckpt_ds: str):
        """Store the RTM of one checkpoint."""
        self.summary[ckpt_ds] = rtm_value

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        horizontal_bar.plot_yright(self.summary, self.summary, "rtm", " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with maximum RTM."""
        max_ckpt_ds = max(self.summary, key=self.summary.get)
        return max_ckpt_ds, self.summary[max_ckpt_ds]

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)


class EAG(Callback):
    """Evaluate the expected anomaly gap of ``normal``, maximised.

    :param output_name: Name of the model output used as anomaly score.
    :param name: Callback name; None takes "eag".
    """

    def __init__(self, output_name: str = "ascore/full", name: str | None = None):
        super().__init__()
        self.output_name = output_name
        self.name = "eag" if name is None else name
        self.summary = {}

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that the normal dataloader is present and reset the buffer."""
        dset_names = list(trainer.test_dataloaders.keys())
        if "normal" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'normal'. "
                f"Available test dataloaders: {dset_names}"
            )

        self.normal_scores = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the normal data set."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        score = torch.as_tensor(outputs[self.output_name]).detach()
        self.normal_scores.append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute, store and log the EAG of the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        eag_value = eag(torch.cat(self.normal_scores))
        self._store_summary(eag_value, utils.misc.get_ckpt_ds_name(ckpt_name))
        utils.mlflow.log_metrics_to_mlflow(
            trainer, {"eag": eag_value}, ckpt_name=ckpt_name, cb_name=self.name
        )

    def _store_summary(self, eag_value: float, ckpt_ds: str):
        """Store the EAG of one checkpoint."""
        self.summary[ckpt_ds] = eag_value

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)
        horizontal_bar.plot_yright(self.summary, self.summary, "eag", " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with maximum EAG."""
        max_ckpt_ds = max(self.summary, key=self.summary.get)
        return max_ckpt_ds, self.summary[max_ckpt_ds]

    def clear_crit_summary(self):
        self.summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            pickle.dump(dict(self.summary), f, protocol=pickle.HIGHEST_PROTOCOL)
