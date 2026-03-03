# Callback that compute the threshold drift between validation and test.
import math
import pickle
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import torch
from pytorch_lightning import Callback

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class ThresholdDriftCallback(Callback):
    """Compute calibration-transfer metric using val-calibrated thresholds stored as buffers.

    Discovers thresholds on the module with names:
        thres_{target_rate}kHz
    where target_rate uses '_' instead of '.'.

    For each discovered target_rate (kHz), on selected dev-test datasets:
        p_hat = P(mse >= T_val)
        L = log((p_hat + eps) / (FPR + eps)), eps = 0.5 / N
    Stores abs(L) by default for minimization.

    :param loss_name: String containing the name of the loss in the output dictionary
        where all loss values are stored (for each event).
    :param ds: List of strings with data set names where to calculate the threshold
        drift on.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    :param name: String specifying the name of the callback for identification in
        later methods that manipulate callbacks.
    """

    _THRES_RE = re.compile(r"^thres_(?P<trate>.+)kHz$")
    def __init__(
        self,
        loss_name: str,
        bc_rate: float = 28608.8064,
        log_raw_mlflow: bool = True,
        name: str = "thres_transfer",
    ):
        self.device = None
        self.loss_name = loss_name
        self.log_raw_mlflow = log_raw_mlflow
        self.name = name
        self.bc_rate = bc_rate

        self.transfer_summary = defaultdict(dict)

        self.target_rates = None
        self.thresholds = None

    def on_test_epoch_start(self, trainer, pl_module):
        self.device = pl_module.device

        dset_names = list(trainer.test_dataloaders.keys())
        if "main_test" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a test dataloader named 'main_test'. "
                f"Available test dataloaders: {dset_names}"
            )

        # Discover thresholds + rates from buffers
        self.thresholds = self._discover_thresholds(pl_module)
        self.target_rates = sorted(self.thresholds.keys())

        # Initialize counters ONLY for main_test
        self.total_counts = 0
        self.exceed_counts = {tr: 0 for tr in self.target_rates}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Count how many values exceed the val threshold on main_test at each rate."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name != "main_test":
            return

        loss = outputs[self.loss_name]
        if loss.ndim == 0:
            raise ValueError(f"outputs['{self.loss_name}'] is scalar. Need loss tensor.")

        loss = loss.detach().to(self.device).view(-1)
        n = int(loss.numel())
        self.total_counts += n

        for tr in self.target_rates:
            thr = self.thresholds[tr].view(-1)[0]
            self.exceed_counts[tr] += int((loss >= thr).sum().item())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Compute the calibration drift on the main_test data set."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        ckpts_dir = Path(pl_module._ckpt_path).parent
        plot_folder = ckpts_dir / "plots" / ckpt_name / "threshold_transfer"
        plot_folder.mkdir(parents=True, exist_ok=True)

        N = int(self.total_counts)
        if N <= 0:
            raise RuntimeError("No samples were processed for 'main_test'.")

        eps = 0.5 / float(N)
        for trate in self.target_rates:
            fp = int(self.exceed_counts[trate])
            p_hat = fp / float(N)

            fpr = float(trate) / float(self.bc_rate)
            L = math.log((p_hat + eps) / (float(fpr) + eps))
            drift_metric = abs(L)

            trate_name = f"{trate} kHz"
            xlabel = (
                f"Calibration drift at threshold: {trate_name}\n"
                f"log((p̂+ε)/(FPR+ε))"
            )
            self._plot({"main_test": drift_metric}, xlabel, plot_folder, percent=False)
            self._store_summary(drift_metric, ckpt_name, trate)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            "thres_transfer",
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"thres_transfer_{self.loss_name.replace('/', '_')}",
        )

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def _store_summary(self, drift_metric: float, ckpt_name: str, trate: float):
        """Store the summary metric for each checkpoint.

        Here, it's just the value of the calibration drift for that checkpoint.
        """
        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
        self.transfer_summary[trate][ckpt_ds] = float(drift_metric)

    def get_optimized_metric(self, ckpt_name: str, target_rate: float):
        """Get one number that one should optimize on this callback.

        This is the value of the drift mteric for a certain bkg rate at a certain
        checkpoint data set.
        """
        if target_rate not in self.transfer_summary:
            log.warn(
                f"Target rate {target_rate} not found."
                f"Available: {list(self.transfer_summary.keys())}"
            )
            return ckpt_name, None

        if ckpt_name not in self.transfer_summary[target_rate]:
            log.warn(
                f"ckpt_name={ckpt_name} not found for target_rate={target_rate}. "
                f"Available: {list(self.transfer_summary[target_rate].keys())}"
            )
            return ckpt_name, None

        optimized_loss = self.transfer_summary[target_rate][ckpt_name]
        return ckpt_name, optimized_loss

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the summary metric.

        In this case, plot the calibration drift for each threhold target rate, for
        each checkpoint.
        """
        plot_folder = root_folder / "plots" / "thres_transfer_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        for trate in sorted(self.transfer_summary.keys()):
            smet = self.transfer_summary[trate]

            trate_name = f"{trate} kHz"
            xlabel = (
                f"Calibration drift across checkpoints at threshold: {trate_name}\n"
                f"log((p̂+ε)/(FPR+ε))"
            )
            self._plot(smet, xlabel, plot_folder, percent=False)

        utils.mlflow.log_plots_to_mlflow(trainer, None, "thres_transfer", plot_folder)

    def clear_crit_summary(self):
        self.transfer_summary.clear()

    def _cache_summary(self, cache_folder: Path):
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.transfer_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _parse_target_rate_from_buffer_name(self, buf_name: str) -> float | None:
        """Get the target rate associated with main_val.

        Get the target rate from the name that is stored during the validation, when
        the threshold is actually set.
        """
        m = ThresholdDriftCallback._THRES_RE.match(buf_name)
        if not m:
            return None
        trate_str = m.group("trate").replace("_", ".")
        try:
            return float(trate_str)
        except ValueError:
            return None

    def _discover_thresholds(self, pl_module) -> dict[float, torch.Tensor]:
        """Discover what thresholds were set during the validation."""
        thresholds = {}
        for name, buf in pl_module.named_buffers():
            tr = self._parse_target_rate_from_buffer_name(name)
            if tr is None:
                continue
            thresholds[tr] = buf.detach().to(pl_module.device)
        if not thresholds:
            raise AttributeError(
                "No threshold buffers found on module. Expected buffers named like "
                "'thres_{target_rate}kHz' registered during validation."
            )
        return thresholds
