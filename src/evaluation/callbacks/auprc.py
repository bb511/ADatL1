from collections import defaultdict
from pathlib import Path
import pickle

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import BinaryAveragePrecision

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar
from src.data.utils import unpack_batch

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class AnomalyAUPRCCallback(Callback):
    """Calculates the AUPRC for each signal dataset against normal data.

    Normal events are treated as class 0 and signal events as class 1.
    The anomaly score is expected to increase with anomalousness.

    :param output_name: Name of the output dict entry used as anomaly score.
    :param ds: List of dataset names to compute AUPRC on.
    :param pure_normal: Whether to use only normal events with l1bit == False.
    :param log_raw_mlflow: Whether to log raw plots to mlflow.
    :param name: Identifier used in plot folders and summaries.
    """

    def __init__(
        self,
        output_name: str,
        ds: list[str],
        pure_normal: bool = False,
        log_raw_mlflow: bool = True,
        name: str = "auprc",
    ):
        super().__init__()
        self.device = None
        self.name = name

        self.output_name = output_name
        self.ds = set(ds)
        self.pure_normal = pure_normal

        self.log_raw_mlflow = log_raw_mlflow
        self.auprc_summary = defaultdict(float)
        self.auprc_min = defaultdict(float)
        self.auprc_med = defaultdict(float)

    def on_test_start(self, trainer, pl_module):
        """Verify expected dataloaders are present."""
        self.device = pl_module.device

        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        if first_test_dset_key != "normal":
            raise ValueError("AUPRC callback needs normal data first in the data dict!")

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear accumulated score data at the start of the epoch."""
        self.normal_score_data = []
        self.sig_score_data = defaultdict(list)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Accumulate normal and signal scores for AUPRC computation."""
        dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]

        b = unpack_batch(batch)
        labels = b.y
        l1bit = b.l1bit

        batch_output = outputs[self.output_name]
        if batch_output.ndim == 0:
            batch_output = batch_output.unsqueeze(0)

        batch_output = batch_output.detach()

        if dataset_name == "normal":
            self._accumulate_normal_output(batch_output, l1bit)
            return

        if dataset_name not in self.ds:
            return

        if torch.all(labels > 0):
            self.sig_score_data[dataset_name].append(batch_output)
        elif torch.all(labels < 0):
            return
        else:
            raise ValueError(
                f"AUPRC callback expects each non-normal dataloader to contain either "
                f"only signal labels > 0 or only background labels < 0. Got mixed labels "
                f"for dataset {dataset_name}."
            )

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Compute and log AUPRC for each signal dataset."""
        auprc_name = f"{self.name}_pure" if self.pure_normal else self.name
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        split = trainer.split
        plot_folder = ckpts_dir / "plots" / split / ckpt_name / auprc_name
        plot_folder.mkdir(parents=True, exist_ok=True)

        auprcs = self._compute_auprcs()

        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
        auprc_data = np.fromiter(auprcs.values(), dtype=float)

        if auprc_data.size:
            self.auprc_summary[ckpt_ds] = float(np.mean(auprc_data))
            self.auprc_med[ckpt_ds] = float(np.median(auprc_data))
            self.auprc_min[ckpt_ds] = float(np.min(auprc_data))
        else:
            self.auprc_summary[ckpt_ds] = 0.0
            self.auprc_med[ckpt_ds] = 0.0
            self.auprc_min[ckpt_ds] = 0.0

        ascore = f"anomaly score: {self.output_name}"
        xlabel = f"AUPRC per signal\n{ascore}"
        self._plot(auprcs, xlabel, plot_folder, percent=False)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            auprc_name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{auprc_name}",
        )

    def plot_summary(self, trainer, root_folder: Path):
        """Plot summary AUPRC metrics accumulated across checkpoints."""
        auprc_name = f"{self.name}_pure" if self.pure_normal else self.name
        split = trainer.split
        plot_folder = root_folder / "plots" / split / (auprc_name + "_summary")
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        ascore = f"anomaly score: {self.output_name}"

        xlabel = f"mean AUPRC across signals\n{ascore}"
        self._plot(self.auprc_summary, xlabel, plot_folder, percent=False)

        xlabel = f"median AUPRC across signals\n{ascore}"
        self._plot(self.auprc_med, xlabel, plot_folder, percent=False)

        xlabel = f"min AUPRC across signals\n{ascore}"
        self._plot(self.auprc_min, xlabel, plot_folder, percent=False)

        utils.mlflow.log_plots_to_mlflow(trainer, None, auprc_name, plot_folder)

    def clear_crit_summary(self):
        self.auprc_summary.clear()
        self.auprc_med.clear()
        self.auprc_min.clear()

    def get_optimized_metric(self):
        """Return the checkpoint with the best mean AUPRC across signals."""
        if not self.auprc_summary:
            raise ValueError(
                f"AUPRC callback for metric {self.output_name} did not calculate "
                "any summary metrics."
            )

        max_ckpt_name = max(self.auprc_summary, key=self.auprc_summary.get)
        max_metric_value = self.auprc_summary[max_ckpt_name]
        return max_ckpt_name, max_metric_value

    def _accumulate_normal_output(
        self, batch_output: torch.Tensor, l1bit: torch.Tensor | None
    ):
        """Accumulate the normal score distribution across batches."""
        if self.pure_normal:
            if l1bit is None:
                raise ValueError(
                    "pure_normal=True requires l1bit to be present in the batch."
                )
            batch_output = batch_output[~l1bit]

        self.normal_score_data.append(batch_output)

    def _compute_auprcs(self):
        """Compute AUPRC for each signal dataset against normal data."""
        if not self.normal_score_data:
            raise ValueError("No normal score data was accumulated for AUPRC.")

        normal_scores = torch.cat(self.normal_score_data, dim=0)
        auprcs = defaultdict(float)

        for ds_name, score_chunks in self.sig_score_data.items():
            if not score_chunks:
                auprcs[ds_name] = 0.0
                continue

            sig_scores = torch.cat(score_chunks, dim=0)
            auprcs[ds_name] = self._average_precision(normal_scores, sig_scores)

        return auprcs

    def _average_precision(
        self, normal_scores: torch.Tensor, sig_scores: torch.Tensor
    ) -> float:
        """Compute binary average precision with normal=0 and signal=1."""
        preds = torch.cat([normal_scores, sig_scores], dim=0).float().cpu()
        target = torch.cat(
            [
                torch.zeros_like(normal_scores, dtype=torch.long),
                torch.ones_like(sig_scores, dtype=torch.long),
            ],
            dim=0,
        ).cpu()

        metric = BinaryAveragePrecision()
        return float(metric(preds, target).item())

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        """Plot AUPRC per dataset for an anomaly metric."""
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionaries."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = {
                "mean": utils.misc.to_plain_dict(self.auprc_summary),
                "median": utils.misc.to_plain_dict(self.auprc_med),
                "min": utils.misc.to_plain_dict(self.auprc_min),
            }
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
