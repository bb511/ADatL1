"""The paired test scores that the CAP and consistency evaluation callbacks share."""

import pickle
from pathlib import Path

import torch
from torch import Tensor

from src.evaluation.callbacks import utils
from src.evaluation.callbacks.components.cap.binary import get_pairing_fn
from src.plot import horizontal_bar


class PairedScores:
    """The scores of two test sets over a checkpoint's pass, paired, and the value
    computed from the pairs at every checkpoint, keyed by the checkpoint's data set.

    Args:
        name: Name of the callback, under which the values are logged and plotted.
        label: Name of the value on the axis of the summary plot.
        output_name: Key of the per-event anomaly scores in the outputs of a batch.
        dataset_1: Name of the first test dataloader.
        dataset_2: Name of the second test dataloader.
        pairing_type: Name of the pairing, as `get_pairing_fn` takes it.
    """

    def __init__(
        self,
        name: str,
        label: str,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
    ):
        self.name, self.label, self.output_name = name, label, output_name
        self.datasets = (dataset_1, dataset_2)
        self.pairing_fn = get_pairing_fn(pairing_type)
        self.summary = {}

    def reset(self) -> None:
        """Drop the scores of the last checkpoint."""
        self.scores = {dataset: [] for dataset in self.datasets}

    def collect(self, trainer, outputs: dict, dataloader_idx: int) -> None:
        """Keep the scores of a batch of either set."""
        dataset = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dataset in self.scores:
            self.scores[dataset].append(outputs[self.output_name])

    def paired(self) -> tuple[Tensor, Tensor]:
        """Return the pass's scores of the two sets, paired and cut to equal length."""
        scores_1, scores_2 = (torch.cat(self.scores[d]) for d in self.datasets)
        index_1, index_2 = self.pairing_fn(scores_1, scores_2)
        paired_1, paired_2 = scores_1[index_1], scores_2[index_2]
        n = min(len(paired_1), len(paired_2))
        return paired_1[:n], paired_2[:n]

    def record(self, trainer, pl_module, value: float, extra: dict | None = None):
        """Keep the checkpoint's value and log it to MLflow, followed by extra."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        self.summary[utils.misc.get_ckpt_ds_name(ckpt_name)] = value
        metrics = {self.name: value} | (extra or {})
        utils.mlflow.log_metrics_to_mlflow(
            trainer, metrics, ckpt_name=ckpt_name, cb_name=self.name
        )

    def plot_summary(self, trainer, root_folder: Path) -> None:
        """Pickle the values, plot them by checkpoint and log the plot to MLflow."""
        plot_folder = root_folder / "plots" / trainer.split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        with open(plot_folder / "summary.pkl", "wb") as f:
            pickle.dump(self.summary, f, protocol=pickle.HIGHEST_PROTOCOL)
        xlabel = f"{self.label}({self.datasets[0]},\n{self.datasets[1]})"
        horizontal_bar.plot_yright(self.summary, self.summary, xlabel, " ", plot_folder)
        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def best(self) -> tuple[str, float]:
        """Return the checkpoint of the largest value, and the value."""
        ckpt_ds = max(self.summary, key=self.summary.get)
        return ckpt_ds, self.summary[ckpt_ds]
