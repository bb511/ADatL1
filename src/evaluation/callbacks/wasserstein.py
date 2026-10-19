import pickle
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


def _wasserstein_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return the exact W1 between the empirical distributions of x and y, the
    integral of |F_x - F_y| over the merged sorted values, in float64; NaN when
    either is empty."""
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")

    x = torch.sort(x.double()).values
    y = torch.sort(y.double()).values
    merged = torch.sort(torch.cat([x, y])).values
    cdf_x = torch.searchsorted(x, merged[:-1], right=True).double() / x.numel()
    cdf_y = torch.searchsorted(y, merged[:-1], right=True).double() / y.numel()
    return torch.sum(torch.abs(cdf_x - cdf_y) * torch.diff(merged)).item()


class Wasserstein(Callback):
    """Evaluate the Wasserstein distance between two score distributions.

    This is a signal-agnostic distribution-similarity baseline. Unlike CAP, it does not
    use pairing and compares the marginal score distributions directly.

    :param output_name: Name of the model output metric used as anomaly score.
    :param dataset_1: Name of the first dataset.
    :param dataset_2: Name of the second dataset.
    :param apply_log1p: Whether to map scores through sign(x) log1p(|x|), which equals
        log1p(x) for non-negative scores, before distance computation.
    :param log_raw_mlflow: Unused, accepted for config compatibility.
    :param name: Callback name.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        apply_log1p: bool = True,
        log_raw_mlflow: bool = True,
        name: str = "wasserstein",
    ):
        super().__init__()
        self.output_name = output_name
        self.dataset_1_name = dataset_1
        self.dataset_2_name = dataset_2
        self.apply_log1p = apply_log1p
        self.name = name

        self.wasserstein_summary = defaultdict(float)

    def on_test_epoch_start(self, trainer, pl_module):
        """Initialize score buffers."""
        self.dataset_1_scores = []
        self.dataset_2_scores = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache scores for the designated datasets."""
        dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dataset_name == self.dataset_1_name:
            self.dataset_1_scores.append(outputs[self.output_name])
        if dataset_name == self.dataset_2_name:
            self.dataset_2_scores.append(outputs[self.output_name])

    def _compute_wasserstein(self) -> float:
        """Compute the 1D Wasserstein distance between the two datasets."""
        x = torch.cat(self.dataset_1_scores, dim=0).detach().flatten().cpu().float()
        y = torch.cat(self.dataset_2_scores, dim=0).detach().flatten().cpu().float()

        if self.apply_log1p:
            # Sign-preserving because RealNVP's negative log-likelihood is unbounded
            # below: log1p is undefined below -1, and a clamp at zero would merge
            # every negative score into 0.
            x = torch.sign(x) * torch.log1p(torch.abs(x))
            y = torch.sign(y) * torch.log1p(torch.abs(y))

        return _wasserstein_1d(x, y)

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute and store Wasserstein distance for the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        wasserstein_value = self._compute_wasserstein()
        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
        self._store_summary(wasserstein_value, ckpt_ds)
        utils.mlflow.log_metrics_to_mlflow(
            trainer,
            {"w1dist": self.wasserstein_summary[ckpt_ds]},
            ckpt_name=ckpt_name,
            cb_name=self.name,
        )

    def _store_summary(self, wasserstein_value: float, ckpt_ds: str):
        """Store Wasserstein summary for one checkpoint."""
        self.wasserstein_summary[ckpt_ds] = wasserstein_value

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        """Plot summary metric."""
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def plot_summary(self, trainer, root_folder: Path):
        """Plot and cache summary metrics."""
        split = trainer.split
        plot_folder = root_folder / "plots" / split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        xlabel = f"W1({self.dataset_1_name},\n{self.dataset_2_name})"
        self._plot(self.wasserstein_summary, xlabel, plot_folder)

        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def get_optimized_metric(self):
        """Return the checkpoint with minimum Wasserstein distance."""
        min_ckpt_ds = min(self.wasserstein_summary, key=self.wasserstein_summary.get)
        min_metric_value = self.wasserstein_summary[min_ckpt_ds]
        return min_ckpt_ds, min_metric_value

    def clear_crit_summary(self):
        self.wasserstein_summary.clear()

    def _cache_summary(self, cache_folder: Path):
        """Cache summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.wasserstein_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
