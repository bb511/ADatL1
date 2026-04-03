from collections import defaultdict
from pathlib import Path
import pickle

import torch

from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar


class Wasserstein(Callback):
    """Evaluate the Wasserstein distance between two score distributions.

    This is a signal-agnostic distribution-similarity baseline. Unlike CAP,
    it does not use pairing and compares the marginal score distributions
    directly.

    Args:
        metric_name: Name of the model output metric used as anomaly score.
        dataset_1: Name of the first dataset.
        dataset_2: Name of the second dataset.
        apply_log1p: Whether to apply log1p transform before distance computation.
        log_raw_mlflow: Whether to log raw plots to mlflow.
        name: Callback name.
    """

    def __init__(
        self,
        metric_name: str,
        dataset_1: str,
        dataset_2: str,
        apply_log1p: bool = True,
        log_raw_mlflow: bool = True,
        name: str = "wasserstein",
    ):
        super().__init__()
        self.metric_name = metric_name
        self.dataset_1_name = dataset_1
        self.dataset_2_name = dataset_2
        self.apply_log1p = apply_log1p
        self.log_raw_mlflow = log_raw_mlflow
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
            self.dataset_1_scores.append(outputs[self.metric_name])
        if dataset_name == self.dataset_2_name:
            self.dataset_2_scores.append(outputs[self.metric_name])

    def _compute_wasserstein(self) -> float:
        """Compute the 1D Wasserstein distance between the two datasets."""
        x = torch.cat(self.dataset_1_scores, dim=0).detach().flatten().cpu().float()
        y = torch.cat(self.dataset_2_scores, dim=0).detach().flatten().cpu().float()

        if self.apply_log1p:
            # Scores should be non-negative; clamp for safety.
            x = torch.log1p(torch.clamp(x, min=0.0))
            y = torch.log1p(torch.clamp(y, min=0.0))

        return self._wasserstein_1d_sorted(x, y)

    def _wasserstein_1d_sorted(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute empirical 1D Wasserstein-1 distance.

        For two empirical 1D distributions, the Wasserstein distance can be
        computed by sorting both samples and averaging absolute quantile gaps.
        If sample sizes differ, the larger sample is downsampled by quantile
        interpolation onto the smaller sample size.
        """
        x = torch.sort(x).values
        y = torch.sort(y).values

        nx = x.numel()
        ny = y.numel()

        if nx == 0 or ny == 0:
            return float("nan")

        if nx == ny:
            return torch.mean(torch.abs(x - y)).item()

        # Interpolate both distributions onto a common quantile grid
        n = min(nx, ny)
        q = torch.linspace(0.0, 1.0, n, dtype=torch.float32)

        xq = self._interp_quantiles(x, q)
        yq = self._interp_quantiles(y, q)

        return torch.mean(torch.abs(xq - yq)).item()

    def _interp_quantiles(
        self, sorted_vals: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate sorted samples at quantiles q in [0, 1]."""
        n = sorted_vals.numel()
        if n == 1:
            return sorted_vals.repeat(len(q))

        pos = q * (n - 1)
        lo = torch.floor(pos).long()
        hi = torch.ceil(pos).long()
        w = pos - lo.float()

        return (1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi]

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute and store Wasserstein distance for the current checkpoint."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        wasserstein_value = self._compute_wasserstein()
        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
        self._store_summary(wasserstein_value, ckpt_ds)

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
