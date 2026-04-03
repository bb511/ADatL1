# Callback that computes an internal split-transfer threshold drift metric.
import math
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning import Callback

from src.evaluation.callbacks import utils
from src.plot import horizontal_bar
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class ThresholdDriftCallback(Callback):
    """Compute a validation-style split-transfer drift metric from one dataset.

    This callback collects all scores from the 'zerobias' dataloader, splits them
    internally into:
        - calibration subset
        - evaluation subset

    Then, for each target rate:
        1) compute a threshold on the calibration subset
        2) apply that threshold on the evaluation subset
        3) measure drift with:

            L = log((p_hat + eps) / (FPR + eps))
            drift = abs(L)

    where:
        - p_hat is the empirical exceedance rate on the evaluation subset
        - FPR = target_rate / bc_rate
        - eps = 0.5 / N_eval

    This is appropriate as a validation-side proxy objective for HPO.

    :param loss_name: Key in outputs dict containing per-event anomaly scores / losses.
    :param target_rates: List of target background rates in kHz.
    :param bc_rate: Bunch crossing rate in kHz.
    :param calibration_fraction: Fraction of zerobias scores used for calibration.
        The remainder is used for evaluation.
    :param split_seed: Seed used for deterministic random splitting.
    :param log_raw_mlflow: Whether to log raw plots to mlflow.
    :param name: Callback identifier.
    """

    def __init__(
        self,
        loss_name: str,
        target_rates: list[float],
        bc_rate: float = 28608.8064,
        calibration_fraction: float = 0.5,
        split_seed: int = 12345,
        log_raw_mlflow: bool = True,
        name: str = "thres_transfer",
    ):
        super().__init__()
        self.device = None
        self.loss_name = loss_name
        self.target_rates = sorted(float(x) for x in target_rates)
        self.bc_rate = float(bc_rate)
        self.calibration_fraction = float(calibration_fraction)
        self.split_seed = int(split_seed)
        self.log_raw_mlflow = log_raw_mlflow
        self.name = name

        if not (0.0 < self.calibration_fraction < 1.0):
            raise ValueError("calibration_fraction must be strictly between 0 and 1.")

        self.transfer_summary = defaultdict(dict)

    def on_test_epoch_start(self, trainer, pl_module):
        self.device = pl_module.device

        dset_names = list(trainer.test_dataloaders.keys())
        if "zerobias" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'zerobias'. "
                f"Available test dataloaders: {dset_names}"
            )

        self.zerobias_scores = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Accumulate zerobias scores across the whole epoch."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name != "zerobias":
            return

        loss = outputs[self.loss_name]
        if loss.ndim == 0:
            raise ValueError(f"outputs['{self.loss_name}'] is scalar. Need a tensor.")

        loss = loss.detach().view(-1).cpu()
        self.zerobias_scores.append(loss)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Compute split-transfer drift on zerobias."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        ckpts_dir = Path(pl_module._ckpt_path).parent
        split = getattr(trainer, "split", "test")
        plot_folder = ckpts_dir / "plots" / split / ckpt_name / self.name
        plot_folder.mkdir(parents=True, exist_ok=True)

        if not self.zerobias_scores:
            raise RuntimeError("No zerobias scores were collected.")

        scores = torch.cat(self.zerobias_scores, dim=0).view(-1)
        n_total = int(scores.numel())
        if n_total < 2:
            raise RuntimeError(
                f"Need at least 2 zerobias scores to split, got {n_total}."
            )

        cal_scores, eval_scores = self._split_scores(scores)
        n_eval = int(eval_scores.numel())
        if n_eval <= 0:
            raise RuntimeError("Evaluation split is empty after internal split.")

        eps = 0.5 / float(n_eval)

        for trate in self.target_rates:
            fpr = float(trate) / float(self.bc_rate)
            thr = self._compute_threshold(cal_scores, exceedance_prob=fpr)

            fp = int((eval_scores >= thr).sum().item())
            p_hat = fp / float(n_eval)

            L = math.log((p_hat + eps) / (fpr + eps))
            drift_metric = abs(L)

            trate_name = f"{trate} kHz"
            xlabel = (
                f"Split-transfer drift at threshold: {trate_name}\n"
                f"log((p̂+ε)/(FPR+ε))"
            )
            self._plot({"zerobias": drift_metric}, xlabel, plot_folder, percent=False)
            self._store_summary(drift_metric, ckpt_name, trate)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            self.name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{self.name}",
        )

    def _split_scores(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministically split scores into calibration and evaluation subsets."""
        n = int(scores.numel())
        n_cal = int(round(self.calibration_fraction * n))
        n_cal = max(1, min(n - 1, n_cal))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.split_seed)
        perm = torch.randperm(n, generator=gen)

        cal_idx = perm[:n_cal]
        eval_idx = perm[n_cal:]

        return scores[cal_idx], scores[eval_idx]

    def _compute_threshold(
        self, scores: torch.Tensor, exceedance_prob: float
    ) -> torch.Tensor:
        """Compute threshold so that P(score >= threshold) ~= exceedance_prob.

        Uses an empirical order-statistics threshold on the calibration scores.
        """
        scores = scores.view(-1)
        n = int(scores.numel())
        if n == 0:
            raise RuntimeError(
                "Cannot compute threshold from an empty calibration set."
            )

        if exceedance_prob <= 0.0:
            return torch.tensor(float("inf"), device=scores.device, dtype=scores.dtype)

        if exceedance_prob >= 1.0:
            return scores.min()

        sorted_scores, _ = torch.sort(scores)  # ascending
        # Keep approximately exceedance_prob mass above threshold.
        q = 1.0 - exceedance_prob
        idx = int(math.ceil(q * n) - 1)
        idx = max(0, min(n - 1, idx))
        return sorted_scores[idx]

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def _store_summary(self, drift_metric: float, ckpt_name: str, trate: float):
        """Store one drift value per checkpoint dataset and target rate."""
        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
        self.transfer_summary[trate][ckpt_ds] = float(drift_metric)

    def get_optimized_metric(self, target_rate: float):
        """Return the drift metric for one target rate at one checkpoint dataset."""
        if not target_rate in list(self.transfer_summary.keys()):
            raise ValueError(
                f"Thres trans callback for metric {self.metric_name} did not calculate "
                f"eff at target rate {target_rate}. Choose {self.transfer_summary.keys()}."
            )

        metric_across_ckpts = self.transfer_summary[target_rate]
        min_ckpt_name = min(metric_across_ckpts, key=metric_across_ckpts.get)
        min_metric_value = metric_across_ckpts[min_ckpt_name]
        return min_ckpt_name, min_metric_value

    def plot_summary(self, trainer, root_folder: Path):
        """Plot drift summary across checkpoints for each target rate."""
        split = getattr(trainer, "split", "test")
        plot_folder = root_folder / "plots" / split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        for trate in sorted(self.transfer_summary.keys()):
            smet = self.transfer_summary[trate]
            trate_name = f"{trate} kHz"
            xlabel = (
                f"split-transfer drift at threshold: {trate_name}\n"
                f"log((p̂+ε)/(FPR+ε))"
            )
            self._plot(smet, xlabel, plot_folder, percent=False)

        utils.mlflow.log_plots_to_mlflow(trainer, None, self.name, plot_folder)

    def clear_crit_summary(self):
        self.transfer_summary.clear()

    def _cache_summary(self, cache_folder: Path):
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.transfer_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
