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
    """Compute a split-transfer threshold drift metric from the normal dataset.

    This callback collects anomaly scores from the 'normal' dataloader, splits them
    internally into calibration and evaluation subsets, and for each target rate:
        1) computes a threshold on the calibration subset,
        2) applies that threshold on the evaluation subset,
        3) measures drift via

            L = log((p_hat + eps) / (FPR + eps)),
            drift = abs(L),

    where p_hat is the empirical exceedance rate on the evaluation subset,
    FPR is obtained from target_rate and base_rate, and eps = 0.5 / N_eval.

    The module target rate is always included and is treated as the operational rate.
    Additional target rates may be passed through the callback config.

    :param output_name: Key in outputs dict containing per-event anomaly scores.
    :param target_rates: Optional extra target rates to evaluate.
    :param base_rate: Optional override for the module base rate.
    :param calibration_fraction: Fraction of normal scores used for calibration.
    :param split_seed: Seed used for deterministic random splitting.
    :param log_raw_mlflow: Whether to log raw plots to mlflow.
    :param name: Callback identifier.
    """

    def __init__(
        self,
        output_name: str,
        target_rates: list[float] | None = None,
        base_rate: float | None = None,
        calibration_fraction: float = 0.5,
        split_seed: int = 12345,
        log_raw_mlflow: bool = True,
        name: str = "thres_transfer",
    ):
        super().__init__()
        self.device = None
        self.output_name = output_name
        self.target_rates = (
            None if target_rates is None else sorted(float(x) for x in target_rates)
        )
        self.base_rate = base_rate
        self.calibration_fraction = float(calibration_fraction)
        self.split_seed = int(split_seed)
        self.log_raw_mlflow = log_raw_mlflow
        self.name = name

        if not (0.0 < self.calibration_fraction < 1.0):
            raise ValueError("calibration_fraction must be strictly between 0 and 1.")

        self.transfer_summary = defaultdict(dict)

    def on_test_epoch_start(self, trainer, pl_module):
        self.device = pl_module.device
        (
            self.target_rates_resolved,
            self.operational_rate,
            self.base_rate_resolved,
        ) = self._resolve_rate_config(pl_module)

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
        """Accumulate normal scores across the whole epoch."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        score = outputs[self.output_name]
        if score.ndim == 0:
            raise ValueError(f"outputs['{self.output_name}'] is scalar. Need a tensor.")

        self.normal_scores.append(score.detach().view(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Compute split-transfer drift on normal."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        ckpts_dir = Path(pl_module._ckpt_path).parent
        split = getattr(trainer, "split", "test")
        plot_folder = ckpts_dir / "plots" / split / ckpt_name / self.name
        plot_folder.mkdir(parents=True, exist_ok=True)

        if not self.normal_scores:
            raise RuntimeError("No normal scores were collected.")

        scores = torch.cat(self.normal_scores, dim=0).view(-1)
        n_total = int(scores.numel())
        if n_total < 2:
            raise RuntimeError(
                f"Need at least 2 normal scores to split, got {n_total}."
            )

        cal_scores, eval_scores = self._split_scores(scores)
        n_eval = int(eval_scores.numel())
        if n_eval <= 0:
            raise RuntimeError("Evaluation split is empty after internal split.")

        eps = 0.5 / float(n_eval)

        for trate in self.target_rates_resolved:
            fpr = self._compute_target_fpr(trate)
            thr = self._compute_threshold(cal_scores, exceedance_prob=fpr)

            fp = int((eval_scores >= thr).sum().item())
            p_hat = fp / float(n_eval)

            drift_metric = abs(math.log((p_hat + eps) / (fpr + eps)))

            if self._is_operational(trate):
                trate_key = "operational"
                display_name = "operational"
            else:
                trate_key = self._target_key(trate)
                display_name = str(trate)

            xlabel = (
                f"Split-transfer drift at threshold: {display_name}\n"
                f"log((p̂+ε)/(FPR+ε))"
            )
            self._plot({"normal": drift_metric}, xlabel, plot_folder, percent=False)
            self._store_summary(drift_metric, ckpt_name, trate_key)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            self.name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{self.name}",
        )

    def _resolve_rate_config(
        self, pl_module
    ) -> tuple[list[float], float, float | None]:
        """Resolve target rates and base rate from module + callback config."""
        module_target = getattr(pl_module.hparams, "target_rate", None)
        module_base = getattr(pl_module.hparams, "base_rate", None)

        if module_target is None:
            raise ValueError(
                "pl_module.hparams.target_rate must be defined for ThresholdDriftCallback."
            )

        rates = [float(module_target)]
        if self.target_rates is not None:
            rates.extend(float(r) for r in self.target_rates)

        seen = set()
        resolved_rates = []
        for r in rates:
            if r not in seen:
                seen.add(r)
                resolved_rates.append(r)

        base_rate = self.base_rate if self.base_rate is not None else module_base
        return resolved_rates, float(module_target), base_rate

    def _compute_target_fpr(self, target_rate: float) -> float:
        """Convert target rate into an exceedance probability."""
        if self.base_rate_resolved is None:
            fpr = float(target_rate)
        else:
            if self.base_rate_resolved <= 0:
                raise ValueError("base_rate must be positive.")
            fpr = float(target_rate) / float(self.base_rate_resolved)

        if not (0.0 < fpr < 1.0):
            raise ValueError(f"Computed FPR must be in (0,1), got {fpr}")

        return fpr

    def _is_operational(self, target_rate: float) -> bool:
        return abs(target_rate - self.operational_rate) < 1e-12

    def _target_key(self, target_rate: float) -> str:
        return f"trate{str(target_rate).replace('.', '_')}kHz"

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
        """Compute threshold so that P(score >= threshold) ~= exceedance_prob."""
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
        q = 1.0 - exceedance_prob
        idx = int(math.ceil(q * n) - 1)
        idx = max(0, min(n - 1, idx))
        return sorted_scores[idx]

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def _store_summary(self, drift_metric: float, ckpt_name: str, trate_key: str):
        """Store one drift value per checkpoint dataset and target key."""
        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
        self.transfer_summary[trate_key][ckpt_ds] = float(drift_metric)

    def get_optimized_metric(self, target_rate: float | None = None):
        """Return the drift metric for one target rate at one checkpoint dataset.

        If target_rate is None, the operational target rate is used.
        """
        if target_rate is None:
            trate_key = "operational"
        elif self._is_operational(target_rate):
            trate_key = "operational"
        else:
            trate_key = self._target_key(target_rate)

        if trate_key not in self.transfer_summary:
            raise ValueError(
                f"Threshold drift callback did not calculate drift for {trate_key}. "
                f"Choose from {self.transfer_summary.keys()}."
            )

        metric_across_ckpts = self.transfer_summary[trate_key]
        min_ckpt_name = min(metric_across_ckpts, key=metric_across_ckpts.get)
        min_metric_value = metric_across_ckpts[min_ckpt_name]
        return min_ckpt_name, min_metric_value

    def plot_summary(self, trainer, root_folder: Path):
        """Plot drift summary across checkpoints for each target rate."""
        split = getattr(trainer, "split", "test")
        plot_folder = root_folder / "plots" / split / f"{self.name}_summary"
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        for trate_key in sorted(self.transfer_summary.keys()):
            smet = self.transfer_summary[trate_key]

            if trate_key == "operational":
                display_name = "operational"
            else:
                display_name = trate_key.replace("trate", "").replace("kHz", "")

            xlabel = (
                f"split-transfer drift at threshold: {display_name}\n"
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
