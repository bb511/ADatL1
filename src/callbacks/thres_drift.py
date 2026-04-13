# Threshold drift callback.
from collections import defaultdict
import math

import torch
from pytorch_lightning import Callback


class ThresholdDriftCallback(Callback):
    """Compute a validation-style split-transfer drift metric from one dataset.

    This callback collects all scores from the 'normal' dataloader, splits them
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

    :param output_name: Key in outputs dict containing per-event anomaly scores / losses.
    :param target_rates: List of target background rates in kHz.
    :param bc_rate: Bunch crossing rate in kHz.
    :param calibration_fraction: Fraction of normal scores used for calibration.
        The remainder is used for evaluation.
    :param split_seed: Seed used for deterministic random splitting.
    :beta: Float that sets parameter of EMA metrics compute here.
    """

    def __init__(
        self,
        output_name: str,
        target_rates: list[float],
        bc_rate: float = 28608.8064,
        calibration_fraction: float = 0.5,
        split_seed: int = 12345,
        beta: float = 0.9,
    ):
        super().__init__()
        self.output_name = output_name
        self.target_rates = sorted(float(x) for x in target_rates)
        self.bc_rate = float(bc_rate)
        self.calibration_fraction = float(calibration_fraction)
        self.split_seed = int(split_seed)
        self.beta = beta

        self.log_kwargs = dict(
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not (0.0 < self.calibration_fraction < 1.0):
            raise ValueError("calibration_fraction must be strictly between 0 and 1.")

    def on_fit_start(self, trainer, pl_module):
        """Instantiate useful quantities."""
        self.drift_ema = defaultdict(float)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Set the device and make sure the normal data is in the used data sets."""
        self.device = pl_module.device

        dset_names = list(trainer.val_dataloaders.keys())
        if "normal" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'normal'. "
                f"Available validation dataloaders: {dset_names}"
            )

        self.normal_scores = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Split the normal validation data set and aggregate anomaly scores."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        loss = outputs[self.output_name]
        if loss.ndim == 0:
            raise ValueError(f"outputs['{self.output_name}'] is scalar. Need a tensor.")

        loss = loss.detach().view(-1)
        self.normal_scores.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the threshold drift metric across the two data sets."""
        if not self.normal_scores:
            raise RuntimeError("No normal validation scores were collected.")

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

        for trate in self.target_rates:
            fpr = float(trate) / float(self.bc_rate)
            thr = self._compute_threshold(cal_scores, exceedance_prob=fpr)

            fp = int((eval_scores >= thr).sum().item())
            p_hat = fp / float(n_eval)

            L = math.log((p_hat + eps) / (fpr + eps))
            drift_metric = abs(L)

            trate_name = f"{trate}".replace(".", "_")
            self._compute_drift_ema(trate_name, drift_metric)

            pl_module.log_dict(
                {
                    f"val/summary/trate{trate_name}kHz_drift_ema": float(
                        self.drift_ema[trate_name]
                    )
                },
                **self.log_kwargs,
            )

    def _compute_drift_ema(self, trate_name: str, drift: float):
        """Compute the cvar estimated moving average."""
        if self.drift_ema[trate_name] == 0.0:
            self.drift_ema[trate_name] = float(drift)
        else:
            self.drift_ema[trate_name] = self.beta * self.drift_ema[trate_name] + (
                1 - self.beta
            ) * float(drift)

    def _split_scores(self, scores: torch.Tensor):
        """Split the anoamly score in a seeded way."""
        scores = scores.view(-1)
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
        """Compute the threshold corresponding to a certain rate -> exceedance prob."""
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

        sorted_scores, _ = torch.sort(scores)
        q = 1.0 - exceedance_prob
        idx = int(math.ceil(q * n) - 1)
        idx = max(0, min(n - 1, idx))
        return sorted_scores[idx]
