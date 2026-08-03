# Threshold drift callback.
import math
from collections import defaultdict

import torch
from pytorch_lightning import Callback


class ThresholdDriftCallback(Callback):
    """Compute a validation-style threshold-transfer drift metric.

    By default this preserves the original behaviour: collect scores from the
    ``normal`` dataloader and split them into calibration and evaluation subsets.
    When ``dataset_2`` is provided, the full ``dataset_1`` sample is instead used
    for calibration and the full ``dataset_2`` sample for evaluation.

    Then, for each target rate:
        1) compute a threshold on the calibration subset
        2) apply that threshold on the evaluation subset
        3) measure drift with:

            L = log((p_hat + eps) / (FPR + eps))
            drift = abs(L)

    where:
        - p_hat is the empirical exceedance rate on the evaluation subset
        - FPR = target_rate / base_rate
        - eps = 0.5 / N_eval

    This is appropriate as a validation-side proxy objective for HPO.

    :param output_name: Key in outputs dict containing per-event anomaly scores / losses.
    :param target_rates: List of target background rates in kHz.
    :param base_rate: Bunch crossing rate in kHz.
    :param calibration_fraction: Fraction of normal scores used for calibration.
        The remainder is used for evaluation.
    :param split_seed: Seed used for deterministic random splitting.
    :beta: Float that sets parameter of EMA metrics compute here.
    """

    def __init__(
        self,
        output_name: str,
        target_rates: list[float] | None = None,
        base_rate: float | None = None,
        calibration_fraction: float = 0.5,
        split_seed: int = 12345,
        beta: float = 0.9,
        metric_name: str | None = None,
        dataset_1: str = "normal",
        dataset_2: str | None = None,
    ):
        super().__init__()
        self.output_name = output_name
        self.target_rates = (
            None if target_rates is None else sorted(float(x) for x in target_rates)
        )
        self.base_rate = base_rate

        self.calibration_fraction = float(calibration_fraction)
        self.split_seed = int(split_seed)
        self.beta = beta
        self.metric_name = None if metric_name is None else str(metric_name)
        self.dataset_1_name = str(dataset_1)
        self.dataset_2_name = None if dataset_2 is None else str(dataset_2)

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
        self.target_rates, self.base_rate = self._resolve_rate_config(pl_module)

        dset_names = list(trainer.val_dataloaders.keys())
        required = [self.dataset_1_name]
        if self.dataset_2_name is not None:
            required.append(self.dataset_2_name)
        missing = [name for name in required if name not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires validation dataloaders {missing}. "
                f"Available validation dataloaders: {dset_names}"
            )

        self.dataset_1_scores = []
        self.dataset_2_scores = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Aggregate scores from the configured calibration/evaluation datasets."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name not in (self.dataset_1_name, self.dataset_2_name):
            return

        loss = outputs[self.output_name]
        if loss.ndim == 0:
            raise ValueError(f"outputs['{self.output_name}'] is scalar. Need a tensor.")

        loss = loss.detach().view(-1)
        if dset_name == self.dataset_1_name:
            self.dataset_1_scores.append(loss)
        if dset_name == self.dataset_2_name:
            self.dataset_2_scores.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the threshold drift metric across the two data sets."""
        if not self.dataset_1_scores:
            raise RuntimeError(f"No validation scores were collected for '{self.dataset_1_name}'.")

        module_target = float(pl_module.hparams.target_rate)

        dataset_1_scores = torch.cat(self.dataset_1_scores, dim=0).view(-1)
        if self.dataset_2_name is None:
            n_total = int(dataset_1_scores.numel())
            if n_total < 2:
                raise RuntimeError(f"Need >=2 scores to split, got {n_total}.")
            cal_scores, eval_scores = self._split_scores(dataset_1_scores)
        else:
            if not self.dataset_2_scores:
                raise RuntimeError(
                    f"No validation scores were collected for '{self.dataset_2_name}'."
                )
            cal_scores = dataset_1_scores
            eval_scores = torch.cat(self.dataset_2_scores, dim=0).view(-1)
        n_eval = int(eval_scores.numel())
        if n_eval <= 0:
            raise RuntimeError("Evaluation split is empty after internal split.")

        eps = 0.5 / float(n_eval)

        for trate in self.target_rates:
            fpr = self._compute_target_fpr(trate)
            thr = self._compute_threshold(cal_scores, exceedance_prob=fpr)

            fp = int((eval_scores >= thr).sum().item())
            p_hat = fp / float(n_eval)

            L = math.log((p_hat + eps) / (fpr + eps))
            drift_metric = abs(L)

            trate_name = f"{trate}".replace(".", "_")
            self._compute_drift_ema(trate_name, drift_metric)

            is_operational = abs(trate - module_target) < 1e-12
            if is_operational:
                component = (
                    "operational_drift_ema"
                    if self.metric_name is None
                    else f"{self.metric_name}_operational_drift_ema"
                )
                key = f"val/summary/{component}"
            else:
                key = f"val/summary/trate{trate_name}kHz_drift_ema"

            pl_module.log_dict(
                {key: float(self.drift_ema[trate_name])},
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

    def _compute_threshold(self, scores: torch.Tensor, exceedance_prob: float) -> torch.Tensor:
        """Compute the threshold corresponding to a certain rate -> exceedance prob."""
        scores = scores.view(-1)
        n = int(scores.numel())
        if n == 0:
            raise RuntimeError("Cannot compute threshold from an empty calibration set.")

        if exceedance_prob <= 0.0:
            return torch.tensor(float("inf"), device=scores.device, dtype=scores.dtype)

        if exceedance_prob >= 1.0:
            return scores.min()

        sorted_scores, _ = torch.sort(scores)
        q = 1.0 - exceedance_prob
        idx = int(math.ceil(q * n) - 1)
        idx = max(0, min(n - 1, idx))
        return sorted_scores[idx]

    def _resolve_rate_config(self, pl_module) -> tuple[list[float], float | None]:
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
        return resolved_rates, base_rate

    def _compute_target_fpr(self, target_rate: float) -> float:
        """Convert target rate into an exceedance probability."""
        if self.base_rate is None:
            fpr = float(target_rate)
        else:
            if self.base_rate <= 0:
                raise ValueError("base_rate must be positive.")
            fpr = float(target_rate) / float(self.base_rate)

        if not (0.0 < fpr < 1.0):
            raise ValueError(f"Computed FPR must be in (0,1), got {fpr}")

        return fpr
