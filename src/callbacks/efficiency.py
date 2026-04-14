# Callback that computes the anomaly rate during training.
from collections import defaultdict
import math
import statistics

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningDataModule

from src.callbacks.metrics.rate import AnomalyRate
from src.data.utils import unpack_batch


class AnomalyEfficiencyCallback(Callback):
    """Calculates the fraction of anomalies detected given a certain bkg rate.

    :target_rate: Float specifying the target rate of bkg.
    :param base_rate: Optional override for the module base rate. If None, the module
        base_rate is used.
    :materic_name: Which model metric to use as the anomaly score to calculate rate on.
    :beta: Float that sets parameter of EMA metrics compute here.
    """

    def __init__(
        self,
        output_name: str,
        target_rates: list[int] | None = None,
        base_rate: float | None = None,
        beta: float = 0.9
    ):
        super().__init__()
        self.device = None
        self.output_name = output_name
        self.target_rates = None if target_rates is None else sorted(float(x) for x in target_rates)
        self.base_rate = base_rate
        self.beta = beta

        self.log_kwargs = dict(
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def on_fit_start(self, trainer, pl_module):
        """Instantiate useful quantities."""
        self.cvar25_ema = {}
        self.cvar10_ema = {}

    def on_validation_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        self.device = pl_module.device
        self.target_rates, self.base_rate = \
            self._resolve_rate_config(pl_module)

        # Check if 'normal' dataset is the first dataset in the validation dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        first_val_dset_key = list(trainer.val_dataloaders.keys())[0]
        if first_val_dset_key != "normal":
            raise ValueError("Rate callback requires normal first in the val dict!")

    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.main_rate = defaultdict(lambda: defaultdict(AnomalyRate))
        self.sig_rates = defaultdict(lambda: defaultdict(AnomalyRate))
        self.bkg_rates = defaultdict(lambda: defaultdict(AnomalyRate))
        self.normal_score_data = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Determine the rate for every given metric for every validation data set.

        First, the desired metrics are computed on the normal_data and accummulated
        across batches. The full metric distribution is used to set a threshold that
        will give a target rate or set of rates, specified by the user.
        These thresholds are then applied on all other data sets used for validation
        to determine, by using the threshold computed on normal, what would the rate
        be on these other data sets.
        """
        self.dataset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        self.total_batches = trainer.num_val_batches[dataloader_idx]
        b = unpack_batch(batch)
        labels = b.y

        if self.dataset_name == "normal":
            self._accumulate_normal_output(outputs, batch_idx, pl_module)
        else:
            if batch_idx == 0:
                self._initialize_rate_metric(labels)
            self._compute_batch_rate(outputs, labels)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        module_target = getattr(pl_module.hparams, "target_rate", None)
        for target_rate in self.target_rates:
            main_rate = self.main_rate[target_rate]["normal"].compute("rate").item()
            bkg_effs = self._compute_efficiencies(self.bkg_rates, target_rate)
            sig_effs = self._compute_efficiencies(self.sig_rates, target_rate)
            pl_module.log_dict(
                {f"val/normal/brate_{target_rate}kHz": main_rate}, **self.log_kwargs
            )
            pl_module.log_dict(bkg_effs, **self.log_kwargs)
            pl_module.log_dict(sig_effs, **self.log_kwargs)

            cvar25 = self._cvar_lower_tail(sig_effs.values(), alpha=0.25)
            cvar10 = self._cvar_lower_tail(sig_effs.values(), alpha=0.10)
            self._compute_cvar25_ema(target_rate, cvar25)
            self._compute_cvar10_ema(target_rate, cvar10)

            min_sig_eff = min(list(sig_effs.values()))
            med_sig_eff = statistics.median(list(sig_effs.values()))
            thres_zb = self.main_rate[target_rate]["normal"].threshold.item()

            is_operational = abs(target_rate - module_target) < 1e-12
            if is_operational:
                trate_name = 'operational'
            else:
                trate_name = str(target_rate).replace(".", "_")

            summaries = {
                f"val/summary/eff_cvar25_{trate_name}": cvar25,
                f"val/summary/eff_cvar10_{trate_name}": cvar10,
                f"val/summary/eff_min_{trate_name}": min_sig_eff,
                f"val/summary/eff_med_{trate_name}": med_sig_eff,
                f"val/summary/eff_cvar25_ema_{trate_name}": self.cvar25_ema[target_rate],
                f"val/summary/eff_cvar10_ema_{trate_name}": self.cvar10_ema[target_rate],
            }
            pl_module.log_dict(summaries, **self.log_kwargs)
            pl_module.log_dict({f"val/normal/thr__brate_{trate_name}kHz": thres_zb})

    def _compute_cvar25_ema(self, target_rate: float, value: float):
        if target_rate not in self.cvar25_ema:
            self.cvar25_ema[target_rate] = float(value)
        else:
            self.cvar25_ema[target_rate] = (
                self.beta * self.cvar25_ema[target_rate]
                + (1 - self.beta) * float(value)
            )

    def _compute_cvar10_ema(self, target_rate: float, value: float):
        """Compute the cvar10 exponential moving average."""
        if target_rate not in self.cvar10_ema:
            self.cvar10_ema[target_rate] = float(value)
        else:
            self.cvar10_ema[target_rate] = (
                self.beta * self.cvar10_ema[target_rate]
                + (1 - self.beta) * float(value)
            )

    def _accumulate_normal_output(self, outputs: dict, batch_idx: int, pl_module):
        """Accummulates the specified metric data across batches.

        Used if currently processing the normal data set, accummulate the values of
        each metric across batches. The whole metric output distribution is needed to
        set a treshold that gives a certain rate.
        """
        batch_output = outputs[self.output_name]
        if batch_output.ndim == 0:
            batch_output = batch_output.unsqueeze(0)
        self.normal_score_data.append(batch_output)

        if batch_idx == self.total_batches - 1:
            self.normal_score_data = torch.cat(self.normal_score_data, dim=0)
            self._compute_normal_rate(pl_module)

    def _compute_normal_rate(self, pl_module):
        """Computes the desired rates on the main validation data set.

        This is a sanity check. The threshold computed on the normal and applied to the
        normal data should return the rate for which this threshold was computed.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.base_rate).to(self.device)
            rate.set_threshold(self.normal_score_data)
            rate.update(self.normal_score_data)
            thres = rate.threshold
            self._set_thres_on_module(pl_module, target_rate, thres)

            self.main_rate[target_rate][self.dataset_name] = rate

    def _initialize_rate_metric(self, labels: torch.Tensor):
        """Initializes the rate metric for a dataset for each given target rate.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the normal metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from normal.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.base_rate).to(self.device)
            rate.set_threshold(self.normal_score_data)
            if torch.all(labels < 0):
                self.bkg_rates[target_rate][self.dataset_name] = rate
            if torch.all(labels > 0):
                self.sig_rates[target_rate][self.dataset_name] = rate

    def _compute_batch_rate(self, outputs: dict, labels: torch.Tensor):
        """Done after knowing the rate thresholds.

        For all the other validation data sets, except the main one, this calculates
        which events pass the rate threshold for each batch and updates the total.
        """
        for tr in self.target_rates:
            if torch.all(labels < 0):
                self.bkg_rates[tr][self.dataset_name].update(outputs[self.output_name])
            if torch.all(labels > 0):
                self.sig_rates[tr][self.dataset_name].update(outputs[self.output_name])

    def _cvar_lower_tail(self, values: list[float], alpha: float = 0.25) -> float:
        """Lower-tail CVaR over a set of values (e.g., per-dataset efficiencies).

        Returns mean of worst alpha-fraction.
        """
        values = list(values)
        x = torch.as_tensor(values, dtype=torch.float32)
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return float("nan")

        x_sorted, _ = torch.sort(x)
        k = max(1, math.ceil(alpha * x_sorted.numel()))
        return float(x_sorted[:k].mean().item())

    def _compute_efficiencies(self, rates: dict, target_rate: float):
        """Compute the efficiencies given a rate dictionary."""
        effs = defaultdict(float)
        clean_output_name = self.output_name.replace("/", "_")
        trate_name = str(target_rate).replace(".", "_")
        for ds_name, rate in rates[target_rate].items():
            logging_name = (
                f"val/{ds_name}/eff__{clean_output_name}__brate_{trate_name}kHz"
            )
            effs[logging_name] = rate.compute("efficiency")

        return effs

    def _get_dsname(self, rate_name: str):
        """Retrieves the data set name from string specifying rate name."""
        dataset_name = rate_name.split("/")[0]
        return dataset_name

    def _set_thres_on_module(self, pl_module, target_rate: float, thres: torch.Tensor):
        """Pass the threshold to the module so it ends up in the checkpoint."""
        target_rate = f"{target_rate}".replace(".", "_")
        name = f"thres_{target_rate}kHz"

        if name not in dict(pl_module.named_buffers()):
            pl_module.register_buffer(name, thres.detach().clone(), persistent=True)
        else:
            buf = getattr(pl_module, name)
            buf.data.copy_(thres.detach())

    def _resolve_rate_config(self, pl_module) -> tuple[list[float], float | None]:
        """Resolve target rates and base rate from module + callback config."""

        module_target = getattr(pl_module.hparams, "target_rate", None)
        module_base = getattr(pl_module.hparams, "base_rate", None)

        if module_target is None:
            raise ValueError(
                "pl_module.hparams.target_rate must be defined for AnomalyEfficiencyCallback."
            )

        # Always include module target
        rates = [float(module_target)]

        # Add extra callback rates if provided
        if self.target_rates is not None:
            rates.extend(float(r) for r in self.target_rates)

        # Deduplicate while preserving order
        seen = set()
        resolved_rates = []
        for r in rates:
            if r not in seen:
                seen.add(r)
                resolved_rates.append(r)

        # Resolve base rate (callback overrides module if provided)
        base_rate = self.base_rate if self.base_rate is not None else module_base

        return resolved_rates, base_rate
