# Callback that computes the anomaly rate during training.
from collections import defaultdict
import math
import statistics

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningDataModule

from src.callbacks.metrics.rate import AnomalyRate
from src.data.components.normalization import L1DataNormalizer


class AnomalyEfficiencyCallback(Callback):
    """Calculates the fraction of anomalies detected given a certain bkg rate.

    :target_rate: Float specifying the target rate of bkg.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total rate of events that are processed by
        the L1 trigger.
    :materic_name: Which model metric to use as the anomaly score to calculate rate on.
    :beta: Float that sets parameter of EMA metrics compute here.
    """

    def __init__(
        self, target_rates: list[int], bc_rate: int, metric_name: str, beta: float = 0.9
    ):
        super().__init__()
        self.device = None
        self.target_rates = target_rates
        self.bc_rate = bc_rate
        self.metric_name = metric_name
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
        self.cvar25_ema = None

    def on_validation_start(self, trainer, pl_module):
        """Do checks required for this callback to work."""

        # Check if 'zerobias' dataset is the first dataset in the validation dictionary.
        # This is required to compute the rate thresholds on the anomaly scores.
        self.device = pl_module.device

        first_val_dset_key = list(trainer.val_dataloaders.keys())[0]
        if first_val_dset_key != "zerobias":
            raise ValueError("Rate callback requires zerobias first in the val dict!")

    def on_validation_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.main_rate = defaultdict(lambda: defaultdict(AnomalyRate))
        self.sig_rates = defaultdict(lambda: defaultdict(AnomalyRate))
        self.bkg_rates = defaultdict(lambda: defaultdict(AnomalyRate))
        self.zerobias_score_data = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Determine the rate for every given metric for every validation data set.

        First, the desired metrics are computed on the zerobias_data and accummulated
        across batches. The full metric distribution is used to set a threshold that
        will give a target rate or set of rates, specified by the user.
        These thresholds are then applied on all other data sets used for validation
        to determine, by using the threshold computed on zerobias, what would the rate
        be on these other data sets.
        """
        self.dataset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        self.total_batches = trainer.num_val_batches[dataloader_idx]
        _, _, _, labels = batch

        if self.dataset_name == "zerobias":
            self._accumulate_zerobias_output(outputs, batch_idx, pl_module)
        else:
            if batch_idx == 0:
                self._initialize_rate_metric(labels)
            self._compute_batch_rate(outputs, labels)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the data sets."""
        for target_rate in self.target_rates:
            main_rate = self.main_rate[target_rate]["zerobias"].compute("rate").item()
            bkg_effs = self._compute_efficiencies(self.bkg_rates, target_rate)
            sig_effs = self._compute_efficiencies(self.sig_rates, target_rate)
            cvar25 = self._cvar_lower_tail(sig_effs.values(), alpha=0.25)
            self._compute_cvar_ema(cvar25)
            min_sig_eff = min(list(sig_effs.values()))
            med_sig_eff = statistics.median(list(sig_effs.values()))
            threshold = self.main_rate[target_rate]["zerobias"].threshold.item()

            pl_module.log_dict(
                {f"val/zerobias/brate_{target_rate}kHz": main_rate}, **self.log_kwargs
            )
            pl_module.log_dict(bkg_effs, **self.log_kwargs)
            pl_module.log_dict(sig_effs, **self.log_kwargs)
            pl_module.log_dict({"val/summary/eff_cvar25": cvar25}, **self.log_kwargs)
            pl_module.log_dict({"val/summary/eff_min": min_sig_eff}, **self.log_kwargs)
            pl_module.log_dict({"val/summary/eff_med": med_sig_eff}, **self.log_kwargs)
            pl_module.log_dict(
                {"val/summary/eff_cvar25_ema": self.cvar25_ema}, **self.log_kwargs
            )
            pl_module.log_dict({f"val/zerobias/thr__brate_{target_rate}kHz": threshold})

    def _compute_cvar_ema(self, cvar25: float):
        """Compute the cvar estimated moving average."""
        if self.cvar25_ema is None:
            self.cvar25_ema = float(cvar25)
        else:
            self.cvar25_ema = self.beta * self.cvar25_ema + (1 - self.beta) * float(
                cvar25
            )

    def _accumulate_zerobias_output(self, outputs: dict, batch_idx: int, pl_module):
        """Accummulates the specified metric data across batches.

        Used if currently processing the zerobias data set, accummulate the values of
        each metric across batches. The whole metric output distribution is needed to
        set a treshold that gives a certain rate.
        """
        batch_output = outputs[self.metric_name]
        if batch_output.ndim == 0:
            batch_output = batch_output.unsqueeze(0)
        self.zerobias_score_data.append(batch_output)

        if batch_idx == self.total_batches - 1:
            self.zerobias_score_data = torch.cat(self.zerobias_score_data, dim=0)
            self._compute_zerobias_rate(pl_module)

    def _compute_zerobias_rate(self, pl_module):
        """Computes the desired rates on the main validation data set.

        This is a sanity check. The threshold computed on the zerobias and applied to the
        zerobias data should return the rate for which this threshold was computed.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.zerobias_score_data)
            rate.update(self.zerobias_score_data)
            thres = rate.threshold
            self._set_thres_on_module(pl_module, target_rate, thres)

            self.main_rate[target_rate][self.dataset_name] = rate

    def _initialize_rate_metric(self, labels: torch.Tensor):
        """Initializes the rate metric for a dataset for each given target rate.

        The anomaly rate metric object is initialised. Then, the threshold is computed
        given the zerobias metric distribution. This threshold is then used to compute
        the rate on the other data sets differing from zerobias.
        """
        for target_rate in self.target_rates:
            rate = AnomalyRate(target_rate, self.bc_rate).to(self.device)
            rate.set_threshold(self.zerobias_score_data)
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
                self.bkg_rates[tr][self.dataset_name].update(outputs[self.metric_name])
            if torch.all(labels > 0):
                self.sig_rates[tr][self.dataset_name].update(outputs[self.metric_name])

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
        clean_metric_name = self.metric_name.replace("/", "_")
        trate_name = str(target_rate).replace(".", "_")
        for ds_name, rate in rates[target_rate].items():
            logging_name = (
                f"val/{ds_name}/eff__ascore_{clean_metric_name}__brate_{trate_name}kHz"
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
