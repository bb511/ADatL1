# Callback that computes the anomaly efficiency.
import csv
import hashlib
import json
import pickle  # nosec B403
import struct
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils
from src.evaluation.callbacks.metrics.rate import AnomalyRate
from src.plot import horizontal_bar
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class AnomalyEfficiencyCallback(Callback):
    """Calculates the fraction of anomalies detected at one or more target rates.

    The module target rate is always included and treated as the operational rate. Additional
    target rates may be passed through the callback config.

    :param target_rates: Optional extra target rates to evaluate in addition to the module target
        rate.
    :param base_rate: Optional override for the module base rate. If None, the module base_rate is
        used.
    :param output_name: Name of the output dict entry used as anomaly score.
    :param pure_thres: Whether to threshold only on events with l1bit == False.
    :param ds: List of dataset names to compute efficiencies on.
    :param cvar_summary: Fraction of worst efficiencies to average in the summary.
    :param log_raw_mlflow: Whether to log raw plots to mlflow.
    :param name: Identifier used in plot folders and summaries.
    :param operating_point_diagnostics_path: Optional sidecar CSV destination for
        validation-threshold/test-normal diagnostics.
    :param operating_point_context: Exact provenance fields copied into the sidecar row.
    :param operating_point_only: Skip all plots and the intervention-level ``values.csv``.
    :param operating_point_threshold_source: Provenance label for the restored or
        externally calibrated operational threshold.
    """

    def __init__(
        self,
        output_name: str,
        ds: list[str],
        target_rates: list[float] | None = None,
        base_rate: float | None = None,
        pure_thres: bool = False,
        cvar_summary: float = 0.25,
        log_raw_mlflow: bool = True,
        name: str = "eff",
        operating_point_diagnostics_path: str | Path | None = None,
        operating_point_context: Mapping[str, Any] | None = None,
        operating_point_only: bool = False,
        operating_point_threshold_source: str = "checkpoint_validation:thres_operational",
        operating_point_threshold_artifact: str | Path | None = None,
        operating_point_threshold_artifact_sha256: str | None = None,
        operating_point_threshold_bytes_sha256: str | None = None,
        operating_point_require_supplied_threshold: bool = False,
    ):
        super().__init__()
        self.device = None
        self.name = name

        self.output_name = output_name
        self.ds = set(ds)
        self.target_rates = None if target_rates is None else [float(x) for x in target_rates]
        self.base_rate = base_rate
        self.pure_thres = pure_thres
        self.cvar_summary = cvar_summary

        self.log_raw_mlflow = log_raw_mlflow
        self.operating_point_diagnostics_path = (
            None
            if operating_point_diagnostics_path is None
            else Path(operating_point_diagnostics_path)
        )
        self.operating_point_context = dict(operating_point_context or {})
        self.operating_point_only = bool(operating_point_only)
        self.operating_point_threshold_source = str(operating_point_threshold_source)
        self.operating_point_threshold_artifact = (
            None
            if operating_point_threshold_artifact is None
            else Path(operating_point_threshold_artifact).resolve()
        )
        self.operating_point_threshold_artifact_sha256 = (
            None
            if operating_point_threshold_artifact_sha256 is None
            else str(operating_point_threshold_artifact_sha256)
        )
        self.operating_point_threshold_bytes_sha256 = (
            None
            if operating_point_threshold_bytes_sha256 is None
            else str(operating_point_threshold_bytes_sha256)
        )
        self.operating_point_require_supplied_threshold = bool(
            operating_point_require_supplied_threshold
        )
        if not self.operating_point_threshold_source:
            raise ValueError("operating_point_threshold_source must not be empty.")
        if self.operating_point_only and self.operating_point_diagnostics_path is None:
            raise ValueError(
                "operating_point_only=True requires operating_point_diagnostics_path."
            )
        if self.operating_point_require_supplied_threshold and (
            self.operating_point_threshold_artifact is None
            or self.operating_point_threshold_artifact_sha256 is None
            or self.operating_point_threshold_bytes_sha256 is None
        ):
            raise ValueError(
                "A required supplied threshold needs artifact path, artifact SHA-256, "
                "and float32-byte SHA-256."
            )
        self.eff_summary = defaultdict(lambda: defaultdict(float))
        self.eff_min = defaultdict(lambda: defaultdict(float))
        self.eff_med = defaultdict(lambda: defaultdict(float))

    def on_test_start(self, trainer, pl_module):
        """Resolve target/base rates and verify expected dataloaders are present."""
        self.device = pl_module.device
        (
            self.target_rates_resolved,
            self.operational_rate,
            self.base_rate_resolved,
        ) = self._resolve_rate_config(pl_module)

        self.thresholds = self._get_thres(pl_module)
        if self.operating_point_require_supplied_threshold:
            artifact_bytes = self.operating_point_threshold_artifact.read_bytes()
            if (
                hashlib.sha256(artifact_bytes).hexdigest()
                != self.operating_point_threshold_artifact_sha256
            ):
                raise ValueError("Supplied threshold artifact SHA-256 differs.")
            artifact = json.loads(artifact_bytes)
            identity = artifact.get("threshold_float32")
            if not isinstance(identity, dict):
                raise ValueError("Supplied threshold artifact has no float32 identity.")
            raw = bytes.fromhex(str(identity.get("little_endian_hex", "")))
            if (
                len(raw) != 4
                or hashlib.sha256(raw).hexdigest() != self.operating_point_threshold_bytes_sha256
                or identity.get("bytes_sha256") != self.operating_point_threshold_bytes_sha256
            ):
                raise ValueError("Supplied threshold artifact byte identity differs.")
            artifact_value = struct.unpack("<f", raw)[0]
            if float(identity.get("value")) != artifact_value:
                raise ValueError("Supplied threshold artifact value differs from its bytes.")
            threshold = self.thresholds.get(self.operational_rate)
            if threshold is None:
                raise ValueError("The supplied operational threshold is missing.")
            raw = struct.pack("<f", float(torch.as_tensor(threshold).float().item()))
            observed = hashlib.sha256(raw).hexdigest()
            if observed != self.operating_point_threshold_bytes_sha256:
                raise ValueError("Operational threshold bytes differ from the supplied artifact.")

        first_test_dset_key = list(trainer.test_dataloaders.keys())[0]
        if first_test_dset_key != "normal":
            raise ValueError("Eff callback needs normal data first in the data dict!")
        if self.operating_point_only and set(trainer.test_dataloaders) != {"normal"}:
            raise ValueError(
                "operating_point_only=True requires exactly the test.normal dataloader."
            )

    def on_test_epoch_start(self, trainer, pl_module):
        """Clear the metrics dictionary at the start of the epoch."""
        self.main_rate = defaultdict(lambda: defaultdict(AnomalyRate))
        self.sig_rates = defaultdict(lambda: defaultdict(AnomalyRate))
        self.bkg_rates = defaultdict(lambda: defaultdict(AnomalyRate))
        self.normal_score_data = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Determine rates for every requested target rate on every test dataset."""
        self.dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        self.total_batches = trainer.num_test_batches[dataloader_idx]

        b = unpack_batch(batch)
        l1bit = b.l1bit
        labels = b.y

        if self.dataset_name == "normal":
            self._accumulate_normal_output(outputs, batch_idx, l1bit)
        else:
            if self.dataset_name not in self.ds:
                return
            if batch_idx == 0:
                self._initialize_rate_metric(labels)
            self._compute_batch_rate(outputs, labels)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Log the anomaly rates computed on each of the datasets."""
        if self.operating_point_diagnostics_path is not None:
            self._write_operating_point_diagnostics(pl_module)
        if self.operating_point_only:
            return

        eff_name = f"{self.name}_pure" if self.pure_thres else self.name
        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        split = trainer.split
        plot_folder = ckpts_dir / "plots" / split / ckpt_name / eff_name
        plot_folder.mkdir(parents=True, exist_ok=True)
        raw_rows = []

        for trate in self.target_rates_resolved:
            main_eff = self._compute_eff(self.main_rate, trate)
            sig_effs = self._compute_eff(self.sig_rates, trate)
            bkg_effs = self._compute_eff(self.bkg_rates, trate)
            effs = sig_effs | bkg_effs | main_eff

            ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt_name)
            sig_data = np.fromiter(sig_effs.values(), dtype=float)
            if sig_data.size:
                self.eff_med[trate][ckpt_ds] = float(np.median(sig_data))
                self.eff_min[trate][ckpt_ds] = float(np.min(sig_data))
            else:
                self.eff_med[trate][ckpt_ds] = 0.0
                self.eff_min[trate][ckpt_ds] = 0.0

            trate_label = self._target_label(trate)
            ascore = f"anomaly score: {self.output_name}"
            xlabel = f"efficiency at threshold: {trate_label}\n{ascore}"
            self._plot(effs, xlabel, plot_folder, percent=True)
            self._store_summary(sig_effs, bkg_effs, ckpt_name, trate)
            raw_rows.extend(
                {
                    "checkpoint": ckpt_name,
                    "intervention": intervention,
                    "metric": f"efficiency_{self._target_label(trate)}",
                    "value": value,
                }
                for intervention, value in sorted(sig_effs.items())
            )

        utils.misc.write_metric_values(plot_folder / "values.csv", raw_rows)

        utils.mlflow.log_plots_to_mlflow(
            trainer,
            ckpt_name,
            eff_name,
            plot_folder,
            log_raw=self.log_raw_mlflow,
            gallery_name=f"{eff_name}",
        )

    def plot_summary(self, trainer, root_folder: Path):
        """Plot summary metrics accumulated across checkpoints and reset state."""
        eff_name = f"{self.name}_pure" if self.pure_thres else self.name
        split = trainer.split
        plot_folder = root_folder / "plots" / split / (eff_name + "_summary")
        plot_folder.mkdir(parents=True, exist_ok=True)
        self._cache_summary(plot_folder)

        for target_rate in self.eff_summary.keys():
            smet = self.eff_summary[target_rate]
            trate_label = self._target_label(target_rate)
            ascore = f"anomaly score: {self.output_name}"

            xlabel = f"25% CVaR at threshold: {trate_label}\n{ascore}"
            self._plot(smet, xlabel, plot_folder, True)

            xlabel = f"med eff at: {trate_label}\n{ascore}"
            self._plot(self.eff_med[target_rate], xlabel, plot_folder, True)

            xlabel = f"min eff at: {trate_label}\n{ascore}"
            self._plot(self.eff_min[target_rate], xlabel, plot_folder, True)

        utils.mlflow.log_plots_to_mlflow(trainer, None, eff_name, plot_folder)

    def clear_crit_summary(self):
        """Clear accumulated checkpoint-level efficiency summaries."""
        self.eff_summary.clear()
        self.eff_med.clear()
        self.eff_min.clear()

    def get_optimized_metric(self, target_rate: float | None = None):
        """Return the best summary metric across checkpoints for one target rate.

        If target_rate is None, the operational rate is used.
        """
        if target_rate is None:
            target_rate = self.operational_rate

        if target_rate not in self.eff_summary:
            raise ValueError(
                f"Efficiency callback for metric {self.output_name} did not calculate "
                f"eff at target rate {target_rate}. Choose {list(self.eff_summary.keys())}."
            )

        metric_across_ckpts = self.eff_summary[target_rate]
        max_ckpt_name = max(metric_across_ckpts, key=metric_across_ckpts.get)
        max_metric_value = 1e3 * metric_across_ckpts[max_ckpt_name]
        return max_ckpt_name, max_metric_value

    def _plot(self, data: dict, xlabel: str, plot_folder: Path, percent: bool = False):
        """Plot the efficiency per dataset for an anomaly metric at target rate."""
        ylabel = " "
        horizontal_bar.plot_yright(data, data, xlabel, ylabel, plot_folder, percent)

    def _store_summary(self, sig_effs: dict, bkg_effs: dict, ckpt: str, trate: float):
        """Store the summary statistic for one checkpoint and one target rate."""
        sig_data = np.fromiter(sig_effs.values(), dtype=float)
        bkg_data = np.fromiter(bkg_effs.values(), dtype=float)
        bkg_mean = bkg_data.mean() if bkg_data.size else 0.0

        alpha = self.cvar_summary
        if sig_data.size:
            k = max(1, int(np.ceil(alpha * sig_data.size)))
            worst = np.partition(sig_data, k - 1)[:k]
            sig_cvar = worst.mean()
        else:
            sig_cvar = 0.0

        summary_metric = sig_cvar - bkg_mean

        ckpt_ds = utils.misc.get_ckpt_ds_name(ckpt)
        self.eff_summary[trate][ckpt_ds] = float(summary_metric)

    def _accumulate_normal_output(self, outputs: dict, batch_idx: int, l1bit: torch.Tensor | None):
        """Accumulate the main normal score distribution across batches."""
        batch_output = outputs[self.output_name]
        if batch_output.ndim == 0:
            batch_output = batch_output.unsqueeze(0)

        if self.pure_thres:
            if l1bit is None:
                raise ValueError("pure_thres=True requires l1bit to be present in the batch.")
            batch_output = batch_output[~l1bit]

        self.normal_score_data.append(batch_output)

        if batch_idx == self.total_batches - 1:
            self.normal_score_data = torch.cat(self.normal_score_data, dim=0)
            self._compute_normal_rate()

    def _compute_normal_rate(self):
        """Compute the requested rates on the normal test dataset."""
        for target_rate in self.target_rates_resolved:
            rate = AnomalyRate(target_rate, self.base_rate_resolved).to(self.device)
            rate.apply_threshold(self.thresholds[target_rate])
            rate.update(self.normal_score_data)
            self.main_rate[target_rate][self.dataset_name] = rate

    def _initialize_rate_metric(self, labels: torch.Tensor):
        """Initialize the rate metric for a signal/background dataset."""
        for target_rate in self.target_rates_resolved:
            rate = AnomalyRate(target_rate, self.base_rate_resolved).to(self.device)
            rate.apply_threshold(self.thresholds[target_rate])
            if torch.all(labels < 0):
                self.bkg_rates[target_rate][self.dataset_name] = rate
            if torch.all(labels > 0):
                self.sig_rates[target_rate][self.dataset_name] = rate

    def _compute_batch_rate(self, outputs: dict, labels: torch.Tensor):
        """Update the thresholded rate for one batch."""
        for tr in self.target_rates_resolved:
            if torch.all(labels < 0):
                self.bkg_rates[tr][self.dataset_name].update(outputs[self.output_name])
            if torch.all(labels > 0):
                self.sig_rates[tr][self.dataset_name].update(outputs[self.output_name])

    def _compute_eff(self, rates: dict, target_rate: float):
        """Compute efficiencies from a given rate dictionary."""
        effs = defaultdict(float)
        for ds_name, rate in rates[target_rate].items():
            effs[ds_name] = rate.compute("efficiency").item()
        return effs

    def _cache_summary(self, cache_folder: Path):
        """Cache the summary metric dictionary."""
        with open(cache_folder / "summary.pkl", "wb") as f:
            plain_dict = utils.misc.to_plain_dict(self.eff_summary)
            pickle.dump(plain_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_thres(self, pl_module):
        """Load thresholds that were stored on the module at validation time."""
        thresholds = defaultdict(float)
        valid_rates = []

        for target_rate in self.target_rates_resolved:
            # --- NEW: operational handling ---
            if self._is_operational(target_rate):
                thres = getattr(pl_module, "thres_operational", None)
                trate_name = "operational"
            else:
                trate_name = str(target_rate).replace(".", "_")
                thres = getattr(pl_module, f"thres_{trate_name}kHz", None)

            if thres is None:
                log.warn(
                    f"No threshold was set for target rate {trate_name}. "
                    "Skipping efficiency computation for this threshold."
                )
                continue

            thresholds[target_rate] = thres
            valid_rates.append(target_rate)

        self.target_rates_resolved = valid_rates
        return thresholds

    def _resolve_rate_config(self, pl_module) -> tuple[list[float], float, float | None]:
        """Resolve target rates and base rate from module + callback config."""
        module_target = getattr(pl_module.hparams, "target_rate", None)
        module_base = getattr(pl_module.hparams, "base_rate", None)

        if module_target is None:
            raise ValueError(
                "pl_module.hparams.target_rate must be defined for AnomalyEfficiencyCallback."
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

    def _is_operational(self, target_rate: float) -> bool:
        """Return whether a target rate is the module operational rate."""
        return abs(target_rate - self.operational_rate) < 1e-12

    def _target_label(self, target_rate: float) -> str:
        """Return the stable output label for one target rate."""
        if self._is_operational(target_rate):
            return "operational"
        return str(target_rate)

    def _write_operating_point_diagnostics(self, pl_module) -> None:
        """Write one provenance-rich, test-normal operating-point row.

        This is deliberately separate from the intervention-level ``values.csv``
        output contract.  The operational threshold is supplied by checkpoint
        restoration or a validation-only calibration; it is never fit on test data.
        """
        if self.base_rate_resolved is not None:
            raise ValueError(
                "Operating-point diagnostics currently require a direct false-positive rate "
                "(base_rate=None)."
            )
        if abs(self.operational_rate - 0.01) > 1e-12:
            raise ValueError(
                "Operating-point diagnostics require target false-positive rate=0.01, got "
                f"{self.operational_rate}."
            )
        if set(self.main_rate[self.operational_rate]) != {"normal"}:
            raise ValueError("Operating-point diagnostics require exactly the test.normal stream.")

        rate = self.main_rate[self.operational_rate]["normal"]
        n_test_normal = int(rate.nsamples.detach().cpu().item())
        n_triggered = int(rate.ntriggered.detach().cpu().item())
        if n_test_normal <= 0:
            raise ValueError("Operating-point diagnostics received no test-normal samples.")
        score = torch.as_tensor(self.normal_score_data)
        n_finite = int(torch.isfinite(score).sum().detach().cpu().item())
        if n_finite != n_test_normal:
            raise ValueError(
                "Operating-point diagnostics require finite test-normal scores; "
                f"found {n_finite}/{n_test_normal}."
            )

        checkpoint = Path(pl_module._ckpt_path).resolve()
        threshold = float(rate.threshold.detach().cpu().item())
        if not np.isfinite(threshold):
            raise ValueError("Operational threshold is not finite.")
        row: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "checkpoint_stem": checkpoint.stem,
            "threshold_source": self.operating_point_threshold_source,
            "validation_derived_threshold": threshold,
            "target_fpr": 0.01,
            "test_normal_count": n_test_normal,
            "test_normal_finite_count": n_finite,
            "triggered_count": n_triggered,
            "achieved_test_normal_acceptance": n_triggered / n_test_normal,
            "nominal_test_normal_acceptance": 0.01,
            "finite_sample_granularity": 1.0 / n_test_normal,
            "nearest_attainable_test_normal_acceptance": (
                round(0.01 * n_test_normal) / n_test_normal
            ),
        }
        if self.operating_point_threshold_artifact is not None:
            row["threshold_artifact"] = str(self.operating_point_threshold_artifact)
            row["threshold_artifact_sha256"] = self.operating_point_threshold_artifact_sha256
            row["threshold_bytes_sha256"] = self.operating_point_threshold_bytes_sha256
        for key, value in sorted(self.operating_point_context.items()):
            if key in row:
                raise ValueError(f"Operating-point context collides with field {key!r}.")
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            elif value is None or isinstance(value, (str, int, float, bool)):
                row[key] = value
            else:
                raise TypeError(f"Operating-point context field {key!r} is not CSV-serializable.")

        output = self.operating_point_diagnostics_path
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            temporary = Path(handle.name)
        temporary.replace(output)
