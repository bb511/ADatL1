from pathlib import Path
from dataclasses import dataclass, field

import yaml
import numpy as np
from colorama import Fore, Back, Style
from pytorch_lightning import loggers

from src.utils import pylogger
from . import plots

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataNormalizer:
    norm_scheme: str = "robust"
    norm_hyperparams: dict = field(default_factory=dict)
    ignore_zeros: bool = False
    processed_data_folder: str = "data/processed/default"
    cache: str = "data/normed/default"

    def fit(self, data: np.ndarray) -> None:
        """Fit the normalization to data."""
        cache_folder = Path(self.cache) / self.norm_scheme
        fit_file = cache_folder / "fit_params.yaml"

        if self._check_fit_exists(fit_file):
            return

        log.info(Fore.GREEN + f"Fitting {self.norm_scheme} norm to data.")
        norm_method = getattr(self, self.norm_scheme + "_fit")
        self.norm_params = norm_method(data, **self.norm_hyperparams)

        self._save_fit_params(cache_folder, fit_file)

    def norm(self, data: np.ndarray, dataset_name: str, logs: loggers) -> np.ndarray:
        """Normalize the data using the hyperparameters from previously fitted data.

        :data: 3D numpy array containing the data to normalise.
        :dataset_name: String describing what kind of data to normalise, i.e., 'train',
            'test', 'val', or it can be the name of the data set.
        :logs: Logger object for pytorch_lightning, such that the normalized data
            plots are logged as artifacts in whatever MLOps client you use.
        """
        cache_file = Path(self.cache) / self.norm_scheme / f"{dataset_name}.npy"
        plots_path = Path(self.cache) / self.norm_scheme / dataset_name
        if self._check_cache_exists(cache_file):
            self._add_plots_to_mlflow(logs, plots_path)
            return np.load(cache_file)

        norm_method = getattr(self, self.norm_scheme)
        data = norm_method(data)
        self._plot_normed_data(data, dataset_name)
        self._add_plots_to_mlflow(logs, plots_path)
        self._cache_normed_data(cache_file, data)

        return data

    def robust(self, data: np.ndarray) -> np.ndarray:
        """Robust normalization, i.e., shift by median and divide by IQ range."""
        return (data - self.norm_params["median"]) / self.norm_params["iq_range"]

    def robust_fit(self, data: np.ndarray, percentiles: list) -> dict:
        """Determines the parameters for robust normalisation.

        Namely, it determines the interquantile range and the median of the data
        feature distributions in a 3D numpy array.

        :data: 3D numpy array of the data to be normalised.
        :percentiles: List of the two percentiles between which the interquantile range
            should be determined. The larger percentile should come first, followed
            by the smaller percentile.
        """
        data_median = []
        interquantile_range = []

        for feature_idx in range(data.shape[-1]):
            data_feature = data[..., feature_idx].flatten()
            if self.ignore_zeros:
                data_feature = data_feature[data_feature > 0.000001]
            data_median.append(float(np.nanmedian(data_feature, axis=0)))
            quant_high, quant_low = np.nanpercentile(data_feature, percentiles)
            interquantile_range.append(float(quant_high - quant_low))

        return {"median": data_median, "iq_range": interquantile_range}

    def changrobust_fit(self, data: np.ndarray, percentiles: list, scale: list) -> dict:
        """Determines the parameters for changrobust normalization.

        This is Chang's (previous student) version of robust normalization and used
        in the v5 version of the level 1 trigger anomaly detector to process the data.

        :data: 3D numpy array of the data to be normalised.
        :percentiles: List of the two percentiles between which the interquantile range
            should be determined. The larger percentile should come first, followed
            by the smaller percentile.
        :scale: List of two floats which are used to determine the distribution shift
            and also used to scale the interquantile range. The larger float should
            come first, followed by the smaller float.
        """
        bias = []
        interquantile_range = []

        scale_larger = scale[0]
        scale_smaller = scale[1]
        scale_width = scale_larger - scale_smaller

        if self.ignore_zeros:
            # Ignore constituents with extremely small pT (basically 0).
            mask = data[:, :, 0] > 0.000001
            data = data[mask]

        for feature_idx in range(data.shape[-1]):
            data_feature = data[..., feature_idx].flatten()
            quant_high, quant_low = np.nanpercentile(data_feature, percentiles)
            scaled_iq = (quant_high - quant_low) / scale_width
            interquantile_range.append(float(scaled_iq))
            chang_shift = (quant_low * scale_larger - quant_high * scale_smaller) / scale_width
            bias.append(float(chang_shift))

        return {"bias": bias, "scaled_iq_range": interquantile_range}

    def changrobust(self, data: np.ndarray):
        """Changrobust normalization.

        Shift by nubmer based on interquantile range and some given user range.
        Scale by interquantile range divided by same given user range as above.
        """
        return (data - self.norm_params["bias"]) / self.norm_params["scaled_iq_range"]

    def _check_fit_exists(self, fit_file: Path) -> bool:
        """Checks if the data has already been fitted for given normalization."""
        if fit_file.exists():
            log.info(Fore.YELLOW + f"Normalization params exist at {fit_file}.")
            log.info("Skipping the fitting...")
            self.norm_params = yaml.safe_load(open(fit_file, "r"))
            return True
        return False

    def _save_fit_params(self, cache_folder: Path, fit_file: Path) -> None:
        """Save the normalization parameters to a file."""
        cache_folder.mkdir(parents=True, exist_ok=True)
        with open(fit_file, "w") as output_file:
            log.info(f"Saving fit parameters to file {fit_file}.")
            yaml.dump(self.norm_params, output_file)

    def _check_cache_exists(self, cache_file: Path) -> bool:
        """Checks if file with cached normalized data exists already."""
        if cache_file.exists():
            log.info(Fore.YELLOW + f"Normalized data exists. Loading {cache_file}.")
            return True

        return False

    def _cache_normed_data(self, cache_file: Path, data: np.ndarray) -> None:
        """Save the normalized data to disk for faster loading if ran again."""
        np.save(cache_file, data)
        log.info(f"Saved normalized data at {cache_file}.")

    def _plot_normed_data(self, data: np.ndarray, dataset_name: str):
        """Plot the normalised data file using the metadata info from processed file."""
        self.processed_data_folder = Path(self.processed_data_folder)
        metadata = yaml.safe_load(open(self.processed_data_folder / "metadata.yaml"))
        plot_folder = Path(self.cache) / self.norm_scheme / dataset_name

        obj_starting_idx = 0
        for obj_name in metadata.keys():
            obj_ending_idx = obj_starting_idx + metadata[obj_name]["const"]
            plots.plot_hist(
                data[:, obj_starting_idx:obj_ending_idx],
                obj_name,
                metadata[obj_name]["feats"],
                plot_folder,
            )
            obj_starting_idx = obj_ending_idx

    def _add_plots_to_mlflow(self, logs: loggers, plots_path: Path):
        """Adds the plots of the normalized data to the mlflow experiment."""
        for logger in logs:
            if isinstance(logger, loggers.mlflow.MLFlowLogger):
                logger.experiment.log_artifact(logger.run_id, plots_path)
