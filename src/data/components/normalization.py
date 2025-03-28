from pathlib import Path
from dataclasses import dataclass, field

import yaml
import numpy as np

from src.utils import pylogger
from colorama import Fore, Back, Style

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataNormalizer:
    norm_scheme: str = "robust"
    norm_hyperparams: dict = field(default_factory=dict)
    cache: str = "/data/normed/"

    def fit(self, data: np.ndarray):
        """Fit the normalization to data."""
        cache_folder = Path(self.cache)
        cache_file = cache_folder / "metadata.yaml"
        if cache_file.exists():
            log.info(Fore.YELLOW + f"Normalization params exist at {cache_folder}.")
            log.info("Skipping the fitting...")
            self.norm_params = yaml.safe_load(open(cache_file, "r"))
            return

        log.info(Fore.GREEN + f"Fitting {self.norm_scheme} norm to data.")
        norm_method = getattr(self, self.norm_scheme + "_fit")
        self.norm_params = norm_method(data, **self.norm_hyperparams)

        cache_folder.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as output_file:
            log.info(f"Saving fit parameters to file {cache_file}.")
            yaml.dump(self.norm_params, output_file)

    def norm(self, data: np.ndarray, flag: str, plot: bool = True) -> np.ndarray:
        """Normalize the data using the hyperparameters from previously fitted data."""
        cache_file = Path(self.cache) / (self.norm_scheme + f"_{flag}.npy")
        if cache_file.exists():
            log.info(Fore.YELLOW + f"Normalized data exists. Loading {cache_file}.")
            return np.load(cache_file)

        log.info(Fore.GREEN + f"Normalizing data with {self.norm_scheme} norm.")
        norm_method = getattr(self, self.norm_scheme)
        data = norm_method(data)

        np.save(cache_file, data)
        log.info(f"Saved normalized data at {cache_file}.")

        return data

    def robust(self, data: np.ndarray) -> np.ndarray:
        """Robust normalization, i.e., shift by median and divide by IQ range."""
        return (data - self.norm_params["median"]) / self.norm_params["iq_range"]

    def robust_fit(self, data: np.ndarray, percentiles: list) -> None:
        """Determines the parameters for robust normalisation.

        Namely, it determines the interquantile range and the median of the data
        feature distributions in a 3D numpy array.
        """
        data_median = []
        interquantile_range = []

        for feature_idx in range(data.shape[-1]):
            data_feature = data[:, :, feature_idx].flatten()
            data_median.append(float(np.nanmedian(data_feature, axis=0)))
            quant_high, quant_low = np.nanpercentile(data_feature, percentiles)
            interquantile_range.append(float(quant_high - quant_low))

        return {"median": data_median, "iq_range": interquantile_range}
