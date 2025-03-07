from pathlib import Path
from dataclasses import dataclass

import yaml
import numpy as np

from src.utils import pylogger
from colorama import Fore, Back, Style

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataNormalizer:
    def __init__(
        self,
        norm_scheme: str = "robust",
        norm_fit_hyperparams: dict = None,
        cache_folder: str  = "/data/robust_norm",
        hyperparams_path: str = None
    ):
        """Normalization schemes for the L1AD data.

        :param norm_scheme: Selected normalization scheme. Defautls to "robust".
        :param norm_fit_hyperparams: Dictionary with hyperparameters used in
            determining the normalisation quantities. For example, the percentiles for
            robust normalisation. Defaults to 'None'.
        :param cache_folder: Path to where to cache the normalized data.
            Defaults to "/data/norm".
        :param hyperparams_path: Path to a yaml file containing the
        """
        self.norm_scheme = norm_scheme
        self.norm_fit_hyperparams = norm_fit_hyperparams
        self.cache_folder = Path(cache_folder)
        self.hyperparams = None

        if hyperparams_path:
            with open(hyperparams_path, 'r') as hyperparams_file:
                self.hyperparams = yaml.safe_load(hyperparams_file)

        self.cache_folder.mkdir(parents=True, exist_ok=True)

    def normalize(self, data: np.ndarray, filename: str, fit: bool = False):
        """Normalize the data given a certain normalization scheme.

        :param data: Numpy array of the data to normalize.
        :param filename: String specifying the name of the file with normalized data.
            If this file does not exist, it is created.
        :param fit: Bool of whether to deduce the hyperparameters of the normalization
            scheme from the data. If False, hyperparameters of the normalization are
            used if present in the class self.hyperparams or they are loaded if
            hyperparams_path file is provided. Defaults to False.
        """
        self.cache_file = self.cache_folder / filename
        if self.cache_file.exists():
            log.info(f"Normalized data exists. Loading {self.cache_file}.")
            return np.load(self.cache_file)

        log.info(Back.GREEN + f"Normalizing {filename} using {self.norm_scheme}.")
        if fit:
            self._fit_norm(data)

        if not self.hyperparams:
            raise ValueError("Normalization not fitted and also hp file not provided!")

        with open(self.cache_folder / "norm_hps.yaml", 'w') as output_file:
            log.info(f"Hps of normalization saved to {self.cache_folder}.")
            yaml.dump(self.hyperparams, output_file)

        norm_method = getattr(self, self.norm_scheme)
        data = norm_method(data)
        np.save(self.cache_file, data)

        return data

    def _fit_norm(self, data: np.ndarray) -> None:
        """Fits the normalization to the data, obtaining the corresp statistics."""
        log.info(f"Fitting {self.norm_scheme} normalization to train data...")
        norm_fit = getattr(self, self.norm_scheme + "_fit")
        norm_fit(data, **self.norm_fit_hyperparams)

    def robust(self, data: np.ndarray) -> np.ndarray:
        """Robust normalization, i.e., shift by median and divide by IQ range."""
        return (data - self.hyperparams["median"])/self.hyperparams["iq_range"]

    def robust_fit(self, data: np.ndarray, percentiles: list) -> None:
        """Gets the parameters for robust normalisation.

        Namely, it determines the interquantile range and the median of the data
        feature distributions.
        """
        data_median = []
        interquantile_range = []

        for feature_idx in range(data.shape[-1]):
            data_feature = data[:, :, feature_idx].flatten()
            data_median.append(np.nanmedian(data_feature, axis=0))
            quant_high, quant_low = np.nanpercentile(data_feature, percentiles)
            interquantile_range.append(quant_high - quant_low)

        self.hyperparams = {"median": data_median, "iq_range": interquantile_range}
