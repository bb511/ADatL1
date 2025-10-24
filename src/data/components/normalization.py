from pathlib import Path
from dataclasses import dataclass

import numpy as np
import awkward as ak
from colorama import Fore, Back, Style
from pytorch_lightning import loggers

from src.utils import pylogger
from . import plots

log = pylogger.RankedLogger(__name__)


@dataclass
class L1DataNormalizer:
    name: str
    hyperparams: dict

    def __post_init__(self):
        self.norm_params = {}

    def fit(self, data: ak.Array, obj_name: str) -> None:
        """Fit the normalization to data to determine normalization params."""
        log.info(Fore.BLUE + f"Fitting {self.name} norm to {obj_name} object...")

        self.obj_name = obj_name
        norm_method = getattr(self, '_' + self.name + '_fit')
        if self.hyperparams:
            norm_method(data, **self.hyperparams)
        else:
            norm_method(data)

    def norm(self, data: ak.Array, obj_name: str) -> np.ndarray:
        """Normalize the data using the hyperparameters from previously fitted data."""
        norm_method = getattr(self, '_' + self.name)
        data = norm_method(data, obj_name)

        return data

    def _unnormalized(self, data: ak.Array, obj_name: str) -> ak.Array:
        return data

    def _unnormalized_fit(self, data: ak.Array, obj_name: str):
        return None

    def _robust(self, data: ak.Array, obj_name: str) -> ak.Array:
        """Robust normalization, i.e., shift by median and divide by IQ range."""
        result = data
        params = self.norm_params[obj_name]
        for feature in ak.fields(data):
            normed_feature = data[feature] - params[feature]['median']
            normed_feature = normed_feature/params[feature]['scale']
            result = ak.with_field(result, normed_feature, where=feature)

        return result

    def _robust_fit(self, data: ak.Array, percentiles: list):
        """Determines the parameters for robust normalisation.

        Namely, it determines the interquantile (iqr) range and the median of the data
        for each feature distribution separately.
        """
        self.norm_params[self.obj_name] = {}
        for feat in ak.fields(data):
            feature_data = ak.to_numpy(ak.flatten(data[feat]))

            median = np.median(feature_data)
            qlow, qhigh = np.quantile(feature_data, percentiles)
            scale = qhigh - qlow
            scale = scale if scale != 0 else 1e-12
            self.norm_params[self.obj_name][feat] = {
                "median": median,
                "scale": scale
            }

    # def changrobust_fit(self, data: np.ndarray, percentiles: list, scale: list) -> dict:
    #     """Determines the parameters for changrobust normalization.

    #     This is Chang's (previous student) version of robust normalization and used
    #     in the v5 version of the level 1 trigger anomaly detector to process the data.

    #     :data: 3D numpy array of the data to be normalised.
    #     :percentiles: List of the two percentiles between which the interquantile range
    #         should be determined. The larger percentile should come first, followed
    #         by the smaller percentile.
    #     :scale: List of two floats which are used to determine the distribution shift
    #         and also used to scale the interquantile range. The larger float should
    #         come first, followed by the smaller float.
    #     """
    #     bias = []
    #     interquantile_range = []

    #     scale_larger = scale[0]
    #     scale_smaller = scale[1]
    #     scale_width = scale_larger - scale_smaller

    #     if self.ignore_zeros:
    #         # Ignore constituents with extremely small pT (basically 0).
    #         mask = data[:, :, 0] > 0.000001
    #         data = data[mask]

    #     for feature_idx in range(data.shape[-1]):
    #         data_feature = data[..., feature_idx].flatten()
    #         quant_high, quant_low = np.nanpercentile(data_feature, percentiles)
    #         scaled_iq = (quant_high - quant_low) / scale_width
    #         interquantile_range.append(scaled_iq)
    #         chang_shift = (
    #             quant_low * scale_larger - quant_high * scale_smaller
    #         ) / scale_width
    #         bias.append(chang_shift)

    #     bias = np.array(bias, dtype=np.float32)
    #     interquantile_range = np.array(interquantile_range, dtype=np.float32)

    #     return {"bias": bias, "scaled_iq_range": interquantile_range}

    # def changrobust(self, data: np.ndarray):
    #     """Changrobust normalization.

    #     Shift by nubmer based on interquantile range and some given user range.
    #     Scale by interquantile range divided by same given user range as above.
    #     """
    #     data = (data - self.norm_params["bias"]) / self.norm_params["scaled_iq_range"]
        # return data.astype(self.output_dtype)

    # def _add_plots_to_mlflow(self, logs: loggers, plots_path: Path):
    #     """Adds the plots of the normalized data to the mlflow experiment."""
    #     for logger in logs:
    #         if isinstance(logger, loggers.mlflow.MLFlowLogger):
    #             logger.experiment.log_artifact(logger.run_id, plots_path)
