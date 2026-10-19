import pickle
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
from colorama import Fore

from src.utils import pylogger

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
        norm_method = getattr(self, "_" + self.name + "_fit")
        if self.hyperparams:
            norm_method(data, **self.hyperparams)
        else:
            norm_method(data)

    def norm(self, data: ak.Array, obj_name: str) -> np.ndarray:
        """Normalize the data using the hyperparameters from previously fitted data."""
        norm_method = getattr(self, "_" + self.name)
        data = norm_method(data, obj_name)
        return data

    def _unnormalized(self, data: ak.Array, obj_name: str) -> ak.Array:
        return data

    def _unnormalized_fit(self, data: ak.Array):
        self.norm_params[self.obj_name] = {}
        for feat in ak.fields(data):
            self.norm_params[self.obj_name][feat] = {"shift": 0, "scale": 1}

    def _affine(self, data: ak.Array, obj_name: str) -> ak.Array:
        """Shift and scale each feature by its previously fitted norm params."""
        result = data
        params = self.norm_params[obj_name]
        for feature in ak.fields(data):
            normed_feature = data[feature] - params[feature]["shift"]
            normed_feature = normed_feature / params[feature]["scale"]
            result = ak.with_field(result, normed_feature, where=feature)

        return result

    def _robust(self, data: ak.Array, obj_name: str) -> ak.Array:
        """Robust normalization: shift by the median, divide by the quantile range."""
        return self._affine(data, obj_name)

    def _robust_fit(self, data: ak.Array, percentiles: list):
        """Determines the parameters for robust normalisation.

        For each feature, the shift is the median and the scale is the range between the
        two quantiles given in percentiles, e.g. 0.05 and 0.95.
        """
        self.norm_params[self.obj_name] = {}
        for feat in ak.fields(data):
            feature_data = ak.to_numpy(ak.flatten(data[feat]))

            median = np.median(feature_data)
            qlow, qhigh = np.quantile(feature_data, percentiles)
            scale = qhigh - qlow
            scale = scale if scale != 0 else 1e-12
            self.norm_params[self.obj_name][feat] = {"shift": median, "scale": scale}

    def _standard(self, data: ak.Array, obj_name: str) -> ak.Array:
        """Apply train-split mean and standard-deviation normalization."""
        return self._affine(data, obj_name)

    def _standard_fit(self, data: ak.Array):
        """Fit feature-wise standard scaling on active training objects only."""
        self.norm_params[self.obj_name] = {}
        for feat in ak.fields(data):
            feature_data = ak.to_numpy(ak.flatten(data[feat]))
            mean = float(np.mean(feature_data))
            scale = float(np.std(feature_data))
            if not np.isfinite(mean) or not np.isfinite(scale):
                raise ValueError(
                    f"Non-finite standard normalization parameters for "
                    f"{self.obj_name}/{feat}."
                )
            if scale <= 0.0:
                scale = 1e-12
            self.norm_params[self.obj_name][feat] = {
                "shift": mean,
                "scale": scale,
            }

    def _robust_axov4_fit(self, data: ak.Array, percentiles: list, scale: list):
        """Determines the parameters for the special kind of robust norm in axov4.

        This is similar to robust normalization, except that the interquantile range is
        rescaled by scale_width = scale[0] - scale[1]. Additionally, a shift is applied
        to each distribution, as defined below. This is to make the interquantile range
        given by the user fit within the domain provided in the scale list, e.g.,
        scale = [2, -2] means that the interquantile range would fit within -2 and 2.
        """
        self.norm_params[self.obj_name] = {}

        scale_width = scale[0] - scale[1]
        for feat in ak.fields(data):
            feature_data = ak.to_numpy(ak.flatten(data[feat]))
            qlow, qhigh = np.quantile(feature_data, percentiles)
            scaled_iqrange = (qhigh - qlow) / scale_width
            shift = (qlow * scale[0] - qhigh * scale[1]) / scale_width
            self.norm_params[self.obj_name][feat] = {
                "shift": shift,
                "scale": scaled_iqrange,
            }

    def _robust_axov4(self, data: ak.Array, obj_name: str):
        """Similar to robust normaliztion, applied to the training of axov4 and v5."""
        return self._affine(data, obj_name)

    def import_norm_params(self, norm_filepath: Path, obj_name: str):
        """Import normalization parameters from a pkl file."""
        if not norm_filepath.is_file():
            raise FileNotFoundError(f"Norm param file not found at {norm_filepath}!")
        if norm_filepath.suffix != ".pkl":
            raise ValueError(
                f"Norm params can only be exported to .pkl! "
                f"Given file path to export to: {norm_filepath}"
            )

        with norm_filepath.open("rb") as params_file:
            self.norm_params[obj_name] = pickle.load(params_file)

    def export_norm_params(self, norm_filepath: Path, obj_name: str):
        """Export normalization parameters from a pkl file."""
        if norm_filepath.suffix != ".pkl":
            raise ValueError(
                f"Norm params can only be exported to .pkl! "
                f"Given file path to export to: {norm_filepath}"
            )

        with norm_filepath.open("wb") as params_file:
            pickle.dump(self.norm_params[obj_name], params_file)
