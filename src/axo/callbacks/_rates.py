# Computes the rates given by the AD algorithm with respect to certain signals.
# Anomaly thresholds for raw rates of 1 kHz, 5 kHz, and  10 kHz are 1192, 904, and
# 724 respectively.

import numpy as np
import tensorflow as tf
from tensorflow import keras


class AnomalyRates(keras.callbacks.Callback):
    """Calculates the rate of a given signal at the end of each min loss epoch.

    This is the number of events captured by the AD trigger but not L1, divided by all
    signal events.

    Args:
        threshold: anomaly detection threshold.
        raw_rate: the raw rate at which the anomaly detection model is tested.
        bc_rate: the average bunch crossing rate at the LHC; for the current runs it is
            28.61 MHz.
    """

    def __init__(self, raw_rate: int, threshold: int, bc_rate: float = 28.61):
        super().__init__()
        self.threshold = threshold
        self.raw_rate = raw_rate
        self.bunch_crossing_rate = bunch_crossing_rate

    def on_train_begin(self, logs=None):
        self.loss_best = np.inf

    def on_epoch_end(self, epoch: int, signal_data: np.ndarray, logs=None):
        loss_current = logs.get("loss")
        if not np.less(loss_current, self.loss_best):
            return

        self.loss_best = loss_current
        model_pred = self.model(signal_data)
        model_pred = model_pred[model_pred > self.threshold]
        leve1_pred = self.level1_criteria(signal_data)

        raw_rate = model_pred.shape[0] / signal_data.shape[0]
        pure_rate = np.setxor1d(model_pred, leve1_pred).shape[0] / signal_data.shape[0]

        print(f"Raw rate: {raw_rate}.")
        print(f"Pure rate: {pure_rate}.")
        print(f"Efficiency: {pure_rate/raw_rate}")

    def level1_criteria(self, signal_data: np.ndarray):
        """Applies current level1 trigger cuts to the data."""
        # Fill in here the level 1 cuts applied currently.
        return

    def convert_to_fpr(self, rate: float):
        """Converts a raw rate to false positive rate, the rates are in MHz."""
        return rate / self.bc_rate
