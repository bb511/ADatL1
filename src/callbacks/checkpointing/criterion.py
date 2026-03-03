# Different modes on how to checkpoint a model based on the values of the metric
# that is being tracked.

import numpy as np
from collections import deque


class Criterion(object):
    """Base implementation of a criterion to decide when to checkpoint a metric.

    Checkpoint top k models that best fulfill the criterion.
    """

    def __init__(self, top_k: int = 1):
        self._validate_topk(top_k)
        self.top_k_values = np.empty(top_k)

    def check(self, metric_value: float) -> bool:
        """Check if the metric value corresponding to epoch  passes criterion."""
        raise NotImplementedError("Test against criterion should be implemented.")

    def _validate_topk(self, top_k: int):
        """Check if given topk is a positive number larger than 0."""
        if top_k <= 0:
            raise ValueError(
                f"top_k in the criterion of the checkpoint callback needs to be "
                + f"strictly larger than 0. Given value is {top_k}."
            )


class Min(Criterion):
    """Save checkpoint for minimum k values of the metric."""

    def __init__(self, top_k: int):
        super().__init__(top_k=top_k)
        self.top_k_values.fill(np.inf)
        self.name = "min"

    def check(self, metric_value: float) -> bool:
        # Get all top k values that are higher then the current value.
        mask = self.top_k_values > metric_value

        if not np.isfinite(metric_value):
            return False

        # If there are any, replace the highest of them with current value.
        if np.any(mask):
            candidate_idxs = np.where(mask)[0]
            replace_idx = candidate_idxs[np.argmax(self.top_k_values[candidate_idxs])]
            self.top_k_values[replace_idx] = metric_value
            return True

        return False


class Max(Criterion):
    """Save checkpoint for maximum k values of the metric."""

    def __init__(self, top_k: int):
        super().__init__(top_k=top_k)
        self.top_k_values.fill(-np.inf)
        self.name = "max"

    def check(self, metric_value: float) -> bool:
        # Get all top k values that are lower then the current value.
        mask = self.top_k_values < metric_value

        # If there are any, replace the lowest of them with current value.
        if np.any(mask):
            replace_idx = np.argmin(self.top_k_values * mask)
            self.top_k_values[replace_idx] = metric_value
            return True

        return False


class Stable(Criterion):
    """Save checkpoint for minimum k values of the metric, but only when stable.

    Stability criterion over a sliding window of length `patience`:
        (max(window) - min(window)) / mean(window) <= threshold

    Args:
        top_k: number of checkpoints to keep (best stable metric values).
        threshold: fractional stability bound in (0, 1), e.g. 0.05 for 5%.
        patience: window length (number of most recent epochs) for stability.
    """

    def __init__(self, top_k: int, threshold: float, patience: int):
        super().__init__(top_k=top_k)
        self.top_k_values.fill(np.inf)
        self.name = "stable"

        self.threshold = float(threshold)
        self.patience = int(patience)

        self._validate_threshold()
        self._validate_patience()

        self._window = deque(maxlen=self.patience)

    def check(self, metric_value: float) -> bool:
        metric_value = float(metric_value)
        self._window.append(metric_value)

        # Need a full window to assess stability
        if len(self._window) < self.patience:
            return False

        w = np.asarray(self._window, dtype=np.float64)
        mean_w = w.mean()

        if not np.isfinite(mean_w):
            return False

        scale = max(np.max(np.abs(w)), 1e-12)
        stability = (w.max() - w.min()) / scale
        if stability > self.threshold:
            return False

        ranking_value = mean_w
        mask = self.top_k_values > ranking_value
        if np.any(mask):
            replace_idx = np.argmax(self.top_k_values * mask)
            self.top_k_values[replace_idx] = ranking_value
            return True

        return False

    def _validate_threshold(self):
        if not (0.0 < self.threshold < 1.0):
            raise ValueError(
                "Threshold given in stability checkpoint callback needs to be "
                f"between 0 and 1. Given value is {self.threshold}"
            )

    def _validate_patience(self):
        if self.patience <= 0:
            raise ValueError(
                "Patience given in stability checkpoint callback needs to be a "
                f"number strictly larger than 0. Given value is {self.patience}."
            )


class Last(Criterion):
    """Save the last value of the metric."""

    def __init__(self, top_k: int = 1):
        self.top_k_values = np.empty(1)
        self.name = "last"

    def check(self, metric_value: float) -> bool:
        self.top_k_values[0] = metric_value
        return False
