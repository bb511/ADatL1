# Different modes on how to checkpoint a model based on the values of the metric
# that is being tracked.

import numpy as np


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
                f"top_k in the criterion of the checkpoint callback needs to be " +
                f"strictly larger than 0. Given value is {top_k}."
            )

class Min(Criterion):
    """Save checkpoint for minimum k values of the metric."""
    def __init__(self, top_k: int):
        super().__init__(top_k=top_k)
        self.top_k_values.fill(np.inf)
        self.name = 'min'

    def check(self, metric_value: float) -> bool:
        # Get all top k values that are higher then the current value.
        mask = self.top_k_values > metric_value

        # If there are any, replace the highest of them with current value.
        if np.any(mask):
            replace_idx = np.argmax(self.top_k_values * mask)
            self.top_k_values[replace_idx] = metric_value
            return True

        return False


class Max(Criterion):
    """Save checkpoint for maximum k values of the metric."""
    def __init__(self, top_k: int):
        super().__init__(top_k=top_k)
        self.top_k_values.fill(-np.inf)
        self.name = 'max'

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
    """Save k most stable values of the metric.

    :threshold: Float between 0 and 1, specifying the fraction up to which the metric
        is allowed to vary between epochs to pass this criterion.
    :patience: How many epochs to track the stability across, i.e., for how many epochs
        should the metric stay within threshold.
    """
    def __init__(self, top_k: int, threshold: float, patience: int):
        super().__init__(top_k=top_k)
        self.top_k_values.fill(np.inf)
        self.name = 'stable'

        self.threshold = threshold
        self.patience = patience
        self.reference_value = None
        self.accummulated_deviation = 0

        self.epoch_counter = 1

        self._validate_threshold()
        self._validate_patience()

    def check(self, metric_value: float) -> bool:
        if self.reference_value is None:
            self.reference_value = metric_value
            return True

        lower_bound = self.reference_value * (1 - self.threshold)
        upper_bound = self.reference_value * (1 + self.threshold)

        if lower_bound <= metric_value <= upper_bound:
            self._increase_count(metric_value)
            if self.epoch_counter == self.patience:
                self._update_topk(self.accummulated_deviation)
                self._reset_count()
        else:
            self._reset_count()

        return False

    def _increase_count(self, metric_value: float):
        """Increase the count for how many epochs the tracked metric has been stable."""
        self.epoch_counter += 1
        self.accummulated_deviation += (metric_value - self.reference_value)**2

    def _reset_count(self):
        """Reset the counting of stable epochs."""
        self.accummulated_deviation = 0
        self.epoch_counter = 1
        self.reference_value = None

    def _update_topk(self, metric_value: float):
        """Update the top_k values."""
        mask = self.top_k_values > metric_value

        # If there are any, replace the highest of them with current value.
        if np.any(mask):
            replace_idx = np.argmax(self.top_k_values * mask)
            self.top_k_values[replace_idx] = metric_value

    def _validate_threshold(self):
        """Check if the threshold given by user is between 0 and 1."""
        below_zero = self.threshold <= 0
        above_one = self.threshold >= 1
        if below_zero or above_one:
            raise ValueError(
                f"Threshold given in stability checkpoint callback needs to be " +
                f"between 0 and 1. Given value is {self.threshold}"
            )

    def _validate_patience(self):
        """Check if the given patience is a positive integer."""
        if self.patience <= 0:
            raise ValueError(
                f"Patience given in stability checkpoint callback needs to be a " +
                f"number strictly larger than 0. Given value is {self.patience}."
            )
