"""The SDS error as an evaluation callback, the mirror of src/callbacks/sds.py over the
``sds_targets`` and ``sds_outliers`` loaders of a test loop.

The cut is ``pl_module.thres_operational`` with ``pl_module.tie_operational``, the
threshold and tie weight set on the epoch's validation normal scores at the module's
target rate.
"""

import torch

from src.callbacks.metrics import label_free
from src.evaluation.callbacks.criterion import CriterionSummary


class SDS(CriterionSummary):
    """SDS error of the pseudo-sets at the operational cut, minimised."""

    key = "sds"
    mode = "min"
    sets = ("sds_targets", "sds_outliers")

    def values(self, pl_module, scores):
        threshold = torch.as_tensor(pl_module.thres_operational).item()
        tie_weight = torch.as_tensor(pl_module.tie_operational).item()
        a_targets, a_outliers = scores["sds_targets"], scores["sds_outliers"]
        error = label_free.sds_error(a_targets, a_outliers, threshold, tie_weight)
        return {self.key: error}
