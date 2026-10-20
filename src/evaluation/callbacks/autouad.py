"""AutoUAD's criteria as evaluation callbacks: NPD, RTM and EAG, the mirrors of
src/callbacks/autouad.py over the named loaders of a test loop."""

from src.callbacks.metrics import label_free
from src.evaluation.callbacks.criterion import CriterionSummary


class NPD(CriterionSummary):
    """Normalised pseudo discrepancy of ``normal`` against ``autouad``, maximised."""

    key = "npd"
    mode = "max"
    sets = ("normal", "autouad")

    def values(self, pl_module, scores):
        return {self.key: label_free.npd(scores["normal"], scores["autouad"])}


class RTM(CriterionSummary):
    """Relative top-median of ``normal``, maximised."""

    key = "rtm"
    mode = "max"

    def values(self, pl_module, scores):
        return {self.key: label_free.rtm(scores["normal"])}


class EAG(CriterionSummary):
    """Expected anomaly gap of ``normal``, maximised."""

    key = "eag"
    mode = "max"

    def values(self, pl_module, scores):
        return {self.key: label_free.eag(scores["normal"])}
