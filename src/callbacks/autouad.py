"""AutoUAD's criteria as training callbacks: NPD, RTM and EAG (Dai and Fan, ICLR 2025).

The logged keys, the EMA and the skipped sanity pass are those of
src.callbacks.criterion.CriterionCallback; the criteria are the functions of
src.callbacks.metrics.label_free. All three are maximised.
"""

from src.callbacks.criterion import AuxSetCallback, CriterionCallback
from src.callbacks.metrics import label_free


class NPDCallback(AuxSetCallback):
    """Normalised pseudo discrepancy of validation normal against the ``autouad`` set,
    N(mu, diag sigma^2) with the per-feature moments of the training inputs."""

    name = "npd"
    aux_set = "autouad"

    def criterion(self, trainer, pl_module) -> float:
        return label_free.npd(self.normal(), self.aux_scores(trainer, pl_module))


class RTMCallback(CriterionCallback):
    """Relative top-median of validation normal, ``label_free.rtm``."""

    name = "rtm"

    def criterion(self, trainer, pl_module) -> float:
        return label_free.rtm(self.normal())


class EAGCallback(CriterionCallback):
    """Expected anomaly gap of validation normal, ``label_free.eag``."""

    name = "eag"

    def criterion(self, trainer, pl_module) -> float:
        return label_free.eag(self.normal())
