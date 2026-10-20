"""Goix's criteria as evaluation callbacks: excess mass and mass volume, the mirrors of
src/callbacks/goix.py over the named loaders of a test loop.

The volume of the uniform box comes from ``pl_module.aux_lower`` and
``pl_module.aux_upper``, the per-feature bounds of the training inputs that the
``uniform`` loader is drawn over.
"""

from src.callbacks.metrics import label_free
from src.evaluation.callbacks.criterion import CriterionSummary


def _volume(pl_module) -> float:
    return label_free.box_volume(pl_module.aux_lower.cpu(), pl_module.aux_upper.cpu())


class ExcessMass(CriterionSummary):
    """Excess-mass area of ``normal`` against ``uniform``, maximised; also logs
    ``em_reached``, 1 when the area reached the level t_max."""

    key = "em"
    mode = "max"
    sets = ("normal", "uniform")

    def values(self, pl_module, scores):
        area, reached = label_free.excess_mass(
            scores["normal"], scores["uniform"], _volume(pl_module)
        )
        return {self.key: area, f"{self.key}_reached": float(reached)}


class MassVolume(CriterionSummary):
    """Mass-volume area of ``normal`` against ``uniform``, minimised."""

    key = "mv"
    mode = "min"
    sets = ("normal", "uniform")

    def values(self, pl_module, scores):
        area = label_free.mass_volume(
            scores["normal"], scores["uniform"], _volume(pl_module)
        )
        return {self.key: area}
