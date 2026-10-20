"""Goix's criteria as training callbacks: excess mass (maximised) and mass volume
(minimised), N. Goix, arXiv:1607.01152.

The logged keys, the EMA and the skipped sanity pass are those of
src.callbacks.criterion.CriterionCallback; the criteria are the functions of
src.callbacks.metrics.label_free. Both read the ``uniform`` set of the score dump,
uniform over the per-feature bounds of the training inputs, and are NaN, with a
warning, when the volume of that box is not finite and positive, as for the 117 physics
features.
"""

from src.callbacks.criterion import AuxSetCallback
from src.callbacks.metrics import label_free


class _UniformBoxCallback(AuxSetCallback):
    aux_set = "uniform"

    def volume(self) -> float:
        return label_free.box_volume(self.aux["aux_lower"], self.aux["aux_upper"])


class ExcessMassCallback(_UniformBoxCallback):
    """Excess-mass area of validation normal, ``label_free.excess_mass``."""

    name = "em"

    def criterion(self, trainer, pl_module) -> float:
        uniform = self.aux_scores(trainer, pl_module)
        return label_free.excess_mass(self.normal(), uniform, self.volume())[0]


class MassVolumeCallback(_UniformBoxCallback):
    """Mass-volume area of validation normal, ``label_free.mass_volume``."""

    name = "mv"

    def criterion(self, trainer, pl_module) -> float:
        uniform = self.aux_scores(trainer, pl_module)
        return label_free.mass_volume(self.normal(), uniform, self.volume())
