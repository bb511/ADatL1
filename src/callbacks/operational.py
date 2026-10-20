"""The operational score Q'' of validation normal as a strategy callback.

The logged key, the EMA and the skipped sanity pass are those of
src.callbacks.criterion.CriterionCallback.
"""

from src.callbacks.criterion import CriterionCallback
from src.callbacks.metrics.selection import EMA_WEIGHT


class OperationalScoreCallback(CriterionCallback):
    """The EMA of the operational score Q'' of validation normal.

    The value is the mean over the batches of ``ascore/operational``, which the module
    logs itself as ``val/normal/ascore_operational``; only the EMA is logged here, under
    ``val/summary/<name>_ema``.
    """

    name = "ascore_operational"

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/operational",
        ema_weight: float = EMA_WEIGHT,
    ):
        super().__init__(name, output_name, ema_weight)

    def criterion(self, trainer, pl_module) -> float:
        return self.normal().double().mean().item()

    def logged(self, value: float) -> dict[str, float]:
        return {f"val/summary/{self.name}_ema": self.ema}
