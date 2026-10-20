"""The loss entropy of EntropyStop as an evaluation callback, the mirror of
src/callbacks/entropy.py over the ``train`` loader of a test loop."""

from src.callbacks.metrics import label_free
from src.evaluation.callbacks.criterion import CriterionSummary


class LossEntropy(CriterionSummary):
    """Entropy of the per-sample loss of the ``train`` loader, minimised; NaN, with a
    warning, when a loss is not positive.

    :param output_name: Key of the per-sample loss in the module's outputs.
    :param name: Callback name; None takes ``key``.
    """

    key = "entropy"
    mode = "min"
    sets = ("train",)

    def __init__(self, output_name: str = "loss/full", name: str | None = None):
        super().__init__(output_name, name)

    def values(self, pl_module, scores):
        return {self.key: label_free.loss_entropy(scores["train"])}
