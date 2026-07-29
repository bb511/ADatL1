"""Drop-in training callback for CAP's posterior-consistency component."""

from src.callbacks.cap import CAPCallback
from src.callbacks.metrics.cap.metric import PosteriorConsistency


class PosteriorConsistencyCallback(CAPCallback):
    """Compute log posterior cosine similarity using the CAP callback contract.

    This class accepts exactly the same constructor arguments as ``CAPCallback``.
    Replacing only the Hydra ``_target_`` therefore preserves existing metric
    names, checkpoint monitors, pairing, EMA handling, and CAP configuration.
    The reported value is the log-cosine component evaluated at the inverse
    temperature selected by the complete CAP objective.
    """

    metric_cls = PosteriorConsistency


__all__ = ["PosteriorConsistencyCallback"]
