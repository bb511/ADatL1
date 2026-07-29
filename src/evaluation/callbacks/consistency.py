"""Drop-in evaluation callback for CAP's posterior-consistency component."""

from src.callbacks.metrics.cap.metric import PosteriorConsistency as ConsistencyMetric
from src.evaluation.callbacks.cap import CAP


class PosteriorConsistency(CAP):
    """Evaluate posterior consistency using exactly the CAP callback contract."""

    metric_cls = ConsistencyMetric


__all__ = ["PosteriorConsistency"]
