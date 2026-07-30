"""Training callback for the consistency-only component of CAP.

Drop-in replacement for :class:`src.callbacks.cap.CAPCallback`: it takes exactly
the same constructor arguments, including the same ``cap_metric_config`` block,
and differs only in the metric it evaluates and the key it logs under.

See :class:`src.callbacks.metrics.cap.metric.PosteriorConsistency` for what the
metric is and why it is taken in the beta-free limit.
"""

from src.callbacks.cap import CAPCallback
from src.callbacks.metrics.cap.metric import PosteriorConsistency


class PosteriorConsistencyCallback(CAPCallback):
    """Log the beta-free consistency limit of CAP between two datasets.

    Reuses the whole CAP callback contract -- score collection, pairing, EMA
    smoothing and logging -- so an experiment config can carry this alongside
    the CAP callback without interfering with it. The logged key is
    ``val/summary/consistency_ema_{dataset_1}_vs_{dataset_2}`` and, like CAP, it
    is maximised.
    """

    metric_cls = PosteriorConsistency
    metric_key = "consistency"


__all__ = ["PosteriorConsistencyCallback"]
