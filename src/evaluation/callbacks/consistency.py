"""Evaluation callback for the consistency-only component of CAP.

Drop-in replacement for :class:`src.evaluation.callbacks.cap.CAP`: it takes
exactly the same constructor arguments, including the same ``cap_metric_config``
block, and differs only in the metric it evaluates and the name it registers
under for checkpoint selection and MLflow logging.

See :class:`src.evaluation.callbacks.metrics.cap.metric.PosteriorConsistency`
for what the metric is and why it is taken in the beta-free limit.
"""

from src.evaluation.callbacks.cap import CAP
from src.evaluation.callbacks.metrics.cap.metric import (
    PosteriorConsistency as PosteriorConsistencyMetric,
)


class PosteriorConsistency(CAP):
    """Evaluate the beta-free consistency limit of CAP between two datasets.

    The summary statistic is maximised across checkpoints, exactly as for CAP,
    so ``get_optimized_metric`` is inherited unchanged. Select it as the HPO
    objective with ``optimized_metric_config.main_metric.callback.name=consistency``.
    """

    metric_cls = PosteriorConsistencyMetric
    metric_label = "Consistency"

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
        cap_metric_config: dict,
        log_raw_mlflow: bool = True,
        name: str = "consistency",
    ):
        super().__init__(
            output_name=output_name,
            dataset_1=dataset_1,
            dataset_2=dataset_2,
            pairing_type=pairing_type,
            cap_metric_config=cap_metric_config,
            log_raw_mlflow=log_raw_mlflow,
            name=name,
        )


__all__ = ["PosteriorConsistency"]
