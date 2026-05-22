"""Test/demo training entrypoint."""

import gc
from typing import Optional

import hydra
from omegaconf import DictConfig

from src.train import _get_directions, train
from src.utils import extras


def _worst_for(direction: str) -> float:
    return float("inf") if direction == "minimize" else -float("inf")


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Run the production training loop with test/demo configs."""
    extras(cfg)
    metric_dict, object_dict = train(cfg)

    evaluator = object_dict.get("evaluator", None)
    metric_value = evaluator.optimized_metric if evaluator else None

    del object_dict
    del metric_dict
    gc.collect()

    if metric_value is None or (
        isinstance(metric_value, (list, tuple)) and any(v is None for v in metric_value)
    ):
        dirs = _get_directions(cfg) or ["minimize"]
        worst = tuple(_worst_for(d) for d in dirs)
        return worst[0] if len(worst) == 1 else worst

    return metric_value


if __name__ == "__main__":
    main()
