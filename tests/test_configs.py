from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.utils.omegaconf import register_resolvers

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = sorted(
    f"{domain}/{path.stem}"
    for domain in ("physics", "cifar10", "robustad")
    for path in (CONFIGS / "experiment" / domain).glob("*.yaml")
)


@pytest.fixture(scope="module", autouse=True)
def _resolvers() -> None:
    # Importing src.train, as test_train.py does at collection, registers them too,
    # and a second registration raises.
    if not OmegaConf.has_resolver("short_hash"):
        register_resolvers()


def test_train_config(cfg_train: DictConfig) -> None:
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.algorithm
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.algorithm)
    hydra.utils.instantiate(cfg_train.trainer)


@pytest.fixture(scope="module")
def experiments() -> dict[str, dict]:
    overrides = ["paths.raw_data_dir=/x", "run_name=dbg"]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        return {
            e: OmegaConf.to_container(
                compose(config_name="train", overrides=[f"experiment={e}", *overrides])
            )
            for e in EXPERIMENTS
        }


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_only_the_candidate_is_evaluated(experiments, experiment):
    cfg = experiments[experiment]
    ckpts = cfg["evaluation"]["evaluator"]["ckpts"]
    candidate = cfg["optimized_metric_config"]["candidate"]

    assert candidate["criterion"] in ckpts[candidate["strategy"]][candidate["metric"]]
    assert "single" not in ckpts
    assert "last" not in ckpts
    assert ckpts == {
        candidate["strategy"]: {candidate["metric"]: [candidate["criterion"]]}
    }
