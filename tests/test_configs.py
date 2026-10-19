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
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
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


def _values_under(node, key: str):
    if isinstance(node, dict):
        for k, value in node.items():
            if k == key:
                yield value
            yield from _values_under(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from _values_under(value, key)


def test_every_domain_is_covered():
    assert {e.split("/")[0] for e in EXPERIMENTS} == {"physics", "cifar10", "robustad"}
    assert len(EXPERIMENTS) == 34


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


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_no_stable_per_signal_or_reco_callbacks(experiments, experiment):
    cfg = experiments[experiment]
    targets = [
        str(node.get("_target_"))
        for group in (cfg["callbacks"], cfg["evaluation"]["callbacks"])
        for node in group.values()
        if isinstance(node, dict)
    ]

    assert not any(t.endswith("SingleDatasetModelCheckpoint") for t in targets)
    assert not any(t.endswith("ReconstructionPlots") for t in targets)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_cap_blocks_carry_the_beta_sweep(experiments, experiment):
    blocks = list(_values_under(experiments[experiment], "cap_metric_config"))

    assert len(blocks) == (4 if experiment.endswith("_agnostic") else 0)
    for block in blocks:
        assert "beta0" not in block
        assert {"beta_min", "beta_max", "n_beta"} <= block.keys()
