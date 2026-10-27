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
    for domain in ("physics", "cifar10", "mscct", "cchamber", "cchamber_wt")
    for path in (CONFIGS / "experiment" / domain).glob("*.yaml")
)
# The LHC keeps the checkpoint regime; every other domain trains and dumps its scores.
PHYSICS = [e for e in EXPERIMENTS if e.startswith("physics/")]
DUMPED = [e for e in EXPERIMENTS if not e.startswith("physics/")]
MODEL_CHECKPOINT = "pytorch_lightning.callbacks.ModelCheckpoint"
SCORE_DUMP = "src.callbacks.score_dump.ScoreDumpCallback"


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


def _saved_folders(callbacks: dict) -> dict[tuple[str, ...], dict]:
    # The evaluator reads <run>/<strategy>/<metric>/<criterion>; the last-epoch
    # checkpoint sits in <run> itself and is never evaluated.
    folders = {}
    for cb in callbacks.values():
        if isinstance(cb, dict) and cb.get("_target_") == MODEL_CHECKPOINT:
            below_run = cb["dirpath"].split("${run_name}")[1].strip("/")
            if below_run:
                folders[tuple(below_run.split("/"))] = cb
    return folders


@pytest.mark.parametrize("experiment", PHYSICS)
def test_every_saved_checkpoint_is_evaluated(experiments, experiment):
    cfg = experiments[experiment]
    ckpts = cfg["evaluation"]["evaluator"]["ckpts"]
    candidate = cfg["optimized_metric_config"]["candidate"]
    assert "single" not in ckpts
    assert "last" not in ckpts

    listed = {
        (strategy, metric, criterion)
        for strategy, metrics in ckpts.items()
        for metric, criteria in metrics.items()
        for criterion in criteria
    }
    saved = _saved_folders(cfg["callbacks"])

    chosen = (candidate["strategy"], candidate["metric"], candidate["criterion"])
    assert chosen in listed
    assert listed == saved.keys()
    for (_, metric, criterion), cb in saved.items():
        assert cb["mode"] == criterion
        assert cb["filename"] == metric


@pytest.mark.parametrize("experiment", DUMPED)
def test_dumped_runs_train_and_dump_only(experiments, experiment):
    cfg = experiments[experiment]
    callbacks = cfg["callbacks"]
    domain, model = experiment.split("/")

    targets = [cb.get("_target_") for cb in callbacks.values() if isinstance(cb, dict)]
    assert MODEL_CHECKPOINT not in targets
    assert not cfg.get("evaluation")
    assert "optimized_metric_config" not in cfg
    assert callbacks["score_dump"]["_target_"] == SCORE_DUMP
    assert cfg["trainer"]["num_sanity_val_steps"] == -1
    assert cfg["trainer"]["enable_checkpointing"] is False
    assert cfg["test"] is False

    stored_sets = ["normal", "reference_normal"] if domain == "cchamber_wt" else None
    assert callbacks["score_dump"]["stored_sets"] == stored_sets
    assert not {"initial_sets", "train_pass"} & callbacks["score_dump"].keys()
    residual = domain.startswith("cchamber") and model in ("ae", "vae")
    assert ("residual_state" in callbacks) == residual


@pytest.mark.parametrize("experiment", PHYSICS)
def test_physics_runs_dump_the_weights_and_the_probes(experiments, experiment):
    cfg = experiments[experiment]
    config = {**cfg["callbacks"]["score_dump"], "dirpath": "d", "run_dir": "r"}
    dump = hydra.utils.instantiate(config)
    npd = hydra.utils.instantiate(cfg["callbacks"]["npd"])
    assert config["_target_"] == SCORE_DUMP
    assert cfg["trainer"]["num_sanity_val_steps"] == -1
    assert dump.keys == ["ascore/full", "fit/full"]
    assert dump.stored_sets == []
    assert dump.initial_sets == ["normal", "SingleNeutrino_E-10-gun"]
    assert dump.train_pass is False
    assert (dump.aux_seed, dump.aux_size) == (npd.aux_seed, npd.aux_size)
    assert dump.aux_batch_size == npd.batch_size


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_cap_parameters_come_from_the_cap_group(experiments, experiment):
    cfg = experiments[experiment]
    evaluation_callbacks = cfg.get("evaluation", {}).get("callbacks", {})
    blocks = [
        cb["cap_metric_config"]
        for group in (cfg["callbacks"], evaluation_callbacks)
        for cb in group.values()
        if isinstance(cb, dict) and "cap_metric_config" in cb
    ]

    assert blocks == ["${cap}"] * (4 if experiment in PHYSICS else 0)
    assert {"beta_min", "beta_max", "n_beta"} <= cfg["cap"].keys()


# checkpoint key: (callback target, mode, name of the block under callbacks)
CRITERIA = {
    "npd": ("src.callbacks.autouad.NPDCallback", "max", "npd"),
    "rtm": ("src.callbacks.autouad.RTMCallback", "max", "rtm"),
    "eag": ("src.callbacks.autouad.EAGCallback", "max", "eag"),
    "entropy": ("src.callbacks.entropy.LossEntropyCallback", "min", "loss_entropy"),
    "sds_286_088064": ("src.callbacks.sds.SDSCallback", "min", "sds"),
    "sds_val_286_088064": ("src.callbacks.sds.SDSCallback", "min", "sds_val"),
    "fit": ("src.callbacks.operational.FitCallback", "min", "fit"),
}
# Criteria left out per experiment. A negative log-likelihood can be negative: entropy
# is then undefined, and the sign of RTM meaningless. The training losses of SVDD reach
# 0, where entropy is undefined too.
SKIPPED = {"physics/realnvp": {"rtm", "entropy"}, "physics/svdd": {"entropy"}}
GOIX = (
    "src.callbacks.goix.ExcessMassCallback",
    "src.callbacks.goix.MassVolumeCallback",
)


@pytest.mark.parametrize("experiment", PHYSICS)
def test_label_free_criteria_are_logged_checkpointed_and_evaluated(
    experiments, experiment
):
    cfg = experiments[experiment]
    callbacks = cfg["callbacks"]
    saved = {cb["monitor"]: cb for cb in _saved_folders(callbacks).values()}
    summary = cfg["evaluation"]["evaluator"]["ckpts"]["summary"]
    skipped = SKIPPED.get(experiment, set())
    for key, (target, mode, block) in CRITERIA.items():
        if key in skipped:
            assert block not in callbacks and f"{key}_ema" not in summary
            continue
        assert callbacks[block]["_target_"] == target
        assert saved[f"val/summary/{key}_ema"]["mode"] == mode
        assert summary[f"{key}_ema"] == [mode]
    assert len(saved) == 17 - len(skipped)
    targets = {cb.get("_target_") for cb in callbacks.values() if isinstance(cb, dict)}
    assert not targets & set(GOIX)
    for block in ("sds", "sds_val"):
        sds = hydra.utils.instantiate(callbacks[block])
        assert f"val/summary/{sds.name}_ema" in saved
    assert callbacks["sds_val"]["source"] == "normal"
    assert callbacks["sds"]["source"] == "train"
    assert not {"npd", "sds"} & cfg["evaluation"]["callbacks"].keys()
