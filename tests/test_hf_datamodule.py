"""The record's configuration tree must compose without disturbing the project's own."""

from pathlib import Path

from hydra import compose, initialize
from hydra.utils import instantiate

from src.data.L1AD_HF_datamodule import compose_record_config


def _mini_record(tmp_path: Path) -> Path:
    """A stand-in for the record: a root holding nothing but a config tree."""
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "config.yaml").write_text("marker: record\nvalue: 1\n")

    return tmp_path


def test_record_config_composes(tmp_path):
    """The record is composed from its own directory, not from this project's."""
    composed = compose_record_config(str(_mini_record(tmp_path)), [])

    assert composed.marker == "record"


def test_record_overrides_reach_the_record(tmp_path):
    composed = compose_record_config(str(_mini_record(tmp_path)), ["value=2"])

    assert composed.value == 2


def test_composing_the_record_leaves_the_running_hydra_alone(tmp_path):
    """The project's own tree must still compose after the record's has been read.

    initialize_config_dir refuses to run while a global Hydra is live and restores only
    what it saw at its own entry, so a naive composition would leave the application
    without the Hydra it started with.
    """
    record = _mini_record(tmp_path)
    with initialize(version_base="1.3", config_path="../configs"):
        before = compose(config_name="train.yaml", overrides=["data=basis_hf"])
        composed = compose_record_config(str(record), [])
        after = compose(config_name="train.yaml", overrides=["data=basis_hf"])

    assert composed.marker == "record"
    assert after.data._target_ == before.data._target_


def test_basis_hf_instantiates_without_reaching_the_hub(tmp_path):
    """Building the datamodule must not download anything; setup does that."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=basis_hf", f"paths.root_dir={tmp_path}"],
        )
    datamodule = instantiate(cfg.data)

    assert datamodule.record is None and datamodule.loader is None
    assert datamodule.hparams.repo_id == "podagiu/anomaly_detection_cmsl1t"
    assert datamodule.hparams.cache_dir == f"{tmp_path}/data/hf"


def _instantiated(tmp_path, *overrides):
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=basis_hf", f"paths.root_dir={tmp_path}", *overrides],
        )

    return instantiate(cfg.data)


def test_a_normalizer_swap_reaches_the_record(tmp_path):
    """The dte experiments swap the normalizer, and the record has to hear about it."""
    datamodule = _instantiated(tmp_path, "data/data_normalizer=standard")

    assert "data/data_normalizer=standard" in datamodule._record_overrides("/record")


def test_the_record_is_asked_for_the_name_it_knows(tmp_path):
    """Two schemes are called something else in the record than they are here."""
    datamodule = _instantiated(tmp_path, "data/data_normalizer=axov4")

    assert "data/data_normalizer=axov4" in datamodule._record_overrides("/record")


def test_record_overrides_are_passed_as_plain_strings(tmp_path):
    """Whatever else the config asks for has to survive as a hydra override."""
    datamodule = _instantiated(
        tmp_path, 'data.overrides=["data/data_awkward2torch=minimal"]'
    )
    overrides = datamodule._record_overrides("/record")

    assert overrides[0] == "paths.root_dir=/record"
    assert overrides[-1] == "data/data_awkward2torch=minimal"
    assert all(isinstance(o, str) for o in overrides)
