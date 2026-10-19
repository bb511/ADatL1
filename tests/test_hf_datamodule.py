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


def test_composing_the_record_leaves_the_running_hydra_alone(tmp_path):
    # A naive composition would leave the application without the Hydra it started with.
    record = _mini_record(tmp_path)
    with initialize(version_base="1.3", config_path="../configs"):
        before = compose(config_name="train.yaml", overrides=["data=basis_hf"])
        composed = compose_record_config(str(record), [])
        after = compose(config_name="train.yaml", overrides=["data=basis_hf"])

    assert composed.marker == "record"
    assert after.data._target_ == before.data._target_


def test_basis_hf_instantiates_without_reaching_the_hub(tmp_path):
    # Building the datamodule must not download anything; setup does that.
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=basis_hf", f"paths.root_dir={tmp_path}"],
        )
    datamodule = instantiate(cfg.data)

    assert datamodule.record is None and datamodule.loader is None
    assert datamodule.hparams.repo_id == "podagiu/anomaly_detection_cmsl1t"
    assert datamodule.hparams.cache_dir == f"{tmp_path}/data/hf"
