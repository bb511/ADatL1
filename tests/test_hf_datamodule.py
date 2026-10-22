"""The record's configuration tree must compose without disturbing the project's own,
and its training split must reach the callbacks as L1ADDataModule's does."""

from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from src.data.L1AD_datamodule import SplitTensors
from src.data.L1AD_HF_datamodule import compose_record_config


def _mini_record(tmp_path: Path) -> Path:
    """A stand-in for the record: a root holding nothing but a config tree."""
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "config.yaml").write_text("marker: record\nvalue: 1\n")

    return tmp_path


def _stand_in_pipeline(n_events: int = 64) -> SimpleNamespace:
    """A stand-in for the record's pipeline, handing back the dtypes and the 39 by 3
    input of its basis configuration, with no auxiliary sets."""
    generator = torch.Generator().manual_seed(0)

    def load(split: str) -> SplitTensors:
        x = torch.randn(n_events, 39, 3, generator=generator)
        mask = torch.rand(n_events, 39, 3, generator=generator) > 0.5
        l1bit = torch.rand(n_events, generator=generator) > 0.5
        y = torch.zeros(n_events, dtype=torch.int64)
        return SplitTensors(x=x, mask=mask, l1bit=l1bit, y=y)

    return SimpleNamespace(load=load, load_aux=lambda split: {})


def _catalogued_config(tmp_path: Path):
    # The catalogue's data=basis_hf wins over the experiment's override /data: basis.
    with initialize(version_base="1.3", config_path="../configs"):
        return compose(
            config_name="train.yaml",
            overrides=[
                "data=basis_hf",
                "experiment=physics/ae",
                f"paths.root_dir={tmp_path}",
            ],
        )


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
    assert datamodule.hparams.repo_id == "CERN/anomaly_detection_cmsl1t"
    # The cache is keyed by the pinned commit, so a bump cannot reuse a stale tree.
    revision = datamodule.hparams.revision[:8]
    assert datamodule.hparams.cache_dir == f"{tmp_path}/data/hf/{revision}"


def test_data_train_is_the_loaded_training_split(tmp_path):
    datamodule = instantiate(_catalogued_config(tmp_path).data)
    datamodule.record = _stand_in_pipeline()
    assert datamodule.data_train is None

    datamodule.setup("fit")
    loaded = datamodule._main["train"]
    tensors = datamodule.data_train.tensors

    expected = (loaded.x, loaded.mask, loaded.l1bit, loaded.y)
    assert all(a is b for a, b in zip(tensors, expected, strict=True))
    datamodule.teardown("fit")
    assert datamodule.data_train is None


def test_criteria_reading_data_train_start_on_the_record(tmp_path):
    cfg = _catalogued_config(tmp_path)
    datamodule = instantiate(cfg.data)
    datamodule.record = _stand_in_pipeline()
    datamodule.setup("fit")
    trainer = SimpleNamespace(datamodule=datamodule)

    for name in ("npd", "loss_entropy", "sds"):
        instantiate(cfg.callbacks[name]).on_fit_start(trainer, None)
