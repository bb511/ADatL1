"""This file prepares config fixtures for other tests."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

# Tests that only compose and instantiate configs do not read the parquet data, so a
# placeholder satisfies the mandatory paths.raw_data_dir. Tests that actually train
# check this variable themselves and skip when it is unset.
RAW_DATA_DIR = os.environ.get("ADL1T_RAW_DATA_DIR", "/nonexistent/raw_data")


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.paths.raw_data_dir = RAW_DATA_DIR
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts
    a temporary logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its
    own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to
        `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# Header of every CSV in the published light-tunnel archive.
CCHAMBER_COLUMNS = (
    "timestamp,config,counter,flag,intervention,red,green,blue,osr_c,v_c,current,"
    "pol_1,pol_2,osr_angle_1,osr_angle_2,v_angle_1,v_angle_2,angle_1,angle_2,ir_1,"
    "vis_1,ir_2,vis_2,ir_3,vis_3,l_11,l_12,l_21,l_22,l_31,l_32,diode_ir_1,"
    "diode_vis_1,diode_ir_2,diode_vis_2,diode_ir_3,diode_vis_3,t_ir_1,t_vis_1,t_ir_2,"
    "t_vis_2,t_ir_3,t_vis_3,camera,v_board,v_reg"
).split(",")


def _write_cchamber_csv(
    path: Path, values: np.ndarray, columns: list[str] = CCHAMBER_COLUMNS
) -> None:
    lines = [",".join(columns)]
    for row in values:
        cells = [repr(float(v)) for v in row]
        cells[columns.index("config")] = "standard"
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def cchamber_data(tmp_path: Path) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    root = tmp_path / "causal_chamber"
    dataset_dir = root / "lt_interventions_standard_v1"
    dataset_dir.mkdir(parents=True)
    reference = rng.uniform(0, 100, (100, len(CCHAMBER_COLUMNS)))
    _write_cchamber_csv(dataset_dir / "uniform_reference.csv", reference)

    signals = ["uniform_red_mid", "uniform_t_ir_3_strong", "uniform_osr_c_weak"]
    for shift, name in enumerate(signals):
        values = rng.uniform(0, 100, (50, len(CCHAMBER_COLUMNS))) + 10.0 * shift
        if name == signals[-1]:
            # Far beyond the clip, so that every ir_1 value of this experiment is cut.
            values[:, CCHAMBER_COLUMNS.index("ir_1")] *= 1e4
        _write_cchamber_csv(dataset_dir / f"{name}.csv", values)

    return SimpleNamespace(data_dir=root, signals=signals, clipped=signals[-1])


# Header of regime_jumps_single.csv in the published wind-tunnel archive wt_walks_v1.
CCHAMBER_WT_COLUMNS = (
    "timestamp,config,counter,flag,intervention,hatch,pot_1,pot_2,osr_1,osr_2,osr_mic,"
    "osr_in,osr_out,osr_upwind,osr_downwind,osr_ambient,osr_intake,v_1,v_2,v_mic,v_in,"
    "v_out,load_in,load_out,current_in,current_out,res_in,res_out,rpm_in,rpm_out,"
    "pressure_upwind,pressure_downwind,pressure_ambient,pressure_intake,mic,signal_1,"
    "signal_2"
).split(",")

# The eleven readout columns of the published file; configs/data/cchamber_wt.yaml reads
# ten of them.
CCHAMBER_WT_READOUTS = [
    "pressure_upwind",
    "pressure_downwind",
    "pressure_ambient",
    "pressure_intake",
    "mic",
    "rpm_in",
    "rpm_out",
    "current_in",
    "current_out",
    "signal_1",
    "signal_2",
]


@pytest.fixture
def cchamber_wt_data(tmp_path: Path) -> SimpleNamespace:
    rng = np.random.default_rng(1)
    root = tmp_path / "causal_chamber"
    dataset_dir = root / "wt_walks_v1"
    dataset_dir.mkdir(parents=True)
    # Unequal, so that the split of every block is seeded differently (seed + n).
    sizes = {0: 100} | {block: 45 + 5 * block for block in range(1, 6)}
    flag = CCHAMBER_WT_COLUMNS.index("flag")
    blocks = []
    for block, n in sizes.items():
        values = rng.uniform(0, 100, (n, len(CCHAMBER_WT_COLUMNS))) + 10.0 * block
        values[:, flag] = block
        blocks.append(values)
    # Shuffled, so that a block is found by its flag and not by its position.
    table = rng.permutation(np.concatenate(blocks))
    path = dataset_dir / "regime_jumps_single.csv"
    _write_cchamber_csv(path, table, CCHAMBER_WT_COLUMNS)

    columns = [CCHAMBER_WT_COLUMNS.index(name) for name in CCHAMBER_WT_READOUTS]
    raw = {block: table[table[:, flag] == block][:, columns] for block in sizes}
    return SimpleNamespace(
        data_dir=root,
        readouts=CCHAMBER_WT_READOUTS,
        signal_blocks={"regime_01_a": 1, "regime_03_b": 3, "regime_04_c": 4},
        null_blocks={"null_regime_02": 2, "null_regime_05": 5},
        raw=raw,
    )
