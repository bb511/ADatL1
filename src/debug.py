
from typing import Optional
import hydra
from omegaconf import DictConfig

from pytorch_lightning import LightningDataModule

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="debug.yaml")
def load_config_test(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # TEMPORAL:
    # h5_filepath = "./data/L1Ntuple_82.h5"
    h5_filepath = "./data/1.h5"

    cfg.data.background_filepath = h5_filepath
    cfg.data.signal_filepath = h5_filepath
    cfg.data.data_processor.processed_data_path = "./data"

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    import ipdb; ipdb.set_trace()
    return

if __name__ == "__main__":
    load_config_test()