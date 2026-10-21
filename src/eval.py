# Offline evaluation of one lattice trained in the dump regime, then the selection.
import os

os.environ["KERAS_BACKEND"] = "torch"

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Add resolvers to evaluate operations in the .yaml configuration files
from src.utils.omegaconf import register_resolvers

register_resolvers()

from src.evaluation.lattice import select_lattice
from src.utils import RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate every run of ``cfg.lattice`` at every epoch, then select on it.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)

    log.info(f"Instantiating evaluator <{cfg.evaluation.evaluator._target_}>")
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)

    log.info(f"Evaluating the runs of {cfg.scores_dir}/{cfg.lattice}")
    evaluator.evaluate_lattice(
        scores_dir=cfg.scores_dir,
        out_dir=cfg.out_dir,
        lattice=cfg.lattice,
        data_dir=cfg.data_dir,
        sets=cfg.sets,
        workers=cfg.workers,
        tracking_uri=cfg.tracking_uri,
    )

    log.info(f"Selecting on the lattice into {cfg.out_dir}")
    select_lattice(
        out_dir=cfg.out_dir,
        selection=OmegaConf.to_container(cfg.selection, resolve=True),
        scores_dir=cfg.scores_dir,
        lattice=cfg.lattice,
        partner=cfg.partner,
    )


if __name__ == "__main__":
    main()
