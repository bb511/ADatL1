from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import itertools

import hydra
import pytorch_lightning as pl

import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.omegaconf import register_resolvers
register_resolvers()

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


from src.utils.checkpoints import find_scan_checkpoints
from capmetric import ApproximationCapacity


@task_wrapper
def eval(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates the model on test data and the stored checkpoints.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)


    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "algorithm": algorithm,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    log.info("Finding checkpoints...")
    filter_groups = lambda dirname: not Path(dirname).stem.startswith("cap")
    checkpoint_dict = find_scan_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        scan=cfg.scan,
        filter_groups=filter_groups,
        filter_checkpoint=None,
        by_combination=False,
    )
    for (epoch, step), ckpt_paths in checkpoint_dict.items():
        # Select the checkpoint and evaluate:
        ckpt_path = ckpt_paths[0]
        trainer.test(model=algorithm, datamodule=datamodule, ckpt_path=ckpt_path)
        
        metric_dict = trainer.callback_metrics
        """
        Metrics obtained here:
        - main_test/loss/...: loss terms with the zerobias test split
        - main_test/...: latent space metrics with the zerobias test split
        - cap/test/baseline: CAP(zerobias, zerobias) for the test split
        - [when signals are also split] cap/test/background0, cap/test/background1
        """

    # BUG @Patrick: This should not be needed, signal datasets should also appear on test:
    datamodule.trainer = trainer
    datamodule.setup("fit")
    dataset_dictionary = datamodule.val_dataloader() # BUG @Patrick: should I reinstantiate datasets?
    
    log.info("Cacheing the data before CAP computation...")
    filter_groups = lambda dirname: (not Path(dirname).stem.startswith("cap")) and (not Path(dirname).stem.startswith("main"))
    checkpoint_dict = find_scan_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        scan=cfg.scan,
        filter_groups=filter_groups,
        filter_checkpoint=None,
        by_combination=True,
    )
    cache = {}
    for prefix in sorted({k[1] for k in checkpoint_dict.keys()}):
        for metric_name in sorted({k[2] for k in checkpoint_dict.keys()}):            
            for dset_key in sorted({k[0] for k in checkpoint_dict.keys()}):
                device = "cuda" if torch.cuda.is_available() else "cpu"

                ckpt_paths = checkpoint_dict.get((dset_key, prefix, metric_name), [])
                if len(ckpt_paths) == 0:
                    continue

                if len(ckpt_paths) > 1:
                    log.warning(
                        f"Multiple checkpoints found for combination "
                        f"(dset_key={dset_key}, prefix={prefix}, metric_name={metric_name}). "
                        f"Using the first one: {ckpt_paths[0]}"
                    )
                ckpt_path = ckpt_paths[0]

                # Load the logits in memory:
                algorithm = hydra.utils.instantiate(cfg.algorithm)
                state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")[
                    "state_dict"
                ]
                algorithm.load_state_dict(state_dict, strict=True)
                algorithm.eval()
                algorithm.to(device)

                ds = dataset_dictionary.get(dset_key)
                with torch.no_grad():
                    logits = []
                    for ibatch in range(len(ds)):
                        batch = ds[ibatch]
                        batch = tuple(t.to(device=device, dtype=torch.float32) for t in batch)
                        output = algorithm.model_step(batch)[cfg.callbacks.cap_callback.output_name]
                        logits.append(output.detach().cpu())

                    cache[(dset_key, prefix, metric_name)] = torch.cat(logits, dim=0)

                del ds, batch, algorithm, state_dict

    results = {}
    for prefix in sorted({k[1] for k in cache}):
        for metric_name in sorted({k[2] for k in cache}):
            # Load existing dataset keys:
            dset_keys = {k[0] for k in cache if k[1] == prefix and k[2] == metric_name}
            for dset_key0, dset_key1 in itertools.combinations(sorted(dset_keys), 2):
                # Load the appropiate logits:
                logits0 = cache.get((dset_key0, prefix, metric_name), torch.empty(0))
                logits1 = cache.get((dset_key1, prefix, metric_name), torch.empty(0))
                if len(logits0) == 0 or len(logits1) == 0:
                    continue

                # Compute CAPmetric
                capmetric = ApproximationCapacity(
                    beta0=1.0,
                    normalization_type="minmax",
                    normalization_params=None,
                    energy_type="baseline",
                    energy_params=None,
                    regularization_type="none",
                    regularization_params=None,
                    binary=True,
                    lr=0.01,
                    n_epochs=50,
                    batch_size=cfg.data.batch_size,
                    device=device,
                    process_group=None,
                    dist_sync_fn=None,
                )
                capmetric.update(
                    logits0.to(device=device),
                    logits1.to(device=device),
                )
                cap = capmetric.compute()
                results[(dset_key, prefix, metric_name)] = cap

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = eval(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
