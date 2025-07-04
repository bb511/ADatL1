from typing import Any, Dict, List
from pathlib import Path
from collections import defaultdict

import hydra
import rootutils
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import torch
from pytorch_lightning.utilities.memory import garbage_collection_cuda

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, task_wrapper, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)

from capmetric import ApproximationCapacity


def find_all_checkpoints(base_path: str) -> Dict[str, List[str]]:
    """Find all checkpoint files organized by dataset key.
    
    Args:
        base_path: Base directory containing checkpoint subdirectories
        
    Returns:
        Dict mapping dataset keys to lists of checkpoint paths
    """
    base_path = Path(base_path)
    checkpoints = {}
    
    # Look for subdirectories (dataset keys)
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            dataset_key = subdir.name
            # Find all .ckpt files in this subdirectory
            ckpt_files = list(subdir.glob("*.ckpt"))
            if ckpt_files:
                checkpoints[dataset_key] = [str(f) for f in ckpt_files]
                log.info(f"Found {len(ckpt_files)} checkpoints for dataset '{dataset_key}'")
    
    return checkpoints


@task_wrapper
def evaluate_checkpoints(cfg: DictConfig) -> Dict[str, Any]:
    """Evaluate CAPmetric on all checkpoints in the given path.

    Args:
        cfg: DictConfig configuration composed by Hydra.
        
    Returns:
        Dict with results for each checkpoint
    """
    if cfg.get("experiment_id"):
        log.info(f"Resuming experiment <{cfg.resume_experiment_id}>")
        cfg.logger.wandb.id = cfg.resume_experiment_id
        cfg.logger.wandb.resume = "must"
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=[], logger=logger
    )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    batch_size = cfg.data.batch_size
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.batch_size_per_device = batch_size
    datamodule.trainer = trainer
    datamodule.prepare_data()
    datamodule.setup("fit")
    dataset_dictionary = datamodule.val_dataloader()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    log.info(f"Finding checkpoints in {cfg.ckpt_path}")
    checkpoints = find_all_checkpoints(cfg.ckpt_path)
    
    if not checkpoints:
        log.warning(f"No checkpoints found in {cfg.ckpt_path}")
        return {}

    results = {}
    for dset_key, ckpt_paths in checkpoints.items():
        log.info(f"Processing {len(ckpt_paths)} checkpoints for dataset '{dset_key}'")
        results[dset_key] = {}
        cache = defaultdict(list)
        for ckpt in ckpt_paths:
            log.info(f"Instantiating model from checkpoint <{cfg.model._target_}>")
            model = hydra.utils.instantiate(cfg.model)
            state_dict = torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"]
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model.to(device)

            ds = dataset_dictionary.get(dset_key)
            with torch.no_grad():
                for ibatch in range(len(ds)):
                    batch = ds[ibatch].flatten(start_dim=1).to(dtype=torch.float32).to(device)
                    output = model.model_step(batch)["loss/total/full"]
                    cache[ckpt].append(output.detach().cpu())

            del batch
            garbage_collection_cuda()
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
            batch_size=batch_size,
            device=device,
            process_group=None,
            dist_sync_fn=None
        )
        capmetric.update(
            *(torch.cat(ds, dim=0).to(device) for ds in cache.values())
        )

        capmetric_value = capmetric.compute()

        garbage_collection_cuda()
        results[dset_key] = capmetric_value
        log.info(f"CAPmetric for {dset_key}: {capmetric_value}")

    # Print summary
    log.info("=" * 80)
    log.info("SUMMARY OF RESULTS")
    log.info("=" * 80)
    for dset_key, capmetric in results.items():
        log.info(f"  {dset_key}: {capmetric}")
    
    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for CAPmetric evaluation on all checkpoints.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    results = evaluate_checkpoints(cfg)
    
    # Optionally save results to file
    if hasattr(cfg, 'save_results') and cfg.save_results:
        import json
        results_path = Path(cfg.ckpt_path) / "capmetric_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
