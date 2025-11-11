from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
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


from src.utils.checkpoints import find_checkpoints
from capmetric import ApproximationCapacity

device = "cuda" if torch.cuda.is_available() else "cpu"

def _defaultdict_to_dict(d):
    """Recursively convert defaultdict to regular dict"""
    if isinstance(d, defaultdict):
        d = {k: _defaultdict_to_dict(v) for k, v in d.items()}
    return d

def _cache_logits_from_checkpoints(
        ckpt_paths: List[Path],
        algorithm: LightningModule,
        dataset: torch.utils.data.Dataset,
        output_name: str = "logits",
    ) -> torch.Tensor:

    if len(ckpt_paths) == 0:
        return torch.empty(0)

    if len(ckpt_paths) > 1:
        log.warning(
            f"Multiple checkpoints found for combination "
            f"Using the first one: {ckpt_paths[0]}"
        )
    ckpt_path = ckpt_paths[0]

    # Load the logits in memory:
    state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")[
        "state_dict"
    ]
    algorithm.load_state_dict(state_dict, strict=True)
    algorithm.eval()
    algorithm.to(device)
    
    with torch.no_grad():
        logits = []
        for ibatch in range(len(dataset)):
            batch = dataset[ibatch]
            batch = tuple([
                t.to(device=device, dtype=torch.float32) if isinstance(t, torch.Tensor) else t
                for t in batch
            ])
            output = algorithm.model_step(batch)[output_name]
            logits.append(output.detach().cpu())

    del dataset, batch, algorithm, state_dict
    return torch.cat(logits, dim=0)

def _compute_cap(
        logits0: torch.Tensor,
        logits1: torch.Tensor,
        batch_size: int
    ) -> Optional[float]:
    
    if len(logits0) == 0 or len(logits1) == 0:
        return None

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
        dist_sync_fn=None,
    )
    capmetric.update(
        logits0.to(device=device),
        logits1.to(device=device),
    )
    cap = capmetric.compute()
    return cap


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
    checkpoint_dict = find_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        by_combination=False,

        exclude_prefix=["cap", "lko"],
    )
    for epoch, ckpt_paths in checkpoint_dict.items():
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
    
    log.info("Cacheing the data before CAP computation for SINGLE and LOO checkpoints...")
    checkpoint_dict = find_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        by_combination=True,

        exclude_prefix=["cap", "lko"],
        exclude_ds=["main", "main_val", "main_test"]
    )
    cache = {}
    for prefix in sorted({k[1] for k in checkpoint_dict.keys()}):
        for metric_name in sorted({k[2] for k in checkpoint_dict.keys()}):            
            for dset_key in sorted({k[0] for k in checkpoint_dict.keys()}):
                logits = _cache_logits_from_checkpoints(
                    ckpt_paths=checkpoint_dict.get((dset_key, prefix, metric_name), []),
                    algorithm=hydra.utils.instantiate(cfg.algorithm),
                    dataset=dataset_dictionary.get(dset_key),
                    output_name=cfg.callbacks.cap_callback.output_name
                )
                if len(logits) == 0:
                    continue

                cache[(dset_key, prefix, metric_name)] = logits
                

    it = -1
    log.info("Computing SINGLE and LOO CAP...")
    results = {}
    for prefix in sorted({k[1] for k in cache}):
        for metric_name in sorted({k[2] for k in cache}):
            # Load existing dataset keys:
            dset_keys = {k[0] for k in cache if k[1] == prefix and k[2] == metric_name}
            for dset_key0, dset_key1 in itertools.combinations(sorted(dset_keys), 2):
                it +=1   

                cap = _compute_cap(
                    logits0=cache.get((dset_key0, prefix, metric_name), torch.empty(0)),
                    logits1=cache.get((dset_key1, prefix, metric_name), torch.empty(0)),
                    batch_size=cfg.data.batch_size
                )
                if cap is not None:
                    results.setdefault(f"cap-{prefix}", {}).setdefault(metric_name, {})[(dset_key0, dset_key1)] = cap

                if it > 10:break
            if it > 10: break
        if it > 10: break

    metric_dict.update(_defaultdict_to_dict(results))
    """
    Here we add all the SINGLE and LOO CAP evaluations for all the signals.
    """

    log.info("Cacheing the data before CAP computation for LKO checkpoints...")
    
    # BUG: This assumes that single-signal checkpoints coincide with the ones available
    # for subset selection. We assume this is true because data is fixed.
    # Alternatively, we could store a metadata file with the leave-k groups used.

    # Get the datasets that should be evaluated:
    available_signals = sorted({k[0] for k in checkpoint_dict.keys()})
    allowed_signals = [
        k for k in available_signals
        if k not in cfg.leave_k_out.ds_keys_to_skip
    ]
    signals_k = [
        k for k in allowed_signals
        if k in cfg.leave_k_out.ds_keys_selected
    ]
    signals_kc = [
        k for k in allowed_signals
        if k not in cfg.leave_k_out.ds_keys_selected
    ]

    def _cache_logits_from_lko(subset_model: str, subset_data: str):
        assert subset_model in ["k", "kc"]; assert subset_data in ["k", "kc"]

        checkpoint_key = "leave-k-in" if subset_model == "k" else "leave-k-out"
        signals = signals_k if subset_data == "k" else signals_kc
        
        logits = [
            _cache_logits_from_checkpoints(
                ckpt_paths=checkpoint_dict.get((checkpoint_key, "lko", metric_name), []),
                algorithm=hydra.utils.instantiate(cfg.algorithm),
                dataset=dataset_dictionary.get(dset_key),
                output_name=cfg.callbacks.cap_callback.output_name
            )
            for dset_key in signals
        ]
        return torch.cat(logits, dim=0)

    # Get the checkpoints of the LKO evaluations:
    checkpoint_dict = find_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        by_combination=True,

        include_prefix=["lko"],
        include_ds=["selected_ds", "left_out_ds"]
    )
    cache = {}
    for metric_name in sorted({k[2] for k in checkpoint_dict.keys()}):    

        """
        Syntax: logits_X_Y where
            - X: data used to select the model, thus originating the checkpoint
            - Y: data used to evaluate the CAP metric

        X, Y can be "k" (subset) or "kc" (complementary)
        """
        logits_k_k = _cache_logits_from_lko(subset_model="k", subset_data="k")
        logits_kc_k = _cache_logits_from_lko(subset_model="kc", subset_data="k")
        logits_k_kc = _cache_logits_from_lko(subset_model="k", subset_data="kc")
        logits_kc_kc = _cache_logits_from_lko(subset_model="kc", subset_data="kc")

        if len(logits_k_k) == 0 or len(logits_kc_k) == 0 or len(logits_k_kc) == 0 or len(logits_kc_kc) == 0:
            continue

        cache[("k", "k", metric_name)] = logits_k_k
        cache[("k", "kc", metric_name)] = logits_k_kc
        cache[("kc", "k", metric_name)] = logits_kc_k
        cache[("kc", "kc", metric_name)] = logits_kc_kc

    # Evaluate cases where CAP has same data but different encoder:
    same_data = [(("kc", "k"), ("k", "k")), (("kc", "kc"), ("k", "kc"))]
    # Evaluate cases where CAP has same encoder but different data:
    same_encoder = [(("k", "k"), ("k", "kc")), (("kc", "k"), ("kc", "kc"))]

    # BUG: This might not be the best strategy
    # Fix the length for the CAP evaluation:
    min_length = min(len(v) for v in cache.values())

    log.info("Computing LKO CAP...")
    results = {}
    for metric_name in sorted({k[2] for k in checkpoint_dict.keys()}): 
        for ((data0, encoder0), (data1, encoder1)) in same_data + same_encoder:
            logits0=cache.get((data0, encoder0, metric_name), torch.empty(0))
            logits1=cache.get((data1, encoder1, metric_name), torch.empty(0))
            cap = _compute_cap(
                logits0=logits0[:min_length],
                logits1=logits1[:min_length],
                batch_size=cfg.data.batch_size
            )
            if cap is not None:
                results.setdefault("cap-lko", {}).setdefault(metric_name, {})[((data0, encoder0), (data1, encoder1))] = cap

    metric_dict.update(_defaultdict_to_dict(results))
    """
    Here we add the LKO CAP evaluations.
    """


    log.info("Caching the data before CAP computation between background and signal simulations...")

    # Get the checkpoints of the LAI evaluations:
    checkpoint_dict = find_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        by_combination=True,

        include_prefix=["lko"],
        include_ds=["all_in"]
    )
    available_metrics_lai = sorted({k[2] for k in checkpoint_dict.keys()})
    cache = {}
    for metric_name in available_metrics_lai: 
        logits = [
            _cache_logits_from_checkpoints(
                ckpt_paths=checkpoint_dict.get(("leave-all-in", "lko", metric_name), []),
                algorithm=hydra.utils.instantiate(cfg.algorithm),
                dataset=dataset_dictionary.get(dset_key),
                output_name=cfg.callbacks.cap_callback.output_name
            )
            for dset_key in allowed_signals
        ]
        logits = torch.cat(logits, dim=0)
        if len(logits) > 0:
            cache[("leave-all-in", "lko", metric_name)] = logits
    
    checkpoint_dict = find_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        by_combination=True,

        include_prefix=["single"],
        include_ds=["main", "SingleNeutrino_E-10-gun", "SingleNeutrino_Pt-2To20-gun"]
    )
    available_metrics_single = sorted({k[2] for k in checkpoint_dict.keys()})
    for metric_name in available_metrics_single:
        for dset_key in sorted({k[0] for k in checkpoint_dict.keys()}):
            logits = _cache_logits_from_checkpoints(
                ckpt_paths=checkpoint_dict.get((dset_key, "single", metric_name), []),
                algorithm=hydra.utils.instantiate(cfg.algorithm),
                dataset=dataset_dictionary.get(dset_key),
                output_name=cfg.callbacks.cap_callback.output_name
            )        
            if len(logits) == 0:
                continue

            cache[(dset_key, "single", metric_name)] = logits

    # BUG: This could be improved, see earlier as well
    min_length = min(len(v) for v in cache.values())

    log.info("Computing CAP between background and signal simulations...")
    results = {}
    for metric_name in set(available_metrics_lai).intersection(set(available_metrics_single)):
        logits_all = cache.get(("leave-all-in", "lko", metric_name), torch.empty(0))
        for dset_key in sorted({k[0] for k in cache.keys() if k[1] == "single" and k[2] == metric_name}):
            logits_single = cache.get((dset_key, "single", metric_name), torch.empty(0))
           
            cap = _compute_cap(
                logits0=logits_all[:min_length],
                logits1=logits_single[:min_length],
                batch_size=cfg.data.batch_size
            )
            if cap is not None:
                results.setdefault("cap-background", {}).setdefault(metric_name, {})[("all-signals", dset_key)] = cap

    metric_dict.update(_defaultdict_to_dict(results))
    """
    Here we add the CAP evaluations between background and signal simulations.
    """
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
