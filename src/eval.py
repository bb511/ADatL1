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


from src.utils.checkpoints import find_checkpoints, _parse_filename
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
        raise ValueError("No checkpoint path has been selected.")

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
    # checkpoint_dict: (prefix, metric_name, criterion, dset_key) -> [ckpt_path, ...]
    checkpoint_dict = find_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        by_combination=True,

        exclude_prefix=["cap", "lko"],
        exclude_ds=["main", "main_val", "main_test"]
    )

    cache: Dict[(int, str), torch.Tensor] = {} # (epoch, dset_key) -> logits
    params_to_epoch: Dict[(int, tuple)] = {} # (prefix, metric_name, criterion, dset_key) -> epoch
    for prefix, metric_name, criterion, dset_key in checkpoint_dict.keys():

        # TODO: Consider logic here, we might want to get the ckpt depending on the value!!
        ckpt_path = checkpoint_dict[(prefix, metric_name, criterion, dset_key)][0]

        # Compute epoch associated to parameter combination and store in the dictionary:
        epoch = _parse_filename(Path(ckpt_path).name).get("epoch")
        params_to_epoch[(prefix, metric_name, criterion, dset_key)] = epoch
                 
        # Skip calculations when logits already exist:
        if (epoch, dset_key) not in cache.keys():
            logits = _cache_logits_from_checkpoints(
                ckpt_paths=[ckpt_path],
                algorithm=hydra.utils.instantiate(cfg.algorithm),
                dataset=dataset_dictionary.get(dset_key),
                output_name=cfg.callbacks.cap_callback.output_name
            )
            if len(logits) == 0:
                continue

            cache[(epoch, dset_key)] = logits
                
    it = -1
    log.info("Computing SINGLE and LOO CAP...")

    results = {}
    directory_hparams = set([(prefix, metric_name, criterion) for prefix, metric_name, criterion, _ in checkpoint_dict.keys()])
    for prefix, metric_name, criterion in directory_hparams:

        # Load existing dataset keys:
        dset_keys = {k[3] for k in checkpoint_dict.keys() if k[0] == prefix and k[1] == metric_name and k[2] == criterion}
        for dset_key0, dset_key1 in itertools.combinations(sorted(dset_keys), 2):
            it +=1   

            epoch0 = params_to_epoch.get((prefix, metric_name, criterion, dset_key0))
            epoch1 = params_to_epoch.get((prefix, metric_name, criterion, dset_key1))
            cap = _compute_cap(
                logits0=cache.get((epoch0, dset_key0)),
                logits1=cache.get((epoch1, dset_key1)),
                batch_size=cfg.data.batch_size
            )
            if cap is not None:
                results.setdefault(f"cap", {}).setdefault(prefix, {}).setdefault(metric_name, {}).setdefault(criterion, {})[(dset_key0, dset_key1)] = cap
                    
            if it > 5:break # just for testing purposes

    metric_dict.update(_defaultdict_to_dict(results))
    """
    Here we add all the SINGLE and LOO CAP evaluations for all the signals.
    """

    log.info("Cacheing the data before CAP computation for LKO checkpoints...")
    
    # BUG: This assumes that single-signal checkpoints coincide with the ones available
    # for subset selection. We assume this is true because data is fixed.
    # Alternatively, we could store a metadata file with the leave-k groups used.

    # Get the datasets that should be evaluated:
    available_signals = sorted({k[3] for k in checkpoint_dict.keys()})
    allowed_signals = [
        k for k in available_signals
        if k not in cfg.leave_k_out.skip_ds
    ]
    signals_k = [
        k for k in allowed_signals
        if k in cfg.leave_k_out.selected_datasets
    ]
    signals_kc = [
        k for k in allowed_signals
        if k not in cfg.leave_k_out.selected_datasets
    ]

    def _cache_logits_from_lko(
            subset_model: str,
            subset_data: str,
            metric_name: str,
            criterion: str
        ):
        assert subset_model in ["k", "kc"]; assert subset_data in ["k", "kc"]

        dset_key = "selected_ds" if subset_model == "k" else "left_out_ds"
        signals = signals_k if subset_data == "k" else signals_kc
        
        ckpt_paths = checkpoint_dict.get(("lko", metric_name, criterion, dset_key), [])
        if len(ckpt_paths) == 0:
            return torch.empty(0)
        
        logits = [
            _cache_logits_from_checkpoints(
                ckpt_paths=[ckpt_paths[0]],
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
    directory_hparams = set([
        (metric_name, criterion) for prefix, metric_name, criterion, dset_key in checkpoint_dict.keys()
        if prefix == "lko" and dset_key in ["selected_ds", "left_out_ds"] # innecessary but just in case
    ])
    for metric_name, criterion in directory_hparams:

        """
        Syntax: logits_X_Y where
            - X: data used to select the model, thus originating the checkpoint
            - Y: data used to evaluate the CAP metric

        X, Y can be "k" (subset) or "kc" (complementary)
        """
        logits_k_k = _cache_logits_from_lko(subset_model="k", subset_data="k", metric_name=metric_name, criterion=criterion)
        logits_kc_k = _cache_logits_from_lko(subset_model="kc", subset_data="k", metric_name=metric_name, criterion=criterion)
        logits_k_kc = _cache_logits_from_lko(subset_model="k", subset_data="kc", metric_name=metric_name, criterion=criterion)
        logits_kc_kc = _cache_logits_from_lko(subset_model="kc", subset_data="kc", metric_name=metric_name, criterion=criterion)

        if len(logits_k_k) == 0 or len(logits_kc_k) == 0 or len(logits_k_kc) == 0 or len(logits_kc_kc) == 0:
            continue

        cache[("k", "k", metric_name, criterion)] = logits_k_k
        cache[("k", "kc", metric_name, criterion)] = logits_k_kc
        cache[("kc", "k", metric_name, criterion)] = logits_kc_k
        cache[("kc", "kc", metric_name, criterion)] = logits_kc_kc


    # Evaluate cases where CAP has same data but different encoder:
    same_data = [(("kc", "k"), ("k", "k")), (("kc", "kc"), ("k", "kc"))]
    # Evaluate cases where CAP has same encoder but different data:
    same_encoder = [(("k", "k"), ("k", "kc")), (("kc", "k"), ("kc", "kc"))]

    # BUG: This might not be the best strategy
    # Fix the length for the CAP evaluation:
    min_length = min(len(v) for v in cache.values())

    log.info("Computing LKO CAP...")
    results = {}
    for metric_name, criterion in directory_hparams:
        for ((data0, encoder0), (data1, encoder1)) in same_data + same_encoder:

            logits0=cache.get((data0, encoder0, metric_name, criterion))
            logits1=cache.get((data1, encoder1, metric_name, criterion))
            if logits0 is None or logits1 is None:
                continue

            cap = _compute_cap(
                logits0=logits0[:min_length],
                logits1=logits1[:min_length],
                batch_size=cfg.data.batch_size
            )
            if cap is not None:
                results.setdefault("cap", {}).setdefault("lko", {}).setdefault(metric_name, {}).setdefault(criterion, {})[((data0, encoder0), (data1, encoder1))] = cap

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
    

    cache = {}
    directory_hparams_lko = set([
        (metric_name, criterion, dset_key) for prefix, metric_name, criterion, dset_key in checkpoint_dict.keys()
        if prefix == "lko" and dset_key == "all_in" # innecessary but just to make sure
    ])
    for metric_name, criterion, _ in directory_hparams_lko: 
        logits = []
        for dset_key in allowed_signals:
            ckpt_paths = checkpoint_dict.get(("lko", metric_name, criterion, dset_key), [])
            if len(ckpt_paths) == 0:
                break # not continue

            logits.append(
                _cache_logits_from_checkpoints(
                    ckpt_paths=[ckpt_paths[0]],
                    algorithm=hydra.utils.instantiate(cfg.algorithm),
                    dataset=dataset_dictionary.get(dset_key),
                    output_name=cfg.callbacks.cap_callback.output_name
                )
            )

        if len(logits) == 0:
            continue

        cache[("lko", metric_name, criterion, "all_in")] = torch.cat(logits, dim=0)
    
    checkpoint_dict = find_checkpoints(
        dirpath=cfg.paths.checkpoints_dir,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        by_combination=True,

        include_prefix=["single"],
        include_ds=["main", "SingleNeutrino_E-10-gun", "SingleNeutrino_Pt-2To20-gun"]
    )
    directory_hparams_lai = set([
        (metric_name, criterion, dset_key) for prefix, metric_name, criterion, dset_key in checkpoint_dict.keys()
        if prefix == "single" and dset_key in ["main", "SingleNeutrino_E-10-gun", "SingleNeutrino_Pt-2To20-gun"] # innecessary but just to make sure
    ])
    for metric_name, criterion, dset_key in directory_hparams_lai:
        ckpt_path = checkpoint_dict.get(("single", metric_name, criterion, dset_key))[0]
        logits = _cache_logits_from_checkpoints(
            ckpt_paths=[ckpt_path],
            algorithm=hydra.utils.instantiate(cfg.algorithm),
            dataset=dataset_dictionary.get(dset_key),
            output_name=cfg.callbacks.cap_callback.output_name
        )        
        if len(logits) == 0:
            continue
        
        cache[("single", metric_name, criterion, dset_key)] = logits


    # BUG: This could be improved, see earlier as well
    min_length = min(len(v) for v in cache.values())

    log.info("Computing CAP between background and signal simulations...")
    results = {}
    intersection_lai_lko = set(
        [(metric_name, criterion) for metric_name, criterion, _ in directory_hparams_lai]
    ).intersection(set(
        [(metric_name, criterion) for metric_name, criterion, _ in directory_hparams_lko]
    ))
    for metric_name, criterion in intersection_lai_lko:
        logits_all = cache.get(("lko", metric_name, criterion, "all_in"))
        if logits_all is None:
            continue

        available_dset_keys = set([
            dset_key for prefix, metric_name_loop, criterion_loop, dset_key in cache.keys()
            if prefix == "single" and metric_name_loop == metric_name and criterion_loop == criterion
        ])
        for dset_key in available_dset_keys:
            logits_single = cache.get(("single", metric_name, criterion, dset_key))
            if logits_single is None:
                continue
           
            cap = _compute_cap(
                logits0=logits_all[:min_length],
                logits1=logits_single[:min_length],
                batch_size=cfg.data.batch_size
            )
            if cap is not None:
                results.setdefault("cap-background", {}).setdefault(metric_name, {}).setdefault(criterion, {})[("all-signals", dset_key)] = cap

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
