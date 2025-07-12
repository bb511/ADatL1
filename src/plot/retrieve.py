import time
import random
import pickle
import os
import os.path as osp
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

import wandb
from wandb.errors.errors import CommError


def retrieve_from_history(run, keyname, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            return [row[keyname] for row in run.scan_history(keys=[keyname]) if row[keyname] is not None]
        except CommError as e:
            if "502" in str(e) and retries < max_retries - 1:
                wait_time = (2 ** retries) + random.random()
                print(f"Encountered 502 error for {keyname}, retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Failed to retrieve {keyname} after {retries} retries")
                return []  # Return empty list as fallback
            

def adatl1_wandb(
        group: str,
        lr: float = None,
        alpha: float = None,
        dirname: str = "results",
        cache: bool = False,
    ) -> pd.DataFrame:

    api = wandb.Api(timeout=100)
    config_filters = [
        {"config.model.loss.alpha": alpha} if alpha != None else {},
        {"config.model.optimizer.lr": lr} if lr != None else {},
    ]
    runs = api.runs(
        "viictorjimenezzz-personal/adatl1",
        filters= {
            "group": group,
            "$and": [config for config in config_filters if len(config) > 0]
        },
    )
    print(f"{len(runs)} runs found.")

    pathdir = osp.join(dirname, group)
    os.makedirs(dirname, exist_ok=True)
    fname = "_".join([
        key.split(".")[-1] + "=" + str(value)
        for config in config_filters
        for key, value in config.items()
    ])
    fname = "joint.pkl" if len(fname) == 0 else fname + ".pkl" 
    fpath = osp.join(pathdir, fname)
    if cache and osp.exists(fpath):
        return pd.read_pickle(fpath), fpath

    data = defaultdict(list)
    # for i, run in tqdm(enumerate(runs), total=len(runs), desc="Runs processed"):
    for irun, run in enumerate(runs):
        print(f"({irun+1}/{len(runs)}) Processing run {run.id}...")
        config = run.config
        # print(
        #     json.dumps(config, indent=4)
        # )
        
        # Check if it's a valid L1AD run
        data_config = config.get("data", {})
        if "signal" not in data_config:
            continue
        
        # Basic info
        data["id"].append(run.id)
        data["seed"].append(config.get("seed"))
        data["tags"].append(config.get("tags"))


        # Data config
        data["data/split"].append(data_config.get("split"))
        data["data/batch_size"].append(data_config.get("batch_size"))
        data["data/n_val_batches"].append(data_config.get("val_batches"))

        data["data/extractor/name"].append(data_config.get("extractor",{}).get("name"))
        data["data/processor/name"].append(data_config.get("processor",{}).get("name"))

        normalizer_cfg = data_config.get("data_normalizer", {})
        data["data/normalizer/scheme"].append(normalizer_cfg.get("norm_scheme"))
        data["data/normalizer/ignore_zeros"].append(normalizer_cfg.get("ignore_zeros"))
        data["data/normalizer/output_dtype"].append(normalizer_cfg.get("output_dtype"))
        data["data/normalizer/norm_hyperparams/scale"].append(
            normalizer_cfg.get("norm_hyperparams", {}).get("scale")
        )
        data["data/normalizer/norm_hyperparams/percentiles"].append(
            normalizer_cfg.get("norm_hyperparams", {}).get("percentiles")
        )
        

        # Model config
        model_config = config.get("model", {})
        loss_cfg = model_config.get("loss", {})
        data["model/loss/_target_"].append(loss_cfg.get("_target_"))
        data["model/loss/alpha"].append(loss_cfg.get("alpha"))

        encoder_cfg = model_config.get("encoder", {})
        data["model/encoder/_target_"].append(encoder_cfg.get("_target_"))
        data["model/encoder/nodes"].append(encoder_cfg.get("nodes"))
        init_bias = encoder_cfg.get("init_bias", {}) or {}
        init_weight = encoder_cfg.get("init_weight", {}) or {}
        data["model/encoder/init_bias"].append(init_bias.get("path"))
        data["model/encoder/init_weight"].append(init_weight.get("path"))

        decoder_cfg = model_config.get("decoder", {})
        data["model/decoder/_target_"].append(decoder_cfg.get("_target_"))
        data["model/decoder/nodes"].append(decoder_cfg.get("nodes"))
        init_bias = decoder_cfg.get("init_bias", {}) or {}
        init_weight = decoder_cfg.get("init_weight", {}) or {}
        data["model/decoder/init_bias"].append(init_bias.get("path"))
        data["model/decoder/init_weight"].append(init_weight.get("path"))
        init_last_bias = decoder_cfg.get("init_last_bias", {}) or {}
        init_last_weight = decoder_cfg.get("init_last_weight", {}) or {}
        data["model/decoder/init_last_bias"].append(init_last_bias.get("path"))
        data["model/decoder/init_last_weight"].append(init_last_weight.get("path"))

        optimizer_cfg = model_config.get("optimizer", {})
        for key, value in optimizer_cfg.items():
            if key not in ["_partial_"]:
                data[f"model/optimizer/{key}"].append(value)

        scheduler_cfg = model_config.get("scheduler", {}).get("scheduler", {})
        for key, value in scheduler_cfg.items():
            if key not in ["_partial_"]:
                data[f"model/scheduler/{key}"].append(value)
        

        # Trainer config
        trainer_cfg = model_config.get("trainer", {})
        data["trainer/max_epochs"].append(trainer_cfg.get("max_epochs"))
        data["trainer/accelerator"].append(trainer_cfg.get("accelerator"))
        data["trainer/devices"].append(trainer_cfg.get("devices"))
        data["trainer/deterministic"].append(trainer_cfg.get("deterministic"))


        # Callbacks config
        callbacks_cfg = model_config.get("callbacks", {})

        # Anomaly Rate callback
        ar_cfg = callbacks_cfg.get("anomaly_rate", {})
        for key, value in ar_cfg.items():
            data[f"callbacks/anomaly_rate/{key}"].append(value)

        # Model Checkpoint callbacks
        for callback_name, callback_cfg in callbacks_cfg.items():
            if callback_name.startswith("model_checkpoint"):
                for key in ["_target_", "mode", "metric_name"]:
                    data[f"callbacks/{callback_name}/{key}"].append(callback_cfg.get(callback_name, {}).get(key))


        cap_cfg = callbacks_cfg.get("approximation_capacity", {})
        for key, value in cap_cfg.items():
            if key in [
                "lr",
                "beta0",
                "n_epochs",
                "energy_type",
                "energy_params",
                "regularization_type",
                "regularization_params",
                "normalization_type",
                "normalization_params",
                "pairing_type",
                "output_name" # same as metric name
            ]:
                data[f"callbacks/cap/{key}"].append(value)

        for key, value in cap_cfg.get("data_pairs", {}).items():
            data[f"callbacks/cap/data_pairs/{key}"].append(value)

        # Metrics (this will depend on what the run logs)
        AVAILABLE_METRICS = run.history().keys()
        for imetric, metric in tqdm(enumerate(AVAILABLE_METRICS), total=len(AVAILABLE_METRICS), desc="Retrieving metrics"):
            # print(f"({imetric+1}/{len(AVAILABLE_METRICS)}) Retrieving {metric}")
            data[metric].append(
                retrieve_from_history(run, metric)
            )

    df = pd.DataFrame(data)

    # Cache
    if cache and not osp.exists(fpath):
        with open(fpath, 'wb') as file:
            pickle.dump(df, file)

    return df, fpath