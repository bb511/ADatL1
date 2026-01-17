import time
import random
import pickle
import os
import os.path as osp
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

import wandb
from wandb.errors.errors import CommError

AVAILABLE_METRICS = None

def retrieve_from_history(run, keyname, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            return [
                row[keyname]
                for row in run.scan_history(keys=[keyname])
                if row[keyname] is not None
            ]
        except CommError as e:
            if "502" in str(e) and retries < max_retries - 1:
                wait_time = (2**retries) + random.random()
                print(
                    f"Encountered 502 error for {keyname}, retrying in {wait_time:.1f} seconds..."
                )
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Failed to retrieve {keyname} after {retries} retries")
                return []  # Return empty list as fallback


def retrieve_many_histories_sparse(run, metrics, *, page_size=200, max_retries=5, summary_fallback=True):
    metrics = list(metrics)          # keeps your desired order
    mset = set(metrics)
    out = {m: [] for m in metrics}

    retries = 0
    while True:
        try:
            # Scan and see what you get back
            for row in run.scan_history(page_size=page_size):
                # row only contains keys that exist at that step
                for k, v in row.items():
                    if k in mset and v is not None:
                        out[k].append(v)

            break

        except CommError as e:
            if "502" in str(e) and retries < max_retries - 1:
                wait = (2 ** retries) + random.random()
                print(f"502 while scanning history, retrying in {wait:.1f}s...")
                time.sleep(wait)
                retries += 1
            else:
                print(f"Failed scanning history after {retries} retries: {e}")
                break

    # OPTIONAL but very useful in practice:
    # many frameworks log final metrics to summary only (no history points).
    if summary_fallback:
        summ = getattr(run, "summary", {})
        for m in metrics:
            if not out[m] and m in summ and summ[m] is not None:
                out[m].append(summ[m])

    return out



def adatl1_wandb(
    group: str,
    filters: dict,
    dirname: str = "results",
    cache: bool = False,
) -> pd.DataFrame:

    api = wandb.Api(timeout=100)
    runs = api.runs(
        "viictorjimenezzz-personal/adatl1",
        filters={
            "group": group,
            "$and": [{k: v} for k, v in filters.items()],
        },
    )
    print(f"{len(runs)} runs found.")

    pathdir = osp.join(dirname, group)
    os.makedirs(dirname, exist_ok=True)
    fname = "_".join(
        [
            k.split(".")[-1] + "=" + str(v)
            for k, v in filters.items()
        ]
    )
    fname = "results.pkl" if len(fname) == 0 else fname + ".pkl"
    fpath = osp.join(pathdir, fname)
    if cache and osp.exists(fpath):
        return pd.read_pickle(fpath), fpath

    data = defaultdict(list)
    for irun, run in tqdm(enumerate(runs)):
        print(f"({irun+1}/{len(runs)}) Processing run {run.id}...")
        config = run.config
        # print(
        #     json.dumps(config, indent=4)
        # )

        ## GENERAL
        data["id"].append(run.id)
        data["seed"].append(config.get("seed"))
        data["tags"].append(config.get("tags"))

        ## DATA
        data_config = config.get("data")
        data["data/split"].append(data_config.get("data_mlready", {}).get("split"))
        data["data/batch_size"].append(data_config.get("batch_size"))
        data["data/features"].append(data_config.get("train_features"))

        data["data/extractor"].append(data_config.get("data_extractor", {}).get("name"))
        data["data/processor"].append(data_config.get("data_processor", {}).get("name"))
        data["data/normalizer"].append(data_config.get("data_mormalizer", {}).get("name"))
        data["data/awkward2torch"].append(data_config.get("data_awkward2torch", {}).get("name"))
        data["data/mlready"].append(data_config.get("data_mlready", {}).get("name"))

        ## ALGORITHM
        algorithm_config = config.get("algorithm", {})
        data["algorithm/_target_"].append(algorithm_config.get("_target_"))

        # Loss:
        for key, value in algorithm_config.get("loss", {}).items():
            if key not in ["_partial_"]:
                data[f"model/optimizer/{key}"].append(value)
        
        # Model:
        model_cfg = algorithm_config.get("model", {})
        data["algorithm/model/_target_"].append(model_cfg.get("_target_"))
        data["algorithm/model/nodes"].append(model_cfg.get("nodes"))
        data["algorithm/model/batchnorm"].append(model_cfg.get("batchnorm"))
        data["algorithm/model/init_bias"].append(model_cfg.get("init_bias"))
        data["algorithm/model/int_weight"].append(model_cfg.get("int_weight"))
        for key, value in model_cfg.items():
            if key not in ["_target_", "nodes", "batchnorm", "init_bias", "init_weight"]:
                data[f"algorithm/model/{key}"].append(value)

        # Optimizer:
        optimizer_cfg = algorithm_config.get("optimizer", {})
        for key, value in optimizer_cfg.items():
            if key not in ["_partial_"]:
                data[f"algorithm/optimizer/{key}"].append(value)

        # Scheduler:
        scheduler_cfg = algorithm_config.get("scheduler", {}).get("scheduler", {})
        for key, value in scheduler_cfg.items():
            if key not in ["_partial_"]:
                data[f"algorithm/scheduler/{key}"].append(value)


        ## TRAINER
        trainer_cfg = algorithm_config.get("trainer", {})
        data["trainer/max_epochs"].append(trainer_cfg.get("max_epochs"))
        data["trainer/accelerator"].append(trainer_cfg.get("accelerator"))
        data["trainer/devices"].append(trainer_cfg.get("devices"))
        data["trainer/deterministic"].append(trainer_cfg.get("deterministic"))

        # cap_cfg = callbacks_cfg.get("approximation_capacity", {})
        # for key, value in cap_cfg.items():
        #     if key in [
        #         "lr",
        #         "beta0",
        #         "n_epochs",
        #         "energy_type",
        #         "energy_params",
        #         "regularization_type",
        #         "regularization_params",
        #         "normalization_type",
        #         "normalization_params",
        #         "pairing_type",
        #         "output_name",  # same as metric name
        #     ]:
        #         data[f"callbacks/cap/{key}"].append(value)

        # for key, value in cap_cfg.get("data_pairs", {}).items():
        #     data[f"callbacks/cap/data_pairs/{key}"].append(value)

        # Metrics (this will depend on what the run logs)
        values = retrieve_many_histories_sparse(run, list(run.history().keys()))
        for m, v in values.items():
            v = v[0]
            if m.startswith("test"):
                data[m].append(v)
       
        # values_by_metric = retrieve_all_from_history(run, AVALIABLE_METRICS)
        # a
        # import ipdb; ipdb.set_trace()
        # for m in tqdm(AVALIABLE_METRICS, desc="Retrieving metrics"):
        #     data[m].append(values_by_metric[m])

        # import ipdb; ipdb.set_trace()

    df = pd.DataFrame(data)

    # Cache
    if cache and not osp.exists(fpath):
        with open(fpath, "wb") as file:
            pickle.dump(df, file)

    return df, fpath
