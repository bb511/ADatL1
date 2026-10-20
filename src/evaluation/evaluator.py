# Evaluator object definition. Produces the desired evaluation plots for given model
# checkpoints saved during the training of a model, and evaluates the per-epoch score
# dumps of a lattice (evaluate_dump, evaluate_lattice).
import csv
import hashlib
import json
import logging
import math
import multiprocessing
import operator
import os
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pytorch_lightning
import torch
import yaml
from colorama import Back, Fore, Style
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

from src.callbacks.metrics.selection import (
    W_STEPS,
    best_epoch,
    ema,
    epoch_pick,
    smoothed,
)
from src.callbacks.sds import sds_inputs
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback
from src.evaluation.callbacks.utils import mlflow as eval_mlflow
from src.evaluation.replay import ReplayModule, extra_loaders
from src.utils.checkpoints import is_valid_ckpt
from src.utils.omegaconf import register_resolvers

ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = ROOT / "configs" / "hparams_search"
# The key of the training efficiency callback for one set's validation efficiency.
SET_EFFICIENCY_KEY = "val/{}/eff__ascore_full__brate_operational"
# The most metrics MLflow accepts in one log_batch call.
MLFLOW_BATCH = 1000


def _lists_candidate(ckpts: dict, candidate: dict) -> bool:
    metrics = ckpts.get(candidate["strategy"])
    if not hasattr(metrics, "get"):
        return False
    return candidate["criterion"] in (metrics.get(candidate["metric"]) or [])


def _container(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def _override(text: str) -> tuple[str, str]:
    key, _, value = text.partition("=")
    return key.lstrip("+~"), value


def hyperpoint(run_dir: Path, search_dir: Path = SEARCH_DIR) -> dict[str, str]:
    """The sampled hyperparameters of a run, as its ``overrides.yaml`` spells them.

    These are the overrides whose key is a parameter of the sweep that the run's
    ``hparams_search=<name>`` override names, read from ``<search_dir>/<name>.yaml``
    (``hydra.sweeper.params``); a run outside a sweep, or without ``overrides.yaml``,
    has none. The values stay strings as Hydra wrote them, so that the runs of two
    lattices drawn with the same sampler seed pair by plain equality.
    """
    path = Path(run_dir) / "overrides.yaml"
    if not path.exists():
        return {}
    overrides = dict(_override(text) for text in yaml.safe_load(path.read_text()))
    search = overrides.get("hparams_search")
    if search is None:
        return {}
    config = yaml.safe_load((Path(search_dir) / f"{search}.yaml").read_text())
    params = config["hydra"]["sweeper"]["params"]
    return {key: overrides[key] for key in params if key in overrides}


def _epoch_files(run_dir: Path) -> list[Path]:
    return [run_dir / "initial.pt", *sorted(run_dir.glob("epoch_*.pt"))]


def _load_record(path: Path, epoch: int) -> dict:
    record = torch.load(path, weights_only=True)
    if record["epoch"] != epoch:
        raise RuntimeError(f"{path} holds epoch {record['epoch']}, not {epoch}.")
    return record


# One entry, kept across the runs a worker evaluates: the runs of a lattice share their
# training inputs, and building the pseudo-sets costs a nearest-neighbour search (on
# CIFAR-10, 75 s and 6 GB), so evaluate_lattice builds them once in the parent and
# hands them to every worker.
_SDS_CACHE: dict = {}


def _cached_sds(x_train: torch.Tensor, max_rows: int, row_seed: int) -> dict:
    digest = hashlib.sha1(x_train.contiguous().numpy().tobytes()).hexdigest()
    key = (digest, tuple(x_train.shape), max_rows, row_seed)
    if key not in _SDS_CACHE:
        _SDS_CACHE.clear()
        _SDS_CACHE[key] = sds_inputs(x_train, max_rows, row_seed)
    return _SDS_CACHE[key]


def _prebuild_sds(run_dir: Path, data_dir: str | None, sds: dict | None) -> None:
    if not sds:
        return
    if not OmegaConf.has_resolver("reverse"):
        register_resolvers()
    datamodule = _rebuild(run_dir, data_dir)[1]
    datamodule.setup("fit")
    _cached_sds(datamodule.data_train.tensors[0], sds["max_rows"], sds["row_seed"])


def _dump_trainer(
    configs: dict, keys: list[str], split: str
) -> pytorch_lightning.Trainer:
    """A CPU trainer without logger, plots or files, with fresh callbacks."""
    trainer = pytorch_lightning.Trainer(
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[hydra.utils.instantiate(configs[key]) for key in keys],
    )
    trainer.plot_checkpoints, trainer.split = False, split
    trainer.strat_name = trainer.metric_name = trainer.criterion_name = None
    return trainer


def _replay_pass(trainer, replay: ReplayModule, split: str, loaders: dict) -> dict:
    """The scalars every callback records in one pass, keyed '<callback>/<scalar>'."""
    replay.serve(split)
    trainer.metric_records = {}
    trainer.test(replay, dataloaders=loaders, verbose=False)
    return trainer.metric_records


def _null_sets(callbacks: list) -> list[str]:
    efficiency = [cb for cb in callbacks if isinstance(cb, AnomalyEfficiencyCallback)]
    return list(efficiency[0].null_ds) if efficiency else []


def _signal_sets(layout: dict) -> list[str]:
    """The sets whose labels are all positive, the efficiency callback's signals."""
    return [name for name in layout["names"] if min(layout["labels"][name]) > 0]


def _strategies(selection: dict) -> dict[str, dict]:
    """Every strategy of the registry with its column in val.csv and its mode.

    The labelled rows are columns under their own names and are maximised; Q'' is
    minimised.
    """
    strategies = {name: dict(spec) for name, spec in selection["criteria"].items()}
    for name, spec in selection["labelled"].items():
        strategies[name] = {**spec, "column": name, "mode": "max"}
    strategies["q_sec"] = {**selection["q_sec"], "mode": "min"}
    return strategies


def _labelled_value(effs: list[float], spec: dict) -> float:
    """The mean, or the CVaR, of the per-set efficiencies.

    As the training efficiency callback computes its CVaR: in float32, without the
    non-finite values, the mean of the ceil(alpha n) lowest.
    """
    if spec["statistic"] not in ("mean", "cvar"):
        raise ValueError(f"Unknown labelled statistic '{spec['statistic']}'.")
    x = torch.as_tensor(effs, dtype=torch.float32)
    x = torch.sort(x[torch.isfinite(x)]).values
    if x.numel() == 0:
        return math.nan
    k = x.numel()
    if spec["statistic"] == "cvar":
        k = max(1, math.ceil(spec["alpha"] * k))
    return x[:k].mean().item()


def _labelled_rows(row: dict, signals: list[str], selection: dict) -> dict:
    effs = [row[selection["efficiency"] + name] for name in signals]
    labelled = selection["labelled"].items()
    return {name: _labelled_value(effs, spec) for name, spec in labelled}


def _add_emas(rows: list[dict], strategies: dict) -> None:
    """Add ema/<strategy> to every row: the EMA over the trained epochs, None before."""
    trained = [row for row in rows if row["epoch"] >= 0]
    untrained = [row for row in rows if row["epoch"] < 0]
    for name, spec in strategies.items():
        values = [row[spec["column"]] for row in trained]
        averages = ema(values, skip_nonfinite=spec.get("skip_nonfinite", False))
        for row, average in zip(trained, averages):
            row[f"ema/{name}"] = average
        for row in untrained:
            row[f"ema/{name}"] = None


def _strategy_picks(name: str, spec: dict, trained: list, q_sec: list, grid: bool):
    """The epoch rule over the w grid, or, without a grid, the best logged EMA of Q'."""
    q_main = [row[spec["column"]] for row in trained]
    skip = spec.get("skip_nonfinite", False)
    if grid:
        indices = {
            k: epoch_pick(q_main, q_sec, k, spec["mode"], skip)
            for k in range(W_STEPS + 1)
        }
    else:
        logged = smoothed(q_main, skip_nonfinite=skip)
        indices = {None: best_epoch(logged, spec["mode"])}
    return [
        {"strategy": name, "k": k, "epoch": trained[index]["epoch"]}
        for k, index in indices.items()
    ]


def _picks(rows: list[dict], strategies: dict, criteria: dict) -> list[dict]:
    trained = [row for row in rows if row["epoch"] >= 0]
    q_sec = [row[strategies["q_sec"]["column"]] for row in trained]
    picks = []
    for name, spec in strategies.items():
        picks += _strategy_picks(name, spec, trained, q_sec, name in criteria)
    return picks


def _test_summary(row: dict, signals: list[str], nulls: list[str], prefix: str):
    null_rates = [row[prefix + name] for name in nulls]
    return {
        "test_mean_eff": float(np.mean([row[prefix + name] for name in signals])),
        "realised_fpr": row[prefix + "normal"],
        "null_rate": float(np.mean(null_rates)) if null_rates else None,
    }


def _run_summary(run_dir: Path, meta: dict, lattice: str | None, nulls: list[str]):
    return {
        "run": run_dir.name,
        "lattice": lattice,
        "dump": str(run_dir),
        "hyperpoint": hyperpoint(run_dir),
        "mlflow_run_id": None,
        "epochs": len(_epoch_files(run_dir)) - 1,
        "module": meta["module"],
        "signals": _signal_sets(meta["val"]),
        "nulls": nulls,
        "stored": meta["stored"],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(out: Path, tables: dict[str, list[dict]], run: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        _write_csv(out / f"{name}.csv", rows)
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


def _mlflow_experiment(tracking_uri: str, name: str) -> str:
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(name)


def _mlflow_columns(strategies: dict, signals: list[str], prefix: str) -> list:
    """(MLflow key, val.csv column) of every quantity logged per epoch."""
    columns = []
    for name, spec in strategies.items():
        columns.append((spec["key"], f"ema/{name}"))
        if spec.get("raw_key"):
            columns.append((spec["raw_key"], spec["column"]))
    return columns + [(SET_EFFICIENCY_KEY.format(s), prefix + s) for s in signals]


def _mlflow_metrics(rows: list[dict], columns: list) -> list[Metric]:
    # As log_metrics_to_mlflow, non-finite values are left out; epoch -1 is too, as
    # Lightning drops the logs of the sanity pass.
    timestamp = int(time.time() * 1000)
    return [
        Metric(key, float(row[column]), timestamp, row["epoch"])
        for row in rows
        if row["epoch"] >= 0
        for key, column in columns
        if row[column] is not None and math.isfinite(row[column])
    ]


def _log_mlflow(tracking_uri: str, experiment_id: str, run: dict, metrics: list):
    client = MlflowClient(tracking_uri)
    run_id = client.create_run(experiment_id, run_name=run["run"]).info.run_id
    params = [Param(key, str(value)) for key, value in run["hyperpoint"].items()]
    client.log_batch(run_id, params=params)
    for start in range(0, len(metrics), MLFLOW_BATCH):
        client.log_batch(run_id, metrics=metrics[start : start + MLFLOW_BATCH])
    client.set_terminated(run_id)
    return run_id


def _rebuild(run_dir: Path, data_dir: str | None) -> tuple:
    """The run's module and datamodule, instantiated from its ``config.yaml``."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if data_dir is not None:
        cfg.data.data_dir = str(data_dir)
    return hydra.utils.instantiate(cfg.algorithm), hydra.utils.instantiate(cfg.data)


def _init_worker(sds_cache: dict | None = None) -> None:
    _SDS_CACHE.update(sds_cache or {})
    torch.set_num_threads(1)
    # The run configs call ${reverse:...} and the like; a second registration raises.
    if not OmegaConf.has_resolver("reverse"):
        register_resolvers()
    # Lightning repeats these at every pass, thousands of times per run.
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*has `__len__` defined.*")


def _lattice_task(task: tuple) -> dict:
    settings, run_dir, out_dir, lattice, data_dir, tracking_uri = task
    module, datamodule = _rebuild(run_dir, data_dir)
    evaluator = Evaluator(**settings)
    return evaluator.evaluate_dump(
        run_dir, module, datamodule, out_dir, lattice, tracking_uri
    )


def _map(function, tasks: list, workers: int) -> list:
    if workers <= 1:
        _init_worker()
        return [function(task) for task in tasks]

    # Spawned, not forked: torch's thread pools do not survive a fork once the parent
    # has used them.
    context = multiprocessing.get_context("spawn")
    initargs = (dict(_SDS_CACHE),)
    with context.Pool(workers, initializer=_init_worker, initargs=initargs) as pool:
        return list(pool.imap(function, tasks))


class Evaluator:
    """Thin wrapper around pl.Trainer that performs the evaluation of a model.

    The test loops are ran for this custom trainer object instantiated in this class.
    The callbacks used here are different from the callbacks used at training time and
    are contained in src/evaluation/callbacks.

    :param ckpts: Dictionary with the following structure:
        'strategy_1':
            'metric_1':
                'criterion_1': ['min', 'max', etc.]
        ...
        This is used for specifying which checkpoints to actually evaluate.
    :param callbacks: List of pl.Callback objects that compute evaluation metrics on
        the test data and do all the handling of these, including plotting.
    :param logger: pl.Logger object that handles the logging for this evaluation stage.
    :param optimized_metric_config: Dictionary naming what is reported to optuna. Its
        'candidate' entry (strategy, metric, criterion) names the one criterion
        folder of ckpts at which 'main_metric' and, if present, 'sec_metric' are
        read; raises ValueError if the candidate is not among ckpts.
    :param dump_callbacks: The dump path's evaluation callbacks, as configs keyed by
        name: every run instantiates its own, and all of them run in the validation
        pass.
    :param test_callbacks: Names among dump_callbacks of those that run in the test
        pass of the dump path.
    :param extra_sets: The sets the validation pass of the dump path appends to the
        validation sets: 'aux' (bool), the three auxiliary sets; 'train' (bool), the
        training set; 'sds' ({max_rows, row_seed} or None), the SDS pseudo-sets.
    :param selection: The registry of strategies of the dump path ('criteria',
        'q_sec', 'labelled', 'efficiency', as in configs/eval/<domain>.yaml).
    :param trainer_kwargs: The other arguments that are given to the trainer object,
        kept the same as the training, e.g., which device to run the computations on.
    """

    def __init__(
        self,
        ckpts: dict = None,
        callbacks: Optional[list[Callback]] = None,
        logger: Optional[Logger] = None,
        optimized_metric_config: dict = None,
        dump_callbacks: dict | None = None,
        test_callbacks: Sequence[str] = ("anomaly_efficiency",),
        extra_sets: dict | None = None,
        selection: dict | None = None,
        **trainer_kwargs,
    ):
        self.ckpts = ckpts or {}
        self.callbacks = callbacks or []
        self.dump_callbacks = _container(dump_callbacks) or {}
        self.test_callbacks = list(test_callbacks)
        self.extra_sets = _container(extra_sets) or {}
        self.selection = _container(selection)
        self.optimized_metric_config = optimized_metric_config or None
        if self.optimized_metric_config is not None:
            candidate = self.optimized_metric_config.get("candidate")
            if candidate is None or not _lists_candidate(self.ckpts, candidate):
                raise ValueError(
                    f"Candidate checkpoint {candidate} is not among the evaluated "
                    f"ckpts {self.ckpts}."
                )

        # Build a *real* Trainer under the hood
        self.evaluator = pytorch_lightning.Trainer(
            logger=logger,
            callbacks=self.callbacks,
            **trainer_kwargs,
        )

        self.evaluator.split = None
        self.evaluator.strat_name = None
        self.evaluator.metric_name = None
        self.evaluator.criterion_name = None
        self.optimized_metric = None
        self.optimized_ckpt_name = None

    def evaluate_run(
        self,
        run_folder: Path,
        model: LightningModule,
        test_loader: dict,
        split: str,
        set_optimized_metric: bool = False,
    ):
        """Evaluates all checkpoints pertraining to a run.

        Checks the self.ckpts to evaluate only checkpoints that are specified there.

        :param run_folder: Path to the folder where the checkpoints for a particular run
            are stored.
        :param model: LightningModule object pertaining to the model to evaluate.
        :param test_loader: The test data loaders. Can correspond to multiple test data
            sets.
        :param split: String specifying which split of the data you are currently eval
            on, either 'val', or 'test'.
        :param set_optimized_metric: Bool whether to set the optimized metric during the
            evaluation procedure or leave it. This is useful when evaluating on
            validation, in which case this should be set to report to optuna. This
            should not be set when evaluating on a test data set.
        """
        self.set_optimized_metric = set_optimized_metric
        self.evaluator.split = split
        if self._check_conditions():
            return

        log.info(Fore.MAGENTA + f"Evaluating run at {run_folder}...")

        for strategy_name in self.ckpts.keys():
            if self.ckpts[strategy_name] is None or self.ckpts[strategy_name] is False:
                continue

            if strategy_name == "last":
                self.evaluate_last(run_folder, model, test_loader)
                continue

            strat_subdir = self._get_subdir(run_folder, strategy_name)
            if strat_subdir is None:
                raise ValueError(f"{strategy_name} strategy not found in {run_folder}.")
            self._evaluate_strategy(strat_subdir, model, test_loader)

        if self.set_optimized_metric and self.optimized_ckpt_name is not None:
            logger = self.evaluator.logger
            if (
                logger is not None
                and hasattr(logger, "experiment")
                and hasattr(logger, "run_id")
            ):
                logger.experiment.set_tag(
                    logger.run_id, "optimized_ckpt_name", self.optimized_ckpt_name
                )
            # Log the exact (main, sec) pair reported to Optuna as metrics, so
            # retrained runs are directly comparable to the search-time
            # objective values.
            vals = self.optimized_metric
            pair = list(vals) if isinstance(vals, (list, tuple)) else [vals]
            eval_mlflow.log_metrics_to_mlflow(
                self.evaluator,
                dict(zip(["optimized_main", "optimized_sec"], pair)),
                cb_name="optimized",
            )

    def _evaluate_strategy(self, strategy_folder: Path, model, test_loader: dict):
        """Evaluate all the checkpoints corresponding to a certain strategy.

        Strategy here means the strategy used to checkpoint, i.e., either a checkpoint
        for each of the datasets, or a single checkpoint at the end of the training...
        Expect that every subdir of a strategy folder corresponds to a certain metric
        that was tracked during that run.
        """
        self.strategy_name = strategy_folder.name
        self.evaluator.strat_name = self.strategy_name
        log.info(Fore.GREEN + f"-> Evaluating strategy '{self.strategy_name}'")

        for metric_name in self.ckpts[self.strategy_name].keys():
            metric_subdir = self._get_subdir(strategy_folder, metric_name)
            if metric_subdir is None:
                raise ValueError(
                    f"{metric_name} metric not found in {strategy_folder}. "
                    f"Are you sure you're checkpointing on {metric_name}???"
                )
            self._evaluate_metric(metric_subdir, model, test_loader)

        self.evaluator.strat_name = None

    def _evaluate_metric(self, metric_folder: Path, model, test_loader: dict):
        """Evaluate the ckpts corresponding to a metric in a run.

        Expect that the folders inside metric_ckpts each correspond to a criteria on
        that metric. By criteria we mean, e.g., min, max, etc. on that metric, where the
        checkpoint was saved.
        """
        self.metric_name = metric_folder.name
        self.evaluator.metric_name = self.metric_name
        log.info(Fore.CYAN + f"-> -> Evaluating metric '{self.metric_name}'")

        for criterion_name in self.ckpts[self.strategy_name][self.metric_name]:
            criterion_subdir = self._get_subdir(metric_folder, criterion_name)
            if criterion_subdir is None:
                raise ValueError(
                    f"{criterion_name} criterion not found in {metric_folder}."
                )
            self._evaluate_criterion(criterion_subdir, model, test_loader)

        self.evaluator.metric_name = None

    def _evaluate_criterion(self, criterion_folder: Path, model, test_loader: dict):
        """Evaluate all checkpoints associated with a given criterion.

        Expect that the folder is filled with .ckpt files corresponding to checkpoints
        on different datasets, for example.
        """
        criterion_name = criterion_folder.name
        self.evaluator.criterion_name = criterion_name
        log.info(Fore.BLUE + f"-> -> -> Evaluating criterion '{criterion_name}'")
        for ckpt_path in criterion_folder.glob("*.ckpt"):
            if not is_valid_ckpt(ckpt_path):
                continue

            self.evaluate_ckpt(ckpt_path, model, test_loader)

        # Skip getting an optimized metric if no checkpoints were made for that
        # specific criterion.
        ckpts = list(criterion_folder.glob("*.ckpt"))
        if ckpts != []:
            if self._is_candidate(criterion_name):
                self._get_optimized_metric(self.optimized_metric_config)
            self._make_criterion_summary_plots(criterion_folder)

        self.evaluator.criterion_name = None

    def evaluate_ckpt(self, ckpt_path: Path, model, test_loader: dict):
        """Run test loops on given a model and a datamodule.

        This runs all the callbacks that were given to it on the test data in the given
        datamodule. The callbacks themselves are responsible for plotting their output
        if this is to be plotted.

        :param ckpt_path: Path to a specific checkopint file, must end in '.ckpt'.
        :param model: LightningModule object pertaining to the model to evaluate.
        :param test_loader: The test data loaders. Can correspond to multiple test data
            sets.
        """
        log.info(f"-> -> -> -> Evaluating checkpoint at {ckpt_path}.")

        state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")[
            "state_dict"
        ]
        # Module-level states registered during training (threshold buffers,
        # the SVDD center) are absent from an eval-only model (train=false).
        model_keys = set(model.state_dict())
        for key, value in state_dict.items():
            if key not in model_keys and "." not in key and torch.is_tensor(value):
                model.register_buffer(key, value.clone(), persistent=True)
        model.load_state_dict(state_dict, strict=True)
        model._ckpt_path = ckpt_path

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        self.evaluator.test(model=model, dataloaders=test_loader, verbose=False)

    def evaluate_last(self, run_folder: Path, model, test_loaders):
        """Evaluate the checkpoint taken at the last epoch."""
        log.info(Fore.GREEN + f"-> Evaluating strategy 'last'")
        self.evaluator.strat_name = "last"
        ckpt_path = run_folder / "last.ckpt"
        self.evaluate_ckpt(ckpt_path, model, test_loaders)
        self._make_criterion_summary_plots(run_folder)
        self.evaluator.strat_name = None

    def _get_subdir(self, main_dir: Path, subdir: str):
        """Checks if a dir exists and returns it if true."""
        subdir = main_dir / subdir
        if not subdir.is_dir():
            return None

        return subdir

    def _check_conditions(self) -> bool:
        """Checks if the criteria for running the evaluation are met."""
        if not self.callbacks:
            log.warning(
                Fore.MAGENTA + "No eval callbacks given. Skipping evaluation..."
            )
            return True

        return False

    def _is_candidate(self, criterion_name: str) -> bool:
        if self.optimized_metric_config is None or not self.set_optimized_metric:
            return False
        candidate = self.optimized_metric_config["candidate"]
        return (self.strategy_name, self.metric_name, criterion_name) == (
            candidate["strategy"],
            candidate["metric"],
            candidate["criterion"],
        )

    def _get_optimized_metric(self, optimized_metric_config: dict):
        """Set the metric reported to optuna from the candidate criterion folder.

        The 'main_metric' entry names an evaluator callback that implements
        'get_optimized_metric' together with its parameters; that callback picks the
        checkpoint within the folder, and 'sec_metric', if given, is read at the same
        checkpoint. A missing or non-finite value leaves optimized_metric None.
        """
        main_cfg = optimized_metric_config["main_metric"]
        main_cb_name = main_cfg["callback"]["name"]
        main_cb_params = main_cfg["callback"]["params"]
        ckpt_name, main_metric = self._get_metric(main_cb_name, main_cb_params)

        metrics = [main_metric]
        sec_cfg = optimized_metric_config.get("sec_metric")
        if sec_cfg is not None:
            metrics.append(self._get_secondary_metric(sec_cfg, ckpt_name)[0])

        if not all(m is not None and math.isfinite(m) for m in metrics):
            log.warning(f"Optimized metric {metrics} at {ckpt_name} is not finite.")
            return

        self.optimized_metric = metrics if sec_cfg is not None else main_metric
        self.optimized_ckpt_name = ckpt_name

    def _get_metric(self, callback_name: str, callback_params: dict):
        """Get callback corresponding to metric and optim direction."""
        available_callbacks = {
            cb.name: cb for cb in self.evaluator.callbacks if hasattr(cb, "name")
        }

        try:
            cb = available_callbacks[callback_name]
        except KeyError:
            raise ValueError(f"Callback {callback_name} not available.")

        ckpt_name, metric_value = cb.get_optimized_metric(**callback_params)

        return ckpt_name, metric_value

    def _get_secondary_metric(self, sec_cfg: dict, ckpt_name: str):
        """Get the secondary metric value.

        Evaluate this metric at the checkpoint that the main metric was evaluated,
        specified by ckpt_name.
        """
        sec_cb_name = sec_cfg["callback"]["name"]
        sec_cb_params = dict(sec_cfg["callback"].get("params", {}))
        sec_optim_dir = sec_cfg["direction"]

        # Fill ckpt_name from main callback if missing/None.
        if sec_cb_params.get("ckpt_name") is None:
            sec_cb_params["ckpt_name"] = ckpt_name

        _, sec_metric = self._get_metric(sec_cb_name, sec_cb_params)
        return sec_metric, sec_optim_dir

    def _make_criterion_summary_plots(self, plot_folder: Path):
        """Make summary plots for each callback.

        Check if any of the callbacks have 'plot_summary' method and if they do call it
        and store result in given folder.
        """
        # Make summary plots for callback across evaluated checkpoints.
        for cb in self.evaluator.callbacks:
            method_name = "plot_summary"
            summary_method = getattr(cb, method_name, None)
            if callable(summary_method):
                summary_method(self.evaluator, plot_folder)
                cb.clear_crit_summary()

    def evaluate_lattice(
        self,
        scores_dir: Path,
        out_dir: Path,
        lattice: str,
        data_dir: str | None = None,
        workers: int = 1,
        tracking_uri: str | None = None,
    ) -> list[dict]:
        """Evaluate every run of a lattice by evaluate_dump; return their run.json.

        The runs are the directories of ``<scores_dir>/<lattice>`` that hold a
        ``meta.pt``, in name order. Each run's module and datamodule are rebuilt from
        its ``config.yaml``, with ``data.data_dir`` replaced by ``data_dir`` when
        given, in a spawn pool of ``workers`` single-threaded processes.
        """
        lattice_dir = Path(scores_dir) / lattice
        runs = sorted(p for p in lattice_dir.iterdir() if (p / "meta.pt").is_file())
        # Created once, before the pool: workers creating one name at once make
        # duplicate experiments.
        if tracking_uri is not None:
            _mlflow_experiment(tracking_uri, f"{lattice}_eval")
        # The run configs read ${oc.env:PROJECT_ROOT}; spawned workers inherit it.
        os.environ.setdefault("PROJECT_ROOT", str(ROOT))
        if workers > 1:
            _prebuild_sds(runs[0], data_dir, self.extra_sets.get("sds"))
        common = (out_dir, lattice, data_dir, tracking_uri)
        tasks = [(self._worker_settings(), run, *common) for run in runs]
        return _map(_lattice_task, tasks, workers)

    def _worker_settings(self) -> dict:
        """Arguments that build this Evaluator's dump path in a worker process."""
        return {
            "dump_callbacks": self.dump_callbacks,
            "test_callbacks": self.test_callbacks,
            "extra_sets": self.extra_sets,
            "selection": self.selection,
            "logger": False,
            "accelerator": "cpu",
            "devices": 1,
        }

    def evaluate_dump(
        self,
        run_dir: Path,
        module: LightningModule,
        datamodule: LightningDataModule,
        out_dir: Path,
        lattice: str | None = None,
        tracking_uri: str | None = None,
    ) -> dict:
        """Evaluate every epoch of one dumped run; write its tables, return run.json.

        ``initial.pt`` (epoch -1) and every ``epoch_<nnnn>.pt`` of ``run_dir`` are
        replayed through ``module`` (``src.evaluation.replay``) in two passes, each on
        a fresh CPU trainer with fresh callbacks and no plots: the validation loaders
        of ``datamodule`` followed by the extra sets, under every callback of
        dump_callbacks, and the test loaders, under those of test_callbacks. The
        scalars the callbacks record are named '<callback>/<scalar>'. Written to
        ``<out_dir>/runs/<run>/``, one row per epoch, -1 first, in the tables:

        - ``val.csv``: ``epoch``; every scalar of the validation pass; the rows of
          the registry's ``labelled``, the mean or the CVaR of the efficiencies
          (columns ``<efficiency><set>``) of the signal sets, those whose labels are
          all positive; ``ema/<strategy>`` for every criterion, labelled row and
          ``q_sec``, the EMA of the training callbacks over epochs >= 0, empty at -1.
        - ``test.csv``: ``epoch``; every scalar of the test pass; ``test_mean_eff``,
          the mean efficiency of the signal sets; ``realised_fpr``, the pass rate of
          ``normal``; ``null_rate``, the mean pass rate of the efficiency callback's
          ``null_ds``, empty without them.
        - ``picks.csv``: ``strategy``, ``k``, ``epoch``. Every criterion has a row per
          k in 0..W_STEPS, the epoch rule ``epoch_pick`` of
          src.callbacks.metrics.selection with Q'' the registry's ``q_sec`` column;
          every labelled row and ``q_sec`` have one, k empty, the best of the logged
          EMA of Q' (``best_epoch``). Epoch -1 never enters.
        - ``run.json``: ``run``, ``lattice``, ``dump``, ``hyperpoint``,
          ``mlflow_run_id``, ``epochs`` (trained), ``module``, ``signals``, ``nulls``,
          ``stored``.

        With ``tracking_uri``, a run named after the dump's run in the MLflow
        experiment ``<lattice>_eval`` receives the hyperpoint as parameters and, at
        step = epoch for every epoch >= 0, every strategy's EMA under its ``key``, its
        value under its ``raw_key`` where the registry gives one, and every signal
        set's validation efficiency under the key of the training efficiency callback.
        """
        if self.selection is None:
            raise ValueError("The dump path needs the selection registry.")
        run_dir = Path(run_dir)
        out = Path(out_dir) / "runs" / run_dir.name
        meta = torch.load(run_dir / "meta.pt", weights_only=True)
        replay, trainers = ReplayModule(module, meta), self._dump_trainers()
        rows = self._replay_run(run_dir, replay, datamodule, trainers, out)
        nulls = _null_sets(trainers["test"].callbacks)
        run = _run_summary(run_dir, replay.meta, lattice, nulls)
        tables = self._tables(rows, replay.meta, nulls)
        if tracking_uri is not None:
            run["mlflow_run_id"] = self._log_run(tables["val"], run, tracking_uri)
        _write_outputs(out, tables, run)
        return run

    def _dump_trainers(self) -> dict:
        keys = {"val": list(self.dump_callbacks), "test": self.test_callbacks}
        return {
            split: _dump_trainer(self.dump_callbacks, names, split)
            for split, names in keys.items()
        }

    def _replay_run(self, run_dir: Path, replay, datamodule, trainers, out: Path):
        """The scalars recorded by both passes, a row per epoch file of the run."""
        loaders = self._dump_loaders(replay.meta, datamodule)
        rows = {split: [] for split in trainers}
        for epoch, path in enumerate(_epoch_files(run_dir), start=-1):
            replay.load_epoch(_load_record(path, epoch), out / path.name)
            for split, trainer in trainers.items():
                records = _replay_pass(trainer, replay, split, loaders[split])
                rows[split].append({"epoch": epoch, **records})
        return rows

    def _dump_loaders(self, meta: dict, datamodule) -> dict[str, dict]:
        """The validation loaders followed by the extra sets, and the test loaders."""
        datamodule.setup("fit")
        val = {**datamodule.val_dataloader(), **self._extra_loaders(meta, datamodule)}
        datamodule.setup("test")
        return {"val": val, "test": datamodule.test_dataloader()}

    def _extra_loaders(self, meta: dict, datamodule) -> dict:
        sets, data_train = self.extra_sets, datamodule.data_train
        sds = sets.get("sds")
        if sds:
            sds = _cached_sds(data_train.tensors[0], sds["max_rows"], sds["row_seed"])
        train = data_train if sets.get("train") else None
        return extra_loaders(meta, train, sds, aux=bool(sets.get("aux")))

    def _tables(self, rows: dict, meta: dict, nulls: list[str]) -> dict:
        """The rows of val.csv, test.csv and picks.csv."""
        selection, strategies = self.selection, _strategies(self.selection)
        signals = _signal_sets(meta["val"])
        val = [{**r, **_labelled_rows(r, signals, selection)} for r in rows["val"]]
        _add_emas(val, strategies)
        signals, prefix = _signal_sets(meta["test"]), selection["efficiency"]
        test = [{**r, **_test_summary(r, signals, nulls, prefix)} for r in rows["test"]]
        picks = _picks(val, strategies, selection["criteria"])
        return {"val": val, "test": test, "picks": picks}

    def _log_run(self, val_rows: list[dict], run: dict, tracking_uri: str) -> str:
        experiment_id = _mlflow_experiment(tracking_uri, f"{run['lattice']}_eval")
        strategies, prefix = _strategies(self.selection), self.selection["efficiency"]
        columns = _mlflow_columns(strategies, run["signals"], prefix)
        metrics = _mlflow_metrics(val_rows, columns)
        return _log_mlflow(tracking_uri, experiment_id, run, metrics)
