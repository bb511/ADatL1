# Evaluator object definition. Produces the desired evaluation plots for given model
# checkpoints saved during the training of a model, and evaluates the per-epoch score
# dumps of a lattice through src/evaluation/dump.py (evaluate_lattice, evaluate_dump).
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pytorch_lightning
import torch
from colorama import Fore
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

from src.evaluation import dump
from src.evaluation.callbacks.utils import mlflow as eval_mlflow
from src.utils.checkpoints import is_valid_ckpt


def _lists_candidate(ckpts: dict, candidate: dict) -> bool:
    metrics = ckpts.get(candidate["strategy"])
    if not hasattr(metrics, "get"):
        return False
    return candidate["criterion"] in (metrics.get(candidate["metric"]) or [])


def _container(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


class Evaluator:
    """Thin wrapper around pl.Trainer that performs the evaluation of a model.

    The test loops are ran for this custom trainer object instantiated in this class.
    The callbacks used here are different from the callbacks used at training time and
    are contained in src/evaluation/callbacks. The dump path (evaluate_lattice,
    evaluate_dump) runs in src/evaluation/dump.py on dump_callbacks, test_callbacks,
    extra_sets and selection alone.

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
            # that the objective recorded in the study can also be read from MLflow.
            vals = self.optimized_metric
            pair = list(vals) if isinstance(vals, (list, tuple)) else [vals]
            eval_mlflow.log_metrics_to_mlflow(
                self.evaluator,
                dict(zip(["optimized_main", "optimized_sec"], pair)),
                cb_name="optimized",
            )

    def _evaluate_strategy(self, strategy_folder: Path, model, test_loader: dict):
        """Evaluate all the checkpoints corresponding to a certain strategy.

        The first folder level of a run's checkpoints is the strategy: `summary` for the
        strategy metrics, `normal` for the loss baseline. Every subdirectory of a
        strategy folder is a metric tracked during that run.
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

        The folder holds the .ckpt its ModelCheckpoint keeps (save_top_k=1), and every
        .ckpt in it is evaluated.
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
        log.info(Fore.GREEN + "-> Evaluating strategy 'last'")
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
        sets: str | None = None,
        workers: int = 1,
        tracking_uri: str | None = None,
    ) -> list[dict]:
        """Evaluate every run of a lattice by ``dump.evaluate_lattice``."""
        settings = self._dump_settings()
        return dump.evaluate_lattice(
            scores_dir,
            out_dir,
            lattice,
            settings,
            data_dir,
            sets,
            workers,
            tracking_uri,
        )

    def evaluate_dump(
        self,
        run_dir: Path,
        module: LightningModule,
        datamodule: LightningDataModule,
        out_dir: Path,
        lattice: str | None = None,
        tracking_uri: str | None = None,
    ) -> dict:
        """Evaluate one dumped run by ``dump.evaluate_dump``, which lists its files."""
        settings = self._dump_settings()
        return dump.evaluate_dump(
            run_dir, module, datamodule, out_dir, lattice, tracking_uri, **settings
        )

    def _dump_settings(self) -> dict:
        if self.selection is None:
            raise ValueError("The dump path needs the selection registry.")
        return {
            "dump_callbacks": self.dump_callbacks,
            "test_callbacks": self.test_callbacks,
            "extra_sets": self.extra_sets,
            "selection": self.selection,
        }
