# Evaluator object definition. Produces the desired evaluation plots for given model
# checkpoints saved during the training of a model.
import logging
import operator
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning
import torch
from colorama import Back, Fore, Style
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

from src.utils.checkpoints import is_valid_ckpt


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
    :param trainer_kwargs: The other arguments that are given to the trainer object,
        kept the same as the training, e.g., which device to run the computations on.
    """

    def __init__(
        self,
        ckpts: dict = None,
        callbacks: Optional[list[Callback]] = None,
        logger: Optional[Logger] = None,
        optimized_metric_config: dict = None,
        optimized_metric_configs: dict = None,
        **trainer_kwargs,
    ):
        self.ckpts = ckpts or {}
        self.callbacks = callbacks or []
        self.optimized_metric_config = optimized_metric_config or None
        self.optimized_metric_configs = optimized_metric_configs or None
        if self.optimized_metric_config is not None and self.optimized_metric_configs is not None:
            raise ValueError(
                "Configure either optimized_metric_config or optimized_metric_configs, not both."
            )

        # Build a *real* Trainer under the hood
        self.evaluator = pytorch_lightning.Trainer(
            logger=logger,
            callbacks=self.callbacks,
            **trainer_kwargs,
        )
        self._all_evaluator_callbacks = list(self.evaluator.callbacks)

        self.evaluator.object_feature_map_1d = None
        self.evaluator.split = None
        self.evaluator.strat_name = None
        self.evaluator.metric_name = None
        self.evaluator.criterion_name = None
        self.optimized_metric = None
        self.optimized_ckpt_name = None
        self.optimized_metrics = {}
        self.optimized_ckpt_names = {}
        if self.optimized_metric_configs is not None:
            self.optimized_metrics = {str(name): None for name in self.optimized_metric_configs}
            self.optimized_ckpt_names = {str(name): None for name in self.optimized_metric_configs}
        self.evaluator.ckpt_path_name = None

    def evaluate_run(
        self,
        run_folder: Path,
        model: LightningModule,
        test_loader: dict,
        split: str,
        set_optimized_metric: bool = False,
    ):
        """Evaluate all checkpoints pertaining to a run.

        Checks the self.ckpts to evaluate only checkpoints that are specified there.

        :param run_folder: Path to the folder where the checkpoints for a particular run are
            stored.
        :param model: LightningModule object pertaining to the model to evaluate.
        :param test_loader: The test data loaders. Can correspond to multiple test data sets.
        :param split: String specifying which split of the data is being evaluated, either 'val',
            or 'test'.
        :param set_optimized_metric: Bool whether to set the optimized metric during the evaluation
            procedure or leave it. This is useful when evaluating on validation, in which case this
            should be set to report to optuna. This should not be set when evaluating on a test
            data set.
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

        if self.set_optimized_metric:
            self._log_optimized_metrics()

    def _log_optimized_metrics(self) -> None:
        """Log selected checkpoint identities and objective values when possible."""
        if self.optimized_metric_configs is None:
            selections = (
                {} if self.optimized_ckpt_name is None else {"optimized": self.optimized_ckpt_name}
            )
            metrics = {} if self.optimized_metric is None else {"optimized": self.optimized_metric}
        else:
            selections = self.optimized_ckpt_names
            metrics = self.optimized_metrics

        if not selections:
            return

        logger = self.evaluator.logger
        if logger is None:
            return
        if hasattr(logger, "experiment") and hasattr(logger, "run_id"):
            for name, checkpoint in selections.items():
                if checkpoint is None:
                    continue
                logger.experiment.set_tag(
                    logger.run_id,
                    f"optimized_ckpt/{name}",
                    str(checkpoint),
                )
        if hasattr(logger, "log_metrics"):
            values = {}
            for name, objective in metrics.items():
                if objective is None:
                    continue
                pair = objective if isinstance(objective, (list, tuple)) else [objective]
                for index, value in enumerate(pair):
                    values[f"optimized/{name}/objective_{index}"] = float(value)
            if values:
                logger.log_metrics(values)

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
            if self.ckpts[self.strategy_name][metric_name] in (None, False):
                continue
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

        Expect that the folders inside metric_ckpts each correspond to a criteria on that metric.
        By criteria we mean, e.g., min, max, etc. on that metric, where the checkpoint was saved.
        """
        self.metric_name = metric_folder.name
        self.evaluator.metric_name = self.metric_name
        log.info(Fore.CYAN + f"-> -> Evaluating metric '{self.metric_name}'")

        for criterion_name in self.ckpts[self.strategy_name][self.metric_name]:
            criterion_subdir = self._get_subdir(metric_folder, criterion_name)
            if criterion_subdir is None:
                raise ValueError(f"{criterion_name} criterion not found in {metric_folder}.")
            self._evaluate_criterion(criterion_subdir, model, test_loader)

        self.evaluator.metric_name = None

    def _evaluate_criterion(self, criterion_folder: Path, model, test_loader: dict):
        """Evaluate all checkpoints associated with a given criterion.

        Expect that the folder is filled with .ckpt files corresponding to checkpoints on different
        datasets, for example.
        """
        criterion_name = criterion_folder.name
        self.evaluator.criterion_name = criterion_name
        log.info(Fore.BLUE + f"-> -> -> Evaluating criterion '{criterion_name}'")
        evaluated = False
        for ckpt_path in criterion_folder.glob("*.ckpt"):
            if not is_valid_ckpt(ckpt_path):
                continue

            self.evaluator.ckpt_path_name = ckpt_path.name
            self.evaluate_ckpt(ckpt_path, model, test_loader)
            evaluated = True

        # Skip getting an optimized metric if no checkpoints were made for that
        # specific criterion.
        if evaluated:
            self._update_optimized_metrics()
            self._make_criterion_summary_plots(criterion_folder)

        self.evaluator.criterion_name = None

    def evaluate_ckpt(self, ckpt_path: Path, model, test_loader: dict):
        """Run test loops on given a model and a datamodule.

        This runs all the callbacks that were given to it on the test data in the given datamodule.
        The callbacks themselves are responsible for plotting their output if this is to be
        plotted.

        :param ckpt_path: Path to a specific checkopint file, must end in '.ckpt'.
        :param model: LightningModule object pertaining to the model to evaluate.
        :param test_loader: The test data loaders. Can correspond to multiple test data sets.
        """
        log.info(f"-> -> -> -> Evaluating checkpoint at {ckpt_path}.")
        self._activate_candidate_callbacks()

        state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model._ckpt_path = ckpt_path

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        self.evaluator.test(model=model, dataloaders=test_loader, verbose=False)

    def _activate_candidate_callbacks(self) -> None:
        """Run only callbacks needed by the current shared-pool candidate."""
        if self.optimized_metric_configs is None or not self.set_optimized_metric:
            self.evaluator.callbacks = list(self._all_evaluator_callbacks)
            return
        candidate = self._candidate_path()
        callback_names = set()
        for config in self.optimized_metric_configs.values():
            if not self._candidate_enabled(config, candidate):
                continue
            callback_names.add(config["main_metric"]["callback"]["name"])
            secondary = config.get("sec_metric")
            if secondary is not None:
                callback_names.add(secondary["callback"]["name"])
        self.evaluator.callbacks = [
            callback
            for callback in self._all_evaluator_callbacks
            if not hasattr(callback, "name") or callback.name in callback_names
        ]

    def evaluate_last(self, run_folder: Path, model, test_loaders):
        """Evaluate the checkpoint taken at the last epoch."""
        log.info(Fore.GREEN + "-> Evaluating strategy 'last'")
        self.evaluator.strat_name = "last"
        ckpt_path = run_folder / "last.ckpt"
        self.evaluate_ckpt(ckpt_path, model, test_loaders)
        self._update_optimized_metrics()
        self._make_criterion_summary_plots(run_folder)

    def _get_subdir(self, main_dir: Path, subdir: str):
        """Checks if a dir exists and returns it if true."""
        subdir = main_dir / subdir
        if not subdir.is_dir():
            return None

        return subdir

    def _check_conditions(self) -> bool:
        """Checks if the criteria for running the evaluation are met."""
        if not self.callbacks:
            log.warn(Fore.MAGENTA + "No eval callbacks given. Skipping evaluation...")
            return True

        return False

    def _get_optimized_metric(self, optimized_metric_config: dict | None):
        """Get one metric across all the checkpoints in a run to optimize for.

        This is used for hyperparameter optimization. The 'optimized_metric_config' dictionary
        contains the name of a callback in the evaluator. This callback should have
        'get_optimized_metric' method implemented. This method returns what is considered the best
        metric, according to its own definition, across all the checkpoints within a checkpointing
        criterion. The optimized_metric_config should also contain all the hyperparameters required
        by the 'get_optimized_metric' method. Finally, it should contain a direction along which to
        pick from best checkpoints given by different criteria.
        """
        if optimized_metric_config is None or self.set_optimized_metric is False:
            return

        main_cfg = optimized_metric_config["main_metric"]
        main_cb_name = main_cfg["callback"]["name"]
        main_cb_params = main_cfg["callback"]["params"]
        main_optim_dir = main_cfg["direction"]
        ckpt_name, main_metric = self._get_metric(main_cb_name, main_cb_params)

        if main_metric is None:
            log.warn("Main metric is None. Optimized HP metric not updated this strategy.")
            return
        elif optimized_metric_config.get("sec_metric") is not None:
            sec_cfg = optimized_metric_config["sec_metric"]
            sec_metric, sec_optim_dir = self._get_secondary_metric(sec_cfg, ckpt_name)
            optimized_metrics = [main_metric, sec_metric]
            optim_directions = [main_optim_dir, sec_optim_dir]
        else:
            self._set_best_across_crit(
                main_metric,
                ckpt_name=ckpt_name,
                direction=main_optim_dir,
            )
            return

        if sec_metric is None:
            log.warn("Sec metric is None. Optimized HP metric not updated this strategy.")
            return

        self._comp_best_across_crit(
            optimized_metrics,
            ckpt_name=ckpt_name,
            directions=optim_directions,
        )

    def _update_optimized_metrics(self) -> None:
        """Update either the legacy objective or all shared-pool objectives."""
        if self.optimized_metric_configs is None:
            self._get_optimized_metric(self.optimized_metric_config)
            return
        if self.set_optimized_metric is False:
            return

        candidate = self._candidate_path()
        for name, config in self.optimized_metric_configs.items():
            if not self._candidate_enabled(config, candidate):
                continue
            self._update_named_optimized_metric(str(name), config)
        self._sync_flattened_optimized_metric()

    def _candidate_path(self) -> str:
        """Return the stable logical path of the checkpoint currently being evaluated."""
        strategy = getattr(self.evaluator, "strat_name", None)
        if strategy == "last":
            return "last"
        metric = getattr(self.evaluator, "metric_name", None)
        criterion = getattr(self.evaluator, "criterion_name", None)
        if strategy is None or metric is None or criterion is None:
            raise RuntimeError("Cannot identify the current checkpoint candidate.")
        return f"{strategy}/{metric}/{criterion}"

    @staticmethod
    def _candidate_enabled(config: dict, candidate: str) -> bool:
        """Restrict one objective to last, its primary EMA, and stable native."""
        if candidate in {"last", "single/ascore_operational/stable"}:
            return True
        primary = config.get("primary_checkpoint_metric")
        if primary is None:
            raise ValueError(
                "Each optimized_metric_configs entry requires primary_checkpoint_metric."
            )
        direction = config["main_metric"]["direction"]
        criterion = "max" if direction == "maximize" else "min"
        return candidate == f"summary/{primary}/{criterion}"

    def _update_named_optimized_metric(self, name: str, config: dict) -> None:
        """Apply the paper checkpoint rule to one logical validation strategy."""
        main_cfg = config["main_metric"]
        ckpt_name, main_metric = self._get_metric(
            main_cfg["callback"]["name"],
            main_cfg["callback"].get("params", {}),
        )
        if main_metric is None:
            return

        sec_cfg = config.get("sec_metric")
        if sec_cfg is None:
            values = main_metric
            directions = [main_cfg["direction"]]
        else:
            sec_metric, sec_direction = self._get_secondary_metric(sec_cfg, ckpt_name)
            if sec_metric is None:
                return
            values = [main_metric, sec_metric]
            directions = [main_cfg["direction"], sec_direction]

        incumbent = self.optimized_metrics[name]
        if incumbent is None or self._contender_wins(values, incumbent, directions):
            self.optimized_metrics[name] = values
            self.optimized_ckpt_names[name] = ckpt_name

    @staticmethod
    def _contender_wins(values, incumbent, directions: list[str]) -> bool:
        """Return whether the fixed 1.25 Q' + Q'' relative-change rule wins."""
        contender = values if isinstance(values, (list, tuple)) else [values]
        current = incumbent if isinstance(incumbent, (list, tuple)) else [incumbent]
        if len(contender) != len(current) or len(contender) != len(directions):
            raise ValueError("Objective values and directions must have equal lengths.")
        if any(value is None for value in [*contender, *current]):
            return False
        relative_changes = [
            ((new - old) if direction == "maximize" else (old - new)) / max(abs(old), 1.0e-12)
            for new, old, direction in zip(contender, current, directions)
        ]
        weights = [1.25, *([1.0] * (len(relative_changes) - 1))]
        return sum(weight * change for weight, change in zip(weights, relative_changes)) > 0

    def _sync_flattened_optimized_metric(self) -> None:
        """Expose the deterministic flattened objective vector to Hydra/Optuna."""
        flattened = []
        for name in self.optimized_metric_configs:
            value = self.optimized_metrics[str(name)]
            if value is None:
                self.optimized_metric = None
                self.optimized_ckpt_name = None
                return
            if isinstance(value, (list, tuple)):
                flattened.extend(value)
            else:
                flattened.append(value)
        self.optimized_metric = tuple(flattened)
        self.optimized_ckpt_name = None

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

        Evaluate this metric at the checkpoint that the main metric was evaluated, specified by
        ckpt_name.
        """
        sec_cb_name = sec_cfg["callback"]["name"]
        sec_cb_params = dict(sec_cfg["callback"].get("params", {}))
        sec_optim_dir = sec_cfg["direction"]

        # Fill the checkpoint identifier expected by the secondary callback.
        if "ckpt_ds" in sec_cb_params:
            if sec_cb_params["ckpt_ds"] is None:
                sec_cb_params["ckpt_ds"] = ckpt_name
        elif sec_cb_params.get("ckpt_name") is None:
            sec_cb_params["ckpt_name"] = ckpt_name

        _, sec_metric = self._get_metric(sec_cb_name, sec_cb_params)
        return sec_metric, sec_optim_dir

    def _set_best_across_crit(self, value: float, *, ckpt_name: str, direction: str):
        """Set the best optimized metric values across all ckpt criteria."""
        if self.optimized_metric is None:
            self.optimized_metric = value
            self.optimized_ckpt_name = ckpt_name
            return

        if direction == "maximize":
            if value > self.optimized_metric:
                self.optimized_metric = value
                self.optimized_ckpt_name = ckpt_name
            return

        if direction == "minimize":
            if value < self.optimized_metric:
                self.optimized_metric = value
                self.optimized_ckpt_name = ckpt_name
            return

        raise ValueError("Optimized metric direction must be 'maximize' or 'minimize'.")

    def _comp_best_across_crit(
        self, values: list[float], *, ckpt_name: str, directions: list[str]
    ):
        """Update best composite metric across checkpoint criteria.

        This is computing the relative change for the main metric and secondary metric. If the sum
        of the relative changes is positive, i.e., a metric's improvement is larger than the other
        metric's worsening, then save this as the new best across checkpointing criteria.
        """
        if any(v is None for v in values):
            return

        if self.optimized_metric is None:
            self.optimized_metric = values
            self.optimized_ckpt_name = ckpt_name
            return

        rel_changes = [
            ((v - b) if d == "maximize" else (b - v)) / max(abs(b), 1e-12)
            for v, b, d in zip(values, self.optimized_metric, directions)
        ]
        weights = [1.25, 1.0]
        score = sum(w * rc for w, rc in zip(weights, rel_changes))

        if score > 0:
            self.optimized_metric = values
            self.optimized_ckpt_name = ckpt_name

    def get_optimized_ckpt_name(self):
        return self.optimized_ckpt_name

    def _make_criterion_summary_plots(self, plot_folder: Path):
        """Make summary plots for each callback.

        Check if any of the callbacks have 'plot_summary' method and if they do call it and store
        result in given folder.
        """
        # Make summary plots for callback across evaluated checkpoints.
        for cb in self.evaluator.callbacks:
            method_name = "plot_summary"
            summary_method = getattr(cb, method_name, None)
            if callable(summary_method):
                summary_method(self.evaluator, plot_folder)
                cb.clear_crit_summary()
        self.evaluator.callbacks = list(self._all_evaluator_callbacks)
