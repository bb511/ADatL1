# Evaluator object definition. Produces the desired evaluation plots for given model
# checkpoints saved during the training of a model.
from typing import Optional
from collections import defaultdict
from pathlib import Path
import logging

import torch
import numpy as np
import pytorch_lightning
from pytorch_lightning.loggers import Logger
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import Callback
from colorama import Fore, Back, Style

from src.utils import pylogger
log = pylogger.RankedLogger(__name__)


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
        This is used for speciyfing which checkpoints to actually evaluate.
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
        **trainer_kwargs,
    ):
        self.ckpts = ckpts or {}
        self.callbacks = callbacks or []
        self.optimized_metric_config = optimized_metric_config or None

        # Build a *real* Trainer under the hood
        self.evaluator = pytorch_lightning.Trainer(
            logger=logger,
            callbacks=self.callbacks,
            **trainer_kwargs,
        )

        self.evaluator.strat_name = None
        self.evaluator.metric_name = None
        self.evaluator.criterion_name = None
        self.optimized_metric = None

    def evaluate_run(self, run_folder: Path, model: LightningModule, test_loader: dict):
        """Evaluates all checkpoints pertraining to a run.

        Checks the self.ckpts to evaluate only checkpoints that are specified there.

        :param run_folder: Path to the folder where the checkpoints for a particular run
            are stored.
        :param model: LightningModule object pertaining to the model to evaulate.
        :param test_loader: The test data loaders. Can correspond to multiple test
            data sets.
        """
        if self._check_conditions():
            return

        log.info(Fore.MAGENTA + f"Evaluating run at {run_folder}...")

        for strategy_name in self.ckpts.keys():
            if self.ckpts[strategy_name] is None or self.ckpts[strategy_name] is False:
                continue

            if strategy_name == 'last':
                self.evaluate_last(run_folder, model, test_loader)
                continue

            strat_subdir = self._get_subdir(run_folder, strategy_name)
            self._evaluate_strategy(strat_subdir, model, test_loader)

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
            self.evaluate_ckpt(ckpt_path, model, test_loader)

        self._get_criterion_optimized_metric(self.optimized_metric_config)
        self._make_summary_plots(criterion_folder)
        self.evaluator.criterion_name = None

    def evaluate_ckpt(self, ckpt_path: Path, model, test_loader: dict):
        """Run test loops on given a model and a datamodule.

        This runs all the callbacks that were given to it on the test data in the
        given datamodule. The callbacks themselves are responsible for plotting their
        output if this is to be plotted.

        :param ckpt_path: Path to a specific checkopint file, must end in '.ckpt'.
        :param model: LightningModule object pertaining to the model to evaulate.
        :param test_loader: The test data loaders. Can correspond to multiple test
            data sets.
        """
        log.info(f"-> -> -> -> Evaluating checkpoint at {ckpt_path}.")

        state_dict = torch.load(
            ckpt_path, weights_only=False, map_location='cpu'
        )['state_dict']
        model.load_state_dict(state_dict, strict=True)
        model._ckpt_path = ckpt_path

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        self.evaluator.test(model=model, dataloaders=test_loader, verbose=False)

    def evaluate_last(self, run_folder: Path, model, test_loaders):
        """Evaluate the checkpoint taken at the last epoch."""
        log.info(Fore.GREEN + f"-> Evaluating strategy 'last'")
        self.evaluator.strat_name = 'last'
        ckpt_path = run_folder / 'last.ckpt'
        self.evaluate_ckpt(ckpt_path, model, test_loaders)
        self._make_summary_plots(run_folder)

    def _get_subdir(self, main_dir: Path, subdir: str):
        """Checks if a dir exists and returns it if true."""
        subdir = main_dir / subdir
        if not subdir.is_dir():
            raise FileNotFoundError(f"Folder not found in {main_dir} for {subdir}.")

        return subdir

    def _check_conditions(self) -> bool:
        """Checks if the criteria for running the evaluation are met."""
        if not self.callbacks:
            log.warn(Fore.MAGENTA + "No eval callbacks given. Skipping evaluation...")
            return True

        return False

    def _get_criterion_optimized_metric(self, optimized_metric_config: dict | None):
        """Get one metric over all the checkpoints that are being probed for a crit.

        This is used for hyperparameter optimization. The optimized_metric_config
        points to a callback that should have a method 'get_optimized_metric'. Then,
        this method returns one number related to what that callback computes and that
        should be optimized. The input dict to this method should also include all the
        args required by the get_optimized_metric method. Finally, it should include
        the way to pick an optimized_metric across checkpointing criteria, i.e., pick
        the maximum across criteria or the minimum, depending on the nature of the
        optimized metric.
        """
        if optimized_metric_config is None:
            return

        callback_name = list(optimized_metric_config.keys())[0]
        cb_names = [cb.__class__.__name__ for cb in self.evaluator.callbacks]
        if not callback_name in cb_names:
            raise ValueError(f"Callback {callback_name} not in evaluator callbacks.")

        for cb in self.evaluator.callbacks:
            if callback_name == cb.__class__.__name__:
                params = optimized_metric_config[callback_name]
                optimized_metric = cb.get_optimized_metric(**params)

        if optimized_metric_config['direction'] == 'maximize':
            if self.optimized_metric is None:
                self.optimized_metric = -np.inf
            if optimized_metric > self.optimized_metric:
                self.optimized_metric = optimized_metric
        elif optimized_metric_config['direction'] == 'minimize':
            if self.optimized_metric is None:
                self.optimized_metric = np.inf
            if optimized_metric < self.optimized_metric:
                self.optimized_metric = optimized_metric
        else:
            raise ValueError("Direction in optimized_metric_config either max or min.")

    def _make_summary_plots(self, plot_folder: Path):
        """Make summary plots for each callback.

        Check if any of the callbacks have 'plot_summary_reset' method and if they do
        call it and store result in given folder.
        """
        # Make summary plots for callback across evaluated checkpoints.
        for cb in self.evaluator.callbacks:
            method_name = 'plot_summary_reset'
            summary_method = getattr(cb, method_name, None)
            if callable(summary_method):
                summary_method(self.evaluator, plot_folder)
