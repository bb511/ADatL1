# Evaluator object definition. Produces the desired evaluation plots for given model
# checkpoints saved during the training of a model.
from typing import Optional
from collections import defaultdict
from pathlib import Path
import re

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

    :param metric_criteria: Dictionary containing the name of each metric for which
        checkpoints have been saved during training and one wants to produce the eval
        plots; to each metric, there corresponds a set of criteria for which these
        checkpoints were saved, e.g., 'min', 'max', etc...
    :param strategy: String specifying the strategy used to checkpoint, e.g., single
        dataset, leave one out, etc...
    :param callbacks: List of pl.Callback objects that compute evaluation metrics on
        the test data and do all the handling of these, including plotting.
    :param logger: pl.Logger object that handles the logging for this evaluation stage.
    :param trainer_kwargs: The other arguments that are given to the trainer object,
        kept the same as the training, e.g., which device to run the computations on.
    """

    def __init__(
        self,
        metric_criteria: dict = None,
        strategy: str = None,
        callbacks: Optional[list[Callback]] = None,
        logger: Optional[Logger] = None,
        **trainer_kwargs,
    ):
        self.strategy = strategy
        self.metric_criteria = metric_criteria
        self.callbacks = callbacks or []

        # Build a *real* Trainer under the hood
        self.evaluator = pytorch_lightning.Trainer(
            **trainer_kwargs, logger=logger, callbacks=self.callbacks
        )

    def evaluate(
        self,
        ckpt_path: Path,
        model: LightningModule,
        datamodule: LightningDataModule,
    ) -> list[dict[str, float]]:
        """Run test loops on given a model and a datamodule.

        This runs all the callbacks that were given to it on the test data in the
        given datamodule. The callbacks themselves are responsible for plotting their
        output if this is to be plotted.
        """
        datamodule.setup("test")

        test_loaders = datamodule.test_dataloader()

        # Delegate to the test loop of a pl trainer object.
        self.evaluator.test(ckpt_path=ckpt_path, model=model, dataloaders=test_loaders)

    def evaluate_run(
        self, run_ckpts: Path, model: LightningModule, datamodule: LightningDataModule,
    ):
        """Computes the evaluation metrics for all the checkpoints inside a folder.

        Assume that the checkpoints are each corresponding to a particular data set.
        """
        self._check_conditions()

        datamodule.setup("test")
        test_loaders = datamodule.test_dataloader()

        ckpt_paths = self._get_ckpt_paths(run_ckpts)
        for metric, criteria in ckpt_paths.items():
            log.info(Fore.GREEN + f"Evaluating checkpoints for metric: {metric}...")
            for criterion, paths in criteria.items():
                log.info(f"-> Evaluating criterion {criterion}")
                for ckpt_path in paths:
                    self.evaluator.test(
                        ckpt_path=ckpt_path,
                        model=model,
                        dataloaders=test_loaders,
                        verbose=False
                    )

    def _get_ckpt_paths(self, ckpt_dir: Path) -> dict[list[Path]]:
        """Finds the ckpts corresponding to given metrics and respective criteria."""
        ckpt_paths = defaultdict(lambda: defaultdict(list))

        strat_dir = self._get_subdir(ckpt_dir, self.strategy)
        for metric in self.metric_criteria.keys():
            metric_dir = self._get_subdir(strat_dir, metric)
            for criterion in self.metric_criteria[metric]:
                criterion_dir = self._get_subdir(metric_dir, criterion)
                for ckpt_filepath in criterion_dir.glob('*.ckpt'):
                    ckpt_paths[metric][criterion].append(ckpt_filepath)

        return ckpt_paths

    def _get_subdir(self, main_dir: Path, subdir: str):
        """Checks if a dir exists and returns it if true."""
        subdir = main_dir / subdir
        if not subdir.is_dir():
            raise FileNotFoundError(f"Folder not found in {main_dir} for {subdir}.")

        return subdir

    def _check_conditions(self):
        """Checks if the criteria for running the evaluation are met."""
        if not self.callbacks:
            log.warn(Fore.MAGENTA + "No eval callbacks given. Skipping evaluation...")
            return
        if not self.metric_criteria:
            log.warn(Fore.MAGENTA + "No metric_criteria given. Skipping evaluation...")
            return

    def _extract_dataset_name(self, ckpt_path: Path):
        """Extracts the dataset name from the name of a checkpoint file.

        Expects the file to have 'ds=[ds_name]' at some point in its name string.
        """
        ckpt_filename = ckpt_path.stem
        match = re.search(r"ds=(.*?)__", ckpt_filename)
        if not match:
            raise ValueError(f"Dataset name not found in {ckpt_filename}")

        ds_name = match.group(1)
        return ds_name
