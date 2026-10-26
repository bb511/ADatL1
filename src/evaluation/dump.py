"""Evaluate the per-epoch score dumps of a lattice, one run at a time."""

import logging
import math
import multiprocessing
import os
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning
import torch
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule

from src.callbacks.components.selection import (
    W_STEPS,
    best_epoch,
    ema,
    epoch_pick,
    smoothed,
)
from src.evaluation.components.pairing import hyperpoint
from src.evaluation.components.rebuild import (
    ROOT,
    SDS_CACHE,
    cached_sds,
    loader_layouts,
    null_sets,
    rebuild,
    signal_sets,
)
from src.evaluation.components.tables import load_record, write_outputs
from src.evaluation.replay import ReplayModule, as_typical, extra_loaders
from src.utils.omegaconf import register_resolvers

# The key of the training efficiency callback for one set's validation efficiency.
SET_EFFICIENCY_KEY = "val/{}/eff__ascore_full__brate_operational"
# The most metrics MLflow accepts in one log_batch call.
MLFLOW_BATCH = 1000
REPLAY_TRAINER = {
    "logger": False,
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "enable_checkpointing": False,
}


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


def _experiment_id(client: MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(name)


class RunEvaluation:
    """The replay of one dumped run through the evaluation callbacks, and its tables.

    Every epoch file, ``initial.pt`` (epoch -1) first, is replayed through the run's
    own module (``src.evaluation.replay``) in two passes, each on a CPU trainer with
    fresh callbacks and no plots: the validation loaders followed by the extra sets
    under every evaluation callback, then the test loaders under the test callbacks.
    The scalars the callbacks record, named '<callback>/<scalar>', make the rows of the
    run's tables. Both trainers carry ``out_dir``, the evaluation directory, where a
    callback may cache what the runs share. The keyword-only arguments are the
    Evaluator's of the same names, apart from ``sets``, the data config whose signals
    and nulls replaced the run's in ``datamodule`` (``rebuild``). With it, the signal
    sets are read from the datamodule's loaders instead of ``meta.pt``: a stored set
    outside the new lists is never served, and a set the dump does not store is
    rescored from the weights.
    """

    def __init__(
        self,
        run_dir: Path,
        module: LightningModule,
        datamodule: LightningDataModule,
        *,
        dump_callbacks: dict,
        test_callbacks: list[str],
        extra_sets: dict,
        selection: dict,
        sets: str | None = None,
        typical: str | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.meta = torch.load(self.run_dir / "meta.pt", weights_only=True)
        self.replay = ReplayModule(module, self.meta, typical)
        self.dump_callbacks, self.extra_sets = dump_callbacks, extra_sets
        self.selection, self.strategies = selection, _strategies(selection)
        self.passes = {"val": list(dump_callbacks), "test": test_callbacks}
        self.trainers = {split: self._trainer(split) for split in self.passes}
        self.loaders = self._loaders(datamodule)
        layouts = self.meta if sets is None else loader_layouts(datamodule)
        self.signals = {split: signal_sets(layouts[split]) for split in self.passes}
        self.nulls = null_sets(self.trainers["test"].callbacks)

    def evaluate(
        self, out_dir: Path, lattice: str | None = None, tracking_uri: str | None = None
    ) -> dict:
        """Write the four files ``evaluate_dump`` lists; return run.json.

        With ``tracking_uri``, the run is also logged to MLflow, as ``<lattice>_eval``.
        """
        out = Path(out_dir) / "runs" / self.run_dir.name
        for trainer in self.trainers.values():
            trainer.out_dir = Path(out_dir)
        rows = self._replay(out)
        run, tables = self._summary(lattice), self._tables(rows)
        if tracking_uri is not None:
            run["mlflow_run_id"] = self._log_mlflow(tracking_uri, tables["val"], run)
        write_outputs(out, tables, run)
        return run

    def _trainer(self, split: str) -> pytorch_lightning.Trainer:
        """A CPU trainer without logger, plots or files, with fresh callbacks."""
        configs = [self.dump_callbacks[name] for name in self.passes[split]]
        callbacks = [hydra.utils.instantiate(config) for config in configs]
        trainer = pytorch_lightning.Trainer(callbacks=callbacks, **REPLAY_TRAINER)
        trainer.plot_checkpoints, trainer.split = False, split
        trainer.strat_name = trainer.metric_name = trainer.criterion_name = None
        return trainer

    def _loaders(self, datamodule: LightningDataModule) -> dict:
        """The validation loaders followed by the extra sets, and the test loaders."""
        datamodule.setup("fit")
        val, data_train = datamodule.val_dataloader(), datamodule.data_train
        sds = self.extra_sets.get("sds")
        if sds:
            sds = cached_sds(data_train.tensors[0], sds["max_rows"], sds["row_seed"])
        train = data_train if self.extra_sets.get("train") else None
        aux = bool(self.extra_sets.get("aux"))
        val = as_typical(val, self.replay.typical)
        val = {**val, **extra_loaders(self.meta, train, sds, aux=aux)}
        datamodule.setup("test")
        return {"val": val, "test": datamodule.test_dataloader()}

    def _epoch_files(self) -> list[Path]:
        return [self.run_dir / "initial.pt", *sorted(self.run_dir.glob("epoch_*.pt"))]

    def _replay(self, out: Path) -> dict[str, list[dict]]:
        """The rows of both passes over every epoch file, -1 first."""
        rows = {split: [] for split in self.trainers}
        for epoch, path in enumerate(self._epoch_files(), start=-1):
            self.replay.load_epoch(load_record(path, epoch), out / path.name)
            for split in self.trainers:
                rows[split].append(self._row(split, epoch))
        return rows

    def _row(self, split: str, epoch: int) -> dict:
        """The epoch and the scalars every callback of a pass records."""
        trainer = self.trainers[split]
        self.replay.serve(split)
        trainer.metric_records = {}
        trainer.test(self.replay, dataloaders=self.loaders[split], verbose=False)
        return {"epoch": epoch, **trainer.metric_records}

    def _tables(self, rows: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """The rows of val.csv, test.csv and picks.csv."""
        val = self._val_rows(rows["val"])
        test = [{**row, **self._test_summary(row)} for row in rows["test"]]
        return {"val": val, "test": test, "picks": self._picks(val)}

    def _labelled(self, row: dict) -> dict[str, float]:
        """The labelled rows of an epoch, mean or CVaR of the signals' efficiencies.

        As the training efficiency callback computes its CVaR: in float32, without the
        non-finite values, the mean of the ceil(alpha n) lowest.
        """
        prefix, values = self.selection["efficiency"], {}
        effs = [row[prefix + s] for s in self.signals["val"]]
        x = torch.as_tensor(effs, dtype=torch.float32)
        x = torch.sort(x[torch.isfinite(x)]).values
        for name, spec in self.selection["labelled"].items():
            if spec["statistic"] not in ("mean", "cvar"):
                raise ValueError(f"Unknown labelled statistic '{spec['statistic']}'.")
            cvar = spec["statistic"] == "cvar"
            k = max(1, math.ceil(spec["alpha"] * x.numel())) if cvar else x.numel()
            values[name] = x[:k].mean().item() if x.numel() else math.nan
        return values

    def _val_rows(self, rows: list[dict]) -> list[dict]:
        """The rows with their labelled rows and ema/<strategy>, None at epoch -1."""
        val = [{**row, **self._labelled(row)} for row in rows]
        trained = [row for row in val if row["epoch"] >= 0]
        for name, spec in self.strategies.items():
            skip = spec.get("skip_nonfinite", False)
            values = [r[spec["column"]] for r in trained]
            averages = iter(ema(values, skip_nonfinite=skip))
            for row in val:
                row[f"ema/{name}"] = next(averages) if row["epoch"] >= 0 else None
        return val

    def _pick(self, name: str, trained: list[dict], k: int | None) -> int:
        """The epoch of the epoch rule at k or, for k None, of its best logged EMA."""
        spec = self.strategies[name]
        skip = spec.get("skip_nonfinite", False)
        q_main = [row[spec["column"]] for row in trained]
        if k is None:
            index = best_epoch(smoothed(q_main, skip_nonfinite=skip), spec["mode"])
        else:
            q_sec = [row[self.strategies["q_sec"]["column"]] for row in trained]
            index = epoch_pick(q_main, q_sec, k, spec["mode"], skip)
        return trained[index]["epoch"]

    def _picks(self, val: list[dict]) -> list[dict]:
        """A row per criterion and k of the w grid, one per other strategy (k None)."""
        trained = [row for row in val if row["epoch"] >= 0]
        criteria, picks = self.selection["criteria"], []
        for name in self.strategies:
            for k in range(W_STEPS + 1) if name in criteria else [None]:
                epoch = self._pick(name, trained, k)
                picks.append({"strategy": name, "k": k, "epoch": epoch})
        return picks

    def _test_summary(self, row: dict) -> dict:
        prefix = self.selection["efficiency"]
        signal_effs = [row[prefix + name] for name in self.signals["test"]]
        null_rates = [row[prefix + name] for name in self.nulls]
        return {
            "test_mean_eff": float(np.mean(signal_effs)),
            "realised_fpr": row[prefix + "normal"],
            "null_rate": float(np.mean(null_rates)) if null_rates else None,
        }

    def _summary(self, lattice: str | None) -> dict:
        return {
            "run": self.run_dir.name,
            "lattice": lattice,
            "dump": str(self.run_dir),
            "hyperpoint": hyperpoint(self.run_dir),
            "mlflow_run_id": None,
            "epochs": len(self._epoch_files()) - 1,
            "module": self.meta["module"],
            "signals": self.signals["val"],
            "nulls": self.nulls,
            "stored": self.meta["stored"],
            "typical": self.replay.typical,
        }

    def _mlflow_columns(self) -> list[tuple[str, str]]:
        """(MLflow key, val.csv column) of every quantity logged per epoch."""
        columns = []
        for name, spec in self.strategies.items():
            columns.append((spec["key"], f"ema/{name}"))
            if spec.get("raw_key"):
                columns.append((spec["raw_key"], spec["column"]))
        prefix, signals = self.selection["efficiency"], self.signals["val"]
        return columns + [(SET_EFFICIENCY_KEY.format(s), prefix + s) for s in signals]

    def _mlflow_metrics(self, val: list[dict]) -> list[Metric]:
        # As log_metrics_to_mlflow, non-finite values are left out; epoch -1 is too, as
        # Lightning drops the logs of the sanity pass.
        columns, timestamp = self._mlflow_columns(), int(time.time() * 1000)
        return [
            Metric(key, float(row[column]), timestamp, row["epoch"])
            for row in val
            if row["epoch"] >= 0
            for key, column in columns
            if row[column] is not None and math.isfinite(row[column])
        ]

    def _log_mlflow(self, tracking_uri: str, val: list[dict], run: dict) -> str:
        client = MlflowClient(tracking_uri)
        experiment_id = _experiment_id(client, f"{run['lattice']}_eval")
        run_id = client.create_run(experiment_id, run_name=run["run"]).info.run_id
        params = [Param(key, str(value)) for key, value in run["hyperpoint"].items()]
        client.log_batch(run_id, params=params)
        metrics = self._mlflow_metrics(val)
        for start in range(0, len(metrics), MLFLOW_BATCH):
            client.log_batch(run_id, metrics=metrics[start : start + MLFLOW_BATCH])
        client.set_terminated(run_id)
        return run_id


def _init_worker(sds_cache: dict | None = None) -> None:
    SDS_CACHE.update(sds_cache or {})
    torch.set_num_threads(1)
    # The run configs call ${reverse:...} and the like; a second registration raises.
    if not OmegaConf.has_resolver("reverse"):
        register_resolvers()
    # Lightning repeats these at every pass, thousands of times per run.
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*has `__len__` defined.*")


def _evaluate_task(task: tuple) -> dict:
    run_dir, out_dir, lattice, data_dir, sets, tracking_uri, settings = task
    module, datamodule = rebuild(run_dir, data_dir, sets)
    run = RunEvaluation(run_dir, module, datamodule, sets=sets, **settings)
    return run.evaluate(out_dir, lattice, tracking_uri)


class LatticeEvaluation:
    """The runs of one lattice, each evaluated by ``RunEvaluation``.

    The runs are the directories of ``<scores_dir>/<lattice>`` that hold a ``meta.pt``,
    in name order. Each run's module and datamodule are rebuilt from its
    ``config.yaml``, with ``data.data_dir`` replaced by ``data_dir`` and the signals
    and nulls by those of ``configs/data/<sets>.yaml`` when given, in a spawn pool of
    ``workers`` single-threaded processes (in this process for one). ``settings``
    holds the keyword-only arguments of ``evaluate_dump``.
    """

    def __init__(
        self,
        scores_dir: Path,
        lattice: str,
        settings: dict,
        data_dir: str | None = None,
        sets: str | None = None,
        workers: int = 1,
        tracking_uri: str | None = None,
    ):
        self.lattice, self.settings, self.data_dir = lattice, settings, data_dir
        self.sets, self.workers, self.tracking_uri = sets, workers, tracking_uri
        paths = (Path(scores_dir) / lattice).iterdir()
        self.runs = sorted(p for p in paths if (p / "meta.pt").is_file())

    def evaluate(self, out_dir: Path) -> list[dict]:
        """Evaluate every run into ``out_dir``; return their run.json."""
        # Created once, before the pool: workers creating one name at once make
        # duplicate experiments.
        if self.tracking_uri is not None:
            _experiment_id(MlflowClient(self.tracking_uri), f"{self.lattice}_eval")
        # The run configs read ${oc.env:PROJECT_ROOT}; spawned workers inherit it.
        os.environ.setdefault("PROJECT_ROOT", str(ROOT))
        task = (out_dir, self.lattice, self.data_dir, self.sets, self.tracking_uri)
        return self._map([(run, *task, self.settings) for run in self.runs])

    def _map(self, tasks: list[tuple]) -> list[dict]:
        if self.workers <= 1:
            _init_worker()
            return [_evaluate_task(task) for task in tasks]
        self._prebuild_sds()
        # Spawned, not forked: torch's thread pools do not survive a fork once the
        # parent has used them.
        context = multiprocessing.get_context("spawn")
        initargs = (dict(SDS_CACHE),)
        with context.Pool(
            self.workers, initializer=_init_worker, initargs=initargs
        ) as pool:
            return list(pool.imap(_evaluate_task, tasks))

    def _prebuild_sds(self) -> None:
        sds = self.settings["extra_sets"].get("sds")
        if not sds:
            return
        if not OmegaConf.has_resolver("reverse"):
            register_resolvers()
        datamodule = rebuild(self.runs[0], self.data_dir, self.sets)[1]
        datamodule.setup("fit")
        cached_sds(datamodule.data_train.tensors[0], sds["max_rows"], sds["row_seed"])


def evaluate_lattice(
    scores_dir: Path,
    out_dir: Path,
    lattice: str,
    settings: dict,
    data_dir: str | None = None,
    sets: str | None = None,
    workers: int = 1,
    tracking_uri: str | None = None,
) -> list[dict]:
    """Evaluate every run of a lattice; return their run.json, in name order."""
    evaluation = LatticeEvaluation(
        scores_dir, lattice, settings, data_dir, sets, workers, tracking_uri
    )
    return evaluation.evaluate(out_dir)


def evaluate_dump(
    run_dir: Path,
    module: LightningModule,
    datamodule: LightningDataModule,
    out_dir: Path,
    lattice: str | None = None,
    tracking_uri: str | None = None,
    *,
    dump_callbacks: dict,
    test_callbacks: list[str],
    extra_sets: dict,
    selection: dict,
    typical: str | None = None,
) -> dict:
    """Evaluate every epoch of one dumped run; write its four files, return run.json.

    Written to ``<out_dir>/runs/<run>/``, a row per epoch, -1 first, a value empty
    where undefined (the EMAs at -1, ``null_rate`` without ``null_ds``):

    - ``val.csv``: ``epoch``, validation scalars, labelled rows, ``ema/<strategy>``.
    - ``test.csv``: ``epoch``, test scalars, ``test_mean_eff``, ``realised_fpr``,
      ``null_rate``.
    - ``picks.csv``: ``strategy``, ``k`` (0..W_STEPS for a criterion, else empty),
      ``epoch``.
    - ``run.json``: ``run``, ``lattice``, ``dump``, ``hyperpoint``, ``mlflow_run_id``,
      ``epochs``, ``module``, ``signals``, ``nulls``, ``stored``, ``typical``.

    With ``tracking_uri``, the run is also logged to MLflow, as ``<lattice>_eval``.
    The keyword-only arguments are the Evaluator's of the same names.
    """
    run = RunEvaluation(
        run_dir,
        module,
        datamodule,
        dump_callbacks=dump_callbacks,
        test_callbacks=test_callbacks,
        extra_sets=extra_sets,
        selection=selection,
        typical=typical,
    )
    return run.evaluate(out_dir, lattice, tracking_uri)
