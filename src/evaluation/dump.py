"""Evaluate the per-epoch score dumps of a lattice, one run at a time."""

import contextlib
import functools
import json
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
from torch.utils.data import DataLoader, Subset

from src.callbacks.components.selection import (
    W_STEPS,
    BandPick,
    band_columns,
    best_epoch,
    ema,
    epoch_band_pick,
    epoch_pick,
    smoothed,
)
from src.callbacks.components.utils import row_subset
from src.callbacks.score_dump import SDS_PARAMS
from src.callbacks.sds import SDS_PUSH_SCALE
from src.data.utils import unpack_batch
from src.evaluation.callbacks.sds import PREFIXES
from src.evaluation.components.pairing import hyperpoint
from src.evaluation.components.rebuild import (
    POINTS_CACHE,
    ROOT,
    SDS_CACHE,
    cached_points,
    cached_sds,
    capped_loader,
    capped_meta,
    capped_record,
    check_record_cache,
    loader_inputs,
    loader_layouts,
    null_sets,
    reads_hub,
    rebuild,
    signal_sets,
)
from src.evaluation.components.tables import load_record, write_outputs, write_partial
from src.evaluation.replay import (
    ReplayModule,
    ScoresCallback,
    as_typical,
    extra_loaders,
)
from src.utils.omegaconf import register_resolvers

log = logging.getLogger(__name__)

# The key of the training efficiency callback for one set's validation efficiency at
# the operational rate; the registry's efficiency_key replaces it.
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
# What becomes of a run with fewer epoch files than its max_epochs (LatticeEvaluation).
PARTIAL_MODES = (None, "skip", "replay")


def _strategies(selection: dict) -> dict[str, dict]:
    """Every strategy of the registry with its column in val.csv and its mode.

    The labelled rows are columns under their own names and are maximised; Q'' is
    minimised. A registry may name none of the three, as that of ANCAP v8 alone
    (``configs/eval/physics_ancap8.yaml``).
    """
    criteria = selection.get("criteria") or {}
    strategies = {name: dict(spec) for name, spec in criteria.items()}
    for name, spec in (selection.get("labelled") or {}).items():
        strategies[name] = {**spec, "column": name, "mode": "max"}
    if selection.get("q_sec"):
        strategies["q_sec"] = {**selection["q_sec"], "mode": "min"}
    return strategies


def _available(strategies: dict, rows: list[dict], run: str) -> dict[str, dict]:
    """The strategies less every banded one whose columns the validation rows lack, as
    where the ancap callback runs without working_point; each is logged once."""
    present, kept = set(rows[0]) if rows else set(), {}
    for name, spec in strategies.items():
        missing = sorted(set(band_columns(spec)) - present) if "band" in spec else []
        if missing:
            log.info(f"{run}: {name} is skipped, its columns {missing} are absent.")
        else:
            kept[name] = spec
    return kept


def _experiment_id(client: MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(name)


def _check_stored_sds(meta: dict, sds: dict) -> None:
    """Raise ValueError where the dump stores the scores of pseudo-sets that ``sds``
    would build otherwise.

    ``meta.pt`` holds no push scale, since every dump stores its pseudo-sets at the
    default: only an entry of ``sds`` at the default, or without a scale, matches them.
    """
    if "sds" not in meta:
        return
    stored = {k: meta[f"sds_{k}"] for k in SDS_PARAMS}
    same_rows = stored == {k: sds[k] for k in SDS_PARAMS}
    if not same_rows or sds.get("push_scale", SDS_PUSH_SCALE) != SDS_PUSH_SCALE:
        raise ValueError(
            f"The dump stores the SDS pseudo-sets of {stored}, the evaluation would "
            f"build those of {dict(sds)}."
        )


def _rates(callbacks: dict) -> list[float]:
    """Every target rate an evaluation callback's config names, whose cuts on validation
    normal the replay sets as training set them."""
    rates = {
        r for config in callbacks.values() for r in config.get("target_rates") or ()
    }
    return sorted(map(float, rates))


def _spec_sds(x: torch.Tensor, mask: torch.Tensor | None, spec: dict) -> dict:
    """``cached_sds`` of the rows x under one entry of ``extra_sets``."""
    push_scale = spec.get("push_scale", SDS_PUSH_SCALE)
    return cached_sds(x, spec["max_rows"], spec["row_seed"], mask, push_scale)


def _sds_val_sets(normal: DataLoader, spec: dict) -> dict:
    """The SDS pseudo-sets of the inputs of validation normal, as ``SDSCallback`` with
    ``source="normal"`` builds them, under the names its mirror reads."""
    view, prefix = loader_inputs(normal), PREFIXES["normal"]
    sets = _spec_sds(view.x, view.mask, spec)
    return {name.replace("sds", prefix, 1): x for name, x in sets.items()}


def _check_rescored(names: list[str], loaders: dict) -> None:
    unknown = sorted(set(names) - loaders.keys())
    if unknown:
        raise ValueError(f"rescored_sets names no validation loader: {unknown}.")


def _kept(loaders: dict, names: list[str] | None) -> dict:
    """The loaders that ``val_sets`` names, in the datamodule's order; all for None."""
    if names is None:
        return loaders
    unknown = sorted(set(names) - loaders.keys())
    if unknown:
        raise ValueError(f"val_sets names no validation loader: {unknown}.")
    return {name: loader for name, loader in loaders.items() if name in names}


def _check_point_names(points: dict, val: dict) -> None:
    # A points loader under a stored set's name would be served that set's scores.
    clash = sorted(points.keys() & val.keys())
    if clash:
        raise ValueError(f"The points {clash} take the names of validation loaders.")


def _served_signals(layout: dict | None, loaders: dict) -> list[str]:
    """The signal sets of a layout among the loaders served.

    The labels of no other set are read: none where ``val_sets`` leaves them out, and
    none at all for a split without loaders, whose layout may be None.
    """
    if not loaders:
        return []
    served = [name for name in layout["names"] if name in loaders]
    return signal_sets({**layout, "names": served})


def _train_loader(data_train, spec, meta: dict) -> DataLoader | None:
    """The loader of ``extra_sets.train``: None for false; for true, every training row
    with the dump's batch size; for ``{max_rows, row_seed, batch_size}``, the seeded
    subset of rows that ``LossEntropyCallback`` draws with the same parameters."""
    if not spec:
        return None
    if spec is True:
        return DataLoader(
            data_train, batch_size=meta["train"]["batch_sizes"]["train"][0]
        )
    rows = row_subset(len(data_train), spec["max_rows"], spec["row_seed"])
    rows = data_train if rows is None else Subset(data_train, rows.tolist())
    return DataLoader(rows, batch_size=spec["batch_size"])


class RunEvaluation:
    """The replay of one dumped run through the evaluation callbacks, and its tables.

    Every epoch file, ``initial.pt`` (epoch -1) first, is replayed through the run's
    own module (``src.evaluation.replay``) in two passes, each on a CPU trainer with
    fresh callbacks and no plots: the validation loaders followed by the extra sets
    under every evaluation callback, then the test loaders under the test callbacks.
    The scalars the callbacks record, named '<callback>/<scalar>', make the rows of the
    run's tables. Both trainers carry ``out_dir``, the evaluation directory, where a
    callback may cache what the runs share, and ``rows``, the rows kept of each capped
    set by split. The keyword-only arguments are the Evaluator's of the same names,
    apart from ``sets``, the data config whose signals and nulls replaced the run's in
    ``datamodule`` (``rebuild``). With it, the signal sets are read from the
    datamodule's loaders instead of ``meta.pt``: a stored set outside the new lists is
    never served, and a set the dump does not store is rescored from the weights.
    The cuts of every ``target_rates`` of the callbacks' configs are set on validation
    normal (``ReplayModule``), which an epoch file that lacks it has rescored first.
    A set of ``rescored_sets`` is replayed as if its file lacked it, so that a callback
    reads it from the same arithmetic as the sets the dump never stored; listing a set
    the dump does not store changes nothing. With ``val_sets``, the validation loaders
    it does not name are dropped before any pass, so that none of them is scored or
    read, and the signal sets are those it names. Without test callbacks no test
    loader is built and no test pass runs: the rows of test.csv hold the epoch alone.
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
        row_caps: dict | None = None,
        rescored_sets: list[str] | None = None,
        val_sets: list[str] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.meta = torch.load(self.run_dir / "meta.pt", weights_only=True)
        self.dump_callbacks, self.extra_sets = dump_callbacks, extra_sets
        self.selection, self.strategies = selection, _strategies(selection)
        self.passes = {"val": list(dump_callbacks), "test": list(test_callbacks)}
        self.trainers = {split: self._trainer(split) for split in self.passes}
        self.loaders = self._loaders(datamodule, typical, val_sets)
        self.rescored_sets = list(rescored_sets or [])
        _check_rescored(self.rescored_sets, self.loaders["val"])
        self.rows = self._cap(row_caps or {})
        meta = capped_meta(self.meta, self.loaders, self.rows)
        self.replay = ReplayModule(module, meta, typical, _rates(dump_callbacks))
        self.signals = self._signals(datamodule, sets)
        self.nulls = null_sets(self.trainers["test"].callbacks)

    def evaluate(
        self, out_dir: Path, lattice: str | None = None, tracking_uri: str | None = None
    ) -> dict:
        """Write the four files ``evaluate_dump`` lists; return run.json.

        With ``tracking_uri``, the run is also logged to MLflow, as ``<lattice>_eval``.
        """
        out = Path(out_dir) / "runs" / self.run_dir.name
        for trainer in self.trainers.values():
            trainer.out_dir, trainer.rows = Path(out_dir), self.rows
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

    def _loaders(
        self,
        datamodule: LightningDataModule,
        typical: str | None,
        val_sets: list[str] | None,
    ) -> dict:
        """The validation loaders followed by the extra sets, and the test loaders."""
        datamodule.prepare_data()
        datamodule.setup("fit")
        val, data_train = datamodule.val_dataloader(), datamodule.data_train
        train = unpack_batch(data_train.tensors)
        loader = _train_loader(data_train, self.extra_sets.get("train"), self.meta)
        aux, pseudo = bool(self.extra_sets.get("aux")), self._pseudo_sets(train, val)
        points = cached_points(self.extra_sets.get("points"))
        extra = extra_loaders(self.meta, loader, pseudo, aux, train.mask, points)
        val = _kept(as_typical(val, typical), val_sets)
        _check_point_names(points, val)
        return {"val": {**val, **extra}, "test": self._test_loaders(datamodule)}

    def _signals(self, datamodule: LightningDataModule, sets: str | None) -> dict:
        """The signal sets that each pass serves.

        From meta.pt's layouts or, with ``sets``, from the datamodule's loaders of the
        splits that have loaders, so that no test loader is built without a test pass.
        """
        splits = tuple(split for split in self.passes if self.loaders[split])
        layouts = self.meta if sets is None else loader_layouts(datamodule, splits)
        served = self.loaders
        return {s: _served_signals(layouts.get(s), served[s]) for s in self.passes}

    def _test_loaders(self, datamodule: LightningDataModule) -> dict:
        if not self.passes["test"]:
            return {}
        datamodule.setup("test")
        return datamodule.test_dataloader()

    def _pseudo_sets(self, train, val: dict) -> dict:
        """The SDS pseudo-sets of ``extra_sets.sds`` (from the training rows) and of
        ``extra_sets.sds_val`` (from validation normal), each point with the mask of the
        row it was shifted from where the data carry one."""
        pseudo, sds = {}, self.extra_sets.get("sds")
        if sds:
            _check_stored_sds(self.meta, sds)
            pseudo |= _spec_sds(train.x, train.mask, sds)
        if self.extra_sets.get("sds_val"):
            pseudo |= _sds_val_sets(val["normal"], self.extra_sets["sds_val"])
        return pseudo

    def _cap(self, row_caps: dict) -> dict[str, dict[str, torch.Tensor]]:
        """Replace the loader of every set ``row_caps`` caps by one of its seeded row
        subset; return the rows kept, by split and set, for the sets it cuts."""
        rows, seed = {}, row_caps.get("seed", 0)
        for split in ("val", "test"):
            for name, cap in (row_caps.get(split) or {}).items():
                loader, kept = capped_loader(self.loaders[split][name], cap, seed)
                self.loaders[split][name] = loader
                if kept is not None:
                    rows.setdefault(split, {})[name] = kept
        return rows

    def _epoch_files(self) -> list[Path]:
        return [self.run_dir / "initial.pt", *sorted(self.run_dir.glob("epoch_*.pt"))]

    def _replay(self, out: Path) -> dict[str, list[dict]]:
        """The rows of both passes over every epoch file, -1 first."""
        rows = {split: [] for split in self.trainers}
        for epoch, path in enumerate(self._epoch_files(), start=-1):
            record = capped_record(load_record(path, epoch), self.rows)
            val = record["val"].items()
            stored = {k: v for k, v in val if k not in self.rescored_sets}
            self.replay.load_epoch({**record, "val": stored}, out / path.name)
            self._score_normal()
            for split in self.trainers:
                rows[split].append(self._row(split, epoch))
        return rows

    def _score_normal(self) -> None:
        """Rescore validation normal from the epoch's weights where its file lacks it,
        in a pass of its own before those that read its cuts."""
        if "normal" in self.replay.record["val"]:
            return
        scores, scorer = self._normal_scorer
        self.replay.serve("val")
        normal = {"normal": self.loaders["val"]["normal"]}
        scorer.test(self.replay, dataloaders=normal, verbose=False)
        self.replay.add_scores(scores.collector.scores())

    @functools.cached_property
    def _normal_scorer(self) -> tuple[ScoresCallback, pytorch_lightning.Trainer]:
        """A CPU trainer that collects the keys the dump stores, built at first use.

        A dump that stores no validation set records no validation keys; its auxiliary
        sets, scored by the same ``model_step`` at every pass, hold the dump's keys.
        """
        scores = ScoresCallback(self.meta["val"]["keys"] or self.meta["aux"]["keys"])
        return scores, pytorch_lightning.Trainer(callbacks=[scores], **REPLAY_TRAINER)

    def _row(self, split: str, epoch: int) -> dict:
        """The epoch and the scalars every callback of a pass records; the epoch alone
        for a pass without callbacks, which is not run."""
        if not self.passes[split]:
            return {"epoch": epoch}
        trainer = self.trainers[split]
        self.replay.serve(split)
        trainer.metric_records = {}
        trainer.test(self.replay, dataloaders=self.loaders[split], verbose=False)
        return {"epoch": epoch, **trainer.metric_records}

    def _tables(self, rows: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """The rows of val.csv, test.csv and picks.csv."""
        self.strategies = _available(self.strategies, rows["val"], self.run_dir.name)
        val = self._val_rows(rows["val"])
        test = [{**row, **self._test_summary(row)} for row in rows["test"]]
        return {"val": val, "test": test, "picks": self._picks(val)}

    def _labelled(self, row: dict) -> dict[str, float]:
        """The labelled rows of an epoch, mean or CVaR of the signals' efficiencies.

        As the training efficiency callback computes its CVaR: in float32, without the
        non-finite values, the mean of the ceil(alpha n) lowest.
        """
        labelled = self.selection.get("labelled") or {}
        prefix, values = self.selection.get("efficiency"), {}
        effs = [row[prefix + s] for s in self.signals["val"]] if labelled else []
        x = torch.as_tensor(effs, dtype=torch.float32)
        x = torch.sort(x[torch.isfinite(x)]).values
        for name, spec in labelled.items():
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
        """The epoch of the epoch rule at k or, for k None, of its best logged EMA; a
        banded strategy's by its band rule."""
        spec = self.strategies[name]
        skip = spec.get("skip_nonfinite", False)
        q_main = [row[spec["column"]] for row in trained]
        if k is None:
            index = best_epoch(smoothed(q_main, skip_nonfinite=skip), spec["mode"])
        elif "band" in spec:
            index = self._band_pick(spec, trained, k).index
        else:
            index = epoch_pick(q_main, self._q_sec(trained), k, spec["mode"], skip)
        return trained[index]["epoch"]

    def _q_sec(self, trained: list[dict]) -> list[float]:
        return [row[self.strategies["q_sec"]["column"]] for row in trained]

    def _band_pick(self, spec: dict, trained: list[dict], k: int) -> BandPick:
        """The band rule's pick at k, on the columns that the entry's ``band`` names."""
        band = spec["band"]
        columns = (band["level"], band["level_se"], spec["column"], band["se"])
        level, level_se, q_main, q_main_se = ([r[c] for r in trained] for c in columns)
        skip = spec.get("skip_nonfinite", False)
        args = (level, level_se, q_main, q_main_se, self._q_sec(trained), k)
        return epoch_band_pick(*args, band["band_se"], spec["mode"], skip)

    def _picks(self, val: list[dict]) -> list[dict]:
        """A row per criterion and k of the w grid, one per other strategy (k None)."""
        trained = [row for row in val if row["epoch"] >= 0]
        criteria, picks = self.selection.get("criteria") or {}, []
        for name in self.strategies:
            for k in range(W_STEPS + 1) if name in criteria else [None]:
                epoch = self._pick(name, trained, k)
                picks.append({"strategy": name, "k": k, "epoch": epoch})
        return picks

    def _test_summary(self, row: dict) -> dict:
        # Without test callbacks no test pass runs, and the row holds the epoch alone.
        if row.keys() == {"epoch"}:
            return {}
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
        prefix = self.selection.get("efficiency")
        signals = [] if prefix is None else self.signals["val"]
        key = self.selection.get("efficiency_key", SET_EFFICIENCY_KEY)
        return columns + [(key.format(s), prefix + s) for s in signals]

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


def _init_worker(
    sds_cache: dict | None = None, threads: int = 1, points_cache: dict | None = None
) -> None:
    SDS_CACHE.update(sds_cache or {})
    POINTS_CACHE.update(points_cache or {})
    torch.set_num_threads(threads)
    # The run configs call ${reverse:...} and the like; a second registration raises.
    if not OmegaConf.has_resolver("reverse"):
        register_resolvers()
    # Lightning repeats these at every pass, thousands of times per run.
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*has `__len__` defined.*")


def _evaluate_task(task: tuple) -> dict:
    run_dir, out_dir, lattice, data, tracking_uri, settings = task
    data_dir, sets, data_overrides, empty_signals = data
    module, datamodule = rebuild(run_dir, data_dir, sets, data_overrides, empty_signals)
    run = RunEvaluation(run_dir, module, datamodule, sets=sets, **settings)
    return run.evaluate(out_dir, lattice, tracking_uri)


def _stored_run(
    run_dir: Path, out_dir: Path, lattice: str, typical: str | None
) -> dict | None:
    """The run.json of a run whose four files in ``out_dir`` are complete, else None.

    Complete means that the three tables exist and that run.json parses and names this
    lattice, run, dump and typical set and the number of epoch files the dump holds.
    ``write_outputs`` removes run.json before it writes the tables and writes it after
    them, so a run.json that parses vouches for tables of the same write wherever a
    kill cut it, and an earlier run.json never sits beside tables being rewritten. A
    check of the tables' contents would not do: a val.csv cut within its last row
    still holds a row per epoch.
    """
    out = Path(out_dir) / "runs" / run_dir.name
    try:
        run = json.loads((out / "run.json").read_text())
    except (FileNotFoundError, ValueError):
        return None
    epochs = len(list(run_dir.glob("epoch_*.pt")))
    same = {"lattice": lattice, "run": run_dir.name, "dump": str(run_dir)}
    same.update(typical=typical, epochs=epochs)
    tables = all((out / f"{name}.csv").is_file() for name in ("val", "test", "picks"))
    return run if tables and all(run.get(k) == v for k, v in same.items()) else None


def _max_epochs(run_dir: Path) -> int | None:
    """``trainer.max_epochs`` of a run's config.yaml, None where it gives none."""
    path = run_dir / "config.yaml"
    if not path.is_file():
        return None
    return OmegaConf.select(
        OmegaConf.load(path), "trainer.max_epochs", throw_on_resolution_failure=False
    )


def partial_runs(runs: list[Path]) -> dict[Path, tuple[int, int]]:
    """The runs with fewer epoch files than their ``trainer.max_epochs``, as a trial
    stopped at a time limit leaves them, with both numbers."""
    partial = {}
    for run in runs:
        epochs, max_epochs = len(list(run.glob("epoch_*.pt"))), _max_epochs(run)
        if max_epochs is not None and epochs < max_epochs:
            partial[run] = (epochs, max_epochs)
    return partial


def _partial_split(runs: list[Path], mode: str | None) -> tuple[dict, list[Path]]:
    """The partial runs under ``mode`` (none for None) and the runs replayed."""
    if mode not in PARTIAL_MODES:
        raise ValueError(f"partial_runs is one of {PARTIAL_MODES}, not {mode!r}.")
    partial = {} if mode is None else partial_runs(runs)
    return partial, [run for run in runs if mode != "skip" or run not in partial]


@contextlib.contextmanager
def _offline_hub(offline: bool):
    """HF_HUB_OFFLINE=1 inside, where ``offline``, for the processes spawned there;
    huggingface_hub reads it at import, so this process is unaffected."""
    before = os.environ.get("HF_HUB_OFFLINE")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        if before is not None:
            os.environ["HF_HUB_OFFLINE"] = before


class LatticeEvaluation:
    """The runs of one lattice, each evaluated by ``RunEvaluation``.

    The runs are the directories of ``<scores_dir>/<lattice>`` that hold a ``meta.pt``,
    in name order. Each run's module and datamodule are rebuilt from its
    ``config.yaml``, with its data directory replaced by ``data_dir``, the signals and
    nulls by those of ``configs/data/<sets>.yaml``, the signals emptied with
    ``empty_signals`` and the record's ``data_overrides`` appended, when given
    (``rebuild``), in a spawn pool of ``workers`` processes of ``threads`` torch
    threads each (in this process, on one thread, for one worker). One thread is the
    reference: more may change float32 results in the last bits. Before the pool, this
    process prepares the first run's data once: the SDS pseudo-sets, which the workers
    receive, and a record of the hub (``rebuild.HUB_DATAMODULES``), which the workers
    then read offline. The points files of ``extra_sets.points`` are loaded and
    checked against their md5s here once, before any run is replayed, and handed to
    the workers; ``data_overrides`` without ``data_dir`` are refused, and with
    ``val_sets`` every hub run's record cache is checked by ``check_record_cache``.
    ``settings`` holds the keyword-only arguments of ``evaluate_dump``. With
    ``resume``, a run complete in ``out_dir`` (``_stored_run``) is neither replayed nor
    logged to MLflow again, and its stored run.json takes its place in the returned
    list. Only what run.json records is compared, so an ``out_dir`` written under
    another evaluation config or other ``sets`` needs ``resume`` off, or a fresh
    ``out_dir``. With ``partial`` None every run is replayed and selected. With
    ``"skip"`` or ``"replay"``, a run with fewer epoch files than its
    ``trainer.max_epochs`` (``partial_runs``) is listed in
    ``<out_dir>/partial_runs.csv`` (run, epochs, max_epochs), which the lattice step
    reads to leave it out, and is replayed only with ``"replay"``.
    """

    def __init__(
        self,
        scores_dir: Path,
        lattice: str,
        settings: dict,
        data_dir: str | None = None,
        sets: str | None = None,
        workers: int = 1,
        threads: int = 1,
        tracking_uri: str | None = None,
        resume: bool = True,
        partial: str | None = None,
        data_overrides: list[str] | None = None,
        empty_signals: bool = False,
    ):
        self.lattice, self.settings = lattice, settings
        # What rebuild takes besides the run, in its order.
        self.data = (data_dir, sets, data_overrides, empty_signals)
        self.workers, self.tracking_uri = workers, tracking_uri
        self.threads, self.resume, self.partial_mode = threads, resume, partial
        paths = (Path(scores_dir) / lattice).iterdir()
        self.runs = sorted(p for p in paths if (p / "meta.pt").is_file())
        self.partial, self.replayed = _partial_split(self.runs, partial)

    def evaluate(self, out_dir: Path) -> list[dict]:
        """Evaluate the runs into ``out_dir``; return their run.json."""
        self._prepare_out(Path(out_dir))
        self._preflight()
        stored = self._stored(out_dir)
        task = (out_dir, self.lattice, self.data, self.tracking_uri)
        runs = [run for run in self.replayed if run not in stored]
        evaluated = iter(self._map([(run, *task, self.settings) for run in runs]))
        return [stored.get(run) or next(evaluated) for run in self.replayed]

    def _preflight(self) -> None:
        """Refuse a configuration, points file or record cache unlike its declaration.

        Before any datamodule is built: ``data_overrides`` without ``data_dir``, which
        would leave in use the run's own record cache, built without them; a points
        file whose md5 differs; and a hub run's record cache that holds a simulated set
        outside ``val_sets``.
        """
        data_dir, _, data_overrides, _ = self.data
        if data_overrides and data_dir is None:
            raise ValueError(
                "data_overrides need data_dir, a record cache built with the same "
                "overrides."
            )
        cached_points(self.settings["extra_sets"].get("points"))
        for run in self.runs:
            check_record_cache(run, data_dir, self.settings.get("val_sets"))

    def _prepare_out(self, out_dir: Path) -> None:
        """Create the MLflow experiment and write the table of partial runs."""
        # Created once, before the pool: workers creating one name at once make
        # duplicate experiments.
        if self.tracking_uri is not None:
            _experiment_id(MlflowClient(self.tracking_uri), f"{self.lattice}_eval")
        # The run configs read ${oc.env:PROJECT_ROOT}; spawned workers inherit it.
        os.environ.setdefault("PROJECT_ROOT", str(ROOT))
        if self.partial_mode is not None:
            rows = [[r.name, *counts] for r, counts in self.partial.items()]
            write_partial(out_dir, rows)

    def _stored(self, out_dir: Path) -> dict[Path, dict]:
        """The stored run.json of every run complete in ``out_dir``, with ``resume``."""
        if not self.resume:
            return {}
        stored = {}
        for run in self.replayed:
            info = _stored_run(run, out_dir, self.lattice, self.settings["typical"])
            if info is not None:
                log.info(f"Skipping {run.name}, complete in {out_dir}")
                stored[run] = info
        return stored

    def _map(self, tasks: list[tuple]) -> list[dict]:
        # A resumed lattice may have no run left, which would spend the SDS prebuild
        # (75 s on CIFAR-10) and the pool's spawn for nothing.
        if not tasks:
            return []
        if self.workers <= 1:
            _init_worker()
            return [_evaluate_task(task) for task in tasks]
        offline = self._prepare()
        # Spawned, not forked: torch's thread pools do not survive a fork once the
        # parent has used them.
        context = multiprocessing.get_context("spawn")
        initargs = (dict(SDS_CACHE), self.threads, dict(POINTS_CACHE))
        with _offline_hub(offline), context.Pool(
            self.workers, initializer=_init_worker, initargs=initargs
        ) as pool:
            return list(pool.imap(_evaluate_task, tasks))

    def _prepare(self) -> bool:
        """Prepare the first run's data here, once: the hub record where its datamodule
        reads one, and the SDS pseudo-sets; return whether it reads one."""
        extra, hub = self.settings["extra_sets"], reads_hub(self.runs[0])
        if not (hub or extra.get("sds") or extra.get("sds_val")):
            return False
        if not OmegaConf.has_resolver("reverse"):
            register_resolvers()
        datamodule = rebuild(self.runs[0], *self.data)[1]
        datamodule.prepare_data()
        _prebuild_sds(datamodule, extra)
        return hub


def _prebuild_sds(datamodule: LightningDataModule, extra: dict) -> None:
    """Build the pseudo-sets of ``extra_sets`` into ``SDS_CACHE``."""
    sds, sds_val = extra.get("sds"), extra.get("sds_val")
    if not (sds or sds_val):
        return
    datamodule.setup("fit")
    train = unpack_batch(datamodule.data_train.tensors)
    if sds:
        _spec_sds(train.x, train.mask, sds)
    if sds_val:
        _sds_val_sets(datamodule.val_dataloader()["normal"], sds_val)


def evaluate_lattice(
    scores_dir: Path,
    out_dir: Path,
    lattice: str,
    settings: dict,
    data_dir: str | None = None,
    sets: str | None = None,
    workers: int = 1,
    threads: int = 1,
    tracking_uri: str | None = None,
    resume: bool = True,
    partial: str | None = None,
    data_overrides: list[str] | None = None,
    empty_signals: bool = False,
) -> list[dict]:
    """Evaluate every run of a lattice; return their run.json, in name order.

    With ``resume``, a run whose four files in ``out_dir`` are complete is not
    evaluated again; ``partial`` says what becomes of a partial run, ``data_overrides``
    what is appended to the record's overrides, and ``empty_signals`` whether every
    run's signals are emptied (``LatticeEvaluation``).
    """
    evaluation = LatticeEvaluation(
        scores_dir,
        lattice,
        settings,
        data_dir,
        sets,
        workers,
        threads,
        tracking_uri,
        resume,
        partial,
        data_overrides,
        empty_signals,
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
    row_caps: dict | None = None,
    rescored_sets: list[str] | None = None,
    val_sets: list[str] | None = None,
) -> dict:
    """Evaluate every epoch of one dumped run; write its four files, return run.json.

    Written to ``<out_dir>/runs/<run>/``, a row per epoch, -1 first, a value empty
    where undefined (the EMAs at -1, ``null_rate`` without ``null_ds``):

    - ``val.csv``: ``epoch``, validation scalars, labelled rows, ``ema/<strategy>``.
    - ``test.csv``: ``epoch``, test scalars, ``test_mean_eff``, ``realised_fpr``,
      ``null_rate``; ``epoch`` alone without test callbacks.
    - ``picks.csv``: ``strategy``, ``k`` (0..W_STEPS for a criterion, else empty),
      ``epoch``; no row where the registry names no strategy.
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
        row_caps=row_caps,
        rescored_sets=rescored_sets,
        val_sets=val_sets,
    )
    return run.evaluate(out_dir, lattice, tracking_uri)
