#!/usr/bin/env python3
"""Replay the pipeline's checkpoint selection on stored scores.

Reads the score dumps of ``src/callbacks/score_dump.py`` (one directory per run) and,
for every CAP configuration given, does per run what training and the evaluator do
in the isolated form of the search catalogues, without retraining:

1. Per epoch, on the validation split and one score key: the training callbacks of
   each strategy (``CAPCallback``, ``ThresholdDriftCallback``, ``WassersteinCallback``),
   driven through their own hooks with ``torch.manual_seed(--torch-seed)`` before
   every epoch's computation, and ``ascore_operational`` of ``normal`` averaged over
   batches as training logs it and as the evaluator's ``OutputCallback`` averages it.
2. Per strategy: one candidate, the epoch at which the EMA of the strategy's quantity
   is best, which is the checkpoint ``ModelCheckpoint(save_top_k=1)`` keeps. Q' is the
   strategy's quantity at that epoch and Q'' is ``ascore_operational`` of ``normal``.
3. Per configuration, strategy, family and seed: the Pareto front of the runs over
   (Q', Q''), the best-Q' run (ties to the lower candidate id), and, at every
   selected epoch, the efficiency at the pipeline's threshold and the AUPRC of every
   signal set on validation and test.

Outputs, in ``--out``:

    selection.csv  one row per (configuration, strategy, run): config, run, family,
                   candidate, seed, score_key, torch_seed, strategy, selected_ckpt,
                   selected_epoch, q_main (Q'), q_sec (Q''), beta_star and at_bound
                   (CAP's maximising beta and grid-bound flag at the selected epoch,
                   empty for the other strategies), efficiency_val_mean,
                   auprc_val_mean, efficiency_test_mean, auprc_test_mean (means over
                   the signal sets), pareto, best
    metrics.csv    one row per (run, selected epoch, split, dataset): threshold,
                   tie_weight, efficiency, AUPRC
    trace.csv      one row per (configuration, run, epoch): every quantity of step 1,
                   with CAP's cap_beta_star and cap_at_bound, and the value, the
                   EMA and the traced columns of every replay strategy

A run directory maps to (family, candidate, seed) through ``--run-pattern``, a regular
expression with those named groups searched in the directory path; the default reads
``<family>_<candidate>_s<seed>``, e.g. ``pool/ae_003_s101/scores``. A CAP configuration
is a YAML file whose stem names it in the outputs, in either of two formats: the nested
format of the design configurations, ``pairing_type`` and ``cap_metric_config``
(optionally ``ema_weight``), as in the CAP blocks of the experiment configs, or the
flat format of the ``cap`` config group, e.g. ``configs/cap/frozen.yaml``, which holds
the metric's parameters alone and is paired with the registered ``cdf`` pairing.
Threshold drift and Wasserstein do not depend on it, so run them once. Every quantity
uses ``--score-key``; ``ascore/mse`` or ``ascore/kl`` gives the native-score
comparison as if that score had been declared. The replay assumes training without a
sanity pass, as in ``configs/trainer/default.yaml``: the training callbacks do not
skip it, so it would enter their averages, while the dump does not store it.

``--pairs`` replaces the CAP pairing at every epoch by the row indices that
``make_pairs.py`` stores, and appends the file's stem to every configuration name, as
in ``frozen_pairs_random``.

Beside the three strategies of the training callbacks, the replay selects with the
label-free criteria of ``scripts/analysis/label_free_criteria.py``, each computed per
epoch on the declared score and selected through the same EMA as drift and w1 (first
value raw, then 0.9 ema + 0.1 raw, no value skipped) and the same checkpoint rule.
Their keys, ``val/summary/<strategy>_ema``, are the replay's own: no training callback
logs them. Strategy (mode): what it reads beside validation ``normal``.

    npd (max)      aux autouad, the dump's own set when the dump holds one, else the
                   side-car's; traced with npd_stored (against the dump's gaussian
                   set) and npd_gap (mean of autouad minus mean of normal)
    em (max)       aux uniform and the dump's box volume; traced with em_100k (against
                   uniform_100k) and em_reached (1 when EM reaches t_max)
    mv (min)       aux uniform; traced with mv_100k
    rtm, eag (max) nothing else
    udr, mc (max)  the values of the consensus side-car
    sds (min)      aux sds_targets and sds_outliers, at the epoch's operational
                   threshold and tie weight
    entropy (min)  the per-sample training loss train/train/loss/full
    last (max)     nothing: the value is the epoch, so the last epoch is selected

The sets and values that a dump lacks come from the side-car roots of
``pool_sidecars.py``, given with ``--sidecar`` (repeatable): ``<root>/<run_name>/``,
where ``run_name`` is the parent of the score directory, holds one file per epoch whose
sets and values are added to the epoch's record. A strategy whose inputs are missing
raises with the run's path.

Usage
-----
    python scripts/cchamber/offline_select.py pool/*/scores \\
        --cap-config design/*.yaml --strategies cap --torch-seed 20261007 \\
        --out design_out --workers 48
    python scripts/cchamber/offline_select.py pool/*/scores \\
        --cap-config configs/cap/frozen.yaml --strategies cap \\
        --torch-seed 20261007 --out frozen_out --workers 48
    python scripts/cchamber/offline_select.py pool/*/scores \\
        --strategies drift w1 --torch-seed 20261007 --out baselines_out
    python scripts/cchamber/offline_select.py pool/*/scores \\
        --strategies npd em mv rtm eag udr mc sds last --torch-seed 20261007 \\
        --sidecar sidecars/rescore --sidecar sidecars/consensus \\
        --out criteria_out --workers 16
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import multiprocessing
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAveragePrecision

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis import label_free_criteria as lfc  # noqa: E402
from src.algorithms import ADLightningModule  # noqa: E402
from src.callbacks.cap import CAPCallback  # noqa: E402
from src.callbacks.metrics.rate import AnomalyRate  # noqa: E402
from src.callbacks.thres_drift import ThresholdDriftCallback  # noqa: E402
from src.callbacks.wasserstein import WassersteinCallback  # noqa: E402

PAIR = {"dataset_1": "normal", "dataset_2": "reference_normal"}
# The registered pairing of registration.yaml, since the cap config group carries the
# metric's parameters only.
PAIRING = "cdf"
# The training callbacks of the agnostic experiments.
DRIFT_KWARGS = {"calibration_fraction": 0.5, "split_seed": 12345, "ema_weight": 0.9}
W1_KWARGS = {"apply_log1p": True, "ema_weight": 0.9}
# Strategy: (EMA key its callback logs and its ModelCheckpoint monitors, mode).
MONITORS = {
    "cap": ("val/summary/cap_ema_normal_vs_reference_normal", "max"),
    "drift": ("val/summary/operational_drift_ema", "min"),
    "w1": ("val/summary/w1dist_ema_normal_vs_reference_normal", "min"),
}
# The replay-only strategies: (EMA key, mode), keys that no training callback logs.
REPLAY_MONITORS = {
    "npd": ("val/summary/npd_ema", "max"),
    "em": ("val/summary/em_ema", "max"),
    "mv": ("val/summary/mv_ema", "min"),
    "rtm": ("val/summary/rtm_ema", "max"),
    "eag": ("val/summary/eag_ema", "max"),
    "udr": ("val/summary/udr_ema", "max"),
    "mc": ("val/summary/mc_ema", "max"),
    "sds": ("val/summary/sds_ema", "min"),
    "entropy": ("val/summary/entropy_ema", "min"),
    "last": ("val/summary/last_ema", "max"),
}
STRATEGIES = {**MONITORS, **REPLAY_MONITORS}
# The (part, name) entries of an epoch's record that a replay strategy reads beside
# validation normal.
INPUTS = {
    "npd": [("aux", "autouad"), ("aux", "gaussian")],
    "em": [("aux", "uniform"), ("aux", "uniform_100k")],
    "mv": [("aux", "uniform"), ("aux", "uniform_100k")],
    "udr": [("values", "udr")],
    "mc": [("values", "mc")],
    "sds": [("aux", "sds_targets"), ("aux", "sds_outliers")],
    "entropy": [("train", "train")],
}
# The weight of the previous value in the EMA of the training callbacks.
EMA_WEIGHT = 0.9
# The per-sample loss that the dump stores for the training set.
TRAIN_KEY = "loss/full"
# The runs that compete for selection: candidates of one family and seed.
GROUP = ("config", "strategy", "family", "seed")
DEFAULT_PATTERN = r"(?P<family>[a-z]+)_(?P<candidate>\d+)_s(?P<seed>\d+)"


def _stored_pairs(pairs: tuple, n_1: int, n_2: int) -> tuple:
    # The indices count rows in loader order, the order in which the dump stores scores.
    if not n_1 == n_2 == len(pairs[0]):
        raise ValueError(f"{len(pairs[0])} stored pairs for {n_1} and {n_2} scores.")
    return pairs


# The training callbacks log the EMA, not the epoch's value; these keep that value too.
class _CAP(CAPCallback):
    def __init__(self, pairs: tuple | None = None, **kwargs):
        super().__init__(**kwargs)
        if pairs is not None:
            self.pairing_fn = lambda s1, s2: _stored_pairs(pairs, len(s1), len(s2))

    def _compute_cap(self) -> float:
        self.raw = super()._compute_cap()
        return self.raw

    def _store_curve(self, trainer) -> dict:
        # The replay writes no curve file; it keeps the two values training logs.
        self.beta_star = self.capmetric.beta_star
        self.at_bound = self.capmetric.at_bound
        prefix = f"val/summary/{self.metric_key}"
        return {
            f"{prefix}_beta_star_{self.pair_name}": self.beta_star,
            f"{prefix}_at_bound_{self.pair_name}": float(self.at_bound),
        }


class _Drift(ThresholdDriftCallback):
    def _compute_drift_ema(self, trate_name: str, drift: float):
        self.raw = drift
        super()._compute_drift_ema(trate_name, drift)


class _W1(WassersteinCallback):
    def _compute_wasserstein(self) -> float:
        self.raw = super()._compute_wasserstein()
        return self.raw


class _Module:
    """The attributes of a LightningModule that the training callbacks read."""

    def __init__(self, meta: dict):
        self.device = torch.device("cpu")
        self.hparams = SimpleNamespace(
            target_rate=meta["target_rate"], base_rate=meta["base_rate"]
        )
        self.logged: dict[str, float] = {}

    def log_dict(self, metrics: dict, **kwargs) -> None:
        self.logged.update(metrics)


def _as_logged(value: float, batch_size: int) -> float:
    """Return a value as Lightning holds one epoch-level `self.log` call of it.

    The value is cast to float32 and mean-reduced over the one call, weighted by the
    batch size of the hook that logged it: the last batch inside
    `on_validation_batch_end`, 1 at epoch end. ModelCheckpoint compares these values,
    not the float64 ones.
    """
    total = torch.tensor(0.0) + torch.tensor(value) * batch_size
    return (total / torch.tensor(batch_size)).item()


def _callbacks(strategies: list[str], key: str, cap_cfg: dict) -> dict:
    make = {
        "cap": lambda: _CAP(output_name=key, **PAIR, **copy.deepcopy(cap_cfg)),
        "drift": lambda: _Drift(output_name=key, **DRIFT_KWARGS),
        "w1": lambda: _W1(output_name=key, **PAIR, **W1_KWARGS),
    }
    callbacks = {s: make[s]() for s in strategies if s in make}
    for callback in callbacks.values():
        callback.on_fit_start(None, None)

    return callbacks


def _validation_epoch(
    callback, module: _Module, val: dict, key: str, seed: int, monitor: str
):
    """Run one validation epoch of a training callback; return its value and EMA.

    Each set enters as one batch, in loader order: the callbacks concatenate their
    batches before computing anything.
    """
    trainer = SimpleNamespace(val_dataloaders=dict.fromkeys(val), sanity_checking=False)
    module.logged.clear()
    callback.on_validation_epoch_start(trainer, module)
    for idx, name in enumerate(val):
        outputs = {key: val[name]}
        callback.on_validation_batch_end(trainer, module, outputs, None, 0, idx)

    torch.manual_seed(seed)
    callback.on_validation_epoch_end(trainer, module)
    return callback.raw, module.logged[monitor]


def _operational(normal: torch.Tensor, batch_sizes: list[int], fpr: float) -> dict:
    """ascore_operational of normal, as training logs it and as the evaluator reads it.

    Training sums the batch values in float64 and logs their mean; the evaluator's
    OutputCallback averages them with a MeanMetric.
    """
    module = SimpleNamespace(target_fpr=fpr)
    values = [
        ADLightningModule.compute_operational_ascore(module, batch)
        for batch in normal.split(batch_sizes)
    ]
    evaluated = MeanMetric()
    for value in values:
        evaluated.update(value)

    logged = _as_logged(sum(values) / len(values), batch_sizes[-1])
    return {"operational_logged": logged, "operational": evaluated.compute().item()}


def _strategy_values(callbacks: dict, module: _Module, val: dict, key: str, seed: int):
    """Each strategy's value and EMA at an epoch, with CAP's beta_star and at_bound."""
    row = {}
    for strategy, callback in callbacks.items():
        monitor = MONITORS[strategy][0]
        value, ema = _validation_epoch(callback, module, val, key, seed, monitor)
        row.update({strategy: value, f"{strategy}_ema": ema})
    if "cap" in callbacks:
        cap = callbacks["cap"]
        row.update(cap_beta_star=cap.beta_star, cap_at_bound=float(cap.at_bound))

    return row


def _volume(meta: dict) -> float:
    """Volume of the box of the dump's uniform set."""
    return float(torch.prod((meta["aux_upper"] - meta["aux_lower"]).double()))


def _npd(scores: dict, meta: dict, epoch: int) -> dict:
    normal, aux = scores["val"]["normal"], scores["aux"]
    row = {"npd": lfc.npd(normal, aux["autouad"])}
    row["npd_stored"] = lfc.npd(normal, aux["gaussian"])
    row["npd_gap"] = (aux["autouad"].double().mean() - normal.double().mean()).item()
    return row


def _em(scores: dict, meta: dict, epoch: int) -> dict:
    normal, aux, volume = scores["val"]["normal"], scores["aux"], _volume(meta)
    em, reached = lfc.excess_mass(normal, aux["uniform"], volume)
    em_100k, _ = lfc.excess_mass(normal, aux["uniform_100k"], volume)
    return {"em": em, "em_100k": em_100k, "em_reached": float(reached)}


def _mv(scores: dict, meta: dict, epoch: int) -> dict:
    normal, aux, volume = scores["val"]["normal"], scores["aux"], _volume(meta)
    mv_100k = lfc.mass_volume(normal, aux["uniform_100k"], volume)
    return {"mv": lfc.mass_volume(normal, aux["uniform"], volume), "mv_100k": mv_100k}


def _sds(scores: dict, meta: dict, epoch: int) -> dict:
    # The operational cut of the epoch, as _epoch_metrics sets it.
    rate = AnomalyRate(meta["target_rate"], meta["base_rate"])
    rate.set_threshold(scores["val"]["normal"])
    cut = (rate.threshold.item(), rate.tie_weight.item())
    aux = scores["aux"]
    return {"sds": lfc.sds_error(aux["sds_targets"], aux["sds_outliers"], *cut)}


# Strategy: the trace columns of one epoch, the strategy's own value under its name.
REPLAY_VALUES = {
    "npd": _npd,
    "em": _em,
    "mv": _mv,
    "rtm": lambda scores, meta, epoch: {"rtm": lfc.rtm(scores["val"]["normal"])},
    "eag": lambda scores, meta, epoch: {"eag": lfc.eag(scores["val"]["normal"])},
    "udr": lambda scores, meta, epoch: {"udr": scores["values"]["udr"]},
    "mc": lambda scores, meta, epoch: {"mc": scores["values"]["mc"]},
    "sds": _sds,
    "entropy": lambda scores, meta, epoch: {
        "entropy": lfc.loss_entropy(scores["train"]["train"])
    },
    "last": lambda scores, meta, epoch: {"last": float(epoch)},
}


class _Criterion:
    """A replay-only strategy, with the EMA of ThresholdDriftCallback and
    WassersteinCallback: the first value as it is, then w ema + (1 - w) value, a
    non-finite value included."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.ema = None

    def update(self, scores: dict, meta: dict, epoch: int) -> dict:
        row = REPLAY_VALUES[self.strategy](scores, meta, epoch)
        value = float(row[self.strategy])
        if self.ema is None:
            self.ema = value
        else:
            self.ema = EMA_WEIGHT * self.ema + (1 - EMA_WEIGHT) * value
        return {**row, f"{self.strategy}_ema": self.ema}


def _replay_values(criteria: list[_Criterion], scores: dict, meta: dict, epoch: int):
    row = {}
    for criterion in criteria:
        row.update(criterion.update(scores, meta, epoch))
    return row


def _trace(run: dict, strategies: list[str], key: str, cap_cfg: dict, seed: int):
    """Per-epoch values of one run, as training computes and logs them."""
    meta, callbacks = run["meta"], _callbacks(strategies, key, cap_cfg)
    criteria = [_Criterion(s) for s in strategies if s in REPLAY_MONITORS]
    module, sizes = _Module(meta), meta["val"]["batch_sizes"]["normal"]
    rows = []
    for epoch, scores in enumerate(run["epochs"]):
        val = scores["val"]
        operational = _operational(val["normal"], sizes, meta["target_fpr"])
        values = _strategy_values(callbacks, module, val, key, seed)
        values.update(_replay_values(criteria, scores, meta, epoch))
        rows.append({"epoch": epoch, **operational, **values})

    return rows


def _checkpoint_epoch(values: list[float], mode: str) -> int:
    """Epoch that ModelCheckpoint(save_top_k=1) keeps for a monitored sequence.

    The first epoch is always saved, a NaN is stored as the worst value, and only a
    strict improvement replaces the kept epoch, so a tie keeps the earlier one.
    """
    first = values[0]
    if math.isnan(first):
        first = -math.inf if mode == "max" else math.inf

    best_epoch, best = 0, first
    for epoch, value in enumerate(values[1:], start=1):
        if (value > best) if mode == "max" else (value < best):
            best_epoch, best = epoch, value

    return best_epoch


def _select(trace: list[dict], strategy: str) -> dict:
    """The one candidate of a strategy: the epoch its ModelCheckpoint keeps."""
    monitor, mode = STRATEGIES[strategy]
    emas = [_as_logged(row[f"{strategy}_ema"], 1) for row in trace]
    epoch = _checkpoint_epoch(emas, mode)
    row, cap = trace[epoch], strategy == "cap"
    unit = {"selected_ckpt": monitor.rsplit("/", 1)[1], "selected_epoch": epoch}
    unit.update(q_main=row[strategy], q_sec=row["operational"])
    unit["beta_star"] = row["cap_beta_star"] if cap else None
    unit["at_bound"] = row["cap_at_bound"] if cap else None
    return unit


def _signal_sets(layout: dict) -> list[str]:
    """Sets the efficiency and AUPRC callbacks treat as signal: every label positive."""
    return [name for name in layout["names"] if min(layout["labels"][name]) > 0]


def _efficiency(
    threshold: torch.Tensor, tie_weight: torch.Tensor, scores: torch.Tensor, meta: dict
) -> float:
    rate = AnomalyRate(meta["target_rate"], meta["base_rate"])
    rate.apply_threshold(threshold, tie_weight)
    rate.update(scores)
    return rate.compute("efficiency").item()


def _auprc(normal: torch.Tensor, signal: torch.Tensor) -> float:
    # As AnomalyAUPRCCallback._average_precision: normal is class 0, signal class 1.
    preds = torch.cat([normal, signal]).float()
    target = torch.cat(
        [
            torch.zeros_like(normal, dtype=torch.long),
            torch.ones_like(signal, dtype=torch.long),
        ]
    )
    return BinaryAveragePrecision()(preds, target).item()


def _metric_row(scores: dict, name: str, cut: tuple, meta: dict) -> dict:
    """Efficiency and AUPRC of one set at a (threshold, tie weight) cut."""
    threshold, tie_weight = cut
    auprc = None if name == "normal" else _auprc(scores["normal"], scores[name])
    efficiency = _efficiency(threshold, tie_weight, scores[name], meta)
    row = {"dataset": name, "threshold": threshold.item()}
    row["tie_weight"] = tie_weight.item()
    return {**row, "efficiency": efficiency, "auprc": auprc}


def _epoch_metrics(scores: dict, meta: dict) -> list[dict]:
    """Efficiency at the operational threshold of normal and of every signal set, and
    AUPRC of every signal set against normal.

    The threshold and its tie weight are those the efficiency callback stores on the
    module at that epoch, from the validation normal scores; the test split reuses them,
    as the evaluation callback does with the checkpoint's buffers.
    """
    rate = AnomalyRate(meta["target_rate"], meta["base_rate"])
    rate.set_threshold(scores["val"]["normal"])
    cut = (rate.threshold, rate.tie_weight)
    return [
        {"split": split, **_metric_row(scores[split], name, cut, meta)}
        for split in ("val", "test")
        for name in ["normal", *_signal_sets(meta[split])]
    ]


def _means(rows: list[dict]) -> dict[str, float]:
    means = {}
    for split in ("val", "test"):
        signal = [r for r in rows if r["split"] == split and r["dataset"] != "normal"]
        for quantity in ("efficiency", "auprc"):
            total = sum(r[quantity] for r in signal)
            means[f"{quantity}_{split}_mean"] = total / len(signal)

    return means


def _keyed(sets: dict, key: str) -> dict:
    return {name: scores[key] for name, scores in sets.items()}


def _merge(scores: dict, path: Path, key: str, epoch: int) -> None:
    """Add the sets and values of a side-car epoch file to an epoch's scores.

    A name already present raises, so that no stored set is replaced.
    """
    sidecar = torch.load(path, weights_only=True)
    if sidecar["epoch"] != epoch:
        raise RuntimeError(f"{path} holds epoch {sidecar['epoch']}, not {epoch}.")
    new = {"aux": _keyed(sidecar["aux"], key), "values": sidecar["values"]}
    for part, entries in new.items():
        repeated = sorted(set(entries) & set(scores[part]))
        if repeated:
            raise ValueError(f"{path} repeats {part} entries {repeated}.")
        scores[part].update(entries)


def _epoch_scores(path: Path, key: str, epoch: int, sidecars: tuple[Path, ...] = ()):
    """Scores of one key in every split of an epoch's record, the training losses, and
    the sets and values of the side-car files of that epoch."""
    record = torch.load(path, weights_only=True)
    if record["epoch"] != epoch:
        raise RuntimeError(f"{path.parent}: epoch {epoch} is missing.")
    if key not in record["val"]["normal"]:
        raise KeyError(f"{path.parent} stores {sorted(record['val']['normal'])}.")

    scores = {s: _keyed(record.get(s, {}), key) for s in ("val", "test", "aux")}
    scores.update(train=_keyed(record.get("train", {}), TRAIN_KEY), values={})
    for sidecar in sidecars:
        _merge(scores, sidecar, key, epoch)
    return scores


def _sidecar_paths(directory: Path, n_epochs: int) -> list[Path]:
    """A run's side-car epoch files under one root, none if the root lacks the run."""
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("epoch_*.pt"))
    if len(paths) != n_epochs:
        raise RuntimeError(f"{directory}: {len(paths)} files for {n_epochs} epochs.")
    return paths


def _load_run(run_dir: Path, key: str, sidecars: tuple[Path, ...] = ()) -> dict:
    """Meta and the per-epoch scores of one key, with the side-cars merged in."""
    paths = sorted(run_dir.glob("epoch_*.pt"))
    if not paths:
        raise RuntimeError(f"{run_dir} holds no epoch files.")

    name, n = run_dir.parent.name, len(paths)
    extra = [_sidecar_paths(root / name, n) for root in sidecars]
    extra = [[files[e] for files in extra if files] for e in range(n)]
    epochs = [_epoch_scores(path, key, e, extra[e]) for e, path in enumerate(paths)]
    meta = torch.load(run_dir / "meta.pt", weights_only=True)
    return {"meta": meta, "epochs": epochs}


def _check_inputs(run: dict, strategies: list[str], run_dir: str) -> None:
    for strategy in strategies:
        for part, name in INPUTS.get(strategy, ()):
            if any(name not in scores[part] for scores in run["epochs"]):
                raise ValueError(
                    f"{run_dir}: {strategy} needs {part}/{name}, which neither the "
                    "dump nor a side-car holds."
                )


def _units(run: dict, trace: list[dict], strategies: list[str], head: dict):
    """Selection rows of one run and the metrics at their selected epochs."""
    units, metrics = [], {}
    for strategy in strategies:
        unit = {**head, "strategy": strategy, **_select(trace, strategy)}
        epoch = unit["selected_epoch"]
        if epoch not in metrics:
            metrics[epoch] = _epoch_metrics(run["epochs"][epoch], run["meta"])
        units.append({**unit, **_means(metrics[epoch])})

    return units, metrics


def _score_task(task: tuple) -> dict:
    """Trace, selection and metrics of one run under one CAP configuration."""
    identity, config, cap_cfg, strategies, key, seed, sidecars = task
    run = _load_run(Path(identity["run"]), key, sidecars)
    _check_inputs(run, strategies, identity["run"])
    trace = _trace(run, strategies, key, cap_cfg, seed)
    head = {"config": config, **identity, "score_key": key, "torch_seed": seed}
    units, metrics = _units(run, trace, strategies, head)
    trace = [{**head, **row} for row in trace]
    return {
        "identity": identity,
        "key": key,
        "units": units,
        "trace": trace,
        "metrics": metrics,
    }


def _init_worker() -> None:
    torch.set_num_threads(1)


def _map(tasks: list[tuple], workers: int):
    if workers <= 1:
        _init_worker()
        yield from map(_score_task, tasks)
        return

    # Spawned, not forked: torch's thread pools do not survive a fork once the parent
    # has used them.
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker) as pool:
        yield from pool.imap_unordered(_score_task, tasks)


class _CsvStream:
    """A DictWriter whose columns are those of the first rows written."""

    def __init__(self, path: Path):
        self.file = open(path, "w", newline="")
        self.writer = None

    def write(self, rows: list[dict]) -> None:
        if self.writer is None and rows:
            self.writer = csv.DictWriter(self.file, fieldnames=list(rows[0]))
            self.writer.writeheader()
        if rows:
            self.writer.writerows(rows)


def _metric_rows(result: dict, seen: set) -> list[dict]:
    """Rows of the (run, epoch) pairs not written yet; they repeat across configs."""
    rows = []
    for epoch, metrics in result["metrics"].items():
        if (result["identity"]["run"], epoch) in seen:
            continue
        seen.add((result["identity"]["run"], epoch))
        head = {**result["identity"], "score_key": result["key"], "epoch": epoch}
        rows += [{**head, **row} for row in metrics]

    return rows


def _report_progress(done: int, total: int, start: float) -> None:
    if done % max(1, total // 100) == 0 or done == total:
        elapsed = time.monotonic() - start
        print(f"{done}/{total} tasks, {elapsed:.0f} s", file=sys.stderr)


def _write_streams(results, out: Path, total: int) -> list[dict]:
    """Write traces and metrics as tasks finish; return the selection rows."""
    trace, metrics = _CsvStream(out / "trace.csv"), _CsvStream(out / "metrics.csv")
    units, seen, start = [], set(), time.monotonic()
    for done, result in enumerate(results, start=1):
        trace.write(result["trace"])
        metrics.write(_metric_rows(result, seen))
        units += result["units"]
        _report_progress(done, total, start)

    trace.file.close()
    metrics.file.close()
    return units


def _badness(value: float | None, maximise: bool) -> float:
    if value is None or math.isnan(value):
        return math.inf
    return -value if maximise else value


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a != b


def _flag_group(group: list[dict]) -> None:
    """Mark the Pareto front over (Q', Q'') and the best-Q' run, ties to the lower
    id."""
    maximise = STRATEGIES[group[0]["strategy"]][1] == "max"
    points = [
        (_badness(u["q_main"], maximise), _badness(u["q_sec"], False)) for u in group
    ]
    best = min(range(len(group)), key=lambda i: (points[i][0], group[i]["candidate"]))
    for i, unit in enumerate(group):
        unit["pareto"] = not any(_dominates(p, points[i]) for p in points)
        unit["best"] = i == best


def _flag_fronts(units: list[dict]) -> None:
    groups = defaultdict(list)
    for unit in units:
        groups[tuple(unit[k] for k in GROUP)].append(unit)

    for group in groups.values():
        _flag_group(group)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_identity(run_dir: str, pattern: str) -> dict:
    match = re.search(pattern, run_dir)
    if match is None:
        raise ValueError(f"{run_dir!r} does not match {pattern!r}.")

    numbers = {"candidate": int(match["candidate"]), "seed": int(match["seed"])}
    return {"run": run_dir, "family": match["family"], **numbers}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="score directories, one per run")
    parser.add_argument("--cap-config", nargs="+", default=[], type=Path)
    parser.add_argument("--torch-seed", type=int, required=True)
    parser.add_argument("--score-key", default="ascore/full")
    parser.add_argument(
        "--strategies", nargs="+", default=list(MONITORS), choices=list(STRATEGIES)
    )
    parser.add_argument("--run-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--sidecar", action="append", default=[], type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _cap_config(mapping: dict) -> dict:
    """A CAP configuration in the nested design format, from either format."""
    if "cap_metric_config" in mapping:
        return mapping
    return {"pairing_type": PAIRING, "cap_metric_config": mapping}


def _with_pairs(configs: list[tuple], path: Path | None) -> list[tuple]:
    if path is None:
        return configs

    pairs = torch.load(path, weights_only=True)
    pairs = (pairs["normal"], pairs["reference_normal"])
    return [(f"{name}_{path.stem}", {**cfg, "pairs": pairs}) for name, cfg in configs]


def _cap_configs(paths: list[Path], pairs: Path | None) -> list[tuple]:
    configs = [(p.stem, _cap_config(yaml.safe_load(p.read_text()))) for p in paths]
    return _with_pairs(configs, pairs)


def _tasks(args: argparse.Namespace) -> list[tuple]:
    """One task per run and CAP configuration."""
    configs = _cap_configs(args.cap_config, args.pairs)
    runs = [_run_identity(r, args.run_pattern) for r in args.runs]
    if len({(r["family"], r["candidate"], r["seed"]) for r in runs}) != len(runs):
        raise SystemExit("Two run directories map to the same run key.")

    fixed = (args.strategies, args.score_key, args.torch_seed, tuple(args.sidecar))
    configs = configs or [("none", {})]
    return [(run, name, cfg, *fixed) for run in runs for name, cfg in configs]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if "cap" in args.strategies and not args.cap_config:
        raise SystemExit("The cap strategy needs at least one --cap-config.")

    tasks = _tasks(args)
    args.out.mkdir(parents=True, exist_ok=True)
    units = _write_streams(_map(tasks, args.workers), args.out, len(tasks))
    _flag_fronts(units)
    units.sort(key=lambda unit: [unit[k] for k in (*GROUP, "candidate")])
    _write_csv(args.out / "selection.csv", units)


if __name__ == "__main__":
    main()
