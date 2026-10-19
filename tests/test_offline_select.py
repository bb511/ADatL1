import copy
import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import average_precision_score

from scripts.cchamber import offline_select as offline
from src.callbacks.cap import CAPCallback
from src.callbacks.metrics.rate import AnomalyRate
from src.callbacks.thres_drift import ThresholdDriftCallback
from src.callbacks.wasserstein import WassersteinCallback
from src.evaluation.callbacks.cap import CAP as EvaluationCAP
from src.evaluation.callbacks.output import OutputCallback
from src.evaluation.evaluator import Evaluator
from tests.test_score_dump import ToyAE, dump_callback, fit

SEED = 5
KEY = "ascore/full"
# The metric's parameters as in configs/cap/provisional.yaml, in the flat format of the
# cap config group and in the nested format of the design configurations.
CAP_METRIC = {
    "beta_min": 0.01,
    "beta_max": 1000.0,
    "n_beta": 200,
    "normalization_type": "sigmoid",
    "normalization_params": None,
    "energy_type": "adaptive",
    "energy_params": {"scale": 0.5},
    "regularization_type": "none",
    "regularization_params": None,
    "binary": True,
}
CAP_CONFIG = {"pairing_type": "cdf", "cap_metric_config": CAP_METRIC}
PAIR_NAME = "normal_vs_reference_normal"
NAMES = ["normal", "reference_normal", "sig_a", "sig_b"]
SIZES = {"normal": 60, "reference_normal": 60, "sig_a": 30, "sig_b": 20}
SHIFTS = {"normal": 0.0, "reference_normal": 0.05, "sig_a": 0.5, "sig_b": 1.0}
LAYOUT = {
    "names": NAMES,
    "keys": [KEY],
    "labels": {"normal": [0], "reference_normal": [0], "sig_a": [1], "sig_b": [2]},
    "batch_sizes": {
        "normal": [25, 25, 10],
        "reference_normal": [60],
        "sig_a": [30],
        "sig_b": [20],
    },
}
META = {"target_rate": 0.1, "base_rate": None, "target_fpr": 0.1}
SELECTION_COLUMNS = [
    "config",
    "run",
    "family",
    "candidate",
    "seed",
    "score_key",
    "torch_seed",
    "strategy",
    "selected_ckpt",
    "selected_epoch",
    "q_main",
    "q_sec",
    "beta_star",
    "at_bound",
    "efficiency_val_mean",
    "auprc_val_mean",
    "efficiency_test_mean",
    "auprc_test_mean",
    "pareto",
    "best",
]


def synthetic_scores(gen, width):
    return {
        split: {
            name: {KEY: width * torch.rand(SIZES[name], generator=gen) + SHIFTS[name]}
            for name in NAMES
        }
        for split in ("val", "test")
    }


def write_run(path, epochs=14, seed=0):
    gen = torch.Generator().manual_seed(seed)
    path.mkdir(parents=True)
    torch.save({**META, "val": LAYOUT, "test": LAYOUT}, path / "meta.pt")
    for epoch in range(epochs):
        scores = synthetic_scores(gen, 1.0 + 0.3 * math.sin(epoch + seed))
        torch.save({"epoch": epoch, **scores}, path / f"epoch_{epoch:04d}.pt")
    return path


def leaf(strategy):
    return offline.MONITORS[strategy][0].rsplit("/", 1)[1]


def test_cap_and_its_logged_keys_follow_the_callback(tmp_path):
    run = offline._load_run(write_run(tmp_path / "ae_000_s1"), KEY)
    trace = offline._trace(run, ["cap"], KEY, CAP_CONFIG, SEED)
    callback = CAPCallback(output_name=KEY, **offline.PAIR, **copy.deepcopy(CAP_CONFIG))
    callback.on_fit_start(None, None)
    module, prefix = offline._Module(META), "val/summary/cap"
    for epoch, (row, scores) in enumerate(zip(trace, run["epochs"])):
        trainer = SimpleNamespace(
            val_dataloaders=dict.fromkeys(NAMES),
            default_root_dir=tmp_path,
            current_epoch=epoch,
        )
        callback.on_validation_epoch_start(trainer, module)
        # Training passes normal in its loader batches, the replay in one.
        val = scores["val"]
        batches = [(0, b) for b in val["normal"].split([25, 25, 10])]
        batches.append((1, val["reference_normal"]))
        for idx, batch in batches:
            outputs = {KEY: batch}
            callback.on_validation_batch_end(trainer, module, outputs, None, 0, idx)
        module.logged.clear()
        callback.on_validation_epoch_end(trainer, module)

        assert callback.capmetric.compute() == row["cap"]
        assert module.logged == {
            f"{prefix}_ema_{PAIR_NAME}": row["cap_ema"],
            f"{prefix}_beta_star_{PAIR_NAME}": row["cap_beta_star"],
            f"{prefix}_at_bound_{PAIR_NAME}": row["cap_at_bound"],
        }


class _Script(Callback):
    def __init__(self, values: dict[str, list[float]]):
        self.values = values

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        pl_module.log_dict({k: v[epoch] for k, v in self.values.items()})


def ema_checkpoint(root, strategy):
    # The layout of the experiment configs: summary/<metric>/<criterion>/<metric>.ckpt.
    monitor, mode = offline.MONITORS[strategy]
    return ModelCheckpoint(
        dirpath=root / "summary" / leaf(strategy) / mode,
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        filename=leaf(strategy),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )


def saved_epochs(paths):
    return sorted(torch.load(p, weights_only=False)["epoch"] for p in paths)


def test_checkpoint_semantics_match_lightning(tmp_path):
    # In float32, 0.7 + 1e-9 ties 0.7: the earlier epoch must be kept.
    maxima = [math.nan, 0.5, 0.7, 0.7, math.nan, 0.7 + 1e-9, 0.6, 0.0]
    minima = [1.0, math.nan, 0.4, 0.4 + 1e-9, 0.3, 0.3, 1.0, 1.0]
    values = {offline.MONITORS["cap"][0]: maxima, offline.MONITORS["drift"][0]: minima}
    checkpoints = {s: ema_checkpoint(tmp_path, s) for s in ("cap", "drift")}
    callbacks = [_Script(values), *checkpoints.values()]
    fit(tmp_path, callbacks, epochs=8, enable_checkpointing=True)

    for strategy, expected in (("cap", 2), ("drift", 4)):
        monitor, mode = offline.MONITORS[strategy]
        logged = [offline._as_logged(v, 1) for v in values[monitor]]
        assert offline._checkpoint_epoch(logged, mode) == expected
        assert saved_epochs([checkpoints[strategy].best_model_path]) == [expected]


def test_threshold_tie_weight_and_efficiency_match_anomaly_rate():
    gen = torch.Generator().manual_seed(3)
    # Scores on a 0.1 grid, so that normal scores tie with the threshold.
    scores = {
        split: {name: torch.round(10 * v[KEY]) / 10 for name, v in sets.items()}
        for split, sets in synthetic_scores(gen, 1.0).items()
    }
    rows = offline._epoch_metrics(scores, {**META, "val": LAYOUT, "test": LAYOUT})
    pairs = {(r["split"], r["dataset"]) for r in rows}
    assert pairs == {
        (s, n) for s in ("val", "test") for n in ("normal", "sig_a", "sig_b")
    }

    rate = AnomalyRate(META["target_rate"], META["base_rate"])
    rate.set_threshold(scores["val"]["normal"])
    threshold, weight = rate.threshold.item(), rate.tie_weight.item()
    assert 0.0 < weight < 1.0
    for row in rows:
        assert (row["threshold"], row["tie_weight"]) == (threshold, weight)
        x = scores[row["split"]][row["dataset"]].numpy()
        expected = np.mean(x > threshold) + weight * np.mean(x == threshold)
        assert row["efficiency"] == pytest.approx(expected)
        if row["dataset"] == "normal":
            assert row["auprc"] is None
            continue
        normal = scores[row["split"]]["normal"].numpy()
        target = np.r_[np.zeros(len(normal)), np.ones(len(x))]
        expected = average_precision_score(target, np.r_[normal, x])
        assert row["auprc"] == pytest.approx(expected, rel=1e-5)

    # The tie weight brings the efficiency of validation normal to the FPR.
    (row,) = (r for r in rows if (r["split"], r["dataset"]) == ("val", "normal"))
    assert row["efficiency"] == pytest.approx(META["target_fpr"])


class _Recorder(Callback):
    def __init__(self, keys):
        self.keys = keys
        self.values = []

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            metrics = trainer.callback_metrics
            self.values.append({k: metrics[k].item() for k in self.keys})


def cap_evaluator(root):
    # The candidate and the reported pair of configs/experiment/cchamber/*.yaml.
    callbacks = [
        EvaluationCAP(output_name=KEY, **offline.PAIR, **copy.deepcopy(CAP_CONFIG)),
        OutputCallback("ascore/operational", ["normal"], name="ascore_operational"),
    ]
    sec = {"name": "ascore_operational", "params": {"test_ds": "normal"}}
    config = {
        "candidate": {"strategy": "summary", "metric": leaf("cap"), "criterion": "max"},
        "main_metric": {
            "callback": {"name": "cap", "params": {}},
            "direction": "maximize",
        },
        "sec_metric": {"callback": sec, "direction": "minimize"},
    }
    return Evaluator(
        ckpts={"summary": {leaf("cap"): ["max"]}},
        callbacks=callbacks,
        logger=False,
        optimized_metric_config=config,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(root),
    )


def test_replay_matches_a_training_run_and_its_evaluation(tmp_path):
    strategies = list(offline.MONITORS)
    keys = [offline.MONITORS[s][0] for s in strategies]
    keys += [f"val/summary/cap_{q}_{PAIR_NAME}" for q in ("beta_star", "at_bound")]
    recorder = _Recorder(keys)
    checkpoints = {s: ema_checkpoint(tmp_path / "ckpts", s) for s in strategies}
    callbacks = [
        dump_callback(tmp_path / "scores"),
        CAPCallback(output_name=KEY, **offline.PAIR, **copy.deepcopy(CAP_CONFIG)),
        ThresholdDriftCallback(output_name=KEY, **offline.DRIFT_KWARGS),
        WassersteinCallback(output_name=KEY, **offline.PAIR, **offline.W1_KWARGS),
        recorder,
        *checkpoints.values(),
    ]
    # No sanity pass, as in configs/trainer/default.yaml: the training callbacks do
    # not skip it, so it would enter their averages, while the dump skips it.
    _, datamodule = fit(
        tmp_path,
        callbacks,
        epochs=14,
        module=ToyAE(lr=0.02),
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )

    run = offline._load_run(tmp_path / "scores", KEY)
    trace = offline._trace(run, strategies, KEY, CAP_CONFIG, SEED)
    assert len(trace) == len(recorder.values) == 14
    for row, logged in zip(trace, recorder.values):
        replayed = [row[f"{s}_ema"] for s in strategies]
        replayed += [row["cap_beta_star"], row["cap_at_bound"]]
        assert [offline._as_logged(v, 1) for v in replayed] == [logged[k] for k in keys]
    for strategy, checkpoint in checkpoints.items():
        unit = offline._select(trace, strategy)
        assert saved_epochs([checkpoint.best_model_path]) == [unit["selected_epoch"]]

    # The evaluator recomputes Q' and Q'' at the candidate checkpoint.
    evaluator = cap_evaluator(tmp_path)
    val = datamodule.loaders("val")
    evaluator.evaluate_run(
        tmp_path / "ckpts", ToyAE(), val, "val", set_optimized_metric=True
    )
    unit = offline._select(trace, "cap")
    assert evaluator.optimized_ckpt_name == unit["selected_ckpt"] == leaf("cap")
    assert evaluator.optimized_metric == [unit["q_main"], unit["q_sec"]]


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def test_main_writes_consistent_tables(tmp_path):
    runs = []
    for family in ("ae", "vae"):
        for candidate in range(3):
            path = tmp_path / "pool" / f"{family}_{candidate:03d}_s101" / "scores"
            runs.append(str(write_run(path, seed=10 * len(runs))))
    # One configuration in both formats, run with one and with two workers.
    for workers, form, config in ((1, "nested", CAP_CONFIG), (2, "flat", CAP_METRIC)):
        path = tmp_path / form / "cap_a.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(config))
        argv = [*runs, "--cap-config", str(path), "--torch-seed", str(SEED)]
        out = tmp_path / f"out_{form}"
        offline.main([*argv, "--out", str(out), "--workers", str(workers)])

    selection = read_csv(tmp_path / "out_nested" / "selection.csv")
    assert list(selection[0]) == SELECTION_COLUMNS
    assert len(selection) == 6 * 3
    for strategy, (_, mode) in offline.MONITORS.items():
        for family in ("ae", "vae"):
            group = [
                u
                for u in selection
                if u["strategy"] == strategy and u["family"] == family
            ]
            sign = -1.0 if mode == "max" else 1.0
            points = [(sign * float(u["q_main"]), float(u["q_sec"])) for u in group]
            ranked = sorted(
                zip(points, group), key=lambda p: (p[0][0], int(p[1]["candidate"]))
            )
            assert [u["best"] for u in group].count("True") == 1
            assert ranked[0][1]["best"] == "True"
            for point, unit in zip(points, group):
                dominated = any(
                    p != point and p[0] <= point[0] and p[1] <= point[1] for p in points
                )
                assert unit["pareto"] == str(not dominated)
                assert unit["selected_ckpt"] == leaf(strategy)

    trace = read_csv(tmp_path / "out_nested" / "trace.csv")
    assert len(trace) == 6 * 14
    epochs = {(row["run"], row["epoch"]): row for row in trace}
    for unit in selection:
        row = epochs[unit["run"], unit["selected_epoch"]]
        main, sec = row[unit["strategy"]], row["operational"]
        assert (unit["q_main"], unit["q_sec"]) == (main, sec)
        cap = [row["cap_beta_star"], row["cap_at_bound"]]
        expected = cap if unit["strategy"] == "cap" else ["", ""]
        assert [unit["beta_star"], unit["at_bound"]] == expected

    metrics = read_csv(tmp_path / "out_nested" / "metrics.csv")
    assert list(metrics[0])[-5:] == [
        "dataset",
        "threshold",
        "tie_weight",
        "efficiency",
        "auprc",
    ]
    for unit in selection:
        rows = [
            m
            for m in metrics
            if m["run"] == unit["run"] and m["epoch"] == unit["selected_epoch"]
        ]
        assert len(rows) == 2 * 3
    for name in ("selection.csv", "metrics.csv", "trace.csv"):
        lines = [
            sorted((tmp_path / f"out_{form}" / name).read_text().splitlines())
            for form in ("nested", "flat")
        ]
        assert lines[0] == lines[1]
