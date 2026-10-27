import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import pytorch_lightning as pl
import torch
import yaml
from hydra import compose, initialize_config_dir
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.components.rate import AnomalyRate
from src.callbacks.components.selection import W_STEPS, best_epoch, ema, smoothed
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback
from src.evaluation.callbacks.utils.mlflow import log_metrics_to_mlflow
from src.evaluation.callbacks.wasserstein import _wasserstein_1d
from src.evaluation.components.pairing import hyperpoint
from src.evaluation.components.rebuild import null_sets, signal_sets
from src.evaluation.dump import RunEvaluation
from src.evaluation.evaluator import Evaluator
from src.evaluation.lattice import select_lattice
from src.evaluation.replay import ReplayModule, as_typical
from src.utils.omegaconf import register_resolvers
from tests.test_score_dump import PASSES, ToyAE, ToyDataModule, dump_callback, fit, load

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DOMAINS = ["cchamber", "cchamber_typical", "cchamber_wt", "cifar10", "mscct"]
# The sets of the toy dump, which has none of the chamber's null pools.
TOY_SETS = {"normal", "reference_normal", "sig_a", "sig_b"}
# The toy's one stored set of label 0 besides normal, served as the typical set.
TYPICAL = "reference_normal"
# Two of the light tunnel's single interventions, pooled under the override.
POOL = ["uniform_t_ir_3_strong", "uniform_osr_c_weak"]
EFF = "eff/eff_operational_"
# The columns that summarise the signal or null sets, which the override changes.
SUMMARIES = set(
    "oracle cvar25 cvar10 ema/oracle ema/cvar25 ema/cvar10 eff/eff_operational "
    "eff/eff_med_operational eff/eff_min_operational test_mean_eff null_rate".split()
)
TABLES = ["picks.csv", "run.json", "test.csv", "val.csv"]
RUN_KEYS = {
    "run",
    "lattice",
    "dump",
    "hyperpoint",
    "mlflow_run_id",
    "epochs",
    "module",
    "signals",
    "nulls",
    "stored",
    "typical",
}
# The overrides of a replay from a scores-only copy of a dump, which leaves SDS out:
# its pseudo-sets are not stored, so they would be rescored from the weights.
SCORES_ONLY = [
    "evaluation.evaluator.extra_sets.sds=null",
    "~evaluation.evaluator.dump_callbacks.sds",
    "~selection.criteria.sds",
]
SELECTION_TABLES = {
    "consensus.csv",
    "selection.csv",
    "validity.csv",
    "tradeoff.csv",
    "oracle_position.csv",
    "interventions.csv",
    "toy.csv",
    "toy_eval.csv",
}
TOY_RUN_CONFIG = {
    "data": {"_target_": "tests.test_score_dump.ToyDataModule"},
    "algorithm": {"_target_": "tests.test_score_dump.ToyAE"},
}
SMOKE_OVERRIDES = [
    "trainer.gradient_clip_val=2.0",
    "algorithm.delta=5.0",
    "algorithm.input_noise_std=0.001",
    "algorithm.encoder.nodes=[32,16]",
    "algorithm.optimizer.lr=0.000748389282753007",
    "experiment=cchamber/ae",
    "hparams_search=cchamber/ae_optuna",
    "trainer=gpu",
    "trainer.max_epochs=2",
]


@pytest.fixture(scope="module", autouse=True)
def _resolvers() -> None:
    # A second registration raises.
    if not OmegaConf.has_resolver("short_hash"):
        register_resolvers()


def _compose(domain: str, *overrides: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        overrides = [f"eval={domain}", "lattice=x", *overrides]
        return compose(config_name="eval", overrides=overrides)


def drop_absent_sets(cfg, sets: set[str]) -> None:
    # A pairing with a set the loaders lack has no scores to compare and raises.
    callbacks = cfg.evaluation.evaluator.dump_callbacks
    criteria = cfg.selection.criteria
    paired = {name: c.get("dataset_2", "normal") for name, c in callbacks.items()}
    absent = [name for name, other in paired.items() if other not in sets]
    with open_dict(cfg):
        for name in absent:
            del callbacks[name]
        for name, spec in list(criteria.items()):
            if spec.column.split("/")[0] in absent:
                del criteria[name]
        nulls = callbacks.anomaly_efficiency.null_ds
        callbacks.anomaly_efficiency.null_ds = [name for name in nulls if name in sets]


def toy_bank(cfg) -> None:
    # ot assigns X1 to subsamples of X2 of X1's size, so it needs N2 >= N1, and the
    # toy's normal (70 rows) outnumbers its reference_normal (40): the control is
    # paired the other way round, with the nn caliper set on normal.
    control = cfg.evaluation.evaluator.dump_callbacks.cap_bank_reference_normal
    control.dataset_1, control.dataset_2 = "reference_normal", "normal"
    control.reference = "normal"


# Placeholders for the beta_fix that the test bed declares, without which every
# posterior callback raises; one per probe, so that a probe read at another's shows.
TOY_BETA_FIX = {"autouad": 3.0, "uniform": 12.0, "gaussian": 40.0}


def toy_posterior(cfg) -> None:
    cfg.cap_posterior.beta_fix = TOY_BETA_FIX


def _evaluator(sets: set[str]) -> Evaluator:
    cfg = _compose("cchamber")
    drop_absent_sets(cfg, sets)
    toy_posterior(cfg)
    if sets == TOY_SETS:
        toy_bank(cfg)
    return hydra.utils.instantiate(cfg.evaluation.evaluator)


@pytest.fixture(scope="module")
def chamber() -> Evaluator:
    return _evaluator(TOY_SETS)


@pytest.fixture(scope="module")
def dump(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("dump")
    path = root / "scores" / "toy" / "run_a"
    fit(root, [dump_callback(path)])
    return SimpleNamespace(root=root, path=path, meta=load(path / "meta.pt"))


class Capture(Callback):
    def __init__(self):
        self.outputs = defaultdict(list)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.outputs[list(trainer.test_dataloaders)[dataloader_idx]].append(outputs)


def _test(replay, split, loaders, callbacks, plots=False) -> dict:
    trainer = pl.Trainer(
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    trainer.split, trainer.metric_records = split, {}
    trainer.strat_name = trainer.metric_name = trainer.criterion_name = None
    if not plots:
        trainer.plot_checkpoints = False
    replay.serve(split)
    trainer.test(replay, dataloaders=loaders, verbose=False)
    return trainer.metric_records


def _replay(dump, epoch_file: str, out: Path, record=None) -> ReplayModule:
    replay = ReplayModule(ToyAE(), dump.meta)
    record = load(dump.path / epoch_file) if record is None else record
    replay.load_epoch(record, out / epoch_file)
    return replay


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def _floats(rows: list[dict], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def test_replay_serves_stored_scores_batch_by_batch(dump, tmp_path):
    record = load(dump.path / "epoch_0001.pt")
    # Shifted, so that only the stored values, not a rescoring, can give them.
    record["val"]["normal"]["ascore/full"] += 100.0
    replay = _replay(dump, "epoch_0001.pt", tmp_path, record)
    capture = Capture()
    _test(replay, "val", ToyDataModule().loaders("val"), [capture])

    for name, scores in record["val"].items():
        sizes = dump.meta["val"]["batch_sizes"][name]
        served = capture.outputs[name]
        assert [len(out["ascore/full"]) for out in served] == sizes
        for key, values in scores.items():
            assert torch.equal(torch.cat([out[key] for out in served]), values)
        chunks = scores["ascore/full"].split(sizes)
        expected = [replay.module.compute_operational_ascore(c) for c in chunks]
        assert [out["ascore/operational"] for out in served] == expected

    rate = AnomalyRate(0.1, None)
    rate.set_threshold(record["val"]["normal"]["ascore/full"])
    assert torch.equal(replay.thres_operational, rate.threshold)
    assert torch.equal(replay.tie_operational, rate.tie_weight)


def test_replay_raises_on_a_foreign_layout(dump, tmp_path):
    replay = _replay(dump, "epoch_0000.pt", tmp_path)
    with pytest.raises(RuntimeError, match="not the dump's"):
        _test(replay, "val", ToyDataModule(batch_size=20).loaders("val"), [])

    loaders = ToyDataModule().loaders("val")
    x, y = loaders["normal"].dataset.tensors
    loaders["normal"] = DataLoader(TensorDataset(x[:64], y[:64]), batch_size=32)
    with pytest.raises(RuntimeError, match="not the dump's"):
        _test(replay, "val", loaders, [])


def test_rescored_sets_match_a_run_that_stored_them(tmp_path):
    stored = ["normal", "reference_normal"]
    fit(tmp_path, [dump_callback(tmp_path / "all")])
    fit(tmp_path, [dump_callback(tmp_path / "some", stored_sets=stored)])
    meta = load(tmp_path / "some" / "meta.pt")
    replay = ReplayModule(ToyAE(), meta)
    for name in PASSES:
        replay.load_epoch(load(tmp_path / "some" / name), tmp_path / name)
        every = load(tmp_path / "all" / name)
        for split in ("val", "test"):
            capture = Capture()
            _test(replay, split, ToyDataModule().loaders(split), [capture])
            for set_name in ("sig_a", "sig_b"):
                outputs = capture.outputs[set_name]
                rescored = torch.cat([out["ascore/full"] for out in outputs])
                expected = every[split][set_name]["ascore/full"]
                torch.testing.assert_close(rescored, expected)


def test_metrics_are_recorded_finite_or_not():
    trainer = SimpleNamespace(metric_records={}, logger=None)
    metrics = {"a": 0.5, "b": torch.tensor(math.inf), "c": None}
    log_metrics_to_mlflow(trainer, metrics, ckpt_name="epoch_0000", cb_name="cb")
    assert trainer.metric_records["cb/a"] == 0.5
    assert trainer.metric_records["cb/b"] == math.inf
    assert math.isnan(trainer.metric_records["cb/c"])

    bare = SimpleNamespace(logger=None)
    log_metrics_to_mlflow(bare, metrics, cb_name="cb")
    assert not hasattr(bare, "metric_records")


def test_the_switch_drops_plots_and_files_only(dump, chamber, tmp_path):
    keys = ["anomaly_efficiency", "thres_drift", "cap", "ascore_operational"]
    # The chamber's evaluation has no CAP: its config is built as the chamber's was.
    cap_metric = OmegaConf.load(CONFIGS / "cap" / "provisional.yaml")
    cap = {
        "_target_": "src.evaluation.callbacks.cap.CAP",
        "output_name": "ascore/full",
        "dataset_1": "normal",
        "dataset_2": "reference_normal",
        "pairing_type": "cdf",
        "cap_metric_config": OmegaConf.to_container(cap_metric),
        "log_raw_mlflow": False,
        "name": "cap",
    }
    available = {**chamber.dump_callbacks, "cap": cap}
    records = {}
    for plots in (True, False):
        out = tmp_path / str(plots)
        replay = _replay(dump, "epoch_0000.pt", out)
        configs = [available[key] for key in keys]
        callbacks = [hydra.utils.instantiate(config) for config in configs]
        loaders = ToyDataModule().loaders("test")
        records[plots] = _test(replay, "test", loaders, callbacks, plots)
        assert (out / "plots").exists() == plots

    assert records[True].keys() == records[False].keys()
    values = list(records[False].values())
    assert list(records[True].values()) == pytest.approx(values, nan_ok=True)
    assert {"cap/cap_beta_star", "thres_drift/drift_operational"} <= set(records[False])
    curves = list((tmp_path / "True").rglob("cap_curve.npz"))
    assert len(curves) == 1 and not list((tmp_path / "False").rglob("*"))


def test_efficiency_takes_signals_by_label_and_reports_nulls(dump, tmp_path):
    replay = _replay(dump, "epoch_0001.pt", tmp_path)

    def run(**kwargs):
        callback = AnomalyEfficiencyCallback(output_name="ascore/full", **kwargs)
        loaders = ToyDataModule().loaders("test")
        return _test(replay, "test", loaders, [callback])

    every, named = run(ds=None), run(ds=["sig_a", "sig_b"])
    nulls = run(ds=None, null_ds=["reference_normal"])
    assert every == named
    per_set = {key for key in every if key.startswith("eff/eff_operational_")}
    assert per_set == {f"eff/eff_operational_{s}" for s in ("sig_a", "sig_b", "normal")}
    null_key = "eff/eff_operational_reference_normal"
    assert {key: v for key, v in nulls.items() if key != null_key} == every

    rate = AnomalyRate(0.1, None)
    rate.apply_threshold(replay.thres_operational, replay.tie_operational)
    rate.update(replay.record["test"]["reference_normal"]["ascore/full"])
    assert nulls[null_key] == pytest.approx(rate.compute("efficiency").item())
    with pytest.raises(ValueError, match="non-zero labels"):
        run(ds=None, null_ds=["sig_a"])


def test_efficiency_reports_a_background_set_outside_the_signal_statistics(
    dump, tmp_path
):
    replay = _replay(dump, "epoch_0001.pt", tmp_path)
    x, _ = ToyDataModule().sets["test"]["sig_a"]
    background = TensorDataset(x, torch.full((len(x),), -1))
    efficiency = dict(output_name="ascore/full", ds=None, null_ds=["reference_normal"])

    def run(with_background: bool) -> dict:
        loaders = ToyDataModule().loaders("test")
        if with_background:
            loaders["bkg_a"] = DataLoader(background, batch_size=32)
        return _test(replay, "test", loaders, [AnomalyEfficiencyCallback(**efficiency)])

    plain, records = run(False), run(True)
    key, summary = "eff/eff_operational_bkg_a", "eff/eff_operational"
    assert set(records) - set(plain) == {key}
    rate = AnomalyRate(0.1, None)
    rate.apply_threshold(replay.thres_operational, replay.tie_operational)
    with torch.no_grad():
        rate.update(replay.module.model_step(background.tensors)["ascore/full"])
    assert records[key] == pytest.approx(rate.compute("efficiency").item())
    # The medians, minima and null rates are untouched; only the callback's own
    # summary takes the background in, as the signals' CVaR minus its pass rate.
    unchanged = [name for name in plain if name != summary]
    assert {k: records[k] for k in unchanged} == {k: plain[k] for k in unchanged}
    assert records[summary] == pytest.approx(plain[summary] - records[key])
    labels = {"normal": [0], "sig_a": [1], "reference_normal": [0], "bkg_a": [-1]}
    assert signal_sets({"names": list(labels), "labels": labels}) == ["sig_a"]
    nulls = null_sets([AnomalyEfficiencyCallback(**efficiency)])
    assert nulls == ["reference_normal"]


def _check_val(val: list[dict], selection: dict) -> None:
    for name, spec in selection["criteria"].items():
        values = _floats(val[1:], spec["column"])
        averages = ema(values, skip_nonfinite=spec.get("skip_nonfinite", False))
        assert val[0][f"ema/{name}"] == ""
        assert _floats(val[1:], f"ema/{name}") == pytest.approx(averages, nan_ok=True)
    for row in val:
        effs = [float(row[f"eff/eff_operational_{s}"]) for s in ("sig_a", "sig_b")]
        assert float(row["oracle"]) == pytest.approx(sum(effs) / 2)
        assert float(row["cvar25"]) == pytest.approx(min(effs))


def _check_picks(picks: list[dict], val: list[dict], selection: dict) -> None:
    labelled = [*selection["labelled"], "q_sec"]
    assert len(picks) == len(selection["criteria"]) * (W_STEPS + 1) + len(labelled)
    assert {row["epoch"] for row in picks} <= {"0", "1"}
    assert [row["k"] for row in picks if row["strategy"] in labelled] == [""] * 4
    drift = _floats(val[1:], "thres_drift/drift_operational")
    pick = [row for row in picks if row["strategy"] == "drift" and row["k"] == "10"]
    assert int(pick[0]["epoch"]) == best_epoch(smoothed(drift), "min")


def _check_mlflow(uri: str, run: dict, val: list[dict]) -> None:
    client = MlflowClient(uri)
    mlflow_run = client.get_run(run["mlflow_run_id"])
    assert mlflow_run.info.run_name == "run_a"
    assert client.get_experiment(mlflow_run.info.experiment_id).name == "toy_eval"
    for key, column in [
        ("val/summary/npd_ema", "ema/npd"),
        ("val/summary/npd", "npd/npd"),
        ("val/sig_a/eff__ascore_full__brate_operational", "eff/eff_operational_sig_a"),
    ]:
        history = client.get_metric_history(run["mlflow_run_id"], key)
        assert [m.step for m in history] == [0, 1]
        assert [m.value for m in history] == pytest.approx(_floats(val[1:], column))


def test_evaluate_dump_writes_the_four_files(dump, chamber, tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    module, datamodule = ToyAE(), ToyDataModule()
    run = chamber.evaluate_dump(dump.path, module, datamodule, tmp_path, "toy", uri)
    out = tmp_path / "runs" / "run_a"
    assert sorted(path.name for path in out.iterdir()) == TABLES
    assert json.loads((out / "run.json").read_text()) == run
    assert set(run) == RUN_KEYS and run["epochs"] == 2 and run["hyperpoint"] == {}
    assert run["signals"] == ["sig_a", "sig_b"] and run["nulls"] == []
    assert run["typical"] is None

    val, test = _read_csv(out / "val.csv"), _read_csv(out / "test.csv")
    selection = chamber.selection
    assert [row["epoch"] for row in val] == [row["epoch"] for row in test]
    assert [row["epoch"] for row in val] == ["-1", "0", "1"]
    columns = [spec["column"] for spec in selection["criteria"].values()]
    emas = [f"ema/{name}" for name in [*selection["criteria"], "oracle", "q_sec"]]
    assert {"oracle", "cvar25", "cvar10", *columns, *emas} <= set(val[0])
    _check_val(val, selection)
    for row in test:
        assert row["realised_fpr"] == row["eff/eff_operational_normal"]
        effs = [float(row[f"eff/eff_operational_{s}"]) for s in ("sig_a", "sig_b")]
        assert float(row["test_mean_eff"]) == pytest.approx(sum(effs) / 2)
        assert row["null_rate"] == ""
    _check_picks(_read_csv(out / "picks.csv"), val, selection)
    _check_mlflow(uri, run, val)


def test_typical_serves_a_stored_set_as_normal(dump, tmp_path):
    record = load(dump.path / "epoch_0001.pt")
    replay = ReplayModule(ToyAE(), dump.meta, TYPICAL)
    replay.load_epoch(record, tmp_path / "epoch_0001.pt")
    capture = Capture()
    loaders = as_typical(ToyDataModule().loaders("val"), TYPICAL)
    _test(replay, "val", loaders, [capture])

    stored, served = record["val"][TYPICAL], capture.outputs["normal"]
    assert list(capture.outputs) == ["normal", "sig_a", "sig_b"]
    sizes = dump.meta["val"]["batch_sizes"][TYPICAL]
    assert [len(out["ascore/full"]) for out in served] == sizes
    for key, values in stored.items():
        assert torch.equal(torch.cat([out[key] for out in served]), values)
    rate = AnomalyRate(0.1, None)
    rate.set_threshold(stored["ascore/full"])
    assert torch.equal(replay.thres_operational, rate.threshold)
    assert torch.equal(replay.tie_operational, rate.tie_weight)


def test_evaluate_dump_with_a_typical_set(dump, tmp_path):
    cfg = _compose("cchamber_typical")
    drop_absent_sets(cfg, TOY_SETS - {TYPICAL})
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator, typical=TYPICAL)
    settings = evaluator._dump_settings()
    run = RunEvaluation(dump.path, ToyAE(), ToyDataModule(), **settings)
    assert list(run.loaders["val"])[:3] == ["normal", "sig_a", "sig_b"]
    rows = sum(dump.meta["val"]["batch_sizes"][TYPICAL])
    assert len(run.loaders["val"]["normal"].dataset) == rows
    info = run.evaluate(tmp_path, "toy")
    out = tmp_path / "runs" / "run_a"
    assert info["typical"] == TYPICAL
    assert json.loads((out / "run.json").read_text()) == info

    # Q'' and the efficiency of sig_a at the cut, both from the typical set's scores.
    for row, name in zip(_read_csv(out / "val.csv"), PASSES):
        stored = load(dump.path / name)["val"]
        fits = stored[TYPICAL]["fit/full"].double()
        assert float(row["fit/normal"]) == pytest.approx(fits.mean().item())
        rate = AnomalyRate(0.1, None)
        rate.set_threshold(stored[TYPICAL]["ascore/full"])
        rate.update(stored["sig_a"]["ascore/full"])
        efficiency = rate.compute("efficiency").item()
        assert float(row[EFF + "sig_a"]) == pytest.approx(efficiency)


def test_evaluate_lattice_rebuilds_each_run(chamber, tmp_path):
    path = tmp_path / "scores" / "toy" / "run_b"
    fit(tmp_path, [dump_callback(path)], epochs=1)
    config = {
        "seed": 0,
        "data": {"_target_": "tests.test_score_dump.ToyDataModule"},
        "algorithm": {"_target_": "tests.test_score_dump.ToyAE"},
    }
    (path / "config.yaml").write_text(yaml.safe_dump(config))
    overrides = ["hparams_search=cchamber/ae_optuna", "algorithm.delta=5.0", "seed=0"]
    (path / "overrides.yaml").write_text(yaml.safe_dump(overrides))
    out = tmp_path / "eval"
    runs = chamber.evaluate_lattice(tmp_path / "scores", out, "toy", workers=1)
    assert [run["run"] for run in runs] == ["run_b"]
    assert runs[0]["hyperpoint"] == {"algorithm.delta": "5.0"}
    assert sorted(p.name for p in (out / "runs" / "run_b").iterdir()) == TABLES


def _strip_weights(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)
    for path in dst.glob("*/epoch_*.pt"):
        record = load(path)
        del record["state_dict"]
        torch.save(record, path)


@pytest.fixture(scope="module")
def lattices(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("lattices")
    for seed, run in enumerate(["run_a", "run_b"]):
        path = root / "full" / "toy" / run
        fit(root, [dump_callback(path)], seed=seed)
        (path / "config.yaml").write_text(yaml.safe_dump(TOY_RUN_CONFIG))
    _strip_weights(root / "full" / "toy", root / "stripped" / "toy")
    return SimpleNamespace(full=root / "full", stripped=root / "stripped")


def _evaluate_and_select(scores_dir: Path, out: Path, *overrides: str) -> None:
    # The two steps of src/eval.py, on the CIFAR-10 evaluation.
    cfg = _compose("cifar10", *overrides)
    drop_absent_sets(cfg, TOY_SETS)
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    evaluator.evaluate_lattice(scores_dir, out, "toy")
    selection = OmegaConf.to_container(cfg.selection, resolve=True)
    select_lattice(out, selection, scores_dir, "toy", None)


def _outputs(out: Path) -> dict[str, bytes]:
    paths = [path for path in out.rglob("*") if path.is_file()]
    return {str(path.relative_to(out)): path.read_bytes() for path in paths}


def test_a_scores_only_copy_replays_to_the_same_tables(lattices, tmp_path):
    stripped = load(lattices.stripped / "toy" / "run_a" / "epoch_0000.pt")
    initial = load(lattices.stripped / "toy" / "run_a" / "initial.pt")
    assert "state_dict" not in stripped and "state_dict" in initial
    outputs = {}
    for name in ("full", "stripped"):
        _evaluate_and_select(getattr(lattices, name), tmp_path / name, *SCORES_ONLY)
        outputs[name] = _outputs(tmp_path / name)

    runs = [f"runs/{run}/{table}" for run in ("run_a", "run_b") for table in TABLES]
    assert set(outputs["full"]) == set(runs) | SELECTION_TABLES
    for path, content in outputs["full"].items():
        if not path.endswith("run.json"):
            assert outputs["stripped"][path] == content, path
            continue
        info = {n: json.loads(outputs[n][path]) for n in outputs}
        assert {**info["full"], "dump": None} == {**info["stripped"], "dump": None}
    assert b"sds/" not in outputs["full"]["runs/run_a/val.csv"]


def test_a_scores_only_copy_raises_where_a_score_is_rescored(lattices, tmp_path):
    message = "Rescoring 'sds_targets' needs the weights, which run_a/epoch_0000.pt"
    with pytest.raises(RuntimeError, match=re.escape(message)):
        _evaluate_and_select(lattices.stripped, tmp_path)

    run_dir = lattices.stripped / "toy" / "run_a"
    replay = ReplayModule(ToyAE(), load(run_dir / "meta.pt"))
    record = load(run_dir / "epoch_0001.pt")
    del record["test"]["sig_a"], record["val"]["normal"]["fit/full"]
    replay.load_epoch(record, tmp_path / "run_a" / "epoch_0001.pt")
    for split, rescored in [("test", "'sig_a'"), ("val", "['fit/full'] of 'normal'")]:
        message = f"Rescoring {rescored} needs the weights, which run_a/epoch_0001.pt"
        with pytest.raises(RuntimeError, match=re.escape(message)):
            _test(replay, split, ToyDataModule().loaders(split), [])


def test_hyperpoint_takes_the_swept_overrides(tmp_path):
    (tmp_path / "overrides.yaml").write_text(yaml.safe_dump(SMOKE_OVERRIDES))
    point = hyperpoint(tmp_path)
    assert point == {
        "trainer.gradient_clip_val": "2.0",
        "algorithm.delta": "5.0",
        "algorithm.input_noise_std": "0.001",
        "algorithm.encoder.nodes": "[32,16]",
        "algorithm.optimizer.lr": "0.000748389282753007",
    }
    outside = [o for o in SMOKE_OVERRIDES if not o.startswith("hparams_search")]
    (tmp_path / "overrides.yaml").write_text(yaml.safe_dump(outside))
    assert hyperpoint(tmp_path) == {}
    assert hyperpoint(tmp_path / "missing") == {}


@pytest.mark.parametrize("domain", DOMAINS)
def test_eval_configs_build_the_evaluator(domain):
    cfg = _compose(domain)
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    assert evaluator.test_callbacks == ["anomaly_efficiency"]
    configs = evaluator.dump_callbacks
    assert all(isinstance(config, dict) for config in configs.values())
    names = {hydra.utils.instantiate(config).name for config in configs.values()}
    selection = evaluator.selection
    columns = [spec["column"] for spec in selection["criteria"].values()]
    columns += [selection["q_sec"]["column"], selection["efficiency"]]
    assert {column.split("/")[0] for column in columns} <= names
    assert configs["wasserstein"]["dataset_2"] == "reference_normal"
    null_ds = configs["anomaly_efficiency"]["null_ds"]
    sizes = {
        "cchamber": 9,
        "cchamber_typical": 9,
        "cchamber_wt": 2,
        "cifar10": 3,
        "mscct": 4,
    }
    assert len(null_ds) == sizes[domain]


@pytest.mark.parametrize("domain", ["cchamber", "cchamber_wt"])
def test_pairings_copy_the_control(domain):
    evaluator = hydra.utils.instantiate(_compose(domain).evaluation.evaluator)
    configs, criteria = evaluator.dump_callbacks, evaluator.selection["criteria"]
    specs = {spec["column"].split("/")[0]: spec for spec in criteria.values()}
    # The posterior's pairings copy a control that no criterion registers; their test
    # is in test_threshold_posterior_callback.py.
    pairings = [
        n
        for n, c in configs.items()
        if "dataset_2" in c and n in specs and not n.startswith("posterior_")
    ]
    assert len(pairings) == (4 if domain == "cchamber" else 3)
    for name in pairings:
        config, spec = configs[name], specs[name]
        control = configs[name.split("_")[0]]
        control_spec = specs[control["name"]]
        renamed = {**config, "name": control["name"], "dataset_2": "reference_normal"}
        assert renamed == control
        assert spec["column"].split("/")[1] in (name, "w1dist")
        assert spec["key"].endswith(f"_ema_normal_vs_{config['dataset_2']}")
        assert spec["mode"] == control_spec["mode"]
        assert spec.get("skip_nonfinite") == control_spec.get("skip_nonfinite")


def test_sets_replace_the_lists_of_a_dump(cchamber_data, monkeypatch, tmp_path):
    path = tmp_path / "scores" / "lt" / "run"
    data = {
        "_target_": "src.data.CausalChamber_datamodule.CausalChamberDataModule",
        "data_dir": str(cchamber_data.data_dir),
        "dataset_name": "lt_interventions_standard_v1",
        "url": "",
        "md5": "",
        "readouts": ["current", "angle_1", "angle_2", "ir_1"],
        "reference": {"file": "uniform_reference"},
        "signals": {name: {"file": name} for name in cchamber_data.signals},
        "batch_size": 16,
    }
    fit(tmp_path, [dump_callback(path)], datamodule=hydra.utils.instantiate(data))
    config = {"data": data, "algorithm": {"_target_": "tests.test_score_dump.ToyAE"}}
    (path / "config.yaml").write_text(yaml.safe_dump(config))
    sets = {
        "signals": {"uniform_red_mid": {"file": "uniform_red_mid"}},
        "nulls": {"sensor_weak": {"members": [{"file": name} for name in POOL]}},
    }
    (tmp_path / "pooled.yaml").write_text(yaml.safe_dump(sets))
    monkeypatch.setattr("src.evaluation.components.rebuild.DATA_CONFIGS", tmp_path)

    before = {"normal", "reference_normal", *cchamber_data.signals}
    after = {"normal", "reference_normal", "uniform_red_mid", "sensor_weak"}
    scores = tmp_path / "scores"
    old = _evaluator(before).evaluate_lattice(scores, tmp_path / "old", "lt")
    new_evaluator = _evaluator(after)
    new = new_evaluator.evaluate_lattice(scores, tmp_path / "new", "lt", sets="pooled")
    assert old[0]["signals"] == cchamber_data.signals
    assert new[0]["signals"] == ["uniform_red_mid"]
    assert new[0]["nulls"] == ["sensor_weak"]
    for split in ("val", "test"):
        old_rows = _read_csv(tmp_path / "old" / "runs" / "run" / f"{split}.csv")
        new_rows = _read_csv(tmp_path / "new" / "runs" / "run" / f"{split}.csv")
        for old_row, new_row in zip(old_rows, new_rows):
            kept = (old_row.keys() & new_row.keys()) - SUMMARIES
            assert {key: new_row[key] for key in kept} == {k: old_row[k] for k in kept}
            members = [float(old_row[EFF + name]) for name in POOL]
            pooled = float(new_row[EFF + "sensor_weak"])
            assert pooled == pytest.approx(sum(members) / 2)

    # The stored scores of the members against the pool rescored from the weights.
    # The pool is scored in other batches than its members were, which moves the
    # float32 rounding of its scores.
    val = _read_csv(tmp_path / "new" / "runs" / "run" / "val.csv")
    for row, name in zip(val, PASSES):
        stored = load(path / name)["val"]
        normal = stored["normal"]["ascore/full"]
        pool = torch.cat([stored[member]["ascore/full"] for member in POOL])
        w1 = _wasserstein_1d(_signed_log1p(normal), _signed_log1p(pool))
        assert float(row["wasserstein_weak/w1dist"]) == pytest.approx(w1, rel=1e-5)
