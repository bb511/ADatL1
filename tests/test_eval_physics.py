import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import yaml
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.components.decoder import Decoder
from src.algorithms.dsae import DeepSetsAE
from src.callbacks.autouad import EAGCallback, NPDCallback, RTMCallback
from src.callbacks.cap_bank import CAPBankCallback as TrainingBank
from src.callbacks.components.rate import AnomalyRate
from src.callbacks.components.selection import W_STEPS, ema
from src.callbacks.components.utils import drawn_masks, row_subset
from src.callbacks.efficiency import AnomalyEfficiencyCallback
from src.callbacks.entropy import LossEntropyCallback
from src.callbacks.operational import FitCallback
from src.callbacks.score_dump import ScoreDumpCallback, aux_inputs
from src.callbacks.sds import SDSCallback, sds_inputs
from src.callbacks.thres_drift import ThresholdDriftCallback
from src.callbacks.wasserstein import WassersteinCallback
from src.evaluation.callbacks.cap_bank import CAPBankCallback as EvaluationBank
from src.evaluation.callbacks.components.cap.binary.pairing import nn_coupling
from src.evaluation.callbacks.wasserstein import _wasserstein_1d
from src.evaluation.components.rebuild import reads_hub, rebuild
from src.evaluation.components.tables import (
    PARTIAL_RUNS,
    load_lattice,
    load_scores,
    write_partial,
)
from src.evaluation.dump import RunEvaluation, _offline_hub
from src.evaluation.lattice import select_lattice
from src.evaluation.replay import ReplayModule
from tests.test_eval_b9 import _assert_float32, _checkpoint, _history, _trained
from tests.test_eval_stage import (
    TOY_RUN_CONFIG,
    TOY_SETS,
    Capture,
    _cifar10_evaluator,
    _compose,
    _read_csv,
    _test,
    drop_absent_sets,
)
from tests.test_physics_models import FEATURE_MAP, _deep_sets
from tests.test_score_dump import (
    MaskedToyAE,
    ToyDataModule,
    dump_callback,
    fit,
    load,
    masked_sets,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/cap_beta_fix.py"
SN = "SingleNeutrino_E-10-gun"
EPOCHS = 10
LATTICE, RUN = "toy", "run_a"
# _compose names every lattice x, which the score writer's directory takes.
WRITTEN = "x"
# The physics rates, so that the tiers of configs/eval/physics.yaml apply: the module's
# 0.25 kHz and the q = 0.99 tier.
RATES = [0.25, 286.088064]
MODULE = {"target_rate": 0.25, "base_rate": 28608.8064, "lr": 0.3}
TIER_KEY = "val/{}/eff__ascore_full__brate_{}"
AUX = {"aux_seed": 7, "aux_size": 50}
BATCH = 16
ENTROPY_ROWS = 100
# ot needs N2 >= N1, and normal (120 rows) outnumbers the gun (80).
BANK_ROWS = 20
TOY_BETA_FIX = {"std": 1.0, "rank": 3.0, "sq": 4.0, "disp": 2.0}
CAPS = {"seed": 0, "val": {"normal": 50, SN: 60}, "test": {"normal": 60}}


def physics_sets(seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)

    def block(n, shift, label):
        x = torch.randn(n, 4, generator=gen) + shift
        return x, torch.full((n,), label, dtype=torch.long)

    sets = {"train": block(200, 0.0, 0)}
    for split in ("val", "test"):
        sets[split] = {
            "normal": block(120, 0.0, 0),
            SN: block(80, 0.5, -1),
            "sig_a": block(40, 2.0, 1),
            "sig_b": block(30, -3.0, 2),
        }
    return masked_sets(sets)


class PhysicsDataModule(ToyDataModule):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.sets, self.calls = physics_sets(seed), []

    def prepare_data(self):
        self.calls.append("prepare_data")

    def setup(self, stage=None):
        self.calls.append(f"setup:{stage}")
        super().setup(stage)


def _module() -> MaskedToyAE:
    return MaskedToyAE(**MODULE)


def _training_callbacks() -> list:
    aux = {**AUX, "batch_size": BATCH}
    pair = {"output_name": "ascore/full", "dataset_1": "normal", "dataset_2": SN}
    sds = {"max_rows": 10000, "batch_size": BATCH, "target_rate": RATES[1]}
    drift = {"calibration_fraction": 0.5, "split_seed": 12345}
    return [
        AnomalyEfficiencyCallback(output_name="ascore/full", target_rates=RATES),
        ThresholdDriftCallback(output_name="ascore/full", target_rates=RATES, **drift),
        WassersteinCallback(**pair, apply_log1p=True),
        NPDCallback(**aux),
        RTMCallback(),
        EAGCallback(),
        LossEntropyCallback(max_rows=ENTROPY_ROWS, batch_size=BATCH),
        SDSCallback(source="train", **sds),
        SDSCallback(name="sds_val", source="normal", **sds),
        FitCallback(),
    ]


def _dump(path: Path) -> ScoreDumpCallback:
    # The physics form: probes and weights at every epoch, normal and the gun at the
    # sanity pass alone, no training pass.
    return ScoreDumpCallback(
        dirpath=str(path),
        run_dir=str(path.parent),
        keys=["ascore/full", "fit/full"],
        stored_sets=[],
        initial_sets=["normal", SN],
        train_pass=False,
        aux_batch_size=BATCH,
        **AUX,
    )


def _stage_config(root: Path, row_caps: dict | None = None):
    cfg = _compose("physics", f"rescored_dir={root / 'rescored'}")
    with open_dict(cfg):
        if row_caps is not None:
            cfg.evaluation.evaluator.row_caps = row_caps
        cfg.cap_bank.beta_fix = TOY_BETA_FIX
        cfg.evaluation.evaluator.dump_callbacks.cap_bank_sn.max_rows_1 = BANK_ROWS
        train = cfg.evaluation.evaluator.extra_sets.train
        train.max_rows, train.batch_size = ENTROPY_ROWS, BATCH
    return cfg


def _train(root: Path, callbacks: list, uri: str) -> str:
    logger = MLFlowLogger(experiment_name=LATTICE, run_name=RUN, tracking_uri=uri)
    pl.seed_everything(0)
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        devices=1,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        default_root_dir=root,
        num_sanity_val_steps=-1,
    )
    trainer.fit(_module(), datamodule=PhysicsDataModule())
    return logger.run_id


def _replay(root: Path, dump: Path, out: Path, uri: str | None, row_caps=None):
    cfg = _stage_config(root, row_caps)
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    datamodule = PhysicsDataModule()
    run = evaluator.evaluate_dump(dump, _module(), datamodule, out, LATTICE, uri)
    return cfg, run, datamodule


@pytest.fixture(scope="module")
def physics(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("physics")
    uri, dump = f"file:{root / 'mlruns'}", root / "scores" / LATTICE / RUN
    selection = OmegaConf.to_container(_stage_config(root).selection, resolve=True)
    strategies = _trained(selection)
    checkpoints = {
        name: _checkpoint(root, spec["key"], spec["mode"])
        for name, spec in strategies.items()
    }
    callbacks = [_dump(dump), *_training_callbacks(), *checkpoints.values()]
    train_id = _train(root, callbacks, uri)
    _, run, datamodule = _replay(root, dump, root / "eval", uri)
    client = MlflowClient(uri)
    return SimpleNamespace(
        root=root,
        dump=dump,
        selection=selection,
        strategies=strategies,
        signals=run["signals"],
        calls=datamodule.calls,
        train=lambda key: _history(client, train_id, key),
        stage=lambda key: _history(client, run["mlflow_run_id"], key),
        val=_read_csv(root / "eval" / "runs" / RUN / "val.csv"),
        picks=_read_csv(root / "eval" / "runs" / RUN / "picks.csv"),
        written=root / "rescored" / WRITTEN / RUN,
        kept={
            name: torch.load(ckpt.best_model_path, weights_only=False)["epoch"]
            for name, ckpt in checkpoints.items()
        },
    )


def _tier_pairs(physics) -> list[tuple[str, str]]:
    # The per-set efficiencies at both tiers, signals and the gun, as training logs them.
    pairs = []
    for name in [*physics.signals, SN]:
        pairs.append(
            (TIER_KEY.format(name, "286_088064kHz"), f"eff/eff_286.088064_{name}")
        )
        pairs.append(
            (TIER_KEY.format(name, "operational"), f"eff/eff_operational_{name}")
        )
    return pairs


def _strategy_pairs(physics) -> list[tuple[str, str]]:
    pairs = []
    for name, spec in physics.strategies.items():
        pairs.append((spec["key"], f"ema/{name}"))
        if "raw_key" in spec:
            pairs.append((spec["raw_key"], spec["column"]))
    return pairs


def test_the_physics_dump_holds_normal_at_the_sanity_pass_alone(physics):
    assert set(load(physics.dump / "initial.pt")["val"]) == {"normal", SN}
    assert load(physics.dump / "epoch_0000.pt")["val"] == {}
    assert physics.calls[:3] == ["prepare_data", "setup:fit", "setup:test"]


def test_stage_reproduces_the_logged_values(physics):
    assert [row["epoch"] for row in physics.val] == [str(e) for e in range(-1, EPOCHS)]
    for key, column in _strategy_pairs(physics) + _tier_pairs(physics):
        steps, logged = physics.train(key)
        assert len(steps) == EPOCHS, key
        _assert_float32(logged, [float(row[column]) for row in physics.val[1:]], 0)
    # The stage logs the strategies and the per-set efficiencies of the working point.
    keys = [key for key, _ in _strategy_pairs(physics)]
    keys += [physics.selection["efficiency_key"].format(s) for s in physics.signals]
    for key in keys:
        stage_steps, stage_logged = physics.stage(key)
        assert stage_steps == list(range(EPOCHS)), key
        _assert_float32(physics.train(key)[1], stage_logged, 0)


def test_the_operational_drift_is_a_descriptor_beside_the_tier(physics):
    drift = [float(row["thres_drift/drift_operational"]) for row in physics.val[1:]]
    _assert_float32(
        physics.train("val/summary/operational_drift_ema")[1], ema(drift), 0
    )


def test_picks_are_the_kept_checkpoints(physics):
    for name in physics.strategies:
        k = str(W_STEPS) if name in physics.selection["criteria"] else ""
        rows = [r for r in physics.picks if r["strategy"] == name and r["k"] == k]
        assert [int(row["epoch"]) for row in rows] == [physics.kept[name]], name
    assert set(physics.kept.values()) - {0, EPOCHS - 1}


def test_the_replay_sets_the_cut_of_every_tier(physics, tmp_path):
    record = load(physics.dump / "initial.pt")
    replay = ReplayModule(_module(), load(physics.dump / "meta.pt"), rates=RATES)
    replay.load_epoch(record, tmp_path / "initial.pt")
    for suffix, rate in (("operational", 0.25), ("286_088064kHz", RATES[1])):
        expected = AnomalyRate(rate, MODULE["base_rate"])
        expected.set_threshold(record["val"]["normal"]["ascore/full"])
        assert torch.equal(getattr(replay, f"thres_{suffix}"), expected.threshold)
        assert torch.equal(getattr(replay, f"tie_{suffix}"), expected.tie_weight)
    replay.load_epoch(load(physics.dump / "epoch_0000.pt"), tmp_path / "epoch_0000.pt")
    assert replay.thres_operational is None and replay.thres_286_088064kHz is None


def test_probes_and_pseudo_points_take_the_training_masks(physics, tmp_path):
    cfg = _stage_config(tmp_path)
    settings = hydra.utils.instantiate(cfg.evaluation.evaluator)._dump_settings()
    run = RunEvaluation(physics.dump, _module(), PhysicsDataModule(), **settings)
    meta, sets = load(physics.dump / "meta.pt"), physics_sets()
    loaders = run.loaders["val"]
    probes = drawn_masks(sets["train"][1], AUX["aux_size"], AUX["aux_seed"])
    for name, x in aux_inputs(meta).items():
        assert torch.equal(loaders[name].dataset.tensors[1], probes)
    pseudo = {"sds": sds_inputs(sets["train"][0], 10000, 0, sets["train"][1])}
    normal = sets["val"]["normal"]
    pseudo["sds_val"] = sds_inputs(normal[0], 10000, 0, normal[1])
    for prefix, inputs in pseudo.items():
        for name in ("targets", "outliers"):
            x, mask = loaders[f"{prefix}_{name}"].dataset.tensors[:2]
            assert torch.equal(mask, inputs[f"sds_{name}_mask"])
            assert torch.equal(x, torch.where(mask, inputs[f"sds_{name}"], 0.0))
    assert len(loaders["train"].dataset) == ENTROPY_ROWS


def test_the_bank_reads_capped_masked_inputs(physics):
    row = physics.val[1]
    for key in ("cdf_std_A-fix", "ot_std_D-fix", "nn_rank_D-fix", "ot_disp_D-fix"):
        assert np.isfinite(float(row[f"cap_bank_sn/{key}"])), key
    labels = [
        json.loads(p.read_text()) for p in (physics.root / "eval").rglob("ot_*.json")
    ]
    assert len(labels) == 1
    assert (labels[0]["n_1"], labels[0]["n_2"]) == (BANK_ROWS, 80)


def _load_script():
    spec = importlib.util.spec_from_file_location("cap_beta_fix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_written_scores_read_back_as_a_scores_only_lattice(physics):
    scores = load_scores(physics.written)
    assert list(scores) == list(range(-1, EPOCHS))
    initial = load(physics.dump / "initial.pt")["val"]
    for name in ("normal", SN):
        for key, values in initial[name].items():
            assert torch.equal(torch.from_numpy(scores[-1]["val"][name][key]), values)
    for row in physics.val:
        fits = torch.from_numpy(scores[int(row["epoch"])]["val"]["normal"]["fit/full"])
        assert float(row["fit/normal"]) == fits.double().mean().item()
    meta = load(physics.written / "meta.pt")
    assert meta["rows"] == {"normal": None, SN: None}
    script = _load_script()
    runs = script._runs([physics.written.parent])
    assert runs == [physics.written]
    quantiles = script.quantiles(runs, OmegaConf.load(CONFIGS / "cap/bank.yaml").maps)
    assert all(np.isfinite(q).all() for q in quantiles.values())


@pytest.fixture(scope="module")
def capped(physics) -> SimpleNamespace:
    root = physics.root / "capped"
    _replay(root, physics.dump, root / "eval", None, CAPS)
    written, out = root / "rescored" / WRITTEN / RUN, root / "eval"
    return SimpleNamespace(
        val=_read_csv(out / "runs" / RUN / "val.csv"),
        scores=load_scores(written),
        meta=load(written / "meta.pt"),
    )


def test_row_caps_keep_the_same_seeded_rows_at_every_epoch(physics, capped):
    rows = {
        name: row_subset(n, CAPS["val"][name], 0)
        for name, n in [("normal", 120), (SN, 80)]
    }
    assert all(torch.equal(capped.meta["rows"][n], rows[n]) for n in rows)
    full = load_scores(physics.written)
    initial = load(physics.dump / "initial.pt")["val"]
    for epoch, scores in capped.scores.items():
        for name, kept in rows.items():
            served = torch.from_numpy(scores["val"][name]["ascore/full"])
            if epoch < 0:
                assert torch.equal(served, initial[name]["ascore/full"][kept])
            expected = torch.from_numpy(full[epoch]["val"][name]["ascore/full"])[kept]
            torch.testing.assert_close(served, expected)


def test_capped_sets_give_the_criteria_of_their_rows(capped):
    log1p = lambda x: torch.sign(x) * torch.log1p(torch.abs(x))  # noqa: E731
    for row in capped.val:
        val = capped.scores[int(row["epoch"])]["val"]
        fits = torch.from_numpy(val["normal"]["fit/full"]).double()
        assert float(row["fit/normal"]) == fits.mean().item()
        normal, gun = (torch.from_numpy(val[n]["ascore/full"]) for n in ("normal", SN))
        w1 = _wasserstein_1d(log1p(normal), log1p(gun))
        assert float(row["wasserstein/w1dist"]) == pytest.approx(w1, rel=1e-6)


def test_the_replay_injects_the_feature_map_of_the_l1ad_loaders(tmp_path):
    decoder = Decoder(nodes=[16, 32], out_dim=9)
    module = DeepSetsAE(_deep_sets(rho_final_activation=False), decoder, base_rate=None)
    gen = torch.Generator().manual_seed(0)

    def loader(n: int, label: int, feature_map: dict | None) -> DataLoader:
        x, mask = torch.randn(n, 9, generator=gen), torch.ones(n, 9)
        mask[:, FEATURE_MAP["FET"]["eta"]] = 0.0
        y = torch.full((n,), label)
        dataset = TensorDataset(x, mask, torch.zeros(n, dtype=torch.bool), y)
        if feature_map is not None:
            dataset.object_feature_map = feature_map
        return DataLoader(dataset, batch_size=16)

    meta = {"target_rate": 0.25, "base_rate": None, "aux_lower": None}
    meta.update(aux_upper=None, val={"batch_sizes": {}}, aux={"batch_sizes": {}})
    record = {"epoch": 0, "val": {}, "aux": {}, "state_dict": module.state_dict()}
    for feature_map in (None, FEATURE_MAP):
        replay = ReplayModule(module, meta)
        replay.load_epoch(record, tmp_path / "epoch_0000.pt")
        loaders = {"normal": loader(40, 0, feature_map), "sig_a": loader(20, 1, None)}
        if feature_map is None:
            with pytest.raises(RuntimeError, match="object_feature_map not found"):
                _test(replay, "val", loaders, [])
            continue
        capture = Capture()
        _test(replay, "val", loaders, [capture])
    assert module.object_feature_map == FEATURE_MAP
    module.eval()
    with torch.no_grad():
        expected = [module.model_step(b)["ascore/full"] for b in loaders["sig_a"]]
    served = [out["ascore/full"] for out in capture.outputs["sig_a"]]
    assert torch.equal(torch.cat(served), torch.cat(expected))


def test_rebuild_reads_the_cache_dir_of_the_hub_datamodule(tmp_path):
    hub = {"_target_": "src.data.L1AD_HF_datamodule.L1ADHFDataModule", "cache_dir": "a"}
    config = {"data": hub, "algorithm": {"_target_": "tests.test_score_dump.ToyAE"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    assert reads_hub(tmp_path)
    datamodule = rebuild(tmp_path, str(tmp_path / "cache"))[1]
    assert datamodule.hparams.cache_dir == str(tmp_path / "cache")
    with pytest.raises(ValueError, match="sets cannot replace"):
        rebuild(tmp_path, None, "cifar10")
    raw = {**config, "data": {"_target_": "src.data.L1AD_datamodule.L1ADDataModule"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    assert not reads_hub(tmp_path)
    with pytest.raises(ValueError, match="data_dir cannot replace"):
        rebuild(tmp_path, str(tmp_path / "cache"))


def test_the_bank_callbacks_cap_and_mask_alike():
    args = ("ascore/full", "normal", SN, "normal", {"pairings": {}, "maps": {}})
    kwargs = {"max_rows_1": 4, "max_rows_2": None, "row_seed": 3}
    banks = [Bank(*args, **kwargs) for Bank in (TrainingBank, EvaluationBank)]
    values = torch.arange(10.0)
    for bank in banks:
        assert torch.equal(bank._capped("normal", values), values[row_subset(10, 4, 3)])
        assert torch.equal(bank._capped(SN, values), values)
    x, mask = torch.ones(2, 3), torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.bool)
    for module in ("src.callbacks.cap_bank", "src.evaluation.callbacks.cap_bank"):
        inputs = importlib.import_module(module)._inputs(
            (x, mask, None, torch.zeros(2))
        )
        assert torch.equal(inputs, mask.float())
        assert torch.equal(importlib.import_module(module)._inputs((x, None)), x)


def test_a_reference_that_is_x1_takes_other_rows_of_it():
    args = ("ascore/full", "normal", SN, "normal", {"pairings": {}, "maps": {}})
    raw = {"normal": torch.arange(10.0)[:, None], SN: torch.zeros(6, 1)}
    for Bank in (TrainingBank, EvaluationBank):
        bank = Bank(*args, max_rows_1=4, row_seed=3)
        inputs = {name: bank._capped(name, x) for name, x in raw.items()}
        reference = bank._reference_inputs(raw, inputs)
        assert len(reference) == 4
        kept = set(inputs["normal"].flatten().tolist())
        assert kept.isdisjoint(reference.flatten().tolist())
        with pytest.raises(ValueError, match="disjoint"):
            Bank(*args, max_rows_1=6)._reference_inputs(raw, inputs)


def test_the_nn_caliper_is_positive_with_x1_as_reference():
    generator = torch.Generator().manual_seed(0)
    raw = {"normal": torch.randn(400, 3, generator=generator)}
    raw[SN] = torch.randn(400, 3, generator=generator)
    bank = EvaluationBank(
        "ascore/full", "normal", SN, "normal", {}, max_rows_1=100, max_rows_2=200
    )
    inputs = {name: bank._capped(name, x) for name, x in raw.items()}
    coupling = nn_coupling(
        inputs["normal"], inputs[SN], bank._reference_inputs(raw, inputs)
    )
    assert coupling.caliper_distance > 0
    assert len(coupling.index_1) > 0


def _toy_run(root: Path, run: str, seed: int, max_epochs: int) -> None:
    path = root / LATTICE / run
    fit(root, [dump_callback(path)], epochs=2, seed=seed)
    config = {**TOY_RUN_CONFIG, "trainer": {"max_epochs": max_epochs}}
    (path / "config.yaml").write_text(yaml.safe_dump(config))


def _selection(**changes) -> dict:
    # The registry of _cifar10_evaluator's toy lattices.
    cfg = _compose("cifar10")
    drop_absent_sets(cfg, TOY_SETS)
    return {**OmegaConf.to_container(cfg.selection, resolve=True), **changes}


@pytest.fixture(scope="module")
def lattice(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("partial")
    scores = root / "scores"
    for seed, (run, max_epochs) in enumerate([("a", 2), ("b", 2), ("c", 3)]):
        _toy_run(scores, f"run_{run}", seed, max_epochs)
    return SimpleNamespace(root=root, scores=scores)


def test_partial_runs_are_kept_unless_a_mode_leaves_them_out(lattice):
    evaluator = _cifar10_evaluator()
    kept = lattice.root / "kept"
    runs = evaluator.evaluate_lattice(lattice.scores, kept, LATTICE)
    assert [run["run"] for run in runs] == ["run_a", "run_b", "run_c"]
    assert not (kept / PARTIAL_RUNS).exists()

    out = lattice.root / "eval"
    runs = evaluator.evaluate_lattice(lattice.scores, out, LATTICE, partial="skip")
    assert [run["run"] for run in runs] == ["run_a", "run_b"]
    assert _read_csv(out / PARTIAL_RUNS) == [
        {"run": "run_c", "epochs": "2", "max_epochs": "3"}
    ]
    assert sorted(p.name for p in (out / "runs").iterdir()) == ["run_a", "run_b"]

    replayed = lattice.root / "replayed"
    runs = evaluator.evaluate_lattice(
        lattice.scores, replayed, LATTICE, partial="replay"
    )
    assert [run["run"] for run in runs] == ["run_a", "run_b", "run_c"]
    select_lattice(replayed, _selection(), lattice.scores, LATTICE, None)
    consensus = _read_csv(replayed / "consensus.csv")
    assert {row["run"] for row in consensus} == {"run_a", "run_b"}
    with pytest.raises(ValueError, match="partial_runs"):
        evaluator.evaluate_lattice(lattice.scores, out, LATTICE, partial="drop")


def test_the_selection_reports_a_lattice_whose_runs_are_all_partial(tmp_path):
    write_partial(tmp_path, [["run_c", 2, 3]])
    with pytest.raises(ValueError, match="No run of"):
        select_lattice(tmp_path, _selection(), tmp_path, LATTICE, None)


def test_lattice_criteria_read_a_row_subset_or_are_left_out(lattice):
    names, rows = ["run_a", "run_b"], {"max_rows": 20, "row_seed": 1}
    lattice_dir = lattice.scores / LATTICE
    full, _ = load_lattice(lattice_dir, names)
    subset, metas = load_lattice(lattice_dir, names, lattice_dir, rows)
    kept = row_subset(full.shape[-1], 20, 1).numpy()
    assert np.array_equal(subset, full[..., kept]) and len(metas) == 2
    absent, _ = load_lattice(lattice_dir, names, lattice.root / "nowhere", rows)
    assert absent is None

    out = lattice.root / "selected"
    _cifar10_evaluator().evaluate_lattice(lattice.scores, out, LATTICE)
    selection = _selection(lattice_rows=rows)
    select_lattice(out, selection, lattice.scores, LATTICE, None)
    assert all(np.isfinite(float(r["mc"])) for r in _read_csv(out / "consensus.csv"))
    select_lattice(out, selection, lattice.scores, LATTICE, None, False, out / "none")
    assert all(r["mc"] == "" for r in _read_csv(out / "consensus.csv"))


def test_physics_config_keys_are_those_training_monitors(tmp_path):
    cfg = _compose("physics")
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    names = {
        hydra.utils.instantiate(config).name
        for name, config in evaluator.dump_callbacks.items()
    }
    selection = evaluator.selection
    columns = [spec["column"] for spec in selection["criteria"].values()]
    columns += [selection["q_sec"]["column"], selection["efficiency"]]
    assert {column.split("/")[0] for column in columns} <= names
    experiment = yaml.safe_load((CONFIGS / "experiment/physics/ae.yaml").read_text())
    monitors = {c.get("monitor") for c in experiment["callbacks"].values() if c}
    specs = [*selection["criteria"].values(), selection["q_sec"]]
    specs += [selection["labelled"][name] for name in ("cvar25", "cvar10")]
    assert {spec["key"] for spec in specs} <= monitors


def test_workers_of_a_hub_record_are_spawned_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with _offline_hub(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ
    with _offline_hub(False):
        assert "HF_HUB_OFFLINE" not in os.environ
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with _offline_hub(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "0"
