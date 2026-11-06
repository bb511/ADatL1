import csv
import math
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import yaml
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks import threshold_posterior as tp
from src.callbacks.components.selection import smoothed
from src.callbacks.components.utils import placeholder_loader
from src.callbacks.sds import sds_inputs
from src.callbacks.threshold_posterior import ThresholdPosteriorCallback, ancap
from src.data.utils import unpack_batch
from src.evaluation.callbacks.threshold_posterior import ThresholdPosterior
from src.evaluation.components.tables import load_scores
from src.evaluation.dump import RunEvaluation
from src.evaluation.lattice import select_lattice
from src.evaluation.replay import _load_weights
from src.utils.omegaconf import register_resolvers
from tests.test_eval_b9 import (
    AUX,
    BATCH,
    EPOCHS,
    LATTICE,
    RUN,
    _history,
    _read_csv,
    _stage_config,
    _train,
)
from tests.test_eval_physics import SN, PhysicsDataModule, _dump, _module
from tests.test_eval_stage import (
    SELECTION_TABLES,
    TOY_POSTERIOR,
    TOY_RUN_CONFIG,
    TOY_SETS,
    TOY_VIEWS,
    _compose,
    drop_absent_sets,
)
from tests.test_score_dump import PASSES, ToyAE, ToyDataModule, dump_callback, fit, load

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
POSTERIOR = OmegaConf.to_container(OmegaConf.load(CONFIGS / "cap" / "posterior.yaml"))
TARGET = "src.evaluation.callbacks.threshold_posterior.ThresholdPosterior"
# Fewer pairs and draws than registered, for speed.
SMALL = POSTERIOR | TOY_POSTERIOR
VIEWS = {"weak": "sensor_weak", "mid": "sensor_mid"}
REFERENCE = "sds_outliers"
SHIFTS = {"normal": 0.0, "sensor_weak": 0.4, "sensor_mid": 0.8, REFERENCE: 2.5}
TUNNELS = {
    "cchamber": {
        "weak": "sensor_weak",
        "mid": "sensor_mid",
        "strong": "sensor_strong",
    },
    "cchamber_wt": {"osr": "sensor_osr"},
}
IMAGES = {
    "cifar10": {name: name for name in ("sensor_weak", "sensor_mid", "sensor_strong")},
    "mscct": {name: name for name in ("nz20", "nz210", "p1000", "gt450")},
}
DOMAINS = TUNNELS | IMAGES
# As extra_sets.sds of configs/eval/mscct.yaml.
SDS = {"max_rows": 10000, "row_seed": 0}
RUNS = ["run_a", "run_b"]
# The callbacks of the reduced pass: the tables need the efficiency and the fit.
REDUCED_KEEP = ("anomaly_efficiency", "ancap", "fit")
# The cheapest evaluation of eval=mscct that stores the reference's scores: the
# efficiency and the fit stay, for the per-run tables, and nothing else is computed.
# rescored_dir and lattice_rows make the selection find no normal in the writer's files,
# so that it leaves UDR and MC out without reading the dumps again.
REFERENCE_ONLY = [
    "tracking_uri=null",
    "+selection.lattice_rows={max_rows:1,row_seed:0}",
    "evaluation.evaluator.extra_sets.aux=false",
    "evaluation.evaluator.extra_sets.train=false",
    "+evaluation.evaluator.row_caps={seed:0,val:{sds_targets:1}}",
    "+evaluation.evaluator.dump_callbacks.score_writer={"
    "_target_:src.evaluation.callbacks.score_writer.ScoreWriter,"
    "dirpath:'${rescored_dir}/${lattice}',sets:[sds_outliers],keys:[ascore/full]}",
    *(
        f"~evaluation.evaluator.dump_callbacks.{name}"
        for name in (
            "thres_drift wasserstein cap_bank_reference_normal cap_bank_nz20 "
            "cap_bank_nz210 cap_bank_p1000 cap_bank_gt450 ancap ascore_operational "
            "npd rtm eag entropy sds"
        ).split()
    ),
    *(f"~selection.criteria.{n}" for n in "drift w1 npd rtm eag entropy sds".split()),
]
# The reference of each ANCAP callback of eval=physics.
PHYSICS_REFERENCES = {"ancap": REFERENCE, "ancap_val": "sds_val_outliers"}
PHYSICS_VIEWS = {"sn": SN}
RESCORED_VIEWS = f"+evaluation.evaluator.rescored_sets=[normal,{SN}]"
# The ANCAP pass of eval=physics: the efficiency stays, since the tables index its
# columns, and each pseudo-target set is cut to one row, since nothing reads it.
PHYSICS_REDUCED = [
    "tracking_uri=null",
    "evaluation.evaluator.extra_sets.aux=false",
    "evaluation.evaluator.extra_sets.train=false",
    "+evaluation.evaluator.row_caps.val.sds_targets=1",
    "+evaluation.evaluator.row_caps.val.sds_val_targets=1",
    "evaluation.evaluator.dump_callbacks.score_writer.keys=[ascore/full]",
    "evaluation.evaluator.dump_callbacks.score_writer.sets="
    f"[normal,{SN},sds_outliers,sds_val_outliers]",
    *(
        f"~evaluation.evaluator.dump_callbacks.{name}"
        for name in (
            "thres_drift wasserstein cap_bank_sn ascore_operational "
            "npd rtm eag entropy sds sds_val"
        ).split()
    ),
    *(
        f"~selection.criteria.{name}"
        for name in "drift w1 npd rtm eag entropy sds sds_val".split()
    ),
]
# Stands for the 50,000 rows per view of the LHC pass; the toy holds 120 and 80.
PHYSICS_CAPS = {"normal": 50, SN: 60}
PHYSICS_PASSES = {
    "stored": [],
    "rescored": [RESCORED_VIEWS],
    "capped": [
        RESCORED_VIEWS,
        *(
            f"evaluation.evaluator.row_caps.val.{k}={n}"
            for k, n in PHYSICS_CAPS.items()
        ),
    ],
}


@pytest.fixture(scope="module", autouse=True)
def _resolvers() -> None:
    # A second registration raises.
    if not OmegaConf.has_resolver("short_hash"):
        register_resolvers()


class Scorer(pl.LightningModule):
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return {"ascore/full": x.sum(dim=1)}


def _loaders(names) -> dict[str, DataLoader]:
    g = torch.Generator().manual_seed(0)
    loaders = {}
    for name in names:
        n = {"normal": 60, "sensor_weak": 90}.get(name, 40)
        x = torch.randn(n, 3, generator=g) + SHIFTS[name]
        y = torch.zeros(n, dtype=torch.long)
        loaders[name] = DataLoader(TensorDataset(x, y), batch_size=16)
    return loaders


def _evaluate(callback, loaders) -> dict:
    trainer = pl.Trainer(
        logger=False,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[callback],
    )
    trainer.metric_records = {}
    module = Scorer()
    module._ckpt_path = "epoch_0003.pt"
    trainer.test(module, dataloaders=loaders, verbose=False)
    return trainer.metric_records


def _scores(loader) -> torch.Tensor:
    return torch.cat([x.sum(dim=1) for x, _ in loader])


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("trained")
    uri, dump = f"file:{root / 'mlruns'}", root / "scores" / LATTICE / RUN
    cfg = _stage_config()
    callback = ThresholdPosteriorCallback(
        "ascore/full",
        "normal",
        TOY_VIEWS,
        OmegaConf.to_container(cfg.cap_posterior),
        batch_size=BATCH,
    )
    callbacks = [dump_callback(dump, aux_batch_size=BATCH, **AUX), callback]
    run_id = _train(root, callbacks, uri)
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    evaluator.evaluate_dump(dump, ToyAE(), ToyDataModule(), root / "eval", LATTICE)
    client = MlflowClient(uri)
    return SimpleNamespace(
        keys=callback.keys,
        train=lambda key: _history(client, run_id, key),
        val=_read_csv(root / "eval" / "runs" / RUN / "val.csv"),
    )


def test_evaluation_records_exactly_ancap_of_the_collected_scores():
    loaders = _loaders(["normal", *VIEWS.values(), REFERENCE])
    callback = ThresholdPosterior("ascore/full", "normal", VIEWS, SMALL, name="x")
    records = _evaluate(callback, loaders)
    scores = {name: _scores(loader) for name, loader in loaders.items()}
    views = {grade: scores[name] for grade, name in VIEWS.items()}
    expected = ancap(scores["normal"], views, scores[REFERENCE], SMALL)
    assert records == {f"x/{key}": value for key, value in expected.items()}
    assert all(math.isfinite(value) for value in records.values())


@pytest.mark.parametrize("missing", ["sensor_mid", REFERENCE, "normal"])
def test_evaluation_names_a_missing_loader(missing):
    sets = {"normal", *VIEWS.values(), REFERENCE} - {missing}
    callback = ThresholdPosterior("ascore/full", "normal", VIEWS, SMALL)
    expected = rf"\['{missing}'\].*Available test dataloaders"
    with pytest.raises(ValueError, match=expected):
        _evaluate(callback, _loaders(sets))


def test_an_empty_reference_leaves_nan_in_the_evaluation():
    loaders = _loaders(["normal", *VIEWS.values(), REFERENCE])
    empty = TensorDataset(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
    loaders[REFERENCE] = DataLoader(empty, batch_size=16)
    records = _evaluate(
        ThresholdPosterior("ascore/full", "normal", VIEWS, SMALL), loaders
    )
    assert len(records) == 5 * len(VIEWS) + 1
    assert all(math.isnan(value) for value in records.values())


def test_a_non_default_reference_is_read():
    loaders = _loaders(["normal", *VIEWS.values(), REFERENCE])
    loaders["other"] = loaders.pop(REFERENCE)
    callback = ThresholdPosterior(
        "ascore/full", "normal", VIEWS, SMALL, reference="other"
    )
    assert math.isfinite(_evaluate(callback, loaders)["ancap/ancap_dir"])


def test_training_builds_the_reference_as_sds_does(tmp_path):
    callback = ThresholdPosteriorCallback(
        "ascore/full", "normal", TOY_VIEWS, SMALL, batch_size=BATCH
    )
    _, datamodule = fit(tmp_path, [callback], epochs=1)
    train = unpack_batch(datamodule.data_train.tensors)
    expected = sds_inputs(train.x, 10000, 0, train.mask)[REFERENCE]
    assert list(callback.loaders) == [REFERENCE]
    assert torch.equal(callback.loaders[REFERENCE].dataset.tensors[0], expected)
    assert math.isfinite(callback.emas["control"])


def test_training_names_a_missing_loader(tmp_path):
    views = {"weak": "sensor_weak"}
    callback = ThresholdPosteriorCallback("ascore/full", "normal", views, SMALL)
    with pytest.raises(ValueError, match=r"\['sensor_weak'\].*Available validation"):
        fit(tmp_path, [callback], epochs=1)


def test_an_empty_reference_leaves_the_training_average_nan(tmp_path, monkeypatch):
    def empty(x, max_rows, row_seed, mask=None):
        return {REFERENCE: x[:0]}

    monkeypatch.setattr(tp, "sds_inputs", empty)
    callback = ThresholdPosteriorCallback("ascore/full", "normal", TOY_VIEWS, SMALL)
    fit(tmp_path, [callback], epochs=1)
    assert math.isnan(callback.emas["control"])


def test_training_logs_the_moving_average_of_the_stage_values(trained):
    assert trained.keys == {"control": "val/summary/ancap_control_ema"}
    steps, logged = trained.train("val/summary/ancap_control_ema")
    assert len(steps) == EPOCHS
    raw = [float(row["ancap/ancap_control"]) for row in trained.val[1:]]
    assert all(math.isfinite(value) for value in raw)
    expected = smoothed(raw, skip_nonfinite=True)
    np.testing.assert_array_equal(np.float32(logged), expected)


def test_stage_writes_every_column(trained):
    suffix = "_control"
    columns = {f"ancap/{p}{suffix}" for p in tp.COLUMNS.values()} | {"ancap/ancap_dir"}
    assert {key for key in trained.val[0] if key.startswith("ancap/")} == columns
    for row in trained.val:
        values = {key: float(row[key]) for key in columns}
        assert all(math.isfinite(value) for value in values.values()), row["epoch"]
        capacity = values["ancap/ancap_control"]
        confidence = values["ancap/ancap_conf_control"]
        assert values["ancap/ancap_cons_control"] == capacity - confidence
        assert values["ancap/ancap_sat_control"] in (0.0, 1.0)


@pytest.mark.parametrize("domain", DOMAINS)
def test_each_domain_composes_one_callback_of_every_grade(domain):
    cfg = _compose(domain)
    assert OmegaConf.to_container(cfg.cap_posterior) == POSTERIOR
    callbacks = cfg.evaluation.evaluator.dump_callbacks
    callbacks = OmegaConf.to_container(callbacks, resolve=True)
    assert [name for name, c in callbacks.items() if c["_target_"] == TARGET] == [
        "ancap"
    ]
    assert not [name for name in callbacks if name.startswith("posterior_")]
    assert callbacks["ancap"] == {
        "_target_": TARGET,
        "output_name": "ascore/full",
        "dataset_1": "normal",
        "views": DOMAINS[domain],
        "posterior_config": POSTERIOR,
        "name": "ancap",
    }
    instance = hydra.utils.instantiate(callbacks["ancap"])
    assert instance.reference == REFERENCE and instance.views == DOMAINS[domain]
    assert cfg.evaluation.evaluator.extra_sets.sds == SDS


@pytest.mark.parametrize("domain", IMAGES)
def test_the_image_views_are_sets_of_the_data_config(domain):
    data = OmegaConf.load(CONFIGS / "data" / f"{domain}.yaml")
    served = list(data.shifts if domain == "cifar10" else data.shift_scanners)
    config = _compose(domain).evaluation.evaluator.dump_callbacks.ancap
    callback = hydra.utils.instantiate(config)
    assert list(callback.views.values()) == served
    loaders = dict.fromkeys(["normal", *served, REFERENCE])
    callback.on_test_epoch_start(SimpleNamespace(test_dataloaders=loaders), None)


@pytest.mark.parametrize("domain", [*DOMAINS, "physics"])
def test_ancap_enters_no_criterion(domain):
    criteria = _compose(domain).selection.criteria
    assert not [n for n in criteria if n.startswith(("posterior", "ancap"))]
    columns = [spec.column for spec in criteria.values()]
    assert not [c for c in columns if c.startswith(("posterior", "ancap"))]


@pytest.fixture(scope="module")
def lattice(tmp_path_factory) -> Path:
    # A dump without the pseudo-sets, whose scores the replay rescores from the weights.
    root = tmp_path_factory.mktemp("reference")
    for seed, run in enumerate(RUNS):
        path = root / "scores" / "toy" / run
        fit(root, [dump_callback(path)], seed=seed)
        (path / "config.yaml").write_text(yaml.safe_dump(TOY_RUN_CONFIG))
    return root / "scores"


def _reference_only(rescored: Path):
    cfg = _compose("mscct", "lattice=toy", f"rescored_dir={rescored}", *REFERENCE_ONLY)
    # The toy has none of the four scanners of null_ds; the callbacks stay as composed.
    drop_absent_sets(cfg, TOY_SETS)
    return cfg


def _module_scores(state_dict: dict, batch_size: int) -> torch.Tensor:
    # The scores of the module on the pseudo-outliers, apart from any replay.
    datamodule = ToyDataModule()
    datamodule.setup("fit")
    train = unpack_batch(datamodule.data_train.tensors)
    points = sds_inputs(train.x, SDS["max_rows"], SDS["row_seed"], train.mask)[
        REFERENCE
    ]
    module = ToyAE()
    _load_weights(module, state_dict)
    module.eval()
    with torch.no_grad():
        loader = placeholder_loader(points, batch_size)
        return torch.cat([module.model_step(batch)["ascore/full"] for batch in loader])


def test_a_replay_stores_the_reference_scores_of_every_epoch(lattice, tmp_path):
    rescored, out = tmp_path / "rescored", tmp_path / "eval"
    cfg = _reference_only(rescored)
    hydra.utils.instantiate(cfg.evaluation.evaluator).evaluate_lattice(
        lattice, out, "toy"
    )
    for run in RUNS:
        written = load_scores(rescored / "toy" / run)
        assert list(written) == [-1, 0, 1]
        meta = load(rescored / "toy" / run / "meta.pt")
        assert meta["stored"] == {"val": [REFERENCE], "test": []}
        assert meta["keys"] == ["ascore/full"] and meta["rows"] == {REFERENCE: None}
        batch_size = load(lattice / "toy" / run / "meta.pt")["aux"]["batch_sizes"]
        for epoch, name in zip(written, PASSES):
            record = load(lattice / "toy" / run / name)
            assert "sds" not in record
            scores = written[epoch]["val"]
            assert list(scores) == [REFERENCE]
            expected = _module_scores(record["state_dict"], batch_size["gaussian"][0])
            assert torch.equal(
                torch.as_tensor(scores[REFERENCE]["ascore/full"]), expected
            )


def test_the_stored_reference_scores_are_those_of_a_training_dump(tmp_path):
    # A twin that stored the pseudo-sets while training, the same seed and weights.
    root = tmp_path / "scores" / "toy"
    fit(tmp_path, [dump_callback(root / "run_a")])
    fit(tmp_path, [dump_callback(tmp_path / "twin", sds=SDS)])
    (root / "run_a" / "config.yaml").write_text(yaml.safe_dump(TOY_RUN_CONFIG))
    cfg = _reference_only(tmp_path / "rescored")
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    evaluator.evaluate_lattice(tmp_path / "scores", tmp_path / "eval", "toy")
    for name in PASSES:
        written = load(tmp_path / "rescored" / "toy" / "run_a" / name)["val"]
        stored = load(tmp_path / "twin" / name)["sds"]
        assert torch.equal(
            written[REFERENCE]["ascore/full"], stored[REFERENCE]["ascore/full"]
        )


def test_the_recipe_leaves_the_efficiency_the_fit_and_the_writer_alone(
    lattice, tmp_path
):
    cfg = _reference_only(tmp_path / "rescored")
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    settings = evaluator._dump_settings()
    assert list(settings["dump_callbacks"]) == [
        "anomaly_efficiency",
        "fit",
        "score_writer",
    ]
    assert settings["selection"]["criteria"] == {}
    run = RunEvaluation(lattice / "toy" / RUNS[0], ToyAE(), ToyDataModule(), **settings)
    sizes = {name: len(loader.dataset) for name, loader in run.loaders["val"].items()}
    assert sizes["sds_targets"] == 1 and sizes[REFERENCE] > 1
    assert not {"gaussian", "uniform", "autouad", "train"} & sizes.keys()


def test_the_recipe_survives_the_tables_and_the_selection(lattice, tmp_path, caplog):
    rescored, out = tmp_path / "rescored", tmp_path / "eval"
    cfg = _reference_only(rescored)
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    evaluator.evaluate_lattice(lattice, out, "toy")
    selection = OmegaConf.to_container(cfg.selection, resolve=True)
    select_lattice(out, selection, lattice, "toy", None, rescored_dir=rescored)
    assert "UDR and MC are left out" in caplog.text
    assert SELECTION_TABLES <= {path.name for path in out.iterdir()}
    with open(out / "runs" / RUNS[0] / "picks.csv", newline="") as file:
        picked = [row["strategy"] for row in csv.DictReader(file)]
    assert picked == ["oracle", "cvar25", "cvar10", "q_sec"]


def _reduced(domain: str) -> list[str]:
    config = OmegaConf.load(CONFIGS / "eval" / f"{domain}.yaml")
    callbacks = config.evaluation.evaluator.dump_callbacks
    return [
        "tracking_uri=null",
        "+selection.lattice_rows={max_rows:1,row_seed:0}",
        "evaluation.evaluator.extra_sets.aux=false",
        "evaluation.evaluator.extra_sets.train=false",
        *(
            f"~evaluation.evaluator.dump_callbacks.{name}"
            for name in callbacks
            if name not in REDUCED_KEEP
        ),
        *(f"~selection.criteria.{name}" for name in config.selection.criteria),
    ]


@pytest.mark.parametrize("domain", IMAGES)
def test_a_scores_only_dump_serves_its_stored_pseudo_sets_to_ancap(domain, tmp_path):
    scores = tmp_path / "scores"
    path = scores / "toy" / RUNS[0]
    fit(tmp_path, [dump_callback(path, sds=SDS)])
    (path / "config.yaml").write_text(yaml.safe_dump(TOY_RUN_CONFIG))
    for name in PASSES:
        record = load(path / name)
        # Doubled, so that a rescored reference would give other columns.
        record["sds"][REFERENCE]["ascore/full"] *= 2.0
        if name != "initial.pt":
            del record["state_dict"]
        torch.save(record, path / name)

    cfg = _compose(domain, "lattice=toy", *_reduced(domain))
    drop_absent_sets(cfg, TOY_SETS)
    cfg.evaluation.evaluator.dump_callbacks.ancap.views = TOY_VIEWS
    cfg.cap_posterior.merge_with(TOY_POSTERIOR)
    out = tmp_path / "eval"
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    evaluator.evaluate_lattice(scores, out, "toy")
    selection = OmegaConf.to_container(cfg.selection, resolve=True)
    select_lattice(out, selection, scores, "toy", None)

    rows = _read_csv(out / "runs" / RUNS[0] / "val.csv")
    assert len(rows) == len(PASSES)
    for row, name in zip(rows, PASSES):
        record = load(path / name)
        normal, control = (
            record["val"][s]["ascore/full"] for s in ("normal", "reference_normal")
        )
        reference = record["sds"][REFERENCE]["ascore/full"]
        expected = ancap(normal, {"control": control}, reference, SMALL)
        for key, value in expected.items():
            assert float(row[f"ancap/{key}"]) == pytest.approx(value, nan_ok=True), key


def test_physics_composes_ancap_beside_its_validation_reference():
    cfg = _compose("physics")
    assert OmegaConf.to_container(cfg.cap_posterior) == POSTERIOR
    callbacks = cfg.evaluation.evaluator.dump_callbacks
    callbacks = OmegaConf.to_container(callbacks, resolve=True)
    names = [name for name, c in callbacks.items() if c["_target_"] == TARGET]
    assert names == list(PHYSICS_REFERENCES)
    for name, reference in PHYSICS_REFERENCES.items():
        instance = hydra.utils.instantiate(callbacks[name])
        assert instance.name == name and instance.dataset_1 == "normal"
        assert instance.views == PHYSICS_VIEWS and instance.reference == reference
        assert instance.config == POSTERIOR


@pytest.fixture(scope="module")
def physics_dump(tmp_path_factory) -> Path:
    # The physics form, its stored scores doubled so that a served file shows.
    root = tmp_path_factory.mktemp("physics")
    path = root / "scores" / "toy" / RUNS[0]
    fit(root, [_dump(path)], module=_module(), datamodule=PhysicsDataModule())
    (path / "config.yaml").write_text(yaml.safe_dump(TOY_RUN_CONFIG))
    record = load(path / "initial.pt")
    for scores in record["val"].values():
        scores["ascore/full"] *= 2.0
    torch.save(record, path / "initial.pt")
    return path


def test_the_physics_pass_serves_the_view_and_both_references(physics_dump):
    evaluator = hydra.utils.instantiate(_compose("physics").evaluation.evaluator)
    settings = evaluator._dump_settings()
    run = RunEvaluation(physics_dump, _module(), PhysicsDataModule(), **settings)
    callbacks = run.trainers["val"].callbacks
    callbacks = [cb for cb in callbacks if isinstance(cb, ThresholdPosterior)]
    assert [cb.name for cb in callbacks] == list(PHYSICS_REFERENCES)
    served = SimpleNamespace(test_dataloaders=run.loaders["val"])
    for callback in callbacks:
        callback.on_test_epoch_start(served, None)


def _physics_pass(dump: Path, root: Path, overrides: list[str]) -> SimpleNamespace:
    written, out = root / "rescored", root / "eval"
    cfg = _compose(
        "physics",
        "lattice=toy",
        f"rescored_dir={written}",
        *PHYSICS_REDUCED,
        *overrides,
    )
    cfg.cap_posterior.merge_with(TOY_POSTERIOR)
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    evaluator.evaluate_dump(dump, _module(), PhysicsDataModule(), out, "toy")
    selection = OmegaConf.to_container(cfg.selection, resolve=True)
    select_lattice(out, selection, dump.parents[1], "toy", None, rescored_dir=written)
    return SimpleNamespace(
        settings=evaluator._dump_settings(),
        out=out,
        scores=load_scores(written / "toy" / dump.name),
        val=_read_csv(out / "runs" / dump.name / "val.csv"),
    )


@pytest.fixture(scope="module")
def physics_passes(physics_dump, tmp_path_factory) -> dict[str, SimpleNamespace]:
    root = tmp_path_factory.mktemp("passes")
    return {
        name: _physics_pass(physics_dump, root / name, overrides)
        for name, overrides in PHYSICS_PASSES.items()
    }


def _written(scores: dict, name: str) -> torch.Tensor:
    return torch.as_tensor(scores["val"][name]["ascore/full"])


@pytest.mark.parametrize("name", PHYSICS_PASSES)
def test_physics_columns_are_ancap_of_the_rows_the_pass_wrote(physics_passes, name):
    run = physics_passes[name]
    assert len(run.val) == len(PASSES)
    for row in run.val:
        scores = run.scores[int(row["epoch"])]
        normal, views = _written(scores, "normal"), {"sn": _written(scores, SN)}
        for callback, reference in PHYSICS_REFERENCES.items():
            expected = ancap(normal, views, _written(scores, reference), SMALL)
            for key, value in expected.items():
                column = float(row[f"{callback}/{key}"])
                assert column == pytest.approx(value, nan_ok=True), key


def test_physics_views_are_rescored_at_the_sanity_pass_on_request(
    physics_dump, physics_passes
):
    initial = load(physics_dump / "initial.pt")["val"]
    for name in ("normal", SN):
        stored = initial[name]["ascore/full"]
        assert torch.equal(_written(physics_passes["stored"].scores[-1], name), stored)
        rescored = _written(physics_passes["rescored"].scores[-1], name)
        torch.testing.assert_close(rescored, stored / 2.0)


def test_physics_caps_reach_the_views_and_leave_the_references_whole(physics_passes):
    capped, whole = physics_passes["capped"], physics_passes["rescored"]
    for epoch, scores in capped.scores.items():
        for name, rows in PHYSICS_CAPS.items():
            assert len(_written(scores, name)) == rows
        for reference in PHYSICS_REFERENCES.values():
            expected = _written(whole.scores[epoch], reference)
            assert torch.equal(_written(scores, reference), expected)


def test_the_physics_pass_keeps_its_callbacks_and_survives_the_selection(
    physics_passes,
):
    run = physics_passes["capped"]
    assert list(run.settings["dump_callbacks"]) == [
        "anomaly_efficiency",
        "ancap",
        "ancap_val",
        "fit",
        "score_writer",
    ]
    assert run.settings["selection"]["criteria"] == {}
    assert SELECTION_TABLES <= {path.name for path in run.out.iterdir()}
