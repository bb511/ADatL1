import math
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from src.evaluation.components.tables import load_lattice
from src.evaluation.dump import RunEvaluation
from src.evaluation.lattice import select_lattice
from tests.test_eval_stage import (
    TOY_RUN_CONFIG,
    TOY_SETS,
    Capture,
    _compose,
    _read_csv,
    _test,
    drop_absent_sets,
    toy_ancap,
    toy_bank,
)
from tests.test_score_dump import (
    PASSES,
    ToyAE,
    ToyDataModule,
    assert_identical,
    dump_callback,
    fit,
    load,
)

# _compose names every lattice x, which the score writer's directory takes.
LATTICE = "x"
RUNS = ["run_a", "run_b"]
SETS = ["normal", "reference_normal", "sig_a", "sig_b"]
# The keys of the dijet and sbc dumps.
KEYS = ["ascore/full", "fit/full"]
NULL_RESCORED = [
    "rescored_dir=null",
    "~evaluation.evaluator.dump_callbacks.score_writer",
]


@pytest.fixture(scope="module")
def lattices(tmp_path_factory) -> SimpleNamespace:
    # Seeded alike, so that both lattices hold the same weights at every epoch. No
    # training pass, as in the dijet and sbc dumps: the evaluation rescores it.
    root = tmp_path_factory.mktemp("nostore")
    for name, stored_sets in (("stored", None), ("nostore", [])):
        for seed, run in enumerate(RUNS):
            path = root / name / LATTICE / run
            callback = dump_callback(path, stored_sets=stored_sets, train_pass=False)
            fit(root, [callback], seed=seed)
            (path / "config.yaml").write_text(yaml.safe_dump(TOY_RUN_CONFIG))
    return SimpleNamespace(stored=root / "stored", nostore=root / "nostore")


def _toy_config(domain: str, *overrides: str):
    # The toy has neither the simulation nor the second sources: their pairings go, and
    # ANCAP and the bank take the toy's sets.
    cfg = _compose(domain, *overrides)
    drop_absent_sets(cfg, TOY_SETS)
    toy_ancap(cfg)
    toy_bank(cfg)
    return cfg


def _evaluate_and_select(
    cfg, scores_dir: Path, out: Path, rescored_dir: Path | None = None
) -> None:
    # The two steps of src/eval.py.
    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)
    evaluator.evaluate_lattice(scores_dir, out, LATTICE)
    selection = OmegaConf.to_container(cfg.selection, resolve=True)
    select_lattice(out, selection, scores_dir, LATTICE, None, False, rescored_dir)


def test_a_nostore_dump_holds_the_weights_of_a_full_one_and_no_set(lattices):
    for run in RUNS:
        meta = load(lattices.nostore / LATTICE / run / "meta.pt")
        assert meta["stored"] == {"val": [], "test": []}
        assert meta["val"]["keys"] == meta["test"]["keys"] == []
        assert meta["aux"]["keys"] == ["ascore/full", "ascore/mse", "fit/full"]
        for name in PASSES:
            full = load(lattices.stored / LATTICE / run / name)
            bare = load(lattices.nostore / LATTICE / run / name)
            assert bare["val"] == bare["test"] == {}
            assert_identical(bare["state_dict"], full["state_dict"])
            assert_identical(bare["aux"], full["aux"])


def test_the_replay_rescores_every_set_as_a_full_dump_stores_it(lattices, tmp_path):
    cfg = _toy_config("dijet", *NULL_RESCORED)
    settings = hydra.utils.instantiate(cfg.evaluation.evaluator)._dump_settings()
    run_dir = lattices.nostore / LATTICE / RUNS[0]
    run = RunEvaluation(run_dir, ToyAE(), ToyDataModule(), **settings)
    for name in PASSES:
        run.replay.load_epoch(load(run_dir / name), tmp_path / name)
        run._score_normal()
        stored = load(lattices.stored / LATTICE / RUNS[0] / name)
        for split in ("val", "test"):
            capture, loaders = Capture(), {s: run.loaders[split][s] for s in SETS}
            _test(run.replay, split, loaders, [capture])
            for s in SETS:
                for key in KEYS:
                    served = torch.cat([out[key] for out in capture.outputs[s]])
                    torch.testing.assert_close(served, stored[split][s][key])


def test_a_nostore_lattice_is_selected_without_its_normal_scores(
    lattices, tmp_path, caplog
):
    _evaluate_and_select(_toy_config("sbc", *NULL_RESCORED), lattices.nostore, tmp_path)
    consensus = _read_csv(tmp_path / "consensus.csv")
    assert {row["run"] for row in consensus} == set(RUNS)
    assert all(row["mc"] == "" for row in consensus)
    strategies = {row["strategy"] for row in _read_csv(tmp_path / "selection.csv")}
    assert "oracle" in strategies and not {"udr", "mc"} & strategies
    assert "lacks validation normal" in caplog.text


def test_the_written_normal_gives_the_lattice_criteria_of_a_full_dump(
    lattices, tmp_path
):
    rescored, out = tmp_path / "rescored", tmp_path / "eval"
    cfg = _toy_config("dijet", f"rescored_dir={rescored}")
    _evaluate_and_select(cfg, lattices.nostore, out, rescored)
    written, _ = load_lattice(lattices.nostore / LATTICE, RUNS, rescored / LATTICE)
    stored, _ = load_lattice(lattices.stored / LATTICE, RUNS)
    torch.testing.assert_close(torch.from_numpy(written), torch.from_numpy(stored))
    consensus = _read_csv(out / "consensus.csv")
    assert all(math.isfinite(float(row["mc"])) for row in consensus)
