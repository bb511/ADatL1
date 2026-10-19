import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

from src.evaluation.evaluator import Evaluator

# (main, sec) per checkpoint stem. The non-candidate drift_ema and last beat the
# candidate cap_ema on both objectives, so any rule that compares checkpoints
# would report them instead.
VALUES = {"cap_ema": (-0.3, 0.2), "drift_ema": (-0.1, 0.1), "last": (-0.05, 0.05)}
CKPTS = {"summary": {"cap_ema": ["max"], "drift_ema": ["min"]}}


class _Summary(Callback):
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index
        self.summary = {}

    def plot_summary(self, trainer, plot_folder):
        pass

    def clear_crit_summary(self):
        self.summary.clear()


class _Main(_Summary):
    def get_optimized_metric(self):
        ckpt_name = max(self.summary, key=self.summary.get)
        return ckpt_name, self.summary[ckpt_name]


class _Sec(_Summary):
    def get_optimized_metric(self, ckpt_name: str, test_ds: str):
        return ckpt_name, self.summary.get(ckpt_name)


def _metric_config(sec: bool = True):
    cfg = {
        "candidate": {"strategy": "summary", "metric": "cap_ema", "criterion": "max"},
        "main_metric": {
            "callback": {"name": "main", "params": {}},
            "direction": "maximize",
        },
    }
    if sec:
        cfg["sec_metric"] = {
            "callback": {
                "name": "sec",
                "params": {"ckpt_name": None, "test_ds": "normal"},
            },
            "direction": "minimize",
        }
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg


def _evaluator(tmp_path: Path, ckpts: dict, values: dict, sec: bool = True):
    evaluator = Evaluator(
        ckpts=OmegaConf.create(ckpts),
        callbacks=[_Main("main", 0), _Sec("sec", 1)],
        logger=None,
        optimized_metric_config=_metric_config(sec),
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        default_root_dir=str(tmp_path),
    )
    evaluator.values = values
    evaluator.strat_names_seen = []

    def evaluate_ckpt(ckpt_path: Path, model, test_loader):
        evaluator.strat_names_seen.append(evaluator.evaluator.strat_name)
        for cb in evaluator.callbacks:
            cb.summary[ckpt_path.stem] = evaluator.values[ckpt_path.stem][cb.index]

    evaluator.evaluate_ckpt = evaluate_ckpt
    return evaluator


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    for metric, criterion in [("cap_ema", "max"), ("drift_ema", "min")]:
        folder = root / "summary" / metric / criterion
        folder.mkdir(parents=True)
        (folder / f"{metric}.ckpt").write_bytes(b"")
    (root / "last.ckpt").write_bytes(b"")
    return root


@pytest.mark.parametrize(
    "ckpts",
    [
        {"summary": {"drift_ema": ["min"]}},
        {"summary": {"cap_ema": ["min"]}},
        {"last": True},
    ],
)
def test_candidate_missing_from_ckpts_raises(tmp_path, ckpts):
    with pytest.raises(ValueError, match="Candidate"):
        _evaluator(tmp_path, ckpts, VALUES)


def test_config_without_candidate_raises(tmp_path):
    cfg = OmegaConf.to_container(_metric_config())
    del cfg["candidate"]
    with pytest.raises(ValueError, match="Candidate"):
        Evaluator(
            ckpts=CKPTS,
            optimized_metric_config=OmegaConf.create(cfg),
            accelerator="cpu",
            default_root_dir=str(tmp_path),
        )


def test_pair_is_read_at_candidate_only(tmp_path, run_folder):
    evaluator = _evaluator(tmp_path, CKPTS, VALUES)
    evaluator.evaluate_run(run_folder, None, {}, "val", set_optimized_metric=True)

    assert evaluator.optimized_metric == [-0.3, 0.2]
    assert evaluator.optimized_ckpt_name == "cap_ema"


def test_main_metric_alone_without_sec_metric(tmp_path, run_folder):
    evaluator = _evaluator(tmp_path, CKPTS, VALUES, sec=False)
    evaluator.evaluate_run(run_folder, None, {}, "val", set_optimized_metric=True)

    assert evaluator.optimized_metric == -0.3


@pytest.mark.parametrize("pair", [(math.nan, 0.2), (-0.3, math.inf), (-0.3, None)])
def test_non_finite_pair_is_not_reported(tmp_path, run_folder, pair):
    evaluator = _evaluator(tmp_path, CKPTS, {**VALUES, "cap_ema": pair})
    evaluator.evaluate_run(run_folder, None, {}, "val", set_optimized_metric=True)

    assert evaluator.optimized_metric is None
    assert evaluator.optimized_ckpt_name is None


def test_test_split_leaves_pair_untouched(tmp_path, run_folder):
    evaluator = _evaluator(tmp_path, CKPTS, VALUES)
    evaluator.evaluate_run(run_folder, None, {}, "val", set_optimized_metric=True)
    evaluator.values = {k: (v[0] + 1.0, v[1]) for k, v in VALUES.items()}
    evaluator.evaluate_run(run_folder, None, {}, "test")

    assert evaluator.optimized_metric == [-0.3, 0.2]


def test_last_is_evaluated_but_never_reported(tmp_path, run_folder):
    ckpts = {**CKPTS, "last": True}
    evaluator = _evaluator(tmp_path, ckpts, VALUES)
    evaluator.evaluate_run(run_folder, None, {}, "val", set_optimized_metric=True)

    assert "last" in evaluator.strat_names_seen
    assert evaluator.evaluator.strat_name is None
    assert evaluator.optimized_metric == [-0.3, 0.2]
