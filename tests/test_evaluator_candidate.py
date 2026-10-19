import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

from src.evaluation.evaluator import Evaluator

# (main, sec) per checkpoint stem. The non-candidates drift_ema, loss and last beat
# the candidate cap_ema on both objectives, so any rule that compares checkpoints
# would report one of them instead.
VALUES = {
    "cap_ema": (-0.3, 0.2),
    "drift_ema": (-0.1, 0.1),
    "loss": (-0.2, 0.15),
    "last": (-0.05, 0.05),
}
CKPTS = {
    "summary": {"cap_ema": ["max"], "drift_ema": ["min"]},
    "normal": {"loss": ["min"]},
}
FOLDERS = [
    ("summary", "cap_ema", "max"),
    ("summary", "drift_ema", "min"),
    ("normal", "loss", "min"),
]


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


def _metric_config():
    cfg = {
        "candidate": {"strategy": "summary", "metric": "cap_ema", "criterion": "max"},
        "main_metric": {
            "callback": {"name": "main", "params": {}},
            "direction": "maximize",
        },
        "sec_metric": {
            "callback": {
                "name": "sec",
                "params": {"ckpt_name": None, "test_ds": "normal"},
            },
            "direction": "minimize",
        },
    }
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg


def _evaluator(tmp_path: Path, ckpts: dict, values: dict):
    evaluator = Evaluator(
        ckpts=OmegaConf.create(ckpts),
        callbacks=[_Main("main", 0), _Sec("sec", 1)],
        logger=None,
        optimized_metric_config=_metric_config(),
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        default_root_dir=str(tmp_path),
    )
    evaluator.values = values
    evaluator.evaluated = []

    def evaluate_ckpt(ckpt_path: Path, model, test_loader):
        evaluator.evaluated.append(ckpt_path.stem)
        for cb in evaluator.callbacks:
            cb.summary[ckpt_path.stem] = evaluator.values[ckpt_path.stem][cb.index]

    evaluator.evaluate_ckpt = evaluate_ckpt
    return evaluator


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    for strategy, metric, criterion in FOLDERS:
        folder = root / strategy / metric / criterion
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


def test_pair_is_read_at_candidate_only(tmp_path, run_folder):
    evaluator = _evaluator(tmp_path, CKPTS, VALUES)
    evaluator.evaluate_run(run_folder, None, {}, "val", set_optimized_metric=True)

    assert evaluator.optimized_metric == [-0.3, 0.2]
    assert evaluator.optimized_ckpt_name == "cap_ema"


@pytest.mark.parametrize("pair", [(math.nan, 0.2), (-0.3, math.inf), (-0.3, None)])
def test_non_finite_pair_is_not_reported(tmp_path, run_folder, pair):
    evaluator = _evaluator(tmp_path, CKPTS, {**VALUES, "cap_ema": pair})
    evaluator.evaluate_run(run_folder, None, {}, "val", set_optimized_metric=True)

    assert evaluator.optimized_metric is None
    assert evaluator.optimized_ckpt_name is None


def test_every_listed_checkpoint_is_evaluated(tmp_path, run_folder):
    evaluator = _evaluator(tmp_path, CKPTS, VALUES)
    evaluator.evaluate_run(run_folder, None, {}, "val", set_optimized_metric=True)

    assert sorted(evaluator.evaluated) == ["cap_ema", "drift_ema", "loss"]
