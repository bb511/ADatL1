"""src/run_eval_metrics.py with eval_split=val (re-evaluation) and eval_split=test."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

pytest.importorskip("torch")


def _import_run_eval_metrics():
    """Import src/run_eval_metrics.py, which registers the OmegaConf resolvers.

    Another entrypoint imported earlier in the session (src.train) has usually
    registered them already, and a second registration raises.
    """
    import src.utils.omegaconf as resolvers

    if not OmegaConf.has_resolver("eval"):
        import src.run_eval_metrics as module

        return module
    register = resolvers.register_resolvers
    resolvers.register_resolvers = lambda: None
    try:
        import src.run_eval_metrics as module
    finally:
        resolvers.register_resolvers = register
    return module


run_eval_metrics = _import_run_eval_metrics()


class FakeDataModule:
    def __init__(self):
        self.calls = []

    def setup(self, stage):
        self.calls.append(("setup", stage))

    def teardown(self, stage):
        self.calls.append(("teardown", stage))

    def val_dataloader(self):
        self.calls.append(("loader", "val"))
        return {"normal": "val-loader"}

    def test_dataloader(self):
        self.calls.append(("loader", "test"))
        return {"normal": "test-loader"}


class FakeEvaluator:
    optimized_metric = 0.42

    def __init__(self):
        self.runs = []

    def evaluate_run(self, run_ckpts, algorithm, loader, split, set_optimized_metric=False):
        self.runs.append((loader, split, set_optimized_metric))

    def release_dataloaders(self):
        pass


def _cfg(split=None):
    cfg = {
        "experiment_name": "Exp",
        "paths": {"output_dir": "/tmp"},
        "evaluation": {"callbacks": {
            "latent_collapse": {"_target_": "x", "evaluation_split": "val"},
            "correlation_matrix": {"_target_": "y"},
            "reco": None,
        }},
    }
    if split is not None:
        cfg["eval_split"] = split
    return OmegaConf.create(cfg)


@pytest.fixture
def stage(monkeypatch, tmp_path):
    record = SimpleNamespace(datamodule=FakeDataModule(), evaluator=FakeEvaluator(),
                             status=[], context_stage=[])

    def build_stage_context(cfg, *, stage_name, strict_manifest):
        record.context_stage.append(stage_name)
        return SimpleNamespace(datamodule=record.datamodule, run_ckpts=tmp_path, algorithm="ae",
                               logger=[], object_dict={}, reused_mlflow_run_id=None)

    monkeypatch.setattr(run_eval_metrics, "build_stage_context", build_stage_context)
    monkeypatch.setattr(run_eval_metrics, "get_evaluator", lambda cfg, logger: record.evaluator)
    monkeypatch.setattr(run_eval_metrics, "write_stage_status",
                        lambda run_ckpts, **kwargs: record.status.append(kwargs))
    monkeypatch.setattr(run_eval_metrics, "finish_stage_loggers", lambda context: None)
    monkeypatch.setattr(run_eval_metrics, "release_accelerator_cache", lambda: None)
    return record


def test_validation_is_the_default(stage, tmp_path):
    cfg = _cfg()
    metric_dict, _ = run_eval_metrics.run_eval_metrics(cfg)

    assert stage.datamodule.calls == [("setup", "validate"), ("loader", "val"), ("teardown", "validate")]
    assert stage.evaluator.runs == [({"normal": "val-loader"}, "val", True)]
    assert stage.context_stage == ["metrics"]
    assert stage.status[0]["stage_name"] == "metrics"
    assert stage.status[0]["artifacts"] == [str(tmp_path / "plots" / "val" / "loss_total")]
    assert metric_dict == {"optimized_metric": 0.42}
    assert cfg.evaluation.callbacks.latent_collapse.evaluation_split == "val"


def test_test_split_replays_test_without_touching_selection(stage, tmp_path):
    cfg = _cfg("test")
    metric_dict, _ = run_eval_metrics.run_eval_metrics(cfg)

    assert stage.datamodule.calls == [("setup", "test"), ("loader", "test"), ("teardown", "test")]
    # No optimized metric on test: it must not feed back into selection.
    assert stage.evaluator.runs == [({"normal": "test-loader"}, "test", False)]
    assert metric_dict == {}
    assert stage.context_stage == ["metrics_test"]
    assert stage.status[0]["stage_name"] == "metrics_test"
    assert stage.status[0]["artifacts"] == [str(tmp_path / "plots" / "test" / "loss_total")]
    # The collapse diagnostic would silently skip test otherwise.
    assert cfg.evaluation.callbacks.latent_collapse.evaluation_split == "test"


def test_unknown_split_is_refused(stage):
    with pytest.raises(ValueError, match="eval_split must be one of"):
        run_eval_metrics.run_eval_metrics(_cfg("train"))
    assert stage.datamodule.calls == []


def test_follow_eval_split_only_touches_split_bound_callbacks():
    cfg = _cfg()
    assert run_eval_metrics.follow_eval_split(cfg, "val") == []
    assert run_eval_metrics.follow_eval_split(cfg, "test") == ["latent_collapse"]
    assert "evaluation_split" not in cfg.evaluation.callbacks.correlation_matrix
    assert run_eval_metrics.stage_name_for("val") == "metrics"
    assert run_eval_metrics.stage_name_for("test") == "metrics_test"


def test_composed_study_config_defaults_to_val_and_switches_collapse_to_test():
    from hydra import compose, initialize_config_dir

    configs = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["experiment=physics/pareto_fet"])
        test_cfg = compose(config_name="train",
                           overrides=["experiment=physics/pareto_fet", "eval_split=test"])
    assert run_eval_metrics.resolve_eval_split(cfg) == "val"
    assert run_eval_metrics.resolve_eval_split(test_cfg) == "test"
    assert run_eval_metrics.follow_eval_split(test_cfg, "test") == ["latent_collapse"]
