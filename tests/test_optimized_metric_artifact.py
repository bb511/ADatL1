"""Tests for persistent model-selection provenance emitted by training."""

import json
from types import SimpleNamespace

from omegaconf import OmegaConf

from src import train


def test_optimized_metric_artifact_records_checkpoint_and_objectives(
    tmp_path, monkeypatch
) -> None:
    """Campaign runs must preserve the exact selected checkpoint and objective pair."""
    monkeypatch.setattr(
        train.HydraConfig,
        "get",
        lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))),
    )
    evaluator = SimpleNamespace(
        optimized_ckpt_name="normal",
        optimized_metric=[0.75, 1.25],
    )

    path = train._write_optimized_metric_artifact(evaluator)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "optimized_ckpt_name": "normal",
        "optimized_metric": [0.75, 1.25],
    }


def test_shared_optimized_metric_artifact_records_every_logical_selection(
    tmp_path, monkeypatch
) -> None:
    """A shared trial must preserve each strategy's independently selected checkpoint."""
    monkeypatch.setattr(
        train.HydraConfig,
        "get",
        lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path))),
    )
    evaluator = SimpleNamespace(
        optimized_metric_configs={"native__cap__cdf": {}, "native__drift": {}},
        optimized_metrics={
            "native__cap__cdf": [0.8, 1.2],
            "native__drift": [0.1, 1.0],
        },
        optimized_ckpt_names={
            "native__cap__cdf": "cap_cdf",
            "native__drift": "last",
        },
        optimized_metric=(0.8, 1.2, 0.1, 1.0),
        optimized_ckpt_name=None,
    )

    path = train._write_optimized_metric_artifact(evaluator)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 2,
        "objective_order": ["native__cap__cdf", "native__drift"],
        "optimized_metric": [0.8, 1.2, 0.1, 1.0],
        "selections": {
            "native__cap__cdf": {
                "optimized_ckpt_name": "cap_cdf",
                "optimized_metric": [0.8, 1.2],
            },
            "native__drift": {
                "optimized_ckpt_name": "last",
                "optimized_metric": [0.1, 1.0],
            },
        },
    }


def test_shared_objective_directions_are_flattened_in_config_order() -> None:
    """Worst-value fallback must have one direction for every returned scalar."""
    cfg = OmegaConf.create(
        {
            "optimized_metric_configs": {
                "cap": {
                    "main_metric": {"direction": "maximize"},
                    "sec_metric": {"direction": "minimize"},
                },
                "drift": {
                    "main_metric": {"direction": "minimize"},
                    "sec_metric": {"direction": "minimize"},
                },
            }
        }
    )
    assert train._get_directions(cfg) == ["maximize", "minimize", "minimize", "minimize"]
