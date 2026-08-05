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


def test_optimized_metric_artifact_records_run_provenance(tmp_path, monkeypatch) -> None:
    """A campaign artifact explicitly binds Hydra, checkpoint, and logger identities."""
    (tmp_path / "hydra").mkdir()
    monkeypatch.setattr(
        train.HydraConfig,
        "get",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(tmp_path / "hydra")),
            job=SimpleNamespace(num=7),
        ),
    )
    cfg = OmegaConf.create(
        {
            "paths": {"checkpoints_dir": str(tmp_path / "checkpoints")},
            "experiment_name": "physics_bgpair_search_ae",
            "run_name": "abc123",
        }
    )
    logger = SimpleNamespace(experiment_id="4", run_id="run-1", save_dir=str(tmp_path / "mlruns"))
    evaluator = SimpleNamespace(optimized_ckpt_name="last", optimized_metric=[1.0])

    artifact = train._write_optimized_metric_artifact(evaluator, cfg=cfg, loggers=[logger])
    provenance = json.loads(artifact.read_text(encoding="utf-8"))["run_provenance"]

    assert provenance["hydra_job_num"] == 7
    assert provenance["run_name"] == "abc123"
    assert provenance["loggers"][0]["run_id"] == "run-1"
    assert provenance["checkpoint_run_dir"].endswith("checkpoints/physics_bgpair_search_ae/abc123")


def test_single_run_provenance_accepts_unset_hydra_job_number(tmp_path, monkeypatch) -> None:
    """Hydra single runs have no multirun job index and must serialize it as null."""
    output = tmp_path / "hydra"
    output.mkdir()
    hydra_cfg = OmegaConf.create({"runtime": {"output_dir": str(output)}, "job": {"num": "???"}})
    monkeypatch.setattr(train.HydraConfig, "get", lambda: hydra_cfg)
    cfg = OmegaConf.create(
        {
            "paths": {"checkpoints_dir": str(tmp_path / "checkpoints")},
            "experiment_name": "physics_bgpair_pilot_ae",
            "run_name": "abc123",
        }
    )
    evaluator = SimpleNamespace(optimized_ckpt_name="last", optimized_metric=[1.0])

    artifact = train._write_optimized_metric_artifact(evaluator, cfg=cfg, loggers=[])

    provenance = json.loads(artifact.read_text(encoding="utf-8"))["run_provenance"]
    assert provenance["hydra_job_num"] is None
