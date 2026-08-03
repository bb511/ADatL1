"""Tests for persistent model-selection provenance emitted by training."""

import json
from types import SimpleNamespace

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
