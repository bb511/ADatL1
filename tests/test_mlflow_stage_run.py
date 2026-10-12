"""Stage 2 and run_eval_metrics must log into the stage-1 MLflow run instead of opening a new one."""

from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.entities import Param  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.utils.mlflow_stage_run import (  # noqa: E402
    _find_with_client,
    find_stage1_run_id,
    tolerate_param_conflicts,
)


def _run(client, exp_id, name, source, start):
    run = client.create_run(
        exp_id,
        start_time=start,
        tags={"mlflow.runName": name, "mlflow.source.name": source},
    )
    return run.info.run_id


@pytest.fixture()
def store(tmp_path: Path):
    root = tmp_path / "mlruns"
    uri = f"file:{root}"
    client = MlflowClient(tracking_uri=uri)
    exp = client.create_experiment("Study")
    ids = {
        "train_old": _run(client, exp, "R1", "src/train.py", 1_000),
        "probes": _run(client, exp, "R1", "src/run_probes.py", 3_000),
        "train_new": _run(client, exp, "R1", "src/train.py", 2_000),
        "deleted": _run(client, exp, "R1", "src/train.py", 4_000),
        "other_name": _run(client, exp, "R2", "src/train.py", 5_000),
    }
    client.delete_run(ids["deleted"])
    return uri, root, client, exp, ids


def test_picks_newest_active_stage1_run(store):
    uri, _root, _client, _exp, ids = store
    assert find_stage1_run_id(uri, "Study", "R1") == ids["train_new"]
    # The triple-slash form written by MLflow itself resolves to the same store.
    assert find_stage1_run_id(uri.replace("file:", "file://"), "Study", "R1") == ids["train_new"]


def test_none_when_no_stage1_run(store):
    uri, *_ = store
    assert find_stage1_run_id(uri, "Study", "missing") is None
    assert find_stage1_run_id(uri, "OtherStudy", "R1") is None


def test_searches_duplicate_experiments_from_merged_sandboxes(store):
    uri, root, client, exp, ids = store
    # A store merged from HTCondor sandboxes: a second experiment, same name.
    dup = client.create_experiment("Study-dup")
    newest = _run(client, dup, "R1", "/abs/path/src/train.py", 9_000)
    meta = root / dup / "meta.yaml"
    meta.write_text(meta.read_text().replace("name: Study-dup", "name: Study"))
    assert find_stage1_run_id(uri, "Study", "R1") == newest


def test_client_fallback_agrees(store):
    uri, _root, _client, _exp, ids = store
    assert _find_with_client(uri, "Study", "R1") == ids["train_new"]


def test_changed_params_do_not_crash_and_are_recorded(store):
    uri, _root, client, _exp, ids = store
    run_id = ids["train_new"]
    client.log_param(run_id, "probe/protocol", "v9")

    class MLFlowLogger:  # duck-typed stand-in for pytorch_lightning's logger
        experiment = client

        def __init__(self):
            self.run_id = run_id

        def log_hyperparams(self, params):
            client.log_batch(self.run_id, params=[Param(k, str(v)) for k, v in params.items()])

    logger = MLFlowLogger()
    with pytest.raises(Exception, match="Changing param values is not allowed"):
        logger.log_hyperparams({"probe/protocol": "v10"})

    tolerate_param_conflicts([logger])
    logger.log_hyperparams({"probe/protocol": "v10", "probe/valid": True})
    run = client.get_run(run_id)
    assert run.data.params["probe/protocol"] == "v9"
    assert run.data.tags["param_update.probe/protocol"] == "v10"
    assert run.data.params["probe/valid"] == "True"
