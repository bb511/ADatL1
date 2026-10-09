"""Keep one MLflow run per checkpoint across the pipeline stages.

Until the split into stages (2026-09-18) a run was trained, evaluated and probed
in one process, so every checkpoint had exactly one MLflow run holding the
training curves, the plots and the analysis metrics. After the split, stages 2
and 3 instantiated the MLflow logger afresh, so every analysis job opened a
*new* run with the same run name, never closed it (it stayed RUNNING), and the
training run never received the analysis metrics.

``find_stage1_run_id`` lets a stage reopen the stage-1 run instead, and
``tolerate_param_conflicts`` keeps a re-run stage from crashing on MLflow's
immutable params. Deliberately free of torch/lightning imports so it can be
tested on its own.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

#: Stage-1 runs are the ones created by ``src/train.py``; MLflow records the
#: entry point in the ``mlflow.source.name`` tag.
TRAIN_ENTRYPOINT = "train.py"

#: Same truncation pytorch_lightning's MLFlowLogger.log_hyperparams applies.
PARAM_VALUE_LIMIT = 250


def _file_store_root(tracking_uri: Optional[str]) -> Optional[str]:
    """Return the directory of a ``file:`` tracking URI, else None."""
    if not tracking_uri or not str(tracking_uri).startswith("file:"):
        return None
    path = str(tracking_uri)[len("file:"):]
    if path.startswith("//"):  # file:///abs/path
        path = path[2:]
    return os.path.normpath(path)


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _meta_value(text: str, key: str) -> Optional[str]:
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def find_stage1_run_id(
    tracking_uri: Optional[str], experiment_name: str, run_name: str
) -> Optional[str]:
    """Return the newest active stage-1 run of ``experiment_name/run_name``.

    Searches every experiment with that name, because a store merged from
    HTCondor job sandboxes holds one experiment per job, all with the same name.
    Returns None when no stage-1 run exists.
    """
    root = _file_store_root(tracking_uri)
    if root is not None:
        return _find_in_file_store(root, experiment_name, run_name)
    return _find_with_client(tracking_uri, experiment_name, run_name)


def _find_in_file_store(root: str, experiment_name: str, run_name: str) -> Optional[str]:
    # Read the store directly: MlflowClient.search_runs on a file store loads
    # every metric of every run, which on EOS takes minutes for one study.
    if not os.path.isdir(root):
        return None
    best: Optional[tuple] = None
    for exp_id in os.listdir(root):
        exp_dir = os.path.join(root, exp_id)
        meta = _read(os.path.join(exp_dir, "meta.yaml"))
        if (
            not meta
            or _meta_value(meta, "name") != experiment_name
            or _meta_value(meta, "lifecycle_stage") != "active"
        ):
            continue
        for run_id in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_id)
            name = _read(os.path.join(run_dir, "tags", "mlflow.runName"))
            if name is None or name.strip() != run_name:
                continue
            run_meta = _read(os.path.join(run_dir, "meta.yaml"))
            if not run_meta or _meta_value(run_meta, "lifecycle_stage") != "active":
                continue
            source = _read(os.path.join(run_dir, "tags", "mlflow.source.name")) or ""
            if not source.strip().endswith(TRAIN_ENTRYPOINT):
                continue
            start = int(_meta_value(run_meta, "start_time") or 0)
            if best is None or start > best[0]:
                best = (start, run_id)
    return best[1] if best else None


def _find_with_client(
    tracking_uri: Optional[str], experiment_name: str, run_name: str
) -> Optional[str]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments(filter_string=f"name = '{experiment_name}'")
    if not experiments:
        return None
    runs = client.search_runs(
        [e.experiment_id for e in experiments],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    for run in runs:
        if run.data.tags.get("mlflow.source.name", "").endswith(TRAIN_ENTRYPOINT):
            return run.info.run_id
    return None


def _flatten_params(params: Any) -> dict:
    """Flatten like pytorch_lightning's MLFlowLogger does ('/' joined keys)."""
    try:
        from lightning_fabric.utilities.logger import _convert_params, _flatten_dict

        return _flatten_dict(_convert_params(params))
    except ImportError:  # pragma: no cover - lightning is always installed in practice
        if hasattr(params, "items") and not isinstance(params, dict):
            from omegaconf import OmegaConf

            params = OmegaConf.to_container(params, resolve=True)
        flat: dict = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(f"{prefix}/{key}" if prefix else str(key), item)
            else:
                flat[prefix] = value

        walk("", dict(params))
        return flat


def log_params_without_conflicts(client: Any, run_id: str, params: Any) -> None:
    """Log only new params; record changed values as ``param_update.<key>`` tags."""
    existing = client.get_run(run_id).data.params
    for key, value in _flatten_params(params).items():
        value = str(value)[:PARAM_VALUE_LIMIT]
        if key not in existing:
            client.log_param(run_id, key, value)
        elif existing[key] != value:
            client.set_tag(run_id, f"param_update.{key}", value)
            log.warning(
                f"MLflow param {key!r} is already {existing[key]!r} in run {run_id}; "
                f"kept it and recorded the new value {value!r} as tag param_update.{key}."
            )


def tolerate_param_conflicts(loggers: Optional[Iterable[Any]]) -> None:
    """Make ``log_hyperparams`` of reopened MLflow loggers survive changed params.

    MLflow params are immutable. A stage that reopens the stage-1 run and logs a
    param with a different value (e.g. a newer probe protocol version on a
    re-run) would otherwise abort the job after the expensive work is done.
    """
    for logger in loggers or []:
        if type(logger).__name__ != "MLFlowLogger":
            continue
        original = logger.log_hyperparams

        def log_hyperparams(params, *args, _logger=logger, _original=original, **kwargs):
            try:
                return _original(params, *args, **kwargs)
            except Exception as err:  # mlflow.exceptions.MlflowException
                if "Changing param values is not allowed" not in str(err):
                    raise
            log_params_without_conflicts(_logger.experiment, _logger.run_id, params)

        logger.log_hyperparams = log_hyperparams
