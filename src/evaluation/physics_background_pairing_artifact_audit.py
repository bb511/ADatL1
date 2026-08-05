"""Fail-closed artifact audit for the physics background-pairing campaign.

This module deliberately performs no campaign writes.  It turns authenticated campaign metadata and
trusted-local retrain artifacts into deterministic rows that a reporter can persist in its final
manifest.
"""

from __future__ import annotations

import hashlib
import math
import pickle  # nosec B403
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

PAIR_ID = "ZB_run396102_vs_ZB_run398183"
OAS_MODELS = frozenset({"ae", "vae"})
KNOWN_MODELS = frozenset({"ae", "vae", "dsae", "dsvae", "svdd", "realnvp"})
KNOWN_SCORES = frozenset({"native", "residual_oas"})
KNOWN_METRICS = frozenset({"cap", "wasserstein", "drift"})


def _sha256(path: Path) -> str:
    """Return the file's streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path, label: str) -> Path:
    """Resolve an existing path and require confinement to ``root``."""
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes campaign root: {path}") from error
    return resolved


def _finite(value: Any, label: str) -> float:
    """Coerce one finite scalar or fail closed."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite scalar.") from error
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be a finite scalar.")
    return scalar


def _objective_index(design: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index unique authenticated design objectives by model and ID."""
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for study in design.get("studies", []):
        model = str(study.get("model"))
        if model not in KNOWN_MODELS:
            raise ValueError(f"Unknown model in design: {model}")
        for raw in study.get("objectives", []):
            objective = dict(raw)
            key = (model, str(objective.get("id")))
            if key in indexed:
                raise ValueError(f"Duplicate design objective: {key}")
            indexed[key] = objective
    if not indexed:
        raise ValueError("Design contains no objectives.")
    return indexed


def _primary_name(model: str, objective: Mapping[str, Any]) -> tuple[str, str]:
    """Derive an objective's exact primary metric name and checkpoint direction."""
    score = str(objective.get("score"))
    metric = str(objective.get("metric"))
    strategy = objective.get("strategy")
    if score not in KNOWN_SCORES or (score == "residual_oas" and model not in OAS_MODELS):
        raise ValueError(f"Invalid score {score!r} for model {model!r}.")
    if metric not in KNOWN_METRICS:
        raise ValueError(f"Unknown selection metric: {metric}")
    if metric == "cap":
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("CAP objectives require a pairing strategy.")
        return f"cap_{score}_{strategy}_ema_{PAIR_ID}", "max"
    if strategy not in (None, ""):
        raise ValueError(f"{metric} objectives must not declare a pairing strategy.")
    if metric == "wasserstein":
        return f"w1dist_{score}_ema_{PAIR_ID}", "min"
    return f"{score}_operational_drift_ema", "min"


def _classify_family(
    model: str, objective: Mapping[str, Any], checkpoint_family_id: str
) -> tuple[str, str | None]:
    """Classify one exact evaluator candidate ID against its objective contract."""
    primary, direction = _primary_name(model, objective)
    primary_id = f"summary/{primary}/{direction}"
    allowed = {
        "last": "last",
        "single/ascore_operational/stable": "stable_native",
        primary_id: "primary_ema",
    }
    try:
        return allowed[checkpoint_family_id], primary
    except KeyError as error:
        raise ValueError(
            f"Checkpoint family {checkpoint_family_id!r} is not valid for "
            f"{model}/{objective.get('id')}; expected one of {sorted(allowed)}."
        ) from error


def _nested(config: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    """Read a required nested mapping from a resolved config."""
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Resolved config is missing {'.'.join(keys)}.")
        value = value[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"Resolved config field {'.'.join(keys)} must be a mapping.")
    return value


def _assert_callback(
    callback: Mapping[str, Any], *, output: str, name: str | None, namespace: str | None
) -> None:
    """Validate the shared callback output, name, and namespace contract."""
    if callback.get("output_name") != output:
        raise ValueError(f"Callback output must be {output!r}.")
    if name is not None and callback.get("name") != name:
        raise ValueError(f"Evaluation callback name must be {name!r}.")
    if callback.get("threshold_namespace") != namespace:
        raise ValueError(f"Callback threshold namespace must be {namespace!r}.")


def _audit_config(config: Mapping[str, Any], model: str) -> tuple[float, float]:
    """Validate retrain leakage, callback, and operating-rate contracts."""
    validation_aux = _nested(config, "data").get("validation_aux_datasets")
    if validation_aux != []:
        raise ValueError("Retrain validation_aux_datasets must remain empty (leakage boundary).")

    callbacks = _nested(config, "callbacks")
    evaluation_callbacks = _nested(config, "evaluation", "callbacks")
    native = _nested(callbacks, "anomaly_eff")
    native_eval = _nested(evaluation_callbacks, "anomaly_efficiency")
    _assert_callback(native, output="ascore/full", name=None, namespace=None)
    _assert_callback(native_eval, output="ascore/full", name="eff_native", namespace=None)
    if native.get("metric_suffix") is not None:
        raise ValueError("Native callback must not define a metric suffix.")

    has_training_oas = "anomaly_eff_oas" in callbacks
    has_evaluation_oas = "anomaly_efficiency_oas" in evaluation_callbacks
    if model in OAS_MODELS:
        if not (has_training_oas and has_evaluation_oas):
            raise ValueError(f"{model} retraining requires native and residual-OAS callbacks.")
        oas = _nested(callbacks, "anomaly_eff_oas")
        oas_eval = _nested(evaluation_callbacks, "anomaly_efficiency_oas")
        _assert_callback(
            oas,
            output="ascore/residual_oas",
            name=None,
            namespace="residual_oas",
        )
        if oas.get("metric_suffix") != "residual_oas":
            raise ValueError("OAS training callback metric_suffix must be 'residual_oas'.")
        _assert_callback(
            oas_eval,
            output="ascore/residual_oas",
            name="eff_residual_oas",
            namespace="residual_oas",
        )
    elif has_training_oas or has_evaluation_oas:
        raise ValueError(f"Non-AE/VAE model {model} must not configure residual OAS.")

    algorithm = _nested(config, "algorithm")
    target_rate = _finite(algorithm.get("target_rate"), "algorithm.target_rate")
    base_rate = _finite(algorithm.get("base_rate"), "algorithm.base_rate")
    return target_rate, base_rate


def _checkpoint_file(root: Path, family_id: str) -> Path:
    """Resolve exactly the selected checkpoint within its authenticated root."""
    if family_id == "last":
        candidate = root / "last.ckpt"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve(strict=True)
    directory = root / family_id
    resolved_directory = _inside(directory, root, "checkpoint family")
    files = sorted(resolved_directory.glob("*.ckpt"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint in {resolved_directory}, found {files}."
        )
    return _inside(files[0], root, "selected checkpoint")


def _checkpoint_identity(path: Path) -> str:
    """Normalize a physical checkpoint stem as the evaluator does."""
    name = path.stem
    if "ds=" in name:
        return name.split("ds=", 1)[1].split("__", 1)[0]
    return name


def _state_dict(path: Path) -> Mapping[str, Any]:
    """Load the state dictionary from a trusted campaign-local checkpoint."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        # Lightning checkpoints can contain benign callables (for example
        # functools.partial) that the weights-only unpickler refuses.  This
        # fallback is limited to a path already resolved inside the authenticated
        # campaign root and is therefore an explicit trusted-local operation.
        payload = torch.load(path, map_location="cpu", weights_only=False)  # nosec
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint state_dict is not a mapping: {path}")
    return state


def _threshold_values(state: Mapping[str, Any], model: str) -> dict[str, float]:
    """Extract exactly the finite threshold buffers required by a model."""
    expected = {"thres_operational"}
    if model in OAS_MODELS:
        expected.add("thres_operational__residual_oas")
    observed = {str(key) for key in state if str(key).startswith("thres_operational")}
    if observed != expected:
        raise ValueError(f"Threshold buffer keys differ: expected {expected}, found {observed}.")
    values = {}
    for key in sorted(expected):
        tensor = torch.as_tensor(state[key]).detach().cpu()
        if tensor.numel() != 1:
            raise ValueError(f"Threshold buffer {key} must be scalar.")
        values[key] = _finite(tensor.item(), f"threshold buffer {key}")
    return values


def _summary_contract(objective: Mapping[str, Any]) -> tuple[str, int, bool]:
    """Return callback folder, pickle count, and nesting for an objective."""
    score = str(objective["score"])
    metric = str(objective["metric"])
    if metric == "cap":
        return f"cap_{score}_{objective['strategy']}_summary", 2, False
    if metric == "wasserstein":
        return f"wasserstein_{score}_summary", 1, False
    return f"drift_{score}_summary", 2, True


def _pickle_objects(path: Path) -> list[Any]:
    """Read every object from an already hashed, trusted-local pickle stream."""
    values = []
    with path.open("rb") as handle:
        while True:
            try:
                values.append(pickle.load(handle))  # nosec B301
            except EOFError:
                return values


def _validate_summary_objects(objects: list[Any], count: int, nested: bool, identity: str) -> None:
    """Validate summary count, shape, identity, and finite values."""
    if len(objects) != count:
        raise ValueError(f"Expected {count} summary objects, found {len(objects)}.")
    for index, payload in enumerate(objects):
        if not isinstance(payload, Mapping):
            raise ValueError(f"Summary object {index} must be a mapping.")
        values = payload
        if nested:
            if set(payload) != {"operational"} or not isinstance(payload["operational"], Mapping):
                raise ValueError("Drift summary must contain only the operational threshold.")
            values = payload["operational"]
        if set(values) != {identity}:
            raise ValueError(
                f"Summary object {index} checkpoint keys differ from "
                f"selected identity {identity!r}."
            )
        _finite(values[identity], f"summary object {index} value")


def audit_physics_background_pairing_artifacts(
    *,
    campaign_root: Path,
    design: Mapping[str, Any],
    manifest: Mapping[str, Any],
    markers: Iterable[Mapping[str, Any]],
    checkpoint_selections: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Audit retrain configs, selected checkpoints, thresholds, and test summaries.

    Each selection must identify its exact evaluator candidate family through
    ``checkpoint_family_id`` or the existing ``optimized_ckpt_name`` field.
    Optional ``config_sha256`` and ``checkpoint_sha256`` values are verified when
    supplied, but frozen campaign markers are not required to contain either.
    """

    root = campaign_root.resolve(strict=True)
    objectives = _objective_index(design)
    retrains = {int(row["retrain_index"]): dict(row) for row in manifest["retrain_runs"]}
    if len(retrains) != len(manifest["retrain_runs"]):
        raise ValueError("Duplicate retrain indices in manifest.")
    marker_rows = [dict(row) for row in markers]
    marker_index = {int(row["retrain_index"]): row for row in marker_rows}
    if len(marker_index) != len(marker_rows):
        raise ValueError("Duplicate retrain indices in markers.")
    if set(marker_index) != set(retrains):
        raise ValueError("Markers do not exactly cover manifest retrains.")

    selections = [dict(row) for row in checkpoint_selections]
    keys = [
        (str(row["model"]), int(row["retrain_index"]), str(row["logical_objective"]))
        for row in selections
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate checkpoint selection rows.")
    if not selections:
        raise ValueError("No checkpoint selections supplied.")
    expected_keys = {
        (str(row["model"]), retrain_index, str(objective_id))
        for retrain_index, row in retrains.items()
        for objective_id in row.get("logical_objectives", [])
    }
    if set(keys) != expected_keys:
        raise ValueError("Checkpoint selections do not exactly cover manifest objectives.")

    family_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    audited_checkpoints: set[tuple[int, str]] = set()
    audited_configs: dict[int, tuple[Path, str, float, float]] = {}

    for selection in sorted(
        selections,
        key=lambda row: (
            str(row["model"]),
            int(row["retrain_index"]),
            str(row["logical_objective"]),
        ),
    ):
        model = str(selection["model"])
        retrain_index = int(selection["retrain_index"])
        objective_id = str(selection["logical_objective"])
        objective = objectives.get((model, objective_id))
        if objective is None:
            raise ValueError(f"Selection is absent from design: {model}/{objective_id}")
        retrain = retrains.get(retrain_index)
        if retrain is None or retrain.get("model") != model:
            raise ValueError(f"Selection differs from manifest retrain {retrain_index}.")
        if objective_id not in retrain.get("logical_objectives", []):
            raise ValueError(
                f"Objective {objective_id} is not assigned to retrain {retrain_index}."
            )
        marker = marker_index[retrain_index]
        marker_row = marker.get("row")
        if marker_row != retrain:
            raise ValueError(f"Marker row differs from manifest retrain {retrain_index}.")

        family_id = str(
            selection.get("checkpoint_family_id", selection.get("optimized_ckpt_name", ""))
        )
        if (
            "checkpoint_family_id" in selection
            and "optimized_ckpt_name" in selection
            and str(selection["optimized_ckpt_name"]) != family_id
        ):
            raise ValueError(f"Checkpoint family declarations differ for {model}/{objective_id}.")
        family, primary = _classify_family(model, objective, family_id)
        checkpoint_root = _inside(Path(str(marker["checkpoint_root"])), root, "checkpoint root")

        if retrain_index not in audited_configs:
            config_path = _inside(
                root / "retraining" / "hydra" / f"{retrain_index:04d}" / ".hydra/config.yaml",
                root,
                "retrain config",
            )
            with config_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if not isinstance(config, Mapping):
                raise ValueError(f"Resolved retrain config must be a mapping: {config_path}")
            target_rate, base_rate = _audit_config(config, model)
            config_hash = _sha256(config_path)
            audited_configs[retrain_index] = (
                config_path,
                config_hash,
                target_rate,
                base_rate,
            )
        config_path, config_hash, target_rate, base_rate = audited_configs[retrain_index]
        expected_config_hash = selection.get("config_sha256")
        if expected_config_hash is not None and expected_config_hash != config_hash:
            raise ValueError(f"Config hash mismatch for retrain {retrain_index}.")

        checkpoint = _checkpoint_file(checkpoint_root, family_id)
        checkpoint_hash = _sha256(checkpoint)
        expected_checkpoint_hash = selection.get("checkpoint_sha256")
        if expected_checkpoint_hash is not None and expected_checkpoint_hash != checkpoint_hash:
            raise ValueError(f"Checkpoint hash mismatch for {model}/{objective_id}.")
        identity = _checkpoint_identity(checkpoint)
        declared_identity = selection.get("checkpoint_identity")
        if declared_identity is not None and str(declared_identity) != identity:
            raise ValueError(f"Selected checkpoint identity differs for {model}/{objective_id}.")

        common = {
            "model": model,
            "retrain_index": retrain_index,
            "logical_objective": objective_id,
            "score": objective["score"],
            "metric": objective["metric"],
            "strategy": objective.get("strategy") or "",
            "checkpoint_family_id": family_id,
            "checkpoint_family": family,
            "primary_criterion": primary,
            "checkpoint_identity": identity,
            "checkpoint_path": str(checkpoint.relative_to(root)),
            "checkpoint_sha256": checkpoint_hash,
            "config_path": str(config_path.relative_to(root)),
            "config_sha256": config_hash,
        }
        family_rows.append(dict(common))

        threshold_key = (retrain_index, str(checkpoint))
        if threshold_key not in audited_checkpoints:
            for buffer_name, value in _threshold_values(_state_dict(checkpoint), model).items():
                threshold_rows.append(
                    {
                        **common,
                        "threshold_buffer": buffer_name,
                        "threshold_value": value,
                        "target_rate": target_rate,
                        "base_rate": base_rate,
                    }
                )
            audited_checkpoints.add(threshold_key)

        callback, object_count, nested = _summary_contract(objective)
        summary_path = _inside(
            checkpoint.parent / "plots" / "test" / callback / "summary.pkl",
            checkpoint_root,
            "metric summary",
        )
        summary_hash = _sha256(summary_path)
        objects = _pickle_objects(summary_path)
        _validate_summary_objects(objects, object_count, nested, identity)
        summary_rows.append(
            {
                **common,
                "callback": callback,
                "summary_path": str(summary_path.relative_to(root)),
                "summary_sha256": summary_hash,
                "object_count": object_count,
            }
        )

    sort_key = lambda row: (  # noqa: E731 - shared deterministic row ordering
        row["model"],
        row["retrain_index"],
        row["logical_objective"],
    )
    family_rows.sort(key=sort_key)
    threshold_rows.sort(key=lambda row: (*sort_key(row), row["threshold_buffer"]))
    summary_rows.sort(key=lambda row: (*sort_key(row), row["callback"]))
    return {
        "checkpoint_families": family_rows,
        "threshold_audit": threshold_rows,
        "metric_summary_files": summary_rows,
    }


__all__ = ["audit_physics_background_pairing_artifacts"]
