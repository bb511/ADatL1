"""Build the authenticated 28-config/42-cell physics confirmation matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
SELECTORS = (
    "cap__flat_physical",
    "cap__physics_summary",
    "cap__typed_sliced_wasserstein",
    "cap__jetclr",
    "cap__cdf",
    "wasserstein",
    "drift",
)
TRAINING_SEEDS = (271828, 314159, 161803)
RESERVE_TRAINING_SEEDS = (141421, 173205)
SEED_EXTENSION_RULE = {
    "trigger_endpoint": "efficiency_operational",
    "practical_delta": 0.005,
    "rule": (
        "After all three primary seeds complete for one model, add both reserve seeds to "
        "every configuration of that model iff any multiplicity-adjusted primary-efficiency "
        "contrast interval spans both -0.005 and +0.005."
    ),
    "outcome_direction_blind": True,
}
TRIALS_BY_MODEL = {
    "ae": (154, 154, 154, 352, 369, 418, 154),
    "vae": (590, 507, 590, 507, 634, 393, 516),
    "dsae": (478, 40, 40, 40, 478, 478, 40),
    "dsvae": (328, 456, 580, 436, 368, 421, 318),
    "svdd": (177, 400, 400, 400, 102, 447, 142),
    "realnvp": (457, 377, 539, 357, 457, 645, 645),
}
EXPECTED_UNIQUE_BY_MODEL = {
    "ae": 4,
    "vae": 5,
    "dsae": 2,
    "dsvae": 7,
    "svdd": 5,
    "realnvp": 5,
}
TRIALS_RELATIVE = Path("results/trials.csv")
FRONTS_RELATIVE = Path("results/endpoint_analysis/primary_front_summary.csv")


def _sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _logical_objective(selector: str) -> tuple[str, str, str | None]:
    """Return logical objective, metric, and pairing for one selector."""
    if selector.startswith("cap__"):
        pairing = selector.split("__", maxsplit=1)[1]
        return f"native__cap__{pairing}", "cap", pairing
    return f"native__{selector}", selector, None


def _expected_cells() -> list[dict[str, Any]]:
    """Materialize the authoritative 42-cell native-score selection matrix."""
    cells = []
    for model in MODELS:
        trials = TRIALS_BY_MODEL[model]
        if len(trials) != len(SELECTORS):
            raise RuntimeError(f"Internal selector/trial cardinality differs for {model}.")
        for selector, trial_number in zip(SELECTORS, trials):
            objective, metric, pairing = _logical_objective(selector)
            cells.append(
                {
                    "cell_index": len(cells),
                    "model": model,
                    "score": "native",
                    "selector": metric,
                    "pairing": pairing,
                    "logical_objective": objective,
                    "trial_number": trial_number,
                }
            )
    if len(cells) != 42 or len({(row["model"], row["logical_objective"]) for row in cells}) != 42:
        raise RuntimeError("Internal confirmation cell matrix is not exactly 42 unique cells.")
    return cells


def _csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV with a nonempty, unique header."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"CSV has a missing or duplicate header: {path}")
        return list(reader)


def _authenticate_front_cells(path: Path, cells: list[dict[str, Any]]) -> None:
    """Require endpoint evidence to name exactly the authoritative native winners."""
    rows = [row for row in _csv_rows(path) if row.get("score") == "native"]
    indexed: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        indexed[(row.get("model", ""), row.get("logical_objective", ""))].append(row)
    expected_keys = {(row["model"], row["logical_objective"]) for row in cells}
    if set(indexed) != expected_keys or any(len(group) != 1 for group in indexed.values()):
        raise ValueError("Primary-front native cell coverage is not exactly the expected 42.")
    for cell in cells:
        row = indexed[(cell["model"], cell["logical_objective"])][0]
        if int(row["trial_number"]) != cell["trial_number"]:
            raise ValueError(
                "Primary-front winner differs from the frozen confirmation matrix for "
                f"{cell['model']} {cell['logical_objective']}."
            )


def _selected_csv_trials(path: Path, keys: set[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """Extract exact parameter dictionaries for all and only requested trial keys."""
    matches: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in _csv_rows(path):
        key = (row.get("model", ""), int(row.get("trial_number", -1)))
        if key in keys:
            matches[key].append(row)
    if set(matches) != keys or any(len(rows) != 1 for rows in matches.values()):
        raise ValueError("trials.csv does not contain exactly one row for every selected key.")
    selected = {}
    for key in sorted(keys):
        row = matches[key][0]
        if row.get("state") != "COMPLETE" or row.get("usable") != "True":
            raise ValueError(f"Selected discovery trial is not usable COMPLETE: {key}.")
        try:
            params = json.loads(row["params_json"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Selected discovery trial has invalid params_json: {key}."
            ) from error
        if not isinstance(params, dict) or not params:
            raise ValueError(f"Selected discovery trial has no parameter dictionary: {key}.")
        selected[key] = params
    return selected


def _database_trials(
    database: Path, study_name: str, trial_numbers: Iterable[int]
) -> dict[int, dict[str, Any]]:
    """Read requested immutable COMPLETE trials from one Optuna SQLite study."""
    import optuna

    before = _sha256(database)
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{database}")
    requested = set(trial_numbers)
    rows = {}
    for trial in study.get_trials(deepcopy=False):
        if int(trial.number) not in requested:
            continue
        if trial.number in rows:
            raise ValueError(f"Duplicate Optuna trial number {trial.number} in {study_name}.")
        state = getattr(trial.state, "name", str(trial.state))
        values = None if trial.values is None else [float(value) for value in trial.values]
        if state != "COMPLETE" or values is None or not all(map(math.isfinite, values)):
            raise ValueError(f"Selected Optuna trial is not finite COMPLETE: {trial.number}.")
        rows[int(trial.number)] = {"params": dict(trial.params), "values": values}
    if set(rows) != requested:
        raise ValueError(
            f"Optuna study {study_name} lacks selected trials {sorted(requested-set(rows))}."
        )
    if _sha256(database) != before:
        raise RuntimeError(f"Read-only Optuna authentication changed database {database}.")
    return rows


def build_confirmation_matrix(
    discovery_root: Path,
    *,
    execution_commit: str,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
) -> dict[str, Any]:
    """Build the exact authenticated confirmation manifest without writing it."""
    root = discovery_root.expanduser().resolve(strict=True)
    if len(execution_commit) != 40 or any(c not in "0123456789abcdef" for c in execution_commit):
        raise ValueError("execution_commit must be a full lowercase Git SHA-1.")
    if (
        len(training_seeds) not in {3, 5}
        or len(set(training_seeds)) != len(training_seeds)
        or 123 in training_seeds
        or any(not isinstance(seed, int) or seed < 0 for seed in training_seeds)
    ):
        raise ValueError("Confirmation requires three or five distinct seeds other than 123.")
    if len(training_seeds) == 5 and training_seeds != TRAINING_SEEDS + RESERVE_TRAINING_SEEDS:
        raise ValueError("Five-seed confirmation must append the preregistered reserve seeds.")

    trials_path = (root / TRIALS_RELATIVE).resolve(strict=True)
    fronts_path = (root / FRONTS_RELATIVE).resolve(strict=True)
    cells = _expected_cells()
    _authenticate_front_cells(fronts_path, cells)
    keys = {(row["model"], int(row["trial_number"])) for row in cells}
    params_by_key = _selected_csv_trials(trials_path, keys)

    database_sources = {}
    optuna_by_key = {}
    for model in MODELS:
        database = (root / "studies" / f"{model}.db").resolve(strict=True)
        study_name = f"physics_{model}_background_all"
        numbers = sorted(number for candidate_model, number in keys if candidate_model == model)
        database_sha256 = _sha256(database)
        trials = _database_trials(database, study_name, numbers)
        if _sha256(database) != database_sha256:
            raise RuntimeError(f"Optuna database changed during matrix authentication: {database}")
        database_sources[model] = {
            "path": str(database),
            "sha256": database_sha256,
            "study_name": study_name,
        }
        for number, trial in trials.items():
            key = (model, number)
            expected_values = 28 if model in {"ae", "vae"} else 14
            if len(trial["values"]) != expected_values:
                raise ValueError(
                    f"Optuna objective cardinality differs for {key}: "
                    f"expected {expected_values}, found {len(trial['values'])}."
                )
            if trial["params"] != params_by_key[key]:
                raise ValueError(f"trials.csv parameters differ from Optuna for {key}.")
            optuna_by_key[key] = trial

    uses: dict[tuple[str, int], list[int]] = defaultdict(list)
    for cell in cells:
        uses[(cell["model"], cell["trial_number"])].append(cell["cell_index"])
    configurations = []
    config_index_by_key = {}
    for key in sorted(keys, key=lambda item: (MODELS.index(item[0]), item[1])):
        model, number = key
        config_index = len(configurations)
        config_index_by_key[key] = config_index
        configurations.append(
            {
                "configuration_index": config_index,
                "model": model,
                "trial_number": number,
                "params": params_by_key[key],
                "params_sha256": _content_hash(params_by_key[key]),
                "source_database": database_sources[model]["path"],
                "source_database_sha256": database_sources[model]["sha256"],
                "source_study_name": database_sources[model]["study_name"],
                "source_objective_values": optuna_by_key[key]["values"],
                "logical_cell_indices": uses[key],
            }
        )
    observed_by_model = {
        model: sum(row["model"] == model for row in configurations) for model in MODELS
    }
    if len(configurations) != 28 or observed_by_model != EXPECTED_UNIQUE_BY_MODEL:
        raise RuntimeError(
            f"Confirmation deduplication differs: total={len(configurations)}, "
            f"by_model={observed_by_model}."
        )
    for cell in cells:
        cell["configuration_index"] = config_index_by_key[(cell["model"], cell["trial_number"])]

    runs = []
    configurations_by_model = {
        model: [row for row in configurations if row["model"] == model] for model in MODELS
    }
    # Interleave model families so the first scheduler wave is scientifically and
    # operationally representative instead of being dominated by slow AE/VAE jobs.
    for position in range(max(map(len, configurations_by_model.values()))):
        for seed in training_seeds:
            for model in MODELS:
                if position >= len(configurations_by_model[model]):
                    continue
                configuration = configurations_by_model[model][position]
                runs.append(
                    {
                        "run_index": len(runs),
                        "configuration_index": configuration["configuration_index"],
                        "model": configuration["model"],
                        "trial_number": configuration["trial_number"],
                        "seed": seed,
                        "epochs": 200,
                        "segment_boundaries": [40, 80, 120, 160, 200],
                        "run_id": (
                            f"{configuration['model']}__trial{configuration['trial_number']:04d}"
                            f"__seed{seed}"
                        ),
                    }
                )
    expected_runs = 28 * len(training_seeds)
    if len(runs) != expected_runs or len({row["run_id"] for row in runs}) != expected_runs:
        raise RuntimeError(
            f"Confirmation run expansion is not exactly {expected_runs} unique identities."
        )

    payload = {
        "schema_version": 1,
        "kind": "physics_background_native_confirmation_matrix",
        "execution_commit": execution_commit,
        "discovery_root": str(root),
        "training_seeds": list(training_seeds),
        "reserve_training_seeds": list(RESERVE_TRAINING_SEEDS),
        "seed_selection_rule": "fixed mathematical-constant digits before confirmation results",
        "seed_extension_rule": SEED_EXTENSION_RULE,
        "source_trials_csv": {"path": str(trials_path), "sha256": _sha256(trials_path)},
        "source_primary_front_summary": {
            "path": str(fronts_path),
            "sha256": _sha256(fronts_path),
        },
        "source_optuna_databases": database_sources,
        "cardinality": {
            "models": 6,
            "logical_cells": 42,
            "unique_configurations": 28,
            "seeds": len(training_seeds),
            "runs": expected_runs,
        },
        "logical_cells": cells,
        "configurations": configurations,
        "runs": runs,
    }
    payload["content_sha256"] = _content_hash(payload)
    return payload


def _atomic_text(path: Path, text: str) -> Path:
    """Persist immutable campaign metadata with byte-identical idempotence."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace different frozen metadata: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


def freeze_confirmation_matrix(
    discovery_root: Path,
    confirmation_root: Path,
    *,
    execution_commit: str,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
) -> Path:
    """Atomically persist the matrix, allowing only byte-identical idempotence."""
    payload = build_confirmation_matrix(
        discovery_root, execution_commit=execution_commit, training_seeds=training_seeds
    )
    output = confirmation_root.expanduser().resolve() / "selection/native_28_manifest.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return _atomic_text(output, encoded)


def freeze_confirmation_campaign(
    discovery_root: Path,
    confirmation_root: Path,
    *,
    execution_commit: str,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
    execution_preflight: Path | None = None,
) -> dict[str, Path]:
    """Freeze matrix plus compatibility metadata for the segmented retrain runner.

    The compatibility manifest is intentionally named like the historical Pareto manifest because
    the battle-tested runner authenticates that exact path.  Its contents identify this as a
    downstream-screened confirmation, not a new Pareto search, and retain the native matrix as the
    authoritative source.
    """
    discovery = discovery_root.expanduser().resolve(strict=True)
    confirmation = confirmation_root.expanduser().resolve()
    matrix_path = freeze_confirmation_matrix(
        discovery,
        confirmation,
        execution_commit=execution_commit,
        training_seeds=training_seeds,
    )
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    configurations = {int(row["configuration_index"]): row for row in matrix["configurations"]}
    cells = {int(row["cell_index"]): row for row in matrix["logical_cells"]}
    run_index = {
        (int(row["configuration_index"]), int(row["seed"])): int(row["run_index"])
        for row in matrix["runs"]
    }
    retrain_rows = []
    for run in matrix["runs"]:
        configuration = configurations[int(run["configuration_index"])]
        objectives = [
            cells[index]["logical_objective"] for index in configuration["logical_cell_indices"]
        ]
        retrain_rows.append(
            {
                "retrain_index": int(run["run_index"]),
                "configuration_index": int(run["configuration_index"]),
                "model": run["model"],
                "trial_number": int(run["trial_number"]),
                "params": configuration["params"],
                "params_sha256": configuration["params_sha256"],
                "logical_objectives": sorted(objectives),
                "seed": int(run["seed"]),
                "epochs": 200,
                "run_id": run["run_id"],
                "confirmation_matrix_sha256": matrix["content_sha256"],
            }
        )
    memberships = []
    for seed in training_seeds:
        for cell in matrix["logical_cells"]:
            memberships.append(
                {
                    "membership_index": len(memberships),
                    "cell_index": int(cell["cell_index"]),
                    "model": cell["model"],
                    "logical_objective": cell["logical_objective"],
                    "score": cell["score"],
                    "metric": cell["selector"],
                    "strategy": cell["pairing"],
                    "trial_number": int(cell["trial_number"]),
                    "seed": int(seed),
                    "retrain_index": run_index[(int(cell["configuration_index"]), int(seed))],
                }
            )
    compatibility = {
        "schema_version": 2,
        "kind": "physics_background_native_confirmation_retrain_manifest",
        "code_commit": execution_commit,
        "selection_uses_downstream_anomalies": True,
        "selection_note": (
            "Discovery endpoint screening selected these fixed configurations; this is a "
            "multi-seed 200-epoch stability confirmation, not a new Optuna search."
        ),
        "confirmation_matrix": str(matrix_path.resolve()),
        "confirmation_matrix_file_sha256": _sha256(matrix_path),
        "confirmation_matrix_content_sha256": matrix["content_sha256"],
        "front_memberships": memberships,
        "retrain_runs": retrain_rows,
    }
    compatibility["content_sha256"] = _content_hash(compatibility)
    compatibility_path = confirmation / "selection/pareto_retrain_manifest.json"
    _atomic_text(
        compatibility_path,
        json.dumps(compatibility, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )

    source_design_path = discovery / "design.json"
    source_design = json.loads(source_design_path.read_text(encoding="utf-8"))
    if source_design.get("schema_version") != 2:
        raise ValueError("Discovery campaign design is not schema version 2.")
    execution_preflight = (
        Path(__file__).resolve().parents[2] / "research/physics_background_pairing/preflight.json"
        if execution_preflight is None
        else execution_preflight
    ).resolve(strict=True)
    if _sha256(execution_preflight) != source_design.get("preflight_sha256"):
        raise ValueError("Execution preflight differs from the frozen discovery inputs.")
    design = {
        **source_design,
        "code_commit": execution_commit,
        "campaign_kind": "physics_background_native_confirmation",
        "selection_uses_downstream_anomalies": True,
        "source_discovery_root": str(discovery),
        "source_discovery_design_sha256": _sha256(source_design_path),
        "source_discovery_preflight": source_design["preflight"],
        "preflight": str(execution_preflight),
        "preflight_sha256": _sha256(execution_preflight),
        "confirmation_matrix": str(matrix_path.resolve()),
        "confirmation_matrix_sha256": _sha256(matrix_path),
        "runner_manifest": str(compatibility_path.resolve()),
        "runner_manifest_sha256": _sha256(compatibility_path),
        "training_seeds": list(training_seeds),
        "reserve_training_seeds": list(RESERVE_TRAINING_SEEDS),
        "seed_extension_rule": SEED_EXTENSION_RULE,
    }
    design.pop("content_sha256", None)
    design["content_sha256"] = _content_hash(design)
    design_path = confirmation / "design.json"
    _atomic_text(design_path, json.dumps(design, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {
        "design": design_path,
        "matrix": matrix_path,
        "runner_manifest": compatibility_path,
    }


def load_authenticated_confirmation_campaign(confirmation_root: Path) -> dict[str, Any]:
    """Rebuild and authenticate every immutable confirmation control-plane file."""
    root = confirmation_root.expanduser().resolve(strict=True)
    design_path = root / "design.json"
    matrix_path = root / "selection/native_28_manifest.json"
    runner_path = root / "selection/pareto_retrain_manifest.json"
    for path in (design_path, matrix_path, runner_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    for label, value in (("design", design), ("matrix", matrix), ("runner", runner)):
        digest = value.get("content_sha256")
        unhashed = {key: item for key, item in value.items() if key != "content_sha256"}
        if not isinstance(digest, str) or digest != _content_hash(unhashed):
            raise ValueError(f"Frozen confirmation {label} content hash differs.")
    if (
        design.get("schema_version") != 2
        or design.get("campaign_kind") != "physics_background_native_confirmation"
        or matrix.get("execution_commit") != design.get("code_commit")
        or runner.get("code_commit") != design.get("code_commit")
        or runner.get("confirmation_matrix_file_sha256") != _sha256(matrix_path)
        or runner.get("confirmation_matrix_content_sha256") != matrix.get("content_sha256")
        or design.get("confirmation_matrix_sha256") != _sha256(matrix_path)
        or design.get("runner_manifest_sha256") != _sha256(runner_path)
    ):
        raise ValueError("Frozen confirmation control-plane identities differ.")
    rebuilt = build_confirmation_matrix(
        Path(matrix["discovery_root"]),
        execution_commit=str(matrix["execution_commit"]),
        training_seeds=tuple(int(seed) for seed in matrix["training_seeds"]),
    )
    if rebuilt != matrix:
        raise ValueError("Frozen confirmation matrix differs from authenticated discovery data.")
    expected_runs = 28 * len(matrix["training_seeds"])
    expected_memberships = 42 * len(matrix["training_seeds"])
    if (
        len(runner.get("retrain_runs", [])) != expected_runs
        or len(runner.get("front_memberships", [])) != expected_memberships
    ):
        raise ValueError("Frozen runner manifest has incorrect run/membership cardinality.")
    if [row.get("retrain_index") for row in runner["retrain_runs"]] != list(range(expected_runs)):
        raise ValueError("Frozen runner indices are not exact and contiguous.")
    configurations = {int(row["configuration_index"]): row for row in matrix["configurations"]}
    matrix_runs = {int(row["run_index"]): row for row in matrix["runs"]}
    runner_runs = {int(row["retrain_index"]): row for row in runner["retrain_runs"]}
    for index, run in matrix_runs.items():
        projected = runner_runs.get(index)
        configuration = configurations[int(run["configuration_index"])]
        expected = {
            "configuration_index": int(run["configuration_index"]),
            "model": run["model"],
            "trial_number": int(run["trial_number"]),
            "seed": int(run["seed"]),
            "epochs": int(run["epochs"]),
            "run_id": run["run_id"],
            "params": configuration["params"],
            "params_sha256": configuration["params_sha256"],
        }
        if projected is None or any(
            projected.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"Frozen runner row differs from matrix run {index}.")
    cells = {int(row["cell_index"]): row for row in matrix["logical_cells"]}
    run_by_configuration_seed = {
        (int(row["configuration_index"]), int(row["seed"])): int(row["run_index"])
        for row in matrix["runs"]
    }
    memberships = runner["front_memberships"]
    expected_keys = {
        (cell_index, int(seed)) for cell_index in cells for seed in matrix["training_seeds"]
    }
    observed_keys = {
        (int(row.get("cell_index", -1)), int(row.get("seed", -1))) for row in memberships
    }
    if observed_keys != expected_keys or len(observed_keys) != len(memberships):
        raise ValueError("Frozen membership cell/seed coverage differs.")
    for membership in memberships:
        cell = cells[int(membership["cell_index"])]
        seed = int(membership["seed"])
        expected_index = run_by_configuration_seed[(int(cell["configuration_index"]), seed)]
        run = runner_runs[expected_index]
        if (
            int(membership.get("retrain_index", -1)) != expected_index
            or membership.get("model") != cell["model"]
            or int(membership.get("trial_number", -1)) != int(cell["trial_number"])
            or membership.get("logical_objective") != cell["logical_objective"]
            or run["seed"] != seed
            or run["configuration_index"] != cell["configuration_index"]
        ):
            raise ValueError(
                f"Frozen membership differs from cell/seed {(cell['cell_index'], seed)}."
            )
    return {"design": design, "matrix": matrix, "runner_manifest": runner}


__all__ = [
    "TRAINING_SEEDS",
    "RESERVE_TRAINING_SEEDS",
    "SEED_EXTENSION_RULE",
    "TRIALS_BY_MODEL",
    "build_confirmation_matrix",
    "freeze_confirmation_matrix",
    "freeze_confirmation_campaign",
    "load_authenticated_confirmation_campaign",
]
