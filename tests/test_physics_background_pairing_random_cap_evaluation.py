"""Focused contracts for offline multi-seed random-CAP evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle  # nosec B403 -- writes synthetic trusted evaluator summaries
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scripts import physics_background_pairing_random_cap_evaluation as runner
from scripts.physics_background_pairing_random_cap_evaluation import (
    _canonical_hash,
    _evaluation_callback_names,
    _marker_valid,
    _structured_value_rows,
    _value_rows,
    _workspace,
    build_confirmation_manifest,
)
from src.evaluation.callbacks import random_cap
from src.evaluation.callbacks.random_cap import RandomBijectionCAP
from src.utils.pairing.random_tables import (
    DATASET_1,
    DATASET_2,
    SEEDS_BY_SPLIT,
    VALIDATION_SEEDS,
    build_random_pair_table,
)
from src.utils.pairing.table import atomic_torch_save, sha256_file


class _Metric:
    """Cheap deterministic replacement for production CAP in callback unit tests."""

    calls = 0

    def __init__(self, **kwargs) -> None:
        del kwargs
        self.value = torch.tensor(0.0)

    def to(self, device):
        del device
        return self

    def update(self, left, right) -> None:
        type(self).calls += 1
        self.value = (left.float() - right.float()).mean()

    def compute(self):
        return self.value


def test_versioned_workspace_is_isolated_and_path_safe(tmp_path: Path) -> None:
    """A code-fixed rerun can use a new immutable namespace without path escape."""
    assert _workspace(tmp_path, "discovery") == tmp_path / "random_cap_evaluation/discovery"
    assert _workspace(tmp_path, "discovery", "discovery_aace145") == (
        tmp_path / "random_cap_evaluation/discovery_aace145"
    )
    with pytest.raises(ValueError, match="Invalid random-CAP workspace"):
        _workspace(tmp_path, "discovery", "../escape")


def test_callback_reuses_scores_across_five_authenticated_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five CAP fits consume one shared pair of collected score/source tensors."""
    source_1 = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    source_2 = source_1 + 100
    tables = []
    for replicate, seed in enumerate(VALIDATION_SEEDS):
        table = build_random_pair_table(source_1, source_2, split="validate", seed=seed)
        path = atomic_torch_save(table, tmp_path / f"seed{seed}.pt")
        tables.append(
            {
                "split": "validate",
                "replicate": replicate,
                "seed": seed,
                "path": str(path),
                "file_sha256": sha256_file(path),
                "permutation_sha256": table["metadata"]["permutation_sha256"],
            }
        )
    callback = RandomBijectionCAP(
        output_name="ascore/full",
        dataset_1=DATASET_1,
        dataset_2=DATASET_2,
        pairing_manifest_path=str(tmp_path / "manifest.json"),
        cap_metric_config={},
        score="native",
        name="random_cap_native",
    )
    callback.device = torch.device("cpu")
    callback.split = "validate"
    callback.manifest = {"tables": tables}
    callback.dataset_1_inputs = [source_1]
    callback.dataset_2_inputs = [source_2]
    callback.dataset_1_scores = [torch.linspace(0, 1, 8)]
    callback.dataset_2_scores = [torch.linspace(1, 2, 8)]
    checkpoint = tmp_path / "store/ds=selected_000__selected.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _Metric.calls = 0
    monkeypatch.setattr(random_cap, "ApproximationCapacity", _Metric)
    callback.on_test_epoch_end(None, SimpleNamespace(_ckpt_path=checkpoint))
    output = checkpoint.parent / "plots/val/ds=selected_000__selected/random_cap_native/values.csv"
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert _Metric.calls == 5
    assert len(rows) == 5
    assert {int(row["seed"]) for row in rows} == set(VALIDATION_SEEDS)
    assert {row["score"] for row in rows} == {"native"}


def test_oas_callback_rejects_checkpoint_without_ready_state(tmp_path: Path) -> None:
    """OAS diagnostics cannot silently consume the model's MSE fallback."""
    callback = RandomBijectionCAP(
        output_name="ascore/residual_oas",
        dataset_1=DATASET_1,
        dataset_2=DATASET_2,
        pairing_manifest_path=str(tmp_path / "random_pairing/manifest.json"),
        cap_metric_config={},
        score="residual_oas",
        name="random_cap_residual_oas",
    )
    module = SimpleNamespace(device=torch.device("cpu"), residual_oas_ready=torch.tensor(False))
    with pytest.raises(RuntimeError, match="OAS-ready checkpoint"):
        callback.on_test_start(SimpleNamespace(test_dataloaders={}), module)


def _task() -> dict:
    """Return one AE checkpoint task requiring native and masked-OAS values."""
    return {
        "task_index": 0,
        "model": "ae",
        "trial_number": 7,
        "scores": ["native", "residual_oas"],
        "checkpoints": [
            {
                "checkpoint_id": "selected_000",
                "source_path": "/checkpoint.ckpt",
                "source_sha256": "a" * 64,
            }
        ],
        "memberships": [],
    }


def _write_values(root: Path, score: str, split: str) -> Path:
    """Write one exact five-replicate callback CSV for marker tests."""
    callback = "random_cap_native" if score == "native" else "random_cap_residual_oas"
    split_folder = "val" if split == "validate" else "test"
    path = (
        root
        / "checkpoint_store/a/b/c/plots"
        / split_folder
        / "ds=selected_000__selected"
        / callback
        / "values.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint",
                "split",
                "score",
                "replicate",
                "seed",
                "metric",
                "value",
                "pair_table_sha256",
                "permutation_sha256",
            ],
        )
        writer.writeheader()
        for replicate in range(5):
            writer.writerow(
                {
                    "checkpoint": "ds=selected_000__selected",
                    "split": split,
                    "score": score,
                    "replicate": replicate,
                    "seed": SEEDS_BY_SPLIT[split][replicate],
                    "metric": "cap",
                    "value": 0.1 + replicate / 100,
                    "pair_table_sha256": f"{replicate + 1:064x}",
                    "permutation_sha256": f"{replicate + 11:064x}",
                }
            )
    return path


def _write_structured_summary(
    root: Path, score: str, strategy: str, split: str, *, cap: float = 0.25
) -> Path:
    """Write one trusted evaluator CAP summary for same-checkpoint tests."""
    split_folder = "val" if split == "validate" else "test"
    path = (
        root
        / "checkpoint_store/a/b/c/plots"
        / split_folder
        / f"cap_{score}_{strategy}_summary"
        / "summary.pkl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"selected_000": cap}, handle)
        pickle.dump({"selected_000": 0.5}, handle)
    return path


def test_marker_requires_complete_score_split_seed_matrix_and_file_hashes(tmp_path: Path) -> None:
    """Missing or changed native/OAS seed-level rows invalidate completion."""
    task = _task()
    for score in task["scores"]:
        for split in ("validate", "test"):
            _write_values(tmp_path, score, split)
    random_pairing = {
        "tables": [
            {
                "split": split,
                "replicate": replicate,
                "seed": SEEDS_BY_SPLIT[split][replicate],
                "file_sha256": f"{replicate + 1:064x}",
                "permutation_sha256": f"{replicate + 11:064x}",
            }
            for split in ("validate", "test")
            for replicate in range(5)
        ]
    }
    values, files = _value_rows(tmp_path, task, random_pairing)
    assert len(values) == 20
    manifest = {"content_sha256": "manifest", "random_pairing": random_pairing}
    marker = {
        "manifest_content_sha256": "manifest",
        "task_sha256": _canonical_hash(task),
        "task_dir": str(tmp_path),
        "values": values,
        "value_files": files,
    }
    assert _marker_valid(manifest, task, marker)
    changed = Path(files[0]["path"])
    changed.write_text(changed.read_text().replace("0.1,", "0.2,"), encoding="utf-8")
    assert not _marker_valid(manifest, task, marker)


def test_confirmation_marker_authenticates_same_checkpoint_structured_cap(
    tmp_path: Path,
) -> None:
    """Native/OAS structured CAP is complete, hash-bound, and never a membership."""
    task = _task()
    task["training_seed"] = 271828
    task["memberships"] = [
        {
            "membership_index": 0,
            "checkpoint_id": "selected_000",
            "logical_objective": "native__cap__flat_physical",
            "score": "native",
            "metric": "cap",
            "strategy": "flat_physical",
        }
    ]
    for score in task["scores"]:
        for split in ("validate", "test"):
            _write_values(tmp_path, score, split)
            _write_structured_summary(tmp_path, score, "flat_physical", split)
    random_pairing = {
        "tables": [
            {
                "split": split,
                "replicate": replicate,
                "seed": SEEDS_BY_SPLIT[split][replicate],
                "file_sha256": f"{replicate + 1:064x}",
                "permutation_sha256": f"{replicate + 11:064x}",
            }
            for split in ("validate", "test")
            for replicate in range(5)
        ]
    }
    values, value_files = _value_rows(tmp_path, task, random_pairing)
    structured, structured_files = _structured_value_rows(tmp_path, task)
    assert len(structured) == 4
    assert {row["score"] for row in structured} == {"native", "residual_oas"}
    assert {row["selection_strategy"] for row in structured} == {"flat_physical"}
    assert len(task["memberships"]) == 1
    marker = {
        "generator": {"commit": "a" * 40, "worktree_clean": True},
        "manifest_content_sha256": "manifest",
        "task_sha256": _canonical_hash(task),
        "task_dir": str(tmp_path),
        "values": values,
        "value_files": value_files,
        "structured_cap_values": structured,
        "structured_cap_value_files": structured_files,
        "structured_cap_values_count": 4,
        "structured_cap_values_sha256": _canonical_hash(structured),
    }
    marker["content_sha256"] = _canonical_hash(marker)
    manifest = {
        "mode": "confirmation",
        "content_sha256": "manifest",
        "generator": {"commit": "a" * 40, "worktree_clean": True},
        "random_pairing": random_pairing,
    }
    assert _marker_valid(manifest, task, marker)
    marker["structured_cap_values_sha256"] = "0" * 64
    assert not _marker_valid(manifest, task, marker)


def test_callback_allowlist_adds_structured_cap_only_for_confirmation() -> None:
    """Discovery execution stays unchanged while confirmation computes matched evidence."""
    task = _task()
    task["memberships"] = [
        {"metric": "cap", "strategy": "cdf"},
        {"metric": "cap", "strategy": "flat_physical"},
        {"metric": "wasserstein", "strategy": ""},
    ]
    assert _evaluation_callback_names(task, "discovery") == (
        "random_cap_native",
        "random_cap_residual_oas",
    )
    assert _evaluation_callback_names(task, "confirmation") == (
        "random_cap_native",
        "random_cap_residual_oas",
        "cap_native_flat_physical",
        "cap_native_cdf",
        "cap_residual_oas_flat_physical",
        "cap_residual_oas_cdf",
    )


def test_confirmation_aggregate_emits_exact_structured_cap_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aggregate rows preserve native selection and OAS evaluation identities."""
    root = tmp_path / "campaign"
    root.mkdir()
    task = _task()
    task.update({"training_seed": 271828, "trial_number": 154})
    task["memberships"] = [
        {
            "membership_index": 7,
            "checkpoint_id": "selected_000",
            "logical_objective": "native__cap__physics_summary",
            "score": "native",
            "metric": "cap",
            "strategy": "physics_summary",
        }
    ]
    task_dir = _workspace(root, "confirmation") / "tasks/000"
    for score in task["scores"]:
        for split in ("validate", "test"):
            _write_values(task_dir, score, split)
            _write_structured_summary(task_dir, score, "physics_summary", split, cap=0.75)
    random_pairing = {
        "tables": [
            {
                "split": split,
                "replicate": replicate,
                "seed": SEEDS_BY_SPLIT[split][replicate],
                "file_sha256": f"{replicate + 1:064x}",
                "permutation_sha256": f"{replicate + 11:064x}",
            }
            for split in ("validate", "test")
            for replicate in range(5)
        ]
    }
    manifest = {
        "mode": "confirmation",
        "content_sha256": "manifest",
        "generator": {"commit": "a" * 40, "worktree_clean": True},
        "random_pairing": random_pairing,
        "tasks": [task],
    }
    values, value_files = _value_rows(task_dir, task, random_pairing)
    structured, structured_files = _structured_value_rows(task_dir, task)
    marker = {
        "generator": manifest["generator"],
        "manifest_content_sha256": "manifest",
        "task_sha256": _canonical_hash(task),
        "task_dir": str(task_dir),
        "values": values,
        "value_files": value_files,
        "structured_cap_values": structured,
        "structured_cap_value_files": structured_files,
        "structured_cap_values_count": len(structured),
        "structured_cap_values_sha256": _canonical_hash(structured),
    }
    marker_path = task_dir / "marker.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    monkeypatch.setattr(runner, "_load_manifest", lambda *_: manifest)
    runner.aggregate(SimpleNamespace(root=root, mode="confirmation", namespace=None))
    payload = json.loads((_workspace(root, "confirmation") / "aggregate.json").read_text())
    assert payload["generator"] == manifest["generator"]
    assert payload["content_sha256"] == _canonical_hash(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    rows = payload["structured_cap_values"]
    assert len(rows) == payload["structured_cap_values_count"] == 4
    assert payload["structured_cap_values_sha256"] == _canonical_hash(rows)
    assert set(rows[0]) == {
        "mode",
        "task_index",
        "membership_index",
        "model",
        "training_seed",
        "logical_objective",
        "selection_score",
        "evaluation_score",
        "selection_metric",
        "selection_strategy",
        "split",
        "checkpoint_id",
        "checkpoint_sha256",
        "cap",
    }
    assert {row["selection_score"] for row in rows} == {"native"}
    assert {row["evaluation_score"] for row in rows} == {"native", "residual_oas"}
    assert {row["logical_objective"] for row in rows} == {"native__cap__physics_summary"}


def test_diagnostic_configs_cannot_optimize_or_checkpoint() -> None:
    """Random CAP configs are evaluation-only and define no selection callback."""
    root = Path(__file__).parents[1] / "configs/diagnostic_suite"
    native = OmegaConf.load(root / "physics_background_random_cap_native.yaml")
    oas = OmegaConf.load(root / "physics_background_random_cap_native_oas.yaml")
    assert native.optimized_metric_config is None
    assert native.optimized_metric_configs is None
    assert native.optimized_metric_artifact is False
    assert native.random_cap_is_diagnostic_only is True
    assert set(native.evaluation.callbacks) == {"random_cap_native"}
    assert set(oas.evaluation.callbacks) == {"random_cap_residual_oas"}
    assert not hasattr(RandomBijectionCAP, "get_optimized_metric")


def test_confirmation_manifest_deduplicates_126_sealed_cell_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sealed 42-cell/three-seed selections become 84 hash-bound offline tasks."""
    root = tmp_path / "confirmation"
    root.mkdir()
    repository = tmp_path / "repository"
    repository.mkdir()
    commit = "a" * 40
    seeds = (271828, 314159, 161803)
    configurations = [
        {
            "configuration_index": index,
            "model": "ae",
            "trial_number": 100 + index,
            "params": {"algorithm.lr": 0.001 + index / 100000},
        }
        for index in range(28)
    ]
    cells = [
        {
            "cell_index": index,
            "configuration_index": index % 28,
            "model": "ae",
            "trial_number": 100 + index % 28,
            "logical_objective": f"native__cap__pairing_{index}",
            "selector": "cap",
            "pairing": f"pairing_{index}",
        }
        for index in range(42)
    ]
    runs = []
    run_index = {}
    for configuration in configurations:
        for seed in seeds:
            index = len(runs)
            run_index[(configuration["configuration_index"], seed)] = index
            runs.append(
                {
                    "retrain_index": index,
                    "configuration_index": configuration["configuration_index"],
                    "model": "ae",
                    "trial_number": configuration["trial_number"],
                    "seed": seed,
                }
            )
    memberships = []
    selection_rows = []
    checkpoint_by_run = {}
    for seed in seeds:
        for cell in cells:
            index = run_index[(cell["configuration_index"], seed)]
            if index not in checkpoint_by_run:
                checkpoint = root / "runtime/checkpoints" / f"run_{index:03d}.ckpt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"checkpoint-{index}".encode())
                checkpoint_by_run[index] = checkpoint
            checkpoint = checkpoint_by_run[index]
            memberships.append(
                {
                    "membership_index": len(memberships),
                    "cell_index": cell["cell_index"],
                    "seed": seed,
                    "retrain_index": index,
                    "logical_objective": cell["logical_objective"],
                }
            )
            selection_rows.append(
                {
                    "cell_index": cell["cell_index"],
                    "model": "ae",
                    "trial_number": cell["trial_number"],
                    "seed": seed,
                    "logical_objective": cell["logical_objective"],
                    "checkpoint_family_id": "last",
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "selection_metric": json.dumps([0.2 + cell["cell_index"] / 1000, 0.1]),
                }
            )
    selections = root / "results/checkpoint_selections.csv"
    selections.parent.mkdir(parents=True)
    with selections.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    receipt = root / "results/endpoint_extraction.json"
    receipt.write_text(
        json.dumps(
            {
                "campaign_root": str(root),
                "artifacts": {
                    "checkpoint_selections.csv": hashlib.sha256(
                        selections.read_bytes()
                    ).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    frozen = {
        "design": {"code_commit": commit},
        "matrix": {
            "content_sha256": "matrix",
            "discovery_root": str(tmp_path / "discovery"),
            "training_seeds": list(seeds),
            "configurations": configurations,
            "logical_cells": cells,
        },
        "runner_manifest": {
            "content_sha256": "runner",
            "retrain_runs": runs,
            "front_memberships": memberships,
        },
    }
    (tmp_path / "discovery").mkdir()
    monkeypatch.setattr(runner, "load_authenticated_confirmation_campaign", lambda _: frozen)
    monkeypatch.setattr(
        runner,
        "_authenticate_execution",
        lambda repo, expected_commit=None: (repo.resolve(), commit),
    )
    monkeypatch.setattr(
        runner,
        "_load_prepared",
        lambda _: {
            "content_sha256": "discovery",
            "inputs": {"data_dir": "/data", "raw_data_dir": "/raw", "pairing_dir": "/pairs"},
        },
    )
    monkeypatch.setattr(runner, "_table_identity", lambda *_: {"content_sha256": "tables"})
    manifest = build_confirmation_manifest(root, repository, selections, receipt)
    assert len(manifest["tasks"]) == 84
    assert sum(len(task["memberships"]) for task in manifest["tasks"]) == 126
    assert sum(len(task["checkpoints"]) for task in manifest["tasks"]) == 84
    assert all(task["scores"] == ["native", "residual_oas"] for task in manifest["tasks"])
    assert all(
        membership["selection_metric_name"] == "cap"
        and membership["selection_metric_direction"] == "maximize"
        and membership["optimized_metric"] == membership["objective_values"]
        for task in manifest["tasks"]
        for membership in task["memberships"]
    )

    # Offline diagnostics have their own immutable execution identity; they need not
    # pretend to run from the older commit that trained the sealed checkpoints.
    frozen["design"]["code_commit"] = "b" * 40
    frozen["reserve_extension"] = {
        "execution_commit": commit,
        "content_sha256": "extension",
    }
    reserve_manifest = build_confirmation_manifest(root, repository, selections, receipt)
    assert reserve_manifest["execution_commit"] == commit
    assert reserve_manifest["reserve_extension_content_sha256"] == "extension"
