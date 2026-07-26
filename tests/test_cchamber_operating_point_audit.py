from __future__ import annotations

import csv
import hashlib
import inspect
import json
import subprocess  # nosec B404
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import cchamber_operating_point_audit as audit
from src.algorithms import ADLightningModule
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback

_REAL_REQUIRE_FINISHED_RUN = audit._require_finished_run
_REAL_AUDIT_REVISION = audit._audit_revision


@pytest.fixture(autouse=True)
def _stable_audit_deployment(monkeypatch):
    """Keep synthetic inventories on one clean, finished audit deployment."""
    monkeypatch.setattr(
        audit, "_audit_revision", lambda: ("synthetic-audit-commit", "research/main")
    )
    monkeypatch.setattr(audit, "_require_finished_run", lambda *args, **kwargs: None)


def _write_json(path: Path, value) -> None:
    """Write one compact synthetic JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_detached_deployment_has_explicit_revision_identity(monkeypatch) -> None:
    """Immutable detached worktrees freeze a non-empty checkout-mode identity."""
    responses = {
        ("status", "--porcelain"): "",
        ("rev-parse", "HEAD"): "audit-commit",
        ("branch", "--show-current"): "",
    }
    monkeypatch.setattr(audit, "_git", lambda *args: responses[args])
    monkeypatch.setattr(audit, "_audit_revision", _REAL_AUDIT_REVISION)
    assert audit._audit_revision() == ("audit-commit", "DETACHED")


def test_algorithm_validation_hook_supports_sealed_single_loader() -> None:
    """Lightning may omit the loader index for a single calibration loader."""
    parameter = inspect.signature(ADLightningModule.on_validation_batch_end).parameters[
        "dataloader_idx"
    ]
    assert parameter.default == 0


def _synthetic_campaign(tmp_path: Path) -> tuple[Path, str]:
    """Create a complete label-free synthetic source campaign."""
    root = tmp_path / "campaign"
    seeds = list(range(1001, 1011))
    dataset_dir = tmp_path / "data" / "causal_chamber" / "lt_interventions_standard_v1"
    dataset_files = []
    for index in range(59):
        name = "uniform_reference.csv" if index == 0 else f"synthetic_{index:02d}.csv"
        path = dataset_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"x\\n{index}\\n", encoding="utf-8")
        dataset_files.append(
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": audit._sha256(path),
            }
        )
    campaign = {
        "schema_version": 1,
        "campaign_id": "synthetic-frozen-campaign",
        "git_commit": audit.CAMPAIGN_TRAINING_COMMIT,
        "dataset": "lt_interventions_standard_v1",
        "feature_set": "readouts",
        "n_features": 11,
        "models": list(audit.MODELS),
        "strategies": list(audit.STRATEGIES),
        "reporting_seeds": seeds,
        "interventions": [f"intervention_{index:02d}" for index in range(58)],
        "dataset_files": dataset_files,
        "dataset_tree_sha256": hashlib.sha256(
            audit._canonical_json(dataset_files).encode("utf-8")
        ).hexdigest(),
    }
    campaign_path = root / "campaign.json"
    _write_json(campaign_path, campaign)

    selection = root / "selection"
    selection.mkdir(parents=True)
    candidate_metrics = selection / "candidate_metrics.csv"
    candidate_metrics.write_text(
        "model,strategy,candidate_id,seed,value,params_json\n" 'ae,cap_random,000,101,1.0,"{}"\n',
        encoding="utf-8",
    )
    selected_trials = selection / "selected_trials.csv"
    selected_trials.write_text(
        "model,strategy,candidate_id\nae,cap_random,000\n", encoding="utf-8"
    )

    manifest = []
    for index, (model, strategy, seed) in enumerate(
        product(audit.MODELS, audit.STRATEGIES, seeds)
    ):
        checkpoint = root / "checkpoints" / f"{index:03d}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": {"weight": torch.tensor(float(index))}},
            checkpoint,
        )
        item = {
            "model": model,
            "strategy": strategy,
            "seed": seed,
            "candidate_id": "000",
            "pool_sha256": f"{model}-pool",
            "params": {"algorithm.optimizer.lr": 0.001},
        }
        manifest.append(item)
        _write_json(
            root / "retrain_results" / f"{index:03d}.json",
            {
                **item,
                "campaign_id": campaign["campaign_id"],
                "git_commit": audit.CAMPAIGN_TRAINING_COMMIT,
                "manifest_index": index,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": audit._sha256(checkpoint),
                "valid_pair_table_sha256": "pending",
                "test_pair_table_sha256": "pending",
                "mlflow_run_id": f"training-{index}",
            },
        )
    retrain_manifest = selection / "retrain_manifest.json"
    _write_json(retrain_manifest, manifest)
    _write_json(
        selection / "candidate_metrics_provenance.json",
        {
            "campaign": str(campaign_path.resolve()),
            "campaign_sha256": audit._sha256(campaign_path),
            "candidate_metrics": str(candidate_metrics.resolve()),
            "candidate_metrics_sha256": audit._sha256(candidate_metrics),
        },
    )
    _write_json(
        selection / "selection_provenance.json",
        {
            "candidate_metrics": str(candidate_metrics.resolve()),
            "candidate_metrics_sha256": audit._sha256(candidate_metrics),
            "selected_trials_sha256": audit._sha256(selected_trials),
            "retrain_manifest_sha256": audit._sha256(retrain_manifest),
            "n_selected": 20,
            "n_retrains": 200,
            "development_seeds": list(audit.DEVELOPMENT_SEEDS),
            "intervention_labels_used": False,
        },
    )

    pairing = root / "pairing"
    encoder = pairing / "seed_123" / "encoder.ckpt"
    encoder.parent.mkdir(parents=True)
    encoder.write_bytes(b"encoder")
    source_1 = torch.arange(11_000, dtype=torch.float32).reshape(1_000, 11)
    source_2 = source_1 + 1
    encoder_sha = audit._sha256(encoder)

    def pair_table(path: Path, split: str) -> str:
        table = {
            "schema_version": 1,
            "dataset_1": "normal",
            "dataset_2": "reference_normal",
            "split": split,
            "encoder_ckpt": str(encoder),
            "idx_1": torch.arange(1_000),
            "idx_2": torch.arange(1_000),
            "distance": torch.zeros(1_000),
            "rank_1_to_2": torch.zeros(1_000, dtype=torch.long),
            "rank_2_to_1": torch.zeros(1_000, dtype=torch.long),
            "metadata": {
                "n_dataset_1": 1_000,
                "n_dataset_2": 1_000,
                "n_pairs": 1_000,
                "encoder_checkpoint_sha256": encoder_sha,
                "source_1_sha256": audit.sha256_tensor(source_1),
                "source_2_sha256": audit.sha256_tensor(source_2),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(table, path)
        return audit._sha256(path)

    valid_table = pairing / "seed_123" / "validate_pairs.pt"
    test_table = pairing / "seed_123" / "test_pairs.pt"
    valid_sha = pair_table(valid_table, "validate")
    test_sha = pair_table(test_table, "test")
    primary = {
        "campaign_id": campaign["campaign_id"],
        "git_commit": audit.CAMPAIGN_TRAINING_COMMIT,
        "data_seed": 314159,
        "encoder_seed": 123,
        "encoder_checkpoint": str(encoder.resolve()),
        "encoder_checkpoint_sha256": encoder_sha,
        "validation_table": str(valid_table.resolve()),
        "validation_table_sha256": valid_sha,
        "test_table": str(test_table.resolve()),
        "test_table_sha256": test_sha,
    }
    _write_json(
        pairing / "comparison" / "pairing_manifest.json",
        {
            "campaign_id": campaign["campaign_id"],
            "primary_encoder_seed": 123,
            "primary_validation_table": str(valid_table.resolve()),
            "primary_validation_table_sha256": valid_sha,
            "primary_test_table": str(test_table.resolve()),
            "primary_test_table_sha256": test_sha,
            "encoder_runs": [
                primary,
                *[
                    {
                        "campaign_id": campaign["campaign_id"],
                        "git_commit": audit.CAMPAIGN_TRAINING_COMMIT,
                        "data_seed": 314159,
                        "encoder_seed": seed,
                    }
                    for seed in (456, 789, 101112, 131415)
                ],
            ],
        },
    )
    for index in range(200):
        marker_path = root / "retrain_results" / f"{index:03d}.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["valid_pair_table_sha256"] = valid_sha
        marker["test_pair_table_sha256"] = test_sha
        _write_json(marker_path, marker)
    return root, audit._sha256(campaign_path)


def test_operating_point_callback_writes_sidecar_without_values_contract(tmp_path) -> None:
    """Operating-only mode must emit diagnostics without plot/value artifacts."""
    checkpoint = tmp_path / "immutable" / "chosen.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "audit" / "operating_point_diagnostics.csv"
    context = {
        "campaign_id": "synthetic",
        "manifest_index": 7,
        "checkpoint_sha256": audit._sha256(checkpoint),
        "validation_normal_count": 100,
        "validation_threshold_granularity": 0.01,
    }
    callback = AnomalyEfficiencyCallback(
        output_name="ascore/full",
        ds=[],
        operating_point_diagnostics_path=output,
        operating_point_context=context,
        operating_point_only=True,
        operating_point_threshold_source="audit_validation.normal:thres_operational",
    )
    trainer = SimpleNamespace(
        test_dataloaders={"normal": object()},
        num_test_batches=[1],
        split="test",
    )
    module = SimpleNamespace(
        device=torch.device("cpu"),
        hparams=SimpleNamespace(target_rate=0.01, base_rate=None),
        thres_operational=torch.tensor(0.8),
        _ckpt_path=checkpoint,
    )
    callback.on_test_start(trainer, module)
    callback.on_test_epoch_start(trainer, module)
    batch = (torch.zeros(4, 1), torch.zeros(4))
    callback.on_test_batch_end(
        trainer,
        module,
        {"ascore/full": torch.tensor([0.1, 0.8, 0.9, 0.2])},
        batch,
        batch_idx=0,
        dataloader_idx=0,
    )
    callback.on_test_epoch_end(trainer, module)

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["threshold_source"] == "audit_validation.normal:thres_operational"
    assert float(row["validation_derived_threshold"]) == pytest.approx(0.8)
    assert int(row["test_normal_count"]) == 4
    assert int(row["triggered_count"]) == 2
    assert float(row["achieved_test_normal_acceptance"]) == pytest.approx(0.5)
    assert float(row["finite_sample_granularity"]) == pytest.approx(0.25)
    assert not (checkpoint.parent / "plots").exists()
    assert not list(tmp_path.rglob("values.csv"))


def test_validation_threshold_callback_uses_higher_quantile() -> None:
    """Synthetic calibration should set an exact persistent operational buffer."""

    class Module(torch.nn.Module):
        """Minimal module contract used by the calibration callback."""

        def __init__(self) -> None:
            super().__init__()
            self.hparams = SimpleNamespace(target_rate=0.01, base_rate=None)

        @property
        def device(self) -> torch.device:
            """Return the synthetic module device."""
            return torch.device("cpu")

    callback = audit._ValidationThresholdCallback()
    trainer = SimpleNamespace(val_dataloaders={"normal": object()})
    module = Module()
    callback.on_validation_start(trainer, module)
    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(
        trainer,
        module,
        {"ascore/full": torch.arange(1_000, dtype=torch.float32)},
        batch={
            "x": torch.arange(11_000, dtype=torch.float32).reshape(1_000, 11),
            "sample_id": torch.arange(1_000),
        },
        batch_idx=0,
    )
    callback.on_validation_epoch_end(trainer, module)

    assert callback.validation_normal_count == 1_000
    assert module.thres_operational.item() == pytest.approx(990.0)
    assert "thres_operational" in dict(module.named_buffers())


def test_inventory_pins_all_200_marker_and_checkpoint_hashes(tmp_path) -> None:
    """Inventory must pin the complete retrain and checkpoint Cartesian set."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    inventory_path = tmp_path / "audit-inputs.json"

    inventory = audit.build_inventory(campaign_root, campaign_hash, inventory_path)

    assert inventory["expected_records"] == 200
    assert len(inventory["records"]) == 200
    assert [row["manifest_index"] for row in inventory["records"]] == list(range(200))
    assert {(row["model"], row["strategy"], row["seed"]) for row in inventory["records"]} == set(
        product(audit.MODELS, audit.STRATEGIES, range(1001, 1011))
    )
    inventory_hash = audit._sha256(inventory_path)
    loaded = audit.load_inventory(inventory_path, inventory_hash)
    assert loaded["campaign_training_commit"] == audit.CAMPAIGN_TRAINING_COMMIT

    marker = Path(inventory["records"][17]["retrain_marker"])
    marker.write_text(marker.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retrain marker"):
        audit._validate_source_record(inventory["records"][17], inventory["campaign_id"])


def test_collect_requires_and_combines_exact_200_diagnostics(tmp_path) -> None:
    """Legacy diagnostic collection remains exact for backward verification."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    inventory_path = tmp_path / "audit-inputs.json"
    inventory = audit.build_inventory(campaign_root, campaign_hash, inventory_path)
    inventory_hash = audit._sha256(inventory_path)
    output_root = tmp_path / "audit-output"

    for index, record in enumerate(inventory["records"]):
        context = audit._diagnostic_context(
            inventory_path,
            inventory_hash,
            inventory,
            record,
            audit_commit="new-audit-code-commit",
            audit_branch="research/main",
        )
        context["validation_normal_count"] = 1000
        context["validation_threshold_granularity"] = 0.001
        diagnostics = output_root / "synthetic" / f"{index:03d}.csv"
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "checkpoint": record["checkpoint"],
            "checkpoint_stem": Path(record["checkpoint"]).stem,
            "threshold_source": "audit_validation.normal:thres_operational",
            "validation_derived_threshold": 1.25,
            "target_fpr": 0.01,
            "test_normal_count": 1000,
            "test_normal_finite_count": 1000,
            "triggered_count": 10,
            "achieved_test_normal_acceptance": 0.01,
            "nominal_test_normal_acceptance": 0.01,
            "finite_sample_granularity": 0.001,
            "nearest_attainable_test_normal_acceptance": 0.01,
            **context,
        }
        with diagnostics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        marker = {
            "schema_version": 1,
            **context,
            "diagnostics_csv": str(diagnostics.resolve()),
            "diagnostics_csv_sha256": audit._sha256(diagnostics),
            "audit_mlflow_run_id": f"audit-{index}",
        }
        _write_json(output_root / "records" / f"{index:03d}.json", marker)

    combined, provenance = audit.collect(inventory_path, inventory_hash, output_root)
    with combined.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert {row["campaign_training_commit"] for row in rows} == {audit.CAMPAIGN_TRAINING_COMMIT}
    assert {row["audit_code_commit"] for row in rows} == {"new-audit-code-commit"}
    assert json.loads(provenance.read_text(encoding="utf-8"))["expected_records"] == 200

    (output_root / "records" / "199.json").unlink()
    with pytest.raises(FileNotFoundError):
        audit.collect(inventory_path, inventory_hash, output_root)


def test_generated_slurm_workflow_is_resource_exact_and_syntax_valid(tmp_path) -> None:
    """Launchers must pack 200 tasks and freeze before any evaluation."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    inventory_path = tmp_path / "sidecar" / "inventory.json"
    audit.build_inventory(campaign_root, campaign_hash, inventory_path)
    scripts = inventory_path.parent / "slurm"
    calibrate = (scripts / "calibrate_packed.sh").read_text(encoding="utf-8")
    evaluate = (scripts / "evaluate_packed.sh").read_text(encoding="utf-8")
    debug = (scripts / "debug_four_models.sh").read_text(encoding="utf-8")
    workflow = (scripts / "submit_workflow.sh").read_text(encoding="utf-8")
    for packed in (calibrate, evaluate):
        assert "#SBATCH --account=a0166" in packed
        assert "#SBATCH --partition=normal" in packed
        assert "#SBATCH --gpus-per-node=4" in packed
        assert "#SBATCH --ntasks-per-node=4" in packed
        assert "#SBATCH --cpus-per-task=72" in packed
        assert "#SBATCH --mem=440G" in packed
        assert "#SBATCH --time=04:00:00" in packed
        assert "#SBATCH --array=0-49%16" in packed
        assert "--cpus-per-task=72 --gpus-per-node=1 --mem=110G" in packed
    assert "#SBATCH --partition=debug" in debug
    assert "#SBATCH --cpus-per-task=72" in debug
    assert "#SBATCH --array=0-3" in debug
    assert "indices=(0 50 100 150)" in debug
    assert 'dependency="afterok:${calibration_job}"' in workflow
    assert 'dependency="afterok:${freeze_job}"' in workflow
    assert 'dependency="afterok:${evaluation_job}"' in workflow
    for path in scripts.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)  # nosec B603 B607


def test_gpu_execution_guard_requires_slurm_and_cuda(monkeypatch) -> None:
    """GPU stages must never execute directly on a login node."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="inside Slurm"):
        audit._require_slurm_gpu("gpu")
    audit._require_slurm_gpu("cpu")


def test_inventory_authentication_rejects_selection_and_pairing_tampering(tmp_path) -> None:
    """Label-use, seed-set, and pairing-source provenance are authenticated."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    selection_path = campaign_root / "selection" / "selection_provenance.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["intervention_labels_used"] = True
    _write_json(selection_path, selection)
    with pytest.raises(ValueError, match="Selection provenance chain"):
        audit._validate_selection_inventory(campaign_root, campaign_hash)

    selection["intervention_labels_used"] = False
    _write_json(selection_path, selection)
    campaign = json.loads((campaign_root / "campaign.json").read_text(encoding="utf-8"))
    pairing_path = campaign_root / "pairing" / "comparison" / "pairing_manifest.json"
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    pairing["encoder_runs"].append(dict(pairing["encoder_runs"][-1]))
    _write_json(pairing_path, pairing)
    with pytest.raises(ValueError, match="primary seed-123 contract"):
        audit._validate_pairing_inventory(campaign_root, campaign)

    pairing["encoder_runs"].pop()
    pairing["encoder_runs"][0]["data_seed"] = 7
    _write_json(pairing_path, pairing)
    with pytest.raises(ValueError, match="campaign/commit/data-seed"):
        audit._validate_pairing_inventory(campaign_root, campaign)


def test_deployment_and_finished_mlflow_resume_are_exact(monkeypatch) -> None:
    """Execution requires the frozen revision and a FINISHED exactly tagged run."""
    inventory = {
        "audit_code_commit": "frozen-commit",
        "audit_code_branch": "research/main",
    }
    monkeypatch.setattr(audit, "_audit_revision", lambda: ("other-commit", "research/main"))
    with pytest.raises(RuntimeError, match="differs from the frozen"):
        audit._require_deployment(inventory)

    run = SimpleNamespace(
        info=SimpleNamespace(status="RUNNING"),
        data=SimpleNamespace(tags={"stage": "threshold_calibration"}),
    )

    class Client:
        """Minimal MLflow resume client."""

        def __init__(self, tracking_uri):
            assert tracking_uri == "file:/frozen/mlruns"

        def get_run(self, run_id):
            assert run_id == "run-1"
            return run

    monkeypatch.setattr(audit, "MlflowClient", Client)
    with pytest.raises(ValueError, match="not FINISHED"):
        _REAL_REQUIRE_FINISHED_RUN(
            "file:/frozen/mlruns", "run-1", {"stage": "threshold_calibration"}
        )
    run.info.status = "FINISHED"
    with pytest.raises(ValueError, match="tag mismatch"):
        _REAL_REQUIRE_FINISHED_RUN(
            "file:/frozen/mlruns", "run-1", {"stage": "threshold_evaluation"}
        )


def test_wilson_interval_is_seed_level_and_finite() -> None:
    """The 10/1000 seed-level false-positive estimate has a valid Wilson interval."""
    low, high = audit._wilson_interval(10, 1_000)
    assert 0.0 < low < 0.01 < high < 1.0
    with pytest.raises(ValueError, match="positive trials"):
        audit._wilson_interval(0, 0)
