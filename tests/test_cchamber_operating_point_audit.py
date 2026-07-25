from __future__ import annotations

import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import cchamber_operating_point_audit as audit
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


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
        "dataset_files": dataset_files,
        "dataset_tree_sha256": hashlib.sha256(
            audit._canonical_json(dataset_files).encode("utf-8")
        ).hexdigest(),
    }
    campaign_path = root / "campaign.json"
    _write_json(campaign_path, campaign)

    manifest = []
    for index, (model, strategy, seed) in enumerate(
        product(audit.MODELS, audit.STRATEGIES, seeds)
    ):
        checkpoint = root / "checkpoints" / f"{index:03d}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"synthetic-checkpoint-{index}".encode())
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
                "valid_pair_table_sha256": "valid-pairs",
                "test_pair_table_sha256": "test-pairs",
                "mlflow_run_id": f"training-{index}",
            },
        )
    _write_json(root / "selection" / "retrain_manifest.json", manifest)
    return root, audit._sha256(campaign_path)


def test_operating_point_callback_writes_sidecar_without_values_contract(tmp_path) -> None:
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
        {"ascore/full": torch.arange(100, dtype=torch.float32)},
        batch=None,
        batch_idx=0,
    )
    callback.on_validation_epoch_end(trainer, module)

    assert callback.validation_normal_count == 100
    assert module.thres_operational.item() == pytest.approx(99.0)
    assert "thres_operational" in dict(module.named_buffers())


def test_inventory_pins_all_200_marker_and_checkpoint_hashes(tmp_path) -> None:
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
