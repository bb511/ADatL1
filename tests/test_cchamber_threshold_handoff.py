from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import cchamber_operating_point_audit as audit
from src.evaluation.callbacks.efficiency import AnomalyEfficiencyCallback
from tests.test_cchamber_operating_point_audit import _synthetic_campaign


@pytest.fixture(autouse=True)
def _stable_audit_deployment(monkeypatch):
    """Keep synthetic handoff artifacts on one clean, finished deployment."""
    monkeypatch.setattr(
        audit, "_audit_revision", lambda: ("synthetic-audit-commit", "research/main")
    )
    monkeypatch.setattr(audit, "_require_finished_run", lambda *args, **kwargs: None)


def _write_json(path: Path, value) -> None:
    """Write a compact synthetic JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _threshold_artifact(
    inventory_path: Path,
    inventory: dict,
    inventory_sha256: str,
    record: dict,
) -> dict:
    """Build one internally consistent synthetic threshold artifact."""
    ids = list(range(1_000))
    threshold = audit._float32_identity(torch.tensor(0.75, dtype=torch.float32))
    return {
        "schema_version": audit.THRESHOLD_SCHEMA_VERSION,
        "created_at": "2026-07-25T12:00:00+00:00",
        "calibration_only": True,
        "campaign_id": inventory["campaign_id"],
        "manifest_index": record["manifest_index"],
        "model": record["model"],
        "strategy": record["strategy"],
        "seed": record["seed"],
        "candidate_id": record["candidate_id"],
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": inventory_sha256,
        "checkpoint": str(Path(record["checkpoint"]).resolve()),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "audit_code_commit": inventory["audit_code_commit"],
        "audit_code_branch": inventory["audit_code_branch"],
        "split": "validate",
        "dataset": "normal",
        "data_seed": 314159,
        "sample_count": 1_000,
        "sample_ids": ids,
        "sample_ids_sha256": audit.sha256_tensor(torch.tensor(ids, dtype=torch.int64)),
        "normalized_tensor_sha256": inventory["pairing_inventory"]["validation_table"][
            "source_1_sha256"
        ],
        "pair_table": inventory["pairing_inventory"]["validation_table"]["path"],
        "pair_table_source_1_sha256": inventory["pairing_inventory"]["validation_table"][
            "source_1_sha256"
        ],
        "pair_table_sha256": inventory["pairing_inventory"]["validation_table"]["sha256"],
        "quantile": 0.99,
        "interpolation": "higher",
        "comparator": ">=",
        "quantile_rank_zero_based": 990,
        "quantile_rank_one_based": 991,
        "threshold_float32": threshold,
        "checkpoint_threshold_float32": None,
        "checkpoint_threshold_present": False,
        "count_below_threshold": 990,
        "count_equal_threshold": 1,
        "count_above_threshold": 9,
        "triggered_count": 10,
        "achieved_validation_acceptance": 0.01,
        "tie_count_at_threshold": 1,
        "finite_sample_granularity": 0.001,
        "audit_mlflow_tracking_uri": inventory["tracking_uri"],
        "audit_mlflow_run_id": f"calibration-{record['manifest_index']}",
        "audit_mlflow_tags": audit._run_tags(
            "threshold_calibration", inventory, inventory_sha256, record
        ),
    }


def test_float32_threshold_identity_rejects_bit_corruption() -> None:
    """Exact threshold identity must fail after any byte mutation."""
    identity = audit._float32_identity(0.1)
    assert audit._decode_float32_identity(identity).dtype == torch.float32
    corrupted = dict(identity)
    corrupted["little_endian_hex"] = "00000000"
    with pytest.raises(ValueError, match="byte identity"):
        audit._decode_float32_identity(corrupted)


def test_threshold_artifact_rejects_declared_field_tampering(tmp_path) -> None:
    """Every generated threshold field participates in authenticated validation."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    inventory = audit.build_inventory(campaign_root, campaign_hash, inventory_path)
    inventory_hash = audit._sha256(inventory_path)
    record = inventory["records"][0]
    artifact_path = tmp_path / "threshold.json"
    baseline = _threshold_artifact(inventory_path, inventory, inventory_hash, record)
    _write_json(artifact_path, baseline)
    audit._validate_threshold_artifact(
        artifact_path, inventory_path, inventory, inventory_hash, record
    )

    mutations = (
        ("pair_table", "/wrong/pairs.pt", "pair_table"),
        ("pair_table_sha256", "0" * 64, "pair_table_sha256"),
        ("quantile_rank_zero_based", 989, "quantile_rank_zero_based"),
        ("count_equal_threshold", 2, "rank/tie diagnostics"),
        ("achieved_validation_acceptance", 0.02, "rank/tie diagnostics"),
        ("checkpoint_threshold_present", True, "presence declaration"),
        ("audit_code_branch", "wrong", "audit_code_branch"),
        ("inventory_sha256", "1" * 64, "inventory_sha256"),
        ("checkpoint_sha256", "2" * 64, "checkpoint_sha256"),
    )
    for field, value, message in mutations:
        tampered = dict(baseline)
        tampered[field] = value
        _write_json(artifact_path, tampered)
        with pytest.raises(ValueError, match=message):
            audit._validate_threshold_artifact(
                artifact_path,
                inventory_path,
                inventory,
                inventory_hash,
                record,
            )


def test_calibration_mlflow_failure_never_publishes_and_retry_recovers(
    tmp_path, monkeypatch
) -> None:
    """A failed logging attempt may leave staging data but no canonical threshold."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    inventory = audit.build_inventory(campaign_root, campaign_hash, inventory_path)
    inventory_hash = audit._sha256(inventory_path)
    record = inventory["records"][0]
    output_root = tmp_path / "threshold-sidecar"
    payload = _threshold_artifact(inventory_path, inventory, inventory_hash, record)
    monkeypatch.setattr(audit, "_calibration_payload", lambda *args, **kwargs: dict(payload))
    monkeypatch.setattr(audit.mlflow, "set_tracking_uri", lambda *args: None)
    monkeypatch.setattr(audit.mlflow, "set_experiment", lambda *args: None)
    monkeypatch.setattr(audit.mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit.mlflow, "log_params", lambda *args, **kwargs: None)
    attempts = []

    class Run:
        """Minimal MLflow context-managed run."""

        def __init__(self, run_id):
            self.info = SimpleNamespace(run_id=run_id)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def start_run(**kwargs):
        attempts.append(kwargs)
        return Run(f"calibration-attempt-{len(attempts)}")

    metric_attempts = 0

    def log_metrics(values):
        nonlocal metric_attempts
        metric_attempts += 1
        if metric_attempts == 1:
            raise RuntimeError("injected MLflow metric failure")

    monkeypatch.setattr(audit.mlflow, "start_run", start_run)
    monkeypatch.setattr(audit.mlflow, "log_metrics", log_metrics)
    canonical = audit._threshold_artifact_path(output_root, 0)
    with pytest.raises(RuntimeError, match="injected MLflow"):
        audit.calibrate_index(
            inventory_path,
            inventory_hash,
            output_root,
            0,
            accelerator="cpu",
        )
    assert not canonical.exists()

    recovered = audit.calibrate_index(
        inventory_path,
        inventory_hash,
        output_root,
        0,
        accelerator="cpu",
    )
    assert canonical.is_file()
    assert recovered["audit_mlflow_run_id"] == "calibration-attempt-2"


def test_evaluation_mlflow_failure_never_publishes_and_retry_recovers(
    tmp_path, monkeypatch
) -> None:
    """A failed evaluation run cannot strand a canonical completion marker."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    inventory = audit.build_inventory(campaign_root, campaign_hash, inventory_path)
    inventory_hash = audit._sha256(inventory_path)
    output_root = tmp_path / "threshold-sidecar"
    for record in inventory["records"]:
        _write_json(
            audit._threshold_artifact_path(output_root, record["manifest_index"]),
            _threshold_artifact(inventory_path, inventory, inventory_hash, record),
        )
    manifest_path, manifest_hash = audit.freeze_threshold_manifest(
        inventory_path, inventory_hash, output_root
    )
    interventions = json.loads((campaign_root / "campaign.json").read_text(encoding="utf-8"))[
        "interventions"
    ]

    class Loader:
        """Mutable synthetic combined-loader leaf."""

        loader = None

    class DataModule:
        """Minimal test-stage datamodule."""

        loader = object()

        def test_dataloader(self):
            return {
                "normal": Loader(),
                **{name: Loader() for name in interventions},
            }

        def teardown(self, stage):
            assert stage == "test"

    monkeypatch.setattr(
        audit,
        "_compose_for_stage",
        lambda *args, **kwargs: (None, DataModule(), SimpleNamespace()),
    )
    monkeypatch.setattr(audit, "_load_checkpoint_strict", lambda *args: ({}, None))
    monkeypatch.setattr(
        audit,
        "apply_threshold_artifact",
        lambda model, artifact: artifact["threshold_float32"]["bytes_sha256"],
    )
    monkeypatch.setattr(audit.pl, "seed_everything", lambda *args, **kwargs: None)

    class Trainer:
        """Write deterministic callback outputs without running a GPU model."""

        def __init__(self, *, callbacks, **kwargs):
            self.callback = callbacks[0]

        def test(self, **kwargs):
            shared = {
                **self.callback.context,
                "test_normal_count": 1_000,
                "test_normal_sample_ids_sha256": "1" * 64,
                "test_normal_tensor_sha256": self.callback.context[
                    "test_pair_table_source_1_sha256"
                ],
            }
            audit._write_csv(
                self.callback.results_path,
                [
                    {
                        **shared,
                        "intervention": intervention,
                        "metric": metric,
                        "value": 0.5,
                    }
                    for intervention in interventions
                    for metric in ("auprc", "efficiency_operational")
                ],
            )
            low, high = audit._wilson_interval(10, 1_000)
            audit._write_csv(
                self.callback.diagnostics_path,
                [
                    {
                        **shared,
                        "target_fpr": 0.01,
                        "comparator": ">=",
                        "triggered_count": 10,
                        "achieved_test_normal_acceptance": 0.01,
                        "achieved_minus_target_fpr": 0.0,
                        "wilson_95_ci_low": low,
                        "wilson_95_ci_high": high,
                        "finite_sample_granularity": 0.001,
                        "validation_test_sample_overlap_count": 0,
                    }
                ],
            )

    monkeypatch.setattr(audit.pl, "Trainer", Trainer)
    monkeypatch.setattr(audit.mlflow, "set_tracking_uri", lambda *args: None)
    monkeypatch.setattr(audit.mlflow, "set_experiment", lambda *args: None)
    monkeypatch.setattr(audit.mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit.mlflow, "end_run", lambda *args, **kwargs: None)
    runs = []

    def start_run(**kwargs):
        runs.append(kwargs)
        return SimpleNamespace(info=SimpleNamespace(run_id=f"evaluation-attempt-{len(runs)}"))

    metric_attempts = 0

    def log_metrics(values):
        nonlocal metric_attempts
        metric_attempts += 1
        if metric_attempts == 1:
            raise RuntimeError("injected evaluation MLflow failure")

    monkeypatch.setattr(audit.mlflow, "start_run", start_run)
    monkeypatch.setattr(audit.mlflow, "log_metrics", log_metrics)
    canonical = output_root / "evaluation" / "000.json"
    with pytest.raises(RuntimeError, match="injected evaluation"):
        audit.evaluate_index(
            inventory_path,
            inventory_hash,
            manifest_path,
            manifest_hash,
            output_root,
            0,
            accelerator="cpu",
        )
    assert not canonical.exists()

    recovered = audit.evaluate_index(
        inventory_path,
        inventory_hash,
        manifest_path,
        manifest_hash,
        output_root,
        0,
        accelerator="cpu",
    )
    assert canonical.is_file()
    assert recovered["audit_mlflow_run_id"] == "evaluation-attempt-2"


def test_threshold_freeze_requires_all_240_immutable_artifacts(tmp_path) -> None:
    """The pre-test gate must authenticate all 240 threshold artifacts."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    inventory = audit.build_inventory(campaign_root, campaign_hash, inventory_path)
    inventory_hash = audit._sha256(inventory_path)
    output_root = tmp_path / "threshold-sidecar"
    for record in inventory["records"]:
        _write_json(
            audit._threshold_artifact_path(output_root, record["manifest_index"]),
            _threshold_artifact(inventory_path, inventory, inventory_hash, record),
        )

    manifest_path, manifest_hash = audit.freeze_threshold_manifest(
        inventory_path, inventory_hash, output_root
    )
    manifest, records = audit._load_threshold_manifest(
        manifest_path, manifest_hash, inventory_hash
    )
    assert manifest["test_or_intervention_data_loaded_before_freeze"] is False
    assert len(records) == 240

    audit._threshold_artifact_path(output_root, 199).unlink()
    with pytest.raises(FileNotFoundError):
        audit.freeze_threshold_manifest(inventory_path, inventory_hash, output_root)


def test_strict_checkpoint_load_then_threshold_injection(tmp_path) -> None:
    """Threshold injection must occur only after strict checkpoint restoration."""

    class Module(torch.nn.Module):
        """Minimal checkpoint-compatible model."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

    checkpoint = tmp_path / "without-threshold.ckpt"
    torch.save({"state_dict": {"weight": torch.tensor(3.0)}}, checkpoint)
    model = Module()
    _, existing = audit._load_checkpoint_strict(model, checkpoint)
    assert existing is None
    artifact = {"threshold_float32": audit._float32_identity(1.25)}
    observed = audit.apply_threshold_artifact(model, artifact)
    assert observed == artifact["threshold_float32"]["bytes_sha256"]
    assert model.weight.item() == pytest.approx(3.0)
    assert model.thres_operational.item() == pytest.approx(1.25)

    with_threshold = tmp_path / "with-threshold.ckpt"
    torch.save(
        {
            "state_dict": {
                "weight": torch.tensor(4.0),
                "thres_operational": torch.tensor(0.5),
            }
        },
        with_threshold,
    )
    model = Module()
    _, existing = audit._load_checkpoint_strict(model, with_threshold)
    assert (
        audit._float32_identity(existing)["bytes_sha256"]
        == audit._float32_identity(0.5)["bytes_sha256"]
    )


def test_threshold_safe_callback_emits_one_provenance_identity_for_all_rows(tmp_path) -> None:
    """All intervention metrics and normal diagnostics must share provenance."""
    interventions = [f"intervention_{index:02d}" for index in range(58)]
    test_inputs = torch.arange(11_000, dtype=torch.float32).reshape(1_000, 11) + 10
    context = {
        "manifest_index": 0,
        "checkpoint_sha256": "checkpoint-sha",
        "threshold_artifact_sha256": "artifact-sha",
        "threshold_bytes_sha256": audit._float32_identity(0.5)["bytes_sha256"],
        "test_pair_table_source_1_sha256": audit.sha256_tensor(test_inputs),
    }
    callback = audit._ThresholdSafeEvaluationCallback(
        interventions,
        tmp_path / "results.csv",
        tmp_path / "diagnostics.csv",
        context,
        validation_sample_ids=range(1_000),
    )
    trainer = SimpleNamespace(
        test_dataloaders={"normal": object(), **{name: object() for name in interventions}}
    )
    module = SimpleNamespace(thres_operational=torch.tensor(0.5))
    callback.on_test_start(trainer, module)
    callback.on_test_epoch_start(trainer, module)
    normal_batch = {"x": test_inputs, "sample_id": torch.arange(1_000, 2_000)}
    callback.on_test_batch_end(
        trainer,
        module,
        {"ascore/full": torch.linspace(0, 1, 1_000)},
        normal_batch,
        0,
        0,
    )
    for index, _ in enumerate(interventions, start=1):
        callback.on_test_batch_end(
            trainer,
            module,
            {"ascore/full": torch.linspace(0.25, 1.25, 1_000)},
            {"x": test_inputs, "sample_id": torch.arange(1_000, 2_000)},
            0,
            index,
        )
    callback.on_test_epoch_end(trainer, module)
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 116
    assert {row["threshold_artifact_sha256"] for row in rows} == {"artifact-sha"}
    assert {row["threshold_bytes_sha256"] for row in rows} == {context["threshold_bytes_sha256"]}
    assert {row["checkpoint_sha256"] for row in rows} == {"checkpoint-sha"}
    with (tmp_path / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        diagnostic = next(csv.DictReader(handle))
    assert int(diagnostic["validation_test_sample_overlap_count"]) == 0
    assert int(diagnostic["test_normal_count"]) == 1_000


def test_evaluation_efficiency_can_require_exact_supplied_threshold(tmp_path) -> None:
    """The generic callback must reject a changed supplied threshold."""
    identity = audit._float32_identity(0.8)
    artifact = tmp_path / "threshold.json"
    _write_json(artifact, {"threshold_float32": identity})
    callback = AnomalyEfficiencyCallback(
        output_name="ascore/full",
        ds=[],
        operating_point_diagnostics_path=tmp_path / "diagnostic.csv",
        operating_point_only=True,
        operating_point_threshold_artifact=artifact,
        operating_point_threshold_artifact_sha256=audit._sha256(artifact),
        operating_point_threshold_bytes_sha256=identity["bytes_sha256"],
        operating_point_require_supplied_threshold=True,
    )
    trainer = SimpleNamespace(test_dataloaders={"normal": object()})
    module = SimpleNamespace(
        device=torch.device("cpu"),
        hparams=SimpleNamespace(target_rate=0.01, base_rate=None),
        thres_operational=torch.tensor(0.8),
    )
    callback.on_test_start(trainer, module)
    module.thres_operational = torch.tensor(0.7)
    with pytest.raises(ValueError, match="bytes differ"):
        callback.on_test_start(trainer, module)

    module.thres_operational = torch.tensor(0.8)
    _write_json(artifact, {"threshold_float32": audit._float32_identity(0.9)})
    with pytest.raises(ValueError, match="artifact SHA-256 differs"):
        callback.on_test_start(trainer, module)


def test_collection_requires_exact_240_by_58_by_2_coverage(tmp_path) -> None:
    """Collection must require the exact 27,840-row Cartesian result table."""
    campaign_root, campaign_hash = _synthetic_campaign(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    inventory = audit.build_inventory(campaign_root, campaign_hash, inventory_path)
    inventory_hash = audit._sha256(inventory_path)
    output_root = tmp_path / "threshold-sidecar"
    artifacts = {}
    for record in inventory["records"]:
        artifact_path = audit._threshold_artifact_path(output_root, record["manifest_index"])
        _write_json(
            artifact_path,
            _threshold_artifact(inventory_path, inventory, inventory_hash, record),
        )
        artifacts[record["manifest_index"]] = artifact_path
    manifest_path, manifest_hash = audit.freeze_threshold_manifest(
        inventory_path, inventory_hash, output_root
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = {row["manifest_index"]: row for row in manifest["records"]}
    interventions = [f"intervention_{index:02d}" for index in range(58)]

    for index, record in enumerate(inventory["records"]):
        artifact = json.loads(artifacts[index].read_text(encoding="utf-8"))
        shared = audit._evaluation_context(
            inventory_path,
            inventory_hash,
            inventory,
            record,
            manifest_path,
            manifest_hash,
            artifacts[index],
            frozen[index],
            artifact,
        )
        shared.update(
            {
                "test_normal_count": 1_000,
                "test_normal_sample_ids_sha256": "1" * 64,
                "test_normal_tensor_sha256": shared["test_pair_table_source_1_sha256"],
            }
        )
        rows = [
            {
                **shared,
                "intervention": intervention,
                "metric": metric,
                "value": 0.5,
            }
            for intervention in interventions
            for metric in ("auprc", "efficiency_operational")
        ]
        results = output_root / "synthetic-results" / f"{index:03d}.csv"
        diagnostics = output_root / "synthetic-diagnostics" / f"{index:03d}.csv"
        audit._write_csv(results, rows)
        audit._write_csv(
            diagnostics,
            [
                {
                    **shared,
                    "test_normal_count": 1_000,
                    "validation_test_sample_overlap_count": 0,
                    "comparator": ">=",
                    "target_fpr": 0.01,
                    "triggered_count": 10,
                    "achieved_test_normal_acceptance": 0.01,
                    "achieved_minus_target_fpr": 0.0,
                    "wilson_95_ci_low": audit._wilson_interval(10, 1_000)[0],
                    "wilson_95_ci_high": audit._wilson_interval(10, 1_000)[1],
                    "finite_sample_granularity": 0.001,
                }
            ],
        )
        _write_json(
            output_root / "evaluation" / f"{index:03d}.json",
            {
                **shared,
                "threshold_manifest_sha256": manifest_hash,
                "audit_code_commit": inventory["audit_code_commit"],
                "audit_code_branch": inventory["audit_code_branch"],
                "audit_mlflow_run_id": f"evaluation-{index}",
                "n_rows": 116,
                "results_csv": str(results.resolve()),
                "results_csv_sha256": audit._sha256(results),
                "diagnostics_csv": str(diagnostics.resolve()),
                "diagnostics_csv_sha256": audit._sha256(diagnostics),
            },
        )

    results, diagnostics, seed_summary, provenance = audit.collect_threshold_safe(
        inventory_path,
        inventory_hash,
        manifest_path,
        manifest_hash,
        output_root,
    )
    with results.open(newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 27_840
    with diagnostics.open(newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 240
    with seed_summary.open(newline="", encoding="utf-8") as handle:
        seed_rows = list(csv.DictReader(handle))
    assert len(seed_rows) == 240
    assert {float(row["achieved_minus_target_fpr"]) for row in seed_rows} == {0.0}
    assert json.loads(provenance.read_text(encoding="utf-8"))["expected_result_rows"] == 27_840

    marker = output_root / "evaluation" / "000.json"
    marker_bytes = marker.read_bytes()
    marker_record = json.loads(marker_bytes)
    result_path = Path(marker_record["results_csv"])
    result_bytes = result_path.read_bytes()
    with result_path.open(newline="", encoding="utf-8") as handle:
        tampered_rows = list(csv.DictReader(handle))
    tampered_rows[0]["threshold_comparator"] = ">"
    audit._write_csv(result_path, tampered_rows)
    marker_record["results_csv_sha256"] = audit._sha256(result_path)
    _write_json(marker, marker_record)
    with pytest.raises(ValueError, match="context mismatch"):
        audit.collect_threshold_safe(
            inventory_path,
            inventory_hash,
            manifest_path,
            manifest_hash,
            output_root,
        )
    result_path.write_bytes(result_bytes)
    marker.write_bytes(marker_bytes)

    marker_record = json.loads(marker_bytes)
    diagnostic_path = Path(marker_record["diagnostics_csv"])
    diagnostic_bytes = diagnostic_path.read_bytes()
    with diagnostic_path.open(newline="", encoding="utf-8") as handle:
        tampered_diagnostics = list(csv.DictReader(handle))
    tampered_diagnostics[0]["test_normal_tensor_sha256"] = "2" * 64
    audit._write_csv(diagnostic_path, tampered_diagnostics)
    marker_record["diagnostics_csv_sha256"] = audit._sha256(diagnostic_path)
    _write_json(marker, marker_record)
    with pytest.raises(ValueError, match="fingerprint contract"):
        audit.collect_threshold_safe(
            inventory_path,
            inventory_hash,
            manifest_path,
            manifest_hash,
            output_root,
        )
    diagnostic_path.write_bytes(diagnostic_bytes)
    marker.write_bytes(marker_bytes)

    marker = output_root / "evaluation" / "199.json"
    marker.unlink()
    with pytest.raises(FileNotFoundError):
        audit.collect_threshold_safe(
            inventory_path,
            inventory_hash,
            manifest_path,
            manifest_hash,
            output_root,
        )
