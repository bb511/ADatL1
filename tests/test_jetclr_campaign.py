from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import jetclr_campaign


def _pairing_metrics(collapse_pass: bool = True) -> dict:
    """Return a complete synthetic pairing artifact."""
    return {
        "selection_score": 0.5,
        "raw_selection_score": 0.5,
        "closure_recall_at_10": 0.8,
        "mnn_coverage": 0.7,
        "embedding_finite_fraction": 1.0,
        "embedding_active_fraction": 0.9,
        "embedding_effective_rank": 40.0,
        "embedding_participation_rank": 35.0,
        "embedding_top_pc_fraction": 0.1,
        "collapse_pass": collapse_pass,
        "collapse_failures": [] if collapse_pass else ["low_effective_rank"],
    }


def _anomaly_metrics(value: float = 0.8) -> dict:
    """Return a complete synthetic anomaly-utility artifact."""
    return {
        "macro_median_auroc": value,
        "macro_mean_auroc": value - 0.01,
        "worst_quartile_mean_auroc": value - 0.1,
        "per_dataset": {"signal": {"auroc": value, "auprc": 0.4}},
    }


def _write_manifest(root: Path) -> dict:
    """Write a minimal authenticated campaign manifest for unit tests."""
    specs = jetclr_campaign.canary_specs()
    stage1 = jetclr_campaign.stage1_specs()
    manifest = {
        "schema_version": 1,
        "campaign_id": "jetclr_test_deadbeef",
        "git": {"commit": "deadbeef", "branch": "test", "source": "/source"},
        "deployment": {"path": "/deployment", "commit": "deadbeef"},
        "config": {"tree_sha256": "config"},
        "data": {"root": "/data", "tree_sha256": "data"},
        "environment": {"uv": "/uv", "venv": "/venv"},
        "canary": {
            "trials": specs,
            "design_sha256": jetclr_campaign._value_sha256(specs),
        },
        "stage1": {
            "seed": jetclr_campaign.STAGE1_SEED,
            "train_batches": jetclr_campaign.STAGE1_TRAIN_BATCHES,
            "candidates": stage1,
            "design_sha256": jetclr_campaign._value_sha256(stage1),
        },
    }
    manifest["manifest_payload_sha256"] = jetclr_campaign._value_sha256(manifest)
    jetclr_campaign._atomic_json(root / "campaign.json", manifest)
    return manifest


def test_canary_specs_are_deterministic_and_cover_four_distinct_recipes() -> None:
    """The canary design must be stable and exercise distinct risk surfaces."""
    first = jetclr_campaign.canary_specs()
    second = jetclr_campaign.canary_specs()

    assert first == second
    assert [item["trial_id"] for item in first] == [0, 1, 2, 3]
    assert len({item["name"] for item in first}) == 4
    assert len({item["spec_sha256"] for item in first}) == 4
    assert any("algorithm.detector_smearing=null" in item["overrides"] for item in first)
    assert any("algorithm.model.d_model=256" in item["overrides"] for item in first)


def test_canary_disables_evaluation_callbacks_without_deleting_config_group() -> None:
    """The physics experiment's evaluation group must remain Hydra-composable."""
    source = Path(jetclr_campaign.__file__).read_text(encoding="utf-8")

    assert '"evaluation.callbacks=null"' in source
    assert '"evaluation=null"' not in source
    assert '"extras.enforce_tags=false"' in source
    assert '"callbacks.rich_progress_bar=null"' in source


def test_stage1_specs_are_fixed_sobol_design_with_anchor_ablations() -> None:
    """Stage 1 must reproducibly combine eight anchors and forty Sobol points."""
    first = jetclr_campaign.stage1_specs()
    second = jetclr_campaign.stage1_specs()

    assert first == second
    assert len(first) == 48
    assert [item["candidate_id"] for item in first] == list(range(48))
    assert {item["kind"] for item in first[:8]} == {"anchor"}
    assert {item["kind"] for item in first[8:]} == {"sobol"}
    assert {item["seed"] for item in first} == {123}
    assert {item["train_batches"] for item in first} == {256}
    assert len({item["spec_sha256"] for item in first}) == 48
    assert first[0]["name"] == "production"
    assert first[4]["name"] == "no_augmentation"


def test_stage1_launchers_pack_48_trials_and_chain_collector(tmp_path: Path) -> None:
    """Twelve array tasks should pack four trials and expose an afterok collector."""
    manifest = _write_manifest(tmp_path)
    launchers = jetclr_campaign._write_stage1_launchers(tmp_path, manifest)
    stage1 = launchers["stage1"].read_text(encoding="utf-8")
    collector = launchers["collector"].read_text(encoding="utf-8")
    submitter = launchers["submitter"].read_text(encoding="utf-8")

    assert "#SBATCH --partition=normal" in stage1
    assert "#SBATCH --array=0-11%4" in stage1
    assert "base=$((SLURM_ARRAY_TASK_ID * 4))" in stage1
    assert "run-stage1" in stage1
    assert "#SBATCH --gpus-per-node" not in collector
    assert "collect-stage1" in collector
    assert 'dependency="afterok:$stage1_job"' in submitter


def test_run_stage1_composes_bounded_training_and_validates_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One trial must use the frozen budget and parse all three metric artifacts."""
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(jetclr_campaign, "_assert_runtime", lambda _: tmp_path)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        output = tmp_path / "stage1" / "candidate_000" / "output"
        metrics = output / "csv" / "version_0" / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("train/loss_mean\n2.5\n", encoding="utf-8")
        pairing = output / "metrics" / "pairing_diagnostics" / "last"
        pairing.mkdir(parents=True)
        (pairing / "pairing_diagnostics.json").write_text(
            json.dumps(_pairing_metrics()), encoding="utf-8"
        )
        anomaly = output / "metrics" / "embedding_anomaly" / "last"
        anomaly.mkdir(parents=True)
        (anomaly / "embedding_anomaly.json").write_text(
            json.dumps(_anomaly_metrics()), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jetclr_campaign.subprocess, "run", fake_run)
    result_path = jetclr_campaign.run_stage1(tmp_path, 0)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    command = observed["command"]

    assert "+trainer.limit_train_batches=256" in command
    assert "+trainer.limit_val_batches=0" in command
    assert "data.max_val_batches=4" in command
    assert "data.max_normal_eval_batches=8" in command
    assert "evaluation.callbacks.pairing_diagnostics.max_events_per_dataset=8192" in command
    assert "evaluation.callbacks.embedding_anomaly.reference_size=8192" in command
    assert "evaluation.callbacks.embedding_anomaly.max_query_events=8192" in command
    assert "callbacks.rich_progress_bar=null" in command
    assert "extras.enforce_tags=false" in command
    assert (tmp_path / "stage1" / "candidate_000" / "output").is_dir()
    assert result["pairing_metrics"]["collapse_pass"] is True
    assert result["anomaly_metrics"]["worst_quartile_mean_auroc"] == pytest.approx(0.7)


def test_campaign_manifest_detects_tampering(tmp_path: Path) -> None:
    """Changing any authenticated manifest field must fail closed."""
    manifest = _write_manifest(tmp_path)
    assert jetclr_campaign._load_campaign(tmp_path)["campaign_id"] == manifest["campaign_id"]

    path = tmp_path / "campaign.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign_id"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        jetclr_campaign._load_campaign(tmp_path)


def test_packed_launcher_assigns_one_gpu_to_each_trial(tmp_path: Path) -> None:
    """The launcher must use every GPU without recursively submitting work."""
    manifest = _write_manifest(tmp_path)
    launcher = jetclr_campaign._write_launcher(tmp_path, manifest)
    text = launcher.read_text(encoding="utf-8")

    assert "#SBATCH --gpus-per-node=4" in text
    assert "#SBATCH --ntasks=4" in text
    assert "for trial_id in 0 1 2 3" in text
    assert "srun --exclusive --ntasks=1" in text
    assert "--gpus-per-node=1" in text
    assert "sbatch " not in text


def test_collect_validates_and_summarizes_all_trials(tmp_path: Path) -> None:
    """Collection should authenticate and summarize four complete trials."""
    manifest = _write_manifest(tmp_path)
    for spec in manifest["canary"]["trials"]:
        trial_root = tmp_path / "canary" / f"{spec['trial_id']:02d}_{spec['name']}"
        metrics = trial_root / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        with metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train/loss_mean"])
            writer.writeheader()
            writer.writerow({"train/loss_mean": 1.0 + spec["trial_id"]})
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "spec_sha256": spec["spec_sha256"],
            "metrics_csv": str(metrics),
            "metrics_csv_sha256": jetclr_campaign._sha256(metrics),
            "metrics": {"train/loss_mean": 1.0 + spec["trial_id"]},
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial_root / "result.json", result)

    output = jetclr_campaign.collect(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["n_trials"] == 4
    assert jetclr_campaign._sha256(Path(summary["summary_csv"])) == summary["summary_csv_sha256"]


def test_collect_rejects_modified_metrics(tmp_path: Path) -> None:
    """Collection must reject metrics modified after trial completion."""
    manifest = _write_manifest(tmp_path)
    spec = manifest["canary"]["trials"][0]
    trial_root = tmp_path / "canary" / f"{spec['trial_id']:02d}_{spec['name']}"
    metrics = trial_root / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("train/loss_mean\n1.0\n", encoding="utf-8")
    result = {
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "spec_sha256": spec["spec_sha256"],
        "metrics_csv": str(metrics),
        "metrics_csv_sha256": "0" * 64,
        "metrics": {"train/loss_mean": 1.0},
    }
    result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
    jetclr_campaign._atomic_json(trial_root / "result.json", result)
    with pytest.raises(ValueError, match="Metrics artifact mismatch"):
        jetclr_campaign.collect(tmp_path)


@pytest.mark.parametrize("all_collapsed", [False, True])
def test_collect_stage1_authenticates_ranks_and_applies_collapse_gate(
    tmp_path: Path, all_collapsed: bool
) -> None:
    """Collection should rank eligible utility while retaining collapsed candidates."""
    manifest = _write_manifest(tmp_path)
    for spec in manifest["stage1"]["candidates"]:
        trial_root = tmp_path / "stage1" / f"candidate_{spec['candidate_id']:03d}"
        trial_root.mkdir(parents=True)
        metrics = trial_root / "metrics.csv"
        pairing_path = trial_root / "pairing_diagnostics.json"
        anomaly_path = trial_root / "embedding_anomaly.json"
        metrics.write_text("train/loss_mean\n2.0\n", encoding="utf-8")
        collapse = False if all_collapsed else spec["candidate_id"] != 1
        pairing = _pairing_metrics(collapse)
        anomaly = _anomaly_metrics(0.99 if spec["candidate_id"] == 1 else 0.75)
        pairing_path.write_text(json.dumps(pairing), encoding="utf-8")
        anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
        artifacts = {
            "training_csv": {
                "path": str(metrics),
                "sha256": jetclr_campaign._sha256(metrics),
            },
            "pairing_json": {
                "path": str(pairing_path),
                "sha256": jetclr_campaign._sha256(pairing_path),
            },
            "anomaly_json": {
                "path": str(anomaly_path),
                "sha256": jetclr_campaign._sha256(anomaly_path),
            },
        }
        result = {
            "campaign_id": manifest["campaign_id"],
            "git_commit": manifest["git"]["commit"],
            "candidate_id": spec["candidate_id"],
            "spec_sha256": spec["spec_sha256"],
            "artifacts": artifacts,
            "training_metrics": {"train/loss_mean": 2.0},
            "pairing_metrics": pairing,
            "anomaly_metrics": anomaly,
        }
        result["result_payload_sha256"] = jetclr_campaign._value_sha256(result)
        jetclr_campaign._atomic_json(trial_root / "result.json", result)

    output = jetclr_campaign.collect_stage1(tmp_path)
    summary = json.loads(output.read_text(encoding="utf-8"))
    with Path(summary["summary_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["n_candidates"] == 48
    expected_eligible = 0 if all_collapsed else 47
    assert summary["n_collapse_pass"] == expected_eligible
    assert summary["n_eligible"] == expected_eligible
    if all_collapsed:
        assert summary["status"] == "complete_no_eligible_candidates"
        assert summary["best_candidate_id"] is None
    else:
        assert summary["status"] == "complete"
        assert summary["best_candidate_id"] != 1
    collapsed = next(row for row in rows if row["candidate_id"] == "1")
    assert collapsed["eligible"] == "False"
    assert float(collapsed["worst_quartile_mean_auroc"]) == pytest.approx(0.89)
