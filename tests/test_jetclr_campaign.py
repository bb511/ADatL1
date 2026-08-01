from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import jetclr_campaign


def _write_manifest(root: Path) -> dict:
    """Write a minimal authenticated campaign manifest for unit tests."""
    specs = jetclr_campaign.canary_specs()
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
