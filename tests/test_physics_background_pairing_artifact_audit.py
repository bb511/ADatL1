"""Tests for the reporting-time physics campaign artifact audit."""

from __future__ import annotations

import copy
import functools
import hashlib
import pickle  # nosec B403
from pathlib import Path

import pytest
import torch
import yaml

from src.evaluation.physics_background_pairing_artifact_audit import (
    audit_physics_background_pairing_artifacts,
)


def _hash(path: Path) -> str:
    """Hash one synthetic artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(model: str) -> dict:
    """Build a minimal resolved retrain config for one model."""
    config = {
        "data": {"validation_aux_datasets": []},
        "algorithm": {"target_rate": 0.00001, "base_rate": 40_000_000.0},
        "callbacks": {
            "anomaly_eff": {
                "output_name": "ascore/full",
                "threshold_namespace": None,
            }
        },
        "evaluation": {
            "callbacks": {
                "anomaly_efficiency": {
                    "name": "eff_native",
                    "output_name": "ascore/full",
                    "threshold_namespace": None,
                }
            }
        },
    }
    if model in {"ae", "vae"}:
        config["callbacks"]["anomaly_eff_oas"] = {
            "output_name": "ascore/residual_oas",
            "threshold_namespace": "residual_oas",
            "metric_suffix": "residual_oas",
        }
        config["evaluation"]["callbacks"]["anomaly_efficiency_oas"] = {
            "name": "eff_residual_oas",
            "output_name": "ascore/residual_oas",
            "threshold_namespace": "residual_oas",
        }
    return config


def _write_summary(path: Path, metric: str, identity: str, *, value: float = 0.2) -> None:
    """Write one callback-compatible synthetic summary pickle stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = {"cap": 2, "wasserstein": 1, "drift": 2}[metric]
    with path.open("wb") as handle:
        for offset in range(count):
            values = {identity: value + offset}
            pickle.dump({"operational": values} if metric == "drift" else values, handle)


def _fixture(tmp_path: Path, model: str = "ae", metric: str = "cap") -> dict:
    """Build one complete campaign-local retrain artifact fixture."""
    root = tmp_path / "campaign"
    root.mkdir()
    score = "residual_oas" if model == "vae" else "native"
    strategy = "cdf" if metric == "cap" else None
    objective_id = f"{score}__{metric}" + ("__cdf" if strategy else "")
    objective = {
        "id": objective_id,
        "score": score,
        "metric": metric,
        "strategy": strategy,
        "directions": ["maximize", "minimize"] if metric == "cap" else ["minimize", "minimize"],
    }
    design = {"studies": [{"model": model, "objectives": [objective]}]}
    retrain = {
        "model": model,
        "trial_number": 2,
        "params": {},
        "logical_objectives": [objective_id],
        "seed": 123,
        "epochs": 200,
        "retrain_index": 0,
    }
    manifest = {"retrain_runs": [retrain]}
    checkpoint_root = root / "checkpoints" / model
    checkpoint_root.mkdir(parents=True)
    buffers = {"thres_operational": torch.tensor(0.4)}
    if model in {"ae", "vae"}:
        buffers["thres_operational__residual_oas"] = torch.tensor(0.4)
    checkpoint = checkpoint_root / "last.ckpt"
    torch.save({"state_dict": buffers}, checkpoint)
    config_path = root / "retraining/hydra/0000/.hydra/config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_config(model)), encoding="utf-8")
    callback = {
        "cap": f"cap_{score}_cdf_summary",
        "wasserstein": f"wasserstein_{score}_summary",
        "drift": f"drift_{score}_summary",
    }[metric]
    summary = checkpoint_root / "plots/test" / callback / "summary.pkl"
    _write_summary(summary, metric, "last")
    selection = {
        "model": model,
        "retrain_index": 0,
        "logical_objective": objective_id,
        "checkpoint_family_id": "last",
        "optimized_ckpt_name": "last",
    }
    return {
        "campaign_root": root,
        "design": design,
        "manifest": manifest,
        "markers": [{"retrain_index": 0, "row": retrain, "checkpoint_root": checkpoint_root}],
        "checkpoint_selections": [selection],
        "config_path": config_path,
        "checkpoint": checkpoint,
        "summary": summary,
    }


def _audit(fixture: dict) -> dict:
    """Call the public API with only its declared inputs."""
    return audit_physics_background_pairing_artifacts(
        campaign_root=fixture["campaign_root"],
        design=fixture["design"],
        manifest=fixture["manifest"],
        markers=fixture["markers"],
        checkpoint_selections=fixture["checkpoint_selections"],
    )


def _relocate_checkpoint(fixture: dict, family_id: str, filename: str, identity: str) -> None:
    """Move the selected artifact and summary into another candidate family."""
    old_checkpoint = fixture["checkpoint"]
    new_checkpoint = old_checkpoint.parent / family_id / filename
    new_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.rename(new_checkpoint)
    old_summary = fixture["summary"]
    relative_summary = old_summary.relative_to(old_checkpoint.parent)
    new_summary = new_checkpoint.parent / relative_summary
    new_summary.parent.mkdir(parents=True)
    old_summary.rename(new_summary)
    metric = fixture["design"]["studies"][0]["objectives"][0]["metric"]
    _write_summary(new_summary, metric, identity)
    fixture["checkpoint"] = new_checkpoint
    fixture["summary"] = new_summary
    fixture["checkpoint_selections"][0]["checkpoint_family_id"] = family_id
    fixture["checkpoint_selections"][0]["optimized_ckpt_name"] = family_id


@pytest.mark.parametrize(
    ("model", "metric", "threshold_count"),
    [("ae", "cap", 2), ("vae", "wasserstein", 2), ("dsae", "drift", 1)],
)
def test_positive_native_ae_vae_artifacts(
    tmp_path: Path, model: str, metric: str, threshold_count: int
) -> None:
    """Accept native-only and native-plus-OAS model contracts."""
    fixture = _fixture(tmp_path, model, metric)
    result = _audit(fixture)

    assert len(result["checkpoint_families"]) == 1
    assert len(result["threshold_audit"]) == threshold_count
    assert result["metric_summary_files"][0]["summary_sha256"] == _hash(fixture["summary"])
    assert result["threshold_audit"][0]["checkpoint_sha256"] == _hash(fixture["checkpoint"])
    assert result["threshold_audit"][0]["target_rate"] == pytest.approx(0.00001)


def test_rejects_native_namespace_reuse(tmp_path: Path) -> None:
    """The native threshold cannot reuse the residual-OAS namespace."""
    fixture = _fixture(tmp_path)
    config = yaml.safe_load(fixture["config_path"].read_text())
    config["callbacks"]["anomaly_eff"]["threshold_namespace"] = "residual_oas"
    fixture["config_path"].write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="threshold namespace"):
        _audit(fixture)


def test_rejects_validation_leakage(tmp_path: Path) -> None:
    """Retrain validation must not contain downstream auxiliary datasets."""
    fixture = _fixture(tmp_path)
    config = yaml.safe_load(fixture["config_path"].read_text())
    config["data"]["validation_aux_datasets"] = ["signal"]
    fixture["config_path"].write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="leakage boundary"):
        _audit(fixture)


@pytest.mark.parametrize("value", [None, float("nan")])
def test_rejects_missing_or_nonfinite_threshold(tmp_path: Path, value: float | None) -> None:
    """Every required threshold must exist as a finite scalar."""
    fixture = _fixture(tmp_path)
    state = {"thres_operational__residual_oas": torch.tensor(0.4)}
    if value is not None:
        state["thres_operational"] = torch.tensor(value)
    torch.save({"state_dict": state}, fixture["checkpoint"])
    with pytest.raises(ValueError, match="Threshold buffer|finite scalar"):
        _audit(fixture)


def test_rejects_wrong_checkpoint_family(tmp_path: Path) -> None:
    """A summary family from another primary objective is invalid."""
    fixture = _fixture(tmp_path)
    fixture["checkpoint_selections"][0]["checkpoint_family_id"] = "summary/wrong/max"
    fixture["checkpoint_selections"][0]["optimized_ckpt_name"] = "summary/wrong/max"
    with pytest.raises(ValueError, match="is not valid"):
        _audit(fixture)


def test_rejects_checkpoint_root_escape(tmp_path: Path) -> None:
    """Authenticated marker roots remain confined to the campaign."""
    fixture = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    fixture["markers"][0]["checkpoint_root"] = outside
    with pytest.raises(ValueError, match="escapes campaign root"):
        _audit(fixture)


@pytest.mark.parametrize("field", ["config_sha256", "checkpoint_sha256"])
def test_rejects_supplied_hash_mismatch(tmp_path: Path, field: str) -> None:
    """Optional predeclared config and checkpoint hashes are enforced."""
    fixture = _fixture(tmp_path)
    fixture["checkpoint_selections"][0][field] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _audit(fixture)


def test_rejects_summary_object_count(tmp_path: Path) -> None:
    """CAP summaries contain exactly CAP and rank-correlation objects."""
    fixture = _fixture(tmp_path)
    with fixture["summary"].open("wb") as handle:
        pickle.dump({"last": 0.2}, handle)
    with pytest.raises(ValueError, match="Expected 2 summary objects"):
        _audit(fixture)


def test_rejects_duplicate_selection(tmp_path: Path) -> None:
    """An objective can have only one authenticated selection row."""
    fixture = _fixture(tmp_path)
    fixture["checkpoint_selections"].append(copy.deepcopy(fixture["checkpoint_selections"][0]))
    with pytest.raises(ValueError, match="Duplicate checkpoint selection"):
        _audit(fixture)


def test_rejects_nonfinite_summary_and_summary_symlink_escape(tmp_path: Path) -> None:
    """Summary values must be finite and summary paths campaign-local."""
    fixture = _fixture(tmp_path)
    _write_summary(fixture["summary"], "cap", "last", value=float("nan"))
    with pytest.raises(ValueError, match="finite scalar"):
        _audit(fixture)

    fixture["summary"].unlink()
    outside = tmp_path / "outside.pkl"
    _write_summary(outside, "cap", "last")
    fixture["summary"].symlink_to(outside)
    with pytest.raises(ValueError, match="escapes campaign root"):
        _audit(fixture)


def test_exact_last_allows_lightning_versioned_sibling(tmp_path: Path) -> None:
    """Exact last wins even when Lightning leaves versioned siblings."""
    fixture = _fixture(tmp_path)
    (fixture["checkpoint"].parent / "last-v1.ckpt").write_bytes(b"unselected")
    assert _audit(fixture)["checkpoint_families"][0]["checkpoint_path"].endswith("/last.ckpt")


@pytest.mark.parametrize("family", ["stable", "primary"])
def test_resolves_exact_stable_and_objective_primary_families(tmp_path: Path, family: str) -> None:
    """Resolve stable and objective-derived primary families exactly."""
    fixture = _fixture(tmp_path)
    if family == "stable":
        family_id = "single/ascore_operational/stable"
        filename = "ascore_operational__ds=normal__stable.ckpt"
        identity = "normal"
    else:
        primary = "cap_native_cdf_ema_ZB_run396102_vs_ZB_run398183"
        family_id = f"summary/{primary}/max"
        filename = f"{primary}.ckpt"
        identity = primary
    _relocate_checkpoint(fixture, family_id, filename, identity)

    row = _audit(fixture)["checkpoint_families"][0]
    assert row["checkpoint_family"] == ("stable_native" if family == "stable" else "primary_ema")
    assert row["checkpoint_identity"] == identity


def test_trusted_local_lightning_payload_falls_back_from_weights_only(tmp_path: Path) -> None:
    """Trusted Lightning metadata can trigger the documented loader fallback."""
    fixture = _fixture(tmp_path)
    state = {
        "thres_operational": torch.tensor(0.4),
        "thres_operational__residual_oas": torch.tensor(0.4),
    }
    torch.save(
        {"state_dict": state, "optimizer_factory": functools.partial(int, base=10)},
        fixture["checkpoint"],
    )
    assert len(_audit(fixture)["threshold_audit"]) == 2
