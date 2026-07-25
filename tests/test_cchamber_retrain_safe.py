from pathlib import Path

from scripts import cchamber_retrain_safe as retrain
from src.utils.pairing.io import compose_config


def test_safe_retrain_disables_legacy_evaluation(monkeypatch, tmp_path: Path) -> None:
    """The sidecar must null the inherited signal-only evaluation callback."""
    campaign = {
        "campaign_id": "campaign",
        "data_seed": 314159,
        "git_commit": "a" * 40,
    }
    monkeypatch.setattr(retrain.campaign_tools, "_campaign_manifest", lambda root: campaign)
    monkeypatch.setattr(retrain, "_git", lambda repository, *args: "b" * 40)
    item = {
        "model": "ae",
        "strategy": "cap_metadata_nearest",
        "candidate_id": "000",
        "seed": 1001,
        "pool_sha256": "c" * 64,
        "params": {"algorithm.optimizer.lr": 0.001},
    }
    pairing = {
        "primary_validation_table": str(tmp_path / "valid.pt"),
        "primary_test_table": str(tmp_path / "test.pt"),
        "primary_validation_table_sha256": "d" * 64,
        "primary_test_table_sha256": "e" * 64,
    }

    overrides, experiment_dir = retrain._build_overrides(
        root=tmp_path,
        manifest_index=0,
        item=item,
        pairing=pairing,
        run_name="safe_attempt",
    )

    assert experiment_dir == "campaign_retrain_ae_cap_metadata_nearest"
    assert "test=false" in overrides
    assert "data.signal_experiments=[]" in overrides
    assert "evaluation.callbacks.anomaly_auprc=null" in overrides
    cfg = compose_config(overrides=overrides)
    assert cfg.evaluation.callbacks.anomaly_auprc is None
    assert cfg.evaluation.callbacks.cap_ref is not None
    assert cfg.evaluation.evaluator.ckpts.last is False
    assert "single" not in cfg.evaluation.evaluator.ckpts
