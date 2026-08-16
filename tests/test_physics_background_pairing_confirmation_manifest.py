"""Tests for authenticated confirmation-manifest freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.physics_background_pairing_confirmation_manifest import (
    build_freeze_payload,
    freeze_confirmation_manifest,
)


def _sha256(path: Path) -> str:
    """Hash a fixture artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(tmp_path: Path) -> Path:
    """Create a minimal authenticated analysis with all required family controls."""
    root = tmp_path / "campaign"
    result = root / "results" / "provisional_search50"
    result.mkdir(parents=True)
    rows = []
    models = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
    for model in models:
        scores = ("native", "residual_oas") if model in {"ae", "vae"} else ("native",)
        for score in scores:
            for index, strategy in enumerate(
                (
                    "flat_physical",
                    "physics_summary",
                    "typed_sliced_wasserstein",
                    "jetclr",
                    "cdf",
                )
            ):
                rows.append(
                    {
                        "model": model,
                        "score": score,
                        "metric": "cap",
                        "strategy": strategy,
                        "mean_efficiency": 0.1 + index / 100,
                        "bottom_quartile_cvar": 0.01,
                        "worst_efficiency": 0.001,
                        "membership_index": len(rows),
                        "trial_number": index,
                        "checkpoint": f"/{model}/{score}/{strategy}.ckpt",
                        "checkpoint_sha256": f"sha-{model}-{score}-{strategy}",
                    }
                )
    analysis = {
        "method": {
            "namespace": "provisional_search50",
            "reported_choices_use_downstream_anomalies": True,
        },
        "front_winners": rows,
    }
    analysis_path = result / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    manifest = {
        "artifacts": [
            {
                "name": "analysis.json",
                "sha256": _sha256(analysis_path),
                "bytes": analysis_path.stat().st_size,
            }
        ]
    }
    (result / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_build_freeze_payload_renders_isolated_strict_commands(tmp_path: Path) -> None:
    """All-score design must render 16 isolated, sequential, two-objective studies."""
    root = _campaign(tmp_path)
    payload = build_freeze_payload(root, scope="all")
    assert len(payload["targets"]) == 16
    assert payload["strict_motpe_requires_n_jobs"] == 1
    assert len({target["database"] for target in payload["targets"]}) == 16
    for target in payload["targets"]:
        command = target["command_argv"]
        assert "hydra.sweeper.n_trials=600" in command
        assert "hydra.sweeper.n_jobs=1" in command
        assert "trainer.min_epochs=50" in command
        assert "trainer.max_epochs=50" in command
        assert any(value.startswith("+confirmation_target=") for value in command)


def test_freeze_requires_preview_hash_and_is_immutable(tmp_path: Path) -> None:
    """Freezing must require affirmative hash authorization and reject replacement."""
    root = _campaign(tmp_path)
    payload = build_freeze_payload(root, scope="native")
    with pytest.raises(ValueError, match="preview hash"):
        freeze_confirmation_manifest(root, scope="native", expected_preview_sha256="wrong")
    output = freeze_confirmation_manifest(
        root,
        scope="native",
        expected_preview_sha256=payload["preview_content_sha256"],
    )
    assert output.is_file()
    assert (
        freeze_confirmation_manifest(
            root,
            scope="native",
            expected_preview_sha256=payload["preview_content_sha256"],
        )
        == output
    )
    with pytest.raises(FileExistsError, match="different immutable"):
        freeze_confirmation_manifest(
            root,
            scope="all",
            expected_preview_sha256=build_freeze_payload(root, scope="all")[
                "preview_content_sha256"
            ],
        )


def test_analysis_tampering_fails_before_preview(tmp_path: Path) -> None:
    """Changing the source analysis after publication must fail closed."""
    root = _campaign(tmp_path)
    analysis = root / "results" / "provisional_search50" / "analysis.json"
    analysis.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        build_freeze_payload(root, scope="native")
