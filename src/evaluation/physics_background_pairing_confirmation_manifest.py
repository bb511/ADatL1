"""Freeze authenticated paper-style confirmation targets and render their commands."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.evaluation.physics_background_pairing_confirmation_targets import (
    build_confirmation_target_preview,
)

ANALYSIS_RELATIVE = Path("results/provisional_search50/analysis.json")
ARTIFACT_MANIFEST_RELATIVE = Path("results/provisional_search50/artifact_manifest.json")
CONFIRMATION_RELATIVE = Path("confirmation")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    """Hash a JSON-compatible payload deterministically."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_authenticated_analysis(campaign_root: Path) -> tuple[dict[str, Any], str, str]:
    """Load analysis only after its portable artifact-manifest hash authenticates."""
    analysis_path = campaign_root / ANALYSIS_RELATIVE
    artifact_manifest_path = campaign_root / ARTIFACT_MANIFEST_RELATIVE
    if not analysis_path.is_file() or not artifact_manifest_path.is_file():
        raise FileNotFoundError("The authenticated provisional search50 result is missing.")
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    rows = artifact_manifest.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("Malformed provisional artifact manifest.")
    matches = [row for row in rows if row.get("name") == analysis_path.name]
    if len(matches) != 1 or matches[0].get("sha256") != _sha256(analysis_path):
        raise ValueError("Provisional analysis hash does not match its artifact manifest.")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    return analysis, _sha256(analysis_path), _sha256(artifact_manifest_path)


def build_freeze_payload(campaign_root: Path, *, scope: str) -> dict[str, Any]:
    """Build, but do not persist, an immutable confirmation manifest payload."""
    campaign_root = campaign_root.resolve()
    analysis, analysis_sha256, artifact_manifest_sha256 = _load_authenticated_analysis(
        campaign_root
    )
    preview = build_confirmation_target_preview(analysis, scope=scope)
    targets = []
    for target in preview["targets"]:
        target_id = (
            f"{int(target['target_index']):02d}_{target['model']}_{target['score']}_"
            f"{target['strategy']}"
        )
        target_root = campaign_root / CONFIRMATION_RELATIVE / "studies" / target_id
        overlay = str(target["logical_objective"]).replace("__", "_")
        study_name = f"physics_bgpair_confirmation_{target_id}"
        database = target_root / "optuna.db"
        sweep_dir = target_root / "sweeps"
        argv = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "python",
            "src/train.py",
            "-m",
            f"experiment=physics/{target['model']}_background_all",
            f"hparams_search={target['hparams_search']}",
            f"+confirmation_target={overlay}",
            f"experiment_name={study_name}",
            "trainer=gpu",
            "trainer.devices=[0]",
            "trainer.min_epochs=50",
            "trainer.max_epochs=50",
            "trainer.num_sanity_val_steps=0",
            "test=false",
            f"hydra.sweeper.storage=sqlite:///{database}?timeout=600",
            f"hydra.sweeper.study_name={study_name}",
            "hydra.sweeper.n_trials=600",
            "hydra.sweeper.n_jobs=1",
            f"hydra.sweep.dir={sweep_dir}",
        ]
        targets.append(
            {
                **target,
                "target_id": target_id,
                "confirmation_target_overlay": overlay,
                "study_name": study_name,
                "database": str(database),
                "sweep_dir": str(sweep_dir),
                "command_argv": argv,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "frozen_confirmation_target_manifest",
        "campaign_root": str(campaign_root),
        "scope": scope,
        "preview_content_sha256": preview["content_sha256"],
        "source_analysis": str(campaign_root / ANALYSIS_RELATIVE),
        "source_analysis_sha256": analysis_sha256,
        "source_artifact_manifest": str(campaign_root / ARTIFACT_MANIFEST_RELATIVE),
        "source_artifact_manifest_sha256": artifact_manifest_sha256,
        "target_rule": preview["target_rule"],
        "uses_downstream_for_target_selection": True,
        "target_selection_bias": preview["target_selection_bias"],
        "strict_motpe_requires_n_jobs": 1,
        "sampler_seed": None,
        "sampler_seed_note": (
            "Null reproduces the paper configuration; persist the realized study database."
        ),
        "targets": targets,
    }
    payload["content_sha256"] = _value_sha256(payload)
    return payload


def freeze_confirmation_manifest(
    campaign_root: Path, *, scope: str, expected_preview_sha256: str
) -> Path:
    """Atomically freeze an explicitly authorized target preview, without overwriting."""
    payload = build_freeze_payload(campaign_root, scope=scope)
    if payload["preview_content_sha256"] != expected_preview_sha256:
        raise ValueError("Expected preview hash does not match the authenticated target set.")
    output = campaign_root.resolve() / CONFIRMATION_RELATIVE / "manifest.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError("A different immutable confirmation manifest already exists.")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


__all__ = ["build_freeze_payload", "freeze_confirmation_manifest"]
