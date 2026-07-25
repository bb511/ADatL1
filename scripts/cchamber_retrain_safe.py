"""Threshold-safe Causal Chamber retraining for the frozen production campaign.

The model process always runs from the repository pinned by ``campaign.json``.
This sidecar only disables the legacy post-training signal callback, because
final threshold calibration and intervention evaluation are performed by the
sealed operating-point audit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from mlflow.tracking import MlflowClient

from scripts import cchamber_campaign as campaign_tools
from scripts import generation


def _git(repository: Path, *args: str) -> str:
    """Run one read-only Git command in a specific deployment."""
    return subprocess.check_output(  # nosec B603 B607
        ["git", "-C", str(repository), *args],
        text=True,
    ).strip()


def _require_clean_revision(repository: Path, commit: str, label: str) -> None:
    """Require one exact clean deployment revision."""
    repository = repository.resolve()
    if _git(repository, "rev-parse", "HEAD") != commit:
        raise RuntimeError(f"{label} HEAD does not match {commit}.")
    if _git(repository, "status", "--porcelain"):
        raise RuntimeError(f"{label} worktree is not clean.")


def _build_overrides(
    *,
    root: Path,
    manifest_index: int,
    item: Mapping[str, Any],
    pairing: Mapping[str, Any],
    run_name: str,
) -> tuple[list[str], str]:
    """Build background-only overrides with signal evaluation disabled."""
    model = str(item["model"])
    strategy = str(item["strategy"])
    seed = int(item["seed"])
    spec = campaign_tools._paper_spec(model, strategy)
    selected = [
        f"{name}={campaign_tools._hydra_value(value)}" for name, value in item["params"].items()
    ]
    overrides = generation.build_retrain_overrides(
        spec,
        seed=seed,
        trainer="gpu",
        devices="[0]",
        selected_overrides=selected,
        run_name=run_name,
    )
    experiment_dir = (
        f"{campaign_tools._campaign_manifest(root)['campaign_id']}_retrain_{model}_{strategy}"
    )
    orchestration_commit = _git(campaign_tools.REPO_ROOT, "rev-parse", "HEAD")
    tags = {
        "campaign_id": campaign_tools._campaign_manifest(root)["campaign_id"],
        "stage": "retrain",
        "model": model,
        "strategy": strategy,
        "candidate_id": item["candidate_id"],
        "model_seed": seed,
        "data_seed": campaign_tools._campaign_manifest(root)["data_seed"],
        "git_commit": campaign_tools._campaign_manifest(root)["git_commit"],
        "retrain_orchestration_commit": orchestration_commit,
        "candidate_pool_sha256": item["pool_sha256"],
        "valid_pair_table_sha256": pairing["primary_validation_table_sha256"],
        "test_pair_table_sha256": pairing["primary_test_table_sha256"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "manifest_index": int(manifest_index),
    }
    tag_override = "{" + ",".join(f"{key}:{value}" for key, value in tags.items()) + "}"
    attempt = run_name.rsplit("_", 1)[-1]
    overrides.extend(
        [
            f"experiment_name={experiment_dir}",
            f"run_name={run_name}",
            f"paths.log_dir={root / 'logs'}",
            f"paths.checkpoints_dir={root / 'checkpoints'}",
            f"hydra.run.dir={root / 'hydra' / 'retrain_safe' / f'{manifest_index:03d}' / attempt}",
            f"logger.mlflow.experiment_name={campaign_tools._experiment_name(str(campaign_tools._campaign_manifest(root)['campaign_id']), 'retrain', model)}",
            f"logger.mlflow.tags={tag_override}",
            "callbacks.clear_ckpts=null",
            "callbacks.last_epoch_ckpt=null",
            "callbacks.stable_ascore_operational_ckpt=null",
            "evaluation.evaluator.ckpts.last=false",
            "~evaluation.evaluator.ckpts.single.ascore_operational",
            "evaluation.callbacks.anomaly_auprc=null",
            "extras.print_config=false",
        ]
    )
    return (
        campaign_tools._replace_pair_env(
            overrides,
            Path(str(pairing["primary_validation_table"])),
            Path(str(pairing["primary_test_table"])),
        ),
        experiment_dir,
    )


def run(root: Path, manifest_index: int, campaign_sha256: str) -> Path:
    """Run one reporting retrain and publish one authenticated checkpoint marker."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Retraining must run on a GPU inside Slurm.")
    root = root.resolve()
    campaign_path = root / "campaign.json"
    if campaign_tools._sha256(campaign_path) != campaign_sha256:
        raise ValueError("Campaign manifest hash changed.")
    campaign = campaign_tools._campaign_manifest(root)
    training_repository = Path(str(campaign["repository"])).resolve()
    _require_clean_revision(
        training_repository,
        str(campaign["git_commit"]),
        "Campaign training deployment",
    )
    orchestration_commit = _git(campaign_tools.REPO_ROOT, "rev-parse", "HEAD")
    _require_clean_revision(
        campaign_tools.REPO_ROOT,
        orchestration_commit,
        "Retrain sidecar deployment",
    )

    item = campaign_tools._manifest_item(root, "retrain_manifest.json", manifest_index)
    pairing = json.loads(
        (root / "pairing" / "comparison" / "pairing_manifest.json").read_text(encoding="utf-8")
    )
    model = str(item["model"])
    strategy = str(item["strategy"])
    seed = int(item["seed"])
    logical_run_name = str(item["run_name"])
    result_path = root / "retrain_results" / f"{manifest_index:03d}.json"
    expected = {
        "campaign_id": campaign["campaign_id"],
        "git_commit": campaign["git_commit"],
        "retrain_orchestration_commit": orchestration_commit,
        "manifest_index": int(manifest_index),
        "model": model,
        "strategy": strategy,
        "seed": seed,
        "logical_run_name": logical_run_name,
        "signal_evaluation_disabled": True,
    }
    if result_path.is_file():
        campaign_tools._validate_marker(
            result_path,
            expected,
            {"checkpoint": "checkpoint_sha256"},
        )
        print(f"[resume] {result_path}")
        return result_path

    attempt = campaign_tools._attempt_id()
    run_name = f"{logical_run_name}_{attempt}"
    overrides, experiment_dir = _build_overrides(
        root=root,
        manifest_index=manifest_index,
        item=item,
        pairing=pairing,
        run_name=run_name,
    )
    training_python = training_repository / ".venv" / "bin" / "python3"
    command = [str(training_python), "src/train.py", *overrides]
    environment = os.environ.copy()
    environment["CCHAMBER_VALID_PAIR_TABLE"] = str(pairing["primary_validation_table"])
    environment["CCHAMBER_TEST_PAIR_TABLE"] = str(pairing["primary_test_table"])
    environment["LOG_DIR"] = str(root / "logs")
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(  # nosec B603
        command,
        cwd=training_repository,
        env=environment,
        check=True,
    )

    checkpoint_run_dir = root / "checkpoints" / experiment_dir / run_name
    relative_checkpoint = campaign_tools._selected_checkpoint_path(
        {"dataset": "cchamber", "strategy": strategy}
    )
    checkpoint = checkpoint_run_dir / relative_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    experiment_name = campaign_tools._experiment_name(
        str(campaign["campaign_id"]),
        "retrain",
        model,
    )
    client = MlflowClient(tracking_uri=str(campaign["tracking_uri"]))
    mlflow_run = campaign_tools._find_mlflow_run(client, experiment_name, run_name)
    result = {
        **item,
        **expected,
        "schema_version": 2,
        "attempt_id": attempt,
        "run_name": run_name,
        "experiment_dir": experiment_dir,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": campaign_tools._sha256(checkpoint),
        "mlflow_experiment": experiment_name,
        "mlflow_run_id": mlflow_run.info.run_id,
        "mlflow_status": mlflow_run.info.status,
        "training_repository": str(training_repository),
        "valid_pair_table_sha256": pairing["primary_validation_table_sha256"],
        "test_pair_table_sha256": pairing["primary_test_table_sha256"],
    }
    campaign_tools._atomic_json(result_path, result)
    client.log_artifact(mlflow_run.info.run_id, str(result_path), artifact_path="campaign")
    client.log_artifact(mlflow_run.info.run_id, str(checkpoint), artifact_path="checkpoint")
    client.set_tag(
        mlflow_run.info.run_id,
        "retrain_result_sha256",
        campaign_tools._sha256(result_path),
    )
    client.set_tag(
        mlflow_run.info.run_id,
        "checkpoint_sha256",
        campaign_tools._sha256(checkpoint),
    )
    print(result_path)
    return result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse sidecar command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-index", type=int, required=True)
    parser.add_argument("--campaign-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested retrain index."""
    args = parse_args(argv)
    run(args.root, args.manifest_index, args.campaign_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
