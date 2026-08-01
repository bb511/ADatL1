#!/usr/bin/env python3
"""Create and execute a reproducible, node-packed JetCLR canary campaign.

The command deliberately does not submit jobs.  ``init`` freezes a clean Git
revision into a deployment, authenticates the data and Python environment, and
writes a reviewed Slurm launcher.  Submission remains an explicit operator step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess  # nosec B404 - commands are fixed argument vectors, never shell strings
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/iopsstor/scratch/cscs/podagiu/data")
DEFAULT_CAMPAIGN_BASE = Path("/iopsstor/scratch/cscs/vjimenez/jetclr/campaigns")
DEFAULT_DEPLOYMENT_BASE = Path("/iopsstor/scratch/cscs/vjimenez/jetclr/deployments")
DEFAULT_VENV = Path("/iopsstor/scratch/cscs/vjimenez/adatl1/.venv-clariden")
DEFAULT_UV = Path("/iopsstor/scratch/cscs/vjimenez/adatl1/tools/uv-0.11.32/uv")

CONFIG_PATHS = (
    "configs/algorithm/jetclr.yaml",
    "configs/data/basis.yaml",
    "configs/experiment/physics/jetclr_pairing.yaml",
    "configs/trainer/gpu.yaml",
    "configs/train.yaml",
)
DATA_CACHE_RELATIVE = Path("data_2025E+G/mlready/eminimalTauFET_pdefaultTauFET_default/robust")
DATA_FILES = tuple(
    Path(split) / name
    for split in ("train", "valid")
    for name in ("torch_cache.pt", "torch_mask.pt", "torch_l1bit.pt")
)


def _canonical_json(value: Any) -> str:
    """Serialize a value into the canonical representation used for identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    """Return a stable SHA-256 identity for a JSON-compatible value."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact on its destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace a non-empty CSV artifact."""
    if not rows:
        raise ValueError("Cannot write an empty result table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _git(source: Path, *args: str) -> str:
    """Run a read-only Git query in a selected worktree."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git")
    return subprocess.check_output(  # nosec B603 - fixed executable and controlled arguments
        [git, *args], cwd=source, text=True
    ).strip()


def canary_specs() -> list[dict[str, Any]]:
    """Return the fixed four-recipe canary design."""
    recipes = [
        ("production", 123, []),
        (
            "no_augmentation",
            124,
            [
                "algorithm.detector_smearing=null",
                "algorithm.object_mask=null",
                "algorithm.lorentz_rotation=null",
            ],
        ),
        (
            "small_encoder",
            125,
            [
                "data.batch_size=512",
                "algorithm.model.d_model=64",
                "algorithm.model.out_dim=64",
                "algorithm.model.n_layers=2",
                "algorithm.model.n_heads=4",
                "algorithm.model.dim_feedforward=256",
            ],
        ),
        (
            "capacity_stress",
            126,
            [
                "data.batch_size=4096",
                "algorithm.model.d_model=256",
                "algorithm.model.out_dim=256",
                "algorithm.model.n_layers=6",
                "algorithm.model.n_heads=8",
                "algorithm.model.dim_feedforward=1024",
            ],
        ),
    ]
    specs = []
    for trial_id, (name, seed, overrides) in enumerate(recipes):
        identity = {"trial_id": trial_id, "name": name, "seed": seed, "overrides": overrides}
        specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    return specs


def _environment_record(venv: Path, uv: Path) -> dict[str, Any]:
    """Authenticate and describe the frozen ARM64 CUDA environment."""
    python = venv / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(python)
    if not uv.is_file():
        raise FileNotFoundError(uv)
    probe = subprocess.check_output(  # nosec B603 - authenticated venv executable
        [
            str(python),
            "-c",
            (
                "import json,platform,sys,torch,pytorch_lightning as pl;"
                "print(json.dumps({'python':platform.python_version(),"
                "'machine':platform.machine(),'torch':torch.__version__,"
                "'torch_cuda':torch.version.cuda,'lightning':pl.__version__}))"
            ),
        ],
        text=True,
    )
    record = json.loads(probe)
    if tuple(map(int, record["python"].split(".")[:2])) != (3, 10):
        raise RuntimeError(f"JetCLR environment must use Python 3.10, found {record['python']}.")
    if record["machine"] != "aarch64" or record["torch_cuda"] is None:
        raise RuntimeError(f"Environment is not the expected CUDA ARM64 build: {record}")
    record.update(
        {
            "venv": str(venv.resolve()),
            "python_executable": str(python.resolve()),
            "uv": str(uv.resolve()),
            "uv_sha256": _sha256(uv),
        }
    )
    record["fingerprint_sha256"] = _value_sha256(record)
    return record


def _fingerprint_files(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Fingerprint every required file, failing closed when one is absent."""
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def _write_launcher(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Write the reviewed four-way packed Clariden canary launcher."""
    deployment = manifest["deployment"]["path"]
    uv = manifest["environment"]["uv"]
    data_dir = manifest["data"]["root"]
    launcher = root / "slurm" / "canary.sbatch"
    text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-canary-{manifest['campaign_id'][-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=debug
        #SBATCH --time=01:30:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --mem=450G
        #SBATCH --output={root}/slurm/%x-%j.out
        #SBATCH --error={root}/slurm/%x-%j.err

        set -euo pipefail
        readonly REPO={deployment}
        readonly CAMPAIGN_ROOT={root}
        readonly UV={uv}
        export PROJECT_ROOT="$REPO"
        export DATA_DIR={data_dir}
        export RAW_DATA_DIR={data_dir}/raw
        export LOG_DIR="$CAMPAIGN_ROOT/logs"
        export OUTPUT_DIR="$CAMPAIGN_ROOT/outputs"
        export CHECKPOINT_DIR="$CAMPAIGN_ROOT/checkpoints"
        export WANDB_MODE=offline
        export HYDRA_FULL_ERROR=1
        export UV_PROJECT_ENVIRONMENT={manifest['environment']['venv']}

        cd "$REPO"
        test "$(git rev-parse HEAD)" = "{manifest['git']['commit']}"
        test -z "$(git status --porcelain)"
        mkdir -p "$CAMPAIGN_ROOT/slurm"
        "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py canary \\
            --root "$CAMPAIGN_ROOT"

        pids=()
        for trial_id in 0 1 2 3; do
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \\
                "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py run-trial \\
                --root "$CAMPAIGN_ROOT" --trial-id "$trial_id" &
            pids+=("$!")
        done
        status=0
        for pid in "${{pids[@]}}"; do
            wait "$pid" || status=1
        done
        test "$status" -eq 0
        "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py collect \\
            --root "$CAMPAIGN_ROOT"
        """
    )
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(text, encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def initialize(
    root: Path,
    deployment: Path,
    source: Path,
    data_dir: Path,
    venv: Path,
    uv: Path,
    campaign_id: str | None = None,
) -> Path:
    """Freeze code and provenance, then create a non-submitting campaign."""
    root, deployment, source, data_dir = (
        item.expanduser().resolve() for item in (root, deployment, source, data_dir)
    )
    if root.exists():
        raise FileExistsError(root)
    if deployment.exists():
        raise FileExistsError(deployment)
    if _git(source, "status", "--porcelain"):
        raise RuntimeError("Refusing to snapshot a dirty source worktree; commit JetCLR first.")
    commit = _git(source, "rev-parse", "HEAD")
    campaign_id = campaign_id or f"jetclr_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{commit[:8]}"
    environment = _environment_record(venv.expanduser().resolve(), uv.expanduser().resolve())
    config_records = _fingerprint_files([source / path for path in CONFIG_PATHS])
    cache_root = data_dir / DATA_CACHE_RELATIVE
    data_records = _fingerprint_files([cache_root / path for path in DATA_FILES])

    deployment.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git")
    subprocess.run(  # nosec B603 - fixed Git executable and explicit paths
        [git, "clone", "--quiet", "--no-hardlinks", str(source), str(deployment)], check=True
    )
    subprocess.run(  # nosec B603 - fixed Git executable and authenticated commit
        [git, "checkout", "--quiet", "--detach", commit], cwd=deployment, check=True
    )
    if _git(deployment, "status", "--porcelain"):
        raise RuntimeError("Fresh deployment snapshot is unexpectedly dirty.")
    (deployment / ".venv").symlink_to(Path(environment["venv"]), target_is_directory=True)

    root.mkdir(parents=True)
    specs = canary_specs()
    _atomic_json(root / "design" / "canary_trials.json", specs)
    manifest = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": commit,
            "branch": _git(source, "branch", "--show-current"),
            "source": str(source),
        },
        "deployment": {"path": str(deployment), "commit": commit},
        "config": {
            "files": config_records,
            "tree_sha256": _value_sha256(config_records),
            "uv_lock_sha256": _sha256(source / "uv.lock"),
        },
        "data": {
            "root": str(data_dir),
            "cache_root": str(cache_root),
            "files": data_records,
            "tree_sha256": _value_sha256(data_records),
        },
        "environment": environment,
        "canary": {"trials": specs, "design_sha256": _value_sha256(specs)},
    }
    manifest["manifest_payload_sha256"] = _value_sha256(manifest)
    _atomic_json(root / "campaign.json", manifest)
    return _write_launcher(root, manifest)


def _load_campaign(root: Path) -> dict[str, Any]:
    """Load a campaign only after authenticating its immutable payload."""
    path = root / "campaign.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("manifest_payload_sha256")
    if _value_sha256(value) != digest:
        raise ValueError("Campaign manifest fingerprint mismatch.")
    value["manifest_payload_sha256"] = digest
    return value


def _assert_runtime(manifest: Mapping[str, Any]) -> Path:
    """Require execution from the exact clean deployment recorded at init."""
    deployment = Path(manifest["deployment"]["path"])
    if _git(deployment, "rev-parse", "HEAD") != manifest["git"]["commit"]:
        raise RuntimeError("Deployment commit does not match the campaign.")
    if _git(deployment, "status", "--porcelain"):
        raise RuntimeError("Campaign deployment is dirty.")
    return deployment


def validate_campaign(root: Path) -> dict[str, Any]:
    """Re-authenticate code, configs, data, and environment before allocation use."""
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    current_configs = _fingerprint_files([deployment / path for path in CONFIG_PATHS])
    expected_configs = manifest["config"]["files"]
    for current, expected in zip(current_configs, expected_configs, strict=True):
        if current["size"] != expected["size"] or current["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Campaign config changed: {current['path']}")
    current_data = _fingerprint_files([Path(item["path"]) for item in manifest["data"]["files"]])
    if _value_sha256(current_data) != manifest["data"]["tree_sha256"]:
        raise RuntimeError("Campaign data cache fingerprints changed.")
    environment = _environment_record(
        Path(manifest["environment"]["venv"]), Path(manifest["environment"]["uv"])
    )
    if environment["fingerprint_sha256"] != manifest["environment"]["fingerprint_sha256"]:
        raise RuntimeError("Campaign Python environment fingerprint changed.")
    return manifest


def _last_finite_metrics(path: Path) -> dict[str, float]:
    """Extract the final finite value of each metric from a Lightning CSV log."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[str, float] = {}
    for row in rows:
        for name, raw in row.items():
            if name in {"epoch", "epoch_idx", "step"} or raw in (None, ""):
                continue
            value = float(raw)
            if math.isfinite(value):
                metrics[name] = value
    if "train/loss_mean" not in metrics:
        raise RuntimeError(f"Canary produced no finite train/loss_mean in {path}.")
    return metrics


def run_trial(root: Path, trial_id: int) -> Path:
    """Run one fixed real-data canary recipe and atomically record its result."""
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    specs = manifest["canary"]["trials"]
    if trial_id < 0 or trial_id >= len(specs):
        raise ValueError(f"trial-id must be between 0 and {len(specs) - 1}.")
    spec = specs[trial_id]
    trial_root = root / "canary" / f"{trial_id:02d}_{spec['name']}"
    result_path = trial_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError(f"Existing result identity mismatch: {result_path}")
        print(result_path)
        return result_path

    command = [
        sys.executable,
        "src/train.py",
        "experiment=physics/jetclr_pairing",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=4",
        "+trainer.limit_val_batches=1",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "logger=csv",
        "evaluation=null",
        "test=false",
        f"seed={spec['seed']}",
        f"experiment_name=jetclr_canary_{manifest['campaign_id']}",
        f"run_name={trial_id:02d}_{spec['name']}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.output_dir={trial_root / 'output'}",
        f"paths.checkpoints_dir={trial_root / 'checkpoints'}",
        f"hydra.run.dir={trial_root / 'hydra'}",
        "extras.print_config=false",
        *spec["overrides"],
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT_ROOT": str(deployment),
            "DATA_DIR": manifest["data"]["root"],
            "LOG_DIR": str(root / "logs"),
            "OUTPUT_DIR": str(root / "outputs"),
            "CHECKPOINT_DIR": str(root / "checkpoints"),
            "WANDB_MODE": "offline",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    trial_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(  # nosec B603 - argv is fixed campaign configuration
        command, cwd=deployment, env=environment, check=False
    )
    if completed.returncode:
        failure = {
            "schema_version": 1,
            "trial_id": trial_id,
            "spec_sha256": spec["spec_sha256"],
            "returncode": completed.returncode,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        }
        _atomic_json(trial_root / "failure.json", failure)
        raise subprocess.CalledProcessError(completed.returncode, command)
    metric_paths = sorted((trial_root / "output").rglob("metrics.csv"))
    if len(metric_paths) != 1:
        raise RuntimeError(f"Expected one metrics.csv for trial {trial_id}, found {metric_paths}.")
    metrics = _last_finite_metrics(metric_paths[0])
    result = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "trial_id": trial_id,
        "name": spec["name"],
        "seed": spec["seed"],
        "spec_sha256": spec["spec_sha256"],
        "command": command,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "metrics_csv": str(metric_paths[0]),
        "metrics_csv_sha256": _sha256(metric_paths[0]),
        "metrics": metrics,
    }
    result["result_payload_sha256"] = _value_sha256(result)
    _atomic_json(result_path, result)
    print(result_path)
    return result_path


def collect(root: Path) -> Path:
    """Validate all four trial artifacts and write an atomic canary summary."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows = []
    for spec in manifest["canary"]["trials"]:
        result_path = root / "canary" / f"{spec['trial_id']:02d}_{spec['name']}" / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_digest = result.pop("result_payload_sha256", None)
        if result_digest is None or _value_sha256(result) != result_digest:
            raise ValueError(f"Result fingerprint mismatch: {result_path}")
        if result.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError(f"Result identity mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
        ):
            raise ValueError(f"Result campaign identity mismatch: {result_path}")
        metrics_path = Path(result["metrics_csv"])
        if not metrics_path.is_file() or _sha256(metrics_path) != result["metrics_csv_sha256"]:
            raise ValueError(f"Metrics artifact mismatch: {metrics_path}")
        rows.append(
            {
                "trial_id": spec["trial_id"],
                "name": spec["name"],
                "seed": spec["seed"],
                "train_loss": result["metrics"]["train/loss_mean"],
                "git_commit": result["git_commit"],
                "spec_sha256": spec["spec_sha256"],
            }
        )
    table = root / "canary" / "summary.csv"
    _atomic_csv(table, rows)
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete",
        "n_trials": len(rows),
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
    }
    output = root / "canary" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def status(root: Path) -> dict[str, Any]:
    """Report trial completion state without mutating the campaign."""
    root = root.resolve()
    manifest = _load_campaign(root)
    trials = []
    for spec in manifest["canary"]["trials"]:
        trial_root = root / "canary" / f"{spec['trial_id']:02d}_{spec['name']}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        trials.append({"trial_id": spec["trial_id"], "name": spec["name"], "state": state})
    value = {
        "campaign_id": manifest["campaign_id"],
        "complete": all(item["state"] == "complete" for item in trials),
        "trials": trials,
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return value


def main() -> None:
    """Dispatch the campaign command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--root", type=Path)
    init_parser.add_argument("--campaign-id")
    init_parser.add_argument("--deployment", type=Path)
    init_parser.add_argument("--source", type=Path, default=REPO_ROOT)
    init_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    init_parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    init_parser.add_argument("--uv", type=Path, default=DEFAULT_UV)
    canary_parser = subparsers.add_parser("canary")
    canary_parser.add_argument("--root", type=Path, required=True)
    run_parser = subparsers.add_parser("run-trial")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--trial-id", type=int, required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--root", type=Path, required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "init":
        campaign_id = args.campaign_id
        if args.root is None and campaign_id is None:
            parser.error("init requires --root or --campaign-id")
        if campaign_id is None:
            campaign_id = args.root.name
        root = args.root or DEFAULT_CAMPAIGN_BASE / campaign_id
        deployment = args.deployment or DEFAULT_DEPLOYMENT_BASE / campaign_id
        launcher = initialize(
            root, deployment, args.source, args.data_dir, args.venv, args.uv, campaign_id
        )
        print(f"Campaign initialized. Review, then submit manually: sbatch {launcher}")
    elif args.command == "canary":
        manifest = validate_campaign(args.root.resolve())
        launcher = args.root.resolve() / "slurm" / "canary.sbatch"
        if not launcher.is_file():
            _write_launcher(args.root.resolve(), manifest)
        print(f"Campaign validation passed. Launcher: {launcher}")
    elif args.command == "run-trial":
        run_trial(args.root, args.trial_id)
    elif args.command == "collect":
        collect(args.root)
    elif args.command == "status":
        status(args.root)


if __name__ == "__main__":
    main()
