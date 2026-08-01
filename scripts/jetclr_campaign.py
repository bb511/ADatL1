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

from scipy.stats import qmc

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
STAGE1_SEED = 123
STAGE1_N_CANDIDATES = 48
STAGE1_TRAIN_BATCHES = 256
STAGE2_SEED = 123
STAGE2_N_CANDIDATES = 12
STAGE2_SOURCE_CAMPAIGN = "jetclr_20260801_866638a"
STAGE2_SOURCE_SUMMARY_SHA256 = "7041ef8770d2c3cc1578255a2e2a5e88bc27977dca24719cfef4b55a36e702f1"
STAGE2_SOURCE_SUMMARY_CSV_SHA256 = (
    "a6b3441cae9c5c0ce5d9da234a78dcf839cbcf3ab68e6d6768fb65e3c1b72e12"
)
STAGE2_SOURCE_SUMMARY = DEFAULT_CAMPAIGN_BASE / STAGE2_SOURCE_CAMPAIGN / "stage1" / "summary.json"
STAGE2_SOURCE_SUMMARY_CSV = STAGE2_SOURCE_SUMMARY.with_name("summary.csv")
STAGE2_PROMOTED_IDS = (43, 3, 37, 11, 34, 27, 39)


def _stage1_base_overrides() -> dict[str, Any]:
    """Return the production augmentation and optimizer anchor."""
    return {
        "data.batch_size": 2048,
        "trainer.gradient_clip_val": 0.1,
        "algorithm.optimizer.lr": 3e-4,
        "algorithm.optimizer.weight_decay": 1e-4,
        "algorithm.loss.temperature": 0.1,
        "algorithm.detector_smearing.prob": 0.8,
        "algorithm.detector_smearing.strength": 0.5,
        "algorithm.object_mask.prob": 0.8,
        "algorithm.object_mask.object_prob": 0.05,
        "algorithm.lorentz_rotation.prob": 0.5,
    }


def _hydra_scalar(value: Any) -> str:
    """Render a scalar as a stable Hydra command-line value."""
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value).lower() if isinstance(value, bool) else str(value)


def stage1_specs() -> list[dict[str, Any]]:
    """Return eight anchor ablations and forty deterministic Sobol candidates."""
    base = _stage1_base_overrides()
    anchors: list[tuple[str, dict[str, Any]]] = [
        ("production", {}),
        ("no_smearing", {"algorithm.detector_smearing": None}),
        ("no_object_dropout", {"algorithm.object_mask": None}),
        ("no_rotation", {"algorithm.lorentz_rotation": None}),
        (
            "no_augmentation",
            {
                "algorithm.detector_smearing": None,
                "algorithm.object_mask": None,
                "algorithm.lorentz_rotation": None,
            },
        ),
        (
            "weak_augmentation",
            {
                "algorithm.detector_smearing.prob": 0.4,
                "algorithm.detector_smearing.strength": 0.25,
                "algorithm.object_mask.prob": 0.4,
                "algorithm.object_mask.object_prob": 0.025,
                "algorithm.lorentz_rotation.prob": 0.25,
            },
        ),
        (
            "strong_augmentation",
            {
                "algorithm.detector_smearing.prob": 1.0,
                "algorithm.detector_smearing.strength": 1.0,
                "algorithm.object_mask.prob": 1.0,
                "algorithm.object_mask.object_prob": 0.1,
                "algorithm.lorentz_rotation.prob": 1.0,
            },
        ),
        (
            "large_batch_low_temperature",
            {"data.batch_size": 4096, "algorithm.loss.temperature": 0.05},
        ),
    ]
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for name, changes in anchors:
        params = dict(base)
        params.update(changes)
        candidates.append((name, "anchor", params))

    unit = qmc.Sobol(d=10, scramble=True, seed=STAGE1_SEED).random_base2(m=6)[:40]
    batches = (2048, 4096, 8192)
    clips = (0.05, 0.1, 0.5)
    weights = (1e-6, 1e-5, 1e-4, 3e-4)
    temperatures = (0.05, 0.1, 0.2)

    def choice(values: Sequence[Any], coordinate: float) -> Any:
        return values[min(int(coordinate * len(values)), len(values) - 1)]

    for index, row in enumerate(unit):
        params = {
            "data.batch_size": choice(batches, row[0]),
            "trainer.gradient_clip_val": choice(clips, row[1]),
            "algorithm.optimizer.lr": 10 ** (-4.30103 + row[2] * 1.30103),
            "algorithm.optimizer.weight_decay": choice(weights, row[3]),
            "algorithm.loss.temperature": choice(temperatures, row[4]),
            "algorithm.detector_smearing.prob": 0.2 + 0.8 * row[5],
            "algorithm.detector_smearing.strength": 0.1 + 1.4 * row[6],
            "algorithm.object_mask.prob": 0.2 + 0.8 * row[7],
            "algorithm.object_mask.object_prob": 0.15 * row[8],
            "algorithm.lorentz_rotation.prob": row[9],
        }
        candidates.append((f"sobol_{index:02d}", "sobol", params))

    specs = []
    for candidate_id, (name, kind, params) in enumerate(candidates):
        overrides = [f"{key}={_hydra_scalar(value)}" for key, value in params.items()]
        identity = {
            "candidate_id": candidate_id,
            "name": name,
            "kind": kind,
            "seed": STAGE1_SEED,
            "train_batches": STAGE1_TRAIN_BATCHES,
            "params": params,
            "overrides": overrides,
        }
        specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE1_N_CANDIDATES:
        raise AssertionError("Stage-1 design must contain exactly 48 candidates.")
    return specs


def stage2_specs() -> list[dict[str, Any]]:
    """Return the frozen Stage-1 Pareto front and five targeted refinements."""
    stage1 = stage1_specs()
    records: list[tuple[str, str, int | None, dict[str, Any]]] = []
    for source_id in STAGE2_PROMOTED_IDS:
        source = stage1[source_id]
        records.append(
            (
                f"stage1_{source_id:02d}_{source['name']}",
                "stage1_promoted",
                source_id,
                dict(source["params"]),
            )
        )

    base = _stage1_base_overrides()
    refinements = [
        (
            "refine_b2048_lr5e-5_t05_rot0",
            3,
            {
                "data.batch_size": 2048,
                "algorithm.optimizer.lr": 5e-5,
                "algorithm.loss.temperature": 0.05,
                "algorithm.detector_smearing.prob": 0.4,
                "algorithm.detector_smearing.strength": 0.2,
                "algorithm.object_mask.prob": 0.4,
                "algorithm.object_mask.object_prob": 0.01,
                "algorithm.lorentz_rotation.prob": 0.0,
            },
        ),
        (
            "refine_b4096_lr1e-4_t05_rot0",
            43,
            {
                "data.batch_size": 4096,
                "algorithm.optimizer.lr": 1e-4,
                "algorithm.loss.temperature": 0.05,
                "algorithm.detector_smearing.prob": 0.5,
                "algorithm.detector_smearing.strength": 0.25,
                "algorithm.object_mask.prob": 0.5,
                "algorithm.object_mask.object_prob": 0.02,
                "algorithm.lorentz_rotation.prob": 0.0,
            },
        ),
        (
            "refine_b4096_lr3e-4_t10_rot002",
            37,
            {
                "data.batch_size": 4096,
                "algorithm.optimizer.lr": 3e-4,
                "algorithm.loss.temperature": 0.1,
                "algorithm.detector_smearing.prob": 0.6,
                "algorithm.detector_smearing.strength": 0.35,
                "algorithm.object_mask.prob": 0.6,
                "algorithm.object_mask.object_prob": 0.03,
                "algorithm.lorentz_rotation.prob": 0.02,
            },
        ),
        (
            "refine_b8192_lr2e-4_t10_rot0",
            11,
            {
                "data.batch_size": 8192,
                "algorithm.optimizer.lr": 2e-4,
                "algorithm.loss.temperature": 0.1,
                "algorithm.detector_smearing.prob": 0.5,
                "algorithm.detector_smearing.strength": 0.3,
                "algorithm.object_mask.prob": 0.5,
                "algorithm.object_mask.object_prob": 0.025,
                "algorithm.lorentz_rotation.prob": 0.0,
            },
        ),
        (
            "refine_b8192_lr5e-4_t20_rot005",
            34,
            {
                "data.batch_size": 8192,
                "algorithm.optimizer.lr": 5e-4,
                "algorithm.loss.temperature": 0.2,
                "algorithm.detector_smearing.prob": 0.8,
                "algorithm.detector_smearing.strength": 0.6,
                "algorithm.object_mask.prob": 0.8,
                "algorithm.object_mask.object_prob": 0.06,
                "algorithm.lorentz_rotation.prob": 0.05,
            },
        ),
    ]
    for name, source_id, changes in refinements:
        params = dict(base)
        params.update(changes)
        records.append((name, "targeted_refinement", source_id, params))

    specs = []
    for candidate_id, (name, kind, source_id, params) in enumerate(records):
        source = stage1[source_id] if source_id is not None else None
        rationale = (
            "Stage-1 promotion preserving Pareto utility and balance-safe diversity."
            if kind == "stage1_promoted"
            else "Targeted modest-augmentation refinement around a Pareto candidate with "
            "zero or near-zero azimuthal rotation."
        )
        identity = {
            "candidate_id": candidate_id,
            "name": name,
            "kind": kind,
            "seed": STAGE2_SEED,
            "full_epochs": 1,
            "source_campaign_id": STAGE2_SOURCE_CAMPAIGN,
            "source_candidate_id": source_id,
            "source_candidate_spec_sha256": source["spec_sha256"] if source else None,
            "rationale": rationale,
            "params": params,
            "overrides": [f"{key}={_hydra_scalar(value)}" for key, value in params.items()],
        }
        specs.append({**identity, "spec_sha256": _value_sha256(identity)})
    if len(specs) != STAGE2_N_CANDIDATES:
        raise AssertionError("Stage-2 design must contain exactly 12 candidates.")
    return specs


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


def _write_stage1_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write the packed Stage-1 array, CPU collector, and dependency submitter."""
    deployment = manifest["deployment"]["path"]
    uv = manifest["environment"]["uv"]
    data_dir = manifest["data"]["root"]
    common = textwrap.dedent(
        f"""\
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
        """
    )
    stage1 = root / "slurm" / "stage1.sbatch"
    stage1_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-s1-{manifest['campaign_id'][-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=12:00:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --mem=450G
        #SBATCH --array=0-11%4
        #SBATCH --output={root}/slurm/%x-%A_%a.out
        #SBATCH --error={root}/slurm/%x-%A_%a.err

        """
    )
    stage1_text += common
    stage1_text += textwrap.dedent(
        """
        base=$((SLURM_ARRAY_TASK_ID * 4))
        pids=()
        for offset in 0 1 2 3; do
            candidate_id=$((base + offset))
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
                "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py run-stage1 \
                --root "$CAMPAIGN_ROOT" --candidate-id "$candidate_id" &
            pids+=("$!")
        done
        status=0
        for pid in "${pids[@]}"; do
            wait "$pid" || status=1
        done
        exit "$status"
        """
    )
    stage1.parent.mkdir(parents=True, exist_ok=True)
    stage1.write_text(stage1_text, encoding="utf-8")
    stage1.chmod(0o755)

    collector = root / "slurm" / "stage1_collect.sbatch"
    collector_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-s1-collect-{manifest['campaign_id'][-8:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=00:30:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=16G
        #SBATCH --output={root}/slurm/%x-%j.out
        #SBATCH --error={root}/slurm/%x-%j.err

        """
    )
    collector_text += common
    collector_text += (
        '"$UV" run --frozen --no-sync python scripts/jetclr_campaign.py collect-stage1 '
        '--root "$CAMPAIGN_ROOT"\n'
    )
    collector.write_text(collector_text, encoding="utf-8")
    collector.chmod(0o755)

    submitter = root / "slurm" / "submit_stage1.sh"
    submitter.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            readonly SCRIPT_DIR={root}/slurm
            stage1_job=$(sbatch --parsable "$SCRIPT_DIR/stage1.sbatch")
            collector_job=$(sbatch --parsable --dependency="afterok:$stage1_job" \
                "$SCRIPT_DIR/stage1_collect.sbatch")
            printf 'stage1=%s collector=%s\\n' "$stage1_job" "$collector_job"
            """
        ),
        encoding="utf-8",
    )
    submitter.chmod(0o755)
    return {"stage1": stage1, "collector": collector, "submitter": submitter}


def _write_stage2_launchers(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Write the three-node packed Stage-2 array and dependent CPU collector."""
    deployment = manifest["deployment"]["path"]
    uv = manifest["environment"]["uv"]
    data_dir = manifest["data"]["root"]
    common = textwrap.dedent(
        f"""\
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
        """
    )
    stage2 = root / "slurm" / "stage2.sbatch"
    stage2_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-s2-{manifest['campaign_id'][-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=12:00:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --mem=450G
        #SBATCH --array=0-2%3
        #SBATCH --output={root}/slurm/%x-%A_%a.out
        #SBATCH --error={root}/slurm/%x-%A_%a.err

        """
    )
    stage2_text += common
    stage2_text += textwrap.dedent(
        """
        base=$((SLURM_ARRAY_TASK_ID * 4))
        pids=()
        for offset in 0 1 2 3; do
            candidate_id=$((base + offset))
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
                "$UV" run --frozen --no-sync python scripts/jetclr_campaign.py run-stage2 \
                --root "$CAMPAIGN_ROOT" --candidate-id "$candidate_id" &
            pids+=("$!")
        done
        status=0
        for pid in "${pids[@]}"; do
            wait "$pid" || status=1
        done
        exit "$status"
        """
    )
    stage2.parent.mkdir(parents=True, exist_ok=True)
    stage2.write_text(stage2_text, encoding="utf-8")
    stage2.chmod(0o755)

    collector = root / "slurm" / "stage2_collect.sbatch"
    collector_text = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=jetclr-s2-collect-{manifest['campaign_id'][-8:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=00:30:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem=16G
        #SBATCH --output={root}/slurm/%x-%j.out
        #SBATCH --error={root}/slurm/%x-%j.err

        """
    )
    collector_text += common
    collector_text += (
        '"$UV" run --frozen --no-sync python scripts/jetclr_campaign.py collect-stage2 '
        '--root "$CAMPAIGN_ROOT"\n'
    )
    collector.write_text(collector_text, encoding="utf-8")
    collector.chmod(0o755)

    submitter = root / "slurm" / "submit_stage2.sh"
    submitter.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            readonly SCRIPT_DIR={root}/slurm
            stage2_job=$(sbatch --parsable "$SCRIPT_DIR/stage2.sbatch")
            collector_job=$(sbatch --parsable --dependency="afterok:$stage2_job" \
                "$SCRIPT_DIR/stage2_collect.sbatch")
            printf 'stage2=%s collector=%s\\n' "$stage2_job" "$collector_job"
            """
        ),
        encoding="utf-8",
    )
    submitter.chmod(0o755)
    return {"stage2": stage2, "collector": collector, "submitter": submitter}


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
    if _sha256(STAGE2_SOURCE_SUMMARY) != STAGE2_SOURCE_SUMMARY_SHA256:
        raise RuntimeError("Frozen Stage-1 source summary fingerprint changed.")
    if _sha256(STAGE2_SOURCE_SUMMARY_CSV) != STAGE2_SOURCE_SUMMARY_CSV_SHA256:
        raise RuntimeError("Frozen Stage-1 source metric table fingerprint changed.")

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
    stage1 = stage1_specs()
    stage2 = stage2_specs()
    _atomic_json(root / "design" / "canary_trials.json", specs)
    _atomic_json(root / "design" / "stage1_candidates.json", stage1)
    _atomic_json(root / "design" / "stage2_candidates.json", stage2)
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
        "stage1": {
            "seed": STAGE1_SEED,
            "train_batches": STAGE1_TRAIN_BATCHES,
            "candidates": stage1,
            "design_sha256": _value_sha256(stage1),
        },
        "stage2": {
            "seed": STAGE2_SEED,
            "full_epochs": 1,
            "source_campaign_id": STAGE2_SOURCE_CAMPAIGN,
            "source_summary": str(STAGE2_SOURCE_SUMMARY),
            "source_summary_sha256": STAGE2_SOURCE_SUMMARY_SHA256,
            "source_summary_csv": str(STAGE2_SOURCE_SUMMARY_CSV),
            "source_summary_csv_sha256": STAGE2_SOURCE_SUMMARY_CSV_SHA256,
            "source_promoted_candidate_ids": list(STAGE2_PROMOTED_IDS),
            "candidates": stage2,
            "design_sha256": _value_sha256(stage2),
        },
    }
    manifest["manifest_payload_sha256"] = _value_sha256(manifest)
    _atomic_json(root / "campaign.json", manifest)
    launcher = _write_launcher(root, manifest)
    _write_stage1_launchers(root, manifest)
    _write_stage2_launchers(root, manifest)
    return launcher


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


def _metric_json(path: Path, required: Sequence[str]) -> dict[str, Any]:
    """Load a metric artifact and require its selection fields to be finite."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Metric artifact must contain an object: {path}")
    for name in required:
        if name not in value:
            raise ValueError(f"Metric artifact {path} is missing {name!r}.")
        metric = value[name]
        if metric is None and name in {
            "value_smd_before_mean",
            "value_smd_after_mean",
            "occupancy_smd_before_mean",
            "occupancy_smd_after_mean",
        }:
            continue
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"Metric {name!r} in {path} must be numeric.")
        if not math.isfinite(float(metric)):
            raise ValueError(f"Metric {name!r} in {path} must be finite.")
    return value


def _validate_optional_metrics(value: Mapping[str, Any], path: Path, names: Sequence[str]) -> None:
    """Allow unavailable metrics as null while rejecting missing or non-finite values."""
    for name in names:
        if name not in value:
            raise ValueError(f"Metric artifact {path} is missing {name!r}.")
        metric = value[name]
        if metric is None:
            continue
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"Optional metric {name!r} in {path} must be numeric or null.")
        if not math.isfinite(float(metric)):
            raise ValueError(f"Optional metric {name!r} in {path} must be finite or null.")


def _single_artifact(root: Path, name: str) -> Path:
    """Resolve exactly one named evaluator artifact below a trial output."""
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {name} below {root}, found {paths}.")
    return paths[0]


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
        "callbacks.rich_progress_bar=null",
        "callbacks.model_summary=null",
        "callbacks.log_data_mlflow=null",
        "logger=csv",
        "evaluation.callbacks=null",
        "test=false",
        f"seed={spec['seed']}",
        f"experiment_name=jetclr_canary_{manifest['campaign_id']}",
        f"run_name={trial_id:02d}_{spec['name']}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.output_dir={trial_root / 'output'}",
        f"paths.checkpoints_dir={trial_root / 'checkpoints'}",
        f"hydra.run.dir={trial_root / 'hydra'}",
        "extras.enforce_tags=false",
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
    (trial_root / "output").mkdir(parents=True, exist_ok=True)
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


def run_stage1(root: Path, candidate_id: int) -> Path:
    """Run one fixed Stage-1 candidate and authenticate its three metric artifacts."""
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    specs = manifest["stage1"]["candidates"]
    if candidate_id < 0 or candidate_id >= len(specs):
        raise ValueError(f"candidate-id must be between 0 and {len(specs) - 1}.")
    spec = specs[candidate_id]
    trial_root = root / "stage1" / f"candidate_{candidate_id:03d}"
    result_path = trial_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Existing Stage-1 result fingerprint mismatch: {result_path}")
        if result.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError(f"Existing Stage-1 result identity mismatch: {result_path}")
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
        f"+trainer.limit_train_batches={manifest['stage1']['train_batches']}",
        "+trainer.limit_val_batches=0",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "callbacks.rich_progress_bar=null",
        "callbacks.model_summary=null",
        "callbacks.log_data_mlflow=null",
        "logger=csv",
        "test=false",
        f"seed={manifest['stage1']['seed']}",
        "data.max_val_batches=4",
        "data.max_normal_eval_batches=8",
        "evaluation.callbacks.pairing_diagnostics.max_events_per_dataset=8192",
        "evaluation.callbacks.embedding_anomaly.reference_size=8192",
        "evaluation.callbacks.embedding_anomaly.max_query_events=8192",
        f"experiment_name=jetclr_stage1_{manifest['campaign_id']}",
        f"run_name=candidate_{candidate_id:03d}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.output_dir={trial_root / 'output'}",
        f"paths.checkpoints_dir={trial_root / 'checkpoints'}",
        f"hydra.run.dir={trial_root / 'hydra'}",
        "extras.print_config=false",
        "extras.enforce_tags=false",
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
    (trial_root / "output").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(  # nosec B603 - fixed campaign argv without a shell
        command, cwd=deployment, env=environment, check=False
    )
    if completed.returncode:
        _atomic_json(
            trial_root / "failure.json",
            {
                "schema_version": 1,
                "campaign_id": manifest["campaign_id"],
                "candidate_id": candidate_id,
                "spec_sha256": spec["spec_sha256"],
                "returncode": completed.returncode,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            },
        )
        raise subprocess.CalledProcessError(completed.returncode, command)

    metrics_csv = _single_artifact(trial_root / "output", "metrics.csv")
    training = _last_finite_metrics(metrics_csv)
    pairing_path = _single_artifact(trial_root / "output", "pairing_diagnostics.json")
    pairing = _metric_json(
        pairing_path,
        (
            "selection_score",
            "raw_selection_score",
            "closure_recall_at_10",
            "mnn_coverage",
            "embedding_finite_fraction",
            "embedding_active_fraction",
            "embedding_effective_rank",
            "embedding_participation_rank",
            "embedding_top_pc_fraction",
        ),
    )
    _validate_optional_metrics(
        pairing,
        pairing_path,
        (
            "value_smd_before_mean",
            "value_smd_after_mean",
            "occupancy_smd_before_mean",
            "occupancy_smd_after_mean",
        ),
    )
    if not isinstance(pairing.get("collapse_pass"), bool) or not isinstance(
        pairing.get("collapse_failures"), list
    ):
        raise ValueError(f"Pairing collapse gate is malformed: {pairing_path}")
    anomaly_path = _single_artifact(trial_root / "output", "embedding_anomaly.json")
    anomaly = _metric_json(
        anomaly_path,
        ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc"),
    )
    if not isinstance(anomaly.get("per_dataset"), dict) or not anomaly["per_dataset"]:
        raise ValueError(f"Embedding anomaly per-dataset metrics are missing: {anomaly_path}")
    for name in ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc"):
        if not 0.0 <= float(anomaly[name]) <= 1.0:
            raise ValueError(f"AUROC {name!r} is outside [0, 1]: {anomaly_path}")

    artifacts = {
        "training_csv": metrics_csv,
        "pairing_json": pairing_path,
        "anomaly_json": anomaly_path,
    }
    result = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "candidate_id": candidate_id,
        "name": spec["name"],
        "kind": spec["kind"],
        "seed": spec["seed"],
        "spec_sha256": spec["spec_sha256"],
        "params": spec["params"],
        "command": command,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in artifacts.items()
        },
        "training_metrics": training,
        "pairing_metrics": pairing,
        "anomaly_metrics": anomaly,
    }
    result["result_payload_sha256"] = _value_sha256(result)
    _atomic_json(result_path, result)
    print(result_path)
    return result_path


def run_stage2(root: Path, candidate_id: int) -> Path:
    """Run one Stage-2 candidate for one complete epoch and authenticate outputs."""
    root = root.resolve()
    manifest = _load_campaign(root)
    deployment = _assert_runtime(manifest)
    specs = manifest["stage2"]["candidates"]
    if candidate_id < 0 or candidate_id >= len(specs):
        raise ValueError(f"candidate-id must be between 0 and {len(specs) - 1}.")
    spec = specs[candidate_id]
    trial_root = root / "stage2" / f"candidate_{candidate_id:03d}"
    result_path = trial_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Existing Stage-2 result fingerprint mismatch: {result_path}")
        if result.get("spec_sha256") != spec["spec_sha256"]:
            raise ValueError(f"Existing Stage-2 result identity mismatch: {result_path}")
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
        "+trainer.limit_val_batches=0",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "callbacks.rich_progress_bar=null",
        "callbacks.model_summary=null",
        "callbacks.log_data_mlflow=null",
        "logger=csv",
        "test=false",
        f"seed={manifest['stage2']['seed']}",
        "data.max_val_batches=4",
        "data.max_normal_eval_batches=8",
        "evaluation.callbacks.pairing_diagnostics.max_events_per_dataset=8192",
        "evaluation.callbacks.embedding_anomaly.reference_size=8192",
        "evaluation.callbacks.embedding_anomaly.max_query_events=8192",
        f"experiment_name=jetclr_stage2_{manifest['campaign_id']}",
        f"run_name=candidate_{candidate_id:03d}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.output_dir={trial_root / 'output'}",
        f"paths.checkpoints_dir={trial_root / 'checkpoints'}",
        f"hydra.run.dir={trial_root / 'hydra'}",
        "extras.print_config=false",
        "extras.enforce_tags=false",
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
    (trial_root / "output").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(  # nosec B603 - fixed campaign argv without a shell
        command, cwd=deployment, env=environment, check=False
    )
    if completed.returncode:
        _atomic_json(
            trial_root / "failure.json",
            {
                "schema_version": 1,
                "campaign_id": manifest["campaign_id"],
                "candidate_id": candidate_id,
                "spec_sha256": spec["spec_sha256"],
                "returncode": completed.returncode,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            },
        )
        raise subprocess.CalledProcessError(completed.returncode, command)

    metrics_csv = _single_artifact(trial_root / "output", "metrics.csv")
    training = _last_finite_metrics(metrics_csv)
    pairing_path = _single_artifact(trial_root / "output", "pairing_diagnostics.json")
    pairing = _metric_json(
        pairing_path,
        (
            "selection_score",
            "raw_selection_score",
            "closure_recall_at_10",
            "mnn_coverage",
            "embedding_finite_fraction",
            "embedding_active_fraction",
            "embedding_effective_rank",
            "embedding_participation_rank",
            "embedding_top_pc_fraction",
            "value_smd_before_mean",
            "value_smd_after_mean",
            "occupancy_smd_before_mean",
            "occupancy_smd_after_mean",
        ),
    )
    if not isinstance(pairing.get("collapse_pass"), bool) or not isinstance(
        pairing.get("collapse_failures"), list
    ):
        raise ValueError(f"Pairing collapse gate is malformed: {pairing_path}")
    anomaly_path = _single_artifact(trial_root / "output", "embedding_anomaly.json")
    anomaly = _metric_json(
        anomaly_path,
        ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc"),
    )
    if not isinstance(anomaly.get("per_dataset"), dict) or not anomaly["per_dataset"]:
        raise ValueError(f"Embedding anomaly per-dataset metrics are missing: {anomaly_path}")
    for name in ("macro_median_auroc", "macro_mean_auroc", "worst_quartile_mean_auroc"):
        if not 0.0 <= float(anomaly[name]) <= 1.0:
            raise ValueError(f"AUROC {name!r} is outside [0, 1]: {anomaly_path}")

    artifacts = {
        "training_csv": metrics_csv,
        "pairing_json": pairing_path,
        "anomaly_json": anomaly_path,
    }
    result = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "git_commit": manifest["git"]["commit"],
        "candidate_id": candidate_id,
        "name": spec["name"],
        "kind": spec["kind"],
        "seed": spec["seed"],
        "spec_sha256": spec["spec_sha256"],
        "source_campaign_id": spec["source_campaign_id"],
        "source_candidate_id": spec["source_candidate_id"],
        "source_candidate_spec_sha256": spec["source_candidate_spec_sha256"],
        "rationale": spec["rationale"],
        "params": spec["params"],
        "command": command,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in artifacts.items()
        },
        "training_metrics": training,
        "pairing_metrics": pairing,
        "anomaly_metrics": anomaly,
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


def collect_stage1(root: Path) -> Path:
    """Authenticate all Stage-1 results and rank collapse-safe candidates."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows: list[dict[str, Any]] = []
    missing = []
    for spec in manifest["stage1"]["candidates"]:
        result_path = root / "stage1" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Stage-1 result fingerprint mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
            or result.get("candidate_id") != spec["candidate_id"]
            or result.get("spec_sha256") != spec["spec_sha256"]
        ):
            raise ValueError(f"Stage-1 result identity mismatch: {result_path}")
        for artifact in result["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Stage-1 artifact mismatch: {path}")
        pairing = result["pairing_metrics"]
        anomaly = result["anomaly_metrics"]
        collapse_pass = bool(pairing["collapse_pass"])
        finite_pass = float(pairing["embedding_finite_fraction"]) == 1.0
        eligible = collapse_pass and finite_pass
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "name": spec["name"],
                "kind": spec["kind"],
                "seed": spec["seed"],
                "eligible": eligible,
                "collapse_pass": collapse_pass,
                "collapse_failures": ";".join(pairing["collapse_failures"]),
                "embedding_finite_fraction": pairing["embedding_finite_fraction"],
                "embedding_active_fraction": pairing["embedding_active_fraction"],
                "embedding_effective_rank": pairing["embedding_effective_rank"],
                "embedding_participation_rank": pairing["embedding_participation_rank"],
                "embedding_top_pc_fraction": pairing["embedding_top_pc_fraction"],
                "pairing_selection_score": pairing["selection_score"],
                "pairing_raw_selection_score": pairing["raw_selection_score"],
                "closure_recall_at_10": pairing["closure_recall_at_10"],
                "mnn_coverage": pairing["mnn_coverage"],
                "macro_median_auroc": anomaly["macro_median_auroc"],
                "macro_mean_auroc": anomaly["macro_mean_auroc"],
                "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
                "train_loss": result["training_metrics"]["train/loss_mean"],
                "params_json": _canonical_json(spec["params"]),
                "spec_sha256": spec["spec_sha256"],
                "result_path": str(result_path),
            }
        )
    if missing:
        raise FileNotFoundError(
            f"Stage 1 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )
    rows.sort(
        key=lambda row: (
            bool(row["eligible"]),
            float(row["worst_quartile_mean_auroc"]),
            float(row["macro_median_auroc"]),
            float(row["pairing_selection_score"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    table = root / "stage1" / "summary.csv"
    _atomic_csv(table, rows)
    eligible = [row for row in rows if row["eligible"]]
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete" if eligible else "complete_no_eligible_candidates",
        "n_candidates": len(rows),
        "n_collapse_pass": sum(bool(row["collapse_pass"]) for row in rows),
        "n_eligible": len(eligible),
        "ranking": [
            "collapse_gate",
            "worst_quartile_mean_auroc",
            "macro_median_auroc",
            "pairing_selection_score",
        ],
        "best_candidate_id": eligible[0]["candidate_id"] if eligible else None,
        "best_candidate_spec_sha256": eligible[0]["spec_sha256"] if eligible else None,
        "proxy_candidate_id": rows[0]["candidate_id"],
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
    }
    output = root / "stage1" / "summary.json"
    _atomic_json(output, summary)
    print(output)
    return output


def _pareto_front(rows: Sequence[Mapping[str, Any]], objectives: Sequence[str]) -> set[int]:
    """Return candidate IDs not dominated when every objective is maximized."""
    front = set()
    for candidate in rows:
        dominated = False
        for other in rows:
            if other["candidate_id"] == candidate["candidate_id"]:
                continue
            weakly_better = all(float(other[key]) >= float(candidate[key]) for key in objectives)
            strictly_better = any(float(other[key]) > float(candidate[key]) for key in objectives)
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(int(candidate["candidate_id"]))
    return front


def _balance_improves(pairing: Mapping[str, Any]) -> bool:
    """Return false for unavailable SMDs and otherwise require both balances to improve."""
    names = (
        "value_smd_before_mean",
        "value_smd_after_mean",
        "occupancy_smd_before_mean",
        "occupancy_smd_after_mean",
    )
    values = [pairing.get(name) for name in names]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return False
    value_before, value_after, occupancy_before, occupancy_after = map(float, values)
    return value_after <= value_before and occupancy_after <= occupancy_before


def collect_stage2(root: Path) -> Path:
    """Authenticate Stage-2 results and report gates plus a three-objective Pareto set."""
    root = root.resolve()
    manifest = _load_campaign(root)
    rows: list[dict[str, Any]] = []
    missing = []
    for spec in manifest["stage2"]["candidates"]:
        result_path = root / "stage2" / f"candidate_{spec['candidate_id']:03d}" / "result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        digest = result.pop("result_payload_sha256", None)
        if digest is None or _value_sha256(result) != digest:
            raise ValueError(f"Stage-2 result fingerprint mismatch: {result_path}")
        if (
            result.get("campaign_id") != manifest["campaign_id"]
            or result.get("git_commit") != manifest["git"]["commit"]
            or result.get("candidate_id") != spec["candidate_id"]
            or result.get("spec_sha256") != spec["spec_sha256"]
            or result.get("source_campaign_id") != spec["source_campaign_id"]
            or result.get("source_candidate_id") != spec["source_candidate_id"]
        ):
            raise ValueError(f"Stage-2 result identity mismatch: {result_path}")
        for artifact in result["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Stage-2 artifact mismatch: {path}")
        pairing = result["pairing_metrics"]
        anomaly = result["anomaly_metrics"]
        collapse_pass = bool(pairing["collapse_pass"])
        finite_pass = float(pairing["embedding_finite_fraction"]) == 1.0
        balance_pass = _balance_improves(pairing)
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "name": spec["name"],
                "kind": spec["kind"],
                "source_candidate_id": spec["source_candidate_id"],
                "rationale": spec["rationale"],
                "seed": spec["seed"],
                "collapse_eligible": collapse_pass and finite_pass,
                "collapse_pass": collapse_pass,
                "collapse_failures": ";".join(pairing["collapse_failures"]),
                "balance_pass": balance_pass,
                "embedding_finite_fraction": pairing["embedding_finite_fraction"],
                "embedding_active_fraction": pairing["embedding_active_fraction"],
                "embedding_effective_rank": pairing["embedding_effective_rank"],
                "embedding_participation_rank": pairing["embedding_participation_rank"],
                "embedding_top_pc_fraction": pairing["embedding_top_pc_fraction"],
                "raw_selection_score": pairing["raw_selection_score"],
                "gated_selection_score": pairing["selection_score"],
                "closure_recall_at_10": pairing["closure_recall_at_10"],
                "mnn_coverage": pairing["mnn_coverage"],
                "value_smd_before_mean": pairing["value_smd_before_mean"],
                "value_smd_after_mean": pairing["value_smd_after_mean"],
                "occupancy_smd_before_mean": pairing["occupancy_smd_before_mean"],
                "occupancy_smd_after_mean": pairing["occupancy_smd_after_mean"],
                "macro_median_auroc": anomaly["macro_median_auroc"],
                "macro_mean_auroc": anomaly["macro_mean_auroc"],
                "worst_quartile_mean_auroc": anomaly["worst_quartile_mean_auroc"],
                "train_loss": result["training_metrics"]["train/loss_mean"],
                "params_json": _canonical_json(spec["params"]),
                "spec_sha256": spec["spec_sha256"],
                "result_path": str(result_path),
            }
        )
    if missing:
        raise FileNotFoundError(
            f"Stage 2 is incomplete: {len(missing)} results missing; first is {missing[0]}"
        )
    objectives = (
        "embedding_effective_rank",
        "raw_selection_score",
        "worst_quartile_mean_auroc",
    )
    front = _pareto_front(rows, objectives)
    for row in rows:
        row["pareto_nondominated"] = row["candidate_id"] in front
    rows.sort(key=lambda row: int(row["candidate_id"]))
    table = root / "stage2" / "summary.csv"
    _atomic_csv(table, rows)
    eligible_ids = [int(row["candidate_id"]) for row in rows if row["collapse_eligible"]]
    balance_ids = [int(row["candidate_id"]) for row in rows if row["balance_pass"]]
    summary = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "complete" if eligible_ids else "complete_no_collapse_eligible_candidates",
        "n_candidates": len(rows),
        "n_collapse_eligible": len(eligible_ids),
        "collapse_eligible_candidate_ids": eligible_ids,
        "n_balance_pass": len(balance_ids),
        "balance_pass_candidate_ids": balance_ids,
        "pareto_objectives": list(objectives),
        "pareto_candidate_ids": sorted(front),
        "selection_policy": "No scalar winner is selected; preserve the validation Pareto set.",
        "summary_csv": str(table),
        "summary_csv_sha256": _sha256(table),
    }
    output = root / "stage2" / "summary.json"
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
    stage1_trials = []
    for spec in manifest.get("stage1", {}).get("candidates", []):
        trial_root = root / "stage1" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage1_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    stage2_trials = []
    for spec in manifest.get("stage2", {}).get("candidates", []):
        trial_root = root / "stage2" / f"candidate_{spec['candidate_id']:03d}"
        state = (
            "complete"
            if (trial_root / "result.json").is_file()
            else "failed"
            if (trial_root / "failure.json").is_file()
            else "pending"
        )
        stage2_trials.append(
            {"candidate_id": spec["candidate_id"], "name": spec["name"], "state": state}
        )
    value = {
        "campaign_id": manifest["campaign_id"],
        "complete": all(item["state"] == "complete" for item in trials),
        "trials": trials,
        "canary": {
            "complete": all(item["state"] == "complete" for item in trials),
            "trials": trials,
        },
        "stage1": {
            "complete": bool(stage1_trials)
            and all(item["state"] == "complete" for item in stage1_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage1_trials),
            "n_total": len(stage1_trials),
            "trials": stage1_trials,
        },
        "stage2": {
            "complete": bool(stage2_trials)
            and all(item["state"] == "complete" for item in stage2_trials),
            "n_complete": sum(item["state"] == "complete" for item in stage2_trials),
            "n_total": len(stage2_trials),
            "trials": stage2_trials,
        },
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
    stage1_parser = subparsers.add_parser("run-stage1")
    stage1_parser.add_argument("--root", type=Path, required=True)
    stage1_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage1_parser = subparsers.add_parser("collect-stage1")
    collect_stage1_parser.add_argument("--root", type=Path, required=True)
    stage1_status_parser = subparsers.add_parser("stage1-status")
    stage1_status_parser.add_argument("--root", type=Path, required=True)
    stage2_parser = subparsers.add_parser("run-stage2")
    stage2_parser.add_argument("--root", type=Path, required=True)
    stage2_parser.add_argument("--candidate-id", type=int, required=True)
    collect_stage2_parser = subparsers.add_parser("collect-stage2")
    collect_stage2_parser.add_argument("--root", type=Path, required=True)
    stage2_status_parser = subparsers.add_parser("stage2-status")
    stage2_status_parser.add_argument("--root", type=Path, required=True)
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
    elif args.command == "run-stage1":
        run_stage1(args.root, args.candidate_id)
    elif args.command == "collect-stage1":
        collect_stage1(args.root)
    elif args.command == "stage1-status":
        status(args.root)
    elif args.command == "run-stage2":
        run_stage2(args.root, args.candidate_id)
    elif args.command == "collect-stage2":
        collect_stage2(args.root)
    elif args.command == "stage2-status":
        status(args.root)
    elif args.command == "status":
        status(args.root)


if __name__ == "__main__":
    main()
