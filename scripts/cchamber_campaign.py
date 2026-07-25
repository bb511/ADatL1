#!/usr/bin/env python3
"""Design, execute, and collect the production Causal Chamber campaign.

The search design is intentionally non-adaptive: every detector receives one
versioned Sobol candidate pool, and every candidate trajectory records all five
label-free model-selection proxies.  Intervention labels and the held-out test
split are disabled during search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient
from scipy import stats
from scipy.stats import qmc

from scripts import generation
from scripts.paper_pipeline import _selected_checkpoint_path, aggregate_results
from src.data.components.causal_chamber import parse_intervention_name
from src.utils.pairing.table import load_pair_table

load_dotenv(REPO_ROOT / ".env")

MODELS = ("ae", "vae", "svdd", "realnvp")
STRATEGIES = (
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "cap_random",
    "drift",
    "wasserstein",
)
DEV_SEEDS = (101, 202, 303, 404, 505)
REPORTING_SEEDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010)
DATA_SEED = 314159
PAIRING_SEED = 271828
PAIR_ENCODER_SEEDS = (123, 456, 789, 101112, 131415)
POOL_SEED = 1729
DEFAULT_N_CANDIDATES = 65
DEFAULT_SEARCH_EPOCHS = 200
DEFAULT_RETRAIN_EPOCHS = 200
DATASET_ARCHIVE_MD5 = "476664d024f88e8b7640998bb5e9ee33"
EQUIVALENCE_MARGIN_AUPRC = 0.02
RANDOM_SENSITIVITY_SEEDS = (271829, 271830, 271831, 271832)
ENCODER_SENSITIVITY_SEEDS = PAIR_ENCODER_SEEDS[1:]
SEMANTIC_FAMILY = {
    "blue": "color",
    "green": "color",
    "red": "color",
}

METRICS = {
    "cap_metadata_nearest": (
        "val/summary/cap_metadata_nearest_ema_normal_vs_reference_normal",
        "maximize",
    ),
    "cap_encoder_nearest": (
        "val/summary/cap_encoder_nearest_ema_normal_vs_reference_normal",
        "maximize",
    ),
    "cap_random": (
        "val/summary/cap_random_ema_normal_vs_reference_normal",
        "maximize",
    ),
    "drift": ("val/summary/operational_drift_ema", "minimize"),
    "wasserstein": (
        "val/summary/w1dist_ema_normal_vs_reference_normal",
        "minimize",
    ),
}
SENSITIVITY_METRICS = {
    **{
        f"cap_random_seed{seed}": (
            f"val/summary/cap_random_seed{seed}_ema_normal_vs_reference_normal",
            "maximize",
        )
        for seed in RANDOM_SENSITIVITY_SEEDS
    },
    **{
        f"cap_encoder_seed{seed}": (
            f"val/summary/cap_encoder_seed{seed}_ema_normal_vs_reference_normal",
            "maximize",
        )
        for seed in ENCODER_SENSITIVITY_SEEDS
    },
}


@dataclass(frozen=True)
class LogRange:
    low: float
    high: float


SPACES: dict[str, dict[str, Sequence[Any] | LogRange]] = {
    "ae": {
        "trainer.gradient_clip_val": (0.0, 0.5, 1.0, 2.0, 5.0),
        "algorithm.delta": (0.5, 1.0, 3.0, 5.0, 7.0, 10.0),
        "algorithm.input_noise_std": (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        "algorithm.encoder.nodes": (
            [16, 8],
            [24, 8],
            [32, 16],
            [32, 16, 4],
            [32, 16, 8],
            [48, 24, 8],
            [48, 24, 16],
            [64, 32, 16],
        ),
        "algorithm.optimizer.lr": LogRange(3e-5, 3e-3),
        "algorithm.optimizer.betas": ([0.9, 0.999], [0.9, 0.99]),
        "algorithm.optimizer.weight_decay": (0.0, 1e-6, 1e-5, 1e-4, 1e-3),
    },
    "vae": {
        "trainer.gradient_clip_val": (0.0, 0.5, 1.0, 2.0),
        "algorithm.encoder.clamp_zlogvar_range": (
            [-20, 10],
            [-10, 6],
            [-8, 6],
            [-6, 4],
        ),
        "algorithm.optimizer.lr": LogRange(5e-5, 1e-3),
        "algorithm.optimizer.betas": ([0.9, 0.999], [0.9, 0.99]),
        "algorithm.optimizer.weight_decay": (0.0, 1e-6, 1e-5, 1e-4, 1e-3),
        "algorithm.kl_scale": (3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 1e-2, 3e-2, 1e-1),
        "algorithm.kl_warmup_frac": (0.0, 0.05, 0.1, 0.2, 0.3),
        "algorithm.encoder.nodes": (
            [16, 4],
            [16, 8],
            [24, 8],
            [32, 16, 4],
            [32, 16, 8],
            [48, 24, 8],
            [48, 24, 16],
            [64, 32, 16],
        ),
        "algorithm.encoder.activation": ("relu", "gelu", "silu"),
    },
    "svdd": {
        "algorithm.optimizer.lr": LogRange(3e-5, 3e-3),
        "trainer.gradient_clip_val": (0.0, 0.5, 1.0, 2.0),
        "algorithm.optimizer.betas": ([0.9, 0.999], [0.9, 0.99]),
        "algorithm.optimizer.weight_decay": (0.0, 1e-6, 1e-5, 1e-4, 1e-3),
        "algorithm.weight_decay": (0.0, 1e-8, 1e-7, 1e-6, 1e-5),
        "algorithm.soft_boundary": (False, True),
        "algorithm.nu": (0.01, 0.05, 0.1, 0.2),
        "algorithm.center_init_method": ("mean", "zeros"),
        "algorithm.encoder.nodes": (
            [16, 8],
            [24, 8],
            [32, 16],
            [32, 16, 8],
            [32, 16, 16],
            [48, 24, 8],
            [48, 24, 16],
            [64, 32, 16],
        ),
        "algorithm.encoder.activation": ("relu", "gelu"),
    },
    "realnvp": {
        "algorithm.optimizer.lr": LogRange(3e-5, 3e-3),
        "trainer.gradient_clip_val": (0.0, 0.5, 1.0, 2.0),
        "algorithm.optimizer.betas": ([0.9, 0.999], [0.9, 0.99]),
        "algorithm.optimizer.weight_decay": (0.0, 1e-6, 1e-5, 1e-4, 1e-3),
        "algorithm.flow.n_flows": (4, 6, 8),
        "algorithm.flow.hidden_dim": (24, 32, 48, 64),
        "algorithm.flow.n_hidden_layers": (1, 2),
        "algorithm.flow.activation": ("relu", "gelu"),
        "algorithm.flow.noise_scale": (0.0, 1e-4, 1e-3, 1e-2),
        "algorithm.flow.scale_clamp": (3.0, 5.0),
    },
}

BASELINES: dict[str, dict[str, Any]] = {
    "ae": {
        "trainer.gradient_clip_val": 0.0,
        "algorithm.delta": 1.0,
        "algorithm.input_noise_std": 0.0,
        "algorithm.encoder.nodes": [32, 16, 4],
        "algorithm.optimizer.lr": 1e-3,
        "algorithm.optimizer.betas": [0.9, 0.999],
        "algorithm.optimizer.weight_decay": 1e-4,
    },
    "vae": {
        "trainer.gradient_clip_val": 0.0,
        "algorithm.encoder.clamp_zlogvar_range": [-10, 6],
        "algorithm.optimizer.lr": 1e-3,
        "algorithm.optimizer.betas": [0.9, 0.999],
        "algorithm.optimizer.weight_decay": 1e-4,
        "algorithm.kl_scale": 0.1,
        "algorithm.kl_warmup_frac": 0.1,
        "algorithm.encoder.nodes": [32, 16, 4],
        "algorithm.encoder.activation": "relu",
    },
    "svdd": {
        "algorithm.optimizer.lr": 1e-3,
        "trainer.gradient_clip_val": 0.0,
        "algorithm.optimizer.betas": [0.9, 0.999],
        "algorithm.optimizer.weight_decay": 1e-4,
        "algorithm.weight_decay": 0.0,
        "algorithm.soft_boundary": False,
        "algorithm.nu": 0.1,
        "algorithm.center_init_method": "mean",
        "algorithm.encoder.nodes": [32, 16, 8],
        "algorithm.encoder.activation": "relu",
    },
    "realnvp": {
        "algorithm.optimizer.lr": 1e-3,
        "trainer.gradient_clip_val": 0.0,
        "algorithm.optimizer.betas": [0.9, 0.999],
        "algorithm.optimizer.weight_decay": 1e-4,
        "algorithm.flow.n_flows": 6,
        "algorithm.flow.hidden_dim": 32,
        "algorithm.flow.n_hidden_layers": 2,
        "algorithm.flow.activation": "relu",
        "algorithm.flow.noise_scale": 0.0,
        "algorithm.flow.scale_clamp": 5.0,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def _attempt_id() -> str:
    """Return a sortable attempt identifier unique within a Slurm task."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    job = os.environ.get("SLURM_JOB_ID", "local")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "none")
    return f"{timestamp}_j{job}_a{task}_p{os.getpid()}"


def _validate_marker(
    path: Path,
    expected: Mapping[str, Any],
    fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load a completion marker only after validating identity and file hashes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"Completion marker identity mismatch in {path}: "
                f"{key}={value.get(key)!r}, expected {expected_value!r}."
            )
    for path_key, hash_key in (fingerprints or {}).items():
        artifact = Path(str(value[path_key]))
        if not artifact.is_file() or _sha256(artifact) != str(value[hash_key]):
            raise ValueError(f"Completion marker artifact mismatch: {artifact}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _campaign_manifest(root: Path) -> dict[str, Any]:
    path = root / "campaign.json"
    if not path.is_file():
        raise FileNotFoundError(f"Campaign has not been designed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_campaign_revision(campaign: Mapping[str, Any]) -> None:
    if _git("rev-parse", "HEAD") != str(campaign["git_commit"]):
        raise RuntimeError("Worktree HEAD does not match the immutable campaign commit.")
    if _git("status", "--porcelain"):
        raise RuntimeError("Campaign execution requires a clean deployment worktree.")
    if "dataset_files" in campaign:
        current = []
        for record in campaign["dataset_files"]:
            path = Path(str(record["path"]))
            if not path.is_file():
                raise FileNotFoundError(path)
            current.append(
                {
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        tree_hash = hashlib.sha256(_canonical_json(current).encode("utf-8")).hexdigest()
        if tree_hash != campaign["dataset_tree_sha256"]:
            raise RuntimeError("Extracted Causal Chamber CSV fingerprints changed.")


def _candidate_manifest(root: Path, model: str) -> list[dict[str, Any]]:
    path = root / "design" / f"{model}_candidates.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return value


def _sample_pool(model: str, n_candidates: int) -> list[dict[str, Any]]:
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least two.")
    space = SPACES[model]
    dimensions = len(space)
    exponent = max(1, math.ceil(math.log2(n_candidates - 1)))
    unit = qmc.Sobol(
        d=dimensions,
        scramble=True,
        seed=POOL_SEED + MODELS.index(model),
    ).random_base2(exponent)

    candidates = [BASELINES[model]]
    for point in unit:
        params: dict[str, Any] = {}
        for coordinate, (name, domain) in zip(point, space.items()):
            if isinstance(domain, LogRange):
                value = math.exp(
                    math.log(domain.low)
                    + float(coordinate) * (math.log(domain.high) - math.log(domain.low))
                )
            else:
                index = min(int(float(coordinate) * len(domain)), len(domain) - 1)
                value = domain[index]
            params[name] = value
        if model == "svdd" and not params["algorithm.soft_boundary"]:
            params["algorithm.nu"] = 0.1
        candidates.append(params)
        if len(candidates) == n_candidates:
            break

    records = []
    for index, params in enumerate(candidates):
        records.append(
            {
                "model": model,
                "candidate_id": f"{index:03d}",
                "params": params,
                "params_sha256": hashlib.sha256(
                    _canonical_json(params).encode("utf-8")
                ).hexdigest(),
                "baseline": index == 0,
            }
        )
    return records


def _tracking_uri(root: Path) -> str:
    return f"file:{(root / 'logs' / 'mlflow' / 'mlruns').resolve()}"


def _experiment_name(campaign_id: str, stage: str, model: str) -> str:
    return f"cchamber_{campaign_id}_{stage}_{model}"


def design(root: Path, campaign_id: str, n_candidates: int) -> None:
    root = root.resolve()
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        raise RuntimeError("Refusing to design a production campaign from a dirty worktree.")
    data_root = Path(
        os.environ.get(
            "DATA_DIR",
            "/iopsstor/scratch/cscs/vjimenez/adatl1/data",
        )
    ).expanduser()
    archive = data_root / "causal_chamber" / "lt_interventions_standard_v1.zip"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_md5 = _md5(archive)
    if archive_md5 != DATASET_ARCHIVE_MD5:
        raise ValueError(
            f"Causal Chamber archive MD5 is {archive_md5}, expected {DATASET_ARCHIVE_MD5}."
        )
    data_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "data" / "causal_chamber.yaml").read_text(encoding="utf-8")
    )
    interventions = [str(value) for value in data_config["signal_experiments"]]
    if len(interventions) != 58 or len(set(interventions)) != 58:
        raise ValueError("Causal Chamber config must contain 58 unique interventions.")
    dataset_dir = data_root / "causal_chamber" / "lt_interventions_standard_v1"
    csv_paths = sorted(dataset_dir.glob("*.csv"))
    if len(csv_paths) != 59:
        raise ValueError(f"Expected 59 extracted Causal Chamber CSVs, found {len(csv_paths)}.")
    dataset_files = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in csv_paths
    ]
    dataset_tree_sha256 = hashlib.sha256(
        _canonical_json(dataset_files).encode("utf-8")
    ).hexdigest()
    root.mkdir(parents=True, exist_ok=False)

    pools: dict[str, str] = {}
    pool_audits: dict[str, dict[str, Any]] = {}
    stress_candidates: dict[str, str] = {}
    for model in MODELS:
        records = _sample_pool(model, n_candidates)
        path = root / "design" / f"{model}_candidates.json"
        _atomic_json(path, records)
        pools[model] = _sha256(path)
        parameter_hashes = [str(record["params_sha256"]) for record in records]
        if len(set(parameter_hashes)) != len(parameter_hashes):
            raise ValueError(f"{model} candidate pool contains duplicate parameter sets.")
        categorical_counts: dict[str, dict[str, int]] = {}
        continuous_ranges: dict[str, dict[str, float]] = {}
        for name, domain in SPACES[model].items():
            values = [record["params"][name] for record in records]
            if isinstance(domain, LogRange):
                numeric = np.asarray(values, dtype=float)
                continuous_ranges[name] = {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                }
            else:
                categorical_counts[name] = {
                    _canonical_json(value): int(
                        sum(_canonical_json(item) == _canonical_json(value) for item in values)
                    )
                    for value in domain
                }
        pool_audits[model] = {
            "n_candidates": len(records),
            "n_unique_parameter_sets": len(set(parameter_hashes)),
            "baseline_candidate_id": "000",
            "categorical_counts": categorical_counts,
            "continuous_ranges": continuous_ranges,
        }

        def runtime_stress_score(record: Mapping[str, Any]) -> tuple[float, ...]:
            params = record["params"]
            nodes = params.get("algorithm.encoder.nodes", [])
            return (
                float(params.get("algorithm.flow.n_flows", 0)),
                float(params.get("algorithm.flow.hidden_dim", 0)),
                float(params.get("algorithm.flow.n_hidden_layers", 0)),
                float(sum(int(value) for value in nodes)),
                float(len(nodes)),
                float(params["algorithm.optimizer.lr"]),
            )

        stress_candidates[model] = str(max(records, key=runtime_stress_score)["candidate_id"])

    tracking_uri = _tracking_uri(root)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    for stage in ("search", "retrain", "evaluate", "pairing"):
        for model in MODELS if stage != "pairing" else ("encoder",):
            name = _experiment_name(campaign_id, stage, model)
            if client.get_experiment_by_name(name) is None:
                client.create_experiment(
                    name,
                    artifact_location=str((root / "mlflow" / "artifacts" / name).resolve()),
                    tags={
                        "campaign_id": campaign_id,
                        "stage": stage,
                        "git_commit": commit,
                    },
                )

    manifest = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO_ROOT),
        "git_commit": commit,
        "git_branch": _git("branch", "--show-current"),
        "dataset": "lt_interventions_standard_v1",
        "dataset_archive": str(archive.resolve()),
        "dataset_archive_md5": archive_md5,
        "dataset_archive_sha256": _sha256(archive),
        "dataset_files": dataset_files,
        "dataset_tree_sha256": dataset_tree_sha256,
        "feature_set": "readouts",
        "n_features": 11,
        "interventions": interventions,
        "models": list(MODELS),
        "strategies": list(STRATEGIES),
        "n_candidates_per_model": n_candidates,
        "development_seeds": list(DEV_SEEDS),
        "reporting_seeds": list(REPORTING_SEEDS),
        "pair_encoder_seeds": list(PAIR_ENCODER_SEEDS),
        "data_seed": DATA_SEED,
        "pairing_seed": PAIRING_SEED,
        "pool_seed": POOL_SEED,
        "search_epochs": DEFAULT_SEARCH_EPOCHS,
        "retrain_epochs": DEFAULT_RETRAIN_EPOCHS,
        "pool_sha256": pools,
        "pool_audit": pool_audits,
        "stress_candidate_by_model": stress_candidates,
        "tracking_uri": tracking_uri,
        "intervention_labels_sealed_during_search": True,
        "test_pairing_diagnostics_not_used_for_selection": True,
        "sensitivity_design": {
            "random_pairing_seeds": [PAIRING_SEED, *RANDOM_SENSITIVITY_SEEDS],
            "encoder_seeds": list(PAIR_ENCODER_SEEDS),
            "metric_definitions": {
                name: {"metric_name": metric, "direction": direction}
                for name, (metric, direction) in SENSITIVITY_METRICS.items()
            },
        },
        "primary_inference": {
            "metric": "auprc",
            "estimand": "intervention-weighted mean within reporting seed",
            "equivalence_margin_absolute": EQUIVALENCE_MARGIN_AUPRC,
            "superiority_contrasts": [
                ["cap_metadata_nearest", "cap_random"],
                ["cap_encoder_nearest", "cap_random"],
            ],
            "equivalence_contrast": [
                "cap_metadata_nearest",
                "cap_encoder_nearest",
            ],
            "multiplicity": "Holm correction within each prespecified contrast family",
        },
    }
    _atomic_json(root / "campaign.json", manifest)
    _write_slurm_scripts(root, campaign_id, n_candidates, stress_candidates)
    print(root / "campaign.json")


def _write_slurm_scripts(
    root: Path,
    campaign_id: str,
    n_candidates: int,
    stress_candidates: Mapping[str, str],
) -> None:
    """Write fixed-commit, four-GPU packed launch scripts into the campaign."""
    uv_path = shutil.which("uv")
    if uv_path is None:
        raise FileNotFoundError("uv is not available while generating Slurm scripts.")
    data_dir = os.environ.get(
        "DATA_DIR",
        "/iopsstor/scratch/cscs/vjimenez/adatl1/data",
    )
    common = textwrap.dedent(
        f"""\
        # Generated by scripts/cchamber_campaign.py.
        set -euo pipefail
        REPO={REPO_ROOT}
        CAMPAIGN_ROOT={root}
        cd "$REPO"
        test "$(git rev-parse HEAD)" = "$({sys.executable} -c 'import json; print(json.load(open(\"'$CAMPAIGN_ROOT'/campaign.json\"))[\"git_commit\"])')"
        test -z "$(git status --porcelain)"
        export PROJECT_ROOT="$REPO"
        export DATA_DIR={shlex.quote(data_dir)}
        export LOG_DIR="$CAMPAIGN_ROOT/logs"
        export CHECKPOINT_DIR="$CAMPAIGN_ROOT/checkpoints"
        UV=({shlex.quote(uv_path)} run --frozen --no-sync python)
        """
    )
    pair_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=cch-pair-{campaign_id[-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=02:00:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --output={root}/slurm/%x-%j.out
        #SBATCH --error={root}/slurm/%x-%j.err
        """
    )
    pair_script = (
        pair_script
        + "\n"
        + common
        + textwrap.dedent(
            f"""
        mkdir -p "$CAMPAIGN_ROOT/slurm"
        seeds=({" ".join(map(str, PAIR_ENCODER_SEEDS))})
        pids=()
        for seed in "${{seeds[@]:0:4}}"; do
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
                "${{UV[@]}}" scripts/cchamber_campaign.py run-pairing-encoder \
                --root "$CAMPAIGN_ROOT" --encoder-seed "$seed" &
            pids+=("$!")
        done
        status=0
        for pid in "${{pids[@]}}"; do
            wait "$pid" || status=1
        done
        test "$status" -eq 0
        srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
            "${{UV[@]}}" scripts/cchamber_campaign.py run-pairing-encoder \
            --root "$CAMPAIGN_ROOT" --encoder-seed "${{seeds[4]}}"
        "${{UV[@]}}" scripts/cchamber_campaign.py collect-pairing \
            --root "$CAMPAIGN_ROOT"
        """
        )
    )
    _write_executable(root / "slurm" / "pairing.sbatch", pair_script)

    calibration_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=cch-cal-{campaign_id[-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=debug
        #SBATCH --time=01:30:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --output={root}/slurm/%x-%j.out
        #SBATCH --error={root}/slurm/%x-%j.err
        """
    )
    calibration_script = (
        calibration_script
        + "\n"
        + common
        + textwrap.dedent(
            f"""
        PAIR_TABLE="$CAMPAIGN_ROOT/pairing/seed_123/validate_pairs.pt"
        test -f "$PAIR_TABLE"
        mkdir -p "$CAMPAIGN_ROOT/slurm"
        declare -A stress=(
            [ae]={stress_candidates["ae"]}
            [vae]={stress_candidates["vae"]}
            [svdd]={stress_candidates["svdd"]}
            [realnvp]={stress_candidates["realnvp"]}
        )
        pids=()
        for model in ae vae svdd realnvp; do
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
                "${{UV[@]}}" scripts/cchamber_campaign.py run-candidate \
                --root "$CAMPAIGN_ROOT" --model "$model" \
                --candidate-id "${{stress[$model]}}" \
                --seeds 101 --pair-table "$PAIR_TABLE" &
            pids+=("$!")
        done
        status=0
        for pid in "${{pids[@]}}"; do
            wait "$pid" || status=1
        done
        exit "$status"
        """
        )
    )
    _write_executable(root / "slurm" / "calibration.sbatch", calibration_script)

    search_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name=cch-search-{campaign_id[-12:]}
        #SBATCH --account=a0166
        #SBATCH --partition=normal
        #SBATCH --time=12:00:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=72
        #SBATCH --gpus-per-node=4
        #SBATCH --array=0-{n_candidates - 1}%8
        #SBATCH --output={root}/slurm/%x-%A_%a.out
        #SBATCH --error={root}/slurm/%x-%A_%a.err
        """
    )
    search_script = (
        search_script
        + "\n"
        + common
        + textwrap.dedent(
            """
        PAIR_TABLE="$CAMPAIGN_ROOT/pairing/seed_123/validate_pairs.pt"
        test -f "$PAIR_TABLE"
        mkdir -p "$CAMPAIGN_ROOT/slurm"
        printf -v candidate_id '%03d' "$SLURM_ARRAY_TASK_ID"
        pids=()
        for model in ae vae svdd realnvp; do
            srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
                "${UV[@]}" scripts/cchamber_campaign.py run-candidate \
                --root "$CAMPAIGN_ROOT" --model "$model" \
                --candidate-id "$candidate_id" --pair-table "$PAIR_TABLE" &
            pids+=("$!")
        done
        status=0
        for pid in "${pids[@]}"; do
            wait "$pid" || status=1
        done
        exit "$status"
        """
        )
    )
    _write_executable(root / "slurm" / "search.sbatch", search_script)

    for stage, count, timeout in (
        ("retrain", len(MODELS) * len(STRATEGIES) * len(REPORTING_SEEDS), "06:00:00"),
        ("evaluate", len(MODELS) * len(STRATEGIES) * len(REPORTING_SEEDS), "04:00:00"),
    ):
        n_array = math.ceil(count / 4)
        stage_script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name=cch-{stage}-{campaign_id[-10:]}
            #SBATCH --account=a0166
            #SBATCH --partition=normal
            #SBATCH --time={timeout}
            #SBATCH --nodes=1
            #SBATCH --ntasks=4
            #SBATCH --cpus-per-task=72
            #SBATCH --gpus-per-node=4
            #SBATCH --array=0-{n_array - 1}%8
            #SBATCH --output={root}/slurm/%x-%A_%a.out
            #SBATCH --error={root}/slurm/%x-%A_%a.err
            """
        )
        stage_script += (
            "\n"
            + common
            + textwrap.dedent(
                f"""
            mkdir -p "$CAMPAIGN_ROOT/slurm"
            pids=()
            for offset in 0 1 2 3; do
                index=$((SLURM_ARRAY_TASK_ID * 4 + offset))
                if (( index >= {count} )); then
                    continue
                fi
                srun --exclusive --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
                    "${{UV[@]}}" scripts/cchamber_campaign.py run-{stage} \
                    --root "$CAMPAIGN_ROOT" --index "$index" &
                pids+=("$!")
            done
        status=0
        for pid in "${{pids[@]}}"; do
            wait "$pid" || status=1
        done
        exit "$status"
        """
            )
        )
        _write_executable(root / "slurm" / f"{stage}.sbatch", stage_script)


def run_pairing_encoder(root: Path, encoder_seed: int) -> None:
    """Train one frozen pairing encoder and create its disjoint val/test tables."""
    root = root.resolve()
    campaign = _campaign_manifest(root)
    _assert_campaign_revision(campaign)
    if int(encoder_seed) not in set(campaign["pair_encoder_seeds"]):
        raise ValueError(f"Seed {encoder_seed} is not a campaign pairing-encoder seed.")
    expected_commit = str(campaign["git_commit"])

    seed = int(encoder_seed)
    campaign_id = str(campaign["campaign_id"])
    experiment_name = _experiment_name(campaign_id, "pairing", "encoder")
    logical_run_name = f"pairing_encoder_seed_{seed}"
    run_root = root / "pairing" / f"seed_{seed}"
    result_path = run_root / "summary.json"
    if result_path.is_file():
        _validate_marker(
            result_path,
            {
                "campaign_id": campaign_id,
                "git_commit": expected_commit,
                "encoder_seed": seed,
            },
            {
                "encoder_checkpoint": "encoder_checkpoint_sha256",
                "validation_table": "validation_table_sha256",
                "test_table": "test_table_sha256",
            },
        )
        print(f"[resume] {result_path}")
        return

    attempt = _attempt_id()
    run_name = f"{logical_run_name}_{attempt}"
    checkpoint = root / "checkpoints" / f"{campaign_id}_pairing_encoder" / run_name / "last.ckpt"
    tracking_uri = str(campaign["tracking_uri"])
    pair_tags = {
        "campaign_id": campaign_id,
        "stage": "pairing",
        "model": "ae_pairing_encoder",
        "encoder_seed": str(seed),
        "logical_run_name": logical_run_name,
        "attempt_id": attempt,
        "data_seed": str(campaign["data_seed"]),
        "git_commit": expected_commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
    }
    tag_override = "{" + ",".join(f"{key}:{value}" for key, value in pair_tags.items()) + "}"
    train_command = [
        sys.executable,
        "src/train.py",
        "experiment=cchamber/ae_pairing",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"seed={seed}",
        f"data.seed={int(campaign['data_seed'])}",
        "data.pairing_strategy=metadata_nearest",
        "data.max_val_batches=-1",
        "data.signal_experiments=[]",
        f"experiment_name={campaign_id}_pairing_encoder",
        f"run_name={run_name}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.checkpoints_dir={root / 'checkpoints'}",
        f"hydra.run.dir={root / 'hydra' / 'pairing' / f'seed_{seed}' / attempt}",
        f"logger.mlflow.experiment_name={experiment_name}",
        f"logger.mlflow.tags={tag_override}",
        "callbacks.log_data_mlflow=null",
        "extras.print_config=false",
        "trainer.min_epochs=100",
        "trainer.max_epochs=100",
        "test=false",
    ]
    environment = os.environ.copy()
    environment["LOG_DIR"] = str(root / "logs")
    print("[run] " + " ".join(train_command), flush=True)
    subprocess.run(train_command, cwd=REPO_ROOT, env=environment, check=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    tables: dict[str, Path] = {}
    for stage in ("validate", "test"):
        destination = run_root / f"{stage}_pairs.pt"
        table_command = [
            sys.executable,
            "-m",
            "src.utils.pairing.build_pair_table",
            "--ckpt",
            str(checkpoint),
            "--out",
            str(destination),
            "--stage",
            stage,
            "--dataset-1",
            "normal",
            "--dataset-2",
            "reference_normal",
            "--pairing-mode",
            "one_to_one_nearest",
            "--k",
            "0",
            "--no-caliper",
            "--device",
            "cuda",
            "--overwrite",
            "experiment=cchamber/ae_pairing",
            "data.pairing_strategy=metadata_nearest",
            "data.max_val_batches=-1",
            "data.signal_experiments=[]",
            f"data.seed={int(campaign['data_seed'])}",
        ]
        print("[run] " + " ".join(table_command), flush=True)
        subprocess.run(table_command, cwd=REPO_ROOT, env=environment, check=True)
        tables[stage] = destination

    client = MlflowClient(tracking_uri=tracking_uri)
    run = _find_mlflow_run(client, experiment_name, run_name)
    summary = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "git_commit": expected_commit,
        "data_seed": int(campaign["data_seed"]),
        "encoder_seed": seed,
        "logical_run_name": logical_run_name,
        "attempt_id": attempt,
        "run_name": run_name,
        "encoder_checkpoint": str(checkpoint.resolve()),
        "encoder_checkpoint_sha256": _sha256(checkpoint),
        "validation_table": str(tables["validate"].resolve()),
        "validation_table_sha256": _sha256(tables["validate"]),
        "test_table": str(tables["test"].resolve()),
        "test_table_sha256": _sha256(tables["test"]),
        "mlflow_run_id": run.info.run_id,
        "mlflow_status": run.info.status,
    }
    _atomic_json(result_path, summary)
    for artifact in (checkpoint, tables["validate"], tables["test"], result_path):
        client.log_artifact(run.info.run_id, str(artifact), artifact_path="pairing")
    client.set_tag(run.info.run_id, "pairing_summary_sha256", _sha256(result_path))
    print(result_path)


def collect_pairing(root: Path) -> None:
    """Compare encoder-seed tables and freeze seed 123 as the primary table."""
    root = root.resolve()
    campaign = _campaign_manifest(root)
    _assert_campaign_revision(campaign)
    summaries = []
    diagnostics: list[dict[str, Any]] = []
    for seed in campaign["pair_encoder_seeds"]:
        path = root / "pairing" / f"seed_{int(seed)}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(summary)
        for split, key in (
            ("validate", "validation_table"),
            ("test", "test_table"),
        ):
            table = load_pair_table(summary[key], expected_split=split)
            distance = table["distance"].detach().cpu().numpy().astype(float)
            coverage = float(table["metadata"]["coverage"])
            if not math.isclose(coverage, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Pairing seed {seed}/{split} has incomplete coverage: {coverage}"
                )
            diagnostics.append(
                {
                    "encoder_seed": int(seed),
                    "split": split,
                    "n_pairs": int(len(distance)),
                    "coverage": coverage,
                    "distance_mean": float(np.mean(distance)),
                    "distance_median": float(np.median(distance)),
                    "distance_q90": float(np.quantile(distance, 0.9)),
                    "distance_max": float(np.max(distance)),
                    "source_1_sha256": table["metadata"]["source_1_sha256"],
                    "source_2_sha256": table["metadata"]["source_2_sha256"],
                    "encoder_checkpoint_sha256": table["metadata"]["encoder_checkpoint_sha256"],
                }
            )

    output_dir = root / "pairing" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    for split, key in (
        ("validate", "validation_table"),
        ("test", "test_table"),
    ):
        output = output_dir / f"{split}_stability.json"
        command = [
            sys.executable,
            "-m",
            "src.utils.pairing.compare_pair_tables",
            "--tables",
            *[str(summary[key]) for summary in summaries],
            "--out",
            str(output),
            "--overwrite",
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        comparisons[split] = json.loads(output.read_text(encoding="utf-8"))

    primary = next(
        summary
        for summary in summaries
        if int(summary["encoder_seed"]) == int(campaign["pair_encoder_seeds"][0])
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "selection_rule": (
            "The first prespecified encoder seed is primary; anomaly labels and "
            "pair-overlap outcomes are not used for selection."
        ),
        "primary_encoder_seed": int(primary["encoder_seed"]),
        "primary_validation_table": primary["validation_table"],
        "primary_validation_table_sha256": primary["validation_table_sha256"],
        "primary_test_table": primary["test_table"],
        "primary_test_table_sha256": primary["test_table_sha256"],
        "encoder_runs": summaries,
        "table_diagnostics": diagnostics,
        "stability": comparisons,
    }
    output = output_dir / "pairing_manifest.json"
    _atomic_json(output, manifest)
    client = MlflowClient(tracking_uri=str(campaign["tracking_uri"]))
    for summary in summaries:
        run_id = summary["mlflow_run_id"]
        for artifact in (
            output,
            output_dir / "validate_stability.json",
            output_dir / "test_stability.json",
        ):
            client.log_artifact(run_id, str(artifact), artifact_path="pairing_audit")
        client.set_tag(run_id, "pairing_manifest_sha256", _sha256(output))
    print(output)


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _find_mlflow_run(
    client: MlflowClient,
    experiment_name: str,
    run_name: str,
) -> Any:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment is missing: {experiment_name}")
    escaped = run_name.replace("'", "\\'")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{escaped}'",
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    finished = [run for run in runs if run.info.status == "FINISHED"]
    if not finished:
        raise RuntimeError(
            f"Expected at least one FINISHED MLflow run named {run_name}, found none."
        )
    # Retries can legitimately leave more than one finished run if postprocessing
    # failed after training. Search order is newest-first; provenance records the
    # exact chosen run ID.
    return finished[0]


def run_candidate(
    root: Path,
    model: str,
    candidate_id: str,
    seeds: Sequence[int],
    pair_table: Path,
) -> None:
    root = root.resolve()
    campaign = _campaign_manifest(root)
    _assert_campaign_revision(campaign)
    expected_commit = str(campaign["git_commit"])
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    candidates = {
        str(record["candidate_id"]): record for record in _candidate_manifest(root, model)
    }
    if candidate_id not in candidates:
        raise ValueError(f"Unknown {model} candidate: {candidate_id}")
    pair_table = pair_table.expanduser().resolve()
    if not pair_table.is_file():
        raise FileNotFoundError(pair_table)
    pairing_manifest = json.loads(
        (root / "pairing" / "comparison" / "pairing_manifest.json").read_text(encoding="utf-8")
    )
    if _sha256(pair_table) != pairing_manifest["primary_validation_table_sha256"]:
        raise ValueError("Search pair table is not the prespecified primary validation table.")
    encoder_tables = {
        int(summary["encoder_seed"]): Path(summary["validation_table"])
        for summary in pairing_manifest["encoder_runs"]
    }
    encoder_table_hashes = {
        int(summary["encoder_seed"]): str(summary["validation_table_sha256"])
        for summary in pairing_manifest["encoder_runs"]
    }

    record = candidates[candidate_id]
    campaign_id = str(campaign["campaign_id"])
    experiment_name = _experiment_name(campaign_id, "search", model)
    tracking_uri = str(campaign["tracking_uri"])
    client = MlflowClient(tracking_uri=tracking_uri)

    for seed in seeds:
        if int(seed) not in set(campaign["development_seeds"]):
            raise ValueError(f"Seed {seed} is not in the campaign development seeds.")
        run_name = f"search_{model}_c{candidate_id}_s{int(seed)}"
        result_path = root / "search_results" / model / candidate_id / f"seed_{int(seed)}.json"
        failure_path = result_path.with_name(f"seed_{int(seed)}.failed.json")
        if result_path.is_file():
            _validate_marker(
                result_path,
                {
                    "campaign_id": campaign_id,
                    "model": model,
                    "candidate_id": candidate_id,
                    "seed": int(seed),
                    "params_sha256": record["params_sha256"],
                    "pool_sha256": campaign["pool_sha256"][model],
                    "git_commit": expected_commit,
                    "pair_table_sha256": pairing_manifest["primary_validation_table_sha256"],
                },
            )
            print(f"[resume] {result_path}", flush=True)
            continue
        if failure_path.is_file():
            print(f"[excluded] {failure_path}", flush=True)
            continue

        tags = {
            "campaign_id": campaign_id,
            "stage": "search",
            "model": model,
            "candidate_id": candidate_id,
            "model_seed": str(int(seed)),
            "data_seed": str(campaign["data_seed"]),
            "git_commit": expected_commit,
            "candidate_pool_sha256": str(campaign["pool_sha256"][model]),
            "pair_table_sha256": _sha256(pair_table),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "none"),
        }
        tag_override = "{" + ",".join(f"{key}:{value}" for key, value in tags.items()) + "}"
        command = [
            sys.executable,
            "src/train.py",
            f"experiment=cchamber/{model}_search",
            "trainer=gpu",
            "trainer.devices=[0]",
            f"trainer.min_epochs={int(campaign['search_epochs'])}",
            f"trainer.max_epochs={int(campaign['search_epochs'])}",
            f"seed={int(seed)}",
            f"data.seed={int(campaign['data_seed'])}",
            f"experiment_name={experiment_name}",
            f"run_name={run_name}",
            f"paths.log_dir={root / 'logs'}",
            f"paths.checkpoints_dir={root / 'checkpoints'}",
            f"hydra.run.dir={root / 'hydra' / 'search' / model / candidate_id / f'seed_{int(seed)}'}",
            f"logger.mlflow.tags={tag_override}",
            "extras.print_config=false",
            *[f"{name}={_hydra_value(value)}" for name, value in record["params"].items()],
        ]
        environment = os.environ.copy()
        environment["CCHAMBER_VALID_PAIR_TABLE"] = str(pair_table)
        for encoder_seed, table in encoder_tables.items():
            environment[f"CCHAMBER_VALID_PAIR_TABLE_SEED{encoder_seed}"] = str(table)
        environment["LOG_DIR"] = str(root / "logs")
        print("[run] " + " ".join(command), flush=True)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            attempt_path = (
                root
                / "attempts"
                / "search"
                / model
                / candidate_id
                / f"seed_{int(seed)}_{_attempt_id()}.json"
            )
            _atomic_json(
                attempt_path,
                {
                    "schema_version": 1,
                    "campaign_id": campaign_id,
                    "model": model,
                    "candidate_id": candidate_id,
                    "seed": int(seed),
                    "params_sha256": record["params_sha256"],
                    "pool_sha256": campaign["pool_sha256"][model],
                    "git_commit": expected_commit,
                    "classification": "transient_or_unclassified",
                    "reason": "training_process_nonzero_exit",
                    "returncode": int(completed.returncode),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
                },
            )
            raise subprocess.CalledProcessError(completed.returncode, command)

        run = _find_mlflow_run(client, experiment_name, run_name)
        extracted: dict[str, Any] = {}
        metric_error = None
        for strategy, (metric_name, direction) in {
            **METRICS,
            **SENSITIVITY_METRICS,
        }.items():
            history = client.get_metric_history(run.info.run_id, metric_name)
            # Lightning logs one validation sanity-check point before the requested
            # epochs. Retain exactly the final N training-epoch points.
            history = sorted(history, key=lambda item: (item.timestamp, item.step))
            history = history[-int(campaign["search_epochs"]) :]
            values = [float(item.value) for item in history if math.isfinite(item.value)]
            if len(values) != int(campaign["search_epochs"]):
                metric_error = (
                    f"{run_name}: {metric_name} has {len(values)} finite training points; "
                    f"expected {campaign['search_epochs']}."
                )
                break
            best = max(values) if direction == "maximize" else min(values)
            extracted[strategy] = {
                "metric_name": metric_name,
                "direction": direction,
                "best": best,
                "last": values[-1],
                "n_points": len(values),
            }
            client.log_metric(run.info.run_id, f"campaign/best/{strategy}", best)
        if metric_error is not None:
            _atomic_json(
                failure_path,
                {
                    "schema_version": 1,
                    "campaign_id": campaign_id,
                    "model": model,
                    "candidate_id": candidate_id,
                    "seed": int(seed),
                    "params_sha256": record["params_sha256"],
                    "pool_sha256": campaign["pool_sha256"][model],
                    "git_commit": expected_commit,
                    "reason": "invalid_metric_history",
                    "detail": metric_error,
                    "mlflow_run_id": run.info.run_id,
                },
            )
            client.set_tag(run.info.run_id, "campaign_candidate_valid", "false")
            continue

        result = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "model": model,
            "candidate_id": candidate_id,
            "seed": int(seed),
            "params": record["params"],
            "params_sha256": record["params_sha256"],
            "pool_sha256": campaign["pool_sha256"][model],
            "git_commit": expected_commit,
            "pair_table": str(pair_table),
            "pair_table_sha256": _sha256(pair_table),
            "encoder_validation_table_sha256_by_seed": {
                str(seed): digest for seed, digest in encoder_table_hashes.items()
            },
            "random_pairing_seeds": [PAIRING_SEED, *RANDOM_SENSITIVITY_SEEDS],
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment": experiment_name,
            "mlflow_run_id": run.info.run_id,
            "mlflow_status": run.info.status,
            "metrics": {name: extracted[name] for name in METRICS},
            "sensitivity_metrics": {name: extracted[name] for name in SENSITIVITY_METRICS},
        }
        _atomic_json(result_path, result)
        client.log_artifact(
            run.info.run_id,
            str(result_path),
            artifact_path="campaign",
        )
        client.set_tag(run.info.run_id, "campaign_result_sha256", _sha256(result_path))
        client.set_tag(run.info.run_id, "campaign_candidate_valid", "true")


def collect_candidates(root: Path) -> None:
    root = root.resolve()
    campaign = _campaign_manifest(root)
    _assert_campaign_revision(campaign)
    rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    pairing_manifest = json.loads(
        (root / "pairing" / "comparison" / "pairing_manifest.json").read_text(encoding="utf-8")
    )
    primary_pair_hash = pairing_manifest["primary_validation_table_sha256"]
    invalid_candidates: dict[str, set[str]] = {model: set() for model in MODELS}
    failure_records: list[dict[str, Any]] = []
    for model in MODELS:
        for record in _candidate_manifest(root, model):
            candidate_id = str(record["candidate_id"])
            for seed in campaign["development_seeds"]:
                result_path = (
                    root / "search_results" / model / candidate_id / f"seed_{int(seed)}.json"
                )
                failure_path = result_path.with_name(f"seed_{int(seed)}.failed.json")
                if failure_path.is_file():
                    invalid_candidates[model].add(candidate_id)
                    failure_records.append(json.loads(failure_path.read_text(encoding="utf-8")))
                elif not result_path.is_file():
                    missing.append(str(result_path))
    for model in MODELS:
        records = _candidate_manifest(root, model)
        for record in records:
            candidate_id = str(record["candidate_id"])
            if candidate_id in invalid_candidates[model]:
                continue
            for seed in campaign["development_seeds"]:
                path = root / "search_results" / model / candidate_id / f"seed_{int(seed)}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                if result["git_commit"] != campaign["git_commit"]:
                    raise ValueError(f"Commit mismatch in {path}")
                if result["pool_sha256"] != campaign["pool_sha256"][model]:
                    raise ValueError(f"Pool mismatch in {path}")
                if (
                    result["model"] != model
                    or result["candidate_id"] != candidate_id
                    or int(result["seed"]) != int(seed)
                    or result["params_sha256"] != record["params_sha256"]
                ):
                    raise ValueError(f"Candidate identity mismatch in {path}")
                if result["pair_table_sha256"] != primary_pair_hash:
                    raise ValueError(f"Primary pair-table mismatch in {path}")
                expected_encoder_hashes = {
                    str(summary["encoder_seed"]): str(summary["validation_table_sha256"])
                    for summary in pairing_manifest["encoder_runs"]
                }
                if result["encoder_validation_table_sha256_by_seed"] != expected_encoder_hashes:
                    raise ValueError(f"Encoder sensitivity pair-table mismatch in {path}")
                if result["random_pairing_seeds"] != [
                    PAIRING_SEED,
                    *RANDOM_SENSITIVITY_SEEDS,
                ]:
                    raise ValueError(f"Random sensitivity seed mismatch in {path}")
                for strategy in STRATEGIES:
                    rows.append(
                        {
                            "dataset": "cchamber",
                            "model": model,
                            "seed": int(seed),
                            "candidate_id": candidate_id,
                            "strategy": strategy,
                            "value": float(result["metrics"][strategy]["best"]),
                            "params_json": _canonical_json(record["params"]),
                            "mlflow_run_id": result["mlflow_run_id"],
                            "pool_sha256": result["pool_sha256"],
                            "pair_table_sha256": result["pair_table_sha256"],
                            "git_commit": result["git_commit"],
                        }
                    )
                for variant, metric in result["sensitivity_metrics"].items():
                    variant_seed = int(variant.rsplit("seed", 1)[1])
                    if variant.startswith("cap_encoder_"):
                        pair_table_sha256 = result["encoder_validation_table_sha256_by_seed"][
                            str(variant_seed)
                        ]
                        pairing_kind = "encoder_nearest"
                    else:
                        pair_table_sha256 = "not_applicable_seeded_permutation"
                        pairing_kind = "random"
                    sensitivity_rows.append(
                        {
                            "dataset": "cchamber",
                            "model": model,
                            "seed": int(seed),
                            "candidate_id": candidate_id,
                            "variant": variant,
                            "pairing_kind": pairing_kind,
                            "pairing_seed": variant_seed,
                            "pair_table_sha256": pair_table_sha256,
                            "value": float(metric["best"]),
                            "params_json": _canonical_json(record["params"]),
                            "mlflow_run_id": result["mlflow_run_id"],
                            "git_commit": result["git_commit"],
                        }
                    )
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Campaign is incomplete: {len(missing)} candidate-seed results are missing.\n"
            f"{preview}"
        )
    minimum_survivors = math.ceil(0.8 * int(campaign["n_candidates_per_model"]))
    survivor_counts = {
        model: int(campaign["n_candidates_per_model"]) - len(invalid_candidates[model])
        for model in MODELS
    }
    if any(count < minimum_survivors for count in survivor_counts.values()):
        raise RuntimeError(
            f"Too many globally invalid candidates; require >= {minimum_survivors}: "
            f"{survivor_counts}"
        )
    output = root / "selection" / "candidate_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sensitivity_output = root / "selection" / "pairing_proxy_sensitivity.csv"
    with sensitivity_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    _atomic_json(
        root / "selection" / "candidate_metrics_provenance.json",
        {
            "campaign": str((root / "campaign.json").resolve()),
            "campaign_sha256": _sha256(root / "campaign.json"),
            "candidate_metrics": str(output.resolve()),
            "candidate_metrics_sha256": _sha256(output),
            "pairing_proxy_sensitivity": str(sensitivity_output.resolve()),
            "pairing_proxy_sensitivity_sha256": _sha256(sensitivity_output),
            "n_rows": len(rows),
            "expected_rows_after_global_exclusion": sum(survivor_counts.values())
            * len(campaign["development_seeds"])
            * len(STRATEGIES),
            "minimum_surviving_candidates_per_model": minimum_survivors,
            "surviving_candidates_per_model": survivor_counts,
            "globally_excluded_candidates": {
                model: sorted(values) for model, values in invalid_candidates.items()
            },
            "failure_records": failure_records,
        },
    )
    print(output)
    print(sensitivity_output)


def select_candidates(root: Path) -> None:
    """Select one shared-pool candidate per model/strategy by mean seed rank."""
    root = root.resolve()
    campaign = _campaign_manifest(root)
    _assert_campaign_revision(campaign)
    source = root / "selection" / "candidate_metrics.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, dtype={"candidate_id": str})
    for model in MODELS:
        model_frame = frame[frame["model"] == model]
        candidate_sets = {
            (str(strategy), int(seed)): frozenset(rows["candidate_id"].astype(str))
            for (strategy, seed), rows in model_frame.groupby(
                ["strategy", "seed"],
                sort=True,
            )
        }
        expected_keys = {
            (strategy, int(seed))
            for strategy in STRATEGIES
            for seed in campaign["development_seeds"]
        }
        if set(candidate_sets) != expected_keys:
            raise ValueError(f"Incomplete strategy/seed coverage for model {model}.")
        reference = next(iter(candidate_sets.values()))
        if any(candidate_ids != reference for candidate_ids in candidate_sets.values()):
            raise ValueError(f"Candidate exclusion was not global for model {model}.")

    selected: list[dict[str, Any]] = []
    retrain: list[dict[str, Any]] = []
    for model in MODELS:
        for strategy in STRATEGIES:
            group = frame[(frame["model"] == model) & (frame["strategy"] == strategy)].copy()
            direction = METRICS[strategy][1]
            group["seed_rank"] = group.groupby("seed")["value"].rank(
                method="average",
                ascending=direction == "minimize",
            )
            summary = (
                group.groupby(["candidate_id", "params_json"], sort=True)
                .agg(
                    mean_rank=("seed_rank", "mean"),
                    mean_value=("value", "mean"),
                    std_value=("value", "std"),
                    n_development_seeds=("seed", "nunique"),
                )
                .reset_index()
            )
            if not (summary["n_development_seeds"] == len(campaign["development_seeds"])).all():
                raise ValueError(f"Incomplete development seeds for {model}/{strategy}.")
            summary = summary.sort_values(
                ["mean_rank", "mean_value", "candidate_id"],
                ascending=[True, direction == "minimize", True],
                kind="stable",
            )
            winner = summary.iloc[0]
            params = json.loads(str(winner["params_json"]))
            selected_row = {
                "dataset": "cchamber",
                "model": model,
                "strategy": strategy,
                "candidate_id": str(winner["candidate_id"]),
                "direction": direction,
                "mean_development_rank": float(winner["mean_rank"]),
                "mean_development_value": float(winner["mean_value"]),
                "std_development_value": float(winner["std_value"]),
                "n_development_seeds": int(winner["n_development_seeds"]),
                "params_json": _canonical_json(params),
                "pool_sha256": campaign["pool_sha256"][model],
                "git_commit": campaign["git_commit"],
            }
            selected.append(selected_row)
            for seed in campaign["reporting_seeds"]:
                retrain.append(
                    {
                        **selected_row,
                        "seed": int(seed),
                        "run_name": (
                            f"retrain_{model}_{strategy}_c{winner['candidate_id']}_s{int(seed)}"
                        ),
                        "params": params,
                    }
                )

    selected_path = root / "selection" / "selected_trials.csv"
    pd.DataFrame(selected).to_csv(selected_path, index=False)
    retrain_path = root / "selection" / "retrain_manifest.json"
    _atomic_json(retrain_path, retrain)
    _atomic_json(
        root / "selection" / "selection_provenance.json",
        {
            "candidate_metrics": str(source.resolve()),
            "candidate_metrics_sha256": _sha256(source),
            "selection_rule": (
                "Minimize mean within-seed candidate rank across all prespecified "
                "development model seeds; break ties by direction-aware mean metric "
                "and then lexical candidate_id."
            ),
            "development_seeds": list(campaign["development_seeds"]),
            "intervention_labels_used": False,
            "n_selected": len(selected),
            "n_retrains": len(retrain),
            "selected_trials_sha256": _sha256(selected_path),
            "retrain_manifest_sha256": _sha256(retrain_path),
        },
    )
    print(selected_path)
    print(retrain_path)


def _manifest_item(root: Path, name: str, index: int) -> dict[str, Any]:
    path = root / "selection" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list.")
    if not 0 <= int(index) < len(records):
        raise IndexError(f"Index {index} is outside {path} ({len(records)} records).")
    return dict(records[int(index)])


def _paper_spec(model: str, strategy: str) -> generation.ExperimentSpecification:
    """Build one production Causal Chamber endpoint specification."""
    return generation.make_experiment_specification(
        dataset=generation.Dataset.CCHAMBER,
        model=generation.Model(model),
        strategy=generation.Strategy(strategy),
        n_trials=1,
        seeds=(REPORTING_SEEDS[0],),
    )


def _replace_pair_env(overrides: Sequence[str], valid: Path, test: Path) -> list[str]:
    return [
        override.replace("$CCHAMBER_VALID_PAIR_TABLE", str(valid)).replace(
            "$CCHAMBER_TEST_PAIR_TABLE",
            str(test),
        )
        for override in overrides
    ]


def _validate_final_values(
    path: Path,
    *,
    checkpoint_stem: str,
    interventions: Sequence[str],
    metric: str,
) -> pd.DataFrame:
    """Validate one final callback table against the frozen result contract."""
    if not path.is_file():
        raise FileNotFoundError(path)
    values = pd.read_csv(path)
    required = {"checkpoint", "intervention", "metric", "value"}
    if not required.issubset(values.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(values.columns))}.")
    expected_interventions = set(map(str, interventions))
    actual_interventions = set(values["intervention"].astype(str))
    if (
        len(values) != len(expected_interventions)
        or actual_interventions != expected_interventions
    ):
        missing = sorted(expected_interventions - actual_interventions)
        extra = sorted(actual_interventions - expected_interventions)
        raise ValueError(
            f"{path}: intervention contract mismatch; rows={len(values)}, "
            f"missing={missing}, extra={extra}."
        )
    if set(values["checkpoint"].astype(str)) != {checkpoint_stem}:
        raise ValueError(f"{path}: checkpoint column does not match {checkpoint_stem}.")
    if set(values["metric"].astype(str)) != {metric}:
        raise ValueError(f"{path}: expected only metric {metric}.")
    numeric = pd.to_numeric(values["value"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or ((numeric < 0.0) | (numeric > 1.0)).any():
        raise ValueError(f"{path}: metric values must be finite and in [0, 1].")
    return values


def run_retrain(root: Path, index: int) -> None:
    """Retrain one selected configuration on one independent reporting seed."""
    root = root.resolve()
    campaign = _campaign_manifest(root)
    _assert_campaign_revision(campaign)
    item = _manifest_item(root, "retrain_manifest.json", index)
    pairing = json.loads(
        (root / "pairing" / "comparison" / "pairing_manifest.json").read_text(encoding="utf-8")
    )
    valid_table = Path(pairing["primary_validation_table"])
    test_table = Path(pairing["primary_test_table"])
    model = str(item["model"])
    strategy = str(item["strategy"])
    seed = int(item["seed"])
    logical_run_name = str(item["run_name"])
    result_path = root / "retrain_results" / f"{int(index):03d}.json"
    if result_path.is_file():
        _validate_marker(
            result_path,
            {
                "campaign_id": campaign["campaign_id"],
                "git_commit": campaign["git_commit"],
                "manifest_index": int(index),
                "model": model,
                "strategy": strategy,
                "seed": seed,
                "logical_run_name": logical_run_name,
            },
            {"checkpoint": "checkpoint_sha256"},
        )
        print(f"[resume] {result_path}")
        return

    attempt = _attempt_id()
    run_name = f"{logical_run_name}_{attempt}"
    experiment_dir = f"{campaign['campaign_id']}_retrain_{model}_{strategy}"
    checkpoint_run_dir = root / "checkpoints" / experiment_dir / run_name
    spec = _paper_spec(model, strategy)
    selected_overrides = [
        f"{name}={_hydra_value(value)}" for name, value in item["params"].items()
    ]
    overrides = generation.build_retrain_overrides(
        spec,
        seed=seed,
        trainer="gpu",
        devices="[0]",
        selected_overrides=selected_overrides,
        run_name=run_name,
    )
    experiment_name = _experiment_name(str(campaign["campaign_id"]), "retrain", model)
    tags = {
        "campaign_id": campaign["campaign_id"],
        "stage": "retrain",
        "model": model,
        "strategy": strategy,
        "candidate_id": item["candidate_id"],
        "model_seed": seed,
        "data_seed": campaign["data_seed"],
        "git_commit": campaign["git_commit"],
        "candidate_pool_sha256": item["pool_sha256"],
        "valid_pair_table_sha256": pairing["primary_validation_table_sha256"],
        "test_pair_table_sha256": pairing["primary_test_table_sha256"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "manifest_index": int(index),
    }
    tag_override = "{" + ",".join(f"{key}:{value}" for key, value in tags.items()) + "}"
    overrides.extend(
        [
            f"experiment_name={experiment_dir}",
            f"run_name={run_name}",
            f"paths.log_dir={root / 'logs'}",
            f"paths.checkpoints_dir={root / 'checkpoints'}",
            f"hydra.run.dir={root / 'hydra' / 'retrain' / f'{int(index):03d}' / attempt}",
            f"logger.mlflow.experiment_name={experiment_name}",
            f"logger.mlflow.tags={tag_override}",
            "callbacks.clear_ckpts=null",
            "callbacks.last_epoch_ckpt=null",
            "callbacks.stable_ascore_operational_ckpt=null",
            "evaluation.evaluator.ckpts.last=false",
            "~evaluation.evaluator.ckpts.single.ascore_operational",
            "extras.print_config=false",
        ]
    )
    overrides = _replace_pair_env(overrides, valid_table, test_table)
    command = [sys.executable, "src/train.py", *overrides]
    environment = os.environ.copy()
    environment["CCHAMBER_VALID_PAIR_TABLE"] = str(valid_table)
    environment["CCHAMBER_TEST_PAIR_TABLE"] = str(test_table)
    environment["LOG_DIR"] = str(root / "logs")
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)

    relative_checkpoint = _selected_checkpoint_path({"dataset": "cchamber", "strategy": strategy})
    checkpoint = checkpoint_run_dir / relative_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    client = MlflowClient(tracking_uri=str(campaign["tracking_uri"]))
    run = _find_mlflow_run(client, experiment_name, run_name)
    result = {
        **item,
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "manifest_index": int(index),
        "logical_run_name": logical_run_name,
        "attempt_id": attempt,
        "run_name": run_name,
        "experiment_dir": experiment_dir,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "mlflow_experiment": experiment_name,
        "mlflow_run_id": run.info.run_id,
        "mlflow_status": run.info.status,
        "valid_pair_table_sha256": pairing["primary_validation_table_sha256"],
        "test_pair_table_sha256": pairing["primary_test_table_sha256"],
    }
    _atomic_json(result_path, result)
    client.log_artifact(run.info.run_id, str(result_path), artifact_path="campaign")
    client.log_artifact(run.info.run_id, str(checkpoint), artifact_path="checkpoint")
    client.set_tag(run.info.run_id, "retrain_result_sha256", _sha256(result_path))
    client.set_tag(run.info.run_id, "checkpoint_sha256", _sha256(checkpoint))
    print(result_path)


def run_evaluate(root: Path, index: int) -> None:
    """Evaluate one exact selected checkpoint on all 58 sealed interventions."""
    root = root.resolve()
    campaign = _campaign_manifest(root)
    _assert_campaign_revision(campaign)
    retrain_path = root / "retrain_results" / f"{int(index):03d}.json"
    if not retrain_path.is_file():
        raise FileNotFoundError(retrain_path)
    item = json.loads(retrain_path.read_text(encoding="utf-8"))
    output = root / "evaluation_results" / f"{int(index):03d}.json"
    if output.is_file():
        _validate_marker(
            output,
            {
                "campaign_id": campaign["campaign_id"],
                "git_commit": campaign["git_commit"],
                "manifest_index": int(index),
            },
            {
                "values_csv": "values_csv_sha256",
                "efficiency_values_csv": "efficiency_values_csv_sha256",
                "checkpoint": "checkpoint_sha256",
            },
        )
        print(f"[resume] {output}")
        return
    if (
        item.get("campaign_id") != campaign["campaign_id"]
        or item.get("git_commit") != campaign["git_commit"]
        or int(item.get("manifest_index", -1)) != int(index)
    ):
        raise ValueError(f"Retrain result identity mismatch: {retrain_path}")
    checkpoint = Path(item["checkpoint"])
    if not checkpoint.is_file() or _sha256(checkpoint) != item["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint fingerprint mismatch: {checkpoint}")
    pairing = json.loads(
        (root / "pairing" / "comparison" / "pairing_manifest.json").read_text(encoding="utf-8")
    )
    valid_table = Path(pairing["primary_validation_table"])
    test_table = Path(pairing["primary_test_table"])
    if (
        item.get("valid_pair_table_sha256") != pairing["primary_validation_table_sha256"]
        or item.get("test_pair_table_sha256") != pairing["primary_test_table_sha256"]
    ):
        raise ValueError(f"Retrain result pair-table identity mismatch: {retrain_path}")
    model = str(item["model"])
    strategy = str(item["strategy"])
    seed = int(item["seed"])
    train_run_name = str(item["run_name"])
    attempt = _attempt_id()
    eval_run_name = f"evaluate_{model}_{strategy}_s{seed}_{attempt}"
    spec = _paper_spec(model, strategy)
    overrides = [
        f"experiment={spec.experiment}",
        *spec.fixed_overrides,
        *spec.strategy_overrides,
        *spec.disabled_overrides,
        f"experiment_name={item['experiment_dir']}",
        f"run_name={train_run_name}",
        "train=false",
        "test=true",
        f"seed={seed}",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"paths.log_dir={root / 'logs'}",
        f"paths.checkpoints_dir={root / 'checkpoints'}",
        f"hydra.run.dir={root / 'hydra' / 'evaluate' / f'{int(index):03d}' / attempt}",
        "~callbacks",
        "callbacks.log_data_mlflow=null",
        "callbacks.anomaly_eff=null",
        "callbacks.thres_drift=null",
        "callbacks.wasserstein_dist=null",
        "callbacks.cap_ref=null",
        "callbacks.stable_ascore_operational_ckpt=null",
        "callbacks.thres_drift_ema_ckpt=null",
        "callbacks.wasserstein_dist_ema_ckpt=null",
        "callbacks.cap_ref_ema_ckpt=null",
        "evaluation.evaluator.ckpts.last=false",
        "~evaluation.evaluator.ckpts.single.ascore_operational",
        (
            "++evaluation.callbacks.anomaly_efficiency="
            "{_target_:src.evaluation.callbacks.efficiency.AnomalyEfficiencyCallback,"
            "output_name:ascore/full,ds:${data.signal_experiments},target_rates:[0.01],"
            "pure_thres:false,cvar_summary:0.25,log_raw_mlflow:false,name:eff}"
        ),
        "extras.print_config=false",
    ]
    if strategy == "cap_encoder_nearest":
        overrides.append(f"+evaluation.callbacks.cap_ref.pairing_test_index_path={test_table}")
    experiment_name = _experiment_name(str(campaign["campaign_id"]), "evaluate", model)
    tags = {
        "campaign_id": campaign["campaign_id"],
        "stage": "evaluate",
        "model": model,
        "strategy": strategy,
        "candidate_id": item["candidate_id"],
        "model_seed": seed,
        "data_seed": campaign["data_seed"],
        "git_commit": campaign["git_commit"],
        "training_mlflow_run_id": item["mlflow_run_id"],
        "checkpoint_sha256": item["checkpoint_sha256"],
        "valid_pair_table_sha256": pairing["primary_validation_table_sha256"],
        "test_pair_table_sha256": pairing["primary_test_table_sha256"],
        "manifest_index": int(index),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
    }
    tag_override = "{" + ",".join(f"{key}:{value}" for key, value in tags.items()) + "}"
    overrides.extend(
        [
            f"logger.mlflow.experiment_name={experiment_name}",
            f"logger.mlflow.run_name={eval_run_name}",
            f"logger.mlflow.tags={tag_override}",
        ]
    )
    overrides = _replace_pair_env(overrides, valid_table, test_table)
    command = [sys.executable, "src/train.py", *overrides]
    environment = os.environ.copy()
    environment["CCHAMBER_VALID_PAIR_TABLE"] = str(valid_table)
    environment["CCHAMBER_TEST_PAIR_TABLE"] = str(test_table)
    environment["LOG_DIR"] = str(root / "logs")
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)

    values_path = checkpoint.parent / "plots" / "test" / checkpoint.stem / "auprc" / "values.csv"
    efficiency_values_path = (
        checkpoint.parent / "plots" / "test" / checkpoint.stem / "eff" / "values.csv"
    )
    _validate_final_values(
        values_path,
        checkpoint_stem=checkpoint.stem,
        interventions=campaign["interventions"],
        metric="auprc",
    )
    _validate_final_values(
        efficiency_values_path,
        checkpoint_stem=checkpoint.stem,
        interventions=campaign["interventions"],
        metric="efficiency_operational",
    )
    client = MlflowClient(tracking_uri=str(campaign["tracking_uri"]))
    run = _find_mlflow_run(client, experiment_name, eval_run_name)
    result = {
        **item,
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "manifest_index": int(index),
        "evaluation_attempt_id": attempt,
        "evaluation_mlflow_experiment": experiment_name,
        "evaluation_mlflow_run_id": run.info.run_id,
        "evaluation_mlflow_status": run.info.status,
        "values_csv": str(values_path.resolve()),
        "values_csv_sha256": _sha256(values_path),
        "efficiency_values_csv": str(efficiency_values_path.resolve()),
        "efficiency_values_csv_sha256": _sha256(efficiency_values_path),
        "n_interventions": 58,
    }
    _atomic_json(output, result)
    client.log_artifact(run.info.run_id, str(values_path), artifact_path="numeric")
    client.log_artifact(
        run.info.run_id,
        str(efficiency_values_path),
        artifact_path="numeric/efficiency",
    )
    client.log_artifact(run.info.run_id, str(output), artifact_path="campaign")
    client.set_tag(run.info.run_id, "values_csv_sha256", _sha256(values_path))
    client.set_tag(
        run.info.run_id,
        "efficiency_values_csv_sha256",
        _sha256(efficiency_values_path),
    )
    client.set_tag(run.info.run_id, "evaluation_result_sha256", _sha256(output))
    print(output)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in original order."""
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _paired_superiority_p(differences: np.ndarray) -> float:
    """Exact one-sided paired sign-flip p-value for a positive mean."""
    differences = np.asarray(differences, dtype=float)
    observed = float(differences.mean())
    bit_grid = np.arange(1 << len(differences), dtype=np.uint64)[:, None]
    positions = np.arange(len(differences), dtype=np.uint64)[None, :]
    signs = 2.0 * ((bit_grid >> positions) & 1).astype(float) - 1.0
    permuted = (signs * differences[None, :]).mean(axis=1)
    return float(np.mean(permuted >= observed - 1e-15))


def _primary_inference(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute the prespecified paired CAP superiority/equivalence tests."""
    primary = seed_summary[seed_summary["metric"] == "auprc"]
    superiority: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    for model in MODELS:
        pivot = primary[primary["model"] == model].pivot(
            index="seed",
            columns="strategy",
            values="value",
        )
        for left in ("cap_metadata_nearest", "cap_encoder_nearest"):
            right = "cap_random"
            differences = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            superiority.append(
                {
                    "model": model,
                    "test_family": "superiority_vs_random",
                    "strategy_left": left,
                    "strategy_right": right,
                    "alternative": "left_greater_than_right",
                    "margin": 0.0,
                    "mean_difference": float(differences.mean()),
                    "p_value": _paired_superiority_p(differences),
                    "n_paired_seeds": len(differences),
                }
            )

        differences = (pivot["cap_metadata_nearest"] - pivot["cap_encoder_nearest"]).to_numpy(
            dtype=float
        )
        n = len(differences)
        mean = float(differences.mean())
        std = float(differences.std(ddof=1))
        standard_error = std / math.sqrt(n)
        margin = EQUIVALENCE_MARGIN_AUPRC
        if standard_error == 0.0:
            p_lower = 0.0 if mean > -margin else 1.0
            p_upper = 0.0 if mean < margin else 1.0
            ci_low = ci_high = mean
        else:
            df = n - 1
            p_lower = float(stats.t.sf((mean + margin) / standard_error, df))
            p_upper = float(stats.t.cdf((mean - margin) / standard_error, df))
            critical = float(stats.t.ppf(0.95, df))
            ci_low = mean - critical * standard_error
            ci_high = mean + critical * standard_error
        equivalence.append(
            {
                "model": model,
                "test_family": "metadata_encoder_equivalence",
                "strategy_left": "cap_metadata_nearest",
                "strategy_right": "cap_encoder_nearest",
                "alternative": "absolute_difference_below_margin",
                "margin": margin,
                "mean_difference": mean,
                "ci90_low": ci_low,
                "ci90_high": ci_high,
                "p_value": max(p_lower, p_upper),
                "n_paired_seeds": n,
            }
        )

    for family in (superiority, equivalence):
        adjusted = _holm_adjust([row["p_value"] for row in family])
        for row, p_adjusted in zip(family, adjusted):
            row["p_value_holm"] = p_adjusted
            row["reject_at_0.05"] = bool(p_adjusted < 0.05)
    return pd.DataFrame([*superiority, *equivalence])


def _write_cchamber_summaries(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write intervention-, family-, target-, and strength-aware paper summaries."""
    keys = ["model", "strategy", "seed", "metric"]
    family_seed = (
        frame.groupby([*keys, "semantic_family"], sort=True)["value"].mean().reset_index()
    )
    equal_family_seed = family_seed.groupby(keys, sort=True)["value"].mean().reset_index()
    intervention_seed = frame.groupby(keys, sort=True)["value"].mean().reset_index()

    paths: list[Path] = []
    for name, table in (
        ("family_seed_summary.csv", family_seed),
        ("equal_family_seed_summary.csv", equal_family_seed),
        ("intervention_weighted_seed_summary.csv", intervention_seed),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        paths.append(path)

    for name, columns in (
        ("family_summary.csv", ["semantic_family"]),
        ("target_summary.csv", ["intervention_target"]),
        ("strength_summary.csv", ["strength"]),
        ("family_strength_summary.csv", ["semantic_family", "strength"]),
    ):
        grouped = (
            frame.groupby(["model", "strategy", "metric", *columns], sort=True, dropna=False)[
                "value"
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        path = output_dir / name
        grouped.to_csv(path, index=False)
        paths.append(path)

    inference = _primary_inference(intervention_seed)
    inference_path = output_dir / "primary_cap_inference.csv"
    inference.to_csv(inference_path, index=False)
    paths.append(inference_path)
    return paths


def collect_final_results(root: Path) -> None:
    """Validate complete evaluation coverage and build paper tables and plots."""
    root = root.resolve()
    campaign = _campaign_manifest(root)
    _assert_campaign_revision(campaign)
    expected_runs = len(MODELS) * len(STRATEGIES) * len(campaign["reporting_seeds"])
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for index in range(expected_runs):
        path = root / "evaluation_results" / f"{index:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_item = _manifest_item(root, "retrain_manifest.json", index)
        result = _validate_marker(
            path,
            {
                "campaign_id": campaign["campaign_id"],
                "git_commit": campaign["git_commit"],
                "manifest_index": int(index),
                "model": expected_item["model"],
                "strategy": expected_item["strategy"],
                "seed": int(expected_item["seed"]),
                "candidate_id": expected_item["candidate_id"],
                "pool_sha256": expected_item["pool_sha256"],
            },
            {
                "checkpoint": "checkpoint_sha256",
                "values_csv": "values_csv_sha256",
                "efficiency_values_csv": "efficiency_values_csv_sha256",
            },
        )
        checkpoint = Path(result["checkpoint"])
        source_records = []
        for path_key, metric in (
            ("values_csv", "auprc"),
            ("efficiency_values_csv", "efficiency_operational"),
        ):
            values_path = Path(result[path_key])
            values = _validate_final_values(
                values_path,
                checkpoint_stem=checkpoint.stem,
                interventions=campaign["interventions"],
                metric=metric,
            )
            values["dataset"] = "cchamber"
            values["model"] = result["model"]
            values["strategy"] = result["strategy"]
            values["seed"] = int(result["seed"])
            values["pairing"] = (
                result["strategy"].removeprefix("cap_")
                if str(result["strategy"]).startswith("cap_")
                else "not_applicable"
            )
            catalog = {
                name: parse_intervention_name(name) for name in values["intervention"].astype(str)
            }
            values["intervention_family"] = values["intervention"].map(
                lambda name: catalog[str(name)]["family"]
            )
            values["semantic_family"] = values["intervention_family"].map(
                lambda family: SEMANTIC_FAMILY.get(str(family), str(family))
            )
            values["intervention_target"] = values["intervention"].map(
                lambda name: catalog[str(name)]["target"]
            )
            values["strength"] = values["intervention"].map(
                lambda name: catalog[str(name)]["strength"]
            )
            frames.append(values)
            source_records.append(
                {
                    "metric": metric,
                    "values_csv": str(values_path.resolve()),
                    "values_csv_sha256": _sha256(values_path),
                }
            )
        provenance.append(
            {
                "manifest_index": index,
                "evaluation_result": str(path.resolve()),
                "evaluation_result_sha256": _sha256(path),
                "numeric_sources": source_records,
                "mlflow_run_id": result["evaluation_mlflow_run_id"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    expected_rows = expected_runs * len(campaign["interventions"]) * 2
    if len(combined) != expected_rows:
        raise ValueError(f"Expected {expected_rows} final rows, found {len(combined)}.")
    group_sizes = combined.groupby(["model", "strategy", "seed", "metric"]).size()
    if (
        not (group_sizes == len(campaign["interventions"])).all()
        or len(group_sizes) != expected_runs * 2
    ):
        raise ValueError("Final result coverage is not exact for both final metrics.")
    output_dir = root / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    combined.to_csv(results_path, index=False)
    summaries = _write_cchamber_summaries(combined, output_dir)
    aggregate_paths = aggregate_results(
        results_path,
        output_dir / "aggregate",
        main_metric="auprc",
    )
    _atomic_json(
        output_dir / "results_provenance.json",
        {
            "campaign": str((root / "campaign.json").resolve()),
            "campaign_sha256": _sha256(root / "campaign.json"),
            "results": str(results_path.resolve()),
            "results_sha256": _sha256(results_path),
            "expected_runs": expected_runs,
            "expected_rows": expected_rows,
            "summary_artifacts": {
                str(path.relative_to(output_dir)): _sha256(path)
                for path in [*summaries, *aggregate_paths]
            },
            "sources": provenance,
        },
    )
    print(results_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    design_parser = subparsers.add_parser("design")
    design_parser.add_argument("--root", type=Path, required=True)
    design_parser.add_argument("--campaign-id", required=True)
    design_parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)

    run_parser = subparsers.add_parser("run-candidate")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--model", choices=MODELS, required=True)
    run_parser.add_argument("--candidate-id", required=True)
    run_parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEV_SEEDS)),
        help="Comma-separated development model seeds.",
    )
    run_parser.add_argument("--pair-table", type=Path, required=True)

    collect_parser = subparsers.add_parser("collect-candidates")
    collect_parser.add_argument("--root", type=Path, required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--root", type=Path, required=True)

    pair_parser = subparsers.add_parser("run-pairing-encoder")
    pair_parser.add_argument("--root", type=Path, required=True)
    pair_parser.add_argument("--encoder-seed", type=int, required=True)

    pair_collect_parser = subparsers.add_parser("collect-pairing")
    pair_collect_parser.add_argument("--root", type=Path, required=True)

    retrain_parser = subparsers.add_parser("run-retrain")
    retrain_parser.add_argument("--root", type=Path, required=True)
    retrain_parser.add_argument("--index", type=int, required=True)

    evaluate_parser = subparsers.add_parser("run-evaluate")
    evaluate_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser.add_argument("--index", type=int, required=True)

    final_parser = subparsers.add_parser("collect-final")
    final_parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "design":
        design(args.root, args.campaign_id, args.n_candidates)
    elif args.command == "run-candidate":
        seeds = tuple(int(value) for value in args.seeds.split(",") if value)
        run_candidate(args.root, args.model, args.candidate_id, seeds, args.pair_table)
    elif args.command == "collect-candidates":
        collect_candidates(args.root)
    elif args.command == "select":
        select_candidates(args.root)
    elif args.command == "run-pairing-encoder":
        run_pairing_encoder(args.root, args.encoder_seed)
    elif args.command == "collect-pairing":
        collect_pairing(args.root)
    elif args.command == "run-retrain":
        run_retrain(args.root, args.index)
    elif args.command == "run-evaluate":
        run_evaluate(args.root, args.index)
    elif args.command == "collect-final":
        collect_final_results(args.root)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
