"""scripts/physics/runae_test.sh: rebuilds the stage-1 overrides from the run's own record."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "physics" / "runae_test.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PARETO_CONFIG = {
    "seed": 180524,
    "tags": ["physics", "pareto", "fet-et", "validation", "protocol-v2"],
    "algorithm": {"mi_gamma": 1, "mi_sensitive_num_bins": 50, "encoder": {"nodes": [64, 32, 8]}},
    "pareto_study": {"candidate": {"autoencoder_seed": 180524, "mi_gamma": 1,
                                   "mi_sensitive_num_bins": 50, "architecture_id": "h64_32",
                                   "encoder_nodes": [64, 32, 8]}},
}
ADHOC_CONFIG = {
    "seed": 7,
    "tags": ["ae", "signal"],
    "algorithm": {
        "mi_gamma": 0.3, "mi_temperature": 6.0, "mi_sensitive_num_bins": 20, "delta": 10.0,
        "input_noise_std": 0.0, "encoder": {"nodes": [128, 64, 8]},
        "optimizer": {"lr": 0.0019859329798336714, "weight_decay": 1e-06, "betas": [0.9, 0.999]},
    },
}


def _run(tmp_path: Path, config: dict, *, experiment="Exp", run="Run01", files=None, **env):
    out = tmp_path / "out"
    run_dir = out / "checkpoints" / experiment / run
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in files if files is not None else ("loss_total.ckpt", "run_manifest.yaml",
                                                  "resolved_config.yaml"):
        content = {"run_manifest.yaml": yaml.safe_dump({"experiment_name": experiment, "run_name": run}),
                   "resolved_config.yaml": yaml.safe_dump(config)}.get(name, "ckpt")
        (run_dir / name).write_text(content)
    project = tmp_path / "project"
    for sub in ("extracted", "processed", "mlready"):
        (project / "data" / "data_2025E+G" / sub).mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in os.environ.items()
             if k not in {"SEED", "MI_GAMMA", "MI_NUM_BINS", "ENCODER_NODES", "LR", "EXPERIMENT"}}
    clean.update(DRY_RUN="1", EXPERIMENT_NAME=experiment, RUN_NAME=run,
                 PROJECT_ROOT=str(project), ADL1T_OUTPUT_ROOT=str(out), **env)
    return subprocess.run(["bash", str(SCRIPT)], env=clean, capture_output=True, text=True)


def _overrides(result) -> list[str]:
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.strip().splitlines()[-1])
    assert command[:2] == ["python3", "src/run_eval_metrics.py"]
    return command[2:]


def test_pareto_run_uses_the_study_experiment_and_its_candidate(tmp_path):
    args = _overrides(_run(tmp_path, PARETO_CONFIG))
    assert "experiment=physics/pareto_fet" in args
    assert "experiment_name=Exp" in args and "run_name=Run01" in args
    # 1 stays 1: the fingerprint would not match 1.0.
    assert "pareto_study.candidate.mi_gamma=1" in args
    assert "pareto_study.candidate.mi_sensitive_num_bins=50" in args
    assert "pareto_study.candidate.encoder_nodes=[64,32,8]" in args
    assert "pareto_study.candidate.autoencoder_seed=180524" in args
    assert args[-1] == "eval_split=test"
    assert not any(a.startswith("algorithm.") for a in args)


def test_adhoc_run_uses_physics_ae_and_its_algorithm_values(tmp_path):
    args = _overrides(_run(tmp_path, ADHOC_CONFIG))
    assert "experiment=physics/ae" in args
    for expected in ("seed=7", "algorithm.mi_gamma=0.3", "algorithm.mi_sensitive_num_bins=20",
                     "algorithm.encoder.nodes=[128,64,8]", "algorithm.optimizer.lr=0.0019859329798336714",
                     "algorithm.optimizer.weight_decay=1e-06", "algorithm.optimizer.betas=[0.9,0.999]",
                     "algorithm.mi_temperature=6.0", "algorithm.delta=10.0"):
        assert expected in args
    assert not any(a.startswith("pareto_study.") for a in args)


def test_test_experiment_overrides_the_choice(tmp_path):
    args = _overrides(_run(tmp_path, PARETO_CONFIG, TEST_EXPERIMENT="physics/other"))
    assert "experiment=physics/other" in args


@pytest.mark.parametrize("missing", ["loss_total.ckpt", "run_manifest.yaml", "resolved_config.yaml"])
def test_runs_without_checkpoint_or_record_are_refused(tmp_path, missing):
    files = [f for f in ("loss_total.ckpt", "run_manifest.yaml", "resolved_config.yaml") if f != missing]
    result = _run(tmp_path, PARETO_CONFIG, files=files)
    assert result.returncode == 2
    assert f"missing" in result.stderr and missing in result.stderr
