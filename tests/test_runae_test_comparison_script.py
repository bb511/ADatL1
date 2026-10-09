"""scripts/physics/runae_test_comparison.sh: where the run list comes from."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "physics" / "runae_test_comparison.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _dry_run(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    out = tmp_path / "out"
    (out / "checkpoints" / "Exp").mkdir(parents=True, exist_ok=True)
    full_env = {k: v for k, v in os.environ.items() if k not in {"RUNS", "FORCE"}}
    full_env.update(DRY_RUN="1", EXPERIMENT_NAME="Exp", PROJECT_ROOT=str(tmp_path / "project"),
                    ADL1T_OUTPUT_ROOT=str(out))
    full_env.update(env)
    return subprocess.run(["bash", str(SCRIPT), *args], env=full_env, capture_output=True, text=True)


def _commands(result) -> list[list[str]]:
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("python3 ")]
    assert len(lines) == 2
    return [shlex.split(line) for line in lines]


def _runs(command: list[str]) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command) if arg == "--run-name"]


def test_runs_from_the_command_line(tmp_path):
    correlation, metrics = _commands(_dry_run(tmp_path, "RunA", "RunB"))
    assert correlation[1:3] == ["-m", "src.evaluation.pareto.correlation_gamma0"]
    assert metrics[1:3] == ["-m", "src.evaluation.pareto.run_vs_gamma0"]
    for command in (correlation, metrics):
        assert _runs(command) == ["RunA", "RunB"]
        assert command[command.index("--split") + 1] == "test"
        assert command[command.index("--experiment-dir") + 1] == str(tmp_path / "out" / "checkpoints" / "Exp")
    assert "--force" not in correlation


def test_runs_from_a_file_skip_comments_and_blank_lines(tmp_path):
    runs = tmp_path / "runs.txt"
    runs.write_text("# selected on val\nRunA\n\n  RunB  # with a comment\n")
    correlation, metrics = _commands(_dry_run(tmp_path, RUNS=str(runs), FORCE="1"))
    assert _runs(correlation) == _runs(metrics) == ["RunA", "RunB"]
    assert "--force" in correlation


def test_default_list_is_batch_test_runs(tmp_path):
    expected = [line.strip() for line in (REPO_ROOT / "batch" / "test_runs.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    result = _dry_run(tmp_path)
    assert "batch/test_runs.txt" in result.stdout
    assert _runs(_commands(result)[1]) == expected


def test_empty_list_and_missing_experiment_are_refused(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing yet\n")
    result = _dry_run(tmp_path, RUNS=str(empty))
    assert result.returncode == 2 and "run list is empty" in result.stderr
    result = _dry_run(tmp_path, "RunA", EXPERIMENT_NAME="Nope")
    assert result.returncode == 2 and "no experiment folder" in result.stderr
