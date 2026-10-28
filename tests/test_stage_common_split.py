"""scripts/physics/_stage_common.sh: DATA_SPLIT_SEED picks the data split of every stage."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

SPLIT_ARGS = (
    "data.data_mlready.seed=666",
    "data.data_mlready.split_suffix=_split666",
    "data.data_mlready.require_split_info=true",
)


def _source(tmp_path: Path, *, trained_split: int | None = None, **env):
    project = tmp_path / "project"
    for sub in ("extracted", "processed", "mlready"):
        (project / "data" / "data_2025E+G" / sub).mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    settings = {"RUN_NAME": "Run01", "EXPERIMENT_NAME": "Study", **env}
    if trained_split is not None:
        run_dir = out / "checkpoints" / settings["EXPERIMENT_NAME"] / settings["RUN_NAME"]
        run_dir.mkdir(parents=True)
        config = {"data": {"data_mlready": {"seed": trained_split}}} if trained_split != 42 else {}
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    clean = {k: v for k, v in os.environ.items() if k not in {"DATA_SPLIT_SEED", "EXPERIMENT_NAME"}}
    clean.update(PROJECT_ROOT=str(project), ADL1T_OUTPUT_ROOT=str(out), CODE_DIR=str(REPO_ROOT),
                 PARETO_CANDIDATE="0", **settings)
    script = 'source scripts/physics/_stage_common.sh && printf "%s\\n" "${COMMON_ARGS[@]}"'
    return subprocess.run(["bash", "-c", script], cwd=REPO_ROOT, env=clean,
                          capture_output=True, text=True)


def _args(result) -> list[str]:
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


@pytest.mark.parametrize("seed", [None, "", "42"])
def test_original_split_adds_nothing(tmp_path, seed):
    env = {} if seed is None else {"DATA_SPLIT_SEED": seed}
    args = _args(_source(tmp_path, **env))
    assert not any(a.startswith("data.data_mlready.") for a in args)


def test_new_split_reads_its_own_prepared_cache(tmp_path):
    args = _args(_source(tmp_path, DATA_SPLIT_SEED="666", EXPERIMENT_NAME="Pareto-Front-261015-split666"))
    assert all(a in args for a in SPLIT_ARGS)
    assert "experiment_name=Pareto-Front-261015-split666" in args


@pytest.mark.parametrize("env", [
    {"DATA_SPLIT_SEED": "666", "EXPERIMENT_NAME": "Pareto-Front-261015"},        # split not named
    {"DATA_SPLIT_SEED": "666", "EXPERIMENT_NAME": "Pareto-Front-261015-split1"}, # another split named
    {"EXPERIMENT_NAME": "Pareto-Front-261015-split666"},                         # seed forgotten
    {"DATA_SPLIT_SEED": "666", "EXPERIMENT_NAME": ""},                           # no study folder
])
def test_study_folder_must_name_the_split(tmp_path, env):
    result = _source(tmp_path, **env)
    assert result.returncode == 2
    assert "does not match DATA_SPLIT_SEED" in result.stderr


def test_existing_run_must_have_been_trained_on_the_split(tmp_path):
    result = _source(tmp_path, trained_split=42, DATA_SPLIT_SEED="666",
                     EXPERIMENT_NAME="Study-split666")
    assert result.returncode == 2
    assert "trained on data split 42" in result.stderr

    ok = _source(tmp_path / "b", trained_split=666, DATA_SPLIT_SEED="666",
                 EXPERIMENT_NAME="Study-split666")
    assert all(a in _args(ok) for a in SPLIT_ARGS)


def test_seed_must_be_an_integer(tmp_path):
    result = _source(tmp_path, DATA_SPLIT_SEED="abc")
    assert result.returncode == 2 and "non-negative integer" in result.stderr
