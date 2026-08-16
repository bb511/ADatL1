"""Tests for bounded local confirmation-study execution and status reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import physics_background_pairing_confirmation as runner


@dataclass
class _Trial:
    """Minimal fake Optuna trial with a named state."""

    state: Any


class _Study:
    """Minimal fake Optuna study exposing immutable trial records."""

    def __init__(self, states: list[str]) -> None:
        self.states = states

    def get_trials(self, *, deepcopy: bool) -> list[_Trial]:
        assert deepcopy is False
        return [_Trial(SimpleNamespace(name=state)) for state in self.states]


def _target(tmp_path: Path) -> dict[str, Any]:
    """Return one frozen-shape runner target rooted in a temporary directory."""
    root = tmp_path / "study"
    database = root / "optuna.db"
    sweep = root / "sweeps"
    return {
        "target_id": "00_ae_native_cap_cdf",
        "study_name": "confirmation_ae_native_cap_cdf",
        "database": str(database),
        "sweep_dir": str(sweep),
        "command_argv": [
            "uv",
            "run",
            "python",
            "src/train.py",
            "-m",
            "hydra.sweeper.n_trials=600",
            "hydra.sweeper.n_jobs=1",
            f"hydra.sweep.dir={sweep}",
        ],
    }


def test_status_reports_required_optuna_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Status preserves COMPLETE/RUNNING/FAIL and accounts for all other states."""
    target = _target(tmp_path)
    monkeypatch.setattr(
        runner,
        "_load_existing_study",
        lambda unused: _Study(["COMPLETE", "COMPLETE", "RUNNING", "FAIL", "PRUNED"]),
    )
    status = runner.target_status(target)
    assert (status["COMPLETE"], status["RUNNING"], status["FAIL"]) == (2, 1, 1)
    assert status["other_states"] == {"PRUNED": 1}
    assert status["total"] == 5
    assert status["remaining_to_cap"] == 595


def test_chunk_is_sequential_bounded_and_uses_unique_sweep_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk replaces only bounded runtime overrides and invokes no scheduler."""
    target = _target(tmp_path)
    states = ["COMPLETE"] * 598
    monkeypatch.setattr(runner, "_load_existing_study", lambda unused: _Study(states))
    monkeypatch.setattr(runner.uuid, "uuid4", lambda: SimpleNamespace(hex="unique"))
    calls: list[tuple[list[str], Path, bool]] = []

    def _run(argv: list[str], *, cwd: Path, check: bool) -> SimpleNamespace:
        calls.append((argv, cwd, check))
        states.extend(["COMPLETE", "COMPLETE"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", _run)
    result = runner.run_sequential_chunk(
        {"targets": [target]}, target_id=target["target_id"], max_new_trials=25
    )
    assert result["new_trial_cap"] == 2
    assert result["after"]["COMPLETE"] == 600
    argv, cwd, check = calls[0]
    assert "hydra.sweeper.n_trials=2" in argv
    assert "hydra.sweeper.n_jobs=1" in argv
    assert f"hydra.sweep.dir={target['sweep_dir']}/chunk_0598_unique" in argv
    assert cwd == Path(runner.__file__).resolve().parents[1]
    assert check is False


@pytest.mark.parametrize("active_state", ["RUNNING", "WAITING"])
def test_chunk_refuses_overlapping_active_trials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, active_state: str
) -> None:
    """An existing active trial must prevent a second coordinator invocation."""
    target = _target(tmp_path)
    monkeypatch.setattr(runner, "_load_existing_study", lambda unused: _Study([active_state]))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )
    with pytest.raises(RuntimeError, match="active RUNNING/WAITING"):
        runner.run_sequential_chunk(
            {"targets": [target]}, target_id=target["target_id"], max_new_trials=1
        )


def test_chunk_propagates_training_failure_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed local process is visible and does not leave a coordinator lock."""
    target = _target(tmp_path)
    monkeypatch.setattr(runner, "_load_existing_study", lambda unused: None)
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7)
    )
    with pytest.raises(runner.subprocess.CalledProcessError) as error:
        runner.run_sequential_chunk(
            {"targets": [target]}, target_id=target["target_id"], max_new_trials=3
        )
    assert error.value.returncode == 7
    assert not (Path(target["database"]).parent / ".chunk.lock").exists()
