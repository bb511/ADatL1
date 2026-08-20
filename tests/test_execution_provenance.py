"""Focused contracts for deterministic post-training generator provenance."""

from __future__ import annotations

import hashlib
import shutil
import subprocess  # nosec B404 -- fixed Git setup commands in isolated test repositories
from pathlib import Path

import pytest

from src.utils.execution_provenance import (
    authenticate_generator,
    reauthenticate_generator,
)


def _git(repository: Path, *args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git")
    return subprocess.run(  # nosec B603 -- fixed test-only Git commands
        [executable, *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "provenance@example.invalid")
    _git(repository, "config", "user.name", "Provenance Test")
    script = repository / "scripts/post_training.py"
    script.parent.mkdir()
    script.write_text("print('sealed')\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "freeze generator")
    return repository, script


def test_generator_provenance_binds_clean_commit_tree_and_script(tmp_path: Path) -> None:
    repository, script = _repository(tmp_path)
    provenance = authenticate_generator(repository, script)
    assert provenance == {
        "schema_version": 1,
        "repository": str(repository.resolve()),
        "commit": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "worktree_clean": True,
        "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "script": "scripts/post_training.py",
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "head_is_detached": False,
        "head_ref": _git(repository, "symbolic-ref", "-q", "--short", "HEAD"),
    }
    reauthenticate_generator(provenance, script)


def test_generator_provenance_rejects_dirty_or_mismatched_checkout(tmp_path: Path) -> None:
    repository, script = _repository(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside the execution repository"):
        authenticate_generator(repository, outside)

    script.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completely clean"):
        authenticate_generator(repository, script)


def test_reauthentication_rejects_tampered_generator_identity(tmp_path: Path) -> None:
    repository, script = _repository(tmp_path)
    provenance = authenticate_generator(repository, script)
    provenance["commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="provenance differs"):
        reauthenticate_generator(provenance, script)
