"""Deterministic Git provenance for immutable post-training artifacts."""

from __future__ import annotations

import hashlib
import shutil
import subprocess  # nosec B404 -- fixed Git inspection commands only
from pathlib import Path
from typing import Any


def _git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run one read-only Git query in an authenticated repository."""
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git")
    return subprocess.run(  # nosec B603 -- executable and arguments are fixed by callers
        [executable, *args],
        cwd=repository,
        check=check,
        capture_output=True,
        text=True,
    )


def sha256_file(path: Path) -> str:
    """Hash one generator source file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_generator(repository: Path, generator_path: Path) -> dict[str, Any]:
    """Require a clean matching checkout and return stable generator provenance."""
    repository = repository.expanduser().resolve(strict=True)
    generator_path = generator_path.expanduser().resolve(strict=True)
    top_level = Path(_git(repository, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top_level != repository:
        raise RuntimeError(
            f"Execution repository is not its Git top level: {repository} != {top_level}."
        )
    if not generator_path.is_relative_to(repository):
        raise RuntimeError(
            f"Generator source is outside the execution repository: {generator_path}."
        )
    status = _git(repository, "status", "--porcelain", "--untracked-files=all").stdout
    if status:
        raise RuntimeError("Post-training execution repository must be completely clean.")
    commit = _git(repository, "rev-parse", "HEAD").stdout.strip()
    tree = _git(repository, "rev-parse", "HEAD^{tree}").stdout.strip()
    relative_script = generator_path.relative_to(repository).as_posix()
    committed_script = _git(repository, "show", f"HEAD:{relative_script}").stdout.encode()
    if hashlib.sha256(committed_script).hexdigest() != sha256_file(generator_path):
        raise RuntimeError("Generator source differs from the committed execution checkout.")
    symbolic = _git(repository, "symbolic-ref", "-q", "--short", "HEAD", check=False)
    head_ref = symbolic.stdout.strip() or None
    return {
        "schema_version": 1,
        "repository": str(repository),
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "script": relative_script,
        "script_sha256": sha256_file(generator_path),
        "head_is_detached": head_ref is None,
        "head_ref": head_ref,
    }


def reauthenticate_generator(provenance: dict[str, Any], generator_path: Path) -> None:
    """Recompute and compare a previously frozen generator identity."""
    if (
        authenticate_generator(Path(str(provenance.get("repository", ""))), generator_path)
        != provenance
    ):
        raise RuntimeError("Post-training generator provenance differs from the frozen identity.")


__all__ = ["authenticate_generator", "reauthenticate_generator", "sha256_file"]
