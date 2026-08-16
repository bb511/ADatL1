#!/usr/bin/env python3
"""Manage authenticated, sequential physics pairing confirmation studies."""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404 - executes only reconstructed, authenticated argv
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.evaluation.physics_background_pairing_confirmation_manifest import (
    build_freeze_payload,
    freeze_confirmation_manifest,
    load_authenticated_confirmation_manifest,
)

MAX_TRIALS = 600


def _target_by_id(manifest: dict[str, Any], target_id: str) -> dict[str, Any]:
    """Resolve exactly one authenticated target identifier."""
    matches = [target for target in manifest["targets"] if target["target_id"] == target_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate confirmation target: {target_id}")
    return matches[0]


def _storage_url(target: dict[str, Any]) -> str:
    """Render the SQLite URL used by the frozen Hydra command."""
    return f"sqlite:///{target['database']}?timeout=600"


def _load_existing_study(target: dict[str, Any]) -> Any | None:
    """Load a study without creating a database when it has not started."""
    database = Path(target["database"])
    if not database.is_file():
        return None
    import optuna

    return optuna.load_study(study_name=target["study_name"], storage=_storage_url(target))


def target_status(target: dict[str, Any]) -> dict[str, Any]:
    """Report authoritative Optuna state counts for one target."""
    study = _load_existing_study(target)
    counts: dict[str, int] = {}
    if study is not None:
        for trial in study.get_trials(deepcopy=False):
            state = str(getattr(trial.state, "name", trial.state)).upper()
            counts[state] = counts.get(state, 0) + 1
    total = sum(counts.values())
    return {
        "target_id": target["target_id"],
        "study_name": target["study_name"],
        "database": target["database"],
        "COMPLETE": counts.get("COMPLETE", 0),
        "RUNNING": counts.get("RUNNING", 0),
        "FAIL": counts.get("FAIL", 0),
        "other_states": {
            state: count
            for state, count in sorted(counts.items())
            if state not in {"COMPLETE", "RUNNING", "FAIL"}
        },
        "total": total,
        "remaining_to_cap": max(0, MAX_TRIALS - total),
    }


def campaign_status(manifest: dict[str, Any]) -> dict[str, Any]:
    """Report state counts for every independently persisted target study."""
    rows = [target_status(target) for target in manifest["targets"]]
    return {
        "manifest_content_sha256": manifest["content_sha256"],
        "scope": manifest["scope"],
        "targets": rows,
        "totals": {
            state: sum(int(row[state]) for row in rows)
            for state in ("COMPLETE", "RUNNING", "FAIL", "total")
        },
    }


@contextmanager
def _target_chunk_lock(target: dict[str, Any]) -> Iterator[None]:
    """Prevent two local coordinators from racing before Optuna records a trial."""
    target_root = Path(target["database"]).parent
    target_root.mkdir(parents=True, exist_ok=True)
    lock = target_root / ".chunk.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Another chunk coordinator owns {lock}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
            handle.write("\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def _replace_override(argv: list[str], prefix: str, value: str) -> list[str]:
    """Replace one and only one authenticated Hydra override."""
    matches = [index for index, item in enumerate(argv) if item.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one frozen override with prefix {prefix!r}.")
    rendered = list(argv)
    rendered[matches[0]] = f"{prefix}{value}"
    return rendered


def run_sequential_chunk(
    manifest: dict[str, Any], *, target_id: str, max_new_trials: int
) -> dict[str, Any]:
    """Resume one study locally for a bounded, strictly sequential trial chunk."""
    if max_new_trials <= 0:
        raise ValueError("max_new_trials must be positive.")
    target = _target_by_id(manifest, target_id)
    with _target_chunk_lock(target):
        before = target_status(target)
        active = before["RUNNING"] + int(before["other_states"].get("WAITING", 0))
        if active:
            raise RuntimeError(
                f"Target {target_id} already has {active} active RUNNING/WAITING trial(s)."
            )
        trial_count = min(max_new_trials, before["remaining_to_cap"])
        if trial_count == 0:
            return {"launched": False, "new_trial_cap": 0, "before": before, "after": before}

        chunk_id = f"chunk_{before['total']:04d}_{uuid.uuid4().hex}"
        sweep_dir = Path(target["sweep_dir"]) / chunk_id
        if sweep_dir.exists():
            raise FileExistsError(f"Refusing to reuse sweep directory: {sweep_dir}")
        sweep_dir.parent.mkdir(parents=True, exist_ok=True)
        argv = list(target["command_argv"])
        argv = _replace_override(argv, "hydra.sweeper.n_trials=", str(trial_count))
        argv = _replace_override(argv, "hydra.sweeper.n_jobs=", "1")
        argv = _replace_override(argv, "hydra.sweep.dir=", str(sweep_dir))
        completed = subprocess.run(  # nosec B603 - argv is authenticated; shell is unused
            argv,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, argv)
        return {
            "launched": True,
            "new_trial_cap": trial_count,
            "sweep_dir": str(sweep_dir),
            "command_argv": argv,
            "before": before,
            "after": target_status(target),
        }


def main() -> None:
    """Run the confirmation manifest command-line interface."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "freeze"):
        command = subparsers.add_parser(name)
        command.add_argument("--campaign-root", type=Path, required=True)
        command.add_argument("--scope", choices=("native", "all"), required=True)
        if name == "freeze":
            command.add_argument("--expected-preview-sha256", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--campaign-root", type=Path, required=True)
    status.add_argument("--target-id")
    chunk = subparsers.add_parser("run-chunk")
    chunk.add_argument("--campaign-root", type=Path, required=True)
    chunk.add_argument("--target-id", required=True)
    chunk.add_argument("--max-new-trials", type=int, required=True)
    args = parser.parse_args()
    if args.command == "preview":
        print(
            json.dumps(
                build_freeze_payload(args.campaign_root, scope=args.scope),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "freeze":
        output = freeze_confirmation_manifest(
            args.campaign_root,
            scope=args.scope,
            expected_preview_sha256=args.expected_preview_sha256,
        )
        print(output)
        return
    manifest = load_authenticated_confirmation_manifest(args.campaign_root)
    if args.command == "status":
        result = (
            target_status(_target_by_id(manifest, args.target_id))
            if args.target_id
            else campaign_status(manifest)
        )
    else:
        result = run_sequential_chunk(
            manifest,
            target_id=args.target_id,
            max_new_trials=args.max_new_trials,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
