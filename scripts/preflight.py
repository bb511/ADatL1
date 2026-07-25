#!/usr/bin/env python3
"""Validate experiment composition and deployment inputs before consuming cloud compute."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import generation  # noqa: E402
from src.utils.omegaconf import register_resolvers  # noqa: E402
from src.utils.pairing.table import (  # noqa: E402
    atomic_json_dump,
    load_pair_table,
    sha256_file,
)

RUNTIME_PATH_VARS = (
    "PROJECT_ROOT",
    "DATA_DIR",
    "LOG_DIR",
    "OUTPUT_DIR",
    "CHECKPOINT_DIR",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("local", "cloud"), default="local")
    parser.add_argument(
        "--require-physics-data",
        action="store_true",
        help="Require every physics raw-data directory and at least one parquet file in each.",
    )
    parser.add_argument(
        "--valid-pair-table",
        type=Path,
        default=os.environ.get("CCHAMBER_VALID_PAIR_TABLE"),
    )
    parser.add_argument(
        "--test-pair-table",
        type=Path,
        default=os.environ.get("CCHAMBER_TEST_PAIR_TABLE"),
    )
    parser.add_argument(
        "--launcher",
        choices=tuple(item.value for item in generation.Launcher),
        default=generation.Launcher.NONE.value,
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Fail if tracked or untracked repository changes exist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "preflight.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(REPOSITORY_ROOT / ".env")
    args = parse_args(argv)
    checks: list[dict[str, Any]] = []

    _check(
        checks,
        "python_version",
        sys.version_info[:2] == (3, 10),
        f"running {sys.version.split()[0]}, required 3.10.x",
    )
    _check_uv_lock(checks)
    _check(
        checks,
        "project_marker",
        (REPOSITORY_ROOT / ".project-root").is_file(),
        ".project-root exists",
    )
    _check_runtime_paths(checks, profile=args.profile)
    _check_git(checks, required=args.require_clean_git or args.profile == "cloud")

    specs = generation.build_paper_experiments(
        n_trials=1,
        seeds=(123,),
        include_cvar10=False,
    )
    launcher = generation.Launcher(args.launcher)
    compose_errors = compose_experiment_matrix(specs.values(), launcher=launcher)
    _check(
        checks,
        "experiment_composition",
        not compose_errors,
        f"{len(specs)} specifications checked",
        errors=compose_errors,
    )

    shell_errors = validate_generated_shells(
        specs.values(),
        launcher=launcher,
    )
    _check(
        checks,
        "generated_shell_syntax",
        not shell_errors,
        f"{len(specs)} generated sweep scripts checked with bash -n",
        errors=shell_errors,
    )

    if args.valid_pair_table or args.test_pair_table or args.profile == "cloud":
        if not (args.valid_pair_table and args.test_pair_table):
            _check(
                checks,
                "pair_tables",
                False,
                "both --valid-pair-table and --test-pair-table are required",
            )
        else:
            _check_pair_tables(checks, args.valid_pair_table, args.test_pair_table)

    if args.require_physics_data or args.profile == "cloud":
        physics_errors = validate_physics_data()
        _check(
            checks,
            "physics_raw_data",
            not physics_errors,
            "all configured raw-data directories contain parquet files",
            errors=physics_errors,
        )

    passed = all(check["status"] in {"passed", "warning"} for check in checks)
    report = {
        "status": "passed" if passed else "failed",
        "profile": args.profile,
        "repository_root": str(REPOSITORY_ROOT),
        "checks": checks,
    }
    output = atomic_json_dump(report, args.output, overwrite=True)
    for check in checks:
        print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        for error in check.get("errors", []):
            print(f"  - {error}")
    print(f"Preflight report: {output}")
    return 0 if passed else 1


def compose_experiment_matrix(
    specs,
    *,
    launcher: generation.Launcher = generation.Launcher.NONE,
) -> list[str]:
    """Compose every paper sweep with the exact generated Hydra overrides."""
    errors = []
    GlobalHydra.instance().clear()
    register_resolvers()
    with initialize_config_dir(
        config_dir=str(REPOSITORY_ROOT / "configs"),
        version_base="1.3",
    ):
        for spec in specs:
            overrides = generation.build_sweep_overrides(
                spec,
                seed=123,
                launcher=launcher,
                trainer="gpu" if launcher != generation.Launcher.NONE else "cpu",
                devices="[0]" if launcher != generation.Launcher.NONE else "1",
                cpus_per_task=2,
                gpus_per_node=1,
                timeout_min=60,
            )
            overrides = [
                item.replace('"${RAW_DATA_DIR}"', "/tmp/raw_l1")
                .replace("$CCHAMBER_VALID_PAIR_TABLE", "/tmp/valid_pairs.pt")
                .replace("$CCHAMBER_TEST_PAIR_TABLE", "/tmp/test_pairs.pt")
                for item in overrides
            ]
            try:
                compose(config_name="train", overrides=overrides)
            except Exception as exc:
                errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")
    GlobalHydra.instance().clear()
    return errors


def validate_generated_shells(specs, *, launcher: generation.Launcher) -> list[str]:
    """Generate portable sweep scripts in a temporary directory and parse with bash."""
    errors = []
    with tempfile.TemporaryDirectory(prefix="adatl1-preflight-") as temporary:
        root = Path(temporary)
        for spec in specs:
            try:
                (root / spec.name).mkdir(parents=True, exist_ok=True)
                commands = generation.sweep_commands_for(
                    spec,
                    launcher=launcher,
                    trainer="gpu" if launcher != generation.Launcher.NONE else "cpu",
                    devices="[0]" if launcher != generation.Launcher.NONE else "1",
                    cpus_per_task=2,
                    gpus_per_node=1,
                    timeout_min=60,
                )
                path = generation.write_script(root / spec.name / "sweep.sh", commands, spec)
                result = subprocess.run(
                    ["bash", "-n", str(path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    errors.append(f"{spec.name}: {result.stderr.strip()}")
            except Exception as exc:
                errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")
    return errors


def validate_physics_data() -> list[str]:
    """Check every resolved physics source directory without loading the dataset."""
    raw_data_dir = os.environ.get("RAW_DATA_DIR")
    if not raw_data_dir:
        return ["RAW_DATA_DIR is not set."]

    GlobalHydra.instance().clear()
    register_resolvers()
    with initialize_config_dir(
        config_dir=str(REPOSITORY_ROOT / "configs"),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=physics/jetclr_pairing",
                f"paths.raw_data_dir={Path(raw_data_dir).expanduser().resolve()}",
            ],
        )
    GlobalHydra.instance().clear()

    errors = []
    for group in ("zerobias", "background", "signal"):
        for name, value in cfg.data[group].items():
            path = Path(str(value))
            if not path.is_dir():
                errors.append(f"{group}.{name}: missing directory {path}")
            elif not any(path.rglob("*.parquet")):
                errors.append(f"{group}.{name}: no parquet files under {path}")
    return errors


def _check_runtime_paths(checks: list[dict[str, Any]], *, profile: str) -> None:
    errors = []
    for name in RUNTIME_PATH_VARS:
        value = os.environ.get(name)
        if not value:
            errors.append(f"{name} is not set")
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            errors.append(f"{name} is not a directory: {path}")
        elif name != "PROJECT_ROOT" and not os.access(path, os.W_OK):
            errors.append(f"{name} is not writable: {path}")
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root and Path(project_root).expanduser().resolve() != REPOSITORY_ROOT:
        errors.append(
            f"PROJECT_ROOT resolves to {Path(project_root).expanduser().resolve()}, "
            f"expected {REPOSITORY_ROOT}"
        )
    if profile == "cloud" and os.environ.get("WANDB_MODE") not in {"offline", "online"}:
        errors.append("WANDB_MODE must be explicitly set to offline or online")
    _check(
        checks,
        "runtime_paths",
        not errors,
        "runtime paths exist and artifact directories are writable",
        errors=errors,
    )


def _check_uv_lock(checks: list[dict[str, Any]]) -> None:
    if not (REPOSITORY_ROOT / "uv.lock").is_file():
        _check(checks, "uv_lock", False, "uv.lock is missing")
        return
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    detail = (result.stdout or result.stderr).strip() or "uv.lock matches pyproject.toml"
    _check(
        checks,
        "uv_lock",
        result.returncode == 0,
        detail,
        errors=[result.stderr.strip()] if result.returncode and result.stderr.strip() else None,
    )


def _check_git(checks: list[dict[str, Any]], *, required: bool) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = bool(result.stdout.strip()) or result.returncode != 0
    status = "failed" if dirty and required else "warning" if dirty else "passed"
    checks.append(
        {
            "name": "git_state",
            "status": status,
            "detail": "repository has uncommitted changes" if dirty else "repository is clean",
        }
    )


def _check_pair_tables(
    checks: list[dict[str, Any]],
    valid_path: Path,
    test_path: Path,
) -> None:
    try:
        valid = load_pair_table(
            valid_path,
            expected_dataset_1="normal",
            expected_dataset_2="reference_normal",
            expected_split="validate",
        )
        test = load_pair_table(
            test_path,
            expected_dataset_1="normal",
            expected_dataset_2="reference_normal",
            expected_split="test",
        )
        errors = []
        for key in ("dataset_1", "dataset_2"):
            if valid[key] != test[key]:
                errors.append(f"{key} differs between validation and test tables")
        valid_sha = valid["metadata"]["encoder_checkpoint_sha256"]
        test_sha = test["metadata"]["encoder_checkpoint_sha256"]
        if valid_sha != test_sha:
            errors.append("validation and test tables use different encoder checkpoints")
        if valid["metadata"].get("data_seed") != test["metadata"].get("data_seed"):
            errors.append("validation and test tables use different data seeds")
        if valid["encoder_ckpt"] != test["encoder_ckpt"]:
            errors.append("validation and test tables reference different encoder paths")
        encoder_path = Path(valid["encoder_ckpt"]).expanduser()
        if not encoder_path.is_file():
            errors.append(f"pairing encoder checkpoint is unavailable: {encoder_path}")
        elif sha256_file(encoder_path) != valid_sha:
            errors.append("pairing encoder checkpoint fingerprint does not match the tables")
        detail = (
            f"valid={sha256_file(valid_path)}, test={sha256_file(test_path)}, "
            f"encoder={valid_sha}"
        )
        _check(checks, "pair_tables", not errors, detail, errors=errors)
    except Exception as exc:
        _check(
            checks,
            "pair_tables",
            False,
            "pair-table validation failed",
            errors=[f"{type(exc).__name__}: {exc}"],
        )


def _check(
    checks: list[dict[str, Any]],
    name: str,
    condition: bool,
    detail: str,
    *,
    errors: list[str] | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "passed" if condition else "failed",
            "detail": detail,
            **({"errors": errors} if errors else {}),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
