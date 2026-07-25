#!/usr/bin/env python3
"""Generate, but never submit, the frozen physical-shift Slurm analysis."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cchamber_physical_shift as physical_shift

ACCOUNT = "a0166"
PARTITION = "normal"
WALLTIME = "04:00:00"
CPUS_PER_TASK = 72
MEMORY = "120G"
FREEZE_NAME = "postselection_analysis_freeze_manifest_v1.json"
PLAN_NAME = "physical_shift_estimand_v1.json"
CATALOG_NAME = "physical_intervention_catalog_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is within parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_scratch_path(path: Path, scratch_root: Path, label: str) -> Path:
    """Require a resolved, non-root path on the explicitly authorized scratch tree."""
    path = path.expanduser().resolve()
    scratch_root = scratch_root.expanduser().resolve()
    if path == scratch_root or not _is_relative_to(path, scratch_root):
        raise ValueError(f"{label} must be below the explicit scratch root {scratch_root}.")
    return path


def _manifest_file(
    freeze_path: Path,
    files: Mapping[str, Any],
    key: str,
    expected_path: Path,
) -> tuple[Path, str]:
    """Resolve and authenticate one exact freeze-manifest entry."""
    record = files.get(key)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"Freeze manifest entry {key!r} has an invalid schema.")
    path = (freeze_path.parent / str(record["path"])).resolve()
    expected_path = expected_path.resolve()
    if path != expected_path:
        raise ValueError(f"Freeze manifest entry {key!r} does not name {expected_path}.")
    expected_hash = str(record["sha256"])
    if len(expected_hash) != 64 or physical_shift._sha256(path) != expected_hash:
        raise ValueError(f"Freeze manifest entry {key!r} failed SHA-256 authentication.")
    return path, expected_hash


def _authenticate_freeze(
    campaign_root: Path,
    freeze_manifest_sha256: str,
) -> tuple[dict[str, Any], Path, str, Path, str, Path, str]:
    """Authenticate the frozen campaign, estimand, and physical catalog."""
    campaign_root = campaign_root.expanduser().resolve()
    freeze_path = campaign_root / "design" / FREEZE_NAME
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    if physical_shift._sha256(freeze_path) != str(freeze_manifest_sha256):
        raise ValueError("Post-selection analysis freeze-manifest SHA-256 mismatch.")
    freeze = _read_json(freeze_path)
    if (
        int(freeze.get("schema_version", -1)) != 1
        or freeze.get("intervention_outcomes_inspected_before_freeze") is not False
    ):
        raise ValueError("Post-selection analysis freeze declaration is invalid.")
    files = freeze.get("files")
    if not isinstance(files, dict):
        raise TypeError("Post-selection analysis freeze manifest has no files object.")
    campaign_path, campaign_hash = _manifest_file(
        freeze_path,
        files,
        "campaign",
        campaign_root / "campaign.json",
    )
    plan_path, plan_hash = _manifest_file(
        freeze_path,
        files,
        "physical_shift_estimand",
        campaign_root / "design" / PLAN_NAME,
    )
    catalog_path, catalog_hash = _manifest_file(
        freeze_path,
        files,
        "physical_intervention_catalog",
        campaign_root / "design" / CATALOG_NAME,
    )
    campaign = _read_json(campaign_path)
    if freeze.get("campaign_id") != campaign.get("campaign_id") or freeze.get(
        "campaign_git_commit"
    ) != campaign.get("git_commit"):
        raise ValueError("Freeze-manifest campaign identity or commit changed.")
    # This validates the semantic contracts and their cross-hashes, but deliberately
    # does not open any intervention CSV.
    physical_shift._validate_frozen_design(campaign_root, plan_path, catalog_path)
    return (
        campaign,
        campaign_path,
        campaign_hash,
        plan_path,
        plan_hash,
        catalog_path,
        catalog_hash,
    )


def _require_clean_repository(
    repository: Path,
    expected_commit: str,
    label: str,
) -> Path:
    """Require one clean repository at an explicitly pinned commit."""
    repository = repository.expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"{label} repository is missing: {repository}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)):
        raise ValueError(f"{label} commit must be a full lowercase Git SHA.")
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required to authenticate repository state.")
    commit = subprocess.check_output(  # nosec B603
        [git, "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(  # nosec B603
        [git, "-C", str(repository), "status", "--porcelain"],
        text=True,
    ).strip()
    if commit != str(expected_commit) or dirty:
        raise RuntimeError(f"{label} repository is not clean at its pinned commit.")
    return repository


def _require_campaign_deployment(campaign: Mapping[str, Any]) -> Path:
    """Require the campaign's clean, exact training deployment checkout."""
    repository = Path(str(campaign.get("repository", ""))).expanduser().resolve()
    return _require_clean_repository(
        repository,
        str(campaign.get("git_commit")),
        "Campaign deployment",
    )


def _require_analysis_deployment(repository: Path, expected_commit: str) -> tuple[Path, str]:
    """Require a clean post-analysis checkout and pin the heavy entry point."""
    repository = _require_clean_repository(
        repository,
        expected_commit,
        "Physical-shift analysis deployment",
    )
    analysis = repository / "scripts" / "cchamber_physical_shift.py"
    if not analysis.is_file():
        raise FileNotFoundError(analysis)
    return repository, physical_shift._sha256(analysis)


def _script_text(
    *,
    campaign: Mapping[str, Any],
    campaign_path: Path,
    campaign_hash: str,
    freeze_path: Path,
    freeze_hash: str,
    plan_path: Path,
    plan_hash: str,
    catalog_path: Path,
    catalog_hash: str,
    selection_path: Path,
    selection_hash: str,
    output_dir: Path,
    repository: Path,
    analysis_commit: str,
    analysis_sha256: str,
    campaign_repository: Path,
    uv: Path,
    log_dir: Path,
) -> str:
    """Render one fully pinned CPU-only production job."""

    def q(value: object) -> str:
        return shlex.quote(str(value))

    return f"""#!/usr/bin/env bash
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --time={WALLTIME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem={MEMORY}
#SBATCH --job-name=cch-physical-shift
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err

# Generated by scripts/cchamber_physical_shift_slurm.py. Submission is manual.
set -euo pipefail
REPO={q(repository)}
CAMPAIGN_REPO={q(campaign_repository)}
CAMPAIGN={q(campaign_path)}
FREEZE={q(freeze_path)}
PLAN={q(plan_path)}
CATALOG={q(catalog_path)}
SELECTION={q(selection_path)}
OUTPUT={q(output_dir)}
UV={q(uv)}

test "$(git -C "$CAMPAIGN_REPO" rev-parse HEAD)" = {q(campaign["git_commit"])}
test -z "$(git -C "$CAMPAIGN_REPO" status --porcelain)"
test "$(git -C "$REPO" rev-parse HEAD)" = {q(analysis_commit)}
test -z "$(git -C "$REPO" status --porcelain)"
test "$(sha256sum "$REPO/scripts/cchamber_physical_shift.py" | awk '{{print $1}}')" = {q(analysis_sha256)}
test "$(sha256sum "$CAMPAIGN" | awk '{{print $1}}')" = {q(campaign_hash)}
test "$(sha256sum "$FREEZE" | awk '{{print $1}}')" = {q(freeze_hash)}
test "$(sha256sum "$PLAN" | awk '{{print $1}}')" = {q(plan_hash)}
test "$(sha256sum "$CATALOG" | awk '{{print $1}}')" = {q(catalog_hash)}
test "$(sha256sum "$SELECTION" | awk '{{print $1}}')" = {q(selection_hash)}

export PROJECT_ROOT="$REPO"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$REPO"
exec srun --nodes=1 --ntasks=1 --cpus-per-task={CPUS_PER_TASK} --mem={MEMORY} \
  "$UV" run --frozen --no-sync python scripts/cchamber_physical_shift.py \
  --campaign-root {q(campaign_path.parent)} \
  --shift-plan "$PLAN" \
  --target-catalog "$CATALOG" \
  --selection-provenance "$SELECTION" \
  --selection-provenance-sha256 {q(selection_hash)} \
  --output-dir "$OUTPUT"
"""


def generate_slurm(
    *,
    campaign_root: Path,
    freeze_manifest_sha256: str,
    selection_provenance_sha256: str,
    output_dir: Path,
    script_output: Path,
    slurm_log_dir: Path,
    scratch_root: Path,
    analysis_repository: Path,
    analysis_commit: str,
    uv: Path,
) -> Path:
    """Authenticate all light inputs and create an immutable Slurm script."""
    scratch_root = scratch_root.expanduser().resolve()
    campaign_root = _require_scratch_path(campaign_root, scratch_root, "Campaign root")
    output_dir = _require_scratch_path(output_dir, scratch_root, "Physical-shift output")
    script_output = _require_scratch_path(script_output, scratch_root, "Slurm script")
    slurm_log_dir = _require_scratch_path(slurm_log_dir, scratch_root, "Slurm log directory")
    if _is_relative_to(script_output, campaign_root) or _is_relative_to(
        slurm_log_dir, campaign_root
    ):
        raise ValueError("Generated scripts and logs must be outside the immutable campaign root.")
    if _is_relative_to(script_output, output_dir) or _is_relative_to(slurm_log_dir, output_dir):
        raise ValueError("Generated scripts and Slurm logs must be outside the analysis output.")
    output_dir = physical_shift._validate_output_path(output_dir, campaign_root)
    (
        campaign,
        campaign_path,
        campaign_hash,
        plan_path,
        plan_hash,
        catalog_path,
        catalog_hash,
    ) = _authenticate_freeze(campaign_root, freeze_manifest_sha256)
    campaign_repository = _require_campaign_deployment(campaign)
    repository, analysis_sha256 = _require_analysis_deployment(
        analysis_repository,
        analysis_commit,
    )
    selection_path = campaign_root / "selection" / "selection_provenance.json"
    physical_shift._validate_selection_frozen(
        campaign_root,
        campaign,
        selection_path,
        selection_provenance_sha256,
    )
    uv = uv.expanduser().resolve()
    if not uv.is_file() or not os.access(uv, os.X_OK):
        raise FileNotFoundError(f"uv executable is missing or not executable: {uv}")
    freeze_path = campaign_root / "design" / FREEZE_NAME
    text = _script_text(
        campaign=campaign,
        campaign_path=campaign_path,
        campaign_hash=campaign_hash,
        freeze_path=freeze_path,
        freeze_hash=freeze_manifest_sha256,
        plan_path=plan_path,
        plan_hash=plan_hash,
        catalog_path=catalog_path,
        catalog_hash=catalog_hash,
        selection_path=selection_path,
        selection_hash=selection_provenance_sha256,
        output_dir=output_dir,
        repository=repository,
        analysis_commit=analysis_commit,
        analysis_sha256=analysis_sha256,
        campaign_repository=campaign_repository,
        uv=uv,
        log_dir=slurm_log_dir,
    )
    script_output.parent.mkdir(parents=True, exist_ok=True)
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    if script_output.exists():
        if script_output.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace a different Slurm script: {script_output}")
        return script_output
    descriptor = os.open(script_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o750)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    return script_output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse generator arguments."""
    default_uv = shutil.which("uv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest-sha256", required=True)
    parser.add_argument("--selection-provenance-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--script-output", type=Path, required=True)
    parser.add_argument("--slurm-log-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--analysis-repository", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--uv", type=Path, default=Path(default_uv) if default_uv else None)
    args = parser.parse_args(argv)
    if args.uv is None:
        parser.error("uv was not found; pass --uv with an absolute executable path.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a production script without submitting it."""
    args = parse_args(argv)
    path = generate_slurm(
        campaign_root=args.campaign_root,
        freeze_manifest_sha256=args.freeze_manifest_sha256,
        selection_provenance_sha256=args.selection_provenance_sha256,
        output_dir=args.output_dir,
        script_output=args.script_output,
        slurm_log_dir=args.slurm_log_dir,
        scratch_root=args.scratch_root,
        analysis_repository=args.analysis_repository,
        analysis_commit=args.analysis_commit,
        uv=args.uv,
    )
    print(path)
    print(f"Review, then submit manually: sbatch {shlex.quote(str(path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
