#!/usr/bin/env python3
"""Extract authenticated efficiency/AUPRC rows from sealed confirmation runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import physics_background_pairing_campaign as campaign
from src.evaluation.physics_background_pairing_confirmation_matrix import (
    load_authenticated_confirmation_campaign,
)

ENDPOINT_FOLDERS = {
    "native": {"efficiency_250hz": "eff_native", "auprc": "auprc_native"},
    "residual_oas": {
        "efficiency_250hz": "eff_residual_oas",
        "auprc": "auprc_residual_oas",
    },
}


def _sha256(path: Path) -> str:
    """Hash one persisted artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    """Hash one JSON-compatible value with deterministic serialization."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Replace one complete CSV atomically after validating its schema."""
    if not rows:
        raise ValueError(f"Refusing to write an empty endpoint table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_bytes() != Path(temporary_name).read_bytes():
                raise FileExistsError(f"Refusing to replace different endpoint table: {path}")
            return path
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


def _values_for_checkpoint(
    marker: dict[str, Any], *, checkpoint: Path, score: str, endpoint: str
) -> dict[str, float]:
    """Read one exact callback/checkpoint output from marker-bound CSV files."""
    if score not in ENDPOINT_FOLDERS or endpoint not in ENDPOINT_FOLDERS[score]:
        raise ValueError(f"Unknown score/endpoint combination: {score}/{endpoint}")
    if endpoint == "efficiency_250hz":
        return campaign._read_downstream_efficiencies(marker, checkpoint, score)
    return campaign._read_downstream_auprcs(marker, checkpoint, score)


def extract_confirmation_endpoints(root: Path, output_dir: Path) -> dict[str, Any]:
    """Authenticate all 84 markers and emit primary plus OAS diagnostic rows."""
    root = root.expanduser().resolve(strict=True)
    frozen = load_authenticated_confirmation_campaign(root)
    matrix = frozen["matrix"]
    runner = frozen["runner_manifest"]
    cells = {int(row["cell_index"]): row for row in matrix["logical_cells"]}
    seed_count = len(matrix["training_seeds"])
    memberships = runner["front_memberships"]
    if len(memberships) != 42 * seed_count:
        raise RuntimeError("Confirmation cell/seed membership cardinality differs.")
    run_rows = {int(row["retrain_index"]): row for row in runner["retrain_runs"]}
    if len(run_rows) != 28 * seed_count:
        raise RuntimeError("Confirmation physical-run cardinality differs.")
    markers: dict[int, dict[str, Any]] = {}
    artifacts: dict[int, dict[str, Any]] = {}
    source_markers = []
    for index, row in run_rows.items():
        marker_path = root / "retraining/markers" / f"{index:04d}.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"Confirmation run {index} is not sealed: {marker_path}")
        campaign._validate_retrain_marker(marker_path, row)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        artifact_path = Path(marker["optimized_metric_artifact"])
        if _sha256(artifact_path) != marker["optimized_metric_artifact_sha256"]:
            raise RuntimeError(f"Optimized-metric artifact changed: {artifact_path}")
        markers[index] = marker
        artifacts[index] = json.loads(artifact_path.read_text(encoding="utf-8"))
        source_markers.append(
            {
                "retrain_index": index,
                "run_id": row["run_id"],
                "marker": str(marker_path.resolve()),
                "marker_sha256": _sha256(marker_path),
                "optimized_metric_artifact": str(artifact_path.resolve()),
                "optimized_metric_artifact_sha256": _sha256(artifact_path),
            }
        )

    primary: list[dict[str, Any]] = []
    oas: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for membership in memberships:
        index = int(membership["retrain_index"])
        cell = cells[int(membership["cell_index"])]
        marker = markers[index]
        selection = artifacts[index]["selections"][cell["logical_objective"]]
        checkpoint_id = str(selection["optimized_ckpt_name"])
        checkpoint_path = campaign._selected_checkpoint_path(
            Path(marker["checkpoint_root"]), checkpoint_id
        )
        checkpoint_sha256 = _sha256(checkpoint_path)
        selection_rows.append(
            {
                "retrain_index": index,
                "run_id": run_rows[index]["run_id"],
                "cell_index": int(cell["cell_index"]),
                "configuration_index": int(cell["configuration_index"]),
                "model": cell["model"],
                "trial_number": int(cell["trial_number"]),
                "seed": int(membership["seed"]),
                "logical_objective": cell["logical_objective"],
                "checkpoint_family_id": checkpoint_id,
                "checkpoint_identity": campaign._checkpoint_identity(checkpoint_path.stem),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "selection_metric": json.dumps(selection["optimized_metric"]),
            }
        )
        for endpoint in ("efficiency_250hz", "auprc"):
            for anomaly, value in sorted(
                _values_for_checkpoint(
                    marker, checkpoint=checkpoint_path, score="native", endpoint=endpoint
                ).items()
            ):
                primary.append(
                    {
                        "cell_index": int(cell["cell_index"]),
                        "configuration_index": int(cell["configuration_index"]),
                        "model": cell["model"],
                        "trial_number": int(cell["trial_number"]),
                        "seed": int(membership["seed"]),
                        "score": "native",
                        "selector": cell["selector"],
                        "pairing": cell["pairing"] or "not_applicable",
                        "logical_objective": cell["logical_objective"],
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "anomaly": anomaly,
                        "endpoint": endpoint,
                        "value": value,
                    }
                )
            if cell["model"] in {"ae", "vae"}:
                for anomaly, value in sorted(
                    _values_for_checkpoint(
                        marker,
                        checkpoint=checkpoint_path,
                        score="residual_oas",
                        endpoint=endpoint,
                    ).items()
                ):
                    oas.append(
                        {
                            **primary[-20],
                            "score": "residual_oas",
                            "anomaly": anomaly,
                            "endpoint": endpoint,
                            "value": value,
                        }
                    )
    expected_primary = 42 * seed_count * 20 * 2
    expected_oas = 2 * 7 * seed_count * 20 * 2
    if len(primary) != expected_primary or len(oas) != expected_oas:
        raise RuntimeError(
            f"Endpoint cardinality differs: primary={len(primary)}/{expected_primary}, "
            f"oas={len(oas)}/{expected_oas}."
        )
    output_dir = output_dir.expanduser().resolve()
    primary_path = _atomic_csv(output_dir / "endpoints_long.csv", primary)
    oas_path = _atomic_csv(output_dir / "oas_diagnostics.csv", oas)
    selections_path = _atomic_csv(output_dir / "checkpoint_selections.csv", selection_rows)
    receipt = {
        "schema_version": 1,
        "execution_commit": frozen["design"]["code_commit"],
        "campaign_root": str(root),
        "matrix_content_sha256": matrix["content_sha256"],
        "runner_manifest_sha256": _sha256(root / "selection/pareto_retrain_manifest.json"),
        "cardinality": {
            "sealed_runs": len(markers),
            "checkpoint_selections": len(selection_rows),
            "primary_endpoint_rows": len(primary),
            "oas_endpoint_rows": len(oas),
        },
        "artifacts": {
            "endpoints_long.csv": _sha256(primary_path),
            "oas_diagnostics.csv": _sha256(oas_path),
            "checkpoint_selections.csv": _sha256(selections_path),
        },
        "primary_rows_content_sha256": _canonical_hash(primary),
        "oas_rows_content_sha256": _canonical_hash(oas),
        "source_markers": source_markers,
        "source_markers_content_sha256": _canonical_hash(source_markers),
    }
    receipt_path = output_dir / "endpoint_extraction.json"
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if receipt_path.exists():
        if receipt_path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"Refusing to replace endpoint extraction receipt: {receipt_path}"
            )
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{receipt_path.name}.", dir=receipt_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, receipt_path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    return receipt


def main() -> None:
    """Parse the campaign and result paths and extract all endpoint rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            extract_confirmation_endpoints(args.root, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
