"""Frozen uniform random-bijection tables for the physics confirmation control."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from src.utils.pairing.physics_tables import load_split
from src.utils.pairing.table import (
    atomic_json_dump,
    atomic_torch_save,
    load_pair_table,
    sha256_file,
    sha256_tensor,
    validate_pair_table,
)
from src.utils.pairing.utils import PairingResult, pair_table_dict

DATASET_1 = "ZB_run396102"
DATASET_2 = "ZB_run398183"
EVENTS = 163_840
VALIDATION_SEEDS = (1_259_077_142, 1_712_736_754, 1_419_173_189, 2_013_784_264, 127_162_851)
TEST_SEEDS = (1_614_427_248, 957_139_045, 438_008_272, 1_352_904_386, 605_064_281)
SEEDS_BY_SPLIT = {"validate": VALIDATION_SEEDS, "test": TEST_SEEDS}
STRATEGY = "random_uniform_bijection"
SCHEMA_VERSION = 1


def _canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def seed_contract() -> dict[str, Any]:
    """Return the explicit preregistered seed contract and its semantic hash."""
    values = {
        "selection_rule": "fixed_before_evaluation",
        "validate": list(VALIDATION_SEEDS),
        "test": list(TEST_SEEDS),
    }
    return {**values, "sha256": _canonical_hash(values)}


def build_random_pair_table(
    source_1: torch.Tensor,
    source_2: torch.Tensor,
    *,
    split: str,
    seed: int,
    dataset_1: str = DATASET_1,
    dataset_2: str = DATASET_2,
) -> dict[str, Any]:
    """Build one deterministic, total, uniform random permutation table."""
    if split not in SEEDS_BY_SPLIT:
        raise ValueError(f"Random pairing split must be one of {tuple(SEEDS_BY_SPLIT)}.")
    if seed not in SEEDS_BY_SPLIT[split]:
        raise ValueError(f"Seed {seed} is not preregistered for split {split!r}.")
    source_1 = torch.flatten(source_1.detach().cpu(), start_dim=1).contiguous()
    source_2 = torch.flatten(source_2.detach().cpu(), start_dim=1).contiguous()
    if source_1.shape[0] != source_2.shape[0] or source_1.shape[0] <= 0:
        raise ValueError("Random bijections require nonempty equal-size source tensors.")

    n_events = int(source_1.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    idx_1 = torch.arange(n_events, dtype=torch.long)
    idx_2 = torch.randperm(n_events, generator=generator, dtype=torch.long)
    zeros_float = torch.zeros(n_events, dtype=torch.float32)
    zeros_long = torch.zeros(n_events, dtype=torch.long)
    pairs = PairingResult(
        idx_1=idx_1,
        idx_2=idx_2,
        distance=zeros_float,
        rank_1_to_2=zeros_long,
        rank_2_to_1=zeros_long.clone(),
    )
    generator_name = "torch.Generator(device=cpu)+torch.randperm"
    generator_version = str(torch.__version__)
    generator_contract_sha256 = _canonical_hash(
        {"generator": generator_name, "generator_version": generator_version, "seed": int(seed)}
    )
    metadata = {
        "producer": "src.utils.pairing.random_tables",
        "strategy": STRATEGY,
        "selection_rule": "fixed_before_evaluation",
        "random_seed": int(seed),
        "generator": generator_name,
        "generator_version": generator_version,
        "generator_contract_sha256": generator_contract_sha256,
        "permutation_sha256": sha256_tensor(idx_2),
        "n_dataset_1": n_events,
        "n_dataset_2": n_events,
        "n_pairs": n_events,
        "coverage": 1.0,
        "encoder_checkpoint_sha256": generator_contract_sha256,
        "source_1_sha256": sha256_tensor(source_1),
        "source_2_sha256": sha256_tensor(source_2),
        "pairing_orientation": "dataset_1_background0_to_dataset_2_background1",
        "distance_semantics": "not_applicable_uniform_random_control",
        "rank_semantics": "not_applicable_uniform_random_control",
    }
    table = pair_table_dict(
        pairs,
        dataset_1=dataset_1,
        dataset_2=dataset_2,
        split=split,
        encoder_ckpt=f"random://torch/cpu/seed/{seed}",
        metadata=metadata,
    )
    table["map_0_to_1"] = idx_2.clone()
    validate_pair_table(table)
    return table


def _authenticate_campaign(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the campaign design and its frozen structured-pairing preflight."""
    design_path = root / "design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    preflight_path = Path(design["preflight"]).resolve(strict=True)
    if sha256_file(preflight_path) != design["preflight_sha256"]:
        raise RuntimeError("Frozen pairing preflight differs from the campaign design.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_1": DATASET_1,
        "dataset_2": DATASET_2,
        "events_per_source": EVENTS,
    }
    for key, value in expected.items():
        if preflight.get(key) != value:
            raise RuntimeError(f"Frozen preflight {key} differs from the random-control contract.")
    if not preflight.get("tables") or not all(
        row.get("authenticated") for row in preflight["tables"]
    ):
        raise RuntimeError("Frozen structured pairing tables are not fully authenticated.")
    for row in preflight["tables"]:
        path = Path(row["path"]).resolve(strict=True)
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Frozen structured pair table changed: {path}")
    return design, preflight


def freeze_random_pairing_tables(root: Path) -> dict[str, Any]:
    """Atomically freeze ten tables without touching existing structured artifacts."""
    root = root.expanduser().resolve(strict=True)
    design, preflight = _authenticate_campaign(root)
    destination = root / "random_pairing"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace frozen random pairing artifacts: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".random_pairing.", dir=destination.parent))
    try:
        table_dir = temporary / "tables"
        table_dir.mkdir(parents=True)
        cache_root = Path(preflight["cache_root"]).resolve(strict=True)
        source_metadata_dir = Path(preflight["artifact_dir"]).resolve(strict=True) / "sources"
        table_rows = []
        for split, seeds in SEEDS_BY_SPLIT.items():
            source_1, _, source_1_folder = load_split(
                cache_root,
                split,
                dataset=DATASET_1,
                limit=EVENTS,
                source_metadata_dir=source_metadata_dir,
            )
            source_2, _, source_2_folder = load_split(
                cache_root,
                split,
                dataset=DATASET_2,
                limit=EVENTS,
                source_metadata_dir=source_metadata_dir,
            )
            if source_1.shape[0] != EVENTS or source_2.shape[0] != EVENTS:
                raise RuntimeError(
                    f"Split {split!r} does not contain the exact {EVENTS} prefixes."
                )
            for replicate, seed in enumerate(seeds):
                table = build_random_pair_table(source_1, source_2, split=split, seed=seed)
                filename = f"{split}_random_seed{seed}_cap_n{EVENTS}.pt"
                temporary_path = table_dir / filename
                atomic_torch_save(table, temporary_path)
                load_pair_table(
                    temporary_path,
                    expected_dataset_1=DATASET_1,
                    expected_dataset_2=DATASET_2,
                    expected_split=split,
                    n_dataset_1=EVENTS,
                    n_dataset_2=EVENTS,
                    source_1_sha256=table["metadata"]["source_1_sha256"],
                    source_2_sha256=table["metadata"]["source_2_sha256"],
                )
                table_rows.append(
                    {
                        "split": split,
                        "replicate": replicate,
                        "seed": seed,
                        "path": str((destination / "tables" / filename).resolve()),
                        "file_sha256": sha256_file(temporary_path),
                        "permutation_sha256": table["metadata"]["permutation_sha256"],
                        "source_1_sha256": table["metadata"]["source_1_sha256"],
                        "source_2_sha256": table["metadata"]["source_2_sha256"],
                        "source_1_folder": str(source_1_folder.resolve()),
                        "source_2_folder": str(source_2_folder.resolve()),
                        "events": EVENTS,
                    }
                )
            del source_1, source_2

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": "physics_background_random_cap_negative_control",
            "campaign_root": str(root),
            "campaign_code_commit": design["code_commit"],
            "preflight": design["preflight"],
            "preflight_sha256": design["preflight_sha256"],
            "dataset_1": DATASET_1,
            "dataset_2": DATASET_2,
            "events": EVENTS,
            "strategy": STRATEGY,
            "seed_contract": seed_contract(),
            "generator": "torch.Generator(device=cpu)+torch.randperm",
            "generator_version": str(torch.__version__),
            "tables": table_rows,
        }
        manifest["content_sha256"] = _canonical_hash(manifest)
        atomic_json_dump(manifest, temporary / "manifest.json")
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_frozen_random_pairing_tables(root: Path) -> dict[str, Any]:
    """Reauthenticate a frozen random table directory and every table file."""
    root = root.expanduser().resolve(strict=True)
    _, preflight = _authenticate_campaign(root)
    manifest_path = root / "random_pairing" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_sha256 = manifest.pop("content_sha256", None)
    if content_sha256 != _canonical_hash(manifest):
        raise RuntimeError("Random pairing manifest content hash differs.")
    manifest["content_sha256"] = content_sha256
    if manifest.get("seed_contract") != seed_contract() or len(manifest.get("tables", [])) != 10:
        raise RuntimeError("Random pairing manifest seed/table coverage differs.")
    expected_pairs = {(split, seed) for split, seeds in SEEDS_BY_SPLIT.items() for seed in seeds}
    observed_pairs = {(row["split"], int(row["seed"])) for row in manifest["tables"]}
    if observed_pairs != expected_pairs:
        raise RuntimeError("Random pairing manifest split/seed coverage differs.")
    observed_replicates = {
        (str(row["split"]), int(row["replicate"])) for row in manifest["tables"]
    }
    expected_replicates = {
        (split, replicate)
        for split, seeds in SEEDS_BY_SPLIT.items()
        for replicate in range(len(seeds))
    }
    if observed_replicates != expected_replicates or any(
        int(row["seed"]) != int(SEEDS_BY_SPLIT[str(row["split"])][int(row["replicate"])])
        for row in manifest["tables"]
    ):
        raise RuntimeError("Random pairing manifest replicate/seed ordering differs.")
    artifact_root = (root / "random_pairing" / "tables").resolve(strict=True)
    for row in manifest["tables"]:
        path = Path(row["path"]).resolve(strict=True)
        if not path.is_relative_to(artifact_root) or sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"Frozen random pair table file differs: {path}")
        table = load_pair_table(
            path,
            expected_dataset_1=DATASET_1,
            expected_dataset_2=DATASET_2,
            expected_split=row["split"],
            n_dataset_1=EVENTS,
            n_dataset_2=EVENTS,
            source_1_sha256=row["source_1_sha256"],
            source_2_sha256=row["source_2_sha256"],
        )
        if (
            table["metadata"]["random_seed"] != row["seed"]
            or table["metadata"]["permutation_sha256"] != row["permutation_sha256"]
        ):
            raise RuntimeError(f"Frozen random pair table semantics differ: {path}")
    if Path(preflight["artifact_dir"]).resolve() == (root / "random_pairing").resolve():
        raise RuntimeError("Random controls must not replace the structured pairing directory.")
    return manifest


__all__ = [
    "DATASET_1",
    "DATASET_2",
    "EVENTS",
    "SEEDS_BY_SPLIT",
    "TEST_SEEDS",
    "VALIDATION_SEEDS",
    "build_random_pair_table",
    "freeze_random_pairing_tables",
    "seed_contract",
    "validate_frozen_random_pairing_tables",
]
