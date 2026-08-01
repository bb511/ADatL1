"""Build deterministic physics-control pairings directly from L1 tensor caches.

This producer intentionally does not instantiate the L1 datamodule: doing so loads
every auxiliary signal sample even though a pairing table needs only ZeroBias and
one background simulation.  It reads the exact ordered caches that CAP later sees,
fits descriptor scaling on ZeroBias *training* data, and records source hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle  # nosec B403 -- trusted local datamaker normalization artifacts only
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.pairing.artifacts import (
    full_pairing_artifact,
    save_full_pairing_artifact,
)
from src.utils.pairing.matching import deterministic_one_to_one_match
from src.utils.pairing.physics import PhysicsFeatureSchema, PhysicsPairingDescriptor
from src.utils.pairing.table import (
    atomic_json_dump,
    atomic_torch_save,
    sha256_file,
    sha256_tensor,
    validate_pair_table,
)
from src.utils.pairing.utils import PairingResult, pair_table_dict

DEFAULT_CACHE_RELATIVE = Path("data_2025E+G/mlready/eminimalTauFET_pdefaultTauFET_default/robust")
DEFAULT_TARGET_DATASET = "SingleNeutrino_E-10-gun"


def parse_args() -> argparse.Namespace:
    """Parse production pair-table generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/iopsstor/scratch/cscs/podagiu/data") / DEFAULT_CACHE_RELATIVE,
        help="Robust mlready cache containing train/valid/test and aux/.",
    )
    parser.add_argument("--stage", choices=("validate", "test"), required=True)
    parser.add_argument("--target-dataset", default=DEFAULT_TARGET_DATASET)
    parser.add_argument(
        "--strategy",
        choices=(
            "flat_physical",
            "physics_summary",
            "typed_sliced_wasserstein",
        ),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--max-target-events",
        type=int,
        default=81_920,
        help="Ordered target prefix collected by the configured CAP experiment.",
    )
    parser.add_argument(
        "--cap-prefix-events",
        type=int,
        nargs="*",
        default=(81_920, 163_840),
        help="Also materialize CAP tables for these common runtime prefixes.",
    )
    parser.add_argument("--fit-events", type=int, default=200_000)
    parser.add_argument("--closure-events", type=int, default=10_000)
    parser.add_argument("--initial-k", type=int, default=64)
    parser.add_argument("--max-k", type=int, default=256)
    parser.add_argument("--caliper-quantile", type=float, default=0.99)
    parser.add_argument("--no-caliper", action="store_true")
    parser.add_argument("--backend", choices=("auto", "faiss", "torch"), default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-batch-size", type=int, default=131_072)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--reference-batch-size", type=int, default=262_144)
    parser.add_argument("--audit-events", type=int, default=20_000)
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_tensor(path: Path, *, limit: int | None = None) -> torch.Tensor:
    """Memory-map a tensor cache and optionally retain an ordered prefix."""
    if not path.is_file():
        raise FileNotFoundError(f"L1 tensor cache does not exist: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not torch.is_tensor(value):
        raise TypeError(f"Expected a tensor cache at {path}.")
    return value if limit is None else value[: min(int(limit), value.shape[0])]


def load_split(
    cache_root: Path,
    split: str,
    *,
    dataset: str | None = None,
    limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Memory-map one ordered ZeroBias or auxiliary split."""
    split_name = "valid" if split == "validate" else split
    folder = (
        cache_root / split_name if dataset is None else cache_root / "aux" / dataset / split_name
    )
    x = _load_tensor(folder / "torch_cache.pt", limit=limit)
    mask = _load_tensor(folder / "torch_mask.pt", limit=limit)
    if x.shape != mask.shape:
        raise ValueError(f"Data/mask shape mismatch under {folder}: {x.shape} vs {mask.shape}.")
    return x, mask, folder


def load_physics_schema(cache_root: Path) -> PhysicsFeatureSchema:
    """Recover normalization calibration and hardware LSBs for a cache."""
    repository_root = Path(__file__).resolve().parents[3]
    with (cache_root / "object_feature_map.json").open(encoding="utf-8") as handle:
        feature_map = json.load(handle)
    with (repository_root / "configs/data/l1_scales/pairing.yaml").open(
        encoding="utf-8"
    ) as handle:
        l1_scales = yaml.safe_load(handle)

    n_features = 1 + max(
        int(index)
        for object_map in feature_map.values()
        for indices in object_map.values()
        for index in indices
    )
    shift = torch.zeros(n_features, dtype=torch.float32)
    scale = torch.ones(n_features, dtype=torch.float32)
    for object_type, object_map in feature_map.items():
        parameter_path = cache_root / f"{object_type}_norm_params.pkl"
        if not parameter_path.is_file():
            raise FileNotFoundError(f"Normalization parameters missing: {parameter_path}")
        with parameter_path.open("rb") as handle:
            # These files are generated locally by the pinned adl1t datamaker and
            # are never accepted from an untrusted runtime input.
            parameters = pickle.load(handle)  # nosec B301
        for feature, indices in object_map.items():
            if feature not in parameters:
                continue
            index = torch.as_tensor(indices, dtype=torch.long)
            shift[index] = float(parameters[feature]["shift"])
            scale[index] = float(parameters[feature]["scale"])
    return PhysicsFeatureSchema(feature_map, l1_scales, shift, scale)


@torch.no_grad()
def transform_in_batches(
    descriptor: PhysicsPairingDescriptor,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Transform a memory-mapped source without materializing physical events."""
    chunks = []
    for start in range(0, x.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), x.shape[0])
        x_chunk = x[start:stop].to(device, non_blocking=False)
        mask_chunk = mask[start:stop].to(device, non_blocking=False)
        chunks.append(descriptor.transform(x_chunk, mask_chunk).cpu())
        del x_chunk, mask_chunk
    return torch.cat(chunks, dim=0).contiguous()


def _state_digest(state: Mapping[str, Any]) -> str:
    """Hash descriptor semantics independently of torch serialization."""
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode())
        value = state[key]
        if torch.is_tensor(value):
            digest.update(sha256_tensor(value).encode())
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _smd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute per-coordinate standardized mean differences."""
    pooled = 0.5 * (x.var(0, unbiased=False) + y.var(0, unbiased=False))
    return (x.mean(0) - y.mean(0)).abs() / torch.sqrt(pooled + 1e-8)


def _energy_distance(x: torch.Tensor, y: torch.Tensor, maximum: int = 2_000) -> float:
    """Estimate multivariate energy distance on deterministic prefixes."""
    n = min(x.shape[0], y.shape[0], maximum)
    x = x[:n].float()
    y = y[:n].float()
    cross = torch.cdist(x, y).mean()
    within_x = torch.cdist(x, x).mean()
    within_y = torch.cdist(y, y).mean()
    return float((2 * cross - within_x - within_y).item())


def _domain_auc(x: torch.Tensor, y: torch.Tensor, maximum: int = 20_000) -> float:
    """Audit separability with a seeded logistic domain classifier."""
    n = min(x.shape[0], y.shape[0], maximum)
    features = np.concatenate((x[:n].numpy(), y[:n].numpy()))
    labels = np.concatenate((np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)))
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.4,
        random_state=271828,
        stratify=labels,
    )
    classifier = LogisticRegression(max_iter=500, random_state=271828)
    classifier.fit(train_x, train_y)
    return float(roc_auc_score(test_y, classifier.predict_proba(test_x)[:, 1]))


def balance_diagnostics(
    target: torch.Tensor,
    reference: torch.Tensor,
    target_to_reference: torch.Tensor,
    *,
    maximum: int,
) -> dict[str, float]:
    """Compare an unpaired reference prefix with the deterministically selected one."""
    valid_target = torch.nonzero(target_to_reference >= 0, as_tuple=False).flatten()
    n = min(int(maximum), target.shape[0], reference.shape[0], valid_target.numel())
    target_rows = valid_target[:n]
    target_sample = target[target_rows]
    before = reference[:n]
    after = reference[target_to_reference[target_rows]]
    smd_before = _smd(target_sample, before)
    smd_after = _smd(target_sample, after)
    return {
        "smd_before_mean": float(smd_before.mean()),
        "smd_before_max": float(smd_before.max()),
        "smd_after_mean": float(smd_after.mean()),
        "smd_after_max": float(smd_after.max()),
        "energy_distance_before": _energy_distance(target_sample, before),
        "energy_distance_after": _energy_distance(target_sample, after),
        "domain_auc_before": _domain_auc(target_sample, before),
        "domain_auc_after": _domain_auc(target_sample, after),
        "audit_events": n,
    }


def _validate_args(args: argparse.Namespace) -> None:
    """Reject inconsistent or unsafe production settings."""
    for name in (
        "max_target_events",
        "fit_events",
        "closure_events",
        "initial_k",
        "max_k",
        "transform_batch_size",
        "query_batch_size",
        "reference_batch_size",
        "audit_events",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.max_k < args.initial_k:
        raise ValueError("--max-k must be at least --initial-k.")
    if any(int(value) <= 0 for value in args.cap_prefix_events):
        raise ValueError("--cap-prefix-events values must be positive.")
    if not 0.0 <= args.caliper_quantile <= 1.0:
        raise ValueError("--caliper-quantile must be between zero and one.")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")


def main() -> None:
    """Build descriptors, assignments, CAP tables, and diagnostics."""
    args = parse_args()
    _validate_args(args)
    cache_root = args.cache_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.stage}_{args.strategy}"
    state_path = out_dir / f"{stem}_descriptor_state.pt"
    full_path = out_dir / f"{stem}_full.pt"
    cap_path = out_dir / f"{stem}_cap.pt"
    diagnostics_path = out_dir / f"{stem}_diagnostics.json"
    candidate_path = out_dir / f"{stem}_candidates.pt"
    prefix_paths = [
        out_dir / f"{stem}_cap_n{int(n)}.pt"
        for n in sorted(set(args.cap_prefix_events))
        if int(n) < args.max_target_events
    ]
    if not args.overwrite:
        outputs = [state_path, full_path, cap_path, diagnostics_path, *prefix_paths]
        if args.save_candidates:
            outputs.append(candidate_path)
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing artifacts: {existing}")

    device = torch.device(args.device)
    schema = load_physics_schema(cache_root)
    train_x, train_mask, _ = load_split(cache_root, "train")
    reference_x, reference_mask, reference_folder = load_split(cache_root, args.stage)
    target_x, target_mask, target_folder = load_split(
        cache_root,
        args.stage,
        dataset=args.target_dataset,
        limit=args.max_target_events,
    )
    descriptor = PhysicsPairingDescriptor(
        schema,
        kind=args.strategy,
        canonicalize_flat=False,
        fit_max_events=args.fit_events,
    )
    descriptor.fit(train_x, train_mask)
    state = descriptor.state_dict()
    atomic_torch_save(state, state_path, overwrite=args.overwrite)
    state_sha256 = sha256_file(state_path)

    print(f"Transforming {reference_x.shape[0]:,} reference events ({args.strategy})...")
    reference = transform_in_batches(
        descriptor,
        reference_x,
        reference_mask,
        batch_size=args.transform_batch_size,
        device=device,
    )
    print(f"Transforming {target_x.shape[0]:,} target events ({args.strategy})...")
    target = transform_in_batches(
        descriptor,
        target_x,
        target_mask,
        batch_size=args.transform_batch_size,
        device=device,
    )

    closure_n = min(args.closure_events, train_x.shape[0])
    closure = transform_in_batches(
        descriptor,
        train_x[:closure_n],
        train_mask[:closure_n],
        batch_size=args.transform_batch_size,
        device=device,
    )
    print(f"Matching {target.shape[0]:,} targets into {reference.shape[0]:,} references...")
    pairing, candidates = deterministic_one_to_one_match(
        target.to(device),
        reference.to(device),
        initial_k=args.initial_k,
        max_k=args.max_k,
        backend=args.backend,
        query_batch_size=args.query_batch_size,
        reference_batch_size=args.reference_batch_size,
    )
    if pairing.n_pairs != target.shape[0]:
        print(
            f"Candidate graph assigned {pairing.n_pairs}/{target.shape[0]} targets; "
            "unassigned target-aligned entries remain explicit (-1/inf/False)."
        )

    caliper = None
    closure_quantiles: dict[str, float] = {}
    if not args.no_caliper:
        closure_pairing, _ = deterministic_one_to_one_match(
            closure.to(device),
            reference.to(device),
            initial_k=args.initial_k,
            max_k=args.max_k,
            backend=args.backend,
            query_batch_size=args.query_batch_size,
            reference_batch_size=args.reference_batch_size,
        )
        closure_distance = closure_pairing.distance[closure_pairing.valid]
        caliper = float(torch.quantile(closure_distance, args.caliper_quantile))
        closure_quantiles = {
            "closure_distance_q50": float(torch.quantile(closure_distance, 0.50)),
            "closure_distance_q95": float(torch.quantile(closure_distance, 0.95)),
            "closure_distance_q99": float(torch.quantile(closure_distance, 0.99)),
        }
    caliper_valid = pairing.valid.clone()
    if caliper is not None:
        caliper_valid &= pairing.distance <= caliper

    target_index = torch.nonzero(caliper_valid, as_tuple=False).flatten()
    reference_index = pairing.target_to_reference[target_index]
    source_reference_sha256 = sha256_tensor(reference_x.flatten(start_dim=1))
    source_target_sha256 = sha256_tensor(target_x.flatten(start_dim=1))
    common_metadata: dict[str, Any] = {
        "producer": "src.utils.pairing.physics_tables",
        "strategy": args.strategy,
        "descriptor_state_sha256": state_sha256,
        "descriptor_state_semantic_sha256": _state_digest(state),
        "schema_signature": schema.signature(),
        "source_reference_sha256": source_reference_sha256,
        "source_target_sha256": source_target_sha256,
        "source_reference_folder": str(reference_folder),
        "source_target_folder": str(target_folder),
        "n_target_full_source": int(_load_tensor(target_folder / "torch_cache.pt").shape[0]),
        "target_prefix_events": int(target_x.shape[0]),
        "descriptor_dimension": int(target.shape[1]),
        "fit_events": int(args.fit_events),
        "fit_source": str(cache_root / "train"),
        "initial_k": int(args.initial_k),
        "final_k": int(candidates.k),
        "max_k": int(args.max_k),
        "search_backend": args.backend,
        "caliper": caliper,
        "caliper_quantile": None if args.no_caliper else float(args.caliper_quantile),
        "caliper_accepted": int(caliper_valid.sum()),
        "caliper_coverage": float(caliper_valid.float().mean()),
        **closure_quantiles,
    }

    accepted_pairing = replace(pairing, caliper_valid=caliper_valid)
    full = full_pairing_artifact(
        accepted_pairing,
        target_dataset=args.target_dataset,
        reference_dataset="normal",
        split=args.stage,
        strategy=args.strategy,
        metadata=common_metadata,
    )
    save_full_pairing_artifact(full, full_path, overwrite=args.overwrite)

    pairs = PairingResult(
        idx_1=reference_index,
        idx_2=target_index,
        distance=pairing.distance[target_index],
        rank_1_to_2=torch.zeros(target_index.numel(), dtype=torch.long),
        rank_2_to_1=pairing.candidate_rank[target_index],
    )
    cap_metadata = {
        **common_metadata,
        "n_dataset_1": int(reference_x.shape[0]),
        "n_dataset_2": int(target_x.shape[0]),
        "n_pairs": int(target_index.numel()),
        "coverage": float(target_index.numel() / target_x.shape[0]),
        "encoder_checkpoint_sha256": state_sha256,
        "source_1_sha256": source_reference_sha256,
        "source_2_sha256": source_target_sha256,
        "data_seed": 123,
        "pairing_orientation": "dataset_2_target_to_dataset_1_reference",
    }
    table = pair_table_dict(
        pairs,
        dataset_1="normal",
        dataset_2=args.target_dataset,
        split=args.stage,
        encoder_ckpt=str(state_path),
        metadata=cap_metadata,
    )
    validate_pair_table(table)
    atomic_torch_save(table, cap_path, overwrite=args.overwrite)

    prefix_tables: dict[str, dict[str, Any]] = {}
    for prefix in sorted({int(n) for n in args.cap_prefix_events}):
        if prefix >= target_x.shape[0]:
            continue
        prefix_valid = caliper_valid[:prefix]
        prefix_target = torch.nonzero(prefix_valid, as_tuple=False).flatten()
        prefix_reference = pairing.target_to_reference[:prefix][prefix_target]
        prefix_source_sha256 = sha256_tensor(target_x[:prefix].flatten(start_dim=1))
        prefix_result = PairingResult(
            idx_1=prefix_reference,
            idx_2=prefix_target,
            distance=pairing.distance[:prefix][prefix_target],
            rank_1_to_2=torch.zeros(prefix_target.numel(), dtype=torch.long),
            rank_2_to_1=pairing.candidate_rank[:prefix][prefix_target],
        )
        prefix_metadata = {
            **cap_metadata,
            "n_dataset_2": prefix,
            "n_pairs": int(prefix_target.numel()),
            "coverage": float(prefix_target.numel() / prefix),
            "source_2_sha256": prefix_source_sha256,
            "source_target_sha256": prefix_source_sha256,
            "target_prefix_events": prefix,
            "parent_full_artifact": str(full_path),
        }
        prefix_table = pair_table_dict(
            prefix_result,
            dataset_1="normal",
            dataset_2=args.target_dataset,
            split=args.stage,
            encoder_ckpt=str(state_path),
            metadata=prefix_metadata,
        )
        validate_pair_table(prefix_table)
        prefix_path = out_dir / f"{stem}_cap_n{prefix}.pt"
        atomic_torch_save(prefix_table, prefix_path, overwrite=args.overwrite)
        prefix_tables[str(prefix)] = {
            "path": str(prefix_path),
            "sha256": sha256_file(prefix_path),
            "n_pairs": int(prefix_target.numel()),
            "coverage": float(prefix_target.numel() / prefix),
        }

    if args.save_candidates:
        atomic_torch_save(
            {
                "squared_distance": candidates.squared_distance,
                "reference_index": candidates.reference_index,
                "target_dataset": args.target_dataset,
                "reference_dataset": "normal",
                "split": args.stage,
                "strategy": args.strategy,
                "descriptor_state_sha256": state_sha256,
            },
            candidate_path,
            overwrite=args.overwrite,
        )

    diagnostics = {
        "status": "passed",
        "strategy": args.strategy,
        "split": args.stage,
        "full_assignment_coverage": pairing.coverage,
        "caliper_coverage": float(caliper_valid.float().mean()),
        "distance_mean": float(pairing.distance[pairing.valid].mean()),
        "distance_p95": float(torch.quantile(pairing.distance[pairing.valid], 0.95)),
        "candidate_rank_mean": float(pairing.candidate_rank[pairing.valid].float().mean()),
        "candidate_rank_max": int(pairing.candidate_rank.max()),
        "state_sha256": state_sha256,
        "full_artifact": str(full_path),
        "full_artifact_sha256": sha256_file(full_path),
        "cap_table": str(cap_path),
        "cap_table_sha256": sha256_file(cap_path),
        "cap_prefix_tables": prefix_tables,
        **closure_quantiles,
        **balance_diagnostics(
            target,
            reference,
            pairing.target_to_reference,
            maximum=args.audit_events,
        ),
    }
    atomic_json_dump(diagnostics, diagnostics_path, overwrite=args.overwrite)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
