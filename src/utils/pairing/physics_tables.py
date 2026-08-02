"""Build deterministic background-to-background pairings from L1 tensor caches.

The producer reads the exact ordered auxiliary prefixes that CAP later sees.  Dataset
1 is background 0 and dataset 2 is background 1; the emitted dense map therefore has
the literal contract ``map_0_to_1[i] == j``.
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
    FullPairingTensors,
    full_pairing_artifact,
    save_full_pairing_artifact,
)
from src.utils.pairing.jetclr import encode_in_batches, load_frozen_encoder
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
DEFAULT_DATASET_1 = "ZB_run396102"
DEFAULT_DATASET_2 = "ZB_run398183"


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
    parser.add_argument(
        "--source-metadata-dir",
        type=Path,
        required=True,
        help="Directory containing zerobias_sources.json and split source-ID arrays.",
    )
    parser.add_argument(
        "--dataset-1",
        default=DEFAULT_DATASET_1,
        help="Background 0; the domain of the emitted dense map.",
    )
    parser.add_argument(
        "--dataset-2",
        default=DEFAULT_DATASET_2,
        help="Background 1; the codomain of the emitted dense map.",
    )
    parser.add_argument(
        "--strategy",
        choices=(
            "flat_physical",
            "physics_summary",
            "typed_sliced_wasserstein",
            "jetclr",
        ),
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--events",
        "--max-target-events",
        dest="events",
        type=int,
        default=81_920,
        help="Ordered prefix from both auxiliary datasets collected by CAP.",
    )
    parser.add_argument(
        "--cap-prefix-events",
        type=int,
        nargs="*",
        default=(),
        help="Also materialize CAP tables for these common runtime prefixes.",
    )
    parser.add_argument("--fit-events", type=int, default=200_000)
    parser.add_argument("--initial-k", type=int, default=64)
    parser.add_argument("--max-k", type=int, default=256)
    parser.add_argument("--caliper-quantile", type=float, default=0.99)
    parser.add_argument(
        "--use-caliper",
        action="store_true",
        help="Compute a diagnostic caliper. It never removes rows from the CAP map.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "faiss", "faiss_hnsw", "torch"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transform-batch-size", type=int, default=131_072)
    parser.add_argument("--jetclr-batch-size", type=int, default=8192)
    parser.add_argument(
        "--jetclr-checkpoint",
        type=Path,
        help="Frozen encoder checkpoint; required when --strategy=jetclr.",
    )
    parser.add_argument("--jetclr-config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--jetclr-config-name", default="train")
    parser.add_argument(
        "--jetclr-config-override",
        action="append",
        dest="jetclr_config_overrides",
        help=(
            "Hydra override used to instantiate the frozen encoder. Repeat as needed; "
            "defaults to experiment=physics/jetclr_aad_best."
        ),
    )
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
    source_metadata_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Memory-map one ordered main or auxiliary split."""
    split_name = "valid" if split == "validate" else split
    folder = cache_root / split_name
    source_root = source_metadata_dir or cache_root
    source_metadata = source_root / "zerobias_sources.json"
    if dataset is not None and source_metadata.is_file():
        with source_metadata.open(encoding="utf-8") as handle:
            source_names = json.load(handle)["sources"]
        if dataset in source_names:
            source_id = torch.from_numpy(
                np.load(source_root / split_name / "zerobias_source_id.npy")
            )
            selected = torch.nonzero(
                source_id == source_names.index(dataset), as_tuple=False
            ).flatten()
            if limit is not None:
                selected = selected[: int(limit)]
            x = _load_tensor(folder / "torch_cache.pt")[selected]
            mask = _load_tensor(folder / "torch_mask.pt")[selected]
        else:
            folder = cache_root / "aux" / dataset / split_name
            x = _load_tensor(folder / "torch_cache.pt", limit=limit)
            mask = _load_tensor(folder / "torch_mask.pt", limit=limit)
    else:
        if dataset is not None:
            folder = cache_root / "aux" / dataset / split_name
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


def _complete_pairing(
    pairing: FullPairingTensors,
    dataset_1: torch.Tensor,
    dataset_2: torch.Tensor,
    *,
    residual_rank: int,
) -> FullPairingTensors:
    """Deterministically complete a sparse candidate assignment.

    The nearest-neighbour graph handles the overwhelming majority of rows. Any residual unmatched
    rows are ordered by a fixed, data-derived scalar projection and paired in rank order. This
    guarantees a total, unique map without allocating an intractable dense N-by-N cost matrix.
    """
    if pairing.n_pairs == dataset_1.shape[0]:
        return pairing
    missing_1 = torch.nonzero(~pairing.valid, as_tuple=False).flatten()
    missing_2 = torch.nonzero(pairing.reference_to_target < 0, as_tuple=False).flatten()
    if missing_2.numel() < missing_1.numel():
        raise ValueError("Not enough unused dataset-2 rows to complete the one-to-one map.")

    dimension = dataset_1.shape[1]
    weights = torch.linspace(1.0, 2.0, dimension, dtype=torch.float64)
    weights /= torch.linalg.vector_norm(weights)
    score_1 = dataset_1[missing_1].double() @ weights
    score_2 = dataset_2[missing_2].double() @ weights
    order_1 = torch.argsort(score_1, stable=True)
    order_2 = torch.argsort(score_2, stable=True)[: missing_1.numel()]
    rows_1 = missing_1[order_1]
    rows_2 = missing_2[order_2]

    target_to_reference = pairing.target_to_reference.clone()
    reference_to_target = pairing.reference_to_target.clone()
    distance = pairing.distance.clone()
    valid = pairing.valid.clone()
    candidate_rank = pairing.candidate_rank.clone()
    target_to_reference[rows_1] = rows_2
    reference_to_target[rows_2] = rows_1
    distance[rows_1] = torch.linalg.vector_norm(
        dataset_1[rows_1].float() - dataset_2[rows_2].float(), dim=1
    )
    valid[rows_1] = True
    candidate_rank[rows_1] = int(residual_rank)
    completed = FullPairingTensors(
        target_to_reference=target_to_reference,
        reference_to_target=reference_to_target,
        distance=distance,
        valid=valid,
        caliper_valid=valid.clone(),
        candidate_rank=candidate_rank,
    )
    completed.validate()
    return completed


def _validate_args(args: argparse.Namespace) -> None:
    """Reject inconsistent or unsafe production settings."""
    for name in (
        "events",
        "fit_events",
        "initial_k",
        "max_k",
        "transform_batch_size",
        "jetclr_batch_size",
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
    if args.strategy == "jetclr":
        if args.jetclr_checkpoint is None:
            raise ValueError("--jetclr-checkpoint is required when --strategy=jetclr.")
        if not args.jetclr_checkpoint.expanduser().is_file():
            raise FileNotFoundError(f"JetCLR checkpoint does not exist: {args.jetclr_checkpoint}")


def main() -> None:
    """Build descriptors, assignments, CAP tables, and diagnostics."""
    args = parse_args()
    _validate_args(args)
    cache_root = args.cache_root.expanduser().resolve()
    source_metadata_dir = args.source_metadata_dir.expanduser().resolve()
    source_metadata_path = source_metadata_dir / "zerobias_sources.json"
    with source_metadata_path.open(encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.stage}_{args.strategy}"
    state_path = out_dir / f"{stem}_descriptor_state.pt"
    full_path = out_dir / f"{stem}_full.pt"
    cap_path = out_dir / f"{stem}_cap_n{int(args.events)}.pt"
    diagnostics_path = out_dir / f"{stem}_diagnostics.json"
    candidate_path = out_dir / f"{stem}_candidates.pt"
    prefix_paths = [
        out_dir / f"{stem}_cap_n{int(n)}.pt"
        for n in sorted(set(args.cap_prefix_events))
        if int(n) < args.events
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
    dataset_1_x, dataset_1_mask, dataset_1_folder = load_split(
        cache_root,
        args.stage,
        dataset=args.dataset_1,
        limit=args.events,
        source_metadata_dir=source_metadata_dir,
    )
    dataset_2_x, dataset_2_mask, dataset_2_folder = load_split(
        cache_root,
        args.stage,
        dataset=args.dataset_2,
        limit=args.events,
        source_metadata_dir=source_metadata_dir,
    )
    if dataset_2_x.shape[0] < dataset_1_x.shape[0]:
        raise ValueError(
            f"A total one-to-one 0->1 map requires dataset 2 to have at least as many "
            f"rows as dataset 1 ({dataset_2_x.shape[0]} < {dataset_1_x.shape[0]})."
        )
    if args.strategy == "jetclr":
        checkpoint = args.jetclr_checkpoint.expanduser().resolve()
        config_overrides = args.jetclr_config_overrides or ["experiment=physics/jetclr_aad_best"]
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        encoder = load_frozen_encoder(
            checkpoint,
            schema.object_feature_map,
            config_dir=args.jetclr_config_dir,
            config_name=args.jetclr_config_name,
            overrides=config_overrides,
            device=device,
        )
        checkpoint_sha256 = sha256_file(checkpoint)
        state = {
            "kind": "jetclr",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "config_dir": str(args.jetclr_config_dir),
            "config_name": args.jetclr_config_name,
            "config_overrides": config_overrides,
            "l2_normalized": True,
            "deterministic_algorithms": True,
            "tf32": False,
            "object_feature_map_sha256": sha256_file(cache_root / "object_feature_map.json"),
        }
        atomic_torch_save(state, state_path, overwrite=args.overwrite)
        print(f"Encoding {dataset_1_x.shape[0]:,} background-0 events (JetCLR)...")
        dataset_1 = encode_in_batches(
            encoder,
            dataset_1_x,
            dataset_1_mask,
            batch_size=args.jetclr_batch_size,
            device=device,
        )
        print(f"Encoding {dataset_2_x.shape[0]:,} background-1 events (JetCLR)...")
        dataset_2 = encode_in_batches(
            encoder,
            dataset_2_x,
            dataset_2_mask,
            batch_size=args.jetclr_batch_size,
            device=device,
        )
        encoder_ckpt = str(checkpoint)
        fit_events = 0
        fit_source = "frozen checkpoint; no pairing-time fitting"
        del encoder
    else:
        descriptor = PhysicsPairingDescriptor(
            schema,
            kind=args.strategy,
            canonicalize_flat=False,
            fit_max_events=args.fit_events,
        )
        # Fit metric scaling symmetrically on equal deterministic prefixes of both
        # backgrounds. Pairing remains entirely independent of anomaly labels.
        fit_each = min(max(1, args.fit_events // 2), dataset_1_x.shape[0], dataset_2_x.shape[0])
        fit_x = torch.cat((dataset_1_x[:fit_each], dataset_2_x[:fit_each]), dim=0)
        fit_mask = torch.cat((dataset_1_mask[:fit_each], dataset_2_mask[:fit_each]), dim=0)
        descriptor.fit(fit_x, fit_mask)
        del fit_x, fit_mask
        state = descriptor.state_dict()
        atomic_torch_save(state, state_path, overwrite=args.overwrite)
        checkpoint_sha256 = sha256_file(state_path)
        print(f"Transforming {dataset_1_x.shape[0]:,} background-0 events ({args.strategy})...")
        dataset_1 = transform_in_batches(
            descriptor,
            dataset_1_x,
            dataset_1_mask,
            batch_size=args.transform_batch_size,
            device=device,
        )
        print(f"Transforming {dataset_2_x.shape[0]:,} background-1 events ({args.strategy})...")
        dataset_2 = transform_in_batches(
            descriptor,
            dataset_2_x,
            dataset_2_mask,
            batch_size=args.transform_batch_size,
            device=device,
        )
        encoder_ckpt = str(state_path)
        fit_events = int(args.fit_events)
        fit_source = "balanced deterministic prefixes of dataset_1 and dataset_2"

    state_sha256 = sha256_file(state_path)

    print(f"Matching {dataset_1.shape[0]:,} background-0 rows into background 1...")
    pairing, candidates = deterministic_one_to_one_match(
        dataset_1.to(device),
        dataset_2.to(device),
        initial_k=args.initial_k,
        max_k=args.max_k,
        backend=args.backend,
        query_batch_size=args.query_batch_size,
        reference_batch_size=args.reference_batch_size,
    )
    pairing = _complete_pairing(pairing, dataset_1, dataset_2, residual_rank=candidates.k + 1)
    if pairing.n_pairs != dataset_1.shape[0]:
        raise RuntimeError("Failed to construct a total background-0 to background-1 map.")

    caliper = None
    closure_quantiles: dict[str, float] = {}
    if args.use_caliper:
        assigned_distance = pairing.distance[pairing.valid]
        caliper = float(torch.quantile(assigned_distance, args.caliper_quantile))
        closure_quantiles = {
            "pair_distance_q50": float(torch.quantile(assigned_distance, 0.50)),
            "pair_distance_q95": float(torch.quantile(assigned_distance, 0.95)),
            "pair_distance_q99": float(torch.quantile(assigned_distance, 0.99)),
        }
    caliper_valid = pairing.valid.clone()
    if caliper is not None:
        caliper_valid &= pairing.distance <= caliper

    dataset_1_index = torch.arange(dataset_1_x.shape[0], dtype=torch.long)
    dataset_2_index = pairing.target_to_reference
    source_1_sha256 = sha256_tensor(dataset_1_x.flatten(start_dim=1))
    source_2_sha256 = sha256_tensor(dataset_2_x.flatten(start_dim=1))
    common_metadata: dict[str, Any] = {
        "producer": "src.utils.pairing.physics_tables",
        "strategy": args.strategy,
        "descriptor_state_sha256": state_sha256,
        "descriptor_state_semantic_sha256": _state_digest(state),
        "schema_signature": schema.signature(),
        "source_1_sha256": source_1_sha256,
        "source_2_sha256": source_2_sha256,
        "source_1_folder": str(dataset_1_folder),
        "source_2_folder": str(dataset_2_folder),
        "source_metadata_path": str(source_metadata_path),
        "source_metadata_sha256": sha256_file(source_metadata_path),
        "n_dataset_1_full_source": int(_load_tensor(dataset_1_folder / "torch_cache.pt").shape[0]),
        "n_dataset_2_full_source": int(_load_tensor(dataset_2_folder / "torch_cache.pt").shape[0]),
        "prefix_events": int(dataset_1_x.shape[0]),
        "descriptor_dimension": int(dataset_1.shape[1]),
        "fit_events": fit_events,
        "fit_source": fit_source,
        "initial_k": int(args.initial_k),
        "final_k": int(candidates.k),
        "max_k": int(args.max_k),
        "search_backend": args.backend,
        "caliper": caliper,
        "caliper_quantile": float(args.caliper_quantile) if args.use_caliper else None,
        "caliper_accepted": int(caliper_valid.sum()),
        "caliper_coverage": float(caliper_valid.float().mean()),
        **closure_quantiles,
    }

    accepted_pairing = replace(pairing, caliper_valid=caliper_valid)
    full = full_pairing_artifact(
        accepted_pairing,
        target_dataset=args.dataset_1,
        reference_dataset=args.dataset_2,
        split=args.stage,
        strategy=args.strategy,
        metadata=common_metadata,
    )
    save_full_pairing_artifact(full, full_path, overwrite=args.overwrite)

    pairs = PairingResult(
        idx_1=dataset_1_index,
        idx_2=dataset_2_index,
        distance=pairing.distance,
        rank_1_to_2=pairing.candidate_rank,
        rank_2_to_1=torch.zeros(dataset_1_index.numel(), dtype=torch.long),
    )
    cap_metadata = {
        **common_metadata,
        "n_dataset_1": int(dataset_1_x.shape[0]),
        "n_dataset_2": int(dataset_2_x.shape[0]),
        "n_pairs": int(dataset_1_index.numel()),
        "coverage": 1.0,
        "encoder_checkpoint_sha256": checkpoint_sha256,
        "source_1_sha256": source_1_sha256,
        "source_2_sha256": source_2_sha256,
        "data_seed": int(source_metadata["seed"]),
        "pairing_orientation": "dataset_1_background0_to_dataset_2_background1",
    }
    table = pair_table_dict(
        pairs,
        dataset_1=args.dataset_1,
        dataset_2=args.dataset_2,
        split=args.stage,
        encoder_ckpt=encoder_ckpt,
        metadata=cap_metadata,
    )
    table["map_0_to_1"] = pairing.target_to_reference.clone()
    validate_pair_table(table)
    atomic_torch_save(table, cap_path, overwrite=args.overwrite)

    prefix_tables: dict[str, dict[str, Any]] = {}
    for prefix in sorted({int(n) for n in args.cap_prefix_events}):
        if prefix >= dataset_1_x.shape[0]:
            continue
        prefix_pairing, prefix_candidates = deterministic_one_to_one_match(
            dataset_1[:prefix].to(device),
            dataset_2[:prefix].to(device),
            initial_k=args.initial_k,
            max_k=args.max_k,
            backend=args.backend,
            query_batch_size=args.query_batch_size,
            reference_batch_size=args.reference_batch_size,
        )
        prefix_pairing = _complete_pairing(
            prefix_pairing,
            dataset_1[:prefix],
            dataset_2[:prefix],
            residual_rank=prefix_candidates.k + 1,
        )
        prefix_dataset_1 = torch.arange(prefix, dtype=torch.long)
        prefix_dataset_2 = prefix_pairing.target_to_reference
        prefix_source_1_sha256 = sha256_tensor(dataset_1_x[:prefix].flatten(start_dim=1))
        prefix_source_2_sha256 = sha256_tensor(dataset_2_x[:prefix].flatten(start_dim=1))
        prefix_result = PairingResult(
            idx_1=prefix_dataset_1,
            idx_2=prefix_dataset_2,
            distance=prefix_pairing.distance,
            rank_1_to_2=prefix_pairing.candidate_rank,
            rank_2_to_1=torch.zeros(prefix, dtype=torch.long),
        )
        prefix_metadata = {
            **cap_metadata,
            "n_dataset_1": prefix,
            "n_dataset_2": prefix,
            "n_pairs": prefix,
            "coverage": 1.0,
            "source_1_sha256": prefix_source_1_sha256,
            "source_2_sha256": prefix_source_2_sha256,
            "prefix_events": prefix,
            "parent_full_artifact": str(full_path),
        }
        prefix_table = pair_table_dict(
            prefix_result,
            dataset_1=args.dataset_1,
            dataset_2=args.dataset_2,
            split=args.stage,
            encoder_ckpt=encoder_ckpt,
            metadata=prefix_metadata,
        )
        prefix_table["map_0_to_1"] = prefix_dataset_2.clone()
        validate_pair_table(prefix_table)
        prefix_path = out_dir / f"{stem}_cap_n{prefix}.pt"
        atomic_torch_save(prefix_table, prefix_path, overwrite=args.overwrite)
        prefix_tables[str(prefix)] = {
            "path": str(prefix_path),
            "sha256": sha256_file(prefix_path),
            "n_pairs": prefix,
            "coverage": 1.0,
        }

    if args.save_candidates:
        atomic_torch_save(
            {
                "squared_distance": candidates.squared_distance,
                "reference_index": candidates.reference_index,
                "target_dataset": args.dataset_1,
                "reference_dataset": args.dataset_2,
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
            dataset_1,
            dataset_2,
            pairing.target_to_reference,
            maximum=args.audit_events,
        ),
    }
    atomic_json_dump(diagnostics, diagnostics_path, overwrite=args.overwrite)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
