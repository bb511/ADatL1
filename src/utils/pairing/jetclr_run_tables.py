"""Encode two processed ZeroBias runs and build a deterministic JetCLR pairing.

The smaller run is the target side. Every target receives one unique reference
from the larger run. The full artifact follows ``src.utils.pairing.artifacts``
and is therefore directly comparable to the deterministic physics artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle  # nosec B403 -- trusted local normalization artifacts only
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import awkward as ak
import hydra
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from src.utils.pairing.artifacts import (
    full_pairing_artifact,
    save_full_pairing_artifact,
)
from src.utils.pairing.io import compose_config
from src.utils.pairing.matching import deterministic_iterative_pairing
from src.utils.pairing.table import atomic_json_dump, sha256_file

SELECTED_FEATURES = {
    "FET": ("Et", "eta", "phi"),
    "egammas": ("Et", "eta", "phi"),
    "jets": ("Et", "eta", "phi"),
    "muons": ("Et", "eta", "phi"),
    "taus": ("Et", "eta", "phi"),
}


@dataclass(frozen=True)
class RunSchema:
    """Frozen preprocessing state used by the selected JetCLR checkpoint."""

    object_feature_map: dict[str, dict[str, list[int]]]
    normalization: dict[str, dict[str, dict[str, float]]]

    @property
    def n_features(self) -> int:
        """Return the flattened feature width."""
        return 1 + max(
            index
            for object_map in self.object_feature_map.values()
            for indices in object_map.values()
            for index in indices
        )


def load_run_schema(training_cache: Path) -> RunSchema:
    """Load the exact feature layout and robust parameters used for training."""
    training_cache = training_cache.expanduser().resolve()
    with (training_cache / "object_feature_map.json").open(encoding="utf-8") as handle:
        feature_map = json.load(handle)
    normalization = {}
    for object_name in SELECTED_FEATURES:
        path = training_cache / f"{object_name}_norm_params.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            normalization[object_name] = pickle.load(handle)  # nosec B301
    schema = RunSchema(feature_map, normalization)
    expected = set(SELECTED_FEATURES)
    if set(feature_map) != expected:
        raise ValueError(
            f"Training feature map has objects {sorted(feature_map)}, expected {sorted(expected)}."
        )
    return schema


class ProcessedRunReader:
    """Stream aligned processed Parquet objects into the trained flat schema."""

    def __init__(
        self,
        run_dir: Path,
        schema: RunSchema,
        batch_size: int,
        max_events: int | None = None,
    ):
        self.run_dir = run_dir.expanduser().resolve()
        self.schema = schema
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.paths = {name: self.run_dir / f"{name}.parquet" for name in SELECTED_FEATURES}
        missing = [path for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Processed run is missing selected objects: {missing}")
        counts = {
            name: pq.ParquetFile(path).metadata.num_rows for name, path in self.paths.items()
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"Processed run object tables are not event aligned: {counts}")
        self.n_source_events = next(iter(counts.values()))
        if max_events is not None and int(max_events) <= 0:
            raise ValueError("max_events must be positive or None.")
        self.n_events = (
            self.n_source_events
            if max_events is None
            else min(int(max_events), self.n_source_events)
        )

    def __iter__(self) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        """Yield ``(row offset, values, masks)`` without changing event order."""
        iterators = {
            name: pq.ParquetFile(path).iter_batches(
                batch_size=self.batch_size,
                use_threads=True,
            )
            for name, path in self.paths.items()
        }
        offset = 0
        while offset < self.n_events:
            batches = {}
            for name, iterator in iterators.items():
                try:
                    batches[name] = next(iterator)
                except StopIteration as error:
                    raise ValueError(f"{name} ended before the other aligned tables.") from error
            sizes = {batch.num_rows for batch in batches.values()}
            if len(sizes) != 1:
                raise ValueError(f"Processed object batch sizes diverged at row {offset}: {sizes}")
            size = sizes.pop()
            size = min(size, self.n_events - offset)
            values = torch.zeros((size, self.schema.n_features), dtype=torch.float32)
            masks = torch.zeros((size, self.schema.n_features), dtype=torch.bool)
            for object_name, batch in batches.items():
                self._fill_object(
                    values,
                    masks,
                    object_name=object_name,
                    array=ak.from_arrow(batch.slice(0, size)),
                )
            yield offset, values, masks
            offset += size
        if self.n_events < self.n_source_events:
            return
        for name, iterator in iterators.items():
            try:
                next(iterator)
            except StopIteration:
                continue
            raise ValueError(f"{name} contains rows beyond the aligned event count.")

    def _fill_object(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        *,
        object_name: str,
        array: ak.Array,
    ) -> None:
        """Normalize, pad, and place one object family in the flat layout."""
        feature_map = self.schema.object_feature_map[object_name]
        parameters = self.schema.normalization[object_name]
        for feature in SELECTED_FEATURES[object_name]:
            indices = feature_map.get(feature, [])
            if not indices:
                continue
            if feature not in array.fields:
                continue
            padded = ak.pad_none(array[feature], len(indices), axis=-1, clip=True)
            present = ~ak.is_none(padded, axis=-1)
            filled = ak.fill_none(padded, 0)
            params = parameters[feature]
            normalized = (filled - float(params["shift"])) / float(params["scale"])
            # Padding is introduced after normalization in the training pipeline.
            normalized = ak.where(present, normalized, 0.0)
            index = torch.as_tensor(indices, dtype=torch.long)
            values[:, index] = torch.from_numpy(
                np.asarray(ak.to_numpy(normalized), dtype=np.float32)
            )
            masks[:, index] = torch.from_numpy(np.asarray(ak.to_numpy(present), dtype=np.bool_))

    def source_hashes(self) -> dict[str, str]:
        """Hash every selected source table independently."""
        return {path.name: sha256_file(path) for path in self.paths.values()}


def load_frozen_encoder(
    checkpoint: Path,
    schema: RunSchema,
    *,
    config_dir: Path,
    config_name: str,
    overrides: list[str],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate only the encoder and strictly restore its checkpoint weights."""
    cfg = compose_config(config_dir=config_dir, config_name=config_name, overrides=overrides)
    encoder = hydra.utils.instantiate(cfg.algorithm.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint has no Lightning state_dict: {checkpoint}")
    encoder_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.set_object_feature_map(schema.object_feature_map)
    encoder.to(device).eval()
    return encoder


@torch.inference_mode()
def encode_processed_run(
    *,
    run_dir: Path,
    training_cache: Path,
    checkpoint: Path,
    output: Path,
    config_dir: Path,
    config_name: str,
    overrides: list[str],
    batch_size: int,
    max_events: int | None,
    device_name: str,
    overwrite: bool,
) -> Path:
    """Encode one complete processed run into an ordered normalized matrix."""
    output = output.expanduser().resolve()
    metadata_path = output.with_suffix(output.suffix + ".json")
    partial = output.with_suffix(output.suffix + ".partial")
    existing = [path for path in (output, metadata_path, partial) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing encoder artifacts: {existing}")
    for path in existing:
        path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    schema = load_run_schema(training_cache)
    reader = ProcessedRunReader(run_dir, schema, batch_size, max_events=max_events)
    encoder = load_frozen_encoder(
        checkpoint.expanduser().resolve(),
        schema,
        config_dir=config_dir,
        config_name=config_name,
        overrides=overrides,
        device=device,
    )
    embedding_dim = int(getattr(encoder, "out_dim"))
    matrix = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(reader.n_events, embedding_dim),
    )
    for offset, values, masks in reader:
        stop = offset + values.shape[0]
        embedding = encoder(values.to(device), masks.to(device)).float()
        embedding = F.normalize(embedding, dim=1)
        if not torch.isfinite(embedding).all():
            raise ValueError(f"Non-finite JetCLR embeddings at rows [{offset}, {stop}).")
        matrix[offset:stop] = embedding.cpu().numpy()
        if offset == 0 or stop == reader.n_events or stop % (100 * batch_size) == 0:
            print(f"Encoded {stop:,}/{reader.n_events:,} events from {reader.run_dir.name}.")
    matrix.flush()
    del matrix
    partial.replace(output)
    metadata = {
        "schema_version": 1,
        "artifact_type": "jetclr_ordered_embeddings",
        "dataset": reader.run_dir.name,
        "source_dir": str(reader.run_dir),
        "n_events": reader.n_events,
        "n_source_events": reader.n_source_events,
        "embedding_dim": embedding_dim,
        "dtype": "float32",
        "l2_normalized": True,
        "event_order": "unchanged processed Parquet row order",
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_cache": str(training_cache.expanduser().resolve()),
        "object_feature_map_sha256": sha256_file(training_cache / "object_feature_map.json"),
        "source_parquet_sha256": reader.source_hashes(),
        "embedding_sha256": sha256_file(output),
        "config_name": config_name,
        "config_overrides": overrides,
        "inference_device": str(device),
        "inference_batch_size": int(batch_size),
        "deterministic_algorithms": True,
        "tf32": False,
    }
    atomic_json_dump(metadata, metadata_path, overwrite=overwrite)
    return output


def _load_embedding_artifact(path: Path) -> tuple[np.ndarray, dict]:
    """Memory-map an authenticated ordered embedding matrix."""
    path = path.expanduser().resolve()
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Embedding artifact or metadata missing for {path}.")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    matrix = np.load(path, mmap_mode="r")
    expected_shape = (int(metadata["n_events"]), int(metadata["embedding_dim"]))
    if matrix.shape != expected_shape or matrix.dtype != np.float32:
        raise ValueError(
            f"Embedding matrix is {matrix.shape}/{matrix.dtype}, expected {expected_shape}."
        )
    if sha256_file(path) != metadata["embedding_sha256"]:
        raise ValueError(f"Embedding artifact hash mismatch: {path}")
    return matrix, metadata


def build_pairing(
    *,
    target_embeddings: Path,
    reference_embeddings: Path,
    output: Path,
    backend: str,
    k: int,
    nlist: int,
    nprobe: int,
    train_events: int,
    train_iterations: int,
    search_batch_size: int,
    add_batch_size: int,
    threads: int,
    seed: int,
    overwrite: bool,
) -> Path:
    """Build and save the complete bidirectional run-to-run pairing artifact."""
    target, target_metadata = _load_embedding_artifact(target_embeddings)
    reference, reference_metadata = _load_embedding_artifact(reference_embeddings)
    if target_metadata["checkpoint_sha256"] != reference_metadata["checkpoint_sha256"]:
        raise ValueError("Both runs must be encoded by the same JetCLR checkpoint.")
    tensors, rounds = deterministic_iterative_pairing(
        target,
        reference,
        backend=backend,
        k=k,
        nlist=nlist,
        nprobe=nprobe,
        train_events=train_events,
        train_iterations=train_iterations,
        search_batch_size=search_batch_size,
        add_batch_size=add_batch_size,
        threads=threads,
        seed=seed,
    )
    finite_distance = tensors.distance[tensors.valid]
    metadata = {
        "producer": "src.utils.pairing.jetclr_run_tables",
        "checkpoint_sha256": target_metadata["checkpoint_sha256"],
        "target_embedding_sha256": target_metadata["embedding_sha256"],
        "reference_embedding_sha256": reference_metadata["embedding_sha256"],
        "pairing_orientation": "target_run_to_reference_run",
        "matching": "iterative_rank_greedy",
        "search_backend": backend,
        "faiss_k": int(k),
        "faiss_nlist": int(nlist),
        "faiss_nprobe": int(nprobe),
        "faiss_train_events": int(train_events),
        "faiss_train_iterations": int(train_iterations),
        "seed": int(seed),
        "rounds": rounds,
        "all_targets_paired": True,
        "one_to_one": True,
        "caliper": None,
        "distance_mean": float(finite_distance.mean()),
        "distance_p50": float(torch.quantile(finite_distance, 0.50)),
        "distance_p95": float(torch.quantile(finite_distance, 0.95)),
        "distance_max": float(finite_distance.max()),
    }
    artifact = full_pairing_artifact(
        tensors,
        target_dataset=target_metadata["dataset"],
        reference_dataset=reference_metadata["dataset"],
        split="full_run",
        strategy="jetclr_cosine",
        metadata=metadata,
    )
    output = save_full_pairing_artifact(artifact, output, overwrite=overwrite)
    diagnostics = {
        **metadata,
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "n_target": tensors.n_target,
        "n_reference": tensors.n_reference,
        "n_pairs": tensors.n_pairs,
        "unused_reference": tensors.n_reference - tensors.n_pairs,
    }
    atomic_json_dump(
        diagnostics,
        output.with_suffix(output.suffix + ".json"),
        overwrite=overwrite,
    )
    return output


def parse_args() -> argparse.Namespace:
    """Parse the staged production commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    encode = subparsers.add_parser("encode")
    encode.add_argument("--run-dir", type=Path, required=True)
    encode.add_argument("--training-cache", type=Path, required=True)
    encode.add_argument("--checkpoint", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--config-dir", type=Path, default=Path("configs"))
    encode.add_argument("--config-name", default="train")
    encode.add_argument("--batch-size", type=int, default=8192)
    encode.add_argument("--max-events", type=int, default=None)
    encode.add_argument("--device", default="cuda:0")
    encode.add_argument("--overwrite", action="store_true")
    encode.add_argument("overrides", nargs="*", default=["experiment=physics/jetclr_aad_best"])

    match = subparsers.add_parser("match")
    match.add_argument("--target-embeddings", type=Path, required=True)
    match.add_argument("--reference-embeddings", type=Path, required=True)
    match.add_argument("--output", type=Path, required=True)
    match.add_argument("--backend", choices=("flat", "ivf_flat"), default="ivf_flat")
    match.add_argument("--k", type=int, default=8)
    match.add_argument("--nlist", type=int, default=16384)
    match.add_argument("--nprobe", type=int, default=32)
    match.add_argument("--train-events", type=int, default=262144)
    match.add_argument("--train-iterations", type=int, default=12)
    match.add_argument("--search-batch-size", type=int, default=131072)
    match.add_argument("--add-batch-size", type=int, default=262144)
    match.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 288))
    match.add_argument("--seed", type=int, default=271828)
    match.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Dispatch encoding or matching."""
    args = parse_args()
    if args.command == "encode":
        output = encode_processed_run(
            run_dir=args.run_dir,
            training_cache=args.training_cache,
            checkpoint=args.checkpoint,
            output=args.output,
            config_dir=args.config_dir,
            config_name=args.config_name,
            overrides=args.overrides,
            batch_size=args.batch_size,
            max_events=args.max_events,
            device_name=args.device,
            overwrite=args.overwrite,
        )
    else:
        output = build_pairing(
            target_embeddings=args.target_embeddings,
            reference_embeddings=args.reference_embeddings,
            output=args.output,
            backend=args.backend,
            k=args.k,
            nlist=args.nlist,
            nprobe=args.nprobe,
            train_events=args.train_events,
            train_iterations=args.train_iterations,
            search_batch_size=args.search_batch_size,
            add_batch_size=args.add_batch_size,
            threads=args.threads,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    print(output)


if __name__ == "__main__":
    main()
