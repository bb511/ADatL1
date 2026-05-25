from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import IterableDataset

META_COLUMNS = (
    "timestamp",
    "config",
    "counter",
    "flag",
    "intervention"
)
READOUT_FEATURES = (
    "current",
    "angle_1",
    "angle_2",
    "ir_1",
    "vis_1",
    "ir_2",
    "vis_2",
    "ir_3",
    "vis_3",
    "v_board",
    "v_reg",
)


@dataclass(frozen=True)
class CausalChamberTable:
    name: str
    x: torch.Tensor
    pairing: torch.Tensor
    metadata: dict[str, torch.Tensor]
    feature_names: list[str]
    pairing_feature_names: list[str]
    all_numeric_features: list[str]


@dataclass(frozen=True)
class CausalChamberNormalizer:
    center: torch.Tensor | None
    scale: torch.Tensor | None
    clip_value: float | None

    @classmethod
    def fit(
        cls,
        x: torch.Tensor,
        *,
        normalize: bool,
        robust_quantiles: list[float] | tuple[float, float],
        clip_value: float | None,
    ) -> "CausalChamberNormalizer":
        if not normalize:
            return cls(center=None, scale=None, clip_value=clip_value)

        q_low, q_high = [float(q) for q in robust_quantiles]
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError("robust_quantiles must satisfy 0 <= low < high <= 1.")

        qs = torch.quantile(x, torch.tensor([q_low, 0.5, q_high]), dim=0)
        return cls(
            center=qs[1],
            scale=(qs[2] - qs[0]).clamp_min(1.0e-6),
            clip_value=clip_value,
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.center is None or self.scale is None:
            return x.float().contiguous()

        out = (x - self.center) / self.scale
        if self.clip_value is not None:
            out = out.clamp(-float(self.clip_value), float(self.clip_value))
        return out.float().contiguous()

    def to_contract(self) -> dict[str, Any]:
        return {
            "center": None if self.center is None else self.center.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "clip_value": self.clip_value,
        }


@dataclass(frozen=True)
class CausalChamberContract:
    dataset_name: str
    model_features: list[str]
    pairing_features: list[str]
    excluded_columns: list[str]
    pairing: dict[str, Any]
    splits: dict[str, int]
    intervention_catalog: list[dict[str, Any]]
    signals: list[str]
    normalizer: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalChamberDataset(IterableDataset):
    """Batched iterable dataset for Causal Chamber tensors and metadata."""

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        feature_names: list[str],
        sample_id: torch.Tensor | None = None,
        pair_id: torch.Tensor | None = None,
        flag: torch.Tensor | None = None,
        intervention: torch.Tensor | None = None,
        max_batches: int | None = None,
        shuffler: torch.Generator | None = None,
    ):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same first dimension.")

        n = x.shape[0]
        self.x = x.float().contiguous()
        self.mask = torch.ones_like(self.x, dtype=torch.bool)
        self.l1bit = torch.zeros(n, dtype=torch.bool)
        self.y = y.long().contiguous()
        self.sample_id = (
            torch.arange(n, dtype=torch.int64)
            if sample_id is None
            else sample_id.long().contiguous()
        )
        self.pair_id = self.sample_id.clone() if pair_id is None else pair_id.long().contiguous()
        self.flag = (
            torch.full((n,), -1, dtype=torch.int64) if flag is None else flag.long().contiguous()
        )
        self.intervention = (
            torch.zeros(n, dtype=torch.int64)
            if intervention is None
            else intervention.long().contiguous()
        )
        self.batch_size = int(batch_size)
        self.max_batches = max_batches
        self.shuffler = shuffler
        self.object_feature_map = {
            "chamber": {feature: [idx] for idx, feature in enumerate(feature_names)}
        }

        self.n = n
        self.num_batches = (self.n + self.batch_size - 1) // self.batch_size
        self.starts = torch.arange(self.num_batches, dtype=torch.int64) * self.batch_size

    def __len__(self) -> int:
        if self.max_batches is None:
            return self.num_batches
        return min(self.num_batches, int(self.max_batches))

    def __iter__(self):
        if self.shuffler is not None:
            order = torch.randperm(self.num_batches, generator=self.shuffler)
            starts = self.starts[order]
        else:
            starts = self.starts

        nb = 0
        for start in starts:
            if self.max_batches is not None and nb >= self.max_batches:
                break

            s = int(start)
            e = min(s + self.batch_size, self.n)
            yield {
                "x": self.x[s:e],
                "mask": self.mask[s:e],
                "l1bit": self.l1bit[s:e],
                "y": self.y[s:e],
                "sample_id": self.sample_id[s:e],
                "pair_id": self.pair_id[s:e],
                "flag": self.flag[s:e],
                "intervention": self.intervention[s:e],
            }
            nb += 1


class CausalChamberDataBuilder:
    """Build Causal Chamber train/evaluation datasets and their contract."""

    def __init__(
        self,
        *,
        dataset_dir: Path,
        dataset_name: str,
        feature_set: str,
        feature_columns: list[str] | None,
        signal_experiments: list[str] | None,
        pairing_columns: list[str] | None,
        pairing_strategy: str,
        train_fraction: float,
        val_fraction: float,
        reference_fraction: float,
        signal_val_fraction: float,
        normalize: bool,
        robust_quantiles: list[float] | tuple[float, float],
        clip_value: float | None,
        seed: int,
    ):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.feature_set = feature_set
        self.feature_columns = feature_columns
        self.signal_experiments = signal_experiments
        self.pairing_columns = pairing_columns
        self.pairing_strategy = pairing_strategy
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.reference_fraction = float(reference_fraction)
        self.signal_val_fraction = float(signal_val_fraction)
        self.normalize = bool(normalize)
        self.robust_quantiles = robust_quantiles
        self.clip_value = clip_value
        self.seed = int(seed)

        self.main: dict[str, CausalChamberDataset] = {}
        self.aux: dict[str, dict[str, CausalChamberDataset]] = {
            "valid": {},
            "test": {},
        }
        self.feature_names: list[str] | None = None
        self.pairing_feature_names: list[str] | None = None
        self.object_feature_map: dict[str, dict[str, list[int]]] | None = None
        self.normalizer: CausalChamberNormalizer | None = None
        self.contract: CausalChamberContract | None = None

    def setup(
        self,
        *,
        stage: str | None,
        batch_size: int,
        max_val_batches: int | None,
        train_shuffler: torch.Generator | None,
    ) -> None:
        self._setup_real(
            stage=stage,
            batch_size=batch_size,
            max_val_batches=max_val_batches,
            train_shuffler=train_shuffler,
        )

    def _setup_real(
        self,
        *,
        stage: str | None,
        batch_size: int,
        max_val_batches: int | None,
        train_shuffler: torch.Generator | None,
    ) -> None:
        reference = self.load_table("uniform_reference")
        train_idx, valid_idx, test_idx = self._split_reference_indices(reference.x.size(0))
        self.normalizer = CausalChamberNormalizer.fit(
            reference.x[train_idx],
            normalize=self.normalize,
            robust_quantiles=self.robust_quantiles,
            clip_value=self.clip_value,
        )

        if stage in (None, "fit"):
            self.main["train"] = self._dataset_from_table(
                reference,
                indices=train_idx,
                label=0,
                batch_size=batch_size,
                max_batches=None,
                shuffler=train_shuffler,
            )
            self._setup_real_eval_split("valid", reference, valid_idx, batch_size, max_val_batches)

        if stage in (None, "validate"):
            self._setup_real_eval_split("valid", reference, valid_idx, batch_size, max_val_batches)

        if stage in (None, "test"):
            self._setup_real_eval_split("test", reference, test_idx, batch_size, max_val_batches)

        self._set_contract(
            splits={
                "train": int(train_idx.numel()),
                "valid_base": int(valid_idx.numel()),
                "test_base": int(test_idx.numel()),
                "valid_pairs": self.main.get("valid", self.aux["valid"].get("reference_normal")).n
                if ("valid" in self.main or "reference_normal" in self.aux["valid"])
                else 0,
                "test_pairs": self.main.get("test", self.aux["test"].get("reference_normal")).n
                if ("test" in self.main or "reference_normal" in self.aux["test"])
                else 0,
            }
        )

    def _setup_real_eval_split(
        self,
        split_name: str,
        reference: CausalChamberTable,
        base_indices: torch.Tensor,
        batch_size: int,
        max_val_batches: int | None,
    ) -> None:
        normal_idx, reference_idx, pair_distance = self._paired_real_indices(
            reference, base_indices
        )

        pair_ids = torch.arange(normal_idx.numel(), dtype=torch.int64)
        normal_x = self.normalizer.transform(reference.x[normal_idx])
        reference_x = self.normalizer.transform(reference.x[reference_idx])

        max_batches = self._resolve_max_batches(max_val_batches)
        self.main[split_name] = self._make_dataset(
            normal_x,
            label=0,
            batch_size=batch_size,
            sample_id=reference.metadata["counter"][normal_idx],
            pair_id=pair_ids,
            flag=reference.metadata["flag"][normal_idx],
            intervention=reference.metadata["intervention"][normal_idx],
            max_batches=max_batches,
        )

        aux: dict[str, CausalChamberDataset] = {
            "reference_normal": self._make_dataset(
                reference_x,
                label=-1,
                batch_size=batch_size,
                sample_id=reference.metadata["counter"][reference_idx],
                pair_id=pair_ids,
                flag=reference.metadata["flag"][reference_idx],
                intervention=reference.metadata["intervention"][reference_idx],
                max_batches=max_batches,
            )
        }
        self._store_pairing_diagnostics(split_name, pair_distance)

        for label, name in enumerate(self._signal_names(), start=1):
            signal = self.load_table(name)
            signal_idx = self._signal_indices(signal.x.size(0), split_name)
            aux[name] = self._dataset_from_table(
                signal,
                indices=signal_idx,
                label=label,
                batch_size=batch_size,
                max_batches=max_batches,
            )

        self.aux[split_name] = aux

    def load_table(self, experiment: str) -> CausalChamberTable:
        path = self.dataset_dir / f"{experiment}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Causal Chamber experiment not found: {path}")

        df = pd.read_csv(path)
        feature_names, pairing_feature_names, all_numeric_features = self._resolve_feature_names(
            df
        )
        self._set_feature_names(feature_names, pairing_feature_names, all_numeric_features)

        x_df = df.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        if x_df.isna().any().any():
            bad = list(x_df.columns[x_df.isna().any()])
            raise ValueError(f"NaN/non-numeric values found in columns: {bad}")
        if pairing_feature_names:
            pairing_df = df.loc[:, pairing_feature_names].apply(pd.to_numeric, errors="coerce")
            if pairing_df.isna().any().any():
                bad = list(pairing_df.columns[pairing_df.isna().any()])
                raise ValueError(f"NaN/non-numeric values found in pairing columns: {bad}")
            pairing = torch.as_tensor(pairing_df.to_numpy(), dtype=torch.float32)
        else:
            pairing = torch.empty((len(df), 0), dtype=torch.float32)

        metadata: dict[str, torch.Tensor] = {}
        for col in META_COLUMNS:
            if col not in df.columns or col == "config":
                continue
            values = pd.to_numeric(df[col], errors="coerce").fillna(-1)
            metadata[col] = torch.as_tensor(values.to_numpy(), dtype=torch.float32)
        if "counter" not in metadata:
            metadata["counter"] = torch.arange(len(df), dtype=torch.float32)
        if "flag" not in metadata:
            metadata["flag"] = torch.full((len(df),), -1.0)
        if "intervention" not in metadata:
            metadata["intervention"] = torch.zeros(len(df))

        return CausalChamberTable(
            name=experiment,
            x=torch.as_tensor(x_df.to_numpy(), dtype=torch.float32),
            pairing=pairing,
            metadata=metadata,
            feature_names=feature_names,
            pairing_feature_names=pairing_feature_names,
            all_numeric_features=all_numeric_features,
        )

    def _dataset_from_table(
        self,
        table: CausalChamberTable,
        *,
        indices: torch.Tensor,
        label: int,
        batch_size: int,
        max_batches: int | None,
        shuffler: torch.Generator | None = None,
    ) -> CausalChamberDataset:
        return self._make_dataset(
            self.normalizer.transform(table.x[indices]),
            label=label,
            batch_size=batch_size,
            sample_id=table.metadata["counter"][indices],
            pair_id=table.metadata["counter"][indices],
            flag=table.metadata["flag"][indices],
            intervention=table.metadata["intervention"][indices],
            max_batches=max_batches,
            shuffler=shuffler,
        )

    def _make_dataset(
        self,
        x: torch.Tensor,
        *,
        label: int,
        batch_size: int,
        sample_id: torch.Tensor | None = None,
        pair_id: torch.Tensor | None = None,
        flag: torch.Tensor | None = None,
        intervention: torch.Tensor | None = None,
        max_batches: int | None = None,
        shuffler: torch.Generator | None = None,
    ) -> CausalChamberDataset:
        if self.feature_names is None:
            raise RuntimeError("Feature names are not initialized.")
        return CausalChamberDataset(
            x=x,
            y=torch.full((x.size(0),), label, dtype=torch.int64),
            batch_size=batch_size,
            feature_names=self.feature_names,
            sample_id=sample_id,
            pair_id=pair_id,
            flag=flag,
            intervention=intervention,
            max_batches=max_batches,
            shuffler=shuffler,
        )

    def _resolve_feature_names(self, df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        all_numeric = [
            c for c in df.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
        ]
        if self.feature_set == "readouts":
            names = list(READOUT_FEATURES)
        elif self.feature_set == "all_numeric_no_meta":
            names = all_numeric
        elif self.feature_set == "custom":
            names = list(self.feature_columns or [])
        else:
            raise ValueError("feature_set must be one of: readouts, all_numeric_no_meta, custom.")

        missing = [name for name in names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing requested Causal Chamber columns: {missing}")
        if not names:
            raise ValueError("No Causal Chamber feature columns were selected.")
        if self.pairing_columns is not None:
            pairing_names = list(self.pairing_columns)
        else:
            pairing_names = [name for name in all_numeric if name not in names]

        missing_pairing = [name for name in pairing_names if name not in df.columns]
        if missing_pairing:
            raise ValueError(
                f"Missing requested Causal Chamber pairing columns: {missing_pairing}"
            )
        return names, pairing_names, all_numeric

    def _set_feature_names(
        self,
        feature_names: list[str],
        pairing_feature_names: list[str],
        all_numeric_features: list[str],
    ) -> None:
        if self.feature_names is None:
            self.feature_names = list(feature_names)
            self.pairing_feature_names = list(pairing_feature_names)
            self.all_numeric_features = list(all_numeric_features)
            self.object_feature_map = {
                "chamber": {feature: [idx] for idx, feature in enumerate(self.feature_names)}
            }
            return

        if feature_names != self.feature_names:
            raise ValueError("Feature columns differ across Causal Chamber datasets.")
        if pairing_feature_names != self.pairing_feature_names:
            raise ValueError("Pairing columns differ across Causal Chamber datasets.")

    def _split_reference_indices(
        self, n_total: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_train = int(round(self.train_fraction * n_total))
        n_valid = int(round(self.val_fraction * n_total))
        n_test = n_total - n_train - n_valid
        if min(n_train, n_valid, n_test) <= 1:
            raise RuntimeError("Reference split is too small. Adjust train_fraction/val_fraction.")

        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n_total, generator=gen)
        return perm[:n_train], perm[n_train : n_train + n_valid], perm[n_train + n_valid :]

    def _paired_real_indices(
        self, table: CausalChamberTable, base_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_pairs = max(1, int(round(self.reference_fraction * base_indices.numel())))
        n_pairs = min(n_pairs, base_indices.numel() // 2)
        if n_pairs <= 0:
            raise RuntimeError("Need at least two rows to build paired validation views.")

        gen = torch.Generator().manual_seed(self.seed + base_indices.numel())
        order = torch.randperm(base_indices.numel(), generator=gen)
        normal_pool = base_indices[order[:n_pairs]]
        reference_pool = base_indices[order[n_pairs : 2 * n_pairs]]

        if table.pairing.size(1) == 0 or self.pairing_strategy == "random":
            return normal_pool, reference_pool, torch.full((n_pairs,), float("nan"))
        if self.pairing_strategy != "nearest":
            raise ValueError("pairing_strategy must be one of: nearest, random.")

        x1 = table.pairing[normal_pool]
        x2 = table.pairing[reference_pool]
        combined = torch.cat([x1, x2], dim=0)
        center = combined.mean(dim=0)
        scale = combined.std(dim=0).clamp_min(1.0e-6)
        dists = torch.cdist((x1 - center) / scale, (x2 - center) / scale)

        flat_order = torch.argsort(dists.flatten())
        used_1 = torch.zeros(n_pairs, dtype=torch.bool)
        used_2 = torch.zeros(n_pairs, dtype=torch.bool)
        matched_1 = []
        matched_2 = []
        matched_dist = []
        for flat_idx in flat_order.tolist():
            i = flat_idx // n_pairs
            j = flat_idx % n_pairs
            if used_1[i] or used_2[j]:
                continue
            used_1[i] = True
            used_2[j] = True
            matched_1.append(normal_pool[i])
            matched_2.append(reference_pool[j])
            matched_dist.append(dists[i, j])
            if len(matched_1) == n_pairs:
                break

        return (
            torch.stack(matched_1).long(),
            torch.stack(matched_2).long(),
            torch.stack(matched_dist).float(),
        )

    def _store_pairing_diagnostics(self, split_name: str, pair_distance: torch.Tensor) -> None:
        if not hasattr(self, "_pairing_diagnostics"):
            self._pairing_diagnostics = {}
        finite = pair_distance[torch.isfinite(pair_distance)]
        self._pairing_diagnostics[split_name] = {
            "n_pairs": int(pair_distance.numel()),
            "mean_distance": None if finite.numel() == 0 else float(finite.mean()),
            "median_distance": None if finite.numel() == 0 else float(finite.median()),
        }

    def _signal_indices(self, n_total: int, split_name: str) -> torch.Tensor:
        n_valid = int(round(self.signal_val_fraction * n_total))
        n_valid = min(max(1, n_valid), n_total - 1)
        gen = torch.Generator().manual_seed(self.seed + n_total)
        perm = torch.randperm(n_total, generator=gen)
        if split_name == "valid":
            return perm[:n_valid]
        if split_name == "test":
            return perm[n_valid:]
        raise ValueError(f"Unsupported split '{split_name}'.")

    def _signal_names(self) -> list[str]:
        if self.signal_experiments:
            return list(self.signal_experiments)
        return sorted(
            p.stem for p in self.dataset_dir.glob("*.csv") if p.stem != "uniform_reference"
        )

    def _resolve_max_batches(self, max_batches: int | None) -> int | None:
        if max_batches is not None and int(max_batches) < 0:
            return None
        return max_batches

    def _intervention_catalog(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.dataset_dir.glob("*.csv")):
            name = path.stem
            info = parse_intervention_name(name)
            try:
                head = pd.read_csv(path, usecols=["flag"], nrows=1)
                flag = int(head["flag"].iloc[0])
                n_rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
            except Exception:
                flag = None
                n_rows = None
            out.append({**info, "flag": flag, "n_rows": n_rows})
        return out

    def _set_contract(self, *, splits: dict[str, int]) -> None:
        if self.feature_names is None or self.normalizer is None:
            return
        excluded = [
            c
            for c in [*META_COLUMNS, *getattr(self, "all_numeric_features", [])]
            if c not in self.feature_names
        ]
        self.contract = CausalChamberContract(
            dataset_name=self.dataset_name,
            model_features=list(self.feature_names),
            pairing_features=list(self.pairing_feature_names or []),
            excluded_columns=excluded,
            pairing={
                "type": self.pairing_strategy,
                "dataset_1": "normal",
                "dataset_2": "reference_normal",
                "pair_key": "pair_id",
                "cap_pairing_type": "none",
                "diagnostics": getattr(self, "_pairing_diagnostics", {}),
            },
            splits=splits,
            intervention_catalog=self._intervention_catalog(),
            signals=self._signal_names(),
            normalizer=self.normalizer.to_contract(),
        )


def parse_intervention_name(name: str) -> dict[str, Any]:
    if name == "uniform_reference":
        return {
            "name": name,
            "family": "reference",
            "target": None,
            "strength": None,
        }

    stem = name.removeprefix("uniform_")
    parts = stem.split("_")
    strength = parts[-1] if parts and parts[-1] in {"weak", "mid", "strong"} else None
    target = "_".join(parts[:-1]) if strength else stem
    family = target.split("_")[0] if target else None
    return {
        "name": name,
        "family": family,
        "target": target,
        "strength": strength,
    }
