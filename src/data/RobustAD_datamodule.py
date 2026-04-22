from dataclasses import dataclass
from pathlib import Path
import gc

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.utils import pylogger
from colorama import Fore, Back
from src.data.components.dataset import CIFARADDataset

log = pylogger.RankedLogger(__name__)


@dataclass(frozen=True)
class SplitTensors:
    """Simple container for one in-memory split."""
    x: torch.Tensor
    y: torch.Tensor


class RobustADDataModule(LightningDataModule):
    """Lightning DataModule for RobustAD anomaly detection experiments.

    This module mirrors the structure of the CIFAR10 and physics datamodules:
    the main normal split is stored separately, while auxiliary evaluation splits
    are exposed through auxiliary dataset dictionaries.

    Dataset structure:
      - one chosen subset: "pcb", "metal_parts", or "piled_bags"
      - source-domain split: "train"
      - multiple shifted target-domain splits: "test0", "test1", ...
      - target splits contain both shifted normal and anomalous samples

    Labeling convention:
      - source-domain normal is assigned label 0
      - shifted normal datasets are assigned labels < 0
      - shifted anomaly datasets are assigned labels > 0

    Evaluation loader naming convention:
      - "normal" is always the first/main dataset
      - "shifted_normal_all" is an auxiliary concatenation of all shifted normal sets
      - "shifted_normal_0", "shifted_normal_1", ... are per-shift normal subsets
      - "shifted_anomaly_0", "shifted_anomaly_1", ... are per-shift anomaly subsets

    :param data_dir: Root directory containing RobustAD subset folders.
    :param subset: One of {"pcb", "metal_parts", "piled_bags"}.
    :param image_size: Optional resize target as [H, W]. If None, keeps original size.
    :param batch_size: Global batch size. It is divided across devices if using DDP.
    :param max_val_batches: Maximum number of batches for auxiliary validation/test
        loaders. -1 means full dataset.
    :param val_fraction: Fraction of source normal reserved for validation.
    :param test_fraction: Fraction of source normal reserved for held-out main test.
    :param normalize: Whether to apply standard normalization.
    :param seed: Random seed used for splitting and shuffling.
    :param num_workers: Number of DataLoader workers.
    :param stats_file: Filename used to cache normalization statistics.
    """

    # Number of shifted test domains per subset.
    SHIFT_MAP = {
        "pcb": [f"test{i}" for i in range(6)],
        "piled_bags": [f"test{i}" for i in range(6)],
        "metal_parts": [f"test{i}" for i in range(7)],
    }

    # Folder names used by the downloaded dataset.
    DIR_MAP = {
        "pcb": "PCB",
        "metal_parts": "MetalParts",
        "piled_bags": "PiledBags",
    }

    def __init__(
        self,
        data_dir: str,
        subset: str,
        image_size: list[int] | tuple[int, int] | None = None,
        batch_size: int = 64,
        max_val_batches: int = -1,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        normalize: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        stats_file: str | None = None,
    ) -> None:
        """Store config and initialize internal split caches."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._main = {}
        self._aux = {"valid": {}, "test": {}}
        self.shuffler = torch.Generator().manual_seed(seed)
        self.mean, self.std = None, None
        self._validate_init()

    def _validate_init(self) -> None:
        """Validate basic configuration."""
        subset = str(self.hparams.subset).lower()
        if subset not in self.SHIFT_MAP:
            raise ValueError(f"Unknown subset '{subset}'.")
        if not (0.0 < self.hparams.val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0, 1).")
        if not (0.0 < self.hparams.test_fraction < 1.0):
            raise ValueError("test_fraction must be in (0, 1).")
        if self.hparams.val_fraction + self.hparams.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1.")

    def prepare_data(self) -> None:
        """Download RobustAD once if it is not already on disk."""
        if self._subset_dir().exists():
            return
        Path(self.hparams.data_dir).mkdir(parents=True, exist_ok=True)
        log.info(Back.GREEN + f"Downloading RobustAD to {self.hparams.data_dir}...")
        snapshot_download(
            repo_id="AmazonScience/RobustAD",
            repo_type="dataset",
            local_dir=str(self.hparams.data_dir),
            local_dir_use_symlinks=False,
        )

    def setup(self, stage: str = None) -> None:
        """Load data into memory and prepare requested stage splits."""
        self._set_batch_size()
        if self.hparams.normalize and self.mean is None:
            self._load_or_compute_stats()
        source_train = self._load_split("train", apply_normalize=self.hparams.normalize)
        source_train = self._keep_only_normal(source_train)

        if stage in (None, "fit", "validate"):
            self._setup_fit_validate(source_train)
        if stage in (None, "test"):
            self._setup_test(source_train)
        if stage == "predict":
            raise ValueError("Predict dataloader is not implemented.")
        self._data_summary(stage)

    def train_dataloader(self):
        """Return training dataloader over source-domain normal."""
        return self._loader(self._main["train"], shuffle=True)

    def val_dataloader(self):
        """Return validation dataloaders with normal first."""
        return self._make_eval_loaders("valid")

    def test_dataloader(self):
        """Return test dataloaders with normal first."""
        return self._make_eval_loaders("test")

    def teardown(self, stage: str | None = None) -> None:
        """Release cached tensors between stages."""
        if stage in ("fit", None):
            self._main.pop("train", None)
            self._main.pop("valid", None)
            self._aux["valid"].clear()
        if stage in ("test", None):
            self._main.pop("test", None)
            self._aux["test"].clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move tuple batch tensors to device efficiently."""
        return tuple(self._move_tensor(t, device) for t in batch)

    def _move_tensor(self, t: torch.Tensor, device):
        """Pin CPU tensors before async CUDA transfer."""
        if device.type != "cuda":
            return t.to(device)
        if t.device.type == "cpu" and not t.is_pinned():
            t = t.pin_memory()
        return t.to(device, non_blocking=True)

    def _subset_dir(self) -> Path:
        """Return the directory of the selected RobustAD subset."""
        root = Path(self.hparams.data_dir)
        name = self.DIR_MAP[self.hparams.subset]
        return root / name if (root / name).exists() else root / "data" / name

    def _split_dir(self, split_name: str) -> Path:
        """Return the directory containing one RobustAD split."""
        stem = f"{self.hparams.subset}_data_dir_{split_name}"
        return self._subset_dir() / stem

    def _stats_path(self) -> Path:
        """Return the cached normalization-stats file path."""
        name = self.hparams.stats_file or f"{self.hparams.subset}_normal_stats.pt"
        return self._subset_dir() / name

    def _load_or_compute_stats(self) -> None:
        """Load cached channel stats or compute them from source train normal."""
        path = self._stats_path()
        if path.exists():
            stats = torch.load(path, map_location="cpu")
            self.mean, self.std = stats["mean"].float(), stats["std"].float()
            return
        source = self._load_split("train", apply_normalize=False)
        source = self._keep_only_normal(source)
        self.mean = source.x.mean(dim=(0, 2, 3))
        self.std = source.x.std(dim=(0, 2, 3)).clamp_min(1e-8)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean, "std": self.std}, path)

    def _setup_fit_validate(self, source: SplitTensors) -> None:
        """Split source normal into train/valid/test and build validation aux data."""
        train_ds, valid_ds, test_ds = self._split_source_normal(source)
        self._main.setdefault("train", train_ds)
        self._main.setdefault("valid", valid_ds)
        self._main.setdefault("test", test_ds)
        if not self._aux["valid"]:
            self._aux["valid"] = self._build_shift_aux()

    def _setup_test(self, source: SplitTensors) -> None:
        """Ensure held-out source test and shifted-domain test aux data exist."""
        if "test" not in self._main:
            _, _, test_ds = self._split_source_normal(source)
            self._main["test"] = test_ds
        if not self._aux["test"]:
            self._aux["test"] = self._build_shift_aux()

    def _split_source_normal(self, source: SplitTensors):
        """Split source normal into train, valid, and test subsets."""
        n = source.x.size(0)
        nv = max(2, round(self.hparams.val_fraction * n))
        nt = max(2, round(self.hparams.test_fraction * n))
        nt = min(nt, n - nv - 1)
        ds = TensorDataset(source.x, source.y)
        parts = random_split(ds, [n - nv - nt, nv, nt], generator=self.shuffler)
        return tuple(self._tensor_subset_to_split(p) for p in parts)

    def _tensor_subset_to_split(self, subset) -> SplitTensors:
        """Convert a TensorDataset subset into stacked tensors."""
        xs, ys = zip(*subset)
        return SplitTensors(
            x=torch.stack(xs).contiguous(),
            y=torch.stack(ys).contiguous(),
        )

    def _build_shift_aux(self) -> dict[str, SplitTensors]:
        """Build shifted normal/anomaly aux datasets for all target domains."""
        out, shifted = {}, []
        neg, pos = 0, 0
        for i, split_name in enumerate(self.SHIFT_MAP[self.hparams.subset]):
            split = self._load_split(split_name, apply_normalize=self.hparams.normalize)
            nmask, amask = split.y == 0, split.y == 1
            if nmask.any():
                neg -= 1
                sn = self._masked_split(split, nmask, neg)
                out[f"shifted_normal_{i}"] = sn
                shifted.append(sn)
            if amask.any():
                pos += 1
                out[f"shifted_anomaly_{i}"] = self._masked_split(split, amask, pos)

        if shifted:
            out["shifted_normal_all"] = self._concat_balanced(shifted)
        return out

    def _masked_split(self, split: SplitTensors, mask: torch.Tensor, label: int) -> SplitTensors:
        """Select a subset of a split and overwrite all labels with one value."""
        x = split.x[mask].contiguous()
        y = torch.full((x.size(0),), label, dtype=torch.int64)
        return SplitTensors(x=x, y=y)

    def _concat_balanced(self, splits: list[SplitTensors]) -> SplitTensors:
        """Merge shifted-normal splits with equal contribution from each domain."""
        n = min(s.x.size(0) for s in splits)
        idxs = [torch.randperm(s.x.size(0), generator=self.shuffler)[:n] for s in splits]
        xs = [s.x[i] for s, i in zip(splits, idxs)]
        ys = [s.y[i] for s, i in zip(splits, idxs)]
        return SplitTensors(
            x=torch.cat(xs).contiguous(),
            y=torch.cat(ys).contiguous(),
        )

    def _load_split(self, split_name: str, apply_normalize: bool = True) -> SplitTensors:
        cache_path = self._cache_path(split_name, normalized=apply_normalize)
        if cache_path.exists():
            data = torch.load(cache_path, map_location="cpu")
            return SplitTensors(x=data["x"], y=data["y"])

        raw_cache_path = self._cache_path(split_name, normalized=False)
        if raw_cache_path.exists():
            data = torch.load(raw_cache_path, map_location="cpu")
            split = SplitTensors(x=data["x"], y=data["y"])
        else:
            ds = load_dataset("imagefolder", data_dir=str(self._split_dir(split_name)), split="train")
            rows = [self._row_to_tensors(r, apply_normalize=False) for r in ds]
            xs, ys = zip(*rows)
            split = SplitTensors(
                x=torch.stack(xs).contiguous(),
                y=torch.tensor(ys, dtype=torch.int64),
            )
            raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"x": split.x, "y": split.y}, raw_cache_path)

        if not apply_normalize:
            return split

        x = self._normalize(split.x, apply=True).contiguous()
        norm_split = SplitTensors(x=x, y=split.y)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": norm_split.x, "y": norm_split.y}, cache_path)
        return norm_split

    def _row_to_tensors(self, row, apply_normalize: bool):
        """Convert one HF row into image tensor and image-level label."""
        x = self._to_image_tensor(row["image"])
        x = self._normalize(x, apply_normalize)
        y = int(row["label"])
        return x, y

    def _to_image_tensor(self, img) -> torch.Tensor:
        img = img if isinstance(img, Image.Image) else Image.open(img)
        img = img.convert("RGB")
        if self.hparams.image_size is not None:
            h, w = map(int, self.hparams.image_size)
            img = img.resize((w, h), Image.BILINEAR)
        x = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0
        return x

    def _normalize(self, x: torch.Tensor, apply: bool) -> torch.Tensor:
        """Apply channel-wise standard normalization if requested."""
        if not (self.hparams.normalize and apply):
            return x
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalization requested but stats are not set.")
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def _make_eval_loaders(self, key: str) -> dict[str, DataLoader]:
        """Construct eval dataloaders with normal first, then aux datasets."""
        loaders = {"normal": self._loader(self._main[key], shuffle=False)}
        for name, split in self._aux[key].items():
            loaders[name] = self._loader(split, shuffle=False, max_b=self.hparams.max_val_batches)
        return loaders

    def _loader(self, split: SplitTensors, shuffle: bool, max_b: int | None = None) -> DataLoader:
        """Wrap one in-memory split in the iterable dataset and DataLoader."""
        ds = CIFARADDataset(
            split.x,
            split.y,
            batch_size=self.batch_size_per_device,
            max_batches=max_b,
            shuffler=self.shuffler if shuffle else None,
        )
        return DataLoader(
            ds,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available(),
        )

    def _set_batch_size(self) -> None:
        """Convert global batch size into per-device batch size."""
        if self.trainer is None:
            self.batch_size_per_device = self.hparams.batch_size
            return
        ws = self.trainer.world_size
        if self.hparams.batch_size % ws != 0:
            raise RuntimeError(f"Batch size {self.hparams.batch_size} not divisible by {ws}.")
        self.batch_size_per_device = self.hparams.batch_size // ws

    def _data_summary(self, stage: str | None) -> None:
        """Print a summary of the currently prepared splits."""
        log.info(Fore.MAGENTA + "-" * 5 + " Data Summary " + "-" * 5)
        if stage in (None, "fit"):
            self._log_split("Training data:", "train")
            self._log_split("Validation data:", "valid", aux_key="valid")
        elif stage == "validate":
            self._log_split("Validation data:", "valid", aux_key="valid")
        elif stage == "test":
            self._log_split("Test data:", "test", aux_key="test")

    def _log_split(self, title: str, key: str, aux_key: str | None = None) -> None:
        """Log the tensor shapes for one main split and optional aux splits."""
        log.info(Fore.GREEN + title)
        if key in self._main:
            log.info(f"normal: {tuple(self._main[key].x.shape)}")
        if aux_key is not None:
            for name, split in self._aux[aux_key].items():
                log.info(f"{name}: {tuple(split.x.shape)}")

    def _keep_only_normal(self, split: SplitTensors) -> SplitTensors:
        mask = split.y == 0
        return SplitTensors(
            x=split.x[mask].contiguous(),
            y=split.y[mask].contiguous(),
        )

    def _cache_path(self, split_name: str, normalized: bool) -> Path:
        size = "orig" if self.hparams.image_size is None else "x".join(map(str, self.hparams.image_size))
        norm = "norm" if normalized else "raw"
        name = f"{self.hparams.subset}_{split_name}_{size}_{norm}.pt"
        return self._subset_dir() / "tensor_cache" / name
