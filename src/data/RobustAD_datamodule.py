from dataclasses import dataclass
from pathlib import Path
import gc

import torch
from PIL import Image
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.utils import pylogger
from colorama import Fore, Back
from src.data.components.dataset import RobustADDataset

log = pylogger.RankedLogger(__name__)


@dataclass(frozen=True)
class SplitTensors:
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor | None = None


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

    SHIFT_MAP = {
        "pcb": [f"test{i}" for i in range(6)],
        "piled_bags": [f"test{i}" for i in range(6)],
        "metal_parts": [f"test{i}" for i in range(7)],
    }

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
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._main = {}
        self._aux = {"valid": {}, "test": {}}
        self.shuffler = torch.Generator().manual_seed(seed)
        self.mean, self.std = None, None
        self._validate_init()

    def _validate_init(self) -> None:
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
        self._set_batch_size()
        if self.hparams.normalize and self.mean is None:
            self._load_or_compute_stats()
        source_train = self._load_split("train", apply_normalize=self.hparams.normalize)
        if stage in (None, "fit", "validate"):
            self._setup_fit_validate(source_train)
        if stage in (None, "test"):
            self._setup_test(source_train)
        if stage == "predict":
            raise ValueError("Predict dataloader is not implemented.")
        self._data_summary(stage)

    def train_dataloader(self):
        return self._loader(self._main["train"], shuffle=True)

    def val_dataloader(self):
        return self._make_eval_loaders("valid")

    def test_dataloader(self):
        return self._make_eval_loaders("test")

    def teardown(self, stage: str | None = None) -> None:
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
        return tuple(self._move_tensor(t, device) for t in batch)

    def _move_tensor(self, t: torch.Tensor, device):
        if device.type != "cuda":
            return t.to(device)
        if t.device.type == "cpu" and not t.is_pinned():
            t = t.pin_memory()
        return t.to(device, non_blocking=True)

    def _subset_dir(self) -> Path:
        root = Path(self.hparams.data_dir)
        name = self.DIR_MAP[self.hparams.subset]
        return root / name if (root / name).exists() else root / "data" / name

    def _split_dir(self, split_name: str) -> Path:
        stem = f"{self.hparams.subset}_data_dir_{split_name}"
        return self._subset_dir() / stem

    def _stats_path(self) -> Path:
        name = self.hparams.stats_file or f"{self.hparams.subset}_normal_stats.pt"
        return self._subset_dir() / name

    def _load_or_compute_stats(self) -> None:
        path = self._stats_path()
        if path.exists():
            stats = torch.load(path, map_location="cpu")
            self.mean, self.std = stats["mean"].float(), stats["std"].float()
            return
        source = self._load_split("train", apply_normalize=False)
        self.mean = source.x.mean(dim=(0, 2, 3))
        self.std = source.x.std(dim=(0, 2, 3)).clamp_min(1e-8)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean, "std": self.std}, path)

    def _setup_fit_validate(self, source: SplitTensors) -> None:
        train_ds, valid_ds, test_ds = self._split_source_normal(source)
        self._main.setdefault("train", train_ds)
        self._main.setdefault("valid", valid_ds)
        self._main.setdefault("test", test_ds)
        if not self._aux["valid"]:
            self._aux["valid"] = self._build_shift_aux()

    def _setup_test(self, source: SplitTensors) -> None:
        if "test" not in self._main:
            _, _, test_ds = self._split_source_normal(source)
            self._main["test"] = test_ds
        if not self._aux["test"]:
            self._aux["test"] = self._build_shift_aux()

    def _split_source_normal(self, source: SplitTensors):
        n = source.x.size(0)
        nv = max(2, round(self.hparams.val_fraction * n))
        nt = max(2, round(self.hparams.test_fraction * n))
        nt = min(nt, n - nv - 1)
        ds = TensorDataset(source.x, source.y)
        parts = random_split(ds, [n - nv - nt, nv, nt], generator=self.shuffler)
        return tuple(self._tensor_subset_to_split(p) for p in parts)

    def _tensor_subset_to_split(self, subset) -> SplitTensors:
        xs, ys = zip(*subset)
        return SplitTensors(x=torch.stack(xs).contiguous(), y=torch.stack(ys).contiguous())

    def _build_shift_aux(self) -> dict[str, SplitTensors]:
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
        x = split.x[mask].contiguous()
        y = torch.full((x.size(0),), label, dtype=torch.int64)
        m = split.mask[mask].contiguous() if split.mask is not None else None
        return SplitTensors(x=x, y=y, mask=m)

    def _concat_balanced(self, splits: list[SplitTensors]) -> SplitTensors:
        n = min(s.x.size(0) for s in splits)
        idxs = [torch.randperm(s.x.size(0), generator=self.shuffler)[:n] for s in splits]
        xs = [s.x[i] for s, i in zip(splits, idxs)]
        ys = [s.y[i] for s, i in zip(splits, idxs)]
        ms = [s.mask[i] for s, i in zip(splits, idxs) if s.mask is not None]
        mask = torch.cat(ms).contiguous() if len(ms) == len(splits) else None
        return SplitTensors(x=torch.cat(xs).contiguous(), y=torch.cat(ys).contiguous(), mask=mask)

    def _load_split(self, split_name: str, apply_normalize: bool = True) -> SplitTensors:
        ds = load_dataset("imagefolder", data_dir=str(self._split_dir(split_name)), split="train")
        rows = [self._row_to_tensors(r, apply_normalize) for r in ds]
        xs, ys, ms = zip(*rows)
        mask = torch.stack(ms).contiguous() if any(m is not None for m in ms) else None
        if mask is None:
            return SplitTensors(x=torch.stack(xs).contiguous(), y=torch.tensor(ys, dtype=torch.int64))
        filled = [m if m is not None else torch.zeros_like(next(mm for mm in ms if mm is not None)) for m in ms]
        return SplitTensors(
            x=torch.stack(xs).contiguous(),
            y=torch.tensor(ys, dtype=torch.int64),
            mask=torch.stack(filled).contiguous(),
        )

    def _row_to_tensors(self, row, apply_normalize: bool):
        x = self._to_image_tensor(row["image"])
        x = self._normalize(x, apply_normalize)
        y = int(row["label"])
        m = self._to_mask_tensor(row["mask"]) if "mask" in row and row["mask"] is not None else None
        return x, y, m

    def _to_image_tensor(self, img) -> torch.Tensor:
        img = img if isinstance(img, Image.Image) else Image.open(img)
        img = img.convert("RGB")
        if self.hparams.image_size is not None:
            h, w = map(int, self.hparams.image_size)
            img = img.resize((w, h), Image.BILINEAR)
        x = torch.as_tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())))
        return x.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float() / 255.0

    def _to_mask_tensor(self, mask) -> torch.Tensor:
        mask = mask if isinstance(mask, Image.Image) else Image.open(mask)
        mask = mask.convert("L")
        if self.hparams.image_size is not None:
            h, w = map(int, self.hparams.image_size)
            mask = mask.resize((w, h), Image.NEAREST)
        m = torch.as_tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes())))
        m = m.view(mask.size[1], mask.size[0]).unsqueeze(0).float() / 255.0
        return (m > 0.5).float()

    def _normalize(self, x: torch.Tensor, apply: bool) -> torch.Tensor:
        if not (self.hparams.normalize and apply):
            return x
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalization requested but stats are not set.")
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def _make_eval_loaders(self, key: str) -> dict[str, DataLoader]:
        loaders = {"normal": self._loader(self._main[key], shuffle=False)}
        for name, split in self._aux[key].items():
            loaders[name] = self._loader(split, shuffle=False, max_b=self.max_val_batches)
        return loaders

    def _loader(self, split: SplitTensors, shuffle: bool, max_b: int | None = None) -> DataLoader:
        ds = RobustADDataset(
            split.x, split.y, mask=split.mask, batch_size=self.batch_size_per_device,
            max_batches=max_b, shuffler=self.shuffler if shuffle else None,
        )
        return DataLoader(
            ds, batch_size=None, shuffle=False, num_workers=self.hparams.num_workers,
            persistent_workers=False, pin_memory=torch.cuda.is_available(),
        )

    def _set_batch_size(self) -> None:
        if self.trainer is None:
            self.batch_size_per_device = self.hparams.batch_size
            return
        ws = self.trainer.world_size
        if self.hparams.batch_size % ws != 0:
            raise RuntimeError(f"Batch size {self.hparams.batch_size} not divisible by {ws}.")
        self.batch_size_per_device = self.hparams.batch_size // ws

    def _data_summary(self, stage: str | None) -> None:
        log.info(Fore.MAGENTA + "-" * 5 + " Data Summary " + "-" * 5)
        if stage in (None, "fit"):
            self._log_split("Training data:", "train")
            self._log_split("Validation data:", "valid", aux_key="valid")
        elif stage == "validate":
            self._log_split("Validation data:", "valid", aux_key="valid")
        elif stage == "test":
            self._log_split("Test data:", "test", aux_key="test")

    def _log_split(self, title: str, key: str, aux_key: str | None = None) -> None:
        log.info(Fore.GREEN + title)
        if key in self._main:
            log.info(f"normal: {tuple(self._main[key].x.shape)}")
        if aux_key is not None:
            for name, split in self._aux[aux_key].items():
                log.info(f"{name}: {tuple(split.x.shape)}")
