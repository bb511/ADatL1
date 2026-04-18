from dataclasses import dataclass
from pathlib import Path
import gc

import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset
from huggingface_hub import snapshot_download

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

    SPLIT_MAP = {
        "pcb": ["test0", "test1", "test2", "test3", "test4", "test5"],
        "piled_bags": ["test0", "test1", "test2", "test3", "test4", "test5"],
        "metal_parts": ["test0", "test1", "test2", "test3", "test4", "test5", "test6"],
    }

    DATA_DIR_MAP = {
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

        self._main: dict[str, SplitTensors] = {}
        self._aux: dict[str, dict[str, SplitTensors]] = {"valid": {}, "test": {}}
        self.shuffler = torch.Generator().manual_seed(seed)
        self.max_val_batches = max_val_batches

        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

        subset = str(subset).lower()
        if subset not in self.SPLIT_MAP:
            raise ValueError(
                f"Unknown subset '{subset}'. Expected one of {list(self.SPLIT_MAP.keys())}."
            )

        if not (0.0 < val_fraction < 1.0):
            raise ValueError("val_fraction must be strictly between 0 and 1.")
        if not (0.0 < test_fraction < 1.0):
            raise ValueError("test_fraction must be strictly between 0 and 1.")
        if val_fraction + test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1.")

    def prepare_data(self) -> None:
        """Ensure the requested RobustAD subset is available locally.

        If the subset directory is missing, download the dataset snapshot from
        Hugging Face into ``data_dir``.
        """
        subset_dir = self._subset_dir()
        if subset_dir.exists():
            log.info(Back.GREEN + f"Found RobustAD subset at {subset_dir}")
            return

        root_dir = Path(self.hparams.data_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            Back.GREEN
            + f"RobustAD subset not found at {subset_dir}. Downloading to {root_dir}..."
        )

        snapshot_download(
            repo_id="AmazonScience/RobustAD",
            repo_type="dataset",
            local_dir=str(root_dir),
            local_dir_use_symlinks=False,
        )

        if not subset_dir.exists():
            raise FileNotFoundError(
                f"Downloaded RobustAD, but expected subset directory still not found: {subset_dir}"
            )

        log.info(Back.GREEN + f"Finished downloading RobustAD subset to {subset_dir}")

    def setup(self, stage: str = None) -> None:
        """Load RobustAD into memory and prepare train/val/test splits."""
        self._set_batch_size()

        if self.hparams.normalize and (self.mean is None or self.std is None):
            self._load_or_compute_stats()

        source_train = self._load_split("train")

        if stage in (None, "fit", "validate"):
            self._setup_fit_validate(source_train)

        if stage in (None, "test"):
            self._setup_test(source_train)

        if stage == "predict":
            raise ValueError("The predict dataloader is not implemented yet.")

        self._data_summary(stage)

    def train_dataloader(self):
        """Create and return the training dataloader."""
        split = self._main["train"]
        dataset = RobustADDataset(
            split.x,
            split.y,
            mask=split.mask,
            batch_size=self.batch_size_per_device,
            shuffler=self.shuffler,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        """Return validation dataloaders with normal first and aux datasets after."""
        return self._make_eval_loaders("valid", "valid", "normal")

    def test_dataloader(self):
        """Return test dataloaders with normal first and aux datasets after."""
        return self._make_eval_loaders("test", "test", "normal")

    def teardown(self, stage: str | None = None) -> None:
        """Release cached tensors."""
        if stage in ("fit", None):
            self._main.pop("train", None)
            self._main.pop("valid", None)
            self._aux.get("valid", {}).clear()

        if stage in ("test", None):
            self._main.pop("test", None)
            self._aux.get("test", {}).clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer RobustAD batches to device.

        Supports:
          - (x, y)
          - (x, mask, y)
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
            if device.type != "cuda":
                return x.to(device), y.to(device)

            if x.device.type == "cpu" and not x.is_pinned():
                x = x.pin_memory()
            if y.device.type == "cpu" and not y.is_pinned():
                y = y.pin_memory()
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            x, mask, y = batch
            if device.type != "cuda":
                return x.to(device), mask.to(device), y.to(device)

            if x.device.type == "cpu" and not x.is_pinned():
                x = x.pin_memory()
            if mask.device.type == "cpu" and not mask.is_pinned():
                mask = mask.pin_memory()
            if y.device.type == "cpu" and not y.is_pinned():
                y = y.pin_memory()
            return (
                x.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )

        raise ValueError(f"Unsupported batch format: {type(batch)}")

    def _subset_dir(self) -> Path:
        root = Path(self.hparams.data_dir)
        subset_name = self.DATA_DIR_MAP[self.hparams.subset]

        direct = root / subset_name
        if direct.exists():
            return direct

        nested = root / "data" / subset_name
        if nested.exists():
            return nested

        return direct

    def _stats_path(self) -> Path:
        if self.hparams.stats_file is not None:
            return self._subset_dir() / self.hparams.stats_file
        return self._subset_dir() / f"{self.hparams.subset}_normal_stats.pt"

    def _load_or_compute_stats(self) -> None:
        """Load cached normalization stats or compute them from source normal train."""
        stats_path = self._stats_path()

        if stats_path.exists():
            stats = torch.load(stats_path, map_location="cpu")
            self.mean = stats["mean"].float()
            self.std = stats["std"].float()
            log.info(f"Loaded RobustAD normalization stats from {stats_path}")
            return

        source_train = self._load_split("train", apply_normalize=False)
        if not torch.all(source_train.y == 0):
            raise RuntimeError("Source train split is expected to contain only normal data.")

        x = source_train.x
        self.mean = x.mean(dim=(0, 2, 3))
        self.std = x.std(dim=(0, 2, 3)).clamp_min(1e-8)

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean, "std": self.std}, stats_path)
        log.info(f"Saved RobustAD normalization stats to {stats_path}")

    def _setup_fit_validate(self, source_train: SplitTensors) -> None:
        """Prepare train and validation splits from source-domain normal train data."""
        n_total = source_train.x.size(0)
        n_valid = max(2, int(round(self.hparams.val_fraction * n_total)))
        n_test = max(2, int(round(self.hparams.test_fraction * n_total)))
        n_train = n_total - n_valid - n_test
        if n_train <= 0:
            raise RuntimeError("Validation/test split too large; no training data remains.")

        full_normal = TensorDataset(source_train.x, source_train.y)
        train_subset, valid_subset, test_subset = random_split(
            full_normal,
            [n_train, n_valid, n_test],
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

        if "train" not in self._main:
            x_train, y_train = self._tensor_dataset_to_tensors(train_subset)
            self._main["train"] = SplitTensors(
                x=x_train.contiguous(),
                y=y_train.contiguous(),
                mask=None,
            )

        if "valid" not in self._main or not self._aux["valid"]:
            x_valid, y_valid = self._tensor_dataset_to_tensors(valid_subset)
            self._main["valid"] = SplitTensors(
                x=x_valid.contiguous(),
                y=y_valid.contiguous(),
                mask=None,
            )
            self._aux["valid"] = self._build_shift_aux()

        if "test" not in self._main:
            x_test, y_test = self._tensor_dataset_to_tensors(test_subset)
            self._main["test"] = SplitTensors(
                x=x_test.contiguous(),
                y=y_test.contiguous(),
                mask=None,
            )

    def _setup_test(self, source_train: SplitTensors) -> None:
        """Prepare held-out main normal test split and target-domain aux sets."""
        if "test" not in self._main:
            n_total = source_train.x.size(0)
            n_valid = max(2, int(round(self.hparams.val_fraction * n_total)))
            n_test = max(2, int(round(self.hparams.test_fraction * n_total)))
            n_train = n_total - n_valid - n_test
            if n_train <= 0:
                raise RuntimeError("Validation/test split too large; no training data remains.")

            full_normal = TensorDataset(source_train.x, source_train.y)
            _, _, test_subset = random_split(
                full_normal,
                [n_train, n_valid, n_test],
                generator=torch.Generator().manual_seed(self.hparams.seed),
            )
            x_test, y_test = self._tensor_dataset_to_tensors(test_subset)
            self._main["test"] = SplitTensors(
                x=x_test.contiguous(),
                y=y_test.contiguous(),
                mask=None,
            )

        if not self._aux["test"]:
            self._aux["test"] = self._build_shift_aux()

    def _build_shift_aux(self) -> dict[str, SplitTensors]:
        """Build auxiliary target-domain datasets.

        Produces:
          - shifted_normal_k
          - shifted_anomaly_k
          - shifted_normal_all
        """
        out: dict[str, SplitTensors] = {}
        shifted_normals: list[SplitTensors] = []

        neg_label = 0
        pos_label = 0

        for shift_idx, split_name in enumerate(self.SPLIT_MAP[self.hparams.subset]):
            split = self._load_split(split_name)

            normal_mask = split.y == 0
            anomaly_mask = split.y == 1

            if normal_mask.any():
                neg_label -= 1
                sn = SplitTensors(
                    x=split.x[normal_mask].contiguous(),
                    y=torch.full(
                        (int(normal_mask.sum().item()),),
                        neg_label,
                        dtype=torch.int64,
                    ),
                    mask=split.mask[normal_mask].contiguous() if split.mask is not None else None,
                )
                out[f"shifted_normal_{shift_idx}"] = sn
                shifted_normals.append(sn)

            if anomaly_mask.any():
                pos_label += 1
                sa = SplitTensors(
                    x=split.x[anomaly_mask].contiguous(),
                    y=torch.full(
                        (int(anomaly_mask.sum().item()),),
                        pos_label,
                        dtype=torch.int64,
                    ),
                    mask=split.mask[anomaly_mask].contiguous() if split.mask is not None else None,
                )
                out[f"shifted_anomaly_{shift_idx}"] = sa

        if shifted_normals:
            out["shifted_normal_all"] = self._concat_splits_balanced(shifted_normals)

        return out

    def _concat_splits_balanced(self, splits: list[SplitTensors]) -> SplitTensors:
        """Concatenate shifted-normal splits with equal-size balancing."""
        min_n = min(s.x.size(0) for s in splits)
        xs = []
        ys = []
        masks = []
        gen = torch.Generator().manual_seed(self.hparams.seed)

        for s in splits:
            perm = torch.randperm(s.x.size(0), generator=gen)[:min_n]
            xs.append(s.x[perm])
            ys.append(s.y[perm])
            if s.mask is not None:
                masks.append(s.mask[perm])

        x = torch.cat(xs, dim=0).contiguous()
        y = torch.cat(ys, dim=0).contiguous()
        mask = torch.cat(masks, dim=0).contiguous() if len(masks) == len(splits) else None

        return SplitTensors(x=x, y=y, mask=mask)

    def _load_split(
        self,
        split_name: str,
        apply_normalize: bool = True,
    ) -> SplitTensors:
        """Load one RobustAD split from disk using Hugging Face datasets."""
        subset_dir = self._subset_dir()

        if split_name == "train":
            data_glob = str(subset_dir / f"{self.hparams.subset}_data_dir_train" / "*")
        else:
            data_glob = str(subset_dir / f"{self.hparams.subset}_data_dir_{split_name}" / "*")

        ds = load_dataset("imagefolder", data_files={split_name: data_glob})[split_name]

        xs = []
        ys = []
        masks = []

        has_any_mask = False

        for sample in ds:
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")

            x = self._pil_to_tensor(img)

            if self.hparams.normalize and apply_normalize:
                if self.mean is None or self.std is None:
                    raise RuntimeError("Normalization requested but mean/std were not set.")
                x = (x - self.mean[:, None, None]) / self.std[:, None, None]

            xs.append(x)
            ys.append(int(sample["label"]))

            mask_obj = sample.get("mask", None)
            if mask_obj is not None:
                has_any_mask = True
                if not isinstance(mask_obj, Image.Image):
                    mask_obj = Image.open(mask_obj)
                mask = self._mask_to_tensor(mask_obj)
                masks.append(mask)
            else:
                masks.append(None)

        x = torch.stack(xs, dim=0)
        y = torch.as_tensor(ys, dtype=torch.int64)

        mask_tensor = None
        if has_any_mask:
            mask_tensor = torch.stack(
                [
                    m if m is not None else torch.zeros((1, x.shape[-2], x.shape[-1]), dtype=torch.float32)
                    for m in masks
                ],
                dim=0,
            )

        return SplitTensors(
            x=x.contiguous(),
            y=y.contiguous(),
            mask=mask_tensor.contiguous() if mask_tensor is not None else None,
        )

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to CHW float tensor in [0, 1], with optional resize."""
        if self.hparams.image_size is not None:
            h, w = int(self.hparams.image_size[0]), int(self.hparams.image_size[1])
            img = img.resize((w, h), Image.BILINEAR)

        x = torch.as_tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())))
        x = x.view(img.size[1], img.size[0], len(img.getbands())).permute(2, 0, 1).float() / 255.0
        return x

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        """Convert optional PIL mask to 1xHxW float tensor."""
        mask = mask.convert("L")
        if self.hparams.image_size is not None:
            h, w = int(self.hparams.image_size[0]), int(self.hparams.image_size[1])
            mask = mask.resize((w, h), Image.NEAREST)

        m = torch.as_tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes())))
        m = m.view(mask.size[1], mask.size[0]).unsqueeze(0).float() / 255.0
        return (m > 0.5).float()

    def _make_eval_loaders(
        self,
        main_key: str,
        aux_key: str,
        main_name: str,
    ) -> dict[str, DataLoader]:
        """Construct evaluation dataloaders with normal first, then auxiliary sets."""
        main = self._main[main_key]
        loaders: dict[str, DataLoader] = {}

        loaders[main_name] = self._to_loader(main, batch_size=self.batch_size_per_device)

        for name, split in self._aux.get(aux_key, {}).items():
            loaders[name] = self._to_loader(
                split,
                batch_size=self.batch_size_per_device,
                max_b=self.max_val_batches,
            )

        return loaders

    def _to_loader(
        self,
        split: SplitTensors,
        batch_size: int,
        max_b: int = None,
    ) -> DataLoader:
        """Transform a split into an iterable DataLoader."""
        ds = RobustADDataset(
            split.x,
            split.y,
            mask=split.mask,
            batch_size=batch_size,
            max_batches=max_b,
        )
        return DataLoader(
            ds,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=torch.cuda.is_available(),
        )

    def _tensor_dataset_to_tensors(self, subset) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a TensorDataset subset into two stacked tensors."""
        xs = []
        ys = []
        for x, y in subset:
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def _set_batch_size(self):
        """Set the batch size per device if multiple devices are available."""
        if self.trainer is None:
            self.batch_size_per_device = self.hparams.batch_size
            return

        world_size = self.trainer.world_size
        if self.hparams.batch_size % world_size != 0:
            raise RuntimeError(
                f"Batch size ({self.hparams.batch_size}) not divisible by the num of "
                f"devices ({world_size})."
            )
        self.batch_size_per_device = self.hparams.batch_size // world_size

    def _data_summary(self, stage: str | None) -> None:
        """Print a summary of the currently loaded data."""
        log.info(Fore.MAGENTA + "-" * 5 + " Data Summary " + "-" * 5)

        def show_split(title: str, key: str, aux_key: str | None = None):
            log.info(Fore.GREEN + title)
            if key in self._main:
                log.info(f"normal: {tuple(self._main[key].x.shape)}")
            if aux_key:
                for name, split in self._aux.get(aux_key, {}).items():
                    log.info(f"{name}: {tuple(split.x.shape)}")

        if stage in (None, "fit"):
            show_split("Training data:", "train")
            show_split("Validation data:", "valid", "valid")
        elif stage == "validate":
            show_split("Validation data:", "valid", "valid")
        elif stage == "test":
            show_split("Test data:", "test", "test")
